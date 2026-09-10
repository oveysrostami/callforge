import json
from dataclasses import replace

import pytest

from callforge.config import AppConfig
from callforge.db import Database
from callforge.metadata import extract_audio_metadata
from callforge.quality import render_markdown


def prepared(tmp_path):
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    metadata = replace(extract_audio_metadata(audio, tmp_path), duration_seconds=10)
    audio_id, _, _ = database.upsert_audio(metadata)
    job = database.claim_jobs(1, "test", 100)[0]
    run = database.start_run(job, "test", config.logs / "test.jsonl", config.logs / "test.stderr")
    rows = [{"id": "s1", "start": 0, "end": 5, "speaker": "گوینده نامشخص", "text": "سلام",
             "raw_text": "سلام", "alternative": "سلام", "flags": [], "uncertain": False}]
    content = render_markdown(audio.name, rows)
    audio.with_suffix(".md").write_text(content, encoding="utf-8")
    transcript_id = database.complete_run(job, run, content, audio.with_suffix(".md"), "fa", None,
                                         quality={"segments": rows, "quality_status": "needs_review"})
    return database, audio, audio_id, transcript_id, rows


def test_human_review_is_versioned_synced_and_optimistically_locked(tmp_path):
    database, audio, audio_id, initial, rows = prepared(tmp_path)
    payload = {"base_transcript_id": initial, "reviewer": "بازبین آزمایشی", "notes": "شنیده شد",
               "status": "approved", "segments": [dict(rows[0], text="سلام وقت بخیر")]}
    second = database.save_review(audio_id, payload)
    detail = database.review_detail(audio_id)
    assert second != initial
    assert detail["status"] == "approved"
    assert detail["markdown_synced"] is True
    assert len(detail["versions"]) == 2
    assert "وقت بخیر" in audio.with_suffix(".md").read_text(encoding="utf-8")
    assert detail["data"]["segments"][0]["raw_text"] == "سلام"
    with pytest.raises(ValueError, match="نسخه"):
        database.save_review(audio_id, payload)
    assert len(database.review_detail(audio_id)["versions"]) == 2


def test_unclear_queue_correction_versions_transcript_and_saves_training_example(tmp_path):
    database, audio, audio_id, initial, rows = prepared(tmp_path)
    unclear = dict(
        rows[0], text="سلام من [نامفهوم] هستم", raw_text="سلام من وینسی هستم",
        alternative="سلام من ونسی هستم", flags=["unclear"], uncertain=True,
        confidence_tier="unresolved", uncertainty_spans=[{"candidates": ["ونسی"]}],
    )
    replacement = render_markdown(audio.name, [unclear])
    audio.with_suffix(".md").write_text(replacement, encoding="utf-8")
    with database.transaction() as connection:
        connection.execute("UPDATE transcripts SET content=?, content_hash=?, unclear_count=1 WHERE id=?",
                           (replacement, __import__("hashlib").sha256(replacement.encode()).hexdigest(), initial))
        connection.execute("UPDATE transcript_reviews SET data_json=? WHERE transcript_id=?",
                           (json.dumps({"segments": [unclear]}, ensure_ascii=False), initial))

    total, queue = database.review_queue()
    assert total == 1
    assert queue[0]["segment_id"] == "s1"
    selected = next(item for item in queue[0]["candidates"] if item["text"] == "سلام من ونسی هستم")
    result, correction = database.save_unclear_correction({
        "audio_id": audio_id, "base_transcript_id": initial, "segment_id": "s1",
        "reviewer": "بازبین", "selection_source": "candidate", "candidate_id": selected["id"],
    })
    assert result != initial
    assert correction > 0
    detail = database.review_detail(audio_id)
    assert detail["status"] == "in_review"
    assert detail["data"]["segments"][0]["text"] == "سلام من ونسی هستم"
    assert detail["data"]["segments"][0]["confidence_tier"] == "human_corrected"
    assert "ونسی" in audio.with_suffix(".md").read_text(encoding="utf-8")
    assert database.review_queue()[0] == 0
    saved = database.correction_examples()[0]
    assert saved["source_transcript_id"] == initial
    assert saved["result_transcript_id"] == result
    assert saved["original_text"] == "سلام من [نامفهوم] هستم"
    assert saved["corrected_text"] == "سلام من ونسی هستم"
    assert json.loads(saved["evidence_json"])["raw_text"] == "سلام من وینسی هستم"


def test_unclear_queue_custom_correction_rejects_stale_and_unresolved_text(tmp_path):
    database, audio, audio_id, initial, rows = prepared(tmp_path)
    unclear = dict(rows[0], text="[نامفهوم]", flags=["unclear"], uncertain=True)
    content = render_markdown(audio.name, [unclear])
    audio.with_suffix(".md").write_text(content, encoding="utf-8")
    with database.transaction() as connection:
        connection.execute("UPDATE transcripts SET content=?, content_hash=?, unclear_count=1 WHERE id=?",
                           (content, __import__("hashlib").sha256(content.encode()).hexdigest(), initial))
        connection.execute("UPDATE transcript_reviews SET data_json=? WHERE transcript_id=?",
                           (json.dumps({"segments": [unclear]}, ensure_ascii=False), initial))
    payload = {"audio_id": audio_id, "base_transcript_id": initial, "segment_id": "s1",
               "reviewer": "بازبین", "selection_source": "custom", "custom_text": "[نامفهوم]"}
    with pytest.raises(ValueError, match="کامل اصلاح"):
        database.save_unclear_correction(payload)
    assert database.correction_examples() == []
    payload["custom_text"] = "پشتیبان ونسی هستم"
    database.save_unclear_correction(payload)
    with pytest.raises(ValueError, match="نسخه تغییر کرده"):
        database.save_unclear_correction(payload)


def test_nonfinite_metrics_are_normalized_in_transcript_and_run_evidence(tmp_path):
    database, audio, audio_id, _, rows = prepared(tmp_path)
    database.queue_transcription(audio_id)
    job = database.claim_audio_job(audio_id, "worker", 100)
    run = database.start_run(job, "worker", tmp_path / "numeric.jsonl", tmp_path / "numeric.stderr")
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "legacy.json").write_text('{"score":Infinity}')
    rows[0]["retry"] = {"avg_logprob": float("nan")}
    transcript = database.complete_run(job, run, render_markdown(audio.name, rows), audio.with_suffix(".md"),
                                       "fa", None, quality={"segments": rows}, evidence_directory=directory)
    def reject(token):
        raise ValueError(token)
    with database.connect() as connection:
        data = json.loads(connection.execute("SELECT data_json FROM transcript_reviews WHERE transcript_id=?", (transcript,)).fetchone()[0], parse_constant=reject)
        evidence = json.loads(connection.execute("SELECT payload_json FROM run_evidence WHERE processing_run_id=?", (run,)).fetchone()[0], parse_constant=reject)
    assert data["segments"][0]["retry"]["avg_logprob"] is None
    assert evidence["legacy.json"]["score"] is None


def test_external_markdown_is_not_overwritten_and_db_revision_survives(tmp_path):
    database, audio, audio_id, initial, rows = prepared(tmp_path)
    audio.with_suffix(".md").write_text("external change", encoding="utf-8")
    with pytest.raises(ValueError, match="بیرون"):
        database.save_review(audio_id, {"base_transcript_id": initial, "reviewer": "test", "status": "in_review", "segments": rows})
    detail = database.review_detail(audio_id)
    assert len(detail["versions"]) == 2
    assert detail["markdown_synced"] is False
    assert audio.with_suffix(".md").read_text() == "external change"


@pytest.mark.parametrize("bad", [float("nan"), -1, 11])
def test_invalid_review_timestamps_do_not_create_revisions(tmp_path, bad):
    database, _, audio_id, initial, rows = prepared(tmp_path)
    rows[0]["start"] = bad
    with pytest.raises(ValueError):
        database.save_review(audio_id, {"base_transcript_id": initial, "reviewer": "test", "status": "in_review", "segments": rows})
    assert len(database.review_detail(audio_id)["versions"]) == 1


def test_review_and_evidence_cascade_on_scoped_reset(tmp_path):
    database, _, audio_id, initial, _ = prepared(tmp_path)
    with database.connect() as connection:
        run = connection.execute("SELECT id FROM processing_runs").fetchone()[0]
        connection.execute("INSERT INTO run_evidence VALUES (?,?,?)", (run, "directory", json.dumps({"raw": "test"})))
    database.reset(tmp_path)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM transcript_reviews").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM run_evidence").fetchone()[0] == 0
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


def test_running_scan_cannot_replace_worker_transcript(tmp_path):
    database, audio, audio_id, initial, _ = prepared(tmp_path)
    database.queue_transcription(audio_id)
    database.claim_audio_job(audio_id, "busy", 100)
    audio.with_suffix(".md").write_text("external", encoding="utf-8")
    assert database.import_markdown(audio_id, audio.with_suffix(".md")) is None
    assert database.audio_file_detail(audio_id)["job_status"] == "running"
    assert database.review_detail(audio_id)["transcript_id"] == initial


def test_truncated_review_is_failure_and_previous_version_survives(tmp_path):
    database, audio, audio_id, initial, rows = prepared(tmp_path)
    database.queue_transcription(audio_id)
    job = database.claim_audio_job(audio_id, "worker", 100)
    run = database.start_run(job, "worker", tmp_path / "run.jsonl", tmp_path / "run.stderr")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "review.json").write_text('{"segments": [', encoding="utf-8")
    content = render_markdown(audio.name, rows)
    with pytest.raises(ValueError, match="incomplete automated review"):
        database.complete_run(job, run, content, audio.with_suffix(".md"), "fa", None,
                              quality={"segments": rows, "automated_review_complete": False},
                              evidence_directory=evidence)
    database.fail_run(job, run, "Incomplete review", retryable=False, evidence_directory=evidence)
    assert database.audio_file_detail(audio_id)["job_status"] == "failed"
    assert database.review_detail(audio_id)["transcript_id"] == initial
    with database.connect() as connection:
        value = json.loads(connection.execute("SELECT payload_json FROM run_evidence WHERE processing_run_id=?", (run,)).fetchone()[0])
    assert value["review.json"]["invalid_json"] is True


def test_speaker_roles_persist_with_transcript_and_incomplete_roles_cannot_publish(tmp_path):
    database, audio, audio_id, initial, rows = prepared(tmp_path)
    database.queue_transcription(audio_id)
    job = database.claim_audio_job(audio_id, "worker", 100)
    run = database.start_run(job, "worker", tmp_path / "roles.jsonl", tmp_path / "roles.stderr")
    data = {"segments": [dict(r, speaker_id="voice1") for r in rows],
            "automated_review_complete": True, "speaker_pipeline_complete": False,
            "speaker_pipeline": {"roles": [{"speaker_id": "voice1", "role": "مشتری", "reason": "شاهد", "evidence": []}]}}
    content = render_markdown(audio.name, data["segments"])
    with pytest.raises(ValueError, match="incomplete automated review"):
        database.complete_run(job, run, content, audio.with_suffix(".md"), "fa", None, quality=data)
    assert database.review_detail(audio_id)["transcript_id"] == initial
    data["speaker_pipeline_complete"] = True
    database.complete_run(job, run, content, audio.with_suffix(".md"), "fa", None, quality=data)
    saved = database.review_detail(audio_id)["data"]
    assert saved["speaker_pipeline"] == data["speaker_pipeline"]
    assert saved["segments"][0]["speaker_id"] == "voice1"


def test_evaluation_cli_uses_human_reference_and_frozen_sample(tmp_path, monkeypatch, capsys):
    from callforge import cli
    database, _, audio_id, initial, rows = prepared(tmp_path)
    monkeypatch.setattr(cli, "workspace", lambda: (AppConfig.for_root(tmp_path), database))
    sample = tmp_path / "sample.json"
    assert cli.main(["evaluation-sample", "--count", "40", "--output", str(sample)]) == 0
    assert json.loads(sample.read_text())["calls"][0]["evaluation_split"] == "holdout"
    database.save_review(audio_id, {"base_transcript_id": initial, "reviewer": "fixture", "status": "approved",
                                    "segments": [dict(rows[0], text="سلام وقت بخیر")]})
    capsys.readouterr()
    assert cli.main(["evaluate", "--sample", str(sample)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["evaluated_calls"] == 1
    assert report["wer"] == pytest.approx(2 / 3)
    assert report["by_split"]["holdout"]["calls"] == 1
    assert report["not_evaluated_audio_ids"] == []
