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
