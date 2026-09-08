import json
from dataclasses import replace

import pytest

from callforge.benchmark import freeze_reference, run_speaker_experiment, speaker_score, speaker_python
from callforge.config import AppConfig
from callforge.db import Database
from callforge.metadata import extract_audio_metadata
from callforge.quality import file_hash


def fixture(tmp_path):
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    db = Database(config.database)
    db.initialize()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"test")
    metadata = replace(extract_audio_metadata(audio, tmp_path), duration_seconds=10)
    audio_id, _, _ = db.upsert_audio(metadata)
    job = db.claim_jobs(1, "test", 100)[0]
    run = db.start_run(job, "test", config.logs / "test.jsonl", config.logs / "test.stderr")
    rows = [{"id": "s1", "start": 0., "end": 5., "text": "سلام", "speaker": "گوینده نامشخص", "uncertain": False, "flags": []}]
    audio.with_suffix(".md").write_text("original")
    baseline = db.complete_run(job, run, "original", audio.with_suffix(".md"), "fa", None,
                               quality={"segments": rows, "automated_review_complete": True})
    return config, db, audio, audio_id, baseline, rows


def test_freeze_requires_approved_and_keeps_immutable_reference(tmp_path):
    config, db, audio, audio_id, baseline, rows = fixture(tmp_path)
    output = tmp_path / "reference.json"
    with pytest.raises(ValueError, match="human-approved"):
        freeze_reference(db, audio_id, output)
    assert not output.exists()
    reference = db.save_review(audio_id, {"base_transcript_id": baseline, "status": "approved", "reviewer": "test",
                                         "segments": [dict(rows[0], speaker="مشتری")]})
    value = freeze_reference(db, audio_id, output)
    assert value["reference"]["transcript_id"] == reference
    assert value["baseline_speaker_score"]["assignment_error_rate"] == 1
    with pytest.raises(FileExistsError):
        freeze_reference(db, audio_id, output)
    db.save_review(audio_id, {"base_transcript_id": reference, "status": "in_review", "reviewer": "test",
                             "segments": [dict(rows[0], text="متن جدید")]})
    assert json.loads(output.read_text())["reference"]["segments"][0]["text"] == "سلام"
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="no longer matches"):
        run_speaker_experiment(config, output, tmp_path / "experiment")


def test_speaker_mapping_is_permutation_invariant_and_not_role_prediction():
    reference = [{"start": 0, "end": 3, "speaker": "customer"}, {"start": 3, "end": 5, "speaker": "support"}]
    hypothesis = [{"start": 0, "end": 3, "speaker_id": "B"}, {"start": 3, "end": 5, "speaker_id": "A"}]
    result = speaker_score(reference, hypothesis)
    assert result["assignment_error_rate"] == 0
    assert result["best_label_mapping_for_evaluation_only"] == {"A": "support", "B": "customer"}
    assert speaker_score(reference, []) ["assignment_error_rate"] == 1
    hypothesis[1]["speaker_id"] = "B"
    assert speaker_score(reference, hypothesis)["assignment_error_rate"] == pytest.approx(.4)


def test_reference_overlap_is_excluded_and_fragmentation_penalized():
    reference = [{"start": 0, "end": 4, "speaker": "customer"}]
    hypothesis = [{"start": 0, "end": 2, "speaker_id": "A"}, {"start": 2, "end": 4, "speaker_id": "B"}]
    assert speaker_score(reference, hypothesis)["assignment_error_rate"] == .5
    reference.append({"start": 3, "end": 4, "speaker": "support"})
    assert speaker_score(reference, hypothesis)["excluded_reference_overlap_seconds"] == 1
    assert speaker_score([], [])["assignment_error_rate"] is None


def test_experiment_never_changes_sources_or_database(tmp_path, monkeypatch):
    config, db, audio, audio_id, baseline, rows = fixture(tmp_path)
    db.save_review(audio_id, {"base_transcript_id": baseline, "status": "approved", "reviewer": "test",
                             "segments": [dict(rows[0], speaker="مشتری")]})
    snapshot = tmp_path / "reference.json"
    freeze_reference(db, audio_id, snapshot)
    before = (file_hash(audio), file_hash(audio.with_suffix(".md")), db.counts())
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "diarize_audio.py" in command[1]:
            json.dump({"turns": [{"start": 0, "end": 5, "speaker_id": "SPEAKER_00"}]}, kwargs["stdout"])
    monkeypatch.setattr("callforge.benchmark.subprocess.run", run)
    output = tmp_path / "experiment"
    interpreter = speaker_python(config)
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    report = run_speaker_experiment(config, snapshot, output)
    assert report["status"] == "completed"
    assert report["speaker_score"]["assignment_error_rate"] == 0
    assert len(calls) == 2  # decode + diarization, never Whisper or Codex
    assert before == (file_hash(audio), file_hash(audio.with_suffix(".md")), db.counts())
    assert not list(output.glob("audio-*"))
    with pytest.raises(FileExistsError):
        run_speaker_experiment(config, snapshot, output)


def test_hf_login_rejects_noninteractive_token_entry(monkeypatch):
    from callforge.cli import command_hf_login
    monkeypatch.setattr("callforge.cli.workspace", lambda: (None, None))
    monkeypatch.setattr("callforge.cli.sys.stdin.isatty", lambda: False)
    with pytest.raises(ValueError, match="interactive terminal"):
        command_hf_login(None)


def test_hf_auth_masks_input_and_never_saves_git_credential(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace
    from callforge import hf_auth
    received = []
    fake = SimpleNamespace(constants=SimpleNamespace(HF_TOKEN_PATH=str(tmp_path / "token"),
                                                    HF_STORED_TOKENS_PATH=str(tmp_path / "stored_tokens")),
                           login=lambda **kwargs: received.append(kwargs))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setattr(hf_auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(hf_auth.getpass, "getpass", lambda prompt: "fake-test-token")
    assert hf_auth.main() == 0
    assert received == [{"token": "fake-test-token", "add_to_git_credential": False}]
    assert "fake-test-token" not in capsys.readouterr().out
    def fail(**kwargs):
        raise ValueError("fake-test-token")
    fake.login = fail
    assert hf_auth.main() == 1
    output = capsys.readouterr()
    assert "fake-test-token" not in output.out + output.err
