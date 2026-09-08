import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from callforge.config import AppConfig
from callforge.speaker_pipeline import SpeakerPipeline
from callforge.quality import write_json


def sample():
    rows = [dict(id="s1", start=0., end=2., text="از پشتیبانی تماس می‌گیرم.", raw_text="raw", notes="note"),
            dict(id="s2", start=2., end=4., text="درخواست شما را بررسی کردم.")]
    acoustic = {"model": "pyannote/speaker-diarization-community-1",
                "turns": [dict(start=0., end=4., speaker_id="voice_B")]}
    roles = {"roles": [dict(speaker_id="voice_B", role="پشتیبان", reason="دو شاهد پشتیبانی",
                     evidence=[dict(segment_id=r["id"], quote=r["text"]) for r in rows])]}
    return rows, acoustic, roles


@pytest.mark.parametrize("failure", [None, "model", "role", "rewrite"])
def test_production_stages_preserve_words_and_fail_closed(tmp_path, monkeypatch, failure):
    config = AppConfig.for_root(tmp_path)
    rows, acoustic, roles = sample()
    source = tmp_path / "out-123-201.mp3"
    source.write_bytes(b"audio")
    source.with_suffix(".md").write_text("human approved")
    log, errors = tmp_path / "events.jsonl", tmp_path / "errors.log"
    pipeline = SpeakerPipeline(config)
    monkeypatch.setattr(pipeline, "preflight", lambda: None)
    monkeypatch.setattr("callforge.speaker_pipeline.subprocess.run", lambda *a, **kw: None)
    def model(command, destination, *args):
        if failure == "model":
            raise RuntimeError("model failed")
        write_json(destination, acoustic)
        return acoustic
    monkeypatch.setattr(pipeline, "_json_process", model)
    def infer(data, *args):
        assert data["direction"] == "outbound"
        assert "human approved" not in json.dumps(data)
        assert "raw_text" not in data["segments"][0]
        if failure == "role":
            raise RuntimeError("role timeout")
        return roles
    monkeypatch.setattr(pipeline, "_roles", infer)
    if failure == "rewrite":
        monkeypatch.setattr("callforge.speaker_pipeline.apply_roles",
                            lambda aligned, roles: [dict(r, text="invented") for r in aligned])
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            pipeline.run(source, rows, tmp_path, log, errors)
    else:
        predicted, report = pipeline.run(source, rows, tmp_path, log, errors)
        assert [r["text"] for r in predicted] == [r["text"] for r in rows]
        assert predicted[0]["raw_text"] == "raw" and predicted[0]["notes"] == "note"
        assert all(r["speaker"] == "پشتیبان" for r in predicted)
        assert report["roles"] == roles["roles"]
        assert set(report["stage_seconds"]) == {"diarization", "word_alignment", "speaker_roles"}
        assert "word_alignment" in log.read_text()
    assert source.with_suffix(".md").read_text() == "human approved"
    assert source.read_bytes() == b"audio"
    assert not (tmp_path / "speaker-raw.wav").exists()
    assert json.loads((tmp_path / "speaker-pipeline.json").read_text())["status"] == ("failed" if failure else "completed")


@pytest.mark.parametrize("bad", [False, True])
def test_ctc_refines_only_ambiguous_reviewed_text(tmp_path, monkeypatch, bad):
    from callforge.alignment import MODEL, REVISION
    rows, acoustic, roles = sample()
    acoustic["turns"] = []
    pipeline = SpeakerPipeline(AppConfig.for_root(tmp_path))
    monkeypatch.setattr(pipeline, "preflight", lambda: None)
    monkeypatch.setattr("callforge.speaker_pipeline.subprocess.run", lambda *a, **kw: None)
    calls = []
    def run(command, destination, *args):
        calls.append(command)
        if "callforge.alignment_worker" in command:
            data = json.loads((tmp_path / "alignment-input.json").read_text())
            assert data["segments"][0]["text"] == rows[0]["text"]
            return dict(model=MODEL, revision="bad" if bad else REVISION,
                        segments=[dict(id=r["id"], status="unalignable", words=[]) for r in rows])
        return acoustic
    monkeypatch.setattr(pipeline, "_json_process", run)
    monkeypatch.setattr(pipeline, "_roles", lambda *a: {"roles": []})
    args = (tmp_path / "external-x.mp3", rows, tmp_path, tmp_path / "events", tmp_path / "errors")
    if bad:
        with pytest.raises(ValueError, match="mismatched"):
            pipeline.run(*args)
    else:
        predicted, _ = pipeline.run(*args)
        assert all(r["speaker"] == "گوینده نامشخص" for r in predicted)
    assert len(calls) == 2


@pytest.mark.parametrize("failure", [False, True])
def test_runner_publishes_only_after_speaker_success(tmp_path, monkeypatch, failure):
    from callforge.codex_runner import CodexRunner
    from callforge.local_transcriber import PreparedTranscription
    config = replace(AppConfig.for_root(tmp_path), codex_model="test")
    config.ensure()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    audio.with_suffix(".md").write_text("previous approved")
    class Local:
        def prepare(self, source, directory, *args):
            write_json(directory / "audio.json", {"duration_seconds": 2})
            data = {"text": "سلام", "segments": [dict(start=0, end=2, text="سلام")]}
            write_json(directory / "raw.json", data)
            write_json(directory / "agc.json", data)
            return PreparedTranscription(directory / "audio.json", directory / "raw.json", directory / "agc.json")
    def popen(command, **kwargs):
        output = Path(command[command.index("--output-last-message") + 1])
        write_json(output, {"segments": [dict(id="s1", text="سلام", speaker="گوینده نامشخص", uncertain=False, notes="")]})
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr("callforge.codex_runner.shutil.which", lambda _: "codex")
    monkeypatch.setattr("callforge.codex_runner.subprocess.Popen", popen)
    monkeypatch.setattr(SpeakerPipeline, "preflight", lambda self: None)
    def run(self, source, rows, *args):
        assert source.with_suffix(".md").read_text() == "previous approved"
        if failure:
            raise RuntimeError("speaker failure")
        return [dict(r, speaker="مشتری", speaker_id="B") for r in rows], {"status": "completed"}
    monkeypatch.setattr(SpeakerPipeline, "run", run)
    runner = CodexRunner(config, Local())
    if failure:
        with pytest.raises(RuntimeError, match="speaker failure"):
            runner.run(audio, config.logs / "events", config.logs / "errors")
        assert audio.with_suffix(".md").read_text() == "previous approved"
        quality = json.loads(next(config.runs.glob("*/quality.json")).read_text())
        assert quality["automated_review_complete"] is False
    else:
        result = runner.run(audio, config.logs / "events", config.logs / "errors")
        assert result.returncode == 0
        assert result.quality["speaker_pipeline_complete"] is True
        assert result.quality["segments"][0]["speaker_id"] == "B"
        assert "مشتری" in audio.with_suffix(".md").read_text()
