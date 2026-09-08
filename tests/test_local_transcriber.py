import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from callforge.codex_runner import CodexRunner
from callforge.config import AppConfig
from callforge.local_transcriber import LocalWhisperPipeline, PreparedTranscription


class FakeLocalPipeline:
    def prepare(self, audio_path, work_directory, log_path, stderr_path):
        metadata = work_directory / "audio.json"
        raw = work_directory / "raw.json"
        agc = work_directory / "agc.json"
        metadata.write_text('{"duration_seconds": 1}', encoding="utf-8")
        raw.write_text('{"text": "raw", "segments": []}', encoding="utf-8")
        agc.write_text('{"text": "agc", "segments": []}', encoding="utf-8")
        return PreparedTranscription(metadata, raw, agc)


class HangingProcess:
    pid = 424242

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise __import__("subprocess").TimeoutExpired("codex", timeout)
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_runtime_environment_preserves_virtualenv_executable(tmp_path, monkeypatch):
    executable = tmp_path / "venv" / "bin" / "python"
    monkeypatch.setattr(sys, "executable", str(executable))
    config = AppConfig.for_root(tmp_path)

    environment = config.runtime_environment()

    assert environment["CALLFORGE_PYTHON"] == str(executable)
    assert environment["HF_HOME"] == str(config.models.resolve())
    assert environment["CALLFORGE_MANAGED_TRANSCRIPTION"] == "1"
    assert config.codex_reasoning_effort == "low"
    assert config.codex_idle_timeout_seconds == 120
    assert config.codex_artifact_grace_seconds == 10
    assert config.codex_ignore_user_config is True
    assert config.codex_ignore_rules is True


def test_local_pipeline_uses_installed_runtime_and_persistent_cache(tmp_path):
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    skill = tmp_path / "skill"
    scripts = skill / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "prepare_audio.py").write_text(
        """
import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
output = Path(args.output_dir)
raw = output / "raw.wav"
agc = output / "agc.wav"
raw.write_bytes(b"raw")
agc.write_bytes(b"agc")
print(json.dumps({"raw_wav": str(raw), "agc_wav": str(agc)}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (scripts / "transcribe_audio.py").write_text(
        """
import json, os, sys
print(json.dumps({
    "backend": "fake",
    "model": "turbo",
    "text": "ok",
    "segments": [],
    "hf_home": os.environ.get("HF_HOME"),
    "python": sys.executable,
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    audio = tmp_path / "external-201-123.mp3"
    audio.write_bytes(b"audio")
    work = config.runs / "test-run"
    work.mkdir()
    log = config.logs / "test.jsonl"
    stderr = config.logs / "test.stderr.log"
    stderr.write_text("", encoding="utf-8")

    prepared = LocalWhisperPipeline(config, skill).prepare(
        audio, work, log, stderr
    )

    raw = json.loads(prepared.raw_transcript_path.read_text(encoding="utf-8"))
    agc = json.loads(prepared.agc_transcript_path.read_text(encoding="utf-8"))
    assert raw["hf_home"] == agc["hf_home"] == str(config.models.resolve())
    assert raw["python"] == agc["python"] == sys.executable
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert events[0]["stage"] == "prepare_audio"
    assert any(event["stage"] == "model_download" for event in events)
    assert events[-1] == {
        "type": "callforge.stage",
        "stage": "whisper",
        "state": "completed",
        "message": "دو پاس مستقل Whisper کامل شد",
    }


def test_codex_prompt_only_requests_review_of_prepared_transcripts(tmp_path):
    config = AppConfig.for_root(tmp_path)
    prepared = PreparedTranscription(
        metadata_path=tmp_path / "audio.json",
        raw_transcript_path=tmp_path / "raw.json",
        agc_transcript_path=tmp_path / "agc.json",
    )
    prompt = CodexRunner(config).build_prompt(tmp_path / "call.mp3", prepared)

    assert "CallForge-managed review mode" in prompt
    assert str(prepared.raw_transcript_path) in prompt
    assert str(prepared.agc_transcript_path) in prompt
    assert "Do not run Whisper" in prompt
    assert "Do not change HF_HOME" in prompt
    assert "last tool action" in prompt


def test_codex_finishes_when_valid_markdown_is_ready(tmp_path, monkeypatch):
    config = replace(
        AppConfig.for_root(tmp_path),
        codex_artifact_grace_seconds=0,
        codex_model="gpt-test",
    )
    config.ensure()
    audio = tmp_path / "external-201-123.mp3"
    audio.write_bytes(b"audio")
    markdown = audio.with_suffix(".md")
    log = config.logs / "run.jsonl"
    stderr = config.logs / "run.stderr.log"
    process = HangingProcess()
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        kwargs["stdout"].write(
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}) + "\n"
        )
        kwargs["stdout"].flush()
        markdown.write_text(
            "# متن تماس\n\n## مکالمه\n\n**مشتری:** سلام\n",
            encoding="utf-8",
        )
        return process

    def fake_stop(selected_process):
        assert selected_process is process
        selected_process.returncode = -15

    monkeypatch.setattr("callforge.codex_runner.shutil.which", lambda _: "codex")
    monkeypatch.setattr("callforge.codex_runner.subprocess.Popen", fake_popen)
    runner = CodexRunner(config, FakeLocalPipeline())
    monkeypatch.setattr(runner, "_terminate_process_tree", fake_stop)

    result = runner.run(audio, log, stderr)

    assert result.returncode == 0
    assert result.thread_id == "thread-1"
    assert "--ignore-user-config" in captured["command"]
    assert "--ignore-rules" in captured["command"]
    assert captured["command"][captured["command"].index("--model") + 1] == "gpt-test"
    if os.name == "nt":
        assert "creationflags" in captured["kwargs"]
    else:
        assert captured["kwargs"]["start_new_session"] is True
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(event.get("stage") == "artifact_ready" for event in events)
    assert events[-1]["stage"] == "review"
    assert events[-1]["state"] == "completed"


def test_codex_hard_timeout_stops_tree_and_fails_without_artifact(
    tmp_path, monkeypatch
):
    config = replace(
        AppConfig.for_root(tmp_path),
        codex_timeout_seconds=1,
        codex_idle_timeout_seconds=100,
        codex_model="gpt-test",
    )
    config.ensure()
    audio = tmp_path / "external-201-123.mp3"
    audio.write_bytes(b"audio")
    audio.with_suffix(".md").write_text(
        "# متن تماس\n\n## مکالمه\n\n**مشتری:** متن قبلی\n",
        encoding="utf-8",
    )
    log = config.logs / "timeout.jsonl"
    stderr = config.logs / "timeout.stderr.log"
    process = HangingProcess()
    stopped = []
    ticks = iter((0.0, 2.0))

    monkeypatch.setattr("callforge.codex_runner.shutil.which", lambda _: "codex")
    monkeypatch.setattr(
        "callforge.codex_runner.subprocess.Popen", lambda command, **kwargs: process
    )
    monkeypatch.setattr("callforge.codex_runner.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("callforge.codex_runner.time.sleep", lambda _: None)
    runner = CodexRunner(config, FakeLocalPipeline())

    def fake_stop(selected_process):
        stopped.append(selected_process)
        selected_process.returncode = -15

    monkeypatch.setattr(runner, "_terminate_process_tree", fake_stop)

    result = runner.run(audio, log, stderr)

    assert result.returncode == 124
    assert stopped == [process]
    assert "exceeded the 1-second timeout" in result.stderr
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["stage"] == "error"
    assert events[-1]["state"] == "failed"
