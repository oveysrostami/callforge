import json
import os
import sys
from dataclasses import replace
from pathlib import Path
import pytest
import subprocess

from callforge.codex_runner import CodexRunner
from callforge.config import AppConfig
from callforge.local_transcriber import LocalWhisperPipeline, PreparedTranscription


def test_known_mlx_native_shutdown_failure_retries_once(tmp_path, monkeypatch):
    config = AppConfig.for_root(tmp_path)
    stderr = tmp_path / "stderr.log"
    responses = [
        subprocess.CompletedProcess([], -6, "", "recursive_mutex lock failed: Invalid argument"),
        subprocess.CompletedProcess([], 0, '{"segments": []}', ""),
    ]
    monkeypatch.setattr("callforge.local_transcriber.subprocess.run", lambda *a, **k: responses.pop(0))
    result = LocalWhisperPipeline(config)._run_json(["whisper"], tmp_path / "result.json", stderr, "Whisper AGC pass")
    assert result["segments"] == []
    assert not responses
    assert "retrying this pass once" in stderr.read_text()


@pytest.fixture(autouse=True)
def isolate_text_review(monkeypatch):
    # This module tests text review; speaker integration has its own tests.
    monkeypatch.setattr("callforge.speaker_pipeline.SpeakerPipeline.preflight", lambda self: None)
    monkeypatch.setattr("callforge.speaker_pipeline.SpeakerPipeline.run",
                        lambda self, source, rows, *args: (rows, {"status": "completed"}))


class FakeLocalPipeline:
    def prepare(self, audio_path, work_directory, log_path, stderr_path):
        metadata = work_directory / "audio.json"
        raw = work_directory / "raw.json"
        agc = work_directory / "agc.json"
        metadata.write_text('{"duration_seconds": 1}', encoding="utf-8")
        raw.write_text(json.dumps({"text": "سلام", "segments": [{"start": 0, "end": 1, "text": "سلام"}]}), encoding="utf-8")
        agc.write_text(raw.read_text(encoding="utf-8"), encoding="utf-8")
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
    "prompt": sys.argv[sys.argv.index("--prompt") + 1],
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
    assert raw["prompt"] == ""
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert events[0]["stage"] == "prepare_audio"
    assert not any(event["stage"] == "model_download" for event in events)
    assert events[-1]["stage"] == "whisper"
    assert events[-1]["state"] == "completed"
    assert (work / "evidence.json").is_file()


def test_local_pipeline_recovers_vad_only_speech_in_a_context_window(tmp_path):
    config = replace(AppConfig.for_root(tmp_path), whisper_retry_segments=0)
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
raw = output / "raw.wav"; raw.write_bytes(b"raw")
agc = output / "agc.wav"; agc.write_bytes(b"agc")
print(json.dumps({"raw_wav": str(raw), "agc_wav": str(agc), "duration_seconds": 12}))
""".strip() + "\n", encoding="utf-8")
    (scripts / "transcribe_audio.py").write_text(
        """
import json, sys
if "--start" not in sys.argv:
    print(json.dumps({"text": "", "segments": [], "speech_regions": [{"start": 0, "end": 4}]}))
else:
    print(json.dumps({"start": 0, "end": 10, "text": "سلام دنیا", "segments": [{
        "start": 1, "end": 3, "text": "سلام دنیا", "avg_logprob": -.2,
        "no_speech_prob": .01, "compression_ratio": 1.0,
        "words": [{"start": 1, "end": 1.5, "word": "سلام "},
                  {"start": 2, "end": 2.5, "word": "دنیا"}]}]}))
""".strip() + "\n", encoding="utf-8")
    audio = tmp_path / "external-201-123.mp3"
    audio.write_bytes(b"audio")
    work = config.runs / "coverage-run"; work.mkdir()
    log = config.logs / "coverage.jsonl"
    stderr = config.logs / "coverage.stderr.log"; stderr.write_text("")

    LocalWhisperPipeline(config, skill).prepare(audio, work, log, stderr)

    result = json.loads((work / "evidence.json").read_text(encoding="utf-8"))
    assert result["coverage_recovery"] == {
        "requested_windows": 1, "completed_windows": 1,
        "recovered_segments": 1, "remaining_segments": 0,
        "audio_variant": "window_quality_selection",
        "cascade": ["mlx-whisper:large-v3-turbo", "mlx-whisper:large-v3"],
    }
    assert result["segments"][0]["retry"]["text"] == "سلام دنیا"
    assert (work / "coverage-1-0-raw.json").is_file()
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(event["stage"] == "whisper_coverage" and event["state"] == "completed"
               for event in events)


def test_codex_prompt_only_requests_review_of_prepared_transcripts(tmp_path):
    config = AppConfig.for_root(tmp_path)
    prepared = PreparedTranscription(
        metadata_path=tmp_path / "audio.json",
        raw_transcript_path=tmp_path / "raw.json",
        agc_transcript_path=tmp_path / "agc.json",
    )
    prompt = CodexRunner(config).build_prompt(tmp_path / "call.mp3", prepared)

    assert "CallForge-managed review mode" in prompt
    assert str(tmp_path / "review-input.json") in prompt
    assert str(prepared.raw_transcript_path) not in prompt
    assert str(prepared.agc_transcript_path) not in prompt
    assert "Do not run Whisper" in prompt
    assert "Do not change HF_HOME" in prompt
    assert "Do not write or edit files" in prompt


def test_codex_publishes_valid_structured_final_output(tmp_path, monkeypatch):
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
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps({"segments": [{"id": "s1", "text": "سلام", "speaker": "مشتری", "uncertain": False, "notes": ""}]}), encoding="utf-8")
        process.returncode = 0
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
    assert result.quality["automated_review_complete"] is True
    assert result.evidence_directory.is_dir()
    assert (result.evidence_directory / "raw.json").is_file()
    assert "سلام" in markdown.read_text(encoding="utf-8")
    assert captured["command"][captured["command"].index("--sandbox") + 1] == "read-only"
    assert events[-1]["stage"] == "review"
    assert events[-1]["state"] == "completed"


def test_codex_hard_timeout_preserves_previous_markdown_without_publication(
    tmp_path, monkeypatch
):
    config = replace(
        AppConfig.for_root(tmp_path),
        codex_timeout_seconds=1,
        codex_idle_timeout_seconds=100,
        codex_model="gpt-test",
        codex_review_attempts=1,
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
    assert result.quality["automated_review_complete"] is False
    assert result.quality["review_error"]
    assert result.quality["quality_status"] == "needs_review"
    assert stopped == [process]
    assert "متن قبلی" in audio.with_suffix(".md").read_text(encoding="utf-8")
    assert not (result.evidence_directory / "transcript.md").exists()
    assert "exceeded the 1-second timeout" in result.stderr
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(event.get("state") == "warning" for event in events)


def test_successful_exit_with_truncated_response_is_not_successful_review(tmp_path, monkeypatch):
    config = replace(AppConfig.for_root(tmp_path), codex_model="gpt-test")
    config.ensure()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    process = HangingProcess()
    process.returncode = 0
    def fake_popen(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text('{"segments": [', encoding="utf-8")
        return process
    monkeypatch.setattr("callforge.codex_runner.shutil.which", lambda _: "codex")
    monkeypatch.setattr("callforge.codex_runner.subprocess.Popen", fake_popen)
    result = CodexRunner(config, FakeLocalPipeline()).run(audio, config.logs / "run.jsonl", config.logs / "run.stderr")
    assert result.returncode != 0
    assert result.quality["automated_review_complete"] is False
    assert result.quality["review_error"]
    assert not audio.with_suffix(".md").exists()
    assert len(list(result.evidence_directory.glob("review-attempt-*.json"))) == 2
    assert (result.evidence_directory / "evidence.json").exists()


def test_review_retry_uses_same_asr_and_never_accepts_stale_response(tmp_path, monkeypatch):
    config = replace(AppConfig.for_root(tmp_path), codex_model="gpt-test")
    config.ensure()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    calls = []
    class CountingPipeline(FakeLocalPipeline):
        count = 0
        def prepare(self, *args):
            self.count += 1
            return super().prepare(*args)
    pipeline = CountingPipeline()
    def fake_popen(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps({"segments": [{"id": "s1", "text": "سلام", "speaker": "مشتری", "uncertain": False, "notes": ""}]}), encoding="utf-8")
        process = HangingProcess()
        process.returncode = 1 if len(calls) == 1 else 0
        return process
    monkeypatch.setattr("callforge.codex_runner.shutil.which", lambda _: "codex")
    monkeypatch.setattr("callforge.codex_runner.subprocess.Popen", fake_popen)
    result = CodexRunner(config, pipeline).run(audio, config.logs / "run.jsonl", config.logs / "run.stderr")
    assert result.returncode == 0
    assert pipeline.count == 1
    assert len(calls) == 2
    assert result.quality["automated_review_complete"] is True
    assert audio.with_suffix(".md").exists()


def test_failed_attempt_output_cannot_satisfy_next_empty_attempt(tmp_path, monkeypatch):
    config = replace(AppConfig.for_root(tmp_path), codex_model="gpt-test")
    config.ensure()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    attempts = []
    def fake_popen(command, **kwargs):
        attempts.append(command)
        process = HangingProcess()
        process.returncode = 1 if len(attempts) == 1 else 0
        if len(attempts) == 1:
            Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps({"segments": [
                {"id": "s1", "text": "سلام", "speaker": "مشتری", "uncertain": False, "notes": ""}]}))
        return process
    monkeypatch.setattr("callforge.codex_runner.shutil.which", lambda _: "codex")
    monkeypatch.setattr("callforge.codex_runner.subprocess.Popen", fake_popen)
    result = CodexRunner(config, FakeLocalPipeline()).run(audio, config.logs / "run.jsonl", config.logs / "run.stderr")
    assert result.returncode != 0
    assert not audio.with_suffix(".md").exists()


def test_external_markdown_edit_during_codex_is_preserved(tmp_path, monkeypatch):
    import pytest
    from callforge.quality import ArtifactConflictError
    config = replace(AppConfig.for_root(tmp_path), codex_model="gpt-test")
    config.ensure()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    markdown = audio.with_suffix(".md")
    markdown.write_text("old", encoding="utf-8")
    process = HangingProcess(); process.returncode = 0
    def fake_popen(command, **kwargs):
        markdown.write_text("user external edit", encoding="utf-8")
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps({"segments": [
            {"id": "s1", "text": "سلام", "speaker": "مشتری", "uncertain": False, "notes": ""}]}), encoding="utf-8")
        return process
    monkeypatch.setattr("callforge.codex_runner.shutil.which", lambda _: "codex")
    monkeypatch.setattr("callforge.codex_runner.subprocess.Popen", fake_popen)
    with pytest.raises(ArtifactConflictError):
        CodexRunner(config, FakeLocalPipeline()).run(audio, config.logs / "run.jsonl", config.logs / "run.stderr")
    assert markdown.read_text() == "user external edit"
    assert list(config.runs.glob("*/transcript.md"))
