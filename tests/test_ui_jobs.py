from pathlib import Path

from callforge.codex_runner import CodexResult
from callforge.config import AppConfig
from callforge.db import Database
from callforge.metadata import extract_audio_metadata
from callforge.ui_jobs import UITranscriptionService


class FakeRunner:
    def run(self, audio_path: Path, log_path: Path, stderr_path: Path) -> CodexResult:
        audio_path.with_suffix(".md").write_text(
            "# متن تماس\n\n## مکالمه\n\n**کارشناس پشتیبانی:** نسخه جدید",
            encoding="utf-8",
        )
        log_path.write_text('{"type":"thread.started","thread_id":"ui-test"}\n', encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return CodexResult(0, "ui-test", "", "")


class FailingRunner:
    def __init__(self):
        self.calls = 0

    def run(self, audio_path: Path, log_path: Path, stderr_path: Path) -> CodexResult:
        self.calls += 1
        log_path.write_text("", encoding="utf-8")
        stderr_path.write_text("model failure", encoding="utf-8")
        return CodexResult(1, None, "", "model failure")


def test_ui_service_processes_only_requested_audio(tmp_path: Path):
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio_ids = []
    for index in range(2):
        audio = tmp_path / f"external-20{index}-123-20260408-14325{index}-id.mp3"
        audio.write_bytes(f"audio-{index}".encode())
        audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, tmp_path))
        audio_ids.append(audio_id)

    service = UITranscriptionService(config, database, runner=FakeRunner())
    try:
        assert service.request(audio_ids[1]) == "queued"
        result = service.wait(audio_ids[1], timeout=3)
        assert result and result[0] is True
    finally:
        service.shutdown(wait=True)

    first = database.audio_file_detail(audio_ids[0])
    second = database.audio_file_detail(audio_ids[1])
    assert first["job_status"] == "pending"
    assert first["transcript_id"] is None
    assert second["job_status"] == "completed"
    assert second["transcript_content"].endswith("نسخه جدید")


def test_ui_service_retries_until_terminal_failure(tmp_path: Path):
    audio = tmp_path / "internal-201-123-20260408-143250-id.mp3"
    audio.write_bytes(b"audio")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, tmp_path))
    runner = FailingRunner()
    service = UITranscriptionService(config, database, runner=runner)
    try:
        assert service.request(audio_id) == "queued"
        result = service.wait(audio_id, timeout=3)
        assert result and result[0] is False
    finally:
        service.shutdown(wait=True)
    assert runner.calls == 3
    assert database.audio_file_detail(audio_id)["job_status"] == "failed"
