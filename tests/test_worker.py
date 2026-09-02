from pathlib import Path

from callforge.codex_runner import CodexResult
from callforge.config import AppConfig
from callforge.db import Database
from callforge.metadata import extract_audio_metadata
from callforge.worker import run_batch


class FakeRunner:
    def run(self, audio_path: Path, log_path: Path, stderr_path: Path) -> CodexResult:
        audio_path.with_suffix(".md").write_text(
            "# متن تماس\n\n## مکالمه\n\n**کارشناس پشتیبانی:** سلام\n\n**مشتری:** سلام",
            encoding="utf-8",
        )
        log_path.write_text('{"type":"thread.started","thread_id":"fake"}\n', encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return CodexResult(0, "fake", "", "")


def test_parallel_batch_persists_files_and_database(tmp_path: Path):
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio_paths = []
    for index in range(3):
        audio = tmp_path / f"external-20{index}-123-20260408-14325{index}-id.mp3"
        audio.write_bytes(f"fake-{index}".encode())
        database.upsert_audio(extract_audio_metadata(audio, tmp_path))
        audio_paths.append(audio)
    result, messages = run_batch(config, database, 3, 2, runner=FakeRunner())
    assert result.claimed == result.completed == 3
    assert result.failed == 0
    assert len(messages) == 3
    assert all(path.with_suffix(".md").is_file() for path in audio_paths)
    assert database.counts()["current_transcripts"] == 3

