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


def test_conflicting_external_edit_does_not_automatically_retry(tmp_path):
    from callforge.quality import ArtifactConflictError
    class ConflictRunner:
        def run(self, *args):
            raise ArtifactConflictError("external edit")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"test")
    audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, tmp_path))
    result, _ = run_batch(config, database, 1, 1, runner=ConflictRunner())
    assert result.failed == 1
    assert database.audio_file_detail(audio_id)["job_status"] == "failed"
    assert database.claim_jobs(1, "retry", 100) == []


def test_speaker_failure_keeps_diagnostics_without_repeating_whisper(tmp_path):
    from callforge.speaker_pipeline import SpeakerProcessingError
    class FailedSpeakerRunner:
        def run(self, *args):
            work = config.runs / "speaker-failure"
            work.mkdir()
            (work / "speaker-pipeline.json").write_text('{"status":"failed"}')
            raise SpeakerProcessingError("role timeout", work)
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"test")
    audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, tmp_path))
    result, _ = run_batch(config, database, 1, 1, runner=FailedSpeakerRunner())
    assert result.failed == 1
    assert database.audio_file_detail(audio_id)["job_status"] == "failed"
    assert database.claim_jobs(1, "retry", 100) == []
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM run_evidence").fetchone()[0] == 1


def test_incomplete_review_never_creates_transcript_and_keeps_failure_evidence(tmp_path):
    class IncompleteRunner:
        def run(self, *args):
            work = config.runs / "failed-review"
            work.mkdir()
            (work / "evidence.json").write_text('{"segments": []}')
            (work / "review-attempt-1.json").write_text('{"segments": [')
            return CodexResult(0, "failed-thread", "", "",  # Even an erroneous zero exit is rejected.
                               {"automated_review_complete": False, "review_error": "timeout"}, work)
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"test")
    audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, tmp_path))
    result, _ = run_batch(config, database, 1, 1, runner=IncompleteRunner())
    assert result.failed == 1
    assert database.audio_file_detail(audio_id)["job_status"] == "failed"
    assert database.counts()["current_transcripts"] == 0
    assert not audio.with_suffix(".md").exists()
    assert database.claim_jobs(1, "retry", 100) == []
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM run_evidence").fetchone()[0] == 1
        assert connection.execute("SELECT codex_thread_id FROM processing_runs").fetchone()[0] == "failed-thread"
