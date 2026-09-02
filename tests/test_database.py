from pathlib import Path

from callforge.config import AppConfig
from callforge.db import SCHEMA, Database
from callforge.metadata import extract_audio_metadata


def make_audio(tmp_path: Path) -> tuple[AppConfig, Database, Path, int]:
    audio = tmp_path / "external-208-123-20260408-143257-id.mp3"
    audio.write_bytes(b"fake mp3")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, tmp_path))
    return config, database, audio, audio_id


def test_claim_and_store_related_versioned_transcript(tmp_path: Path):
    config, database, audio, audio_id = make_audio(tmp_path)
    jobs = database.claim_jobs(5, "test-worker", 300)
    assert len(jobs) == 1
    run_id = database.start_run(
        jobs[0], "test-worker", config.logs / "run.jsonl", config.logs / "run.stderr"
    )
    markdown = audio.with_suffix(".md")
    markdown.write_text("# متن تماس\n\n## مکالمه\n\n**مشتری:** سلام", encoding="utf-8")
    transcript_id = database.complete_run(
        jobs[0], run_id, markdown.read_text(encoding="utf-8"), markdown, "fa", "thread-1"
    )
    with database.connect() as connection:
        transcript = connection.execute(
            "SELECT * FROM transcripts WHERE id=?", (transcript_id,)
        ).fetchone()
        artifact = connection.execute(
            "SELECT * FROM artifacts WHERE transcript_id=?", (transcript_id,)
        ).fetchone()
        job = connection.execute("SELECT * FROM jobs WHERE audio_file_id=?", (audio_id,)).fetchone()
    assert transcript["audio_file_id"] == audio_id
    assert transcript["content"] == markdown.read_text(encoding="utf-8")
    assert artifact["path"] == str(markdown.resolve())
    assert job["status"] == "completed"


def test_existing_sibling_markdown_is_imported_once(tmp_path: Path):
    _, database, audio, audio_id = make_audio(tmp_path)
    markdown = audio.with_suffix(".md")
    markdown.write_text("## مکالمه\n\n**کارشناس پشتیبانی:** الو", encoding="utf-8")
    assert database.import_markdown(audio_id, markdown) is not None
    assert database.import_markdown(audio_id, markdown) is None
    assert database.counts()["transcripts"] == 1


def test_interactive_queue_is_distinct_from_scan_pending(tmp_path: Path):
    _, database, _, audio_id = make_audio(tmp_path)
    initial = database.audio_file_detail(audio_id)
    assert initial["job_status"] == "pending"
    assert initial["job_priority"] == 0

    assert database.queue_transcription(audio_id) == "queued"
    queued = database.audio_file_detail(audio_id)
    assert queued["job_status"] == "pending"
    assert queued["job_priority"] == 1000


def test_schema_one_is_migrated_without_deleting_database(tmp_path: Path):
    database = Database(tmp_path / "legacy.sqlite3")
    legacy_schema = SCHEMA.replace("    audio_content_hash TEXT NOT NULL,\n", "").replace(
        "PRAGMA user_version = 3;", "PRAGMA user_version = 1;"
    )
    with database.connect() as connection:
        connection.executescript(legacy_schema)
    database.initialize()
    with database.connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(transcripts)")
        }
    assert version == 3
    assert "audio_content_hash" in columns


def test_schema_two_call_directions_are_migrated_from_filename(tmp_path: Path):
    database = Database(tmp_path / "legacy-directions.sqlite3")
    legacy_schema = SCHEMA.replace(
        "direction IN ('inbound', 'outbound', 'internal')",
        "direction IN ('inbound', 'outbound')",
    ).replace("PRAGMA user_version = 3;", "PRAGMA user_version = 2;")
    with database.connect() as connection:
        connection.executescript(legacy_schema)
        rows = [
            (
                "external-201-09013663769-20260419-193423-id.mp3",
                "outbound",
            ),
            ("internal-202-205-20260428-143902-id.mp3", "inbound"),
            ("out-09124975161-206-20260816-132339-id.mp3", None),
        ]
        for filename, direction in rows:
            connection.execute(
                """
                INSERT INTO audio_files(
                    relative_path, absolute_path, filename, size_bytes, mtime_ns,
                    content_sha256, direction, discovered_at, updated_at
                ) VALUES (?, ?, ?, 1, 1, ?, ?, '2026-01-01', '2026-01-01')
                """,
                (filename, str(tmp_path / filename), filename, filename, direction),
            )

    database.initialize()

    with database.connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        directions = {
            row["filename"]: row["direction"]
            for row in connection.execute("SELECT filename, direction FROM audio_files")
        }
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert version == 3
    assert directions == {
        "external-201-09013663769-20260419-193423-id.mp3": "inbound",
        "internal-202-205-20260428-143902-id.mp3": "internal",
        "out-09124975161-206-20260816-132339-id.mp3": "outbound",
    }
    assert foreign_key_errors == []
