from dataclasses import replace
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


def test_zero_duration_audio_is_inventory_only_and_never_claimed(tmp_path: Path):
    audio = tmp_path / "external-208-123-20260408-143257-zero.mp3"
    audio.write_bytes(b"header only")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    metadata = replace(
        extract_audio_metadata(audio, tmp_path),
        duration_seconds=0.144,
        metadata_error=None,
    )
    audio_id, _, _ = database.upsert_audio(metadata)

    detail = database.audio_file_detail(audio_id)
    assert detail["job_status"] == "skipped"
    assert database.queue_transcription(audio_id) == "skipped"
    assert database.claim_jobs(5, "test-worker", 300) == []
    markdown = audio.with_suffix(".md")
    markdown.write_text("# متن قدیمی", encoding="utf-8")
    assert database.import_markdown(audio_id, markdown) is None
    assert database.counts() == {
        "total_audio_files": 1,
        "audio_files": 0,
        "skipped": 1,
        "transcripts": 0,
        "current_transcripts": 0,
    }
    total, items = database.list_audio_files(status="skipped")
    assert total == 1
    assert items[0]["job_status"] == "skipped"


def test_corrected_duration_requeues_previously_skipped_audio(tmp_path: Path):
    audio = tmp_path / "out-09120000000-208-20260408-143257-zero.mp3"
    audio.write_bytes(b"audio")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    metadata = replace(extract_audio_metadata(audio, tmp_path), duration_seconds=0.144)
    audio_id, _, _ = database.upsert_audio(metadata)
    assert database.audio_file_detail(audio_id)["job_status"] == "skipped"

    database.upsert_audio(replace(metadata, duration_seconds=1.0))
    assert database.audio_file_detail(audio_id)["job_status"] == "pending"


def test_historical_transcript_is_retained_but_hidden_after_audio_becomes_skipped(tmp_path: Path):
    audio = tmp_path / "external-208-123-20260408-143257-history.mp3"
    audio.write_bytes(b"audio")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    metadata = replace(extract_audio_metadata(audio, tmp_path), duration_seconds=2.0)
    audio_id, _, _ = database.upsert_audio(metadata)
    markdown = audio.with_suffix(".md")
    markdown.write_text("# متن قدیمی", encoding="utf-8")
    assert database.import_markdown(audio_id, markdown) is not None

    database.upsert_audio(replace(metadata, duration_seconds=0.144))
    detail = database.audio_file_detail(audio_id)
    with database.connect() as connection:
        retained = connection.execute(
            "SELECT COUNT(*) FROM transcripts WHERE audio_file_id=?", (audio_id,)
        ).fetchone()[0]

    assert detail["job_status"] == "skipped"
    assert detail["transcript_id"] is None
    assert detail["transcript_content"] is None
    assert retained == 1
    assert database.recent_transcripts(10) == []
    assert database.counts()["current_transcripts"] == 0


def test_scoped_reset_cascades_database_relations_but_keeps_source_files(tmp_path: Path):
    selected_directory = tmp_path / "2026" / "04"
    selected_directory.mkdir(parents=True)
    selected_audio = selected_directory / "external-201-123-20260419-193423-a.mp3"
    remaining_audio = tmp_path / "external-202-456-20260419-193423-b.mp3"
    selected_audio.write_bytes(b"selected")
    remaining_audio.write_bytes(b"remaining")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    selected_id, _, _ = database.upsert_audio(
        extract_audio_metadata(selected_audio, tmp_path)
    )
    remaining_id, _, _ = database.upsert_audio(
        extract_audio_metadata(remaining_audio, tmp_path)
    )
    markdown = selected_audio.with_suffix(".md")
    markdown.write_text("# متن تماس", encoding="utf-8")
    assert database.import_markdown(selected_id, markdown) is not None

    assert database.reset_count(selected_directory) == 1
    assert database.reset(selected_directory) == 1

    assert database.audio_file_detail(selected_id) is None
    assert database.audio_file_detail(remaining_id) is not None
    assert selected_audio.is_file()
    assert markdown.is_file()
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM transcripts WHERE audio_file_id=?", (selected_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE audio_file_id=?", (selected_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE audio_file_id=?", (selected_id,)
        ).fetchone()[0] == 0


def test_full_reset_empties_all_database_data(tmp_path: Path):
    _, database, audio, _ = make_audio(tmp_path)
    assert database.reset_count() == 1
    assert database.reset() == 1
    assert database.reset_count() == 0
    assert audio.is_file()
    with database.connect() as connection:
        for table in ("audio_files", "jobs", "processing_runs", "transcripts", "artifacts"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_reset_refuses_to_delete_an_actively_processing_job(tmp_path: Path):
    _, database, _, audio_id = make_audio(tmp_path)
    assert len(database.claim_jobs(1, "active-worker", 300)) == 1

    try:
        database.reset()
    except RuntimeError as error:
        assert "actively processing" in str(error)
    else:
        raise AssertionError("Expected reset to reject an active job")

    assert database.audio_file_detail(audio_id) is not None


def test_schema_one_is_migrated_without_deleting_database(tmp_path: Path):
    database = Database(tmp_path / "legacy.sqlite3")
    legacy_schema = SCHEMA.replace("    audio_content_hash TEXT NOT NULL,\n", "").replace(
        "PRAGMA user_version = 4;", "PRAGMA user_version = 1;"
    )
    with database.connect() as connection:
        connection.executescript(legacy_schema)
    database.initialize()
    with database.connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(transcripts)")
        }
    assert version == 4
    assert "audio_content_hash" in columns


def test_schema_two_call_directions_are_migrated_from_filename(tmp_path: Path):
    database = Database(tmp_path / "legacy-directions.sqlite3")
    legacy_schema = SCHEMA.replace(
        "direction IN ('inbound', 'outbound', 'internal')",
        "direction IN ('inbound', 'outbound')",
    ).replace("PRAGMA user_version = 4;", "PRAGMA user_version = 2;")
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

    assert version == 4
    assert directions == {
        "external-201-09013663769-20260419-193423-id.mp3": "inbound",
        "internal-202-205-20260428-143902-id.mp3": "internal",
        "out-09124975161-206-20260816-132339-id.mp3": "outbound",
    }
    assert foreign_key_errors == []


def test_schema_three_migrates_zero_duration_jobs_to_skipped(tmp_path: Path):
    database = Database(tmp_path / "legacy-zero-duration.sqlite3")
    legacy_schema = SCHEMA.replace(
        "status IN ('pending', 'running', 'completed', 'failed', 'skipped')",
        "status IN ('pending', 'running', 'completed', 'failed')",
    ).replace("PRAGMA user_version = 4;", "PRAGMA user_version = 3;")
    with database.connect() as connection:
        connection.executescript(legacy_schema)
        for filename, duration in (("zero.mp3", 0.144), ("normal.mp3", 2.0)):
            cursor = connection.execute(
                """
                INSERT INTO audio_files(
                    relative_path, absolute_path, filename, size_bytes, mtime_ns,
                    content_sha256, duration_seconds, discovered_at, updated_at
                ) VALUES (?, ?, ?, 1, 1, ?, ?, '2026-01-01', '2026-01-01')
                """,
                (filename, str(tmp_path / filename), filename, filename, duration),
            )
            connection.execute(
                """
                INSERT INTO jobs(
                    audio_file_id, stage, status, created_at, updated_at
                ) VALUES (?, 'transcribe', 'pending', '2026-01-01', '2026-01-01')
                """,
                (cursor.lastrowid,),
            )

    database.initialize()
    with database.connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        statuses = {
            row["filename"]: row["status"]
            for row in connection.execute(
                "SELECT a.filename, j.status FROM jobs j JOIN audio_files a ON a.id=j.audio_file_id"
            )
        }
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert version == 4
    assert statuses == {"zero.mp3": "skipped", "normal.mp3": "pending"}
    assert foreign_key_errors == []
