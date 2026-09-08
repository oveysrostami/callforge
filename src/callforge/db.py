from __future__ import annotations

import hashlib
import json
import sqlite3
import os
import tempfile
import math
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

from callforge.metadata import AudioMetadata
from callforge.quality import file_hash, render_markdown, text_flags


MIN_PROCESSABLE_DURATION_SECONDS = 0.5


def is_zero_duration(duration_seconds: float | None) -> bool:
    return duration_seconds is not None and duration_seconds < MIN_PROCESSABLE_DURATION_SECONDS


SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_files (
    id INTEGER PRIMARY KEY,
    absolute_path TEXT NOT NULL UNIQUE,
    relative_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    direction TEXT CHECK(direction IN ('inbound', 'outbound', 'internal') OR direction IS NULL),
    agent_extension TEXT,
    remote_number TEXT,
    call_id TEXT,
    recorded_at TEXT,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    duration_seconds REAL,
    codec TEXT,
    bitrate INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
    metadata_error TEXT,
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    priority INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    claimed_by TEXT,
    claimed_at TEXT,
    lease_expires_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(audio_file_id, stage)
);

CREATE TABLE IF NOT EXISTS processing_runs (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
    worker_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    codex_thread_id TEXT,
    log_path TEXT,
    stderr_path TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    processing_run_id INTEGER REFERENCES processing_runs(id) ON DELETE SET NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    format TEXT NOT NULL DEFAULT 'markdown',
    language TEXT NOT NULL DEFAULT 'fa',
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    audio_content_hash TEXT NOT NULL,
    markdown_path TEXT NOT NULL,
    unclear_count INTEGER NOT NULL DEFAULT 0,
    is_current INTEGER NOT NULL DEFAULT 1 CHECK(is_current IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE(audio_file_id, version)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY,
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    processing_run_id INTEGER REFERENCES processing_runs(id) ON DELETE SET NULL,
    transcript_id INTEGER REFERENCES transcripts(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(stage, status, priority DESC, id);
CREATE INDEX IF NOT EXISTS idx_transcripts_current ON transcripts(audio_file_id, is_current);
CREATE INDEX IF NOT EXISTS idx_runs_audio ON processing_runs(audio_file_id, started_at);
PRAGMA user_version = 4;
"""

QUALITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript_reviews (
    transcript_id INTEGER PRIMARY KEY REFERENCES transcripts(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('needs_review', 'in_review', 'approved')),
    reviewer TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    data_json TEXT NOT NULL DEFAULT '{}',
    base_transcript_id INTEGER REFERENCES transcripts(id) ON DELETE SET NULL,
    markdown_synced INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_evidence (
    processing_run_id INTEGER PRIMARY KEY REFERENCES processing_runs(id) ON DELETE CASCADE,
    directory TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
PRAGMA user_version = 5;
"""

MIGRATION_1_TO_2 = """
ALTER TABLE transcripts ADD COLUMN audio_content_hash TEXT;
UPDATE transcripts
SET audio_content_hash = (
    SELECT audio_files.content_sha256
    FROM audio_files
    WHERE audio_files.id = transcripts.audio_file_id
);
PRAGMA user_version = 2;
"""

MIGRATION_2_TO_3 = """
PRAGMA foreign_keys = OFF;
CREATE TABLE audio_files_v3 (
    id INTEGER PRIMARY KEY,
    absolute_path TEXT NOT NULL UNIQUE,
    relative_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    direction TEXT CHECK(direction IN ('inbound', 'outbound', 'internal') OR direction IS NULL),
    agent_extension TEXT,
    remote_number TEXT,
    call_id TEXT,
    recorded_at TEXT,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    duration_seconds REAL,
    codec TEXT,
    bitrate INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
    metadata_error TEXT,
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
INSERT INTO audio_files_v3 (
    id, absolute_path, relative_path, filename, direction, agent_extension,
    remote_number, call_id, recorded_at, size_bytes, mtime_ns, content_sha256,
    duration_seconds, codec, bitrate, sample_rate, channels, metadata_error,
    discovered_at, updated_at
)
SELECT
    id, absolute_path, relative_path, filename,
    CASE
        WHEN lower(filename) LIKE 'external-%' THEN 'inbound'
        WHEN lower(filename) LIKE 'internal-%' THEN 'internal'
        WHEN lower(filename) LIKE 'out-%' THEN 'outbound'
        ELSE direction
    END,
    agent_extension, remote_number, call_id, recorded_at, size_bytes, mtime_ns,
    content_sha256, duration_seconds, codec, bitrate, sample_rate, channels,
    metadata_error, discovered_at, updated_at
FROM audio_files;
DROP TABLE audio_files;
ALTER TABLE audio_files_v3 RENAME TO audio_files;
PRAGMA foreign_keys = ON;
PRAGMA user_version = 3;
"""

MIGRATION_3_TO_4 = """
PRAGMA foreign_keys = OFF;
CREATE TABLE jobs_v4 (
    id INTEGER PRIMARY KEY,
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    priority INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    claimed_by TEXT,
    claimed_at TEXT,
    lease_expires_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(audio_file_id, stage)
);
INSERT INTO jobs_v4 (
    id, audio_file_id, stage, status, priority, attempts, max_attempts,
    claimed_by, claimed_at, lease_expires_at, last_error, created_at, updated_at
)
SELECT
    j.id, j.audio_file_id, j.stage,
    CASE
        WHEN j.status != 'running'
         AND a.duration_seconds IS NOT NULL
         AND a.duration_seconds < 0.5 THEN 'skipped'
        ELSE j.status
    END,
    CASE
        WHEN a.duration_seconds IS NOT NULL AND a.duration_seconds < 0.5 THEN 0
        ELSE j.priority
    END,
    j.attempts, j.max_attempts, j.claimed_by, j.claimed_at, j.lease_expires_at,
    j.last_error, j.created_at, j.updated_at
FROM jobs j
JOIN audio_files a ON a.id = j.audio_file_id;
DROP TABLE jobs;
ALTER TABLE jobs_v4 RENAME TO jobs;
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(stage, status, priority DESC, id);
PRAGMA foreign_keys = ON;
PRAGMA user_version = 4;
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 5:
                raise RuntimeError(
                    f"Database schema {version} is newer than this CallForge supports"
                )
            if 0 < version < 5:
                backup_directory = self.path.parent / "backups"
                backup_directory.mkdir(exist_ok=True)
                descriptor, backup_path = tempfile.mkstemp(prefix=f"schema-{version}-", suffix=".sqlite3", dir=backup_directory)
                os.close(descriptor)
                destination = sqlite3.connect(backup_path)
                try:
                    connection.backup(destination)
                finally:
                    destination.close()
            if version == 0:
                connection.executescript(SCHEMA)
                version = 4
            if version == 1:
                connection.executescript(MIGRATION_1_TO_2)
                version = 2
            if version == 2:
                connection.executescript(MIGRATION_2_TO_3)
                version = 3
            if version == 3:
                connection.executescript(MIGRATION_3_TO_4)
            connection.executescript(QUALITY_SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def upsert_audio(
        self, metadata: AudioMetadata, max_attempts: int = 3
    ) -> tuple[int, bool, bool]:
        values = asdict(metadata)
        now = utcnow()
        desired_status = "skipped" if is_zero_duration(metadata.duration_seconds) else "pending"
        with self.transaction() as connection:
            previous = connection.execute(
                "SELECT id, content_sha256, size_bytes, mtime_ns FROM audio_files WHERE absolute_path = ?",
                (metadata.absolute_path,),
            ).fetchone()
            created = previous is None
            changed = created or previous["content_sha256"] != metadata.content_sha256
            columns = list(values)
            placeholders = ", ".join("?" for _ in columns)
            assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "absolute_path")
            connection.execute(
                f"INSERT INTO audio_files ({', '.join(columns)}, discovered_at, updated_at) "
                f"VALUES ({placeholders}, ?, ?) ON CONFLICT(absolute_path) DO UPDATE SET {assignments}, updated_at=excluded.updated_at",
                (*values.values(), now, now),
            )
            audio_id = int(
                connection.execute(
                    "SELECT id FROM audio_files WHERE absolute_path = ?", (metadata.absolute_path,)
                ).fetchone()["id"]
            )
            connection.execute(
                "INSERT INTO jobs (audio_file_id, stage, status, max_attempts, created_at, updated_at) "
                "VALUES (?, 'transcribe', ?, ?, ?, ?) ON CONFLICT(audio_file_id, stage) DO NOTHING",
                (audio_id, desired_status, max_attempts, now, now),
            )
            if changed and previous is not None:
                connection.execute(
                    "UPDATE transcripts SET is_current=0 WHERE audio_file_id=? AND is_current=1",
                    (audio_id,),
                )
                connection.execute(
                    "UPDATE jobs SET status=?, priority=0, attempts=0, claimed_by=NULL, claimed_at=NULL, "
                    "lease_expires_at=NULL, last_error=NULL, updated_at=? "
                    "WHERE audio_file_id=? AND stage='transcribe' AND status != 'running'",
                    (desired_status, now, audio_id),
                )
            elif desired_status == "skipped":
                connection.execute(
                    "UPDATE jobs SET status='skipped', priority=0, claimed_by=NULL, claimed_at=NULL, "
                    "lease_expires_at=NULL, last_error=NULL, updated_at=? "
                    "WHERE audio_file_id=? AND stage='transcribe' AND status != 'running'",
                    (now, audio_id),
                )
            else:
                connection.execute(
                    "UPDATE jobs SET status='pending', priority=0, attempts=0, last_error=NULL, updated_at=? "
                    "WHERE audio_file_id=? AND stage='transcribe' AND status='skipped'",
                    (now, audio_id),
                )
            return audio_id, changed, created

    def import_markdown(self, audio_id: int, path: Path, language: str = "fa") -> int | None:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return None
        with self.transaction() as connection:
            job = connection.execute(
                "SELECT status FROM jobs WHERE audio_file_id=? AND stage='transcribe'",
                (audio_id,),
            ).fetchone()
            if job is not None and job["status"] in {"skipped", "running"}:
                return None
            transcript_id, inserted = self._store_transcript(
                connection, audio_id, None, content, path, language, "existing_markdown"
            )
            connection.execute(
                "UPDATE jobs SET status='completed', last_error=NULL, updated_at=? "
                "WHERE audio_file_id=? AND stage='transcribe'",
                (utcnow(), audio_id),
            )
            return transcript_id if inserted else None

    def _store_transcript(
        self,
        connection: sqlite3.Connection,
        audio_id: int,
        run_id: int | None,
        content: str,
        markdown_path: Path,
        language: str,
        source: str,
        force_version: bool = False,
    ) -> tuple[int, bool]:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        audio_content_hash = str(
            connection.execute(
                "SELECT content_sha256 FROM audio_files WHERE id=?", (audio_id,)
            ).fetchone()["content_sha256"]
        )
        current = connection.execute(
            "SELECT id, content_hash FROM transcripts WHERE audio_file_id=? AND is_current=1",
            (audio_id,),
        ).fetchone()
        if current and current["content_hash"] == content_hash and not force_version:
            return int(current["id"]), False
        version = int(
            connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS version FROM transcripts WHERE audio_file_id=?",
                (audio_id,),
            ).fetchone()["version"]
        )
        connection.execute("UPDATE transcripts SET is_current=0 WHERE audio_file_id=?", (audio_id,))
        cursor = connection.execute(
            "INSERT INTO transcripts (audio_file_id, processing_run_id, version, content, language, "
            "source, content_hash, audio_content_hash, markdown_path, unclear_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                audio_id,
                run_id,
                version,
                content,
                language,
                source,
                content_hash,
                audio_content_hash,
                str(markdown_path.resolve()),
                content.count("[نامفهوم]"),
                utcnow(),
            ),
        )
        transcript_id = int(cursor.lastrowid)
        connection.execute(
            "INSERT INTO artifacts (audio_file_id, processing_run_id, transcript_id, kind, path, "
            "content_hash, metadata_json, created_at) VALUES (?, ?, ?, 'transcript_markdown', ?, ?, ?, ?)",
            (
                audio_id,
                run_id,
                transcript_id,
                str(markdown_path.resolve()),
                content_hash,
                json.dumps({"language": language, "version": version}),
                utcnow(),
            ),
        )
        return transcript_id, True

    def claim_jobs(self, limit: int, worker_id: str, lease_seconds: int) -> list[sqlite3.Row]:
        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE jobs SET status='pending', claimed_by=NULL, claimed_at=NULL, lease_expires_at=NULL, "
                "last_error='Previous worker lease expired', updated_at=? "
                "WHERE status='running' AND lease_expires_at < ?",
                (now_text, now_text),
            )
            rows = connection.execute(
                "SELECT j.*, a.absolute_path, a.filename FROM jobs j "
                "JOIN audio_files a ON a.id=j.audio_file_id "
                "WHERE j.stage='transcribe' AND j.status='pending' AND j.attempts < j.max_attempts "
                "AND (a.duration_seconds IS NULL OR a.duration_seconds >= 0.5) "
                "ORDER BY j.priority DESC, j.id LIMIT ?",
                (limit,),
            ).fetchall()
            claimed: list[sqlite3.Row] = []
            for row in rows:
                updated = connection.execute(
                    "UPDATE jobs SET status='running', attempts=attempts+1, claimed_by=?, claimed_at=?, "
                    "lease_expires_at=?, updated_at=? WHERE id=? AND status='pending'",
                    (worker_id, now_text, lease, now_text, row["id"]),
                ).rowcount
                if updated:
                    claimed.append(
                        connection.execute(
                            "SELECT j.*, a.absolute_path, a.filename FROM jobs j "
                            "JOIN audio_files a ON a.id=j.audio_file_id WHERE j.id=?",
                            (row["id"],),
                        ).fetchone()
                    )
            return claimed

    def queue_transcription(self, audio_id: int) -> str | None:
        """Queue one exact file from an interactive request.

        Returns ``queued``, ``running``, ``skipped``, or ``None`` when the audio id does not exist.
        """
        now = utcnow()
        with self.transaction() as connection:
            audio = connection.execute(
                "SELECT id, duration_seconds FROM audio_files WHERE id=?", (audio_id,)
            ).fetchone()
            if audio is None:
                return None
            if is_zero_duration(audio["duration_seconds"]):
                connection.execute(
                    "UPDATE jobs SET status='skipped', priority=0, claimed_by=NULL, claimed_at=NULL, "
                    "lease_expires_at=NULL, last_error=NULL, updated_at=? "
                    "WHERE audio_file_id=? AND stage='transcribe' AND status != 'running'",
                    (now, audio_id),
                )
                return "skipped"
            job = connection.execute(
                "SELECT id, status FROM jobs WHERE audio_file_id=? AND stage='transcribe'",
                (audio_id,),
            ).fetchone()
            if job is None:
                connection.execute(
                    "INSERT INTO jobs (audio_file_id, stage, status, priority, attempts, max_attempts, "
                    "created_at, updated_at) VALUES (?, 'transcribe', 'pending', 1000, 0, 3, ?, ?)",
                    (audio_id, now, now),
                )
                return "queued"
            if job["status"] == "running":
                return "running"
            connection.execute(
                "UPDATE jobs SET status='pending', priority=1000, attempts=0, claimed_by=NULL, "
                "claimed_at=NULL, lease_expires_at=NULL, last_error=NULL, updated_at=? WHERE id=?",
                (now, job["id"]),
            )
            return "queued"

    def claim_audio_job(
        self, audio_id: int, worker_id: str, lease_seconds: int
    ) -> sqlite3.Row | None:
        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE jobs SET status='pending', claimed_by=NULL, claimed_at=NULL, "
                "lease_expires_at=NULL, last_error='Previous worker lease expired', updated_at=? "
                "WHERE audio_file_id=? AND stage='transcribe' AND status='running' "
                "AND lease_expires_at < ?",
                (now_text, audio_id, now_text),
            )
            job = connection.execute(
                "SELECT j.*, a.absolute_path, a.filename FROM jobs j "
                "JOIN audio_files a ON a.id=j.audio_file_id "
                "WHERE j.audio_file_id=? AND j.stage='transcribe' AND j.status='pending' "
                "AND (a.duration_seconds IS NULL OR a.duration_seconds >= 0.5) "
                "AND j.attempts < j.max_attempts",
                (audio_id,),
            ).fetchone()
            if job is None:
                return None
            updated = connection.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, claimed_by=?, claimed_at=?, "
                "lease_expires_at=?, updated_at=? WHERE id=? AND status='pending'",
                (worker_id, now_text, lease, now_text, job["id"]),
            ).rowcount
            if not updated:
                return None
            return connection.execute(
                "SELECT j.*, a.absolute_path, a.filename FROM jobs j "
                "JOIN audio_files a ON a.id=j.audio_file_id WHERE j.id=?",
                (job["id"],),
            ).fetchone()

    def start_run(self, job: sqlite3.Row, worker_id: str, log_path: Path, stderr_path: Path) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO processing_runs (job_id, audio_file_id, stage, status, worker_id, attempt, "
                "log_path, stderr_path, started_at) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)",
                (
                    job["id"],
                    job["audio_file_id"],
                    job["stage"],
                    worker_id,
                    job["attempts"],
                    str(log_path),
                    str(stderr_path),
                    utcnow(),
                ),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _store_run_evidence(connection, run_id: int, directory: Path | None) -> None:
        if directory is None:
            return
        payload = {}
        for path in directory.glob("*.json"):
            original = path.read_text(encoding="utf-8", errors="replace")
            try:
                payload[path.name] = json.loads(original)
            except json.JSONDecodeError:
                payload[path.name] = {"invalid_json": True, "original": original}
        connection.execute("INSERT OR REPLACE INTO run_evidence VALUES (?, ?, ?)",
                           (run_id, str(directory), json.dumps(payload, ensure_ascii=False)))

    def complete_run(
        self,
        job: sqlite3.Row,
        run_id: int,
        content: str,
        markdown_path: Path,
        language: str,
        codex_thread_id: str | None,
        quality: dict | None = None,
        evidence_directory: Path | None = None,
    ) -> int:
        if quality is not None and (quality.get("automated_review_complete") is False
                                    or quality.get("speaker_pipeline_complete") is False):
            raise ValueError("Cannot publish an incomplete automated review")
        with self.transaction() as connection:
            transcript_id, _ = self._store_transcript(
                connection,
                int(job["audio_file_id"]),
                run_id,
                content,
                markdown_path,
                language,
                "codex_skill",
                force_version=quality is not None,
            )
            connection.execute(
                "INSERT OR REPLACE INTO transcript_reviews (transcript_id, status, data_json, markdown_synced, created_at) VALUES (?, 'needs_review', ?, 1, ?)",
                (transcript_id, json.dumps(quality or {}, ensure_ascii=False), utcnow()),
            )
            self._store_run_evidence(connection, run_id, evidence_directory)
            now = utcnow()
            connection.execute(
                "UPDATE processing_runs SET status='completed', codex_thread_id=?, finished_at=? WHERE id=?",
                (codex_thread_id, now, run_id),
            )
            connection.execute(
                "UPDATE jobs SET status='completed', claimed_by=NULL, claimed_at=NULL, lease_expires_at=NULL, "
                "last_error=NULL, updated_at=? WHERE id=?",
                (now, job["id"]),
            )
            return transcript_id

    def fail_run(self, job: sqlite3.Row, run_id: int, error: str, *, retryable: bool = True,
                 evidence_directory: Path | None = None, codex_thread_id: str | None = None) -> None:
        with self.transaction() as connection:
            self._store_run_evidence(connection, run_id, evidence_directory)
            terminal = not retryable or int(job["attempts"]) >= int(job["max_attempts"])
            status = "failed" if terminal else "pending"
            now = utcnow()
            connection.execute(
                "UPDATE processing_runs SET status='failed', error=?, finished_at=?, codex_thread_id=? WHERE id=?",
                (error, now, codex_thread_id, run_id),
            )
            connection.execute(
                "UPDATE jobs SET status=?, claimed_by=NULL, claimed_at=NULL, lease_expires_at=NULL, "
                "last_error=?, updated_at=? WHERE id=?",
                (status, error, now, job["id"]),
            )

    def retry_failed(self) -> int:
        with self.transaction() as connection:
            return connection.execute(
                "UPDATE jobs SET status='pending', attempts=0, last_error=NULL, claimed_by=NULL, "
                "claimed_at=NULL, lease_expires_at=NULL, updated_at=? WHERE status='failed'",
                (utcnow(),),
            ).rowcount

    @staticmethod
    def _audio_rows_in_scope(
        connection: sqlite3.Connection, directory: Path | None
    ) -> list[sqlite3.Row]:
        rows = connection.execute(
            "SELECT a.id, a.absolute_path, j.status AS job_status "
            "FROM audio_files a LEFT JOIN jobs j "
            "ON j.audio_file_id=a.id AND j.stage='transcribe'"
        ).fetchall()
        if directory is None:
            return list(rows)
        target = directory.expanduser().resolve()
        selected: list[sqlite3.Row] = []
        for row in rows:
            audio_path = Path(row["absolute_path"]).expanduser().resolve()
            if audio_path == target or target in audio_path.parents:
                selected.append(row)
        return selected

    def reset_count(self, directory: Path | None = None) -> int:
        with self.connect() as connection:
            return len(self._audio_rows_in_scope(connection, directory))

    def reset(self, directory: Path | None = None) -> int:
        """Delete indexed database records without touching source or Markdown files."""

        with self.transaction() as connection:
            rows = self._audio_rows_in_scope(connection, directory)
            running = [row for row in rows if row["job_status"] == "running"]
            if running:
                raise RuntimeError(
                    f"Cannot reset {len(running)} actively processing file(s). "
                    "Stop the active run or UI worker and try again."
                )
            connection.executemany(
                "DELETE FROM audio_files WHERE id=?",
                ((row["id"],) for row in rows),
            )
            return len(rows)

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            result = {
                "total_audio_files": int(
                    connection.execute("SELECT COUNT(*) FROM audio_files").fetchone()[0]
                ),
                "audio_files": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM audio_files a LEFT JOIN jobs j "
                        "ON j.audio_file_id=a.id AND j.stage='transcribe' "
                        "WHERE j.status IS NULL OR j.status != 'skipped'"
                    ).fetchone()[0]
                ),
            }
            for row in connection.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"):
                result[str(row["status"])] = int(row["count"])
            eligible_transcripts = (
                "FROM transcripts t JOIN jobs j ON j.audio_file_id=t.audio_file_id "
                "AND j.stage='transcribe' WHERE j.status != 'skipped'"
            )
            result["transcripts"] = int(
                connection.execute(f"SELECT COUNT(*) {eligible_transcripts}").fetchone()[0]
            )
            result["current_transcripts"] = int(
                connection.execute(
                    f"SELECT COUNT(*) {eligible_transcripts} AND t.is_current=1"
                ).fetchone()[0]
            )
            return result

    def recent_transcripts(self, limit: int) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT a.relative_path, t.version, t.source, t.markdown_path, t.created_at "
                "FROM transcripts t JOIN audio_files a ON a.id=t.audio_file_id "
                "JOIN jobs j ON j.audio_file_id=a.id AND j.stage='transcribe' "
                "WHERE t.is_current=1 AND j.status != 'skipped' "
                "ORDER BY t.created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def processing_snapshot(self, audio_id: int) -> dict[str, object] | None:
        """Return the queue state and latest run used by the live UI stream."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT a.id AS audio_id, a.filename, "
                "j.id AS job_id, j.status AS job_status, j.attempts, j.max_attempts, "
                "j.claimed_at, j.lease_expires_at, j.last_error, "
                "pr.id AS run_id, pr.status AS run_status, pr.attempt AS run_attempt, "
                "pr.started_at, pr.finished_at, pr.log_path, pr.stderr_path, pr.error AS run_error, "
                "COALESCE((SELECT SUM((julianday(COALESCE(history.finished_at, CURRENT_TIMESTAMP)) "
                "- julianday(history.started_at)) * 86400.0) FROM processing_runs history "
                "WHERE history.audio_file_id=a.id AND history.stage='transcribe'), 0) "
                "AS processing_total_seconds "
                "FROM audio_files a "
                "LEFT JOIN jobs j ON j.audio_file_id=a.id AND j.stage='transcribe' "
                "LEFT JOIN processing_runs pr ON pr.id=("
                "SELECT MAX(latest.id) FROM processing_runs latest "
                "WHERE latest.audio_file_id=a.id AND latest.stage='transcribe') "
                "WHERE a.id=?",
                (audio_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_audio_files(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
        direction: str = "",
        status: str = "",
        transcript: str = "",
        review: str = "",
    ) -> tuple[int, list[dict[str, object]]]:
        conditions: list[str] = []
        parameters: list[object] = []
        if query:
            conditions.append(
                "(a.filename LIKE ? ESCAPE '\\' OR a.relative_path LIKE ? ESCAPE '\\' "
                "OR COALESCE(a.remote_number, '') LIKE ? ESCAPE '\\' "
                "OR COALESCE(a.agent_extension, '') LIKE ? ESCAPE '\\')"
            )
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            parameters.extend([pattern, pattern, pattern, pattern])
        if direction in {"inbound", "outbound", "internal"}:
            conditions.append("a.direction = ?")
            parameters.append(direction)
        if status in {"pending", "running", "completed", "failed", "skipped"}:
            conditions.append("j.status = ?")
            parameters.append(status)
        if transcript == "yes":
            conditions.append("t.id IS NOT NULL")
        elif transcript == "no":
            conditions.append("t.id IS NULL")
        if review in {"needs_review", "in_review", "approved"}:
            conditions.append("t.id IS NOT NULL AND COALESCE(r.status, 'needs_review')=? AND j.status != 'skipped'")
            parameters.append(review)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        joined = (
            "FROM audio_files a "
            "LEFT JOIN jobs j ON j.audio_file_id=a.id AND j.stage='transcribe' "
            "LEFT JOIN transcripts t ON t.audio_file_id=a.id AND t.is_current=1 "
            "AND (j.status IS NULL OR j.status != 'skipped') "
            "LEFT JOIN transcript_reviews r ON r.transcript_id=t.id "
        )
        with self.connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) {joined} {where}", parameters
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT a.id, a.filename, a.relative_path, a.direction, a.agent_extension, "
                "a.remote_number, a.recorded_at, a.duration_seconds, a.size_bytes, a.codec, "
                "a.bitrate, a.sample_rate, a.channels, a.metadata_error, "
                "j.status AS job_status, j.priority AS job_priority, j.attempts, j.max_attempts, j.last_error, "
                "(SELECT MAX(history.id) FROM processing_runs history "
                "WHERE history.audio_file_id=a.id AND history.stage='transcribe') AS latest_run_id, "
                "COALESCE((SELECT SUM((julianday(COALESCE(history.finished_at, CURRENT_TIMESTAMP)) "
                "- julianday(history.started_at)) * 86400.0) FROM processing_runs history "
                "WHERE history.audio_file_id=a.id AND history.stage='transcribe'), 0) "
                "AS processing_total_seconds, "
                "t.id AS transcript_id, t.version AS transcript_version, t.source AS transcript_source, "
                "t.unclear_count, t.created_at AS transcript_created_at, COALESCE(r.status, 'needs_review') AS review_status "
                f"{joined} {where} "
                "ORDER BY (a.recorded_at IS NULL), a.recorded_at DESC, a.id DESC LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return total, [dict(row) for row in rows]

    def audio_file_detail(self, audio_id: int) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT a.*, j.status AS job_status, j.priority AS job_priority, j.attempts, j.max_attempts, j.last_error, "
                "t.id AS transcript_id, t.version AS transcript_version, t.content AS transcript_content, "
                "t.source AS transcript_source, t.unclear_count, t.created_at AS transcript_created_at, "
                "t.markdown_path, pr.id AS latest_run_id, pr.status AS latest_run_status, "
                "pr.attempt AS latest_run_attempt, pr.started_at AS processing_started_at, "
                "pr.finished_at AS processing_finished_at, "
                "COALESCE((SELECT SUM((julianday(COALESCE(history.finished_at, CURRENT_TIMESTAMP)) "
                "- julianday(history.started_at)) * 86400.0) FROM processing_runs history "
                "WHERE history.audio_file_id=a.id AND history.stage='transcribe'), 0) "
                "AS processing_total_seconds "
                "FROM audio_files a "
                "LEFT JOIN jobs j ON j.audio_file_id=a.id AND j.stage='transcribe' "
                "LEFT JOIN transcripts t ON t.audio_file_id=a.id AND t.is_current=1 "
                "AND (j.status IS NULL OR j.status != 'skipped') "
                "LEFT JOIN processing_runs pr ON pr.id=("
                "SELECT MAX(latest.id) FROM processing_runs latest "
                "WHERE latest.audio_file_id=a.id AND latest.stage='transcribe') "
                "WHERE a.id=?",
                (audio_id,),
            ).fetchone()
        return dict(row) if row else None

    def audio_path(self, audio_id: int) -> Path | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT absolute_path FROM audio_files WHERE id=?", (audio_id,)
            ).fetchone()
        return Path(row["absolute_path"]) if row else None

    def review_detail(self, audio_id: int) -> dict:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT t.id, t.version, t.source, t.content, t.is_current, t.created_at, "
                "r.status, r.reviewer, r.notes, r.data_json, r.markdown_synced "
                "FROM transcripts t LEFT JOIN transcript_reviews r ON r.transcript_id=t.id "
                "WHERE t.audio_file_id=? ORDER BY t.version DESC", (audio_id,),
            ).fetchall()
            current = next((row for row in rows if row["is_current"]), None)
            return {
                "transcript_id": current["id"] if current else None,
                "status": (current["status"] or "needs_review") if current else "needs_review",
                "reviewer": current["reviewer"] if current else "",
                "notes": current["notes"] if current else "",
                "data": json.loads(current["data_json"] or "{}") if current else {},
                "markdown_synced": bool(current["markdown_synced"]) if current else False,
                "versions": [{key: row[key] for key in ("id", "version", "source", "content", "created_at", "status", "reviewer", "notes")} for row in rows],
            }

    def save_review(self, audio_id: int, payload: dict) -> int:
        """Append a human revision with optimistic locking; publication is recoverable."""
        status = payload.get("status")
        reviewer = payload.get("reviewer", "")
        notes = payload.get("notes", "")
        if status not in {"in_review", "approved"}:
            raise ValueError("Invalid review status")
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
            raise ValueError("نام بازبین را وارد کنید")
        if not isinstance(notes, str) or len(notes) > 10000:
            raise ValueError("Invalid review notes")
        with self.transaction() as connection:
            audio = connection.execute("SELECT a.*, j.status AS job_status FROM audio_files a LEFT JOIN jobs j ON j.audio_file_id=a.id AND j.stage='transcribe' WHERE a.id=?", (audio_id,)).fetchone()
            if audio is None:
                raise ValueError("Audio not found")
            if audio["job_status"] in {"running", "skipped"}:
                raise ValueError("بازبینی فایل در حال پردازش یا مدت صفر مجاز نیست")
            current = connection.execute("SELECT * FROM transcripts WHERE audio_file_id=? AND is_current=1", (audio_id,)).fetchone()
            if current is None or payload.get("base_transcript_id") != current["id"]:
                raise ValueError("نسخه تغییر کرده است؛ صفحه را تازه‌سازی کنید و اصلاحات را مقایسه کنید")
            if file_hash(Path(audio["absolute_path"])) != current["audio_content_hash"]:
                raise ValueError("فایل صوتی تغییر کرده است؛ ابتدا دوباره scan کنید")
            previous = connection.execute("SELECT data_json FROM transcript_reviews WHERE transcript_id=?", (current["id"],)).fetchone()
            data = json.loads(previous[0]) if previous else {}
            segments = payload.get("segments")
            if segments is not None:
                if not isinstance(segments, list) or len(segments) > 10000:
                    raise ValueError("Invalid segments")
                checked = []
                evidence_by_id = {row["id"]: row for row in data.get("segments", [])}
                seen = set()
                for index, row in enumerate(segments):
                    if not isinstance(row, dict):
                        raise ValueError("Invalid segment")
                    start, end = row.get("start"), row.get("end")
                    if type(start) not in {int, float} or type(end) not in {int, float} or not math.isfinite(start + end) or start < 0 or end <= start or end > (audio["duration_seconds"] or 0) + .1:
                        raise ValueError("زمان بخش خارج از محدودهٔ صوت است")
                    if not isinstance(row.get("text"), str) or not row["text"].strip() or len(row["text"]) > 20000:
                        raise ValueError("متن بخش نمی‌تواند خالی باشد")
                    if not isinstance(row.get("speaker"), str) or not row["speaker"].strip() or len(row["speaker"]) > 200:
                        raise ValueError("گوینده را مشخص کنید")
                    identifier = str(row.get("id", f"human-{index}"))
                    if identifier in seen:
                        raise ValueError("شناسهٔ بخش تکراری است")
                    seen.add(identifier)
                    original = evidence_by_id.get(identifier, {})
                    checked.append(dict(original, id=identifier, start=float(start), end=float(end),
                                        text=row["text"].strip(), speaker=row["speaker"].strip(),
                                        uncertain=bool(row.get("uncertain", False)),
                                        flags=text_flags(row["text"])))
                checked.sort(key=lambda row: (row["start"], row["end"]))
                if status == "approved" and (not checked or set(evidence_by_id) - seen) and not notes.strip():
                    raise ValueError("برای تأیید متن خالی یا حذف بخش‌های قبلی، توضیح بازبینی لازم است")
                if status == "approved" and any(row["uncertain"] or "unclear" in row["flags"] for row in checked) and not notes.strip():
                    raise ValueError("برای تأیید متن دارای ابهام، توضیح بازبینی لازم است")
                data["segments"] = checked
                content = render_markdown(audio["filename"], checked, human_reviewed=status == "approved")
            else:
                content = payload.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("متن بازبینی نمی‌تواند خالی باشد")
                # Legacy Markdown can be reviewed without fabricated timestamps.
                if data.get("segments"):
                    raise ValueError("برای متن زمان‌بندی‌شده از ویرایش بخش‌ها استفاده کنید")
                if status == "approved" and "[نامفهوم]" in content and not notes.strip():
                    raise ValueError("برای تأیید متن دارای ابهام، توضیح بازبینی لازم است")
            data["quality_status"] = status
            transcript_id, _ = self._store_transcript(connection, audio_id, None, content,
                Path(audio["absolute_path"]).with_suffix(".md"), current["language"], "human_review", force_version=True)
            connection.execute(
                "INSERT INTO transcript_reviews (transcript_id,status,reviewer,notes,data_json,base_transcript_id,created_at) VALUES (?,?,?,?,?,?,?)",
                (transcript_id,status,reviewer.strip(),notes,json.dumps(data,ensure_ascii=False),current["id"],utcnow()),
            )
        # DB revision survives disk errors; the UI can explicitly retry publication.
        self.sync_review_markdown(audio_id, transcript_id)
        return transcript_id

    def sync_review_markdown(self, audio_id: int, transcript_id: int) -> None:
        with self.transaction() as connection:
            row = connection.execute("SELECT t.*, r.base_transcript_id FROM transcripts t JOIN transcript_reviews r ON r.transcript_id=t.id WHERE t.id=? AND t.audio_file_id=? AND t.is_current=1", (transcript_id, audio_id)).fetchone()
            if row is None:
                raise ValueError("نسخهٔ انتخاب‌شده دیگر نسخهٔ جاری نیست")
            destination = Path(row["markdown_path"])
            if destination.exists():
                allowed = {row["content_hash"]}
                base = connection.execute("SELECT content_hash FROM transcripts WHERE id=?", (row["base_transcript_id"],)).fetchone()
                if base:
                    allowed.add(base[0])
                if file_hash(destination) not in allowed:
                    raise ValueError("اصلاح در دیتابیس ذخیره شد اما Markdown بیرون از برنامه تغییر کرده است؛ فایل بیرونی بازنویسی نشد")
            descriptor, temporary = tempfile.mkstemp(prefix=".callforge-review-", suffix=".md", dir=destination.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(row["content"])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                connection.execute("UPDATE transcript_reviews SET markdown_synced=1 WHERE transcript_id=?", (transcript_id,))
            finally:
                Path(temporary).unlink(missing_ok=True)
