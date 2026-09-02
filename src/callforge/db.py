from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

from callforge.metadata import AudioMetadata


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
            if version > 4:
                raise RuntimeError(
                    f"Database schema {version} is newer than this CallForge supports"
                )
            if version == 0:
                connection.executescript(SCHEMA)
                return
            if version == 1:
                connection.executescript(MIGRATION_1_TO_2)
                version = 2
            if version == 2:
                connection.executescript(MIGRATION_2_TO_3)
                version = 3
            if version == 3:
                connection.executescript(MIGRATION_3_TO_4)

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
            if job is not None and job["status"] == "skipped":
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
        if current and current["content_hash"] == content_hash:
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

    def complete_run(
        self,
        job: sqlite3.Row,
        run_id: int,
        content: str,
        markdown_path: Path,
        language: str,
        codex_thread_id: str | None,
    ) -> int:
        with self.transaction() as connection:
            transcript_id, _ = self._store_transcript(
                connection,
                int(job["audio_file_id"]),
                run_id,
                content,
                markdown_path,
                language,
                "codex_skill",
            )
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

    def fail_run(self, job: sqlite3.Row, run_id: int, error: str) -> None:
        with self.transaction() as connection:
            terminal = int(job["attempts"]) >= int(job["max_attempts"])
            status = "failed" if terminal else "pending"
            now = utcnow()
            connection.execute(
                "UPDATE processing_runs SET status='failed', error=?, finished_at=? WHERE id=?",
                (error, now, run_id),
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

    def list_audio_files(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
        direction: str = "",
        status: str = "",
        transcript: str = "",
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
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        joined = (
            "FROM audio_files a "
            "LEFT JOIN jobs j ON j.audio_file_id=a.id AND j.stage='transcribe' "
            "LEFT JOIN transcripts t ON t.audio_file_id=a.id AND t.is_current=1 "
            "AND (j.status IS NULL OR j.status != 'skipped') "
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
                "t.id AS transcript_id, t.version AS transcript_version, t.source AS transcript_source, "
                "t.unclear_count, t.created_at AS transcript_created_at "
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
                "t.markdown_path "
                "FROM audio_files a "
                "LEFT JOIN jobs j ON j.audio_file_id=a.id AND j.stage='transcribe' "
                "LEFT JOIN transcripts t ON t.audio_file_id=a.id AND t.is_current=1 "
                "AND (j.status IS NULL OR j.status != 'skipped') "
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
