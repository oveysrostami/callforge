from __future__ import annotations

import hashlib
import os
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from callforge.codex_runner import CodexRunner
from callforge.config import AppConfig
from callforge.db import Database


@dataclass
class BatchResult:
    claimed: int = 0
    completed: int = 0
    failed: int = 0


def worker_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def process_job(
    config: AppConfig,
    database: Database,
    runner: CodexRunner,
    job,
    identity: str,
) -> tuple[bool, str]:
    audio_path = Path(job["absolute_path"])
    token = f"job-{job['id']}-attempt-{job['attempts']}-{uuid.uuid4().hex[:8]}"
    log_path = config.logs / f"{token}.jsonl"
    stderr_path = config.logs / f"{token}.stderr.log"
    run_id = database.start_run(job, identity, log_path, stderr_path)
    try:
        if not audio_path.is_file():
            raise RuntimeError(f"Audio file disappeared: {audio_path}")
        markdown_path = audio_path.with_suffix(".md")
        previous_markdown = None
        if markdown_path.is_file():
            previous_markdown = (
                markdown_path.stat().st_mtime_ns,
                hashlib.sha256(markdown_path.read_bytes()).hexdigest(),
            )
        result = runner.run(audio_path, log_path, stderr_path)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()[-2000:]
            raise RuntimeError(f"Codex exited with status {result.returncode}: {detail}")
        if not markdown_path.is_file():
            raise RuntimeError(f"Expected Markdown was not created: {markdown_path}")
        content = markdown_path.read_text(encoding="utf-8")
        if not content.strip():
            raise RuntimeError(f"Generated Markdown is empty: {markdown_path}")
        if previous_markdown:
            current_markdown = (
                markdown_path.stat().st_mtime_ns,
                hashlib.sha256(markdown_path.read_bytes()).hexdigest(),
            )
            if current_markdown == previous_markdown:
                raise RuntimeError("Codex completed without refreshing the existing Markdown file")
        database.complete_run(
            job, run_id, content, markdown_path, config.language, result.thread_id
        )
        return True, str(audio_path)
    except Exception as exc:
        database.fail_run(job, run_id, str(exc))
        return False, f"{audio_path}: {exc}"


def run_batch(
    config: AppConfig,
    database: Database,
    batch_size: int,
    workers: int,
    runner: CodexRunner | None = None,
) -> tuple[BatchResult, list[str]]:
    if batch_size < 1 or workers < 1:
        raise ValueError("batch_size and workers must be positive")
    identity = worker_identity()
    jobs = database.claim_jobs(batch_size, identity, config.lease_seconds)
    summary = BatchResult(claimed=len(jobs))
    messages: list[str] = []
    if not jobs:
        return summary, messages
    selected_runner = runner or CodexRunner(config)
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                process_job, config, database, selected_runner, job, identity
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            success, message = future.result()
            messages.append(message)
            if success:
                summary.completed += 1
            else:
                summary.failed += 1
    return summary, messages
