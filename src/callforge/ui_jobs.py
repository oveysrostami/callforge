from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from callforge.codex_runner import CodexRunner
from callforge.config import AppConfig
from callforge.db import Database
from callforge.worker import process_job, worker_identity


class UITranscriptionService:
    """Run user-selected transcription jobs without blocking HTTP requests."""

    def __init__(
        self,
        config: AppConfig,
        database: Database,
        runner: CodexRunner | None = None,
    ):
        self.config = config
        self.database = database
        self.runner = runner or CodexRunner(config)
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, config.workers), thread_name_prefix="callforge-ui"
        )
        self._active: dict[int, Future[tuple[bool, str] | None]] = {}
        self._lock = threading.Lock()

    def request(self, audio_id: int) -> str | None:
        state = self.database.queue_transcription(audio_id)
        if state is None or state == "running":
            return state
        with self._lock:
            previous = self._active.get(audio_id)
            if previous is not None and not previous.done():
                return "queued"
            self._active[audio_id] = self.executor.submit(self._process, audio_id)
        return "queued"

    def _process(self, audio_id: int) -> tuple[bool, str] | None:
        identity = f"ui:{worker_identity()}"
        last_result: tuple[bool, str] | None = None
        while True:
            job = self.database.claim_audio_job(
                audio_id, identity, self.config.lease_seconds
            )
            if job is None:
                return last_result
            last_result = process_job(
                self.config, self.database, self.runner, job, identity
            )
            if last_result[0]:
                return last_result
            detail = self.database.audio_file_detail(audio_id)
            if detail is None or detail["job_status"] != "pending":
                return last_result

    def wait(self, audio_id: int, timeout: float = 10) -> tuple[bool, str] | None:
        with self._lock:
            future = self._active.get(audio_id)
        return future.result(timeout=timeout) if future else None

    def shutdown(self, *, wait: bool = False) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=True)
