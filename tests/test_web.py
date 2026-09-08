import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from callforge.config import AppConfig
from callforge.db import Database
from callforge.metadata import extract_audio_metadata
from callforge.web import create_server


def prepared_server(tmp_path: Path, service_factory=None):
    audio = tmp_path / "external-208-09120000000-20260408-143257-id.mp3"
    audio.write_bytes(b"0123456789")
    markdown = audio.with_suffix(".md")
    markdown.write_text("# متن تماس\n\n**مشتری:** سلام", encoding="utf-8")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, tmp_path))
    database.import_markdown(audio_id, markdown)
    service = service_factory(config, database, audio_id) if service_factory else None
    server = create_server(
        config, database, "127.0.0.1", 0, transcription_service=service
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, audio_id


def get_json(url: str):
    with urlopen(url, timeout=3) as response:
        return response.status, json.loads(response.read())


def test_human_review_http_roundtrip_history_and_filter(tmp_path):
    server, thread, audio_id = prepared_server(tmp_path)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _, review = get_json(f"{base}/api/files/{audio_id}/review")
        assert review["status"] == "needs_review"
        payload = {"base_transcript_id": review["transcript_id"], "content": "## مکالمه\n\n**مشتری:** سلام وقت بخیر",
                   "status": "approved", "reviewer": "test", "notes": "fixture only"}
        request = Request(f"{base}/api/files/{audio_id}/review", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json", "X-CallForge-UI": "1"})
        with urlopen(request, timeout=3) as response:
            saved = json.loads(response.read())
        assert saved["status"] == "approved"
        assert saved["markdown_synced"] is True
        assert len(saved["versions"]) == 2
        _, listing = get_json(f"{base}/api/files?review=approved")
        assert listing["total"] == 1
        assert listing["items"][0]["review_status"] == "approved"
        _, listing = get_json(f"{base}/api/files?review=needs_review")
        assert listing["total"] == 0
        try:
            urlopen(request, timeout=3)
            assert False, "stale revision must be rejected"
        except HTTPError as error:
            assert error.code == 409
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


def test_ui_api_lists_details_and_streams_range(tmp_path: Path):
    server, thread, audio_id = prepared_server(tmp_path)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, listing = get_json(f"{base}/api/files?q=09120000000&transcript=yes")
        assert status == 200
        assert listing["total"] == 1
        assert listing["items"][0]["direction"] == "inbound"

        _, detail = get_json(f"{base}/api/files/{audio_id}")
        assert detail["transcript_content"].endswith("سلام")
        assert detail["audio_url"] == f"/api/files/{audio_id}/audio"

        request = Request(
            f"{base}/api/files/{audio_id}/audio", headers={"Range": "bytes=2-5"}
        )
        with urlopen(request, timeout=3) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.read() == b"2345"

        with urlopen(f"{base}/", timeout=3) as response:
            assert response.status == 200
            page = response.read()
            assert b"CallForge" in page
            assert b'class="markdown-body" id="transcript"' in page
            assert b'<option value="skipped">' in page

        with urlopen(f"{base}/app.js", timeout=3) as response:
            javascript = response.read()
            assert b"function renderMarkdown" in javascript
            assert b'renderMarkdown($("transcript")' in javascript
            assert b"new EventSource" in javascript

        with urlopen(f"{base}/", timeout=3) as response:
            assert b'id="progress-card"' in response.read()

        with urlopen(f"{base}/Vazirmatn.woff2", timeout=3) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "font/woff2"
            assert response.read(4) == b"wOF2"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_progress_endpoint_streams_snapshot_and_summarized_codex_events(tmp_path: Path):
    server, thread, audio_id = prepared_server(tmp_path)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    config = AppConfig.for_root(tmp_path)
    database = Database(config.database)
    try:
        assert database.queue_transcription(audio_id) == "queued"
        job = database.claim_audio_job(audio_id, "test-worker", 120)
        assert job is not None
        log_path = config.logs / "live.jsonl"
        stderr_path = config.logs / "live.stderr.log"
        log_path.write_text(
            '{"type":"thread.started","thread_id":"test"}\n'
            '{"type":"item.started","item":{"type":"command_execution",'
            '"status":"in_progress","command":"python prepare_audio.py call.mp3"}}\n',
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        run_id = database.start_run(job, "test-worker", log_path, stderr_path)

        with urlopen(f"{base}/api/files/{audio_id}/progress?once=1", timeout=3) as response:
            body = response.read().decode("utf-8")
            assert response.headers["Content-Type"].startswith("text/event-stream")
            assert "event: snapshot" in body
            assert '"job_status":"running"' in body
            assert "event: progress" in body
            assert "آماده‌سازی، اندازه‌گیری و تقویت صدا" in body

        markdown_path = tmp_path / "external-208-09120000000-20260408-143257-id.md"
        content = "# متن تماس\n\n**کارشناس:** انجام شد"
        markdown_path.write_text(content, encoding="utf-8")
        database.complete_run(job, run_id, content, markdown_path, "fa", "test")
        detail = database.audio_file_detail(audio_id)
        assert detail and detail["latest_run_id"] == run_id
        assert detail["latest_run_status"] == "completed"
        assert float(detail["processing_total_seconds"]) >= 0
        assert log_path.is_file()
        _, listed = database.list_audio_files()
        assert listed[0]["latest_run_id"] == run_id
        assert float(listed[0]["processing_total_seconds"]) >= 0

        with urlopen(f"{base}/api/files/{audio_id}/progress?once=1", timeout=3) as response:
            historical = response.read().decode("utf-8")
            assert '"job_status":"completed"' in historical
            assert "آماده‌سازی، اندازه‌گیری و تقویت صدا" in historical
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_invalid_audio_range_returns_416(tmp_path: Path):
    server, thread, audio_id = prepared_server(tmp_path)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = Request(
            f"{base}/api/files/{audio_id}/audio", headers={"Range": "bytes=99-100"}
        )
        try:
            urlopen(request, timeout=3)
        except HTTPError as error:
            assert error.code == 416
            assert error.headers["Content-Range"] == "bytes */10"
        else:
            raise AssertionError("Expected a 416 response")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_transcribe_endpoint_requires_ui_header_and_queues_exact_file(tmp_path: Path):
    class RecordingService:
        def __init__(self):
            self.requested = []

        def request(self, audio_id: int):
            self.requested.append(audio_id)
            return "queued"

    holder = {}

    def service_factory(config, database, audio_id):
        service = RecordingService()
        holder["service"] = service
        return service

    server, thread, audio_id = prepared_server(tmp_path, service_factory)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        unauthorized = Request(
            f"{base}/api/files/{audio_id}/transcribe",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        try:
            urlopen(unauthorized, timeout=3)
        except HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("Expected a 403 response")

        request = Request(
            f"{base}/api/files/{audio_id}/transcribe",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json", "X-CallForge-UI": "1"},
        )
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read())
            assert response.status == 202
            assert payload == {"audio_id": audio_id, "status": "queued"}
        assert holder["service"].requested == [audio_id]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
