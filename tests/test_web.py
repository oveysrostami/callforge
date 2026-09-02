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

        with urlopen(f"{base}/Vazirmatn.woff2", timeout=3) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "font/woff2"
            assert response.read(4) == b"wOF2"
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
