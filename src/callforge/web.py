from __future__ import annotations

import json
import mimetypes
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from callforge import __version__
from callforge.config import AppConfig
from callforge.db import Database
from callforge.ui_jobs import UITranscriptionService


AUDIO_ROUTE = re.compile(r"^/api/files/(?P<id>\d+)/audio$")
DETAIL_ROUTE = re.compile(r"^/api/files/(?P<id>\d+)$")
TRANSCRIBE_ROUTE = re.compile(r"^/api/files/(?P<id>\d+)/transcribe$")
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.css": "app.css",
    "/app.js": "app.js",
    "/Vazirmatn.woff2": "Vazirmatn.woff2",
}


def _integer(values: dict[str, list[str]], key: str, default: int, maximum: int) -> int:
    try:
        value = int(values.get(key, [str(default)])[0])
    except ValueError:
        return default
    return min(max(value, 0), maximum)


def make_handler(
    config: AppConfig,
    database: Database,
    transcription_service: UITranscriptionService,
):
    resources = files("callforge").joinpath("resources/web")

    class CallForgeHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"CallForge/{__version__}"

        def do_GET(self) -> None:
            self._route(send_body=True)

        def do_HEAD(self) -> None:
            self._route(send_body=False)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            match = TRANSCRIBE_ROUTE.match(parsed.path)
            if match is None:
                self._error(HTTPStatus.NOT_FOUND, "مسیر پیدا نشد", True)
                return
            if self.headers.get("X-CallForge-UI") != "1":
                self._error(HTTPStatus.FORBIDDEN, "درخواست UI معتبر نیست", True)
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
            if content_type != "application/json":
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "Content-Type باید application/json باشد",
                    True,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Content-Length نامعتبر است", True)
                return
            if length > 1024:
                self.close_connection = True
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "بدنهٔ درخواست بیش از حد بزرگ است", True)
                return
            if length:
                self.rfile.read(length)
            audio_id = int(match.group("id"))
            state = transcription_service.request(audio_id)
            if state is None:
                self._error(HTTPStatus.NOT_FOUND, "فایل در دیتابیس پیدا نشد", True)
                return
            self._json(
                {"audio_id": audio_id, "status": state},
                send_body=True,
                status=HTTPStatus.ACCEPTED,
            )

        def _route(self, send_body: bool) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path in STATIC_FILES:
                    self._static(STATIC_FILES[parsed.path], send_body)
                    return
                if parsed.path == "/api/stats":
                    self._json(database.counts(), send_body=send_body)
                    return
                if parsed.path == "/api/files":
                    self._files(parse_qs(parsed.query), send_body)
                    return
                audio_match = AUDIO_ROUTE.match(parsed.path)
                if audio_match:
                    self._audio(int(audio_match.group("id")), send_body)
                    return
                detail_match = DETAIL_ROUTE.match(parsed.path)
                if detail_match:
                    self._detail(int(detail_match.group("id")), send_body)
                    return
                self._error(HTTPStatus.NOT_FOUND, "مسیر پیدا نشد", send_body)
            except BrokenPipeError:
                return
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), send_body)

        def _files(self, values: dict[str, list[str]], send_body: bool) -> None:
            limit = _integer(values, "limit", 50, 200) or 50
            offset = _integer(values, "offset", 0, 10_000_000)
            query = values.get("q", [""])[0].strip()[:200]
            direction = values.get("direction", [""])[0]
            status = values.get("status", [""])[0]
            transcript = values.get("transcript", [""])[0]
            total, items = database.list_audio_files(
                limit=limit,
                offset=offset,
                query=query,
                direction=direction,
                status=status,
                transcript=transcript,
            )
            self._json(
                {"items": items, "total": total, "limit": limit, "offset": offset},
                send_body=send_body,
            )

        def _detail(self, audio_id: int, send_body: bool) -> None:
            detail = database.audio_file_detail(audio_id)
            if detail is None:
                self._error(HTTPStatus.NOT_FOUND, "فایل در دیتابیس پیدا نشد", send_body)
                return
            detail["audio_url"] = f"/api/files/{audio_id}/audio"
            self._json(detail, send_body=send_body)

        def _audio(self, audio_id: int, send_body: bool) -> None:
            audio_path = database.audio_path(audio_id)
            if audio_path is None:
                self._error(HTTPStatus.NOT_FOUND, "فایل در دیتابیس پیدا نشد", send_body)
                return
            resolved = audio_path.expanduser().resolve()
            root = config.root.resolve()
            if resolved != root and root not in resolved.parents:
                self._error(HTTPStatus.FORBIDDEN, "مسیر صوت خارج از workspace است", send_body)
                return
            if not resolved.is_file():
                self._error(HTTPStatus.NOT_FOUND, "فایل صوتی روی دیسک موجود نیست", send_body)
                return
            size = resolved.stat().st_size
            if size == 0:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "فایل صوتی خالی است", send_body)
                return
            start, end, partial = self._byte_range(size)
            if start is None or end is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "private, max-age=3600")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return
            with resolved.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _byte_range(self, size: int) -> tuple[int | None, int | None, bool]:
            header = self.headers.get("Range")
            if not header:
                return 0, max(size - 1, 0), False
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not match or size == 0:
                return None, None, True
            first, last = match.groups()
            if not first:
                suffix = int(last or "0")
                if suffix <= 0:
                    return None, None, True
                start = max(0, size - suffix)
                return start, size - 1, True
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
            if start >= size or end < start:
                return None, None, True
            return start, end, True

        def _static(self, name: str, send_body: bool) -> None:
            resource = resources.joinpath(name)
            content = resource.read_bytes()
            mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if send_body:
                self.wfile.write(content)

        def _json(
            self,
            value: Any,
            *,
            send_body: bool,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            content = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(content)

        def _error(self, status: HTTPStatus, message: str, send_body: bool) -> None:
            content = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[ui] {self.address_string()} - {format % args}")

    return CallForgeHandler


def create_server(
    config: AppConfig,
    database: Database,
    host: str,
    port: int,
    transcription_service: UITranscriptionService | None = None,
) -> ThreadingHTTPServer:
    service = transcription_service or UITranscriptionService(config, database)
    server = ThreadingHTTPServer(
        (host, port), make_handler(config, database, service)
    )
    server.transcription_service = service  # type: ignore[attr-defined]
    return server


def serve_ui(
    config: AppConfig,
    database: Database,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    server = create_server(config, database, host, port)
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}"
    print(f"CallForge UI: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.transcription_service.shutdown(wait=False)  # type: ignore[attr-defined]
