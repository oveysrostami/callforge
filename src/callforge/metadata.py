from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


CALL_FILENAME = re.compile(
    r"^(?P<prefix>external|internal|out)-(?P<first>[^-]+)-(?P<second>[^-]+)-"
    r"(?P<date>\d{8})-(?P<time>\d{6})-(?P<call_id>.+)\.mp3$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AudioMetadata:
    absolute_path: str
    relative_path: str
    filename: str
    direction: str | None
    agent_extension: str | None
    remote_number: str | None
    call_id: str | None
    recorded_at: str | None
    size_bytes: int
    mtime_ns: int
    content_sha256: str
    duration_seconds: float | None
    codec: str | None
    bitrate: int | None
    sample_rate: int | None
    channels: int | None
    metadata_error: str | None


def parse_call_filename(filename: str) -> dict[str, str | None]:
    match = CALL_FILENAME.match(filename)
    if not match:
        return {
            "direction": None,
            "agent_extension": None,
            "remote_number": None,
            "call_id": None,
            "recorded_at": None,
        }
    values = match.groupdict()
    prefix = values["prefix"].lower()
    direction = {
        "external": "inbound",
        "internal": "internal",
        "out": "outbound",
    }[prefix]
    if prefix == "out":
        agent_extension, remote_number = values["second"], values["first"]
    else:
        agent_extension, remote_number = values["first"], values["second"]
    recorded_at = datetime.strptime(
        values["date"] + values["time"], "%Y%m%d%H%M%S"
    ).isoformat()
    return {
        "direction": direction,
        "agent_extension": agent_extension,
        "remote_number": remote_number,
        "call_id": values["call_id"],
        "recorded_at": recorded_at,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_audio_metadata(path: Path, root: Path) -> AudioMetadata:
    resolved = path.resolve()
    stat = resolved.stat()
    parsed = parse_call_filename(resolved.name)
    duration = codec = bitrate = sample_rate = channels = None
    error = None
    try:
        from mutagen.mp3 import MP3

        info = MP3(resolved).info
        duration = round(float(info.length), 6)
        codec = "mp3"
        bitrate = int(info.bitrate) if info.bitrate else None
        sample_rate = int(info.sample_rate) if info.sample_rate else None
        channels = int(info.channels) if info.channels else None
    except Exception as exc:  # Broken audio is indexed and reported instead of skipped.
        error = f"{type(exc).__name__}: {exc}"
    return AudioMetadata(
        absolute_path=str(resolved),
        relative_path=str(resolved.relative_to(root.resolve())),
        filename=resolved.name,
        direction=parsed["direction"],
        agent_extension=parsed["agent_extension"],
        remote_number=parsed["remote_number"],
        call_id=parsed["call_id"],
        recorded_at=parsed["recorded_at"],
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        content_sha256=sha256_file(resolved),
        duration_seconds=duration,
        codec=codec,
        bitrate=bitrate,
        sample_rate=sample_rate,
        channels=channels,
        metadata_error=error,
    )
