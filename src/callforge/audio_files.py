from __future__ import annotations

import mimetypes
from collections import defaultdict
from pathlib import Path


SUPPORTED_AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".flac", ".ogg"})


def is_supported_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES


def transcript_path(path: Path) -> Path:
    return path.with_suffix(".md")


def audio_mime_type(path: Path) -> str:
    explicit = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    return explicit.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def colliding_transcript_targets(paths: list[Path]) -> dict[Path, tuple[Path, ...]]:
    """Return every audio whose sibling transcript target is shared.

    Paths are resolved case-sensitively by the filesystem, while extensions are
    intentionally ignored because every supported container publishes to
    ``<stem>.md``.
    """
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[transcript_path(path).resolve()].append(path.resolve())
    return {
        audio: tuple(sorted(group, key=str))
        for group in grouped.values()
        if len(group) > 1
        for audio in group
    }


def sibling_audio_collision(path: Path) -> tuple[Path, ...]:
    candidates = [
        candidate
        for candidate in path.parent.iterdir()
        if is_supported_audio(candidate) and candidate.stem == path.stem
    ]
    return tuple(sorted((candidate.resolve() for candidate in candidates), key=str)) if len(candidates) > 1 else ()
