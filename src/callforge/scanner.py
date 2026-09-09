from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from callforge.config import AppConfig, WORKSPACE_NAME
from callforge.audio_files import colliding_transcript_targets, is_supported_audio, transcript_path
from callforge.db import Database, is_zero_duration
from callforge.metadata import extract_audio_metadata


@dataclass
class ScanResult:
    discovered: int = 0
    changed: int = 0
    metadata_errors: int = 0
    imported_markdown: int = 0
    skipped: int = 0
    collisions: int = 0


def discover_audio(root: Path):
    for path in root.rglob("*"):
        if WORKSPACE_NAME in path.parts:
            continue
        if is_supported_audio(path):
            yield path


# Public compatibility alias used by older integrations.
discover_mp3 = discover_audio


def scan(config: AppConfig, database: Database, import_markdown: bool = True) -> ScanResult:
    result = ScanResult()
    paths = list(discover_audio(config.root))
    collisions = colliding_transcript_targets(paths)
    result.collisions = len(collisions)
    for path in paths:
        if path.resolve() in collisions:
            result.discovered += 1
            result.metadata_errors += 1
            result.skipped += 1
            continue
        metadata = extract_audio_metadata(path, config.root)
        audio_id, changed, created = database.upsert_audio(metadata, config.max_attempts)
        result.discovered += 1
        result.changed += int(changed)
        result.metadata_errors += int(metadata.metadata_error is not None)
        result.skipped += int(is_zero_duration(metadata.duration_seconds))
        markdown = transcript_path(path)
        if (
            import_markdown
            and not is_zero_duration(metadata.duration_seconds)
            and markdown.is_file()
            and not (changed and not created)
        ):
            result.imported_markdown += int(
                database.import_markdown(audio_id, markdown, config.language) is not None
            )
    return result
