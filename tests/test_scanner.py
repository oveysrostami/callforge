from pathlib import Path

from callforge.config import AppConfig
from callforge.db import Database
from callforge.scanner import scan


def test_scan_recurses_and_imports_existing_markdown(tmp_path: Path):
    nested = tmp_path / "2026" / "04" / "19"
    nested.mkdir(parents=True)
    audio = nested / "external-201-123-20260419-193423-id.mp3"
    audio.write_bytes(b"fake")
    audio.with_suffix(".md").write_text("## مکالمه\n\n**مشتری:** تست", encoding="utf-8")
    ignored = tmp_path / ".callforge" / "ignored.mp3"
    ignored.parent.mkdir()
    ignored.write_bytes(b"ignored")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    result = scan(config, database)
    assert result.discovered == 1
    assert result.imported_markdown == 1
    assert database.counts()["audio_files"] == 1


def test_changed_audio_invalidates_old_transcript(tmp_path: Path):
    audio = tmp_path / "external-201-123-20260419-193423-id.mp3"
    audio.write_bytes(b"first audio")
    audio.with_suffix(".md").write_text("## مکالمه\n\n**مشتری:** نسخه قبلی", encoding="utf-8")
    config = AppConfig.for_root(tmp_path)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    scan(config, database)
    assert database.counts()["current_transcripts"] == 1
    audio.write_bytes(b"different audio")
    result = scan(config, database)
    counts = database.counts()
    assert result.changed == 1
    assert result.imported_markdown == 0
    assert counts["current_transcripts"] == 0
    assert counts["pending"] == 1
