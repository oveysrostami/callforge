"""Opt-in bounded real-backend smoke run on a copied recording.

Usage: python tests/manual_pipeline_smoke.py SOURCE_MP3 EXISTING_MODEL_CACHE [--full]
Defaults to 12 seconds; --full copies the complete source without re-encoding.
Runs Codex review (normal account usage). Never touches the source or active DB.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from callforge.codex_runner import CodexRunner
from callforge.config import AppConfig
from callforge.db import Database
from callforge.metadata import extract_audio_metadata
from callforge.worker import run_batch


def main():
    import imageio_ffmpeg
    root = Path(tempfile.mkdtemp(prefix="callforge-pipeline-smoke-"))
    full = "--full" in sys.argv[3:]
    audio = root / Path(sys.argv[1]).name
    if full:
        shutil.copy2(sys.argv[1], audio)
    else:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", sys.argv[1], "-t", "12", str(audio)], check=True, timeout=30)
    os.environ["HF_HUB_OFFLINE"] = "1"
    config = replace(AppConfig.for_root(root), models=Path(sys.argv[2]),
                     whisper_timeout_seconds=300 if full else 90, whisper_retry_seconds=120 if full else 30,
                     whisper_retry_segments=3 if full else 1, codex_timeout_seconds=300 if full else 120,
                     codex_idle_timeout_seconds=120 if full else 60)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, root))
    print(f"Isolated smoke workspace: {root}", flush=True)
    result, messages = run_batch(config, database, 1, 1, runner=CodexRunner(config))
    review = database.review_detail(audio_id)
    print(json.dumps({"completed": result.completed, "failed": result.failed,
                      "quality_status": review["status"], "segments": len(review["data"].get("segments", [])),
                      "automated_review_complete": review["data"].get("automated_review_complete"),
                      "review_error": review["data"].get("review_error"),
                      "failure": messages if result.failed else None}, ensure_ascii=False), flush=True)
    return int(bool(result.failed))


if __name__ == "__main__":
    raise SystemExit(main())
