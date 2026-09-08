"""Isolated browser QA fixture. Does not read or change the active workspace."""
import tempfile
from pathlib import Path

from callforge.config import AppConfig
from callforge.db import Database
from callforge.metadata import extract_audio_metadata
from callforge.quality import render_markdown
from callforge.web import serve_ui


def main():
    import imageio_ffmpeg
    import subprocess
    root = Path(tempfile.mkdtemp(prefix="callforge-review-ui-"))
    config = AppConfig.for_root(root)
    config.ensure()
    database = Database(config.database)
    database.initialize()
    audio = root / "external-201-09120000000-20260408-143257-demo.mp3"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "sine=frequency=400:duration=10", str(audio)], check=True)
    audio_id, _, _ = database.upsert_audio(extract_audio_metadata(audio, root))
    job = database.claim_jobs(1, "fixture", 120)[0]
    run = database.start_run(job, "fixture", config.logs / "demo.jsonl", config.logs / "demo.stderr")
    rows = [
        {"id": "s1", "start": 0, "end": 4, "text": "سلام، برای پیگیری درخواست تماس گرفتم.", "speaker": "گوینده نامشخص", "raw_text": "سلام برای پیگیری درخواست تماس گرفتم", "alternative": "سلام برای پیگیری تماس گرفتم", "uncertain": False, "flags": []},
        {"id": "s2", "start": 4, "end": 9, "text": "مبلغ ۲۰۰ هزار تومان [نامفهوم]", "speaker": "مشتری", "raw_text": "مبلغ ۲۰۰ هزار تومان", "alternative": "مبلغ ۳۰۰ هزار تومان", "uncertain": True, "flags": ["pass_disagreement", "verify_numbers", "unclear"], "notes": "عدد در دو پاس متفاوت است؛ صوت باید بررسی شود."},
    ]
    content = render_markdown(audio.name, rows)
    markdown = audio.with_suffix(".md")
    markdown.write_text(content, encoding="utf-8")
    database.complete_run(job, run, content, markdown, "fa", None, quality={"segments": rows, "quality_status": "needs_review"})
    print(f"QA fixture: {root}", flush=True)
    serve_ui(config, database, port=18765, open_browser=False)


if __name__ == "__main__":
    main()
