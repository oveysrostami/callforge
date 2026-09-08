from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _compact(value: object, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("`", "")
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _command_stage(command: str) -> tuple[str, str]:
    lowered = command.lower()
    if "prepare_audio.py" in lowered:
        return "prepare_audio", "آماده‌سازی، اندازه‌گیری و تقویت صدا"
    if "transcribe_audio.py" in lowered or "mlx_whisper" in lowered:
        if "large-v3-4bit" in lowered:
            return "whisper_review", "اجرای گذر بازبینی Whisper"
        return "whisper", "اجرای تشخیص گفتار Whisper"
    if "ffprobe" in lowered or "afinfo" in lowered or re.search(r"(^|[ ;])file ", lowered):
        return "inspect_audio", "بررسی مشخصات و سلامت فایل صوتی"
    if "pip install" in lowered or "venv" in lowered:
        return "runtime", "آماده‌سازی ابزارهای پردازش محلی"
    if "skill.md" in lowered:
        return "instructions", "بارگذاری دستورالعمل تبدیل تماس"
    if ".md" in lowered and any(word in lowered for word in ("write", "sed", "test", "ls")):
        return "finalize", "ساخت و کنترل فایل Markdown نهایی"
    return "processing", "اجرای یکی از مراحل پردازش"


def describe_codex_event(event: dict[str, Any]) -> dict[str, str] | None:
    """Turn one Codex JSONL event into a small user-facing progress event."""

    event_type = event.get("type")
    if event_type == "callforge.stage":
        stage = _compact(event.get("stage"), 80) or "processing"
        state = _compact(event.get("state"), 40) or "active"
        message = _compact(event.get("message")) or "پردازش محلی CallForge"
        return {"stage": stage, "state": state, "message": message}
    if event_type == "thread.started":
        return {"stage": "starting", "state": "completed", "message": "Codex شروع شد"}
    if event_type == "turn.started":
        return {
            "stage": "starting",
            "state": "active",
            "message": "فرآیند بررسی و تبدیل تماس آغاز شد",
        }
    if event_type in {"turn.failed", "error"}:
        message = _compact(event.get("message") or event.get("error") or "پردازش با خطا روبه‌رو شد")
        return {"stage": "error", "state": "failed", "message": message}
    if event_type == "turn.completed":
        return {
            "stage": "finalize",
            "state": "completed",
            "message": "اجرای Codex پایان یافت؛ نتیجه در حال ثبت است",
        }

    item = event.get("item")
    if not isinstance(item, dict):
        return None
    item_type = item.get("type")
    item_status = str(item.get("status") or "")

    if item_type == "agent_message":
        message = _compact(item.get("text"))
        if not message:
            return None
        return {"stage": "review", "state": "active", "message": message}

    if item_type != "command_execution":
        return None
    stage, label = _command_stage(str(item.get("command") or ""))
    if event_type == "item.started":
        return {"stage": stage, "state": "active", "message": label}
    if event_type != "item.completed":
        return None
    if item_status == "failed" or item.get("exit_code") not in (None, 0):
        output = _compact(item.get("aggregated_output"), 180)
        detail = f"{label} ناموفق بود"
        if output:
            detail += f": {output}"
        return {"stage": stage, "state": "failed", "message": detail}
    return {"stage": stage, "state": "completed", "message": f"{label} انجام شد"}


def read_progress_events(
    log_path: str | Path | None,
    *,
    after_line: int = 0,
    maximum: int = 40,
) -> tuple[int, list[dict[str, object]]]:
    """Read new complete JSONL lines and return summarized progress events."""

    if not log_path:
        return after_line, []
    path = Path(log_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return after_line, []
    start = min(max(after_line, 0), len(lines))
    if after_line == 0 and len(lines) > maximum:
        start = len(lines) - maximum
    described: list[dict[str, object]] = []
    for index, line in enumerate(lines[start:], start=start + 1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        event = describe_codex_event(raw)
        if event:
            described.append({"line": index, **event})
    return len(lines), described[-maximum:]
