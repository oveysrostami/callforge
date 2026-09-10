"""Build a conservative human-review queue from persisted ASR evidence."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from callforge.quality import normalize, text_flags, usable_retry_text


UNRESOLVED_MARKER = "[نامفهوم]"
_REJECT_FLAGS = {"repetition", "compression", "unclear"}
_SOURCE_LABELS = {
    "raw": "خروجی صوت خام",
    "alternative": "خروجی صوت تقویت‌شده",
    "retry": "بازخوانی تکمیلی",
    "consensus": "جمع‌بندی مدل‌ها",
    "uncertainty": "پیشنهاد مدل برای بخش نامفهوم",
    "entity": "پیشنهاد واژه‌نامه با شاهد صوتی",
}


def is_unresolved_segment(row: dict[str, Any]) -> bool:
    flags = set(row.get("flags") or [])
    return (
        UNRESOLVED_MARKER in str(row.get("text") or "")
        or row.get("confidence_tier") == "unresolved"
        or "unclear" in flags
    )


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text or len(text) > 20_000 or UNRESOLVED_MARKER in text:
        return None
    if set(text_flags(text)) & _REJECT_FLAGS:
        return None
    return text


def _full_candidate(current: str, value: Any, *, fragment: bool) -> str | None:
    candidate = _safe_text(value)
    if candidate is None:
        return None
    if fragment:
        if current.count(UNRESOLVED_MARKER) != 1:
            return None
        candidate = current.replace(UNRESOLVED_MARKER, candidate, 1)
    return _safe_text(candidate)


def candidate_options(row: dict[str, Any]) -> list[dict[str, str]]:
    """Return deduplicated full-segment suggestions with stable opaque IDs."""
    current = str(row.get("text") or "").strip()
    values: list[tuple[str, Any, bool]] = [
        ("raw", row.get("raw_text"), False),
        ("consensus", row.get("consensus_text"), False),
    ]
    if "alternative_prompt_leakage" not in set(row.get("flags") or []):
        values.append(("alternative", row.get("alternative"), False))
    retry = usable_retry_text(row) if isinstance(row.get("retry") or {}, dict) else None
    if retry:
        values.append(("retry", retry, False))
    for span in row.get("uncertainty_spans") or []:
        if not isinstance(span, dict):
            continue
        candidates = span.get("candidates") or span.get("candidate_texts") or []
        if isinstance(candidates, str):
            candidates = [candidates]
        for value in candidates:
            if isinstance(value, dict):
                value = value.get("text") or value.get("candidate")
            values.append(("uncertainty", value, True))
    for decision in row.get("entity_resolutions") or []:
        if isinstance(decision, dict) and decision.get("accepted", True):
            values.append(("entity", decision.get("canonical"), True))

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, value, fragment in values:
        text = _full_candidate(current, value, fragment=fragment)
        key = normalize(text or "")
        if not text or not key or key == normalize(current) or key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256(f"{source}\0{text}".encode("utf-8")).hexdigest()[:16]
        result.append({
            "id": f"{source}-{digest}",
            "source": source,
            "source_label": _SOURCE_LABELS[source],
            "text": text,
        })
    return result


def public_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the evidence needed for future glossary mining without audio copies."""
    keys = (
        "raw_text", "alternative", "consensus_text", "retry", "uncertainty_spans",
        "entity_candidates", "entity_resolutions", "candidate_sources", "confidence_tier",
        "reason", "flags", "notes", "speaker", "speaker_id",
    )
    return {key: row[key] for key in keys if key in row}
