"""Timed transcription evidence, conservative review validation and rendering.

Quality flags are review priorities, never calibrated confidence scores.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path


class ArtifactConflictError(RuntimeError):
    """A user changed the output during processing; automatic retry is unsafe."""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(str.maketrans("يكى۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "یکی01234567890123456789"))).strip()


def text_flags(text: str, segment: dict | None = None) -> list[str]:
    flags = []
    if re.search(r"(.)\1{5,}|(\b\S+\s+)\2{3,}", text):
        flags.append("repetition")
    if re.search(r"\d", normalize(text)):
        flags.append("verify_numbers")
    if "[نامفهوم]" in text or not text.strip():
        flags.append("unclear")
    segment = segment or {}
    if segment.get("avg_logprob") is not None and segment["avg_logprob"] < -1:
        flags.append("low_logprob")
    if segment.get("compression_ratio") is not None and segment["compression_ratio"] > 2.2:
        flags.append("compression")
    if segment.get("no_speech_prob") is not None and segment["no_speech_prob"] > .5:
        flags.append("possible_non_speech")
    return flags


def retry_priority(row: dict) -> tuple[int, float]:
    """Repair decoder loops before spending a bounded retry budget on gaps."""
    flags = set(row.get("flags", []))
    priority = (0 if "repetition" in flags else 1 if "compression" in flags
                else 2 if "low_logprob" in flags else 3 if "pass_disagreement" in flags else 4)
    return priority, float(row["start"])


def compact_review_input(evidence: dict) -> dict:
    """Keep the complete call context, not duplicated word-level diagnostics."""
    return {"duration_seconds": evidence["duration_seconds"], "segments": [
        {**{key: row.get(key) for key in ("id", "start", "end", "text", "alternative", "flags", "speaker")},
         "retry_text": (row.get("retry") or {}).get("text")}
        for row in evidence["segments"]]}


def timed_units(segments: list[dict], max_seconds: float = 8) -> list[dict]:
    """Split on real word boundaries, retaining unsplittable text without guessing times."""
    units = []
    for segment in segments:
        words = segment.get("words") or []
        joined = "".join(word.get("word", "") for word in words)
        if not words or SequenceMatcher(None, normalize(joined), normalize(segment["text"])).ratio() < .8:
            units.append(segment)
            continue
        group = []
        def flush():
            if group:
                units.append(dict(segment, start=group[0]["start"], end=group[-1]["end"],
                                  text="".join(word["word"] for word in group).strip(), words=list(group)))
        for word in words:
            if group and (word["end"] - group[0]["start"] > max_seconds or word["start"] - group[-1]["end"] > .65):
                flush(); group = []
            group.append(word)
        flush()
    return units


def build_evidence(raw: dict, enhanced: dict, duration: float, speech: list[dict] | None = None) -> dict:
    """Keep a canonical raw timeline and overlapping alternatives without claiming diarization."""
    rows = []
    enhanced_units = timed_units(enhanced.get("segments", []))
    for index, item in enumerate(timed_units(raw.get("segments", []))):
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start + end) or start < 0 or end < start or start >= duration:
            continue
        end = min(end, duration)
        alternatives = [other for other in enhanced_units
                        if float(other["start"]) < end and float(other["end"]) > start]
        alternative = " ".join(other["text"] for other in alternatives).strip()
        original = item["text"].strip()
        flags = text_flags(original, item)
        if alternative and SequenceMatcher(None, normalize(original), normalize(alternative)).ratio() < .65:
            flags.append("pass_disagreement")
        rows.append({"id": f"s{index + 1}", "start": start, "end": end,
                     "text": original, "alternative": alternative, "flags": flags,
                     "speaker": "گوینده نامشخص", "words": item.get("words", [])})
    # Enhanced-only speech must not disappear merely because the raw pass omitted it.
    candidates = list(enhanced_units)
    # VAD uses 400 ms padding: it is not evidence of missing words at boundaries.
    candidates.extend({"start": s["start"] + .4, "end": s["end"] - .4,
                       "text": "[نامفهوم]", "vad_only": True} for s in speech or [])
    for item in candidates:
        start, end = max(0., float(item["start"])), min(duration, float(item["end"]))
        if end <= start:
            continue
        uncovered = [(start, end)]
        for row in rows:
            next_parts = []
            for a, b in uncovered:
                if row["end"] <= a or row["start"] >= b:
                    next_parts.append((a, b))
                else:
                    if a < row["start"]:
                        next_parts.append((a, row["start"]))
                    if row["end"] < b:
                        next_parts.append((row["end"], b))
            uncovered = next_parts
        for a, b in uncovered:
            if b - a < (1.0 if item.get("vad_only") else .25):
                continue
            rows.append({"id": f"s{len(rows) + 1}", "start": a, "end": b,
                         "text": "[نامفهوم]", "alternative": item["text"],
                         "flags": ["speech_gap"], "speaker": "گوینده نامشخص", "words": []})
    rows.sort(key=lambda row: (row["start"], row["end"]))
    # Assign ids only after all gap rows have been inserted.
    for index, row in enumerate(rows):
        row["id"] = f"s{index + 1}"
    return {"schema_version": 1, "duration_seconds": duration, "segments": rows,
            "speech_regions": speech, "quality_status": "needs_review"}


def review_schema(ids: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": ["segments"], "properties": {"segments": {
                "type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["id", "text", "speaker", "uncertain", "notes"],
                "properties": {"id": {"type": "string", "enum": ids or ["no-speech"]},
                               "text": {"type": "string"}, "speaker": {"type": "string"},
                               "uncertain": {"type": "boolean"}, "notes": {"type": "string"}}}}}}


def apply_speaker_evidence(evidence: dict, turns: list[dict]) -> None:
    """Attach acoustic ids, not customer/agent roles; preserve overlap as uncertain."""
    def speaker_for(start, end):
        scores = {}
        for turn in turns:
            overlap = max(0., min(end, turn["end"]) - max(start, turn["start"]))
            if overlap:
                scores[turn["speaker_id"]] = scores.get(turn["speaker_id"], 0.) + overlap
        ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        duration = max(.01, end - start)
        if not ordered or ordered[0][1] / duration < .6 or (len(ordered) > 1 and ordered[1][1] / duration > .15):
            return None
        return str(ordered[0][0])
    rows = []
    for row in evidence["segments"]:
        words = row.get("words") or []
        # Reviewed wording may differ from the old decoder word array. Never
        # resurrect those raw words merely to attach acoustic speaker labels.
        if normalize("".join(word.get("word", "") for word in words)) != normalize(row["text"]):
            words = []
        groups = []
        for word in words:
            speaker = speaker_for(word["start"], word["end"])
            if not groups or groups[-1][0] != speaker:
                groups.append((speaker, []))
            groups[-1][1].append(word)
        if len(groups) > 1:
            candidates = [dict(row, start=group[0]["start"], end=group[-1]["end"],
                               text="".join(word["word"] for word in group).strip(), words=group,
                               speaker_id=speaker, source_segment_id=row["id"])
                          for speaker, group in groups]
        else:
            candidates = [dict(row, speaker_id=speaker_for(row["start"], row["end"]))]
        for candidate in candidates:
            candidate["speaker"] = candidate["speaker_id"] or "گوینده نامشخص"
            candidate["flags"] = list(row["flags"])
            if candidate["speaker_id"] is None:
                candidate["flags"].append("speaker_uncertain")
            rows.append(candidate)
    for index, row in enumerate(rows):
        row["id"] = f"s{index + 1}"
    evidence["segments"] = rows
    evidence["speaker_turns"] = turns


def validate_review(value: dict, evidence: dict) -> list[dict]:
    """Require every evidence segment exactly once; never accept a greeting-only draft."""
    if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
        raise ValueError("Review must contain a segments array")
    source = {row["id"]: row for row in evidence["segments"]}
    seen = set()
    result = []
    for entry in value["segments"]:
        if not isinstance(entry, dict) or entry.get("id") not in source or entry["id"] in seen:
            raise ValueError("Review contains unknown or duplicate segment ids")
        seen.add(entry["id"])
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip() or text.strip() in {"...", "…"}:
            raise ValueError("Review contains an empty/placeholder segment")
        if not isinstance(entry.get("uncertain"), bool):
            raise ValueError("Review must explicitly mark uncertainty")
        if not isinstance(entry.get("speaker"), str) or not entry["speaker"].strip():
            raise ValueError("Review must identify a speaker or mark unknown")
        if not isinstance(entry.get("notes"), str):
            raise ValueError("Review notes must be text")
        if re.search(r"(.)\1{7,}|(\b\S+\s+)\2{7,}", text):
            raise ValueError(f"Review retains a decoder repetition loop in {entry['id']}")
        row = dict(source[entry["id"]])
        row["raw_text"] = row["text"]
        row.update({key: entry[key] for key in ("text", "speaker", "uncertain", "notes")})
        row["flags"] = sorted(set(row["flags"] + text_flags(text)))
        originals = set(re.findall(r"\d+", normalize(row["raw_text"] + " " + row["alternative"]
                                                    + " " + (row.get("retry") or {}).get("text", ""))))
        if any(number not in originals for number in re.findall(r"\d+", normalize(text))):
            row["flags"].append("unsupported_number")
            row["uncertain"] = True
        if entry["uncertain"] and "[نامفهوم]" not in text:
            row["flags"].append("uncertain_wording")
        if SequenceMatcher(None, normalize(row["raw_text"]), normalize(text)).ratio() < .25:
            row["flags"].append("large_revision")
        result.append(row)
    if seen != set(source):
        raise ValueError("Review omitted evidence segments")
    return sorted(result, key=lambda row: (row["start"], row["end"]))


def timestamp(value: float) -> str:
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes:02}:{seconds:02}"


def render_markdown(filename: str, segments: list[dict], *, human_reviewed: bool = False) -> str:
    lines = ["# متن تماس", "", f"- فایل صوتی: `{filename}`",
             "- بازبینی: " + ("تأیید انسانی" if human_reviewed else "نیازمند بازبینی انسانی"), "", "## مکالمه", ""]
    if not segments:
        lines.append("[گفتاری شناسایی نشد؛ نیازمند بررسی صوت]")
    for row in segments:
        label = re.sub(r"[\r\n*]", "", row["speaker"])
        warning = " [نیازمند بازبینی]" if row.get("uncertain") and "[نامفهوم]" not in row["text"] else ""
        lines.extend([f"**{label}:** `{timestamp(row['start'])}–{timestamp(row['end'])}` {row['text']}{warning}", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
