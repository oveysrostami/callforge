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
from callforge.json_utils import dumps as json_dumps


class ArtifactConflictError(RuntimeError):
    """A user changed the output during processing; automatic retry is unsafe."""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(str.maketrans("يكى۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "یکی01234567890123456789"))).strip()


def text_flags(text: str, segment: dict | None = None) -> list[str]:
    flags = list((segment or {}).get("flags", []))
    if re.search(r"(.)\1{5,}|(\b\S+\s+)\2{3,}", text):
        flags.append("repetition")
    if re.search(r"\d", normalize(text)):
        flags.append("verify_numbers")
    if "[نامفهوم]" in text or not text.strip():
        flags.append("unclear")
    segment = segment or {}
    if any(isinstance(segment.get(key), (int, float)) and not math.isfinite(segment[key])
           for key in ("avg_logprob", "no_speech_prob", "compression_ratio")):
        flags.append("invalid_asr_metrics")
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
    priority = (0 if flags & {"repetition", "invalid_asr_metrics", "decoder_truncated"} else 1 if "compression" in flags
                else 2 if "low_logprob" in flags else 3 if "pass_disagreement" in flags else 4)
    return priority, float(row["start"])


def prompt_leakage(text: str, prompt: str) -> bool:
    """Detect a decoder echo of the neutral context prompt, not a content guess."""
    left, right = normalize(text), normalize(prompt)
    return bool(left and right and len(left) >= 8
                and SequenceMatcher(None, left, right).ratio() >= .65)


def usable_retry_text(row: dict) -> str | None:
    retry = row.get("retry") or {}
    prompt = str((retry.get("settings") or {}).get("prompt") or "")
    if prompt_leakage(str(retry.get("text") or ""), prompt):
        return None
    if any(set(text_flags(s.get("text", ""), s)) & {
               "repetition", "compression", "possible_non_speech",
               "invalid_asr_metrics", "decoder_truncated", "prompt_leakage"}
           for s in retry.get("segments", [])):
        return None
    text = retry.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    if set(text_flags(text)) & {"repetition", "unclear"}:
        return None
    return text.strip()


def coverage_retry_candidates(evidence: dict) -> list[dict]:
    """Return speech spans for which neither main Whisper pass has usable text."""
    speech = evidence.get("speech_regions") or []

    def overlaps_detected_speech(row: dict) -> bool:
        if not speech:
            return True
        return any(
            float(region["start"]) < float(row["end"])
            and float(region["end"]) > float(row["start"])
            for region in speech
        )

    rows = []
    for row in evidence.get("segments", []):
        flags = set(row.get("flags", []))
        primary_unusable = (
            normalize(str(row.get("text", ""))) in {"", "[نامفهوم]"}
            or bool(flags & {"repetition", "compression", "invalid_asr_metrics", "decoder_truncated", "prompt_leakage"})
        )
        alternative_flags = set(text_flags(str(row.get("alternative", ""))))
        alternative_unusable = (
            normalize(str(row.get("alternative", ""))) in {"", "[نامفهوم]"}
            or "alternative_prompt_leakage" in flags
            or bool(alternative_flags & {"repetition", "compression", "invalid_asr_metrics", "decoder_truncated", "prompt_leakage"})
        )
        if (primary_unusable and alternative_unusable and usable_retry_text(row) is None
                and overlaps_detected_speech(row)):
            rows.append(row)
    return sorted(rows, key=lambda row: (float(row["start"]), float(row["end"])))


def coverage_retry_windows(
    evidence: dict,
    *,
    minimum_seconds: float = 30.0,
    maximum_seconds: float = 30.0,
    context_seconds: float = 3.0,
    merge_silence_seconds: float = 3.0,
) -> list[dict]:
    """Group VAD-only gaps into contextual retries without decoding tiny turns alone."""
    duration = float(evidence.get("duration_seconds", 0))
    if duration <= 0 or not 0 < minimum_seconds <= maximum_seconds:
        return []
    gaps = coverage_retry_candidates(evidence)
    if not gaps:
        return []

    maximum_core = max(0.1, maximum_seconds - 2 * context_seconds)
    groups: list[list[dict]] = []
    for gap in gaps:
        if (groups
                and float(gap["start"]) - float(groups[-1][-1]["end"]) <= merge_silence_seconds
                and float(gap["end"]) - float(groups[-1][0]["start"]) <= maximum_core):
            groups[-1].append(gap)
        else:
            groups.append([gap])

    plans = []
    for group in groups:
        core_start = float(group[0]["start"])
        core_end = float(group[-1]["end"])
        target = min(
            duration,
            max(minimum_seconds, min(maximum_seconds, core_end - core_start + 2 * context_seconds)),
        )
        centered_start = (core_start + core_end - target) / 2
        start = max(0.0, min(duration - target, centered_start))
        end = start + target
        plans.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "gap_start": round(core_start, 3),
            "gap_end": round(core_end, 3),
            "segment_ids": [str(row["id"]) for row in group],
        })
    return plans


def aligned_retry_for_gap(retry: dict, start: float, end: float) -> dict | None:
    """Keep only finite, non-looping retry words that land inside one VAD gap."""
    selected_segments = []
    prompt = str((retry.get("settings") or {}).get("prompt") or "")
    for segment in retry.get("segments", []):
        try:
            segment_start, segment_end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        segment_text = str(segment.get("text", ""))
        flags = set(text_flags(segment_text, segment))
        if prompt_leakage(segment_text, prompt):
            flags.add("prompt_leakage")
        if (not math.isfinite(segment_start + segment_end) or segment_end <= segment_start
                or flags & {"repetition", "compression", "possible_non_speech",
                            "invalid_asr_metrics", "decoder_truncated", "prompt_leakage", "unclear"}):
            continue
        words = []
        for word in segment.get("words") or []:
            try:
                word_start, word_end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            midpoint = (word_start + word_end) / 2
            if (math.isfinite(word_start + word_end) and word_end > word_start
                    and start <= midpoint <= end and str(word.get("word", "")).strip()):
                words.append(dict(word))
        if words:
            text = "".join(str(word["word"]) for word in words).strip()
            probabilities = [float(word["probability"]) for word in words
                             if isinstance(word.get("probability"), (int, float))
                             and math.isfinite(float(word["probability"]))]
            mean_probability = sum(probabilities) / len(probabilities) if probabilities else None
            # A lone low-probability token is not a recovery. Numbers require
            # stronger evidence because a plausible-looking wrong amount is
            # worse than an explicit unresolved marker.
            if (mean_probability is not None
                    and (mean_probability < .3
                         or (len(words) == 1 and mean_probability < .5)
                         or (re.search(r"\d", normalize(text)) and mean_probability < .5))):
                continue
            clipped = dict(segment, start=words[0]["start"], end=words[-1]["end"],
                           text=text, words=words, word_probability_mean=mean_probability)
        else:
            overlap = max(0.0, min(end, segment_end) - max(start, segment_start))
            if overlap / max(0.001, segment_end - segment_start) < 0.8:
                continue
            text = str(segment.get("text", "")).strip()
            clipped = dict(segment, start=max(start, segment_start),
                           end=min(end, segment_end), text=text, words=[])
        if not text or set(text_flags(text, clipped)) & {
                "repetition", "compression", "possible_non_speech", "unclear",
                "invalid_asr_metrics", "decoder_truncated"}:
            continue
        selected_segments.append(clipped)
    text = " ".join(segment["text"] for segment in selected_segments).strip()
    if not text or set(text_flags(text)) & {"repetition", "unclear"}:
        return None
    return {
        "text": text,
        "segments": selected_segments,
        "source_window": {"start": retry.get("start"), "end": retry.get("end")},
        "alignment": "word_midpoint_or_dominant_segment",
    }


def attach_coverage_retry(evidence: dict, retry: dict, segment_ids: list[str]) -> int:
    """Attach a contextual result once, partitioned by canonical gap timestamps."""
    wanted = set(segment_ids)
    recovered = 0
    for row in evidence.get("segments", []):
        if row.get("id") not in wanted:
            continue
        aligned = aligned_retry_for_gap(retry, float(row["start"]), float(row["end"]))
        if aligned is None:
            continue
        row["retry"] = aligned
        row["flags"] = sorted(set(row.get("flags", []) + ["coverage_recovered"]))
        recovered += 1
    return recovered


def compact_review_input(evidence: dict) -> dict:
    """Keep the complete call context, not duplicated word-level diagnostics."""
    return {"duration_seconds": evidence["duration_seconds"], "segments": [
        {**{key: row.get(key) for key in ("id", "start", "end", "text", "alternative", "flags", "speaker")},
         "retry_text": usable_retry_text(row)}
        for row in evidence["segments"]]}


def aligned_words(segment: dict) -> list[dict]:
    """Only partition wording with a complete, finite, ordered word timeline."""
    words = segment.get("words") or []
    if not words or normalize("".join(str(w.get("word", "")) for w in words)) != normalize(segment["text"]):
        return []
    previous = -math.inf
    for word in words:
        start, end = word.get("start"), word.get("end")
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start + end) or start < 0 or end < start or start < previous):
            return []
        previous = end
    return words


def timed_units(segments: list[dict], max_seconds: float = 8) -> list[dict]:
    """Split on real word boundaries, retaining unsplittable text without guessing times."""
    units = []
    for segment in segments:
        words = aligned_words(segment)
        if not words:
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
    """Each enhanced word/unsplittable utterance has exactly one canonical owner.

    Segment-level overlap is not word alignment. An unsplittable enhanced phrase
    spanning several raw rows makes a joint review unit, never repeated copies.
    """
    rows = []
    def valid(item, *, allow_point=False):
        start, end = item.get("start"), item.get("end")
        return (isinstance(start, (int, float)) and isinstance(end, (int, float))
                and math.isfinite(start + end) and 0 <= start < duration
                and (end >= start if allow_point else end > start))

    enhanced_units = [item for item in enhanced.get("segments", []) if valid(item)]
    for index, item in enumerate(timed_units(raw.get("segments", []))):
        if not valid(item):
            continue
        start, end = float(item["start"]), min(float(item["end"]), duration)
        original = item["text"].strip()
        flags = text_flags(original, item)
        rows.append({"id": f"s{index + 1}", "start": start, "end": end,
                     "text": original, "alternative": "", "flags": flags,
                     "speaker": "گوینده نامشخص", "words": aligned_words(item), "alternative_sources": []})

    def overlap(left, right):
        return max(0., min(left["end"], right["end"]) - max(left["start"], right["start"]))

    # Merge only when word alignment is unavailable; do not invent where an
    # indivisible enhanced sentence should be cut. Preserve all raw wording.
    for item in enhanced_units:
        if aligned_words(item):
            continue
        touched = [row for row in rows if overlap(item, row) > 0]
        if len(touched) > 1:
            touched.sort(key=lambda row: (row["start"], row["end"]))
            merged = dict(touched[0], start=min(row["start"] for row in touched),
                          end=max(row["end"] for row in touched),
                          text=" ".join(row["text"] for row in touched), words=[],
                          flags=sorted({flag for row in touched for flag in row["flags"]}
                                       | {"alternative_timing_uncertain"}))
            rows = [row for row in rows if row not in touched] + [merged]
    rows.sort(key=lambda row: (row["start"], row["end"]))
    canonical = list(rows)
    for source_index, item in enumerate(enhanced_units):
        words = aligned_words(item)
        atoms = words or [dict(item, word=item["text"])]
        for word_index, atom in enumerate(atoms):
            if not valid(atom, allow_point=True) or not atom["word"].strip():
                continue
            matches = [(overlap(atom, row), i, row) for i, row in enumerate(canonical)]
            # Point words at an exact boundary belong to the following interval.
            if atom["start"] == atom["end"]:
                matches = [(1. if row["start"] <= atom["start"] < row["end"] else 0., i, row)
                           for _, i, row in matches]
            matches = [match for match in matches if match[0] > 0]
            if matches:
                owner = max(matches, key=lambda match: (match[0], -match[1]))[2]
            else:
                # Real enhanced-only words remain available, including brief
                # replies. Never copy a full phrase into an uncovered sliver.
                owner = rows[-1] if rows else None
                if not (owner and owner.get("enhanced_only") and owner["end"] <= atom["start"]
                        and atom["start"] - owner["end"] <= .65 and atom["end"] - owner["start"] <= 8):
                    if atom["end"] == atom["start"]:
                        # A lone point has no acoustic duration. Use its
                        # enclosing decoder interval only when no adjacent
                        # enhanced-only unit can own it.
                        atom = dict(atom, start=item["start"], end=item["end"])
                    owner = {"start": atom["start"], "end": min(duration, atom["end"]),
                             "text": "[نامفهوم]", "alternative": "", "flags": ["speech_gap"],
                             "speaker": "گوینده نامشخص", "words": [], "alternative_sources": [],
                             "enhanced_only": True}
                    rows.append(owner)
                owner["end"] = min(duration, atom["end"])
            owner["alternative_sources"].append({"segment_index": source_index,
                "word_index": word_index if words else None, "start": atom["start"],
                "end": atom["end"], "text": atom["word"]})
            if "prompt_leakage" in item.get("flags", []):
                owner["flags"].append("alternative_prompt_leakage")
    for row in rows:
        parts = row["alternative_sources"]
        # Within a segment Whisper word tokens retain their own spacing.
        groups = []
        for part in parts:
            if groups and groups[-1][0] == part["segment_index"]:
                groups[-1][1] += part["text"]
            else:
                groups.append([part["segment_index"], part["text"]])
        row["alternative"] = " ".join(text.strip() for _, text in groups).strip()
        if row["alternative"] and SequenceMatcher(None, normalize(row["text"]), normalize(row["alternative"])).ratio() < .65:
            row["flags"].append("pass_disagreement")
        row["flags"] = sorted(set(row["flags"]))
    # VAD uses 400 ms padding: it is not evidence of missing words at boundaries.
    candidates = [{"start": s["start"] + .4, "end": s["end"] - .4,
                   "text": "[نامفهوم]", "vad_only": True} for s in speech or []]
    for item in candidates:
        start, end = max(0., float(item["start"])), min(duration, float(item["end"]))
        if end <= start:
            continue
        uncovered = [(start, end)]
        # Broad decoder segment envelopes can contain long omitted passages.
        # Only wording actually assigned above counts as enhanced coverage.
        covered = rows + [part for row in rows for part in row.get("alternative_sources", [])]
        for row in covered:
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
            # Long detector-only regions become reviewable timeline units, but
            # contextual recovery still groups them before running Whisper.
            pieces = max(1, math.ceil((b - a) / 8.0)) if item.get("vad_only") else 1
            width = (b - a) / pieces
            for piece in range(pieces):
                part_start = a + piece * width
                part_end = b if piece == pieces - 1 else a + (piece + 1) * width
                gap_flags = ["speech_gap"]
                if "prompt_leakage" in item.get("flags", []):
                    gap_flags.append("alternative_prompt_leakage")
                rows.append({"id": f"s{len(rows) + 1}", "start": part_start, "end": part_end,
                             "text": "[نامفهوم]", "alternative": item["text"],
                             "flags": gap_flags, "speaker": "گوینده نامشخص", "words": []})
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
        if set(row["flags"]) & {"invalid_asr_metrics", "decoder_truncated"}:
            row["uncertain"] = True
        originals = set(re.findall(r"\d+", normalize(row["raw_text"] + " " + row["alternative"]
                                                    + " " + (usable_retry_text(row) or ""))))
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
    path.write_text(json_dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
