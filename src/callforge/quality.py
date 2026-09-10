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


def tight_vad_windows(
    speech_regions: list[dict] | None,
    duration: float,
    *,
    maximum_seconds: float = 15.0,
    merge_silence_seconds: float = 0.5,
) -> list[dict]:
    """Return short, non-overlapping speech windows without adding broad silence.

    Whole-call decoding is useful as a cheap first hypothesis, but telephone
    silence can poison the decoder before quiet Persian speech.  These windows
    retain the detector's own small padding and never expand to a minimum size.
    Long continuous regions are split only to keep Whisper below its 30-second
    context limit.
    """
    if not math.isfinite(duration) or duration <= 0 or maximum_seconds <= 0:
        return []
    normalized = []
    for region in speech_regions or []:
        try:
            start = max(0.0, float(region["start"]))
            end = min(duration, float(region["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start + end) or end - start < .2:
            continue
        if (normalized and start - normalized[-1]["end"] <= merge_silence_seconds
                and end - normalized[-1]["start"] <= maximum_seconds):
            normalized[-1]["end"] = end
        else:
            normalized.append({"start": start, "end": end})

    windows = []
    for region in normalized:
        start = region["start"]
        while region["end"] - start > maximum_seconds:
            windows.append({"start": round(start, 3),
                            "end": round(start + maximum_seconds, 3)})
            start += maximum_seconds
        if region["end"] - start >= .2:
            windows.append({"start": round(start, 3), "end": round(region["end"], 3)})
    return windows


def coverage_attempt_windows(plan: dict, duration: float, expand_seconds: float) -> list[dict]:
    """Try the exact speech gap before contextual and finally expanded audio.

    Model fallback and window expansion are intentionally independent.  A
    stronger model must first hear the same tight acoustic evidence that the
    primary model failed on.
    """
    tight = {
        "start": max(0.0, float(plan["gap_start"]) - .4),
        "end": min(duration, float(plan["gap_end"]) + .4),
        "mode": "tight_vad",
    }
    contextual = {"start": float(plan["start"]), "end": float(plan["end"]),
                  "mode": "contextual"}
    center = (float(plan["gap_start"]) + float(plan["gap_end"])) / 2
    width = min(duration, float(expand_seconds))
    expanded_start = max(0.0, min(duration - width, center - width / 2))
    expanded = {"start": expanded_start, "end": expanded_start + width,
                "mode": "expanded"}
    result = []
    seen = set()
    for window in (tight, contextual, expanded):
        key = (round(window["start"], 3), round(window["end"], 3))
        if window["end"] <= window["start"] or key in seen:
            continue
        seen.add(key)
        result.append(dict(window, start=key[0], end=key[1]))
    return result


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
        {**{key: row.get(key) for key in (
            "id", "start", "end", "text", "alternative", "flags", "speaker",
            "consensus_text", "confidence_tier", "uncertainty_spans",
            "candidate_sources", "reason",
            "entity_resolutions", "entity_candidates",
        )},
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


QUARANTINE_FLAGS = {
    "repetition", "compression", "invalid_asr_metrics", "decoder_truncated",
    "invalid_timestamp", "prompt_leakage",
}


def segment_timing_flags(segment: dict, duration: float) -> list[str]:
    flags = set(text_flags(str(segment.get("text", "")), segment))
    start, end = segment.get("start"), segment.get("end")
    if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(start + end) or start < 0 or end <= start
            or start >= duration or end > duration + .25):
        flags.add("invalid_timestamp")
    words = segment.get("words") or []
    if words and not aligned_words(segment):
        flags.add("invalid_timestamp")
    if sum(word.get("start") == word.get("end") for word in words) > 1:
        flags.add("invalid_timestamp")
    return sorted(flags)


def segment_is_quarantined(segment: dict, duration: float) -> bool:
    return bool(set(segment_timing_flags(segment, duration)) & QUARANTINE_FLAGS)


def validate_timeline(rows: list[dict], duration: float) -> None:
    previous_end = 0.0
    intervals = set()
    owned = set()
    for row in sorted(rows, key=lambda item: (item["start"], item["end"])):
        start, end = float(row["start"]), float(row["end"])
        if not (0 <= start < end <= duration + 1e-6):
            raise ValueError("Evidence contains a non-positive or out-of-range interval")
        interval = (round(start, 6), round(end, 6))
        if interval in intervals:
            raise ValueError("Evidence contains a duplicate interval")
        intervals.add(interval)
        if start < previous_end - 1e-6:
            raise ValueError("Evidence contains overlapping canonical intervals")
        previous_end = end
        for part in row.get("alternative_sources", []):
            key = (part.get("segment_index"), part.get("word_index"))
            if key in owned:
                raise ValueError("A hypothesis token has more than one owner")
            owned.add(key)


def consensus_for_row(row: dict) -> dict:
    """Keep agreed prefix/suffix and isolate only the unresolved span."""
    candidates = []
    for source, value in (("primary", row.get("text")), ("enhanced", row.get("alternative")),
                          ("recovery", usable_retry_text(row))):
        if source == "enhanced" and set(row.get("flags", [])) & {
                "alternative_failed_decode", "alternative_prompt_leakage"}:
            continue
        text = normalize(str(value or ""))
        if text and text != "[نامفهوم]" and not set(text_flags(text)) & {"repetition", "compression"}:
            candidates.append({"source": source, "text": text})
    groups: dict[str, list[dict]] = {}
    for candidate in candidates:
        groups.setdefault(candidate["text"], []).append(candidate)
    agreed = max(groups.values(), key=len, default=[])
    if len(agreed) >= 2:
        sensitive = bool(re.search(r"\d|تومان|ریال|درصد|شماره|کد|تاریخ", agreed[0]["text"]))
        providers = {str(row.get("primary_provider", "")), str(row.get("alternative_provider", ""))} - {""}
        families = {"whisper" if "whisper" in provider else provider for provider in providers}
        probabilities = [float(word["probability"])
                         for word in (row.get("words") or []) + row.get("alternative_sources", [])
                         if isinstance(word.get("probability"), (int, float))]
        acoustic_support = len(probabilities) >= 2 and sum(probabilities) / len(probabilities) >= .65
        if sensitive and len(families) < 2 and not acoustic_support:
            guarded = re.sub(r"\d+", "[نامفهوم]", agreed[0]["text"])
            return {"consensus_text": guarded, "confidence_tier": "unresolved",
                    "uncertainty_spans": [{"text": agreed[0]["text"],
                                           "reason": "sensitive_requires_independent_evidence"}],
                    "candidate_sources": [item["source"] for item in agreed],
                    "reason": "sensitive_requires_independent_evidence"}
        return {"consensus_text": agreed[0]["text"], "confidence_tier": "high",
                "uncertainty_spans": [], "candidate_sources": [item["source"] for item in agreed],
                "reason": "two_valid_hypotheses_agree"}
    if not candidates:
        return {"consensus_text": "[نامفهوم]", "confidence_tier": "unresolved",
                "uncertainty_spans": [{"text": "[نامفهوم]", "reason": "no_usable_decode"}],
                "candidate_sources": [], "reason": "no_usable_decode"}
    if len(candidates) == 1:
        retry_segments = (row.get("retry") or {}).get("segments") or []
        aligned = any(float(segment.get("alignment_coverage") or 0) >= .8 for segment in retry_segments)
        source_words = (retry_segments[0].get("words", []) if candidates[0]["source"] == "recovery" and retry_segments
                        else row.get("words") or [])
        probabilities = [float(word["probability"]) for word in source_words
                         if isinstance(word.get("probability"), (int, float))]
        strong = aligned or (len(probabilities) >= 2 and sum(probabilities) / len(probabilities) >= .7)
        return {"consensus_text": candidates[0]["text"], "confidence_tier": "medium" if strong else "low",
                "uncertainty_spans": [], "candidate_sources": [candidates[0]["source"]],
                "reason": "strong_acoustic_alignment" if strong else "single_valid_hypothesis"}
    primary = next((candidate for candidate in candidates if candidate["source"] == "primary"), None)
    primary_probabilities = [float(word["probability"]) for word in row.get("words", [])
                             if isinstance(word.get("probability"), (int, float))
                             and math.isfinite(float(word["probability"]))]
    primary_strong = (len(primary_probabilities) >= 2
                      and sum(primary_probabilities) / len(primary_probabilities) >= .7)
    disagreement_text = " ".join(candidate["text"] for candidate in candidates)
    sensitive = bool(re.search(
        r"\d|تومان|ریال|درصد|شماره|کد|تاریخ|نام خانوادگی|اسم(?:م| من)?",
        disagreement_text,
    ))
    if primary and primary_strong and not sensitive:
        return {"consensus_text": primary["text"], "confidence_tier": "medium",
                "uncertainty_spans": [], "candidate_sources": ["primary"],
                "reason": "strong_primary_alignment_over_disagreement"}
    left, right = candidates[0]["text"].split(), candidates[1]["text"].split()
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(left) - prefix, len(right) - prefix) and left[-1 - suffix] == right[-1 - suffix]:
        suffix += 1
    left_middle = left[prefix:len(left) - suffix if suffix else len(left)]
    right_middle = right[prefix:len(right) - suffix if suffix else len(right)]
    sensitive = bool(re.search(r"\d|تومان|ریال|درصد|شماره|کد|تاریخ|نام خانوادگی|اسم(?:م| من)?",
                               " ".join(left_middle + right_middle)))
    output = left[:prefix] + ["[نامفهوم]"] + (left[len(left) - suffix:] if suffix else [])
    reason = "sensitive_disagreement" if sensitive else "hypothesis_disagreement"
    return {"consensus_text": " ".join(output), "confidence_tier": "unresolved",
            "uncertainty_spans": [{"text": " ".join(left_middle),
                                   "candidates": [" ".join(left_middle), " ".join(right_middle)],
                                   "reason": reason}],
            "candidate_sources": [item["source"] for item in candidates], "reason": reason}


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

    enhanced_segments = enhanced.get("segments", [])
    quarantined_enhanced = [item for item in enhanced_segments if segment_is_quarantined(item, duration)]
    enhanced_units = [item for item in enhanced_segments if valid(item) and item not in quarantined_enhanced]
    raw_units = []
    for item in raw.get("segments", []):
        if segment_is_quarantined(item, duration):
            if not str(item.get("text", "")).strip() and item.get("start") == item.get("end"):
                continue
            start = item.get("start") if isinstance(item.get("start"), (int, float)) else 0.0
            end = item.get("end") if isinstance(item.get("end"), (int, float)) else start + .001
            start = min(max(0.0, float(start)), max(0.0, duration - .001))
            end = min(duration, max(start + .001, float(end)))
            failed = dict(item, start=start, end=end, text="[نامفهوم]", words=[],
                          flags=segment_timing_flags(item, duration) + ["failed_decode"],
                          failed_decode_texts=[str(item.get("text", "")).strip()])
            if (raw_units and "failed_decode" in raw_units[-1].get("flags", [])
                    and start <= float(raw_units[-1]["end"]) + .1):
                previous = raw_units[-1]
                previous["end"] = max(float(previous["end"]), end)
                previous["flags"] = sorted(set(previous["flags"] + failed["flags"]))
                previous.setdefault("failed_decode_texts", []).extend(failed["failed_decode_texts"])
            else:
                raw_units.append(failed)
        else:
            raw_units.extend(timed_units([item]))
    previous_end = 0.0
    for index, item in enumerate(sorted(raw_units, key=lambda value: (value.get("start", 0), value.get("end", 0)))):
        if not valid(item):
            continue
        start, end = max(previous_end, float(item["start"])), min(float(item["end"]), duration)
        if end <= start:
            if rows:
                rows[-1]["flags"] = sorted(set(rows[-1]["flags"] + ["invalid_timeline", "failed_decode"]))
            continue
        previous_end = end
        original = item["text"].strip()
        flags = text_flags(original, item)
        rows.append({"id": f"s{index + 1}", "start": start, "end": end,
                     "text": original, "alternative": "", "flags": flags,
                     "speaker": "گوینده نامشخص", "words": aligned_words(item), "alternative_sources": [],
                     "primary_provider": raw.get("provider") or raw.get("backend"),
                     "alternative_provider": enhanced.get("provider") or enhanced.get("backend")})

    # Never allocate tokens from a corrupt decoder segment. Its entire interval
    # is one evidence unit (or a flag on existing primary evidence), regardless
    # of how many zero-duration words the decoder emitted.
    for source_index, item in enumerate(quarantined_enhanced):
        try:
            start = max(0.0, min(duration, float(item.get("start", 0))))
            end = max(start, min(duration, float(item.get("end", start))))
        except (TypeError, ValueError):
            continue
        touched = [row for row in rows if max(0.0, min(end, row["end"]) - max(start, row["start"])) > 0]
        flags = sorted(set(segment_timing_flags(item, duration) + ["failed_decode"]))
        if touched:
            for row in touched:
                # A broken alternative is quarantined evidence; it must not
                # downgrade or erase a valid primary decode of the same audio.
                row["flags"] = sorted(set(row["flags"] + [
                    "alternative_failed_decode",
                    *(f"alternative_{flag}" for flag in flags if flag != "failed_decode"),
                ]))
        elif end > start:
            rows.append({"id": "", "start": start, "end": end, "text": "[نامفهوم]",
                         "alternative": str(item.get("text", "")).strip(), "flags": flags,
                         "speaker": "گوینده نامشخص", "words": [],
                         "alternative_sources": [{"segment_index": source_index, "word_index": None,
                                                   "start": start, "end": end,
                                                   "text": str(item.get("text", "")).strip()}],
                         "failed_decode_source": source_index})

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
    disjoint = []
    for row in rows:
        if disjoint and row["start"] < disjoint[-1]["end"]:
            if row.get("failed_decode_source") is not None:
                disjoint[-1]["flags"] = sorted(set(disjoint[-1]["flags"] + row["flags"]))
                continue
            row["start"] = disjoint[-1]["end"]
        if row["end"] > row["start"]:
            disjoint.append(row)
    rows = disjoint
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
                        if canonical:
                            # A point token outside every valid primary
                            # interval has no acoustic ownership. Dropping it
                            # is safer than publishing a burst of millisecond
                            # review units around otherwise valid speech.
                            continue
                        # A lone point has no acoustic duration. Bound the
                        # decoder envelope to the uncovered canonical gap;
                        # expanding across a valid primary row would create an
                        # impossible overlapping timeline.
                        point = float(atom["start"])
                        left = max((row["end"] for row in canonical if row["end"] <= point),
                                   default=float(item["start"]))
                        right = min((row["start"] for row in canonical if row["start"] >= point),
                                    default=float(item["end"]))
                        atom = dict(atom, start=max(float(item["start"]), left),
                                    end=min(float(item["end"]), right))
                        if atom["end"] <= atom["start"]:
                            continue
                    owner = {"start": atom["start"], "end": min(duration, atom["end"]),
                             "text": "[نامفهوم]", "alternative": "", "flags": ["speech_gap"],
                             "speaker": "گوینده نامشخص", "words": [], "alternative_sources": [],
                             "enhanced_only": True}
                    rows.append(owner)
                owner["end"] = min(duration, atom["end"])
            owner["alternative_sources"].append({"segment_index": source_index,
                "word_index": word_index if words else None, "start": atom["start"],
                "end": atom["end"], "text": atom["word"],
                "probability": atom.get("probability")})
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
        row.update(consensus_for_row(row))
    validate_timeline(rows, duration)
    return {"schema_version": 2, "duration_seconds": duration, "segments": rows,
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
        unsupported = {number for number in re.findall(r"\d+", normalize(text)) if number not in originals}
        if unsupported:
            row["flags"].append("unsupported_number")
            row["uncertain"] = True
            # Fail closed at the token span: retain supported surrounding text,
            # but never publish a plausible-looking number with no ASR source.
            text = re.sub(r"\d+", lambda match: "[نامفهوم]" if normalize(match.group()) in unsupported else match.group(), text)
            row["text"] = text
        if entry["uncertain"] and "[نامفهوم]" not in text:
            row["flags"].append("uncertain_wording")
        if SequenceMatcher(None, normalize(row["raw_text"]), normalize(text)).ratio() < .25:
            row["flags"].append("large_revision")
        unresolved = [
            {"start_char": match.start(), "end_char": match.end(),
             "text": match.group(), "reason": "review_unresolved"}
            for match in re.finditer(r"\[نامفهوم\]", text)
        ]
        row["uncertainty_spans"] = unresolved or list(row.get("uncertainty_spans") or [])
        row["confidence_tier"] = "unresolved" if unresolved else (
            "low" if row["uncertain"] else str(row.get("confidence_tier") or "medium"))
        row["candidate_sources"] = list(row.get("candidate_sources") or [])
        row["reason"] = "review_unresolved" if unresolved else str(row.get("reason") or "reviewed")
        result.append(row)
    if seen != set(source):
        raise ValueError("Review omitted evidence segments")
    return sorted(result, key=lambda row: (row["start"], row["end"]))


def validate_publish_quality(rows: list[dict], *, maximum_unresolved_ratio: float = .35) -> None:
    """Fail closed when an automated review is still mostly unresolved or looping.

    A segment containing one unresolved span can still preserve a useful long
    prefix and suffix.  Count only marker-only segments here; token-span detail
    remains available in ``uncertainty_spans`` and the human review queue.
    """
    if not rows:
        raise ValueError("No reviewed speech was produced")
    looping = [row.get("id", "?") for row in rows
               if "repetition" in text_flags(str(row.get("text", "")))]
    if looping:
        raise ValueError(f"Reviewed output still contains decoder repetition: {', '.join(looping)}")
    unresolved = sum(normalize(str(row.get("text", ""))) == "[نامفهوم]" for row in rows)
    ratio = unresolved / len(rows)
    if ratio > maximum_unresolved_ratio:
        raise ValueError(
            f"Reviewed output is too unresolved to publish ({unresolved}/{len(rows)}, {ratio:.0%})"
        )


def timestamp(value: float, *, round_up: bool = False) -> str:
    seconds_total = math.ceil(value) if round_up else math.floor(value)
    minutes, seconds = divmod(max(0, seconds_total), 60)
    return f"{minutes:02}:{seconds:02}"


def display_rows(segments: list[dict], *, maximum_seconds: float = 15.0,
                 maximum_gap_seconds: float = .35) -> list[dict]:
    """Coalesce adjacent turns for readable Markdown without changing evidence.

    Review audio and quality JSON intentionally retain their precise, short
    segments.  Only presentation rows from an acoustically identified speaker
    are joined: merging unknown speakers could incorrectly combine two people.
    """
    rows: list[dict] = []
    for source in segments:
        row = dict(source)
        speaker = str(row.get("speaker") or "گوینده نامشخص").strip()
        row["speaker"] = speaker
        previous = rows[-1] if rows else None
        gap = float(row["start"]) - float(previous["end"]) if previous else math.inf
        can_join = bool(
            previous
            and speaker == previous["speaker"]
            and speaker != "گوینده نامشخص"
            and 0 <= gap <= maximum_gap_seconds
            and float(row["end"]) - float(previous["start"]) <= maximum_seconds
        )
        if not can_join:
            rows.append(row)
            continue
        previous["end"] = row["end"]
        previous["text"] = f'{str(previous.get("text", "")).rstrip()} {str(row.get("text", "")).lstrip()}'.strip()
        previous["uncertain"] = bool(previous.get("uncertain") or row.get("uncertain"))
    return rows


def render_markdown(filename: str, segments: list[dict], *, human_reviewed: bool = False) -> str:
    lines = ["# متن تماس", "", f"- فایل صوتی: `{filename}`",
             "- بازبینی: " + ("تأیید انسانی" if human_reviewed else "نیازمند بازبینی انسانی"), "", "## مکالمه", ""]
    if not segments:
        lines.append("[گفتاری شناسایی نشد؛ نیازمند بررسی صوت]")
    for row in display_rows(segments):
        label = re.sub(r"[\r\n*]", "", row["speaker"])
        warning = " [نیازمند بازبینی]" if row.get("uncertain") and "[نامفهوم]" not in row["text"] else ""
        lines.extend([f"**{label}:** `{timestamp(row['start'])}–{timestamp(row['end'], round_up=True)}` {row['text']}{warning}", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json_dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
