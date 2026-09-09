"""Local, reproducible evaluation against explicitly human-approved revisions."""
from __future__ import annotations

import random
import re
from collections import Counter

from callforge.quality import normalize


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize(text).replace("\u200c", " "), flags=re.UNICODE)


def edit_distance(reference: list, hypothesis: list) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str) -> dict:
    ref, hyp = tokens(reference), tokens(hypothesis)
    ref_chars, hyp_chars = list("".join(ref)), list("".join(hyp))
    reference_numbers = Counter(re.findall(r"\d+", normalize(reference)))
    hypothesis_numbers = Counter(re.findall(r"\d+", normalize(hypothesis)))
    matched = sum((reference_numbers & hypothesis_numbers).values())
    return {"word_errors": edit_distance(ref, hyp), "reference_words": len(ref),
            "character_errors": edit_distance(ref_chars, hyp_chars), "reference_characters": len(ref_chars),
            "wer": edit_distance(ref, hyp) / len(ref) if ref else None,
            "cer": edit_distance(ref_chars, hyp_chars) / len(ref_chars) if ref_chars else None,
            "matched_numbers": matched, "reference_numbers": sum(reference_numbers.values()),
            "hypothesis_numbers": sum(hypothesis_numbers.values()),
            "number_recall": matched / sum(reference_numbers.values()) if reference_numbers else None,
            "number_precision": matched / sum(hypothesis_numbers.values()) if hypothesis_numbers else None}


def sample_calls(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin duration/direction strata, deterministic within a frozen inventory."""
    rng = random.Random(seed)
    strata = {}
    for row in sorted(rows, key=lambda row: row["id"]):
        duration = row.get("duration_seconds") or 0
        band = "short" if duration < 60 else "medium" if duration < 180 else "long"
        strata.setdefault((row.get("direction") or "unknown", band), []).append(row)
    for group in strata.values():
        rng.shuffle(group)
    result = []
    while strata and len(result) < count:
        for key in sorted(list(strata)):
            if len(result) >= count:
                break
            row = dict(strata[key].pop())
            result.append(row)
            if not strata[key]:
                del strata[key]
    if len(result) == 40:
        for index, row in enumerate(result):
            row["evaluation_split"] = "development" if index < 30 else "holdout"
    else:
        # Preserve the legacy deterministic split for partial review batches;
        # the promotion gate still requires the fixed 30/10 corpus.
        for index, row in enumerate(result):
            row["evaluation_split"] = "holdout" if index % 5 == 0 else "development"
    return result


def transcript_text(content: str, data: dict) -> str:
    if "segments" in data:
        return " ".join(row["text"] for row in data["segments"])
    body = content.split("## مکالمه", 1)[-1]
    body = re.sub(r"\*\*[^*\n]+:\*\*", "", body)
    return re.sub(r"(?:\[\d+:\d+[–-]\d+:\d+\]|`\d+:\d+[–-]\d+:\d+`)", "", body).strip()
