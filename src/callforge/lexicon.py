"""Evidence-grounded normalization for domain terms and sensitive names."""
from __future__ import annotations

import re
from dataclasses import replace

from callforge.config import GlossaryTerm
from callforge.quality import normalize


def _words(value: str) -> list[str]:
    return [normalize(part).replace("\u200c", "")
            for part in re.findall(r"[\w\u200c]+", value, flags=re.UNICODE)]


def _aliases(term: GlossaryTerm) -> tuple[str, ...]:
    return tuple(dict.fromkeys((term.canonical, *term.aliases)))


def _contains_alias(text: str, term: GlossaryTerm) -> str | None:
    tokens = _words(text)
    for alias in sorted(_aliases(term), key=lambda item: len(_words(item)), reverse=True):
        wanted = _words(alias)
        if wanted and any(tokens[index:index + len(wanted)] == wanted
                          for index in range(len(tokens) - len(wanted) + 1)):
            return alias
    return None


def _replace_text(text: str, aliases: tuple[str, ...], canonical: str) -> tuple[str, int]:
    changed = 0
    result = text
    for alias in sorted(aliases, key=lambda item: len(_words(item)), reverse=True):
        if _words(alias) == _words(canonical):
            continue
        parts = [re.escape(part) for part in alias.split() if part]
        if not parts:
            continue
        pattern = re.compile(r"(?<![\w\u200c])" + r"\s+".join(parts) + r"(?![\w\u200c])",
                             flags=re.UNICODE | re.IGNORECASE)
        result, count = pattern.subn(canonical, result)
        changed += count
    return result, changed


def _rewrite_word_array(words: list[dict], term: GlossaryTerm) -> tuple[list[dict], int]:
    result = list(words)
    changed = 0
    for alias in sorted(_aliases(term), key=lambda item: len(_words(item)), reverse=True):
        wanted = _words(alias)
        if not wanted or wanted == _words(term.canonical):
            continue
        index = 0
        while index <= len(result) - len(wanted):
            if [_words(str(row.get("word", "")))[0] if len(_words(str(row.get("word", "")))) == 1 else ""
                    for row in result[index:index + len(wanted)]] != wanted:
                index += 1
                continue
            group = result[index:index + len(wanted)]
            first, last = str(group[0].get("word", "")), str(group[-1].get("word", ""))
            leading = re.match(r"^\s*", first).group()
            trailing = re.search(r"[^\w\u0600-\u06ff\u200c]*$", last, flags=re.UNICODE).group()
            merged = dict(group[0])
            merged["word"] = f"{leading}{term.canonical}{trailing}"
            merged["start"] = group[0].get("start")
            merged["end"] = group[-1].get("end")
            probabilities = [float(row["probability"]) for row in group
                             if isinstance(row.get("probability"), (int, float))]
            if probabilities:
                merged["probability"] = min(probabilities)
            merged["lexicon_alias"] = alias
            merged["lexicon_canonical"] = term.canonical
            result[index:index + len(wanted)] = [merged]
            changed += 1
            index += 1
    return result, changed


def normalize_result(result: dict, terms: tuple[GlossaryTerm, ...], *, include_sensitive: bool = False) -> int:
    """Rewrite only aliases that already exist in a hypothesis."""
    replacements = 0
    selected = [term for term in terms
                if include_sensitive or not term.requires_acoustic_validation]
    for segment in result.get("segments", []):
        for term in selected:
            words = segment.get("words") or []
            if words:
                rewritten, count = _rewrite_word_array(words, term)
                if count:
                    segment["words"] = rewritten
                    segment["text"] = "".join(str(word.get("word", "")) for word in rewritten).strip()
                    replacements += count
                    continue
            segment["text"], count = _replace_text(
                str(segment.get("text", "")), _aliases(term), term.canonical)
            replacements += count
    if result.get("segments"):
        result["text"] = " ".join(str(row.get("text", "")) for row in result["segments"]).strip()
    else:
        for term in selected:
            result["text"], count = _replace_text(
                str(result.get("text", "")), _aliases(term), term.canonical)
            replacements += count
    if replacements:
        provenance = result.setdefault("provenance", {})
        provenance["glossary_mode"] = "present_alias_normalization"
        provenance["glossary_replacements"] = int(provenance.get("glossary_replacements", 0)) + replacements
    return replacements


def _word_spans(words: list[dict], aliases: tuple[str, ...], key: str) -> list[dict]:
    values = [_words(str(row.get(key, ""))) for row in words]
    spans = []
    for alias in aliases:
        wanted = _words(alias)
        if not wanted:
            continue
        for index in range(len(values) - len(wanted) + 1):
            if [part[0] if len(part) == 1 else "" for part in values[index:index + len(wanted)]] != wanted:
                continue
            group = words[index:index + len(wanted)]
            start, end = group[0].get("start"), group[-1].get("end")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
                spans.append({"start": float(start), "end": float(end), "observed": alias})
    return spans


def alignment_requests(evidence: dict, terms: tuple[GlossaryTerm, ...]) -> tuple[list[dict], list[dict]]:
    """Build one CTC comparison group for each observed person-name alias."""
    sensitive = tuple(term for term in terms if term.requires_acoustic_validation)
    requests, groups = [], []
    for row in evidence.get("segments", []):
        combined = " ".join(str(value or "") for value in
                            (row.get("text"), row.get("alternative"),
                             (row.get("retry") or {}).get("text")))
        for trigger in sensitive:
            observed = _contains_alias(combined, trigger)
            if not observed:
                continue
            if trigger.contexts and not any(normalize(context) in normalize(combined)
                                            for context in trigger.contexts):
                continue
            spans = _word_spans(row.get("words") or [], _aliases(trigger), "word")
            spans += _word_spans(row.get("alternative_sources") or [], _aliases(trigger), "text")
            for segment in (row.get("retry") or {}).get("segments", []):
                spans += _word_spans(segment.get("words") or [], _aliases(trigger), "word")
            if not spans and float(row.get("end", 0)) - float(row.get("start", 0)) <= 8:
                spans = [{"start": float(row["start"]), "end": float(row["end"]),
                          "observed": observed}]
            seen = set()
            for span in spans:
                interval = (round(span["start"], 2), round(span["end"], 2))
                if interval in seen:
                    continue
                seen.add(interval)
                group_index = len(groups)
                candidates = []
                for candidate_index, candidate in enumerate(sensitive):
                    request_id = f"lexicon-{group_index}-{candidate_index}"
                    requests.append({"id": request_id, "start": span["start"], "end": span["end"],
                                     "text": candidate.canonical})
                    candidates.append({"id": request_id, "term": candidate})
                groups.append({"row": row, "trigger": trigger, "observed": span["observed"],
                               "start": span["start"], "end": span["end"],
                               "candidates": candidates})
    return requests, groups


def apply_alignment(evidence: dict, groups: list[dict], alignment: dict) -> list[dict]:
    """Apply a person name only when CTC score and runner-up margin pass."""
    aligned = {row.get("id"): row for row in alignment.get("segments", [])}
    decisions = []
    resolved_rows: set[tuple[str, str]] = set()
    for group in groups:
        scores = []
        for candidate in group["candidates"]:
            row = aligned.get(candidate["id"], {})
            words = row.get("words") or []
            usable = bool(words and all(isinstance(word.get("score"), (int, float))
                                        and isinstance(word.get("character_hits"), (int, float))
                                        for word in words))
            score = (sum(float(word["score"]) for word in words) / len(words)) if usable else 0.0
            hits = (sum(float(word["character_hits"]) for word in words) / len(words)) if usable else 0.0
            scores.append((score, hits, candidate["term"]))
        scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_hits, best = scores[0] if scores else (0.0, 0.0, group["trigger"])
        runner_up = scores[1][0] if len(scores) > 1 else None
        accepted = (best_score >= best.min_alignment_score
                    and best_hits >= best.min_character_hits
                    and (runner_up is None or best_score - runner_up >= best.min_alignment_margin))
        decision = {"row_id": group["row"].get("id"), "observed": group["observed"],
                    "canonical": best.canonical if accepted else None,
                    "score": round(best_score, 4), "character_hits": round(best_hits, 4),
                    "runner_up_score": round(runner_up, 4) if runner_up is not None else None,
                    "accepted": accepted, "method": "persian_ctc_forced_alignment",
                    "candidates": [{"canonical": term.canonical, "score": round(score, 4),
                                    "character_hits": round(hits, 4)}
                                   for score, hits, term in scores]}
        decisions.append(decision)
        group["row"].setdefault("entity_candidates", []).append(decision)
        if not accepted:
            continue
        row = group["row"]
        identity = (str(row.get("id")), best.canonical)
        if identity in resolved_rows:
            continue
        resolved_rows.add(identity)
        replacement = replace(group["trigger"], canonical=best.canonical, kind="term",
                              aliases=tuple(dict.fromkeys((*group["trigger"].aliases,
                                                          group["trigger"].canonical))))
        for key in ("text", "alternative"):
            row[key], _ = _replace_text(str(row.get(key, "")), _aliases(replacement), best.canonical)
        rewritten, _ = _rewrite_word_array(row.get("words") or [], replacement)
        if rewritten:
            row["words"] = rewritten
        for source in row.get("alternative_sources") or []:
            source["text"], _ = _replace_text(str(source.get("text", "")),
                                                _aliases(replacement), best.canonical)
        retry = row.get("retry") or {}
        normalize_result(retry, (replacement,), include_sensitive=True)
        row.setdefault("entity_resolutions", []).append(decision)
        row["flags"] = sorted(set(row.get("flags", []) + ["lexicon_ctc_validated"]))
    evidence["lexicon_validation"] = decisions
    return decisions


def guard_unverified_entities(row: dict) -> None:
    """Prevent an unvalidated configured person alias from becoming consensus."""
    rejected = [item for item in row.get("entity_candidates", []) if not item.get("accepted")]
    if not rejected or row.get("entity_resolutions"):
        return
    text = str(row.get("consensus_text", ""))
    changed = 0
    for item in rejected:
        text, count = _replace_text(text, (str(item.get("observed", "")),), "[نامفهوم]")
        changed += count
    if not changed:
        return
    row["consensus_text"] = text
    row["confidence_tier"] = "unresolved"
    row["reason"] = "sensitive_entity_requires_acoustic_evidence"
    row["uncertainty_spans"] = [{"text": item.get("observed"),
                                  "reason": "ctc_score_or_margin_below_threshold"}
                                 for item in rejected]
    row["flags"] = sorted(set(row.get("flags", []) + ["sensitive_entity_unverified"]))


def term_snapshot(terms: tuple[GlossaryTerm, ...]) -> list[dict]:
    return [{"canonical": term.canonical, "type": term.kind,
             "aliases": list(term.aliases), "contexts": list(term.contexts),
             "requires_acoustic_validation": term.requires_acoustic_validation,
             "min_alignment_score": term.min_alignment_score,
             "min_alignment_margin": term.min_alignment_margin,
             "min_character_hits": term.min_character_hits}
            for term in terms]
