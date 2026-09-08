"""Local CTC forced alignment of existing text; no transcription or role inference."""
from __future__ import annotations

import re
import unicodedata

MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
REVISION = "234714078a1398a9db88194c5a40fefe6f376dc1"


def text_units(text: str, vocabulary: dict) -> list[dict]:
    """Preserve literal character offsets. Unknown words/numbers are never expanded."""
    units = []
    for match in re.finditer(r"\S+", text):
        literal = match.group()
        normalized = literal.translate(str.maketrans("يك", "یک")).lower()
        normalized = "".join(c for c in normalized if unicodedata.category(c)[0] not in {"P", "M"}
                             and c not in {"\u200c", "\u200d"})
        ids = ([vocabulary[c] for c in normalized] if normalized and "[" not in literal
               and not any(c.isdigit() for c in literal) and all(c in vocabulary for c in normalized) else [])
        units.append({"text": literal, "char_start": match.start(), "char_end": match.end(), "token_ids": ids})
    return units


def ctc_path(log_probs, targets: list[int], blank: int) -> list[list[int]]:
    """Viterbi CTC path with free prefix/suffix and wildcard (-1) for unalignable words.

    Wildcard timing is never emitted as a known word. No nearest-speaker filling.
    """
    import numpy as np
    emission = np.asarray(log_probs)
    if emission.ndim != 2 or not np.isfinite(emission).all() or not targets:
        raise ValueError("Invalid CTC emissions/targets")
    states = [blank]
    for token in targets:
        states.extend([token, blank])
    if any(t < -1 or t >= emission.shape[1] or t == blank for t in targets):
        raise ValueError("Invalid CTC target")
    size = len(states)
    previous = np.full(size, -np.inf)
    previous[0] = 0.
    back = np.zeros((len(emission), size), dtype=np.int8)
    skip = np.array([s > 1 and states[s] != blank and states[s] != states[s - 2] for s in range(size)])
    for frame, values in enumerate(emission):
        options = np.stack([previous, np.r_[-np.inf, previous[:-1]], np.r_[[-np.inf] * 2, previous[:-2]]])
        options[2, ~skip] = -np.inf
        choice = options.argmax(axis=0)
        wildcard = np.max(np.delete(values, blank))
        scores = np.array([wildcard if t == -1 else values[t] for t in states])
        # Padding can contain unrelated speech, but cannot swallow the first/
        # last target character for free (that would collapse word boundaries).
        if targets[0] >= 0:
            scores[0] = np.max(np.delete(values, targets[0]))
        if targets[-1] >= 0:
            scores[-1] = np.max(np.delete(values, targets[-1]))
        previous = options[choice, np.arange(size)] + scores
        back[frame] = choice
    state = size - 1 if previous[-1] >= previous[-2] else size - 2
    if not np.isfinite(previous[state]):
        raise ValueError("No complete CTC path")
    frames = [[] for _ in targets]
    for frame in range(len(emission) - 1, -1, -1):
        if state % 2:
            frames[state // 2].append(frame)
        state -= int(back[frame, state])
    if any(not part for part in frames):
        raise ValueError("Incomplete CTC alignment")
    return [sorted(part) for part in frames]


def align_emissions(text: str, vocabulary: dict, blank: int, log_probs, offset: float, frame_seconds: float) -> list[dict]:
    import numpy as np
    units = text_units(text, vocabulary)
    targets, spans = [], []
    delimiter = vocabulary.get("|")
    for unit in units:
        if targets and delimiter is not None:
            targets.append(delimiter)
        start = len(targets)
        targets.extend(unit["token_ids"] or [-1])
        spans.append((start, len(targets)))
    paths = ctc_path(log_probs, targets, blank)
    greedy = np.asarray(log_probs).argmax(axis=1)
    result = []
    for unit, (start, end) in zip(units, spans):
        row = {key: unit[key] for key in ("text", "char_start", "char_end")}
        row.update(start=None, end=None, score=None, character_hits=None, accepted=False)
        if unit["token_ids"]:
            frames = [f for part in paths[start:end] for f in part]
            scores = [float(log_probs[f, targets[t]]) for t in range(start, end) for f in paths[t]]
            hits = sum(any(greedy[f] == targets[t] for f in paths[t]) for t in range(start, end)) / (end - start)
            confidence = float(np.exp(np.mean(scores)))
            row.update(start=offset + min(frames) * frame_seconds,
                       end=offset + (max(frames) + 1) * frame_seconds,
                       score=confidence, character_hits=hits,
                       accepted=confidence >= .3 and hits >= .5)
        result.append(row)
    return result
