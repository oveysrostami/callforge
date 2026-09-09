"""Common local ASR provider interface and quality-aware candidate selection."""
from __future__ import annotations

import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path

from callforge.quality import prompt_leakage, text_flags


@dataclass(frozen=True)
class ASRSpec:
    provider: str
    model: str

    @property
    def heavy(self) -> bool:
        return self.model in {"large-v3", "Qwen3-ASR-1.7B"} or self.provider == "qwen3-asr"

    @property
    def family(self) -> str:
        return "whisper" if "whisper" in self.provider else self.provider


def parse_spec(value: str) -> ASRSpec:
    aliases = {
        "turbo": "mlx-whisper:large-v3-turbo",
        "turbo-q4": "mlx-whisper:large-v3-turbo-q4",
        "full": "mlx-whisper:large-v3",
    }
    value = aliases.get(value.strip(), value.strip())
    if ":" not in value:
        raise ValueError(f"ASR model must be provider:model, got {value!r}")
    provider, model = value.split(":", 1)
    if provider not in {"mlx-whisper", "faster-whisper", "qwen3-asr"} or not model:
        raise ValueError(f"Unsupported ASR provider: {provider}")
    return ASRSpec(provider, model)


def provider_command(skill_directory: Path, audio: Path, spec: ASRSpec, language: str,
                     prompt: str = "", start: float | None = None,
                     end: float | None = None, seed: int | None = None) -> list[str]:
    import sys
    if spec.provider == "qwen3-asr":
        from callforge.qwen_runtime import python
        command = [str(python()), str(Path(__file__).with_name("qwen_worker.py")), "--audio", str(audio),
                   "--model", spec.model, "--language", language]
    else:
        backend = "mlx" if spec.provider == "mlx-whisper" else "faster"
        # Explicit MLX configs remain portable: non-Apple installations use
        # faster-whisper while recording the actual backend in provenance.
        if backend == "mlx" and not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            backend = "faster"
        command = [sys.executable, str(skill_directory / "scripts" / "transcribe_audio.py"),
                   str(audio), "--backend", backend, "--model", spec.model,
                   "--language", language, "--prompt", prompt]
    if start is not None:
        command += ["--start", str(start)]
    if end is not None:
        command += ["--end", str(end)]
    if seed is not None:
        command += ["--seed", str(seed)]
    return command


def candidate_defects(result: dict) -> set[str]:
    defects = set()
    prompt = str((result.get("settings") or {}).get("prompt") or "")
    previous_end = -math.inf
    for segment in result.get("segments", []):
        flags = set(text_flags(str(segment.get("text", "")), segment))
        if prompt_leakage(str(segment.get("text", "")), prompt):
            flags.add("prompt_leakage")
        start, end = segment.get("start"), segment.get("end")
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start + end) or end <= start or start < previous_end):
            flags.add("invalid_timestamp")
        if isinstance(end, (int, float)):
            previous_end = end
        defects.update(flags & {"repetition", "compression", "invalid_asr_metrics",
                                "decoder_truncated", "prompt_leakage", "invalid_timestamp"})
    if not str(result.get("text", "")).strip():
        defects.add("empty")
    return defects


def candidate_score(result: dict) -> tuple[int, float, float]:
    """Lower is better; compare variants for the same acoustic window."""
    defects = candidate_defects(result)
    segments = result.get("segments", [])
    logprobs = [float(row["avg_logprob"]) for row in segments
                if isinstance(row.get("avg_logprob"), (int, float)) and math.isfinite(row["avg_logprob"])]
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else -10.0
    coverage = sum(max(0.0, float(row.get("end", 0)) - float(row.get("start", 0))) for row in segments)
    return len(defects), -mean_logprob, -coverage


def choose_candidate(candidates: list[tuple[str, dict]]) -> tuple[str, dict]:
    if not candidates:
        raise ValueError("No ASR candidates were supplied")
    return min(candidates, key=lambda item: candidate_score(item[1]))


def contains_sensitive_entity(text: str) -> bool:
    return bool(re.search(r"\d|تومان|ریال|درصد|شماره|کد|تاریخ", text))
