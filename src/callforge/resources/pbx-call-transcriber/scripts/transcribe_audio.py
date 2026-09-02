#!/usr/bin/env python3
"""Run the best local Whisper backend and emit review-friendly JSON."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import tempfile
import wave
from contextlib import ExitStack
from pathlib import Path

import numpy as np


MLX_MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo-q4",
    "full": "mlx-community/whisper-large-v3-mlx",
}
FASTER_MODELS = {"turbo": "large-v3-turbo", "full": "large-v3"}


def read_clip(path: Path, start: float, end: float | None) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise RuntimeError("Input must be mono PCM16 WAV")
        rate = audio.getframerate()
        first = max(0, round(start * rate))
        total = audio.getnframes()
        last = total if end is None else min(total, round(end * rate))
        if last <= first:
            raise RuntimeError("The selected time interval is empty")
        audio.setpos(first)
        data = audio.readframes(last - first)
    if rate != 16_000:
        raise RuntimeError(f"Expected 16000 Hz WAV, got {rate} Hz")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx_whisper"):
        return "mlx"
    if importlib.util.find_spec("faster_whisper"):
        return "faster"
    raise RuntimeError("No supported Whisper backend is installed")


def compatible_mlx_model(repo_or_path: str, stack: ExitStack) -> str:
    from huggingface_hub import snapshot_download

    candidate = Path(repo_or_path).expanduser()
    model_path = candidate.resolve() if candidate.exists() else Path(snapshot_download(repo_id=repo_or_path))
    expected = model_path / "weights.safetensors"
    alternate = model_path / "model.safetensors"
    if expected.exists() or not alternate.exists():
        return str(model_path)
    alias = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="mlx-whisper-model-")))
    for item in model_path.iterdir():
        target_name = "weights.safetensors" if item.name == "model.safetensors" else item.name
        os.symlink(item, alias / target_name)
    return str(alias)


def mlx_transcribe(samples: np.ndarray, args) -> tuple[str, list[dict], str]:
    import mlx_whisper

    requested = MLX_MODELS.get(args.model, args.model)
    with ExitStack() as stack:
        model = compatible_mlx_model(requested, stack)
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=model,
            language=args.language,
            task="transcribe",
            verbose=False,
            temperature=0,
            condition_on_previous_text=False,
            initial_prompt=args.prompt,
            word_timestamps=False,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.2,
            hallucination_silence_threshold=1.2,
        )
    segments = [
        {
            "start": round(args.start + float(item["start"]), 3),
            "end": round(args.start + float(item["end"]), 3),
            "text": item["text"].strip(),
            "avg_logprob": round(float(item.get("avg_logprob", 0.0)), 4),
            "no_speech_prob": round(float(item.get("no_speech_prob", 0.0)), 4),
        }
        for item in result.get("segments", [])
    ]
    return result.get("text", "").strip(), segments, requested


def faster_transcribe(samples: np.ndarray, args) -> tuple[str, list[dict], str]:
    from faster_whisper import WhisperModel

    requested = FASTER_MODELS.get(args.model, args.model)
    model = WhisperModel(requested, device="auto", compute_type="default")
    generated, _ = model.transcribe(
        samples,
        language=args.language,
        task="transcribe",
        beam_size=5,
        temperature=0,
        initial_prompt=args.prompt,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    segments = []
    texts = []
    for item in generated:
        text = item.text.strip()
        texts.append(text)
        segments.append(
            {
                "start": round(args.start + float(item.start), 3),
                "end": round(args.start + float(item.end), 3),
                "text": text,
                "avg_logprob": round(float(item.avg_logprob), 4),
                "no_speech_prob": round(float(item.no_speech_prob), 4),
            }
        )
    return " ".join(texts).strip(), segments, requested


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--backend", choices=("auto", "mlx", "faster"), default="auto")
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--language", default="fa")
    parser.add_argument("--prompt", default="این یک مکالمه تلفنی فارسی میان کارشناس پشتیبانی و مشتری است.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    audio = args.audio.expanduser().resolve()
    samples = read_clip(audio, args.start, args.end)
    backend = choose_backend(args.backend)
    if backend == "mlx":
        text, segments, model = mlx_transcribe(samples, args)
    else:
        text, segments, model = faster_transcribe(samples, args)
    output = {
        "audio": str(audio),
        "backend": backend,
        "model": model,
        "language": args.language,
        "start": args.start,
        "end": args.start + len(samples) / 16_000,
        "text": text,
        "segments": segments,
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
