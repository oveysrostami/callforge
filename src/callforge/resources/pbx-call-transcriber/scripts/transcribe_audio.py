#!/usr/bin/env python3
"""Run the best local Whisper backend and emit review-friendly JSON."""

from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import wave
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np


MLX_MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo-q4": "mlx-community/whisper-large-v3-turbo-q4",
    "large-v3-turbo-q4": "mlx-community/whisper-large-v3-turbo-q4",
    "full": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}
FASTER_MODELS = {
    "turbo": "large-v3-turbo", "large-v3-turbo": "large-v3-turbo",
    "turbo-q4": "large-v3-turbo", "large-v3-turbo-q4": "large-v3-turbo",
    "full": "large-v3", "large-v3": "large-v3",
}


def finite_json(value):
    # Keep the standalone skill helper independent of the CallForge package.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return value


def sanitize_segments(segments):
    for row in segments:
        invalid = [key for key in ("avg_logprob", "no_speech_prob", "compression_ratio")
                   if isinstance(row.get(key), (int, float)) and not math.isfinite(row[key])]
        if invalid:
            row["flags"] = sorted(set(row.get("flags", []) + ["invalid_asr_metrics"]))
            row["invalid_metrics"] = invalid
    return finite_json(segments)


@contextmanager
def guarded_mlx_decoder():
    """End an impossible token path instead of computing -inf - (-inf).

    Scoped to this helper's transcription; no installed library files change.
    A forced end marks the result as truncated, never as a confident decode.
    """
    import mlx.core as mx
    import mlx_whisper.decoding as decoding
    original = decoding.GreedyDecoder
    warnings = []
    class GuardedDecoder(original):
        def update(self, tokens, logits, sum_logprobs):
            if bool(mx.any(mx.isnan(logits) | (logits == mx.inf))):
                raise FloatingPointError("Whisper produced invalid decoder logits")
            finished = tokens[:, -1] == self.eot
            dead = mx.all(logits == -mx.inf, axis=-1)
            if bool(mx.any(dead & ~finished)):
                warnings.append("decoder_truncated")
            # Finished hypotheses must not evaluate -inf * 0 either.
            forced = mx.full_like(logits, -mx.inf)
            forced[:, self.eot] = 0.
            logits = mx.where((dead | finished)[:, None], forced, logits)
            # Mark only this failed hypothesis as unusable. Whisper's existing
            # bounded temperature fallback now sees a low score and can retry;
            # NaN previously made its threshold comparison silently false.
            sum_logprobs = mx.where(dead & ~finished, -mx.inf, sum_logprobs)
            return super().update(tokens, logits, sum_logprobs)
    decoding.GreedyDecoder = GuardedDecoder
    try:
        yield warnings
    finally:
        decoding.GreedyDecoder = original


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


def compatible_mlx_model(repo_or_path: str, stack: ExitStack) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    candidate = Path(repo_or_path).expanduser()
    model_path = candidate.resolve() if candidate.exists() else Path(snapshot_download(repo_id=repo_or_path))
    expected = model_path / "weights.safetensors"
    alternate = model_path / "model.safetensors"
    if expected.exists() or not alternate.exists():
        return str(model_path), str(model_path)
    alias = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="mlx-whisper-model-")))
    for item in model_path.iterdir():
        target_name = "weights.safetensors" if item.name == "model.safetensors" else item.name
        os.symlink(item, alias / target_name)
    return str(alias), str(model_path)


def mlx_transcribe(samples: np.ndarray, args) -> tuple[str, list[dict], str]:
    import mlx_whisper
    if getattr(args, "seed", None) is not None:
        import mlx.core as mx
        mx.random.seed(args.seed)

    requested = MLX_MODELS.get(args.model, args.model)
    with ExitStack() as stack, guarded_mlx_decoder() as decoder_warnings:
        model, args.resolved_model_path = compatible_mlx_model(requested, stack)
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=model,
            language=args.language,
            task="transcribe",
            verbose=False,
            temperature=tuple(getattr(args, "temperatures", None) or (0.0, 0.2)),
            condition_on_previous_text=False,
            initial_prompt=args.prompt,
            word_timestamps=True,
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
            "compression_ratio": item.get("compression_ratio"),
            "temperature": item.get("temperature"),
            "words": [dict(word, start=round(args.start + word["start"], 3),
                           end=round(args.start + word["end"], 3)) for word in item.get("words", [])],
        }
        for item in result.get("segments", [])
    ]
    if decoder_warnings:
        for row in segments:
            if not math.isfinite(row["avg_logprob"]):
                row["flags"] = ["decoder_truncated"]
    return result.get("text", "").strip(), sanitize_segments(segments), requested


def faster_transcribe(samples: np.ndarray, args) -> tuple[str, list[dict], str]:
    from faster_whisper import WhisperModel

    requested = FASTER_MODELS.get(args.model, args.model)
    model = WhisperModel(requested, device="auto", compute_type="default")
    generated, _ = model.transcribe(
        samples,
        language=args.language,
        task="transcribe",
        beam_size=5,
        temperature=getattr(args, "temperatures", None) or [0.0, 0.2],
        initial_prompt=args.prompt,
        condition_on_previous_text=False,
        # VAD is retained as independent coverage evidence on both backends.
        # Do not discard quiet telephone speech solely on a detector decision.
        vad_filter=False,
        word_timestamps=True,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.2,
        hallucination_silence_threshold=1.2,
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
                "compression_ratio": item.compression_ratio,
                "temperature": item.temperature,
                "words": [{"start": round(args.start + word.start, 3),
                           "end": round(args.start + word.end, 3), "word": word.word,
                           "probability": word.probability} for word in item.words or []],
            }
        )
    return " ".join(texts).strip(), sanitize_segments(segments), requested


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--backend", choices=("auto", "mlx", "faster"), default="auto")
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--language", default="fa")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--windows-file", type=Path,
                        help="JSON array of exact {start,end} speech windows; keeps one model process alive")
    parser.add_argument("--temperature", dest="temperatures", type=float, action="append",
                        help="Ordered fallback temperatures; repeat to override the default 0, 0.2")
    parser.add_argument("--seed", type=int, help="MLX sampling seed for controlled experiments")
    args = parser.parse_args()
    if args.temperatures and any(not math.isfinite(t) or not 0 <= t <= 1 for t in args.temperatures):
        parser.error("temperatures must be finite values between 0 and 1")
    audio = args.audio.expanduser().resolve()
    backend = choose_backend(args.backend)
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    if args.windows_file:
        try:
            requested_windows = json.loads(args.windows_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read windows file: {exc}")
        if not isinstance(requested_windows, list) or not requested_windows:
            parser.error("windows file must contain a non-empty JSON array")
    else:
        requested_windows = [{"start": args.start, "end": args.end}]

    all_segments = []
    all_speech = []
    texts = []
    decoded_windows = []
    model = None
    for index, window in enumerate(requested_windows):
        try:
            start = float(window["start"])
            end = None if window.get("end") is None else float(window["end"])
        except (KeyError, TypeError, ValueError):
            parser.error(f"invalid window at index {index}")
        if not math.isfinite(start) or start < 0 or (end is not None and
                (not math.isfinite(end) or end <= start)):
            parser.error(f"invalid window bounds at index {index}")
        args.start, args.end = start, end
        samples = read_clip(audio, start, end)
        actual_end = start + len(samples) / 16_000
        speech = get_speech_timestamps(
            samples,
            vad_options=VadOptions(threshold=.35, min_silence_duration_ms=500,
                                   speech_pad_ms=400),
        )
        if backend == "mlx":
            text, segments, model = mlx_transcribe(samples, args)
        else:
            text, segments, model = faster_transcribe(samples, args)
        texts.append(text)
        all_segments.extend(segments)
        all_speech.extend({"start": round(start + region["start"] / 16000, 3),
                           "end": round(start + region["end"] / 16000, 3)}
                          for region in speech)
        decoded_windows.append({"start": round(start, 3), "end": round(actual_end, 3),
                                "segment_count": len(segments)})

    all_segments.sort(key=lambda row: (row.get("start", 0), row.get("end", 0)))
    all_speech.sort(key=lambda row: (row["start"], row["end"]))
    text = " ".join(value for value in texts if value).strip()
    output_start = decoded_windows[0]["start"]
    output_end = decoded_windows[-1]["end"]
    output = {
        "schema_version": 1,
        "audio": str(audio),
        "backend": backend,
        "provider": f"{backend}-whisper" if backend != "faster" else "faster-whisper",
        "model": model,
        "resolved_model_path": getattr(args, "resolved_model_path", None),
        "language": args.language,
        "start": output_start,
        "end": output_end,
        "text": text,
        "segments": all_segments,
        "speech_regions": all_speech,
        "settings": {"temperature": args.temperatures or [0, .2], "seed": args.seed, "word_timestamps": True,
                     "condition_on_previous_text": False,
                     "vad_mode": "tight_windows" if args.windows_file else "coverage_only",
                     "decoder_numerics_guard": backend == "mlx",
                     "vad_threshold": .35, "prompt": args.prompt},
        "provenance": {"provider": f"{backend}-whisper" if backend != "faster" else "faster-whisper",
                       "model": model, "backend": backend, "offline": True,
                       "decode_windows": decoded_windows},
        "metrics": {"segment_count": len(all_segments),
                    "word_count": sum(len(row.get("words", [])) for row in all_segments),
                    "window_count": len(decoded_windows)},
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "faster-whisper", "onnxruntime") + (("mlx-whisper",) if backend == "mlx" else ())},
    }
    print(json.dumps(finite_json(output), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
