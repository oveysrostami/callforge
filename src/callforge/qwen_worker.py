"""Offline Qwen3-ASR worker. Unsupported backends are reported, never fatal to a benchmark."""
from __future__ import annotations

import argparse
import json
import importlib.metadata
import os
import platform
import time
import tempfile
import wave
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="fa")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.monotonic()
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
        if args.seed is not None:
            torch.manual_seed(args.seed)
        model_name = args.model if "/" in args.model else f"Qwen/{args.model}"
        temporary = None
        audio_input = args.audio
        with wave.open(str(args.audio), "rb") as handle:
            rate, channels, width, frames = (handle.getframerate(), handle.getnchannels(),
                                              handle.getsampwidth(), handle.getnframes())
            duration = frames / rate
            first = max(0, round(args.start * rate))
            last = min(frames, round(args.end * rate)) if args.end is not None else frames
            if first or last < frames:
                handle.setpos(first)
                payload = handle.readframes(max(0, last - first))
                temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temporary.close()
                with wave.open(temporary.name, "wb") as target:
                    target.setnchannels(channels); target.setsampwidth(width); target.setframerate(rate)
                    target.writeframes(payload)
                audio_input = Path(temporary.name)
        devices = (["mps", "cpu"] if platform.system() == "Darwin" and torch.backends.mps.is_available()
                   else ["cpu"])
        errors = []
        result = None
        device = devices[-1]
        for device in devices:
            try:
                dtype = torch.float16 if device == "mps" else torch.float32
                model = Qwen3ASRModel.from_pretrained(model_name, dtype=dtype, device_map=device)
                result = model.transcribe(audio=str(audio_input), language=args.language)
                break
            except Exception as backend_error:
                errors.append(f"{device}:{type(backend_error).__name__}")
                if "model" in locals():
                    del model
                if device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                    continue
                raise
        if result is None:
            raise RuntimeError("No compatible Qwen backend: " + ", ".join(errors))
        item = result[0] if isinstance(result, list) else result
        text = str(getattr(item, "text", item.get("text", "") if isinstance(item, dict) else item)).strip()
        end = min(duration, args.end) if args.end is not None else duration
        output = {"schema_version": 1, "status": "completed", "audio": str(args.audio), "provider": "qwen3-asr",
                  "model": model_name, "language": args.language, "start": args.start, "end": end,
                  "text": text, "segments": [{"start": args.start, "end": end, "text": text,
                                                 "words": [], "flags": ["requires_ctc_alignment"]}],
                  "speech_regions": [], "settings": {"prompt": "", "word_timestamps": False,
                                                         "offline": True, "seed": args.seed},
                  "metrics": {"segment_count": 1, "word_count": 0, "alignment_coverage": None},
                  "provenance": {"provider": "qwen3-asr", "model": model_name, "device": device,
                                   "backend_attempts": devices[:devices.index(device) + 1],
                                   "revision": getattr(getattr(model, "config", None), "_commit_hash", None),
                                   "offline": True},
                  "runtime_versions": {name: importlib.metadata.version(name)
                                       for name in ("qwen-asr", "transformers", "torch")},
                  "elapsed_seconds": time.monotonic() - started}
        if temporary is not None:
            Path(temporary.name).unlink(missing_ok=True)
    except Exception as exc:
        name = type(exc).__name__
        output = {"schema_version": 1, "status": "unsupported", "provider": "qwen3-asr", "model": args.model,
                  "error": f"{name}: {exc}", "segments": [], "text": "",
                  "elapsed_seconds": time.monotonic() - started}
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
