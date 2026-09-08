"""Offline attention/DTW alignment using already-cached MLX Whisper, not decoding."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
import wave
from contextlib import ExitStack
from pathlib import Path

MODEL = "mlx-community/whisper-large-v3-turbo-q4"


def literal_words(text, timings, offset):
    """Map tokenizer pieces to exact literal spans, never replace text with pieces."""
    pieces, cursor = [], -1  # The teacher-forced input has one leading space.
    for item in timings:
        pieces.append((cursor, cursor + len(item.word), item))
        cursor += len(item.word)
    if "".join(item.word for item in timings) != " " + text:
        raise ValueError("Attention alignment changed supplied text")
    result = []
    for match in re.finditer(r"\S+", text):
        parts = [item for a, b, item in pieces if a < match.end() and b > match.start()]
        a, b = min(float(p.start) for p in parts), max(float(p.end) for p in parts)
        probability = min(float(p.probability) for p in parts)
        result.append({"text": match.group(), "char_start": match.start(), "char_end": match.end(),
                       "start": offset + a, "end": offset + b, "score": probability,
                       "method": "whisper_attention", "character_hits": None,
                       "accepted": probability >= .3 and b > a})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import mlx.core as mx
    import numpy as np
    from huggingface_hub import snapshot_download
    from mlx_whisper.load_models import load_model
    from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
    from mlx_whisper.tokenizer import get_tokenizer
    from mlx_whisper.timing import find_alignment
    started = time.monotonic()
    snapshot = Path(snapshot_download(MODEL, local_files_only=True))
    helper_path = Path(__file__).parent / "resources/pbx-call-transcriber/scripts/transcribe_audio.py"
    spec = importlib.util.spec_from_file_location("callforge_audio_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    with wave.open(str(args.audio), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            raise ValueError("Expected mono 16kHz PCM16")
        waveform = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.
    with ExitStack() as stack:
        path, _ = helper.compatible_mlx_model(str(snapshot), stack)
        model = load_model(path, dtype=mx.float16)
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="fa", task="transcribe")
        ready = time.monotonic()
        print("Cached Whisper loaded; aligning fixed text without decoding", file=sys.stderr, flush=True)
        result = []
        for row in data["segments"]:
            a, b = max(0., row["start"] - .3), min(len(waveform) / 16000, row["end"] + .3)
            if "[" in row["text"] or any(c.isdigit() for c in row["text"]) or b - a > 29 or b <= a:
                result.append({"id": row["id"], "status": "skipped", "reason": "unclear_numeric_or_long_text", "words": []})
                continue
            clip = waveform[int(a * 16000):int(b * 16000)]
            mel = log_mel_spectrogram(clip, n_mels=model.dims.n_mels)
            frames = mel.shape[0]
            mel = pad_or_trim(mel, N_FRAMES, axis=0).astype(mx.float16)
            alignment = find_alignment(model, tokenizer, tokenizer.encode(" " + row["text"]), mel, frames)
            words = literal_words(row["text"], alignment, a)
            result.append({"id": row["id"], "status": "completed", "words": words})
            print(f"Aligned {row['id']}", file=sys.stderr, flush=True)
    print(json.dumps({"model": MODEL, "revision": snapshot.name, "backend": "mlx",
                      "independent_of_whisper": False, "model_was_cached": True,
                      "mlx_whisper_version": importlib.metadata.version("mlx-whisper"),
                      "model_load_seconds": ready - started, "alignment_seconds": time.monotonic() - ready,
                      "elapsed_seconds": time.monotonic() - started, "segments": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
