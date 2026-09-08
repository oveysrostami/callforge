"""Subprocess worker in the isolated speaker runtime. Audio never leaves the host."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
import sys
import wave
from pathlib import Path

from callforge.alignment import MODEL, REVISION, align_emissions, text_units


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    print("Loading local alignment dependencies", file=sys.stderr, flush=True)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # Transformers may otherwise start a non-daemon background conversion and
    # second weight download even when use_safetensors=False. It delays exit.
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    import numpy as np
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from huggingface_hub import try_to_load_from_cache
    started = time.monotonic()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    with wave.open(str(args.audio), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            raise ValueError("Expected mono 16kHz PCM16")
        waveform = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cached = isinstance(try_to_load_from_cache(MODEL, "pytorch_model.bin", revision=REVISION), str)
    print("Loading cached alignment model" if cached else "Downloading/loading alignment model (first use)",
          file=sys.stderr, flush=True)
    # Restricted weights-only loading, fixed revision, no remote Python code.
    processor = Wav2Vec2Processor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False,
                                          weights_only=True, use_safetensors=False).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    vocabulary = processor.tokenizer.get_vocab()
    model_ready = time.monotonic()
    print("Alignment model ready; matching existing words to audio", file=sys.stderr, flush=True)
    result = []
    for segment in data["segments"]:
        if not any(unit["token_ids"] for unit in text_units(segment["text"], vocabulary)):
            result.append({"id": segment["id"], "status": "unalignable", "reason": "no_supported_words", "words": []})
            continue
        a = max(0., float(segment["start"]) - .3)
        b = min(len(waveform) / 16000, float(segment["end"]) + .3)
        if b <= a or b - a > 30:
            result.append({"id": segment["id"], "status": "skipped", "reason": "invalid_or_long_segment", "words": []})
            continue
        inputs = processor(waveform[int(a * 16000):int(b * 16000)], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**{key: value.to(device) for key, value in inputs.items()}).logits[0]
            probabilities = logits.log_softmax(-1).cpu().numpy()
        try:
            words = align_emissions(segment["text"], vocabulary, model.config.pad_token_id,
                                    probabilities, a, (b - a) / len(probabilities))
            result.append({"id": segment["id"], "status": "completed", "words": words})
        except ValueError:
            result.append({"id": segment["id"], "status": "unalignable", "words": []})
        print(f"Aligned {segment['id']}", file=sys.stderr, flush=True)
    print(json.dumps({"model": MODEL, "revision": REVISION, "device": str(device),
                      "transformers_version": importlib.metadata.version("transformers"),
                      "torch_version": torch.__version__, "elapsed_seconds": time.monotonic() - started,
                      "model_was_cached": cached, "model_load_seconds": model_ready - started,
                      "alignment_seconds": time.monotonic() - model_ready,
                      "segments": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
