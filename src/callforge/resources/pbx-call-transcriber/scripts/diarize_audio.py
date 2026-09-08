#!/usr/bin/env python3
"""Opt-in local community-1 diarization. Never uses the hosted precision service."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import wave
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    import numpy as np
    import torch
    from pyannote.audio import Pipeline

    with wave.open(str(args.audio), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16")
        rate = handle.getframerate()
        waveform = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=os.environ.get("HF_TOKEN"))
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    output = pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": rate})
    annotation = output.speaker_diarization
    if hasattr(annotation, "itertracks"):
        turns = [{"start": turn.start, "end": turn.end, "speaker_id": label}
                 for turn, _, label in annotation.itertracks(yield_label=True)]
    else:
        turns = [{"start": turn.start, "end": turn.end, "speaker_id": label} for turn, label in annotation]
    print(json.dumps({"model": "pyannote/speaker-diarization-community-1", "version": importlib.metadata.version("pyannote.audio"), "turns": turns}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
