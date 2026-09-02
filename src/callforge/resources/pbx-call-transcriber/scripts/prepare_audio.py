#!/usr/bin/env python3
"""Decode a call recording, measure it, and make a conservative AGC copy."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


TARGET_RATE = 16_000


def ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def decode(source: Path, destination: Path) -> str:
    ffmpeg = ffmpeg_executable()
    if ffmpeg:
        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(TARGET_RATE),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        decoder = "ffmpeg"
    elif shutil.which("afconvert"):
        command = [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            f"LEI16@{TARGET_RATE}",
            "-c",
            "1",
            str(source),
            str(destination),
        ]
        decoder = "afconvert"
    else:
        raise RuntimeError("No FFmpeg or afconvert decoder is available")
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{decoder} failed: {detail}")
    return decoder


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise RuntimeError("Decoded WAV is not mono PCM16")
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0, rate


def write_pcm16(path: Path, samples: np.ndarray, rate: int) -> None:
    encoded = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(encoded.tobytes())


def rms(samples: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(samples * samples)))) if samples.size else 0.0


def apply_agc(samples: np.ndarray, rate: int, target_rms: float, max_gain: float, gate_rms: float) -> np.ndarray:
    window = max(1, round(rate * 0.25))
    levels = [rms(samples[i : i + window]) for i in range(0, len(samples), window)]
    output = np.zeros_like(samples)
    softness = 1.4
    normalizer = math.tanh(softness)
    for index, offset in enumerate(range(0, len(samples), window)):
        chunk = samples[offset : offset + window]
        nearby = levels[max(0, index - 1) : min(len(levels), index + 2)]
        level = max(nearby, default=0.0)
        gain = 1.0 if level <= gate_rms else min(max_gain, max(1.0, target_rms / level))
        output[offset : offset + len(chunk)] = np.tanh(chunk * gain * softness) / normalizer
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-rms", type=float, default=0.06)
    parser.add_argument("--max-gain", type=float, default=18.0)
    parser.add_argument("--gate-rms", type=float, default=0.0025)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{source.stem}.16k-mono.wav"
    agc_path = args.output_dir / f"{source.stem}.16k-mono-agc.wav"
    decoder = decode(source, raw_path)
    samples, rate = read_pcm16(raw_path)
    enhanced = apply_agc(samples, rate, args.target_rms, args.max_gain, args.gate_rms)
    write_pcm16(agc_path, enhanced, rate)
    result = {
        "source": str(source),
        "decoder": decoder,
        "raw_wav": str(raw_path),
        "agc_wav": str(agc_path),
        "sample_rate": rate,
        "channels": 1,
        "duration_seconds": round(len(samples) / rate, 6),
        "peak": round(float(np.max(np.abs(samples))) if samples.size else 0.0, 6),
        "rms": round(rms(samples), 6),
        "likely_empty": len(samples) / rate < 1.0 or rms(samples) < 0.0001,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

