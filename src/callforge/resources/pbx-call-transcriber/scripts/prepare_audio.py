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


def read_pcm16(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        if audio.getsampwidth() != 2:
            raise RuntimeError("Decoded WAV is not PCM16")
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return (samples.reshape(-1, channels).astype(np.float32) / 32768.0), rate, channels


def write_pcm16(path: Path, samples: np.ndarray, rate: int) -> None:
    encoded = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(encoded.tobytes())


def independent_conversation_channels(samples: np.ndarray, rate: int) -> tuple[bool, dict]:
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < rate:
        return False, {"reason": "not_stereo_or_too_short"}
    window = max(1, round(rate * .25))
    levels = np.array([[rms(samples[offset:offset + window, channel]) for channel in range(2)]
                       for offset in range(0, len(samples), window)])
    threshold = max(.0025, float(np.percentile(levels, 60)) * .3)
    active = levels > threshold
    both = float(np.mean(active[:, 0] & active[:, 1]))
    exclusive = float(np.mean(active[:, 0] ^ active[:, 1]))
    energy = [rms(samples[:, channel]) for channel in range(2)]
    correlation = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]) if all(energy) else 1.0
    independent = all(value > .0005 for value in energy) and exclusive >= .15 and abs(correlation) < .45
    return independent, {"correlation": round(correlation, 4), "exclusive_activity": round(exclusive, 4),
                         "simultaneous_activity": round(both, 4), "channel_rms": [round(v, 6) for v in energy]}


def rms(samples: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(samples * samples)))) if samples.size else 0.0


def apply_agc(samples: np.ndarray, rate: int, target_rms: float, max_gain: float, gate_rms: float) -> np.ndarray:
    window = max(1, round(rate * 0.25))
    levels = [rms(samples[i : i + window]) for i in range(0, len(samples), window)]
    output = np.zeros_like(samples)
    for index, offset in enumerate(range(0, len(samples), window)):
        chunk = samples[offset : offset + window]
        nearby = levels[max(0, index - 1) : min(len(levels), index + 2)]
        level = max(nearby, default=0.0)
        gain = 1.0 if level <= gate_rms else min(max_gain, max(1.0, target_rms / level))
        # Linear peak-limited gain: gain=1 must be an exact no-op, including silence.
        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.
        if peak:
            gain = min(gain, max(1., .95 / peak))
        output[offset : offset + len(chunk)] = chunk * gain
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-rms", type=float, default=0.06)
    parser.add_argument("--max-gain", type=float, default=18.0)
    parser.add_argument("--gate-rms", type=float, default=0.0025)
    parser.add_argument("--denoise", action="store_true", help="Benchmark-only FFmpeg spectral denoise variant")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded_path = args.output_dir / f"{source.stem}.16k-preserved.wav"
    raw_path = args.output_dir / f"{source.stem}.16k-mono.wav"
    agc_path = args.output_dir / f"{source.stem}.16k-mono-agc.wav"
    decoder = decode(source, decoded_path)
    multichannel, rate, channels = read_pcm16(decoded_path)
    samples = np.mean(multichannel, axis=1)
    write_pcm16(raw_path, samples, rate)
    channel_paths = []
    for channel in range(channels):
        channel_path = args.output_dir / f"{source.stem}.16k-channel-{channel + 1}.wav"
        write_pcm16(channel_path, multichannel[:, channel], rate)
        channel_paths.append(str(channel_path))
    independent, channel_metrics = independent_conversation_channels(multichannel, rate)
    enhancement_recommended = rms(samples) < args.target_rms * .75
    enhanced = (apply_agc(samples, rate, args.target_rms, args.max_gain, args.gate_rms)
                if enhancement_recommended else samples.copy())
    write_pcm16(agc_path, enhanced, rate)
    denoise_path = None
    if args.denoise:
        executable = ffmpeg_executable()
        if not executable:
            raise RuntimeError("FFmpeg is required for the benchmark denoise variant")
        denoise_path = args.output_dir / f"{source.stem}.16k-mono-denoise.wav"
        completed = subprocess.run([executable, "-nostdin", "-y", "-i", str(raw_path),
                                    "-af", "afftdn=nf=-25", str(denoise_path)],
                                   capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"FFmpeg denoise failed: {completed.stderr[-1000:]}")
    result = {
        "source": str(source),
        "decoder": decoder,
        "raw_wav": str(raw_path),
        "agc_wav": str(agc_path),
        "sample_rate": rate,
        "channels": channels,
        "channel_wavs": channel_paths,
        "independent_conversation_channels": independent,
        "channel_analysis": channel_metrics,
        "duration_seconds": round(len(samples) / rate, 6),
        "peak": round(float(np.max(np.abs(samples))) if samples.size else 0.0, 6),
        "rms": round(rms(samples), 6),
        "clipping_fraction": float(np.mean(np.abs(samples) >= .999)) if samples.size else 0.,
        "enhancement_recommended": enhancement_recommended,
        "enhancement_applied": bool(np.any(samples != enhanced)),
        "enhancement": "linear_peak_limited_agc",
        "agc_rms": round(rms(enhanced), 6),
        "likely_empty": len(samples) / rate < 1.0 or rms(samples) < 0.0001,
    }
    if denoise_path is not None:
        result["denoise_wav"] = str(denoise_path)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
