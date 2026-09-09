"""Isolated Python 3.12 runtime for optional Qwen3-ASR candidates."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import venv
from pathlib import Path

from callforge.registry import callforge_home


MODELS = ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B")


def paths(config=None) -> tuple[Path, Path]:
    base = callforge_home()
    models = config.models if config is not None else base / "models"
    return base / "qwen3-asr-python312", models


def python(config=None) -> Path:
    runtime, _ = paths(config)
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def environment(config=None, *, offline: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(paths(config)[1])
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def install(config=None) -> None:
    runtime, models = paths(config)
    models.mkdir(parents=True, exist_ok=True)
    py312 = shutil.which("python3.12")
    if not py312 and os.name != "nt" and shutil.which("brew"):
        subprocess.run([shutil.which("brew"), "install", "python@3.12"], check=True)
        py312 = shutil.which("python3.12")
        if not py312:
            candidate = Path(shutil.which("brew")).resolve().parent / "python3.12"
            py312 = str(candidate) if candidate.is_file() else None
    if not py312:
        raise RuntimeError("Qwen3-ASR setup requires python3.12 on PATH (Homebrew installs it automatically on macOS)")
    if not python(config).is_file():
        subprocess.run([py312, "-m", "venv", str(runtime)], check=True)
    subprocess.run([str(python(config)), "-m", "pip", "install", "qwen-asr", "transformers", "accelerate"], check=True)
    ready = runtime / "ready.json"
    ready.write_text(json.dumps({"schema": 1, "python": "3.12", "models": list(MODELS)}), encoding="utf-8")


def ready(config=None) -> tuple[bool, str]:
    runtime, _ = paths(config)
    marker = runtime / "ready.json"
    return (python(config).is_file() and marker.is_file(), str(marker))
