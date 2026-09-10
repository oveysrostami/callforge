"""Shared speaker models/runtime, usable before an audio workspace is selected."""
from __future__ import annotations

import json
import os
from pathlib import Path

from callforge.registry import callforge_home, get_active_root


def paths(config=None) -> tuple[Path, Path]:
    registration = callforge_home() / "speaker-runtime.json"
    if registration.is_file():
        value = json.loads(registration.read_text(encoding="utf-8"))
        return Path(value["runtime"]), Path(value["models"])
    # Reuse an existing installation and its accepted HF credentials/cache.
    workspace = config.workspace if config is not None else None
    if workspace is None:
        try:
            workspace = get_active_root() / ".callforge"
        except RuntimeError:
            pass
    if workspace is not None and (workspace / "diarization-runtime").is_dir():
        return workspace / "diarization-runtime", workspace / "models"
    return callforge_home() / "speaker-runtime", callforge_home() / "models"


def python(config=None) -> Path:
    runtime, _ = paths(config)
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def register(config=None) -> None:
    runtime, models = paths(config)
    from callforge.quality import write_json
    target = callforge_home() / "speaker-runtime.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"speaker-runtime.{os.getpid()}.tmp")
    write_json(temporary, {"runtime": str(runtime), "models": str(models)})
    temporary.replace(target)


def environment(config=None, *, offline: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(paths(config)[1])
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    env["PYANNOTE_METRICS_ENABLED"] = "0"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["ORT_DISABLE_TELEMETRY"] = "1"
    env["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def install(config=None) -> None:
    import subprocess
    import venv
    runtime, models = paths(config)
    models.mkdir(parents=True, exist_ok=True)
    if not python(config).is_file():
        venv.EnvBuilder(with_pip=True).create(runtime)
    subprocess.run([str(python(config)), "-m", "pip", "install",
                    "pyannote.audio==4.0.7", "transformers==5.16.1"], check=True)
    register(config)
