from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import as_file, files
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def skill_destination() -> Path:
    return Path.home() / ".agents" / "skills" / "pbx-call-transcriber"


def whisper_package() -> tuple[str, str]:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx_whisper", "mlx-whisper"
    return "faster_whisper", "faster-whisper"


def run_version(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (completed.stdout.strip() or completed.stderr.strip()).splitlines()
    return completed.returncode == 0, detail[0] if detail else f"exit {completed.returncode}"


def checks() -> list[Check]:
    codex = shutil.which("codex")
    if codex:
        codex_ok, codex_detail = run_version([codex, "--version"])
        auth_ok, auth_detail = run_version([codex, "login", "status"])
    else:
        codex_ok, codex_detail = False, "not found"
        auth_ok, auth_detail = False, "Codex is not installed"
    module, package = whisper_package()
    whisper_ok = importlib.util.find_spec(module) is not None
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_ok = Path(ffmpeg).is_file()
        ffmpeg_detail = ffmpeg
    except Exception as exc:
        ffmpeg_ok, ffmpeg_detail = False, str(exc)
    destination = skill_destination()
    return [
        Check("Python", sys.version_info >= (3, 11), platform.python_version()),
        Check("Codex CLI", codex_ok, codex_detail),
        Check("Codex login", auth_ok, auth_detail),
        Check("FFmpeg", ffmpeg_ok, ffmpeg_detail),
        Check("Whisper backend", whisper_ok, package if whisper_ok else f"missing: {package}"),
        Check("Transcription skill", (destination / "SKILL.md").is_file(), str(destination)),
    ]


def install_codex() -> None:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        raise RuntimeError(
            "Codex is missing and npm is unavailable. Install Node.js, then run "
            "`npm install -g @openai/codex`."
        )
    subprocess.run([npm, "install", "-g", "@openai/codex"], check=True)


def install_whisper() -> None:
    _, package = whisper_package()
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


def install_skill(force: bool = False) -> Path:
    destination = skill_destination()
    resource = files("callforge").joinpath("resources/pbx-call-transcriber")
    if destination.exists():
        if not force:
            if (destination / "SKILL.md").is_file():
                return destination
            raise RuntimeError(f"Skill destination already exists: {destination}")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = destination.with_name(destination.name + f".backup-{timestamp}")
        destination.rename(backup)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with as_file(resource) as source:
        shutil.copytree(source, destination)
    return destination


def setup(install_missing: bool, force_skill: bool = False) -> list[Check]:
    current = checks()
    if not install_missing:
        return current
    by_name = {item.name: item for item in current}
    if not by_name["Codex CLI"].ok:
        install_codex()
    if not by_name["Whisper backend"].ok:
        install_whisper()
    if not by_name["Transcription skill"].ok or force_skill:
        install_skill(force=force_skill)
    return checks()
