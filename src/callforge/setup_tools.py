from __future__ import annotations

import importlib.util
import importlib.metadata
import platform
import shutil
import subprocess
import sys
import os
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import as_file, files
from pathlib import Path

TESTED_ASR_VERSIONS = {"mlx-whisper": "0.4.3", "faster-whisper": "1.2.1"}


def tested_package_installed(package: str) -> bool:
    try:
        return importlib.metadata.version(package) == TESTED_ASR_VERSIONS[package]
    except importlib.metadata.PackageNotFoundError:
        return False


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
    whisper_ok = importlib.util.find_spec(module) is not None and tested_package_installed(package)
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
        Check("Speech detector", tested_package_installed("faster-whisper"),
              "faster-whisper / Silero ONNX (all platforms)"),
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
    packages = sorted({package, "faster-whisper"})
    subprocess.run([sys.executable, "-m", "pip", "install",
                    *(f"{name}=={TESTED_ASR_VERSIONS[name]}" for name in packages)], check=True)


def install_skill(force: bool = False) -> Path:
    destination = skill_destination()
    resource = files("callforge").joinpath("resources/pbx-call-transcriber")
    if destination.exists():
        if not force:
            if (destination / "SKILL.md").is_file():
                return destination
            raise RuntimeError(f"Skill destination already exists: {destination}")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = destination.parent.parent / "skill-backups" / (destination.name + f".backup-{timestamp}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(backup)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with as_file(resource) as source:
        shutil.copytree(source, destination)
    return destination


def speaker_check() -> Check:
    from callforge.speaker_runtime import paths, python
    ready = paths()[0] / "ready.json"
    try:
        value = json.loads(ready.read_text(encoding="utf-8"))
        ok = value.get("schema") == 1 and value.get("models") == str(paths()[1]) and python().is_file()
    except (OSError, ValueError):
        ok = False
    return Check("Speaker / role pipeline", ok, "models verified by setup" if ok else "run callforge setup --yes")


def quality_checks() -> list[Check]:
    try:
        from callforge.registry import get_active_root
        from callforge.config import AppConfig
        config = AppConfig.for_root(get_active_root())
        hub = config.models / "hub"
    except RuntimeError:
        config = None
        hub = Path.home() / ".cache" / "huggingface" / "hub"
    from callforge.qwen_runtime import ready
    qwen_ok, qwen_detail = ready(config)
    turbo = any(hub.glob("models--mlx-community--whisper-large-v3-turbo/snapshots/*")) or any(
        hub.glob("models--Systran--faster-whisper-large-v3-turbo/snapshots/*"))
    full = any(hub.glob("models--mlx-community--whisper-large-v3-mlx/snapshots/*")) or any(
        hub.glob("models--Systran--faster-whisper-large-v3/snapshots/*"))
    qwen_models = all(any(hub.glob(f"models--Qwen--{name}/snapshots/*"))
                      for name in ("Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B"))
    try:
        import torch
        mps = bool(torch.backends.mps.is_available())
    except Exception:
        mps = False
    return [
        Check("ASR primary cache", turbo, "non-Q4 large-v3-turbo" if turbo else "run setup --quality-models"),
        Check("ASR fallback cache", full, "full large-v3" if full else "run setup --quality-models"),
        Check("Qwen Python 3.12 runtime", qwen_ok, qwen_detail),
        Check("Qwen candidate cache", qwen_models, "0.6B + 1.7B" if qwen_models else "optional; run setup --quality-models"),
        Check("MPS", mps, "available" if mps else "unavailable; Qwen falls back to CPU"),
    ]


def hf_instructions() -> None:
    print("Hugging Face setup (audio stays local):", flush=True)
    print("1. Sign in and personally accept the model conditions: https://huggingface.co/pyannote/speaker-diarization-community-1", flush=True)
    print("2. Create a Read token (or fine-grained token with read access to public gated models): https://huggingface.co/settings/tokens", flush=True)
    print("3. Paste it only into the hidden terminal prompt, never chat, CLI arguments or project config.", flush=True)
    print("--yes installs software; it does NOT accept model conditions on your behalf.", flush=True)


def setup_models() -> None:
    from callforge.speaker_runtime import environment, install, paths, python
    from callforge.quality import write_json
    from callforge.registry import get_active_root
    from callforge.config import AppConfig
    install()
    env = environment()
    ready = paths()[0] / "ready.json"
    ready.unlink(missing_ok=True)
    command = [str(python()), "-m", "callforge.setup_models", "access"]
    def access():
        try:
            return subprocess.run(command, env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=90).returncode
        except subprocess.TimeoutExpired:
            return 5
    result = access()
    if result:
        hf_instructions()
        if not sys.stdin.isatty():
            raise RuntimeError("Hugging Face setup is incomplete. Run `callforge setup` in your interactive terminal.")
        if result in {2, 3}:
            input("After accepting the model conditions and creating your token, press Enter: ")
            login = subprocess.run([str(python()), "-m", "callforge.hf_auth"], env=env, check=False)
            if login.returncode:
                raise RuntimeError("Hugging Face login incomplete. Rerun callforge setup.")
        elif result == 4:
            input("This account lacks model access. Accept the conditions with the SAME account, then press Enter: ")
        else:
            raise RuntimeError("Cannot verify Hugging Face access. Check your network and rerun setup; credentials were not changed.")
        result = access()
    if result:
        raise RuntimeError("Model access is still unavailable. Check the account, token read permissions and model acceptance, then rerun setup.")
    print("Hugging Face model access verified; existing credentials are reused.", flush=True)
    subprocess.run([str(python()), "-m", "callforge.setup_models", "models"], env=env, check=True, timeout=3600)
    # Whisper keeps its previous workspace cache, if one already exists.
    whisper_env = dict(env)
    try:
        config = AppConfig.for_root(get_active_root())
        whisper_env = config.runtime_environment()
        whisper_env.pop("HF_HUB_OFFLINE", None)
        whisper_env.pop("TRANSFORMERS_OFFLINE", None)
        whisper_env["PYTHONPATH"] = env["PYTHONPATH"]
    except RuntimeError:
        config = None
    subprocess.run([sys.executable, "-m", "callforge.setup_models", "whisper"], env=whisper_env, check=True, timeout=3600)
    # Verify cached speaker loading offline too: calls must never download models.
    subprocess.run([str(python()), "-m", "callforge.setup_models", "models"],
                   env=environment(offline=True), check=True, timeout=180)
    write_json(ready, {"schema": 1, "models": str(paths()[1]), "verified_at": datetime.now(UTC).isoformat()})
    if config is not None:
        enable_speakers(config)


def enable_speakers(config) -> None:
    import re
    path = config.workspace / "config.toml"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    # Limit the edit to the callforge table; preserve unrelated settings/comments.
    pattern = r"(?ms)(^\[callforge\][^\n]*\n)(.*?)(?=^\[|\Z)"
    def update(match):
        body = match[2]
        if re.search(r"(?m)^diarization\s*=", body):
            body = re.sub(r"(?m)^(diarization\s*=\s*)(?:true|false)", r"\g<1>true", body)
        else:
            body = "diarization = true\n" + body
        return match[1] + body
    updated = re.sub(pattern, update, text)
    if updated != text:
        temporary = path.with_name(f"config.setup-{os.getpid()}.tmp")
        temporary.write_text(updated, encoding="utf-8")
        temporary.replace(path)
    print("Speaker/role pipeline enabled for the active workspace (run and UI). Restart an already-running UI to load settings.", flush=True)


def setup(install_missing: bool, force_skill: bool = False, diarization: bool = True,
          quality_models: bool = False) -> list[Check]:
    current = checks()
    if not install_missing:
        return current + [speaker_check()] + quality_checks()
    by_name = {item.name: item for item in current}
    if not by_name["Codex CLI"].ok:
        install_codex()
    if not by_name["Whisper backend"].ok or not by_name["Speech detector"].ok:
        install_whisper()
    if not by_name["Transcription skill"].ok or force_skill:
        install_skill(force=force_skill)
    if not checks()[2].ok:
        if not sys.stdin.isatty():
            raise RuntimeError("Codex login required. Run callforge setup in an interactive terminal.")
        subprocess.run([shutil.which("codex"), "login"], check=True)
    setup_models()
    if quality_models:
        from callforge.qwen_runtime import install as install_qwen, environment as qwen_environment
        from callforge.config import AppConfig
        from callforge.registry import get_active_root
        try:
            config = AppConfig.for_root(get_active_root())
        except RuntimeError:
            config = None
        install_qwen(config)
        env = qwen_environment(config, offline=False)
        subprocess.run([sys.executable, "-m", "callforge.setup_models", "quality-models"],
                       env=env, check=True, timeout=7200)
    return checks() + [speaker_check()] + quality_checks()
