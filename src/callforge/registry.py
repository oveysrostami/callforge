from __future__ import annotations

import json
import os
import platform
from pathlib import Path


STATE_FILENAME = "state.json"


def callforge_home() -> Path:
    """Return the per-user CallForge state directory.

    CALLFORGE_HOME is primarily useful for automation and tests. The defaults
    follow each operating system's normal per-user application-data location.
    """

    override = os.environ.get("CALLFORGE_HOME")
    if override:
        return Path(override).expanduser().resolve()
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "CallForge"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "CallForge"
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "callforge"


def state_path() -> Path:
    return callforge_home() / STATE_FILENAME


def set_active_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {resolved}")
    target = state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"active_root": os.fspath(resolved)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return resolved


def get_active_root() -> Path:
    environment_root = os.environ.get("CALLFORGE_ROOT")
    if environment_root:
        return Path(environment_root).expanduser().resolve()
    source = state_path()
    if not source.is_file():
        raise RuntimeError(
            "No active audio directory. Run `callforge init /path/to/audio` "
            "or `callforge scan /path/to/audio` first."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        value = payload["active_root"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"CallForge state is invalid: {source}. Run `callforge init /path/to/audio` again."
        ) from exc
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"CallForge state is invalid: {source}. Run `callforge init /path/to/audio` again."
        )
    return Path(value).expanduser().resolve()
