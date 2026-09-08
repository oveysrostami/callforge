from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_NAME = ".callforge"


@dataclass(frozen=True)
class AppConfig:
    root: Path
    workspace: Path
    database: Path
    logs: Path
    runs: Path
    models: Path
    batch_size: int = 5
    workers: int = 2
    max_attempts: int = 3
    lease_seconds: int = 7200
    whisper_timeout_seconds: int = 1800
    codex_timeout_seconds: int = 600
    codex_idle_timeout_seconds: int = 120
    codex_artifact_grace_seconds: int = 10
    codex_reasoning_effort: str = "low"
    codex_model: str = ""
    codex_ignore_user_config: bool = True
    codex_ignore_rules: bool = True
    language: str = "fa"
    skill_name: str = "pbx-call-transcriber"

    @classmethod
    def for_root(cls, root: Path) -> "AppConfig":
        resolved = root.expanduser().resolve()
        workspace = resolved / WORKSPACE_NAME
        config_path = workspace / "config.toml"
        values: dict[str, object] = {}
        if config_path.is_file():
            with config_path.open("rb") as handle:
                values = tomllib.load(handle).get("callforge", {})
        config = cls(
            root=resolved,
            workspace=workspace,
            database=workspace / "callforge.sqlite3",
            logs=workspace / "logs",
            runs=workspace / "runs",
            models=workspace / "models",
            batch_size=int(values.get("batch_size", 5)),
            workers=int(values.get("workers", 2)),
            max_attempts=int(values.get("max_attempts", 3)),
            lease_seconds=int(values.get("lease_seconds", 7200)),
            whisper_timeout_seconds=int(values.get("whisper_timeout_seconds", 1800)),
            codex_timeout_seconds=int(values.get("codex_timeout_seconds", 600)),
            codex_idle_timeout_seconds=int(
                values.get("codex_idle_timeout_seconds", 120)
            ),
            codex_artifact_grace_seconds=int(
                values.get("codex_artifact_grace_seconds", 10)
            ),
            codex_reasoning_effort=str(values.get("codex_reasoning_effort", "low")),
            codex_model=str(values.get("codex_model", "")).strip(),
            codex_ignore_user_config=bool(
                values.get("codex_ignore_user_config", True)
            ),
            codex_ignore_rules=bool(values.get("codex_ignore_rules", True)),
            language=str(values.get("language", "fa")),
            skill_name=str(values.get("skill_name", "pbx-call-transcriber")),
        )
        if (
            config.whisper_timeout_seconds < 1
            or config.codex_timeout_seconds < 1
            or config.codex_idle_timeout_seconds < 1
            or config.codex_artifact_grace_seconds < 0
        ):
            raise ValueError("Processing timeouts must be positive integers")
        if config.codex_reasoning_effort not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
            "ultra",
        }:
            raise ValueError(
                "codex_reasoning_effort must be none, minimal, low, medium, high, "
                "xhigh, max, or ultra"
            )
        return config

    def ensure(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {self.root}")
        self.workspace.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        self.runs.mkdir(exist_ok=True)
        self.models.mkdir(exist_ok=True)
        config_path = self.workspace / "config.toml"
        if not config_path.exists():
            config_path.write_text(
                "[callforge]\n"
                "batch_size = 5\n"
                "workers = 2\n"
                "max_attempts = 3\n"
                "lease_seconds = 7200\n"
                "whisper_timeout_seconds = 1800\n"
                "codex_timeout_seconds = 600\n"
                "codex_idle_timeout_seconds = 120\n"
                "codex_artifact_grace_seconds = 10\n"
                'codex_reasoning_effort = "low"\n'
                'codex_model = ""\n'
                "codex_ignore_user_config = true\n"
                "codex_ignore_rules = true\n"
                'language = "fa"\n'
                'skill_name = "pbx-call-transcriber"\n',
                encoding="utf-8",
            )

    def runtime_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        # Resolving this symlink escapes CallForge's virtual environment and
        # makes installed Whisper packages invisible to child processes.
        environment["CALLFORGE_PYTHON"] = os.fspath(Path(os.sys.executable))
        environment["HF_HOME"] = os.fspath(self.models.resolve())
        environment["CALLFORGE_MANAGED_TRANSCRIPTION"] = "1"
        return environment
