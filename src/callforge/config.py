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
        return cls(
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
            language=str(values.get("language", "fa")),
            skill_name=str(values.get("skill_name", "pbx-call-transcriber")),
        )

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
                'language = "fa"\n'
                'skill_name = "pbx-call-transcriber"\n',
                encoding="utf-8",
            )

    def runtime_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CALLFORGE_PYTHON"] = os.fspath(Path(os.sys.executable).resolve())
        environment["HF_HOME"] = os.fspath(self.models.resolve())
        return environment
