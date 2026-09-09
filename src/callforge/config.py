from __future__ import annotations

import os
import re
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
    codex_review_attempts: int = 2
    codex_artifact_grace_seconds: int = 10
    codex_reasoning_effort: str = "low"
    codex_model: str = ""
    codex_ignore_user_config: bool = True
    codex_ignore_rules: bool = True
    language: str = "fa"
    skill_name: str = "pbx-call-transcriber"
    whisper_model: str = "turbo"
    whisper_retry_model: str = "turbo"
    whisper_retry_segments: int = 3
    whisper_retry_seconds: int = 120
    transcription_profile: str = "max_quality_v2"
    asr_primary: str = "mlx-whisper:large-v3-turbo"
    asr_fallback: str = "mlx-whisper:large-v3"
    asr_alternative: str = ""
    asr_context_prompt: bool = False
    audio_enhancement: str = "adaptive"
    recovery_min_seconds: int = 8
    recovery_max_seconds: int = 15
    recovery_expand_seconds: int = 30
    recovery_context_seconds: int = 2
    asr_heavy_concurrency: int = 1
    glossary: tuple[str, ...] = ()
    diarization: bool = False
    diarization_timeout_seconds: int = 180
    alignment_timeout_seconds: int = 180
    role_timeout_seconds: int = 180

    @classmethod
    def for_root(cls, root: Path) -> "AppConfig":
        resolved = root.expanduser().resolve()
        workspace = resolved / WORKSPACE_NAME
        config_path = workspace / "config.toml"
        values: dict[str, object] = {}
        if config_path.is_file():
            with config_path.open("rb") as handle:
                values = tomllib.load(handle).get("callforge", {})
        if not isinstance(values.get("glossary", []), list) or not all(isinstance(item, str) for item in values.get("glossary", [])):
            raise ValueError("glossary must be a TOML array of spelling hints")
        legacy_primary = str(values.get("whisper_model", "turbo"))
        legacy_fallback = str(values.get("whisper_retry_model", "full"))
        legacy_names = {"turbo": "large-v3-turbo", "turbo-q4": "large-v3-turbo-q4", "full": "large-v3"}
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
            codex_review_attempts=int(values.get("codex_review_attempts", 2)),
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
            whisper_model=str(values.get("whisper_model", "turbo")),
            whisper_retry_model=str(values.get("whisper_retry_model", "turbo")),
            whisper_retry_segments=int(values.get("whisper_retry_segments", 3)),
            whisper_retry_seconds=int(values.get("whisper_retry_seconds", 120)),
            transcription_profile=str(values.get("transcription_profile", "max_quality_v2")),
            asr_primary=str(values.get("asr_primary", f"mlx-whisper:{legacy_names.get(legacy_primary, legacy_primary)}")),
            asr_fallback=str(values.get("asr_fallback", f"mlx-whisper:{legacy_names.get(legacy_fallback, legacy_fallback)}")),
            asr_alternative=str(values.get("asr_alternative", "")),
            asr_context_prompt=bool(values.get("asr_context_prompt", False)),
            audio_enhancement=str(values.get("audio_enhancement", "adaptive")),
            recovery_min_seconds=int(values.get("recovery_min_seconds", 8)),
            recovery_max_seconds=int(values.get("recovery_max_seconds", 15)),
            recovery_expand_seconds=int(values.get("recovery_expand_seconds", 30)),
            recovery_context_seconds=int(values.get("recovery_context_seconds", 2)),
            asr_heavy_concurrency=int(values.get("asr_heavy_concurrency", 1)),
            glossary=tuple(str(item) for item in values.get("glossary", [])),
            diarization=bool(values.get("diarization", True)),
            diarization_timeout_seconds=int(values.get("diarization_timeout_seconds", 180)),
            alignment_timeout_seconds=int(values.get("alignment_timeout_seconds", 180)),
            role_timeout_seconds=int(values.get("role_timeout_seconds", 180)),
        )
        if not 1 <= config.codex_review_attempts <= 3:
            raise ValueError("codex_review_attempts must be between 1 and 3")
        if config.whisper_retry_segments < 0 or config.whisper_retry_seconds < 1 or config.diarization_timeout_seconds < 1:
            raise ValueError("Retry segments must be nonnegative and retry seconds positive")
        if not 0 < config.recovery_min_seconds <= config.recovery_max_seconds <= config.recovery_expand_seconds:
            raise ValueError("Recovery windows must satisfy 0 < min <= max <= expand")
        if config.recovery_context_seconds < 0 or config.asr_heavy_concurrency != 1:
            raise ValueError("Recovery context must be nonnegative and heavy concurrency must be 1")
        if config.audio_enhancement not in {"off", "adaptive"}:
            raise ValueError("audio_enhancement must be off or adaptive")
        if (
            config.whisper_timeout_seconds < 1
            or config.codex_timeout_seconds < 1
            or config.codex_idle_timeout_seconds < 1
            or config.codex_artifact_grace_seconds < 0
            or config.alignment_timeout_seconds < 1
            or config.role_timeout_seconds < 1
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
                "codex_review_attempts = 2\n"
                "codex_artifact_grace_seconds = 10\n"
                'codex_reasoning_effort = "low"\n'
                'codex_model = ""\n'
                "codex_ignore_user_config = true\n"
                "codex_ignore_rules = true\n"
                'language = "fa"\n'
                'skill_name = "pbx-call-transcriber"\n'
                'whisper_model = "turbo"\n'
                'whisper_retry_model = "turbo"\n'
                'whisper_retry_segments = 3\n'
                'whisper_retry_seconds = 120\n'
                'transcription_profile = "max_quality_v2"\n'
                'asr_primary = "mlx-whisper:large-v3-turbo"\n'
                'asr_fallback = "mlx-whisper:large-v3"\n'
                'asr_alternative = ""\n'
                'asr_context_prompt = false\n'
                'audio_enhancement = "adaptive"\n'
                'recovery_min_seconds = 8\n'
                'recovery_max_seconds = 15\n'
                'recovery_expand_seconds = 30\n'
                'recovery_context_seconds = 2\n'
                'asr_heavy_concurrency = 1\n'
                'glossary = []\n'
                'diarization = true\n'
                'diarization_timeout_seconds = 180\n'
                'alignment_timeout_seconds = 180\n'
                'role_timeout_seconds = 180\n',
                encoding="utf-8",
            )
        else:
            # Idempotent additive migration: preserve every user value and only
            # add v2 keys that older workspaces cannot contain.
            with config_path.open("rb") as handle:
                existing = tomllib.load(handle).get("callforge", {})
            defaults = {
                "transcription_profile": '"max_quality_v2"',
                "asr_primary": '"mlx-whisper:large-v3-turbo"',
                "asr_fallback": '"mlx-whisper:large-v3"',
                "asr_alternative": '""',
                "asr_context_prompt": "false",
                "audio_enhancement": '"adaptive"',
                "recovery_min_seconds": "8",
                "recovery_max_seconds": "15",
                "recovery_expand_seconds": "30",
                "recovery_context_seconds": "2",
                "asr_heavy_concurrency": "1",
            }
            missing = [(key, value) for key, value in defaults.items() if key not in existing]
            if missing:
                text = config_path.read_text(encoding="utf-8")
                match = re.search(r"(?ms)^\[callforge\][^\n]*\n.*?(?=^\[|\Z)", text)
                if match is None:
                    raise ValueError("config.toml has no [callforge] table")
                addition = "\n# max_quality_v2 (added by an idempotent migration)\n" + "".join(
                    f"{key} = {value}\n" for key, value in missing
                )
                updated = text[:match.end()] + addition + text[match.end():]
                temporary = config_path.with_name(f"config.{os.getpid()}.tmp")
                temporary.write_text(updated, encoding="utf-8")
                temporary.replace(config_path)

    def runtime_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        # Resolving this symlink escapes CallForge's virtual environment and
        # makes installed Whisper packages invisible to child processes.
        environment["CALLFORGE_PYTHON"] = os.fspath(Path(os.sys.executable))
        environment["HF_HOME"] = os.fspath(self.models.resolve())
        # Fresh installations download once during setup, before init/scan.
        # Existing Whisper caches stay in place and are never silently copied.
        from callforge.registry import callforge_home
        if not (self.models / "hub").is_dir() and (callforge_home() / "speaker-runtime.json").is_file():
            from callforge.speaker_runtime import paths
            environment["HF_HOME"] = str(paths(self)[1])
        environment["CALLFORGE_MANAGED_TRANSCRIPTION"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        return environment
