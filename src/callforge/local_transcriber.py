from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from callforge.config import AppConfig


@dataclass(frozen=True)
class PreparedTranscription:
    metadata_path: Path
    raw_transcript_path: Path
    agc_transcript_path: Path


class LocalWhisperPipeline:
    """Run the deterministic audio and Whisper stages before Codex review."""

    def __init__(self, config: AppConfig, skill_directory: Path | None = None):
        self.config = config
        self.skill_directory = skill_directory or (
            Path.home() / ".agents" / "skills" / self.config.skill_name
        )

    def _event(
        self,
        log_path: Path,
        stage: str,
        state: str,
        message: str,
    ) -> None:
        record = {
            "type": "callforge.stage",
            "stage": stage,
            "state": state,
            "message": message,
        }
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_stderr(self, stderr_path: Path, label: str, content: str) -> None:
        if not content.strip():
            return
        with stderr_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(f"\n--- {label} ---\n")
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")

    def _run_json(
        self,
        command: list[str],
        destination: Path,
        stderr_path: Path,
        label: str,
    ) -> dict:
        try:
            completed = subprocess.run(
                command,
                cwd=self.config.root,
                env=self.config.runtime_environment(),
                capture_output=True,
                text=True,
                timeout=self.config.whisper_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            self._append_stderr(stderr_path, label, str(exc))
            raise RuntimeError(
                f"{label} exceeded the {self.config.whisper_timeout_seconds}-second timeout"
            ) from exc
        self._append_stderr(stderr_path, label, completed.stderr)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"{label} failed: {detail[-2000:]}")
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} returned invalid JSON") from exc
        destination.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return parsed

    def prepare(
        self,
        audio_path: Path,
        work_directory: Path,
        log_path: Path,
        stderr_path: Path,
    ) -> PreparedTranscription:
        scripts = self.skill_directory / "scripts"
        prepare_script = scripts / "prepare_audio.py"
        transcribe_script = scripts / "transcribe_audio.py"
        if not prepare_script.is_file() or not transcribe_script.is_file():
            raise RuntimeError(
                f"Installed skill scripts were not found under {scripts}. "
                "Run `callforge setup --yes --force-skill`."
            )

        metadata_path = work_directory / "audio.json"
        raw_transcript_path = work_directory / "raw-turbo.json"
        agc_transcript_path = work_directory / "agc-turbo.json"

        self._event(
            log_path,
            "prepare_audio",
            "active",
            "آماده‌سازی، اندازه‌گیری و تقویت صدا",
        )
        metadata = self._run_json(
            [
                sys.executable,
                str(prepare_script),
                str(audio_path),
                "--output-dir",
                str(work_directory),
            ],
            metadata_path,
            stderr_path,
            "audio preparation",
        )
        raw_wav = Path(str(metadata.get("raw_wav", "")))
        agc_wav = Path(str(metadata.get("agc_wav", "")))
        if not raw_wav.is_file() or not agc_wav.is_file():
            raise RuntimeError("Audio preparation did not create both WAV files")
        self._event(
            log_path,
            "prepare_audio",
            "completed",
            "نسخه‌های خام و تقویت‌شدهٔ صدا آماده شدند",
        )

        hub = self.config.models / "hub"
        model_cached = any(
            path.is_file()
            for pattern in (
                "models--*whisper*turbo*/snapshots/*/weights.safetensors",
                "models--*whisper*turbo*/snapshots/*/model.safetensors",
                "models--*whisper*turbo*/snapshots/*/model.bin",
            )
            for path in hub.glob(pattern)
        )
        if not model_cached:
            self._event(
                log_path,
                "model_download",
                "active",
                "دریافت یک‌بارهٔ مدل Whisper در cache پایدار workspace",
            )

        self._event(
            log_path,
            "whisper",
            "active",
            "اجرای پاس اول Whisper روی صدای خام",
        )
        self._run_json(
            [
                sys.executable,
                str(transcribe_script),
                str(raw_wav),
                "--model",
                "turbo",
                "--language",
                self.config.language,
            ],
            raw_transcript_path,
            stderr_path,
            "Whisper raw pass",
        )
        if not model_cached:
            self._event(
                log_path,
                "model_download",
                "completed",
                "مدل Whisper در cache پایدار ذخیره شد",
            )

        self._event(
            log_path,
            "whisper",
            "active",
            "اجرای پاس دوم Whisper روی صدای تقویت‌شده",
        )
        self._run_json(
            [
                sys.executable,
                str(transcribe_script),
                str(agc_wav),
                "--model",
                "turbo",
                "--language",
                self.config.language,
            ],
            agc_transcript_path,
            stderr_path,
            "Whisper AGC pass",
        )
        self._event(
            log_path,
            "whisper",
            "completed",
            "دو پاس مستقل Whisper کامل شد",
        )
        return PreparedTranscription(
            metadata_path=metadata_path,
            raw_transcript_path=raw_transcript_path,
            agc_transcript_path=agc_transcript_path,
        )
