from __future__ import annotations

import json
import subprocess
import sys
import time
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from callforge.config import AppConfig
from callforge import __version__
from callforge.asr import choose_candidate, parse_spec, provider_command
from callforge.quality import (
    attach_coverage_retry,
    build_evidence,
    coverage_retry_candidates,
    coverage_retry_windows,
    prompt_leakage,
    consensus_for_row,
    normalize,
    retry_priority,
    usable_retry_text,
    write_json,
)
from callforge.json_utils import finite_json


_HEAVY_ASR_LOCK = threading.Semaphore(1)


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
            Path(__file__).parent / "resources" / self.config.skill_name
        )

    def _whisper_prompt(self) -> str:
        """Anchor the language/domain without suggesting call-specific facts."""
        if not self.config.asr_context_prompt:
            return ""
        base = (
            "این یک مکالمه تلفنی فارسی میان مشتری و کارشناس پشتیبانی است."
            if self.config.language == "fa" else ""
        )
        glossary = "، ".join(self.config.glossary)
        if base and glossary:
            return f"{base} املای واژه‌های احتمالی: {glossary}"
        return base or glossary

    def _asr_command(self, audio: Path, spec_value: str, prompt: str = "",
                     start: float | None = None, end: float | None = None,
                     seed: int | None = None) -> list[str]:
        return provider_command(self.skill_directory, audio, parse_spec(spec_value),
                                self.config.language, prompt, start, end, seed)

    def _model_cached(self, spec_value: str) -> bool:
        spec = parse_spec(spec_value)
        candidate = Path(spec.model).expanduser()
        if candidate.exists():
            return True
        if spec.provider == "qwen3-asr":
            from callforge.qwen_runtime import ready
            runtime_ready, _ = ready(self.config)
            prefix = f"models--Qwen--{spec.model.replace('Qwen/', '')}"
        elif spec.provider == "mlx-whisper":
            name = {"large-v3-turbo": "whisper-large-v3-turbo",
                    "large-v3-turbo-q4": "whisper-large-v3-turbo-q4",
                    "large-v3": "whisper-large-v3-mlx"}.get(spec.model, spec.model.split("/")[-1])
            prefix = f"models--mlx-community--{name}"
            runtime_ready = True
        else:
            name = {"large-v3-turbo": "faster-whisper-large-v3-turbo",
                    "large-v3": "faster-whisper-large-v3"}.get(spec.model, spec.model.split("/")[-1])
            prefix = f"models--Systran--{name}"
            runtime_ready = True
        hub = Path(self.config.runtime_environment()["HF_HOME"]) / "hub"
        return runtime_ready and any((hub / prefix / "snapshots").glob("*"))

    @staticmethod
    def _merge_channel_results(results: list[dict]) -> dict:
        segments = []
        speech = []
        for channel, result in enumerate(results, 1):
            segments.extend(dict(row, channel_id=channel) for row in result.get("segments", []))
            speech.extend(dict(row, channel_id=channel) for row in result.get("speech_regions", []))
        segments.sort(key=lambda row: (row.get("start", 0), row.get("end", 0), row.get("channel_id", 0)))
        merged = []
        for row in segments:
            if merged and row.get("start", 0) < merged[-1].get("end", 0):
                previous = merged[-1]
                previous["end"] = max(previous["end"], row["end"])
                previous["text"] = f"{previous.get('text', '').strip()} {row.get('text', '').strip()}".strip()
                previous["words"] = []
                previous["flags"] = sorted(set(previous.get("flags", []) + row.get("flags", []) + ["channel_overlap"]))
                previous["channel_id"] = [previous.get("channel_id"), row.get("channel_id")]
            else:
                merged.append(row)
        segments = merged
        speech.sort(key=lambda row: (row.get("start", 0), row.get("end", 0)))
        return {"provider": results[0].get("provider") if results else None,
                "model": results[0].get("model") if results else None,
                "text": " ".join(str(row.get("text", "")) for row in segments).strip(),
                "segments": segments, "speech_regions": speech,
                "settings": (results[0].get("settings") if results else {}),
                "provenance": {"mode": "independent_channels", "channels": len(results),
                               "sources": [item.get("provenance") for item in results]}}

    def _align_qwen_result(self, result: dict, audio: Path, directory: Path,
                           stderr_path: Path, label: str) -> dict | None:
        """Require >=80% Persian CTC coverage before Qwen becomes evidence."""
        from callforge.speaker_runtime import environment, python
        text = str(result.get("text") or "").strip()
        if not text or not python(self.config).is_file():
            return None
        request = directory / f"{label.replace(' ', '-')}-alignment-input.json"
        write_json(request, {"segments": [{"id": "qwen", "start": result.get("start", 0),
                                            "end": result.get("end", 0), "text": text}]})
        completed = subprocess.run(
            [str(python(self.config)), "-m", "callforge.alignment_worker", "--audio", str(audio),
             "--input", str(request)], env=environment(self.config, offline=True),
            capture_output=True, text=True, timeout=self.config.alignment_timeout_seconds,
        )
        self._append_stderr(stderr_path, f"{label} CTC", completed.stderr)
        if completed.returncode:
            return None
        aligned = json.loads(completed.stdout)
        words = (aligned.get("segments") or [{}])[0].get("words") or []
        supported = [word for word in words if word.get("score") is not None]
        accepted = [word for word in supported if word.get("accepted")]
        coverage = len(accepted) / len(supported) if supported else 0.0
        if coverage < .8:
            return None
        asr_words = [{"word": (" " if index else "") + word["text"], "start": word["start"],
                      "end": word["end"], "probability": word["score"]}
                     for index, word in enumerate(accepted)]
        candidate = dict(result)
        candidate["segments"] = [{"start": asr_words[0]["start"], "end": asr_words[-1]["end"],
                                  "text": " ".join(word["text"] for word in accepted),
                                  "words": asr_words, "flags": ["ctc_aligned"],
                                  "alignment_coverage": coverage}]
        candidate["text"] = candidate["segments"][0]["text"]
        candidate.setdefault("metrics", {})["alignment_coverage"] = coverage
        candidate.setdefault("provenance", {})["ctc_alignment_coverage"] = coverage
        return candidate

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
            "timestamp": datetime.now(UTC).isoformat(),
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

    def _normalize_glossary_spelling(self, parsed: dict) -> None:
        """Correct only an already-present orthographic equivalent.

        This cannot insert absent glossary entities and deliberately ignores
        multi-word terms and anything containing a digit.
        """
        terms = {normalize(term): term for term in self.config.glossary
                 if term.strip() and not any(char.isdigit() for char in term) and len(term.split()) == 1}
        if not terms:
            return
        changed = False
        for segment in parsed.get("segments", []):
            pieces = str(segment.get("text", "")).split()
            corrected = [terms.get(normalize(piece), piece) for piece in pieces]
            if corrected != pieces:
                segment["text"] = " ".join(corrected)
                changed = True
        if changed:
            parsed["text"] = " ".join(str(row.get("text", "")) for row in parsed.get("segments", [])).strip()
            parsed.setdefault("provenance", {})["glossary_mode"] = "orthographic_equivalence_only"

    def _run_json(
        self,
        command: list[str],
        destination: Path,
        stderr_path: Path,
        label: str,
        timeout: float | None = None,
    ) -> dict:
        started = time.monotonic()
        heavy = (any(value.endswith(("transcribe_audio.py", "qwen_worker.py")) for value in command)
                 or any(value in {"large-v3", "full", "Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B"}
                        for value in command))
        guard = _HEAVY_ASR_LOCK if heavy else nullcontext()
        with guard:
            for native_attempt in range(2):
                try:
                    completed = subprocess.run(
                        command,
                        cwd=self.config.root,
                        env=self.config.runtime_environment(),
                        capture_output=True,
                        text=True,
                        timeout=timeout or self.config.whisper_timeout_seconds,
                    )
                except subprocess.TimeoutExpired as exc:
                    self._append_stderr(stderr_path, label, str(exc))
                    raise RuntimeError(
                        f"{label} exceeded the {self.config.whisper_timeout_seconds}-second timeout"
                    ) from exc
                self._append_stderr(stderr_path, label, completed.stderr)
                if not completed.returncode:
                    break
                detail = completed.stderr.strip() or completed.stdout.strip()
                if native_attempt == 0 and "recursive_mutex lock failed" in detail:
                    self._append_stderr(stderr_path, f"{label} native retry",
                                        "MLX terminated after decoding; retrying this pass once in a fresh process.")
                    continue
                raise RuntimeError(f"{label} failed: {detail[-2000:]}")
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{label} returned an invalid result object")
        # Older helpers may still emit non-standard NaN/Infinity tokens.
        import math
        prompt = str((parsed.get("settings") or {}).get("prompt") or "")
        for row in parsed.get("segments", []):
            if prompt_leakage(str(row.get("text") or ""), prompt):
                row["flags"] = sorted(set(row.get("flags", []) + ["prompt_leakage"]))
            if any(isinstance(row.get(key), (int, float)) and not math.isfinite(row[key])
                   for key in ("avg_logprob", "no_speech_prob", "compression_ratio")):
                row["flags"] = sorted(set(row.get("flags", []) + ["invalid_asr_metrics"]))
            if prompt_leakage(str(row.get("text") or ""), prompt):
                row["flags"] = sorted(set(row.get("flags", []) + ["prompt_leakage"]))
        parsed = finite_json(parsed)
        self._normalize_glossary_spelling(parsed)
        parsed["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(destination, parsed)
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
        whisper_prompt = self._whisper_prompt()

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

        model_cached = self._model_cached(self.config.asr_primary)
        # The helper also enforces Hub offline mode, but fail here with a clear
        # instruction before a provider can even attempt network resolution.
        if not model_cached and self.skill_directory == Path(__file__).parent / "resources" / self.config.skill_name:
            raise RuntimeError(
                f"ASR model is not cached: {self.config.asr_primary}. "
                "Run `callforge setup --yes --quality-models`; runtime downloads are disabled."
            )

        self._event(
            log_path,
            "whisper",
            "active",
            "اجرای پاس اول Whisper روی صدای خام",
        )
        if metadata.get("independent_conversation_channels") and len(metadata.get("channel_wavs") or []) == 2:
            channel_results = []
            for channel, channel_wav in enumerate(metadata["channel_wavs"], 1):
                channel_results.append(self._run_json(
                    self._asr_command(Path(channel_wav), self.config.asr_primary, whisper_prompt),
                    work_directory / f"raw-channel-{channel}.json", stderr_path,
                    f"primary ASR channel {channel}"))
            raw_result = self._merge_channel_results(channel_results)
            write_json(raw_transcript_path, raw_result)
        else:
            raw_result = self._run_json(
                self._asr_command(raw_wav, self.config.asr_primary, whisper_prompt),
                raw_transcript_path, stderr_path, "primary ASR raw pass")
        self._event(
            log_path,
            "whisper",
            "active",
            "اجرای پاس دوم Whisper روی صدای تقویت‌شده",
        )
        use_enhanced = (self.config.audio_enhancement == "adaptive"
                        and metadata.get("enhancement_recommended")
                        and not metadata.get("independent_conversation_channels"))
        if use_enhanced:
            enhanced_result = self._run_json(
                self._asr_command(agc_wav, self.config.asr_primary, whisper_prompt),
                agc_transcript_path, stderr_path, "primary ASR AGC pass")
        else:
            enhanced_result = dict(raw_result)
            enhanced_result["provenance"] = dict(raw_result.get("provenance") or {},
                                                   audio_variant="raw", enhancement="skipped")
            write_json(agc_transcript_path, enhanced_result)
        self._event(
            log_path,
            "whisper",
            "completed",
            "دو پاس Whisper کامل شد؛ بررسی اختلاف‌ها و بخش‌های مشکوک",
        )
        evidence = build_evidence(raw_result, enhanced_result if use_enhanced else {"segments": []},
                                  float(metadata.get("duration_seconds", 0)),
                                  raw_result.get("speech_regions"))
        deadline = time.monotonic() + self.config.whisper_retry_seconds
        retry_variants = [("raw", raw_wav)]
        if use_enhanced:
            retry_variants.append(("agc", agc_wav))
        coverage_plans = coverage_retry_windows(
            evidence, minimum_seconds=self.config.recovery_min_seconds,
            maximum_seconds=self.config.recovery_max_seconds,
            context_seconds=self.config.recovery_context_seconds,
        )
        coverage_summary = {
            "requested_windows": len(coverage_plans),
            "completed_windows": 0,
            "recovered_segments": 0,
            "remaining_segments": len(coverage_retry_candidates(evidence)),
            "audio_variant": "window_quality_selection",
            "cascade": [self.config.asr_primary, self.config.asr_fallback] +
                       ([self.config.asr_alternative] if self.config.asr_alternative else []),
        }
        if coverage_plans:
            self._event(
                log_path,
                "whisper_coverage",
                "active",
                f"بازیابی {len(coverage_plans)} پنجرهٔ گفتاری فاقد متن با زمینهٔ کامل",
            )
        for index, plan in enumerate(coverage_plans):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._event(log_path, "whisper_coverage", "warning", "مهلت بازیابی پوشش گفتار پایان یافت")
                break
            self._event(
                log_path,
                "whisper_coverage",
                "active",
                f"بازیابی پنجرهٔ {index + 1} از {len(coverage_plans)}: {plan['start']:.1f} تا {plan['end']:.1f} ثانیه",
            )
            recovered = 0
            specs = coverage_summary["cascade"]
            for cascade_index, spec in enumerate(specs):
                window = dict(plan)
                if cascade_index and recovered == 0:
                    center = (plan["gap_start"] + plan["gap_end"]) / 2
                    width = min(float(evidence["duration_seconds"]), float(self.config.recovery_expand_seconds))
                    window["start"] = max(0.0, min(float(evidence["duration_seconds"]) - width, center - width / 2))
                    window["end"] = window["start"] + width
                candidates = []
                for variant, retry_wav in retry_variants:
                    try:
                        retry = self._run_json(
                            self._asr_command(retry_wav, spec, whisper_prompt,
                                              window["start"], window["end"]),
                            work_directory / f"coverage-{index + 1}-{cascade_index}-{variant}.json",
                            stderr_path, f"ASR coverage recovery {spec} {variant}", timeout=remaining)
                        if spec.startswith("qwen3-asr:") and retry.get("status") == "completed":
                            retry = self._align_qwen_result(
                                retry, retry_wav, work_directory, stderr_path,
                                f"coverage-{index + 1}-{cascade_index}-{variant}",
                            ) or {"status": "unsupported", "error": "ctc_alignment_coverage_below_80pct"}
                        if retry.get("status") not in {"unsupported", "failed"}:
                            candidates.append((variant, retry))
                        else:
                            coverage_summary.setdefault("unsupported", []).append({"provider": spec,
                                                                                   "status": retry.get("status"),
                                                                                   "error": retry.get("error")})
                    except RuntimeError as exc:
                        coverage_summary.setdefault("errors", []).append(str(exc))
                if candidates:
                    variant, retry = choose_candidate(candidates)
                    retry.setdefault("provenance", {})["selected_audio_variant"] = variant
                    recovered = attach_coverage_retry(evidence, retry, plan["segment_ids"])
                if recovered:
                    break
            coverage_summary["completed_windows"] += 1
            coverage_summary["recovered_segments"] += recovered
            if not recovered:
                self._event(log_path, "whisper_coverage", "warning",
                            "cascade این پنجره evidence کافی نداشت؛ بخش نامفهوم باقی ماند")
        coverage_summary["remaining_segments"] = len(coverage_retry_candidates(evidence))
        evidence["coverage_recovery"] = coverage_summary
        if coverage_plans:
            self._event(
                log_path,
                "whisper_coverage",
                "completed" if coverage_summary["remaining_segments"] == 0 else "warning",
                f"بازیابی پوشش تمام شد: {coverage_summary['recovered_segments']} بخش بازیابی شد و "
                f"{coverage_summary['remaining_segments']} بخش نیازمند بازبینی ماند",
            )
        candidates = [row for row in evidence["segments"] if set(row["flags"]) & {
            "repetition", "compression", "pass_disagreement", "speech_gap", "low_logprob",
            "invalid_asr_metrics", "decoder_truncated"} and usable_retry_text(row) is None]
        candidates.sort(key=retry_priority)
        for index, row in enumerate(candidates[:self.config.whisper_retry_segments]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._event(log_path, "whisper_retry", "active", f"بازخوانی محدود بخش مشکوک {index + 1}: {row['start']:.1f} تا {row['end']:.1f} ثانیه")
            try:
                retry = self._run_json(
                    self._asr_command(raw_wav, self.config.asr_fallback, whisper_prompt,
                                      max(0, row["start"] - 2),
                                      min(evidence["duration_seconds"], row["end"] + 2)),
                    work_directory / f"retry-{row['id']}.json", stderr_path,
                    "fallback ASR selective retry", timeout=remaining)
                row["retry"] = retry
                if any(set(s.get("flags", [])) & {"invalid_asr_metrics", "decoder_truncated"}
                       for s in retry.get("segments", [])):
                    row["flags"] = sorted(set(row["flags"] + ["invalid_asr_metrics"]))
                    self._event(log_path, "whisper_retry", "warning", "بازخوانی امتیاز معتبر ندارد؛ برای بازبینی انسانی علامت‌گذاری شد")
            except RuntimeError as exc:
                row["retry_error"] = str(exc)
                self._event(log_path, "whisper_retry", "warning", "بازخوانی تکمیلی کامل نشد؛ بخش برای بازبینی انسانی علامت‌گذاری شد")
        # Speaker inference runs after text review, in its isolated runtime.
        for row in evidence["segments"]:
            row.update(consensus_for_row(row))
        evidence["provenance"] = {
            "callforge_version": __version__,
            "pipeline": self.config.transcription_profile,
            "primary": self.config.asr_primary,
            "fallback": self.config.asr_fallback,
            "alternative": self.config.asr_alternative or None,
            "context_prompt": self.config.asr_context_prompt,
            "audio_enhancement": self.config.audio_enhancement,
            "heavy_concurrency": self.config.asr_heavy_concurrency,
            "cascade": coverage_summary,
        }
        write_json(work_directory / "evidence.json", evidence)
        return PreparedTranscription(
            metadata_path=metadata_path,
            raw_transcript_path=raw_transcript_path,
            agc_transcript_path=agc_transcript_path,
        )
