from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from callforge.config import AppConfig, GlossaryTerm
from callforge import __version__
from callforge.asr import choose_candidate, parse_spec, provider_command
from callforge.lexicon import (alignment_requests, apply_alignment,
                               guard_unverified_entities, normalize_result,
                               term_snapshot)
from callforge.quality import (
    attach_coverage_retry,
    build_evidence,
    coverage_attempt_windows,
    coverage_retry_candidates,
    coverage_retry_windows,
    prompt_leakage,
    consensus_for_row,
    retry_priority,
    segment_is_quarantined,
    tight_vad_windows,
    usable_retry_text,
    write_json,
)
from callforge.json_utils import finite_json
from callforge.runtime_locks import HEAVY_MODEL_LOCK


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
        glossary = "، ".join(dict.fromkeys(
            (*self.config.glossary, *(term.canonical for term in self.config.terms))))
        if base and glossary:
            return f"{base} املای واژه‌های احتمالی: {glossary}"
        return base or glossary

    def _asr_command(self, audio: Path, spec_value: str, prompt: str = "",
                     start: float | None = None, end: float | None = None,
                     seed: int | None = 0) -> list[str]:
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
        """Normalize an observed non-sensitive alias without inventing a term."""
        legacy = tuple(GlossaryTerm(canonical=term) for term in self.config.glossary
                       if term.strip() and not any(char.isdigit() for char in term))
        normalize_result(parsed, (*legacy, *self.config.terms))

    def _validate_sensitive_terms(self, evidence: dict, audio: Path, directory: Path,
                                  stderr_path: Path) -> dict:
        """Confirm observed person-name aliases with the cached Persian CTC model."""
        requests, groups = alignment_requests(evidence, self.config.terms)
        summary = {"terms": term_snapshot(self.config.terms), "requested": len(groups),
                   "accepted": 0, "status": "not_needed" if not groups else "pending"}
        if not groups:
            evidence["lexicon"] = summary
            return summary
        from callforge.speaker_runtime import environment, python
        executable = python(self.config)
        if not executable.is_file():
            summary.update(status="unsupported", error="persian_ctc_runtime_missing")
            evidence["lexicon"] = summary
            return summary
        request_path = directory / "lexicon-alignment-input.json"
        result_path = directory / "lexicon-alignment.json"
        write_json(request_path, {"segments": requests})
        try:
            with HEAVY_MODEL_LOCK:
                completed = subprocess.run(
                    [str(executable), "-m", "callforge.alignment_worker", "--audio", str(audio),
                     "--input", str(request_path)],
                    env=environment(self.config, offline=True), capture_output=True, text=True,
                    timeout=self.config.alignment_timeout_seconds,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._append_stderr(stderr_path, "lexicon CTC", str(exc))
            summary.update(status="unsupported", error=type(exc).__name__)
            evidence["lexicon"] = summary
            return summary
        self._append_stderr(stderr_path, "lexicon CTC", completed.stderr)
        if completed.returncode:
            summary.update(status="unsupported",
                           error=(completed.stderr.strip() or completed.stdout.strip())[-1000:])
            evidence["lexicon"] = summary
            return summary
        try:
            result = finite_json(json.loads(completed.stdout))
        except (json.JSONDecodeError, TypeError) as exc:
            summary.update(status="unsupported", error=f"invalid_alignment_result: {exc}")
            evidence["lexicon"] = summary
            return summary
        write_json(result_path, result)
        decisions = apply_alignment(evidence, groups, result)
        summary.update(status="completed", accepted=sum(bool(row["accepted"]) for row in decisions),
                       decisions=decisions)
        evidence["lexicon"] = summary
        return summary

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
        guard = HEAVY_MODEL_LOCK if heavy else nullcontext()
        with guard:
            effective_timeout = timeout or self.config.whisper_timeout_seconds
            for native_attempt in range(2):
                try:
                    completed = subprocess.run(
                        command,
                        cwd=self.config.root,
                        env=self.config.runtime_environment(),
                        capture_output=True,
                        text=True,
                        timeout=effective_timeout,
                    )
                except subprocess.TimeoutExpired as exc:
                    self._append_stderr(stderr_path, label, str(exc))
                    raise RuntimeError(
                        f"{label} exceeded the {effective_timeout:.0f}-second timeout"
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
        fallback_transcript_path = work_directory / "fallback-vad.json"
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
            "انتخاب تطبیقی نسخهٔ صوت برای فرضیهٔ Turbo",
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
            "active",
            "اجرای large-v3 روی پنجره‌های دقیق گفتار",
        )
        duration = float(metadata.get("duration_seconds", 0))
        speech_regions = raw_result.get("speech_regions") or []
        vad_windows = tight_vad_windows(
            speech_regions, duration, maximum_seconds=self.config.recovery_max_seconds)
        fallback_result = None
        fallback_error = None
        if (vad_windows and not metadata.get("independent_conversation_channels")
                and parse_spec(self.config.asr_fallback).provider != "qwen3-asr"):
            windows_path = work_directory / "fallback-vad-windows.json"
            windows_path.write_text(json.dumps(vad_windows, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
            command = self._asr_command(raw_wav, self.config.asr_fallback, "")
            command.extend(["--windows-file", str(windows_path)])
            try:
                fallback_result = self._run_json(
                    command, fallback_transcript_path, stderr_path,
                    "fallback ASR tight VAD pass",
                )
            except RuntimeError as exc:
                fallback_error = str(exc)
                self._event(log_path, "whisper", "warning",
                            "پاس large-v3 کامل نشد؛ مسیر بازیابی محدود ادامه دارد")

        turbo_candidates = [("raw", raw_result)]
        if use_enhanced:
            turbo_candidates.append(("agc", enhanced_result))
        turbo_variant, turbo_result = choose_candidate(turbo_candidates)
        turbo_path = agc_transcript_path if turbo_variant == "agc" else raw_transcript_path
        fallback_has_usable_segment = bool(fallback_result and any(
            not segment_is_quarantined(row, duration)
            for row in fallback_result.get("segments", [])
        ))
        if fallback_has_usable_segment:
            evidence_primary = fallback_result
            evidence_alternative = turbo_result
            prepared_primary_path = fallback_transcript_path
            prepared_alternative_path = turbo_path
            primary_mode = "large_v3_tight_vad"
        else:
            evidence_primary = turbo_result
            evidence_alternative = (enhanced_result if use_enhanced and turbo_variant == "raw"
                                    else raw_result if use_enhanced else {"segments": []})
            prepared_primary_path = turbo_path
            prepared_alternative_path = (agc_transcript_path if turbo_variant == "raw"
                                         else raw_transcript_path)
            primary_mode = f"turbo_{turbo_variant}"
        self._event(
            log_path,
            "whisper",
            "completed",
            "فرضیه‌های ASR آماده شد؛ بررسی اختلاف‌ها و بخش‌های مشکوک",
        )
        evidence = build_evidence(evidence_primary, evidence_alternative, duration, speech_regions)
        coverage_deadline = time.monotonic() + self.config.whisper_retry_seconds
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
            "attempts": [],
            "coverage_budget_seconds": self.config.whisper_retry_seconds,
            "selective_retry_budget_seconds": self.config.whisper_retry_seconds,
            "selective_retry_attempts": [],
        }
        if coverage_plans:
            self._event(
                log_path,
                "whisper_coverage",
                "active",
                f"بازیابی {len(coverage_plans)} پنجرهٔ گفتاری فاقد متن با زمینهٔ کامل",
            )
        for index, plan in enumerate(coverage_plans):
            self._event(
                log_path,
                "whisper_coverage",
                "active",
                f"بازیابی پنجرهٔ {index + 1} از {len(coverage_plans)}: {plan['start']:.1f} تا {plan['end']:.1f} ثانیه",
            )
            recovered = 0
            specs = coverage_summary["cascade"]
            attempts = coverage_attempt_windows(
                plan, float(evidence["duration_seconds"]), self.config.recovery_expand_seconds)
            for attempt_index, window in enumerate(attempts):
                for cascade_index, spec in enumerate(specs):
                    remaining = coverage_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    candidates = []
                    for variant, retry_wav in retry_variants:
                        try:
                            retry = self._run_json(
                                self._asr_command(retry_wav, spec, whisper_prompt,
                                                  window["start"], window["end"]),
                                work_directory / f"coverage-{index + 1}-{attempt_index}-{cascade_index}-{variant}.json",
                                stderr_path, f"ASR coverage recovery {spec} {variant}",
                                timeout=remaining)
                            coverage_summary["attempts"].append({
                                "plan": index + 1, "window_mode": window["mode"],
                                "start": window["start"], "end": window["end"],
                                "provider": spec, "audio_variant": variant,
                            })
                            if spec.startswith("qwen3-asr:") and retry.get("status") == "completed":
                                retry = self._align_qwen_result(
                                    retry, retry_wav, work_directory, stderr_path,
                                    f"coverage-{index + 1}-{attempt_index}-{cascade_index}-{variant}",
                                ) or {"status": "unsupported", "error": "ctc_alignment_coverage_below_80pct"}
                            if retry.get("status") not in {"unsupported", "failed"}:
                                candidates.append((variant, retry))
                            else:
                                coverage_summary.setdefault("unsupported", []).append({
                                    "provider": spec, "status": retry.get("status"),
                                    "error": retry.get("error")})
                        except RuntimeError as exc:
                            coverage_summary.setdefault("errors", []).append(str(exc))
                    if candidates:
                        variant, retry = choose_candidate(candidates)
                        retry.setdefault("provenance", {})["selected_audio_variant"] = variant
                        recovered = attach_coverage_retry(evidence, retry, plan["segment_ids"])
                    if recovered:
                        break
                if recovered or coverage_deadline - time.monotonic() <= 0:
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
        # Coverage recovery and selective repair solve different failures. A
        # call with several silent/failed windows must not consume the entire
        # budget before a compact, high-priority decoder loop can be repaired.
        selective_deadline = time.monotonic() + self.config.whisper_retry_seconds
        for index, row in enumerate(candidates[:self.config.whisper_retry_segments]):
            remaining = selective_deadline - time.monotonic()
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
                coverage_summary["selective_retry_attempts"].append({
                    "segment_id": row["id"], "start": max(0, row["start"] - 2),
                    "end": min(evidence["duration_seconds"], row["end"] + 2),
                    "provider": self.config.asr_fallback, "audio_variant": "raw",
                })
                if any(set(s.get("flags", [])) & {"invalid_asr_metrics", "decoder_truncated"}
                       for s in retry.get("segments", [])):
                    row["flags"] = sorted(set(row["flags"] + ["invalid_asr_metrics"]))
                    self._event(log_path, "whisper_retry", "warning", "بازخوانی امتیاز معتبر ندارد؛ برای بازبینی انسانی علامت‌گذاری شد")
            except RuntimeError as exc:
                row["retry_error"] = str(exc)
                self._event(log_path, "whisper_retry", "warning", "بازخوانی تکمیلی کامل نشد؛ بخش برای بازبینی انسانی علامت‌گذاری شد")
        lexicon_summary = self._validate_sensitive_terms(
            evidence, raw_wav, work_directory, stderr_path)
        # Speaker inference runs after text review, in its isolated runtime.
        for row in evidence["segments"]:
            row.update(consensus_for_row(row))
            guard_unverified_entities(row)
            if row.get("entity_resolutions"):
                row["candidate_sources"] = list(dict.fromkeys(
                    [*row.get("candidate_sources", []), "persian_ctc_lexicon"]))
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
            "lexicon": lexicon_summary,
            "primary_mode": primary_mode,
            "tight_vad_windows": vad_windows,
            "fallback_error": fallback_error,
        }
        write_json(work_directory / "evidence.json", evidence)
        return PreparedTranscription(
            metadata_path=metadata_path,
            raw_transcript_path=prepared_primary_path,
            agc_transcript_path=prepared_alternative_path,
        )
