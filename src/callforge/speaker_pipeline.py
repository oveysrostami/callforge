"""Production speaker/role stages shared by CLI and UI. No human reference input."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from callforge.alignment import MODEL, REVISION
from callforge.alignment_experiment import refine_segments
from callforge.quality import write_json
from callforge.roles import align_segments, apply_roles, role_input, role_prompt, role_schema, validate_roles
from callforge.speaker_runtime import environment, python


class SpeakerProcessingError(RuntimeError):
    """Do not repeat costly ASR when a later speaker stage or setup fails."""
    def __init__(self, message, evidence_directory):
        super().__init__(message)
        self.evidence_directory = evidence_directory


class SpeakerPipeline:
    def __init__(self, config):
        self.config = config

    def preflight(self):
        from callforge.speaker_runtime import paths
        runtime, models = paths(self.config)
        try:
            ready = json.loads((runtime / "ready.json").read_text(encoding="utf-8"))
            valid = ready.get("schema") == 1 and ready.get("models") == str(models)
        except (OSError, ValueError):
            valid = False
        if not python(self.config).is_file() or not valid:
            raise RuntimeError("Speaker models are not set up. Run `callforge setup --yes` first.")

    def _json_process(self, command, destination, errors, timeout):
        from callforge.codex_runner import CodexRunner
        with destination.open("w", encoding="utf-8") as out, errors.open("a", encoding="utf-8") as err:
            process = subprocess.Popen(command, env=environment(self.config, offline=True),
                                       stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                       text=True, **CodexRunner._process_options())
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Local speaker stage exceeded {timeout} seconds") from None
            finally:
                if process.poll() is None:
                    CodexRunner._terminate_process_tree(process)
        if code:
            raise RuntimeError("Local speaker model failed. Run `callforge setup --yes`; details are in the processing log.")
        return json.loads(destination.read_text(encoding="utf-8"))

    def _roles(self, data, directory, log, errors):
        from callforge.codex_runner import CodexRunner
        write_json(directory / "role-input.json", data)
        write_json(directory / "role-schema.json", role_schema())
        prompt = directory / "role-prompt.txt"
        prompt.write_text(role_prompt(data), encoding="utf-8")
        if not data["speaker_ids"]:
            value = {"roles": []}
            write_json(directory / "roles.json", value)
            return value
        runner = CodexRunner(replace(self.config, codex_timeout_seconds=self.config.role_timeout_seconds,
                                    codex_idle_timeout_seconds=min(self.config.role_timeout_seconds,
                                                                   self.config.codex_idle_timeout_seconds)))
        executable = shutil.which("codex")
        if not executable:
            raise RuntimeError("Codex CLI missing; run callforge setup")
        command = [executable, "exec", "--ephemeral", "--json", "--sandbox", "read-only",
                   "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--cd", str(directory),
                   "--output-schema", str(directory / "role-schema.json"),
                   "--output-last-message", str(directory / "roles.json"),
                   "--config", f'model_reasoning_effort="{self.config.codex_reasoning_effort}"']
        model = runner._configured_model()
        if model:
            command += ["--model", model]
        command.append("-")
        offset = log.stat().st_size
        with log.open("a", encoding="utf-8") as out, errors.open("a", encoding="utf-8") as err, prompt.open(encoding="utf-8") as inp:
            process = subprocess.Popen(command, cwd=directory, env=self.config.runtime_environment(),
                                       stdin=inp, stdout=out, stderr=err, text=True, **runner._process_options())
            try:
                result = runner._wait_for_codex(process, log, errors)
            finally:
                if process.poll() is None:
                    runner._terminate_process_tree(process)
        if result.returncode:
            raise RuntimeError(f"Role inference failed: {result.forced_reason or result.returncode}; no transcript published")
        with log.open("rb") as handle:
            handle.seek(offset)
            for line in handle:
                event = json.loads(line)
                if "item" in event and event["item"].get("type") not in {"agent_message", "reasoning"}:
                    raise ValueError("Role inference consulted tools; result rejected")
        return json.loads((directory / "roles.json").read_text(encoding="utf-8"))

    def run(self, source, segments, directory, log, errors):
        from callforge.codex_runner import CodexRunner
        import imageio_ffmpeg
        event = lambda stage, state, message: CodexRunner._append_event(log, stage, state, message)
        self.preflight()
        started = time.monotonic()
        report = {"status": "failed", "reference_used_for_inference": False, "stage_seconds": {}}
        stage = "diarization"
        wav = directory / "speaker-raw.wav"
        try:
            event(stage, "active", "تفکیک محلی صداهای گوینده‌ها؛ مدل community-1")
            tick = time.monotonic()
            # Decode the original signal, not the gain-adjusted Whisper pass.
            with errors.open("a", encoding="utf-8") as err:
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(source),
                                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                               check=True, stdout=subprocess.DEVNULL, stderr=err, timeout=60)
            script = Path(__file__).parent / "resources/pbx-call-transcriber/scripts/diarize_audio.py"
            acoustic = self._json_process([str(python(self.config)), str(script), str(wav)],
                        directory / "diarization.json", errors, self.config.diarization_timeout_seconds)
            if acoustic.get("model") != "pyannote/speaker-diarization-community-1":
                raise ValueError("Unexpected diarization model")
            aligned = [dict(original, **assigned) for original, assigned in
                       zip(segments, align_segments(segments, acoustic["turns"]))]
            report["stage_seconds"][stage] = round(time.monotonic() - tick, 3)
            event(stage, "completed", "شناسه‌های صوتی ثبت شد؛ هنوز نقش افراد تعیین نشده است")
            stage = "word_alignment"
            tick = time.monotonic()
            selected = [{key: row[key] for key in ("id", "start", "end", "text")}
                        for row in aligned if row["speaker_id"] is None]
            write_json(directory / "alignment-input.json", {"segments": selected})
            event(stage, "active", f"تطبیق کلمات با صدا در {len(selected)} بخش مبهم؛ بدون تغییر متن")
            if selected:
                timing = self._json_process([str(python(self.config)), "-m", "callforge.alignment_worker",
                          "--audio", str(wav), "--input", str(directory / "alignment-input.json")],
                          directory / "alignment.json", errors, self.config.alignment_timeout_seconds)
                if (timing.get("model") != MODEL or timing.get("revision") != REVISION
                        or len(timing.get("segments", [])) != len(selected)
                        or {r["id"] for r in timing["segments"]} != {r["id"] for r in selected}):
                    raise ValueError("Incomplete or mismatched alignment output")
                aligned = refine_segments(aligned, timing, acoustic["turns"])
            report["stage_seconds"][stage] = round(time.monotonic() - tick, 3)
            event(stage, "completed", "تطبیق کلمات کامل شد؛ مرزهای نامطمئن برای بازبینی علامت‌گذاری شدند")
            stage = "speaker_roles"
            tick = time.monotonic()
            event(stage, "active", "تشخیص پشتیبان و مشتری از محتوای هر صدای شناسایی‌شده")
            prefix = source.name.split("-", 1)[0]
            direction = {"external": "inbound", "out": "outbound", "internal": "internal"}.get(prefix, "unknown")
            data = role_input(aligned, direction)
            value = self._roles(data, directory, log, errors)
            predicted = apply_roles(aligned, validate_roles(value, data))
            if [(r["id"], r["start"], r["end"], r["text"]) for r in predicted] != [
                    (r["id"], r["start"], r["end"], r["text"]) for r in segments]:
                raise ValueError("Speaker stage changed reviewed text or timing")
            report["stage_seconds"][stage] = round(time.monotonic() - tick, 3)
            report.update(status="completed", roles=value["roles"],
                          assigned_segments=sum(r["speaker"] != "گوینده نامشخص" for r in predicted),
                          total_segments=len(predicted))
            event(stage, "completed", f"نقش {report['assigned_segments']} از {len(predicted)} بخش مشخص شد؛ موارد نامطمئن بدون حدس باقی ماندند")
            return predicted, report
        except Exception:
            event(stage, "failed", "مرحلهٔ تشخیص گوینده/نقش ناموفق بود؛ متن جدید منتشر نشد. جزئیات در لاگ پردازش است")
            raise
        finally:
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            write_json(directory / "speaker-pipeline.json", report)
            wav.unlink(missing_ok=True)
