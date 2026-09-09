"""Frozen local references and isolated speaker experiments; never publish transcripts."""
from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from callforge import __version__
from callforge.evaluation import score, transcript_text
from callforge.quality import file_hash, write_json


UNKNOWN = {"", "گوینده نامشخص", "unknown"}


def speaker_python(config) -> Path:
    from callforge.speaker_runtime import python
    return python(config)


def setup_speaker_runtime(config) -> None:
    """Keep experimental torch/pyannote dependencies out of the Whisper runtime."""
    from callforge.speaker_runtime import install
    install(config)


def speaker_score(reference: list[dict], hypothesis: list[dict]) -> dict:
    """Time-weighted best one-to-one label mapping on singly-labelled reference speech.

    This is NOT full DER: reference gaps/overlap and false alarms outside the
    annotated regions are excluded. Roles in the reference stand in for speaker
    identity only for calls with one person per role.
    """
    def clean(rows):
        result = []
        for row in rows:
            start, end = float(row["start"]), float(row["end"])
            if not math.isfinite(start + end) or start < 0 or end <= start:
                raise ValueError("Invalid speaker interval")
            label = str(row.get("speaker_id", row.get("speaker", "")))
            result.append((start, end, label))
        return result
    ref, hyp = clean(reference), clean(hypothesis)
    boundaries = sorted({value for a, b, _ in ref + hyp for value in (a, b)})
    matrix, total, unknown, excluded_overlap = {}, 0., 0., 0.
    for a, b in zip(boundaries, boundaries[1:]):
        middle = (a + b) / 2
        r = {label for start, end, label in ref if start <= middle < end and label not in UNKNOWN}
        if len(r) != 1:
            if len(r) > 1:
                excluded_overlap += b - a
            continue
        total += b - a
        h = {label for start, end, label in hyp if start <= middle < end and label not in UNKNOWN}
        if len(h) != 1:
            unknown += b - a
            continue
        key = (next(iter(h)), next(iter(r)))
        matrix[key] = matrix.get(key, 0.) + b - a
    labels = sorted({h for h, _ in matrix})
    roles = sorted({r for _, r in matrix})
    if len(labels) > 8 or len(roles) > 8:
        raise ValueError("Speaker benchmark supports at most 8 speakers")
    best, mapping = 0., {}
    # Padding penalizes fragmentation: several predicted voices cannot all map
    # to the same reference speaker.
    targets = roles + [None] * max(0, len(labels) - len(roles))
    for assignment in itertools.permutations(targets, len(labels)):
        matched = sum(matrix.get((label, role), 0.) for label, role in zip(labels, assignment))
        if matched > best:
            best, mapping = matched, dict(zip(labels, assignment))
    return {"scored_seconds": total, "matched_seconds": best,
            "unknown_or_overlap_seconds": unknown, "excluded_reference_overlap_seconds": excluded_overlap,
            "assignment_error_rate": (total - best) / total if total else None,
            "best_label_mapping_for_evaluation_only": mapping,
            "note": "Not full DER; ignores unannotated audio and reference overlap. Assumes one person per reference role. Mapping is evaluation-only, not predicted customer/support roles."}


def freeze_reference(database, audio_id: int, destination: Path, *, split: str = "development") -> dict:
    if split not in {"development", "holdout"}:
        raise ValueError("Reference split must be development or holdout")
    with database.connect() as connection:
        audio = connection.execute("SELECT * FROM audio_files WHERE id=?", (audio_id,)).fetchone()
        reference = connection.execute(
            "SELECT t.*, r.status, r.data_json FROM transcripts t JOIN transcript_reviews r ON r.transcript_id=t.id "
            "WHERE t.audio_file_id=? AND t.is_current=1", (audio_id,)).fetchone()
        if audio is None or reference is None or reference["status"] != "approved":
            raise ValueError("Current transcript must be human-approved; approve the latest corrections in the UI first")
        baseline = connection.execute(
            "SELECT t.*, r.data_json FROM transcripts t LEFT JOIN transcript_reviews r ON r.transcript_id=t.id "
            "WHERE t.audio_file_id=? AND t.source='codex_skill' AND t.version<? AND t.audio_content_hash=? "
            "ORDER BY t.version DESC LIMIT 1", (audio_id, reference["version"], reference["audio_content_hash"])).fetchone()
    if baseline is None:
        raise ValueError("No matching machine baseline")
    source = Path(audio["absolute_path"])
    if file_hash(source) != reference["audio_content_hash"]:
        raise ValueError("Audio changed since human review")
    ref_data, base_data = json.loads(reference["data_json"]), json.loads(baseline["data_json"] or "{}")
    baseline_text = transcript_text(baseline["content"], base_data)
    baseline_tokens = max(1, len(baseline_text.split()))
    value = {"schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
             "callforge_version": __version__, "split": split,
             "audio_id": audio_id, "audio_path": str(source), "audio_sha256": reference["audio_content_hash"],
             "duration_seconds": audio["duration_seconds"], "direction": audio["direction"],
             "reference": {"transcript_id": reference["id"], "version": reference["version"],
                           "content": reference["content"], "segments": ref_data.get("segments", [])},
             "baseline": {"transcript_id": baseline["id"], "version": baseline["version"],
                          "content": baseline["content"], "segments": base_data.get("segments", [])},
             "baseline_text_score": score(transcript_text(reference["content"], ref_data),
                                          baseline_text),
             "baseline_unresolved_token_rate": baseline_text.count("[نامفهوم]") / baseline_tokens,
             "baseline_speaker_score": speaker_score(ref_data.get("segments", []), base_data.get("segments", []))}
    # Immutable snapshots: reruns must choose a new name.
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return value


def run_speaker_experiment(config, benchmark: Path, output: Path, *, timeout: int = 300) -> dict:
    import imageio_ffmpeg
    snapshot = json.loads(benchmark.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark schema")
    if snapshot.get("split") != "development":
        raise ValueError("Do not use holdout references for development experiments")
    source = Path(snapshot["audio_path"])
    if file_hash(source) != snapshot["audio_sha256"]:
        raise ValueError("Audio no longer matches the frozen benchmark")
    if not snapshot["reference"]["segments"]:
        raise ValueError("Speaker evaluation requires human-reviewed timed segments")
    interpreter = speaker_python(config)
    if not interpreter.is_file():
        raise ValueError("Isolated speaker runtime is missing. Run callforge setup-diarization first")
    output.mkdir(parents=True, exist_ok=False)
    environment = config.runtime_environment()
    environment["PYANNOTE_METRICS_ENABLED"] = "0"
    script = Path(__file__).parent / "resources/pbx-call-transcriber/scripts/diarize_audio.py"
    started = time.monotonic()
    report = {"schema_version": 1, "callforge_version": __version__, "benchmark_sha256": file_hash(benchmark),
              "audio_sha256": snapshot["audio_sha256"], "source_unchanged": True,
              "model": "pyannote/speaker-diarization-community-1", "status": "failed"}
    try:
        with tempfile.TemporaryDirectory(prefix="audio-", dir=output) as scratch:
            wav = Path(scratch) / "raw.wav"
            with (output / "stderr.log").open("w", encoding="utf-8") as errors:
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(source),
                                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                               check=True, stdout=subprocess.DEVNULL, stderr=errors, timeout=60)
                with (output / "diarization.json").open("w", encoding="utf-8") as result:
                    subprocess.run([str(interpreter), str(script), str(wav)], env=environment,
                                   check=True, stdout=result, stderr=errors, timeout=timeout)
        prediction = json.loads((output / "diarization.json").read_text(encoding="utf-8"))
        report.update(status="completed", runtime=prediction,
                      speaker_score=speaker_score(snapshot["reference"]["segments"], prediction["turns"]))
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: experiment failed; see stderr.log. Check model access and diarization dependencies."
        raise RuntimeError(report["error"]) from None
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        try:
            report["source_unchanged"] = file_hash(source) == snapshot["audio_sha256"]
        except OSError:
            report["source_unchanged"] = False
        if not report["source_unchanged"]:
            report["status"] = "failed"
            report["error"] = "Source audio changed during experiment; results invalid"
        write_json(output / "report.json", report)


def run_role_experiment(config, benchmark: Path, diarization_directory: Path, output: Path,
                        *, timeout: int = 180, alignment_directory: Path | None = None) -> dict:
    """Reuse frozen machine text and acoustic evidence. Never publish or read the DB."""
    from dataclasses import replace
    import shutil
    from callforge.codex_runner import CodexRunner
    from callforge.quality import render_markdown
    from callforge.roles import align_segments, apply_roles, role_input, role_prompt, role_schema, role_score, validate_roles

    if timeout < 1:
        raise ValueError("Timeout must be positive")
    benchmark_hash = file_hash(benchmark)
    snapshot = json.loads(benchmark.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or snapshot.get("split") != "development":
        raise ValueError("Role experiments require a schema-1 development benchmark")
    source = Path(snapshot["audio_path"])
    if file_hash(source) != snapshot["audio_sha256"]:
        raise ValueError("Audio no longer matches the frozen benchmark")
    acoustic = json.loads((diarization_directory / "report.json").read_text(encoding="utf-8"))
    if (acoustic.get("status") != "completed" or acoustic.get("source_unchanged") is not True
            or acoustic.get("audio_sha256") != snapshot["audio_sha256"]
            or acoustic.get("benchmark_sha256") != benchmark_hash):
        raise ValueError("Diarization does not match this benchmark")
    # Use the attested runtime within the completed report, not an unrelated JSON.
    aligned = align_segments(snapshot["baseline"]["segments"], acoustic["runtime"]["turns"])
    alignment_hash = None
    if alignment_directory is not None:
        from callforge.alignment_experiment import refine_segments
        alignment_path = alignment_directory / "report.json"
        alignment_hash = file_hash(alignment_path)
        timing = json.loads(alignment_path.read_text(encoding="utf-8"))
        if (timing.get("status") != "completed" or timing.get("source_unchanged") is not True
                or timing.get("benchmark_unchanged") is not True
                or timing.get("audio_sha256") != snapshot["audio_sha256"]
                or timing.get("benchmark_sha256") != benchmark_hash
                or timing.get("diarization_report_sha256") != file_hash(diarization_directory / "report.json")):
            raise ValueError("Word alignment does not match this experiment")
        aligned = refine_segments(aligned, timing["runtime"], acoustic["runtime"]["turns"])
    data = role_input(aligned, snapshot["direction"])
    if len({row["id"] for row in aligned}) != len(aligned) or not aligned:
        raise ValueError("Machine baseline requires unique non-empty timed segments")
    executable = shutil.which("codex")
    if not executable:
        raise ValueError("Codex CLI missing")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original_md_hash = CodexRunner._file_digest(source.with_suffix(".md"))
    started = time.monotonic()
    runner = CodexRunner(replace(config, codex_timeout_seconds=timeout,
                                 codex_idle_timeout_seconds=min(timeout, config.codex_idle_timeout_seconds)))
    model = runner._configured_model()
    report = {"schema_version": 1, "status": "failed", "callforge_version": __version__,
              "benchmark_sha256": benchmark_hash, "audio_sha256": snapshot["audio_sha256"],
              "diarization_report_sha256": file_hash(diarization_directory / "report.json"),
              "alignment_report_sha256": alignment_hash,
              "codex_model": model, "reasoning_effort": config.codex_reasoning_effort,
              "prediction_input": "machine_baseline_only", "reference_used_for_inference": False,
              "text_unchanged": True, "published": False}
    write_json(output / "input.json", data)
    write_json(output / "aligned.json", {"segments": aligned})
    write_json(output / "schema.json", role_schema())
    prompt = role_prompt(data)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    report["prompt_sha256"] = file_hash(output / "prompt.txt")
    command = [executable, "exec", "--ephemeral", "--json", "--sandbox", "read-only",
               "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--cd", str(output),
               "--output-schema", str(output / "schema.json"),
               "--output-last-message", str(output / "roles.json"),
               "--config", f'model_reasoning_effort="{config.codex_reasoning_effort}"']
    if model:
        command.extend(["--model", model])
    command.append("-")
    try:
        if data["speaker_ids"]:
            with ((output / "events.jsonl").open("w", encoding="utf-8") as events,
                  (output / "stderr.log").open("w", encoding="utf-8") as errors,
                  (output / "prompt.txt").open("r", encoding="utf-8") as prompt_file):
                process = subprocess.Popen(command, cwd=output, env=config.runtime_environment(),
                                           stdin=prompt_file, stdout=events, stderr=errors, text=True,
                                           **runner._process_options())
                try:
                    result = runner._wait_for_codex(process, output / "events.jsonl", output / "stderr.log")
                finally:
                    if process.poll() is None:
                        runner._terminate_process_tree(process)
            if result.returncode:
                raise ValueError(f"Role inference failed: {result.forced_reason or result.returncode}")
            # Reject runs that consulted extra evidence, even if the final JSON is valid.
            allowed_items = {"agent_message", "reasoning"}
            for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if "item" in event and event["item"].get("type") not in allowed_items:
                    raise ValueError("Role inference used tools; isolated comparison invalid")
            value = json.loads((output / "roles.json").read_text(encoding="utf-8"))
        else:
            value = {"roles": []}
            write_json(output / "roles.json", value)
        roles = validate_roles(value, data)
        predicted = apply_roles(aligned, roles)
        if [(r["id"], r["start"], r["end"], r["text"]) for r in predicted] != [
                (r["id"], r["start"], r["end"], r["text"]) for r in snapshot["baseline"]["segments"]]:
            raise ValueError("Role assignment changed transcript text or timing")
        write_json(output / "prediction.json", {"segments": predicted, "roles": value["roles"]})
        (output / "preview.md").write_text(render_markdown(source.name, predicted), encoding="utf-8")
        if alignment_directory is not None:
            from callforge.alignment_experiment import render_word_review
            (output / "word-review.md").write_text(render_word_review(predicted, roles), encoding="utf-8")
        report.update(status="completed", roles=value["roles"],
                      assigned_segments=sum(r["speaker"] != "گوینده نامشخص" for r in predicted),
                      total_segments=len(predicted),
                      refined_segments=sum("word_aligned" in r["flags"] for r in predicted),
                      mixed_segments=sum("mixed_speakers" in r["flags"] for r in predicted),
                      role_score=role_score(snapshot["reference"]["segments"], predicted),
                      baseline_role_score=role_score(snapshot["reference"]["segments"], snapshot["baseline"]["segments"]))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: role experiment failed; see local logs. No transcript published."
        raise RuntimeError(report["error"]) from exc
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["source_unchanged"] = CodexRunner._file_digest(source) == snapshot["audio_sha256"]
        report["markdown_unchanged"] = CodexRunner._file_digest(source.with_suffix(".md")) == original_md_hash
        report["benchmark_unchanged"] = CodexRunner._file_digest(benchmark) == benchmark_hash
        if not all(report[key] for key in ("source_unchanged", "markdown_unchanged", "benchmark_unchanged")):
            report["status"] = "failed"
            report["error"] = "Source/reference changed during experiment; results invalid"
        write_json(output / "report.json", report)
    return report
