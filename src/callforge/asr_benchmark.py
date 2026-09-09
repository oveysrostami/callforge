"""Offline, immutable ASR comparisons against frozen human references."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from callforge import __version__
from callforge.asr import ASRSpec, candidate_defects, parse_spec, provider_command
from callforge.config import AppConfig
from callforge.evaluation import score, transcript_text
from callforge.local_transcriber import LocalWhisperPipeline
from callforge.quality import file_hash, normalize, write_json


def _spec(value: str, provider: str | None = None) -> ASRSpec:
    return parse_spec(value) if ":" in value else ASRSpec(provider or "mlx-whisper", value)


def _prompt(pipeline: LocalWhisperPipeline, mode: str) -> str:
    if mode == "off":
        return ""
    if mode == "domain":
        return "این یک مکالمه تلفنی فارسی است."
    if mode == "glossary":
        return "املای واژه‌های احتمالی: " + "، ".join(pipeline.config.glossary)
    raise ValueError(f"Unknown prompt mode: {mode}")


def _frozen(paths: list[Path], split: str) -> list[dict]:
    snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({item["audio_id"] for item in snapshots}) != len(snapshots):
        raise ValueError("Each reference audio must occur once")
    for item in snapshots:
        if item.get("schema_version") != 1 or item.get("split") != split:
            raise ValueError(f"Expected frozen {split} references")
        if file_hash(Path(item["audio_path"])) != item["audio_sha256"]:
            raise ValueError("Audio no longer matches its frozen reference")
    return snapshots


def _qwen_ctc_alignment(config: AppConfig, audio: Path, result: dict,
                        directory: Path) -> tuple[float, list[dict]]:
    from callforge.speaker_runtime import environment, python
    if not python(config).is_file() or not result.get("text"):
        return 0.0, []
    request = directory / "qwen-ctc-input.json"
    write_json(request, {"segments": [{"id": "qwen", "start": result.get("start", 0),
                                        "end": result.get("end", 0), "text": result["text"]}]})
    completed = subprocess.run(
        [str(python(config)), "-m", "callforge.alignment_worker", "--audio", str(audio),
         "--input", str(request)], env=environment(config, offline=True), capture_output=True,
        text=True, timeout=config.alignment_timeout_seconds,
    )
    if completed.returncode:
        return 0.0, []
    aligned = json.loads(completed.stdout)
    words = (aligned.get("segments") or [{}])[0].get("words") or []
    supported = [word for word in words if word.get("score") is not None]
    coverage = sum(bool(word.get("accepted")) for word in supported) / len(supported) if supported else 0.0
    return coverage, words


def _aggregate(entries: list[dict]) -> dict:
    values = [entry["score"] for entry in entries]
    def mean(key: str) -> float:
        available = [value[key] for value in values if value.get(key) is not None]
        return sum(available) / len(available) if available else 0.0
    token_count = sum(max(1, len(normalize(entry["hypothesis"]).split())) for entry in entries)
    unresolved = sum(normalize(entry["hypothesis"]).count("[نامفهوم]") for entry in entries)
    return {"macro_wer": mean("wer"), "macro_cer": mean("cer"),
            "unresolved_token_rate": unresolved / token_count,
            "number_precision": mean("number_precision"), "runs": len(entries),
            "elapsed_seconds": sum(entry.get("elapsed_seconds") or 0 for entry in entries)}


def _promotion(winner: dict, snapshots: list[dict], holdout_count: int = 0) -> dict:
    baselines = [item.get("baseline_text_score") or {} for item in snapshots]
    if not baselines or any(value.get("wer") is None or value.get("cer") is None for value in baselines):
        return {"eligible": False, "reason": "baseline_v0.12_metrics_missing"}
    baseline_wer = sum(value["wer"] for value in baselines) / len(baselines)
    baseline_cer = sum(value["cer"] for value in baselines) / len(baselines)
    baseline_precision_values = [value["number_precision"] for value in baselines
                                 if value.get("number_precision") is not None]
    baseline_precision = (sum(baseline_precision_values) / len(baseline_precision_values)
                          if baseline_precision_values else None)
    metrics = winner["metrics"]
    checks = {
        "reference_split_30_10": len(snapshots) == 30 and holdout_count == 10,
        "seeded_repeats_at_least_3": metrics["runs"] >= len(snapshots) * 3,
        "wer_relative_reduction_20pct": metrics["macro_wer"] <= baseline_wer * .8,
        "cer_relative_reduction_20pct": metrics["macro_cer"] <= baseline_cer * .8,
        "number_precision_not_lower": baseline_precision is None or metrics["number_precision"] >= baseline_precision,
        "no_unsupported_numbers": all(not entry.get("unsupported_numbers") for entry in winner["entries"]),
        "zero_decoder_loops": all(not entry.get("decoder_loop") for entry in winner["entries"]),
        "zero_duplicate_intervals": all(not entry.get("duplicate_intervals") for entry in winner["entries"]),
    }
    baseline_unresolved = [item.get("baseline_unresolved_token_rate") for item in snapshots]
    checks["unresolved_relative_reduction_35pct"] = (
        all(value is not None for value in baseline_unresolved)
        and metrics["unresolved_token_rate"] <= sum(baseline_unresolved) / len(baseline_unresolved) * .65
    )
    return {"eligible": all(checks.values()), "checks": checks,
            "baseline": {"macro_wer": baseline_wer, "macro_cer": baseline_cer,
                         "number_precision": baseline_precision}}


def run_comparison(config: AppConfig, references: list[Path], output: Path,
                   models: list[str], *, temperatures: list[float] | None = None,
                   seed: int | None = None, context_prompt: bool | None = None,
                   providers: list[str] | None = None, prompt_modes: list[str] | None = None,
                   audio_variants: list[str] | None = None, repeats: int = 1,
                   holdout_references: list[Path] | None = None) -> dict:
    if not references or not models or repeats < 1:
        raise ValueError("Provide references/models and a positive repeat count")
    if temperatures and any(not math.isfinite(value) or not 0 <= value <= 1 for value in temperatures):
        raise ValueError("Temperatures must be finite values between 0 and 1")
    development = _frozen(references, "development")
    holdout = _frozen(holdout_references or [], "holdout") if holdout_references else []
    if providers and len(providers) not in {1, len(models)}:
        raise ValueError("Provide one provider or one provider per model")
    specs = [_spec(model, providers[min(index, len(providers) - 1)] if providers else None)
             for index, model in enumerate(models)]
    prompt_modes = list(prompt_modes or (["domain"] if context_prompt else ["off"]))
    audio_variants = list(audio_variants or ["raw", "agc"])
    if any(value not in {"raw", "agc", "denoise"} for value in audio_variants):
        raise ValueError("Audio variants are raw, agc, or denoise")
    immutable = {path: file_hash(path) for path in references + (holdout_references or [])}
    for snapshot in development + holdout:
        source = Path(snapshot["audio_path"])
        immutable[source] = file_hash(source)
        markdown = source.with_suffix(".md")
        if markdown.is_file():
            immutable[markdown] = file_hash(markdown)
    output.mkdir(parents=True, exist_ok=False)
    pipeline = LocalWhisperPipeline(config)
    report = {"schema_version": 2, "callforge_version": __version__, "status": "running",
              "reference_hashes": [file_hash(path) for path in references], "results": [],
              "temperatures": temperatures or [0, .2], "seed": seed, "repeats": repeats,
              "prompt_modes": prompt_modes, "audio_variants": audio_variants,
              "models": [f"{spec.provider}:{spec.model}" for spec in specs],
              "network_policy": "offline; run callforge setup --quality-models first",
              "evaluation_scope": "ASR only; references are scorer-only and never inference input"}
    groups: dict[str, list[dict]] = defaultdict(list)

    def evaluate(snapshots: list[dict], selected_specs: list[ASRSpec], phase: str,
                 selected_prompt_modes: list[str], selected_variants: list[str]) -> None:
        for snapshot in snapshots:
            source = Path(snapshot["audio_path"])
            work = output / phase / str(snapshot["audio_id"])
            work.mkdir(parents=True, exist_ok=True)
            stderr = work / "stderr.log"
            prepare = [sys.executable, str(pipeline.skill_directory / "scripts" / "prepare_audio.py"),
                       str(source), "--output-dir", str(work)]
            if "denoise" in selected_variants:
                prepare.append("--denoise")
            metadata = pipeline._run_json(prepare, work / "audio.json", stderr, "prepare", timeout=120)
            reference = transcript_text(snapshot["reference"].get("content", ""), snapshot["reference"])
            for spec in selected_specs:
                for prompt_mode in selected_prompt_modes:
                    for variant in selected_variants:
                        for repeat in range(repeats):
                            run_seed = (seed or 0) + repeat
                            key = f"{spec.provider}:{spec.model}|{prompt_mode}|{variant}"
                            destination = work / f"candidate-{len(report['results'])}.json"
                            try:
                                options = ([value for temperature in temperatures or []
                                            for value in ("--temperature", str(temperature))]
                                           if spec.provider != "qwen3-asr" else [])
                                command = provider_command(
                                    pipeline.skill_directory, Path(metadata[f"{variant}_wav"]), spec,
                                    config.language, _prompt(pipeline, prompt_mode), seed=run_seed,
                                ) + options
                                result = pipeline._run_json(command, destination, stderr, key)
                                if result.get("status") in {"unsupported", "failed"}:
                                    report["results"].append({"phase": phase, "audio_id": snapshot["audio_id"],
                                                              "candidate": key, "repeat": repeat,
                                                              "status": result.get("status"), "error": result.get("error")})
                                    continue
                                if spec.provider == "qwen3-asr":
                                    coverage, words = _qwen_ctc_alignment(
                                        config, Path(metadata["raw_wav"]), result, work)
                                    if coverage < .8:
                                        report["results"].append({"phase": phase, "audio_id": snapshot["audio_id"],
                                                                  "candidate": key, "repeat": repeat,
                                                                  "status": "unsupported",
                                                                  "error": "ctc_alignment_coverage_below_80pct",
                                                                  "alignment_coverage": coverage})
                                        continue
                                    result["aligned_words"] = words
                                    result.setdefault("metrics", {})["alignment_coverage"] = coverage
                                hypothesis = " ".join(row.get("text", "") for row in result.get("segments", []))
                                intervals = [(row.get("start"), row.get("end")) for row in result.get("segments", [])]
                                metrics = score(reference, hypothesis)
                                defects = candidate_defects(result)
                                entry = {"phase": phase, "audio_id": snapshot["audio_id"], "candidate": key,
                                         "provider": spec.provider, "model": spec.model,
                                         "prompt_mode": prompt_mode, "variant": variant, "repeat": repeat,
                                         "seed": run_seed, "status": "completed", "score": metrics,
                                         "hypothesis": hypothesis, "elapsed_seconds": result.get("elapsed_seconds"),
                                         "decoder_loop": "repetition" in defects,
                                         "duplicate_intervals": len(intervals) != len(set(intervals)),
                                         "unsupported_numbers": metrics["hypothesis_numbers"] > metrics["matched_numbers"]}
                                report["results"].append(entry)
                                if phase == "development":
                                    groups[key].append(entry)
                            except Exception as exc:
                                report["results"].append({"phase": phase, "audio_id": snapshot["audio_id"],
                                                          "candidate": key, "repeat": repeat, "status": "unsupported",
                                                          "error": f"{type(exc).__name__}: {exc}"})
                            write_json(output / "report.json", report)

    try:
        evaluate(development, specs, "development", prompt_modes, audio_variants)
        candidates = [{"candidate": key, "metrics": _aggregate(entries), "entries": entries}
                      for key, entries in groups.items() if len(entries) == len(development) * repeats]
        candidates.sort(key=lambda item: (item["metrics"]["macro_wer"],
                                          item["metrics"]["unresolved_token_rate"],
                                          item["metrics"]["macro_cer"]))
        report["candidates"] = [{key: value for key, value in item.items() if key != "entries"}
                                for item in candidates]
        if candidates:
            best_wer = candidates[0]["metrics"]["macro_wer"]
            close = [item for item in candidates if item["metrics"]["macro_wer"] - best_wer < .02]
            winner = min(close, key=lambda item: (item["metrics"]["unresolved_token_rate"],
                                                  item["metrics"]["macro_cer"]))
            report["winner"] = {key: value for key, value in winner.items() if key != "entries"}
            report["promotion"] = _promotion(winner, development, len(holdout))
            if holdout:
                provider_model, winner_prompt, winner_variant = winner["candidate"].split("|")
                selected = next(spec for spec in specs if f"{spec.provider}:{spec.model}" == provider_model)
                evaluate(holdout, [selected], "holdout", [winner_prompt], [winner_variant])
                holdout_entries = [entry for entry in report["results"]
                                   if entry.get("phase") == "holdout"
                                   and entry.get("candidate") == winner["candidate"]
                                   and entry.get("status") == "completed"]
                if len(holdout_entries) == len(holdout) * repeats:
                    holdout_metrics = _aggregate(holdout_entries)
                    report["holdout_metrics"] = holdout_metrics
                    baseline = [item.get("baseline_text_score") or {} for item in holdout]
                    unresolved = [item.get("baseline_unresolved_token_rate") for item in holdout]
                    baseline_precision_values = [value["number_precision"] for value in baseline
                                                 if value.get("number_precision") is not None]
                    holdout_checks = {
                        "wer_relative_reduction_20pct": holdout_metrics["macro_wer"] <=
                            (sum(value["wer"] for value in baseline) / len(baseline)) * .8,
                        "cer_relative_reduction_20pct": holdout_metrics["macro_cer"] <=
                            (sum(value["cer"] for value in baseline) / len(baseline)) * .8,
                        "unresolved_relative_reduction_35pct": all(value is not None for value in unresolved)
                            and holdout_metrics["unresolved_token_rate"] <= sum(unresolved) / len(unresolved) * .65,
                        "number_precision_not_lower": not baseline_precision_values or
                            holdout_metrics["number_precision"] >= sum(baseline_precision_values) / len(baseline_precision_values),
                        "no_unsupported_numbers": all(not entry["unsupported_numbers"] for entry in holdout_entries),
                        "zero_decoder_loops": all(not entry["decoder_loop"] for entry in holdout_entries),
                        "zero_duplicate_intervals": all(not entry["duplicate_intervals"] for entry in holdout_entries),
                    }
                    report["promotion"]["holdout_checks"] = holdout_checks
                    report["promotion"]["eligible"] = report["promotion"].get("eligible", False) and all(holdout_checks.values())
                else:
                    report["promotion"]["eligible"] = False
                    report["promotion"]["holdout_error"] = "winner_did_not_complete_all_holdout_runs"
        else:
            report["promotion"] = {"eligible": False, "reason": "no_complete_candidate"}
        approved_seconds = sum(float(item.get("duration_seconds") or 0) for item in development)
        report["fine_tuning_readiness"] = {
            "approved_development_hours": approved_seconds / 3600,
            "minimum_hours": 5,
            "eligible": approved_seconds >= 5 * 3600 and not report["promotion"].get("eligible", False),
            "holdout_excluded": True,
        }
        report["status"] = "completed"
        return report
    finally:
        changed = [str(path) for path, digest in immutable.items()
                   if not path.is_file() or file_hash(path) != digest]
        report["immutable_inputs_unchanged"] = not changed
        report["changed_inputs"] = changed
        if changed:
            report["status"] = "failed"
        write_json(output / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--holdout-reference", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--provider", choices=("mlx-whisper", "faster-whisper", "qwen3-asr"), action="append")
    parser.add_argument("--prompt-mode", choices=("off", "domain", "glossary"), action="append")
    parser.add_argument("--audio-variant", choices=("raw", "agc", "denoise"), action="append")
    parser.add_argument("--temperature", type=float, action="append")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from callforge.registry import get_active_root
    run_comparison(AppConfig.for_root(get_active_root()), args.reference, args.output.resolve(), args.model,
                   temperatures=args.temperature, seed=args.seed, providers=args.provider,
                   prompt_modes=args.prompt_mode, audio_variants=args.audio_variant, repeats=args.repeats,
                   holdout_references=args.holdout_reference)


if __name__ == "__main__":
    main()
