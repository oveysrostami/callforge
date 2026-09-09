"""Isolated ASR comparisons against frozen references; never publishes to the DB.

Run with ``python -m callforge.asr_benchmark --help`` in the CallForge runtime.
Human corrections are used only by the scorer, never as decoder prompts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from callforge import __version__
from callforge.config import AppConfig
from callforge.evaluation import score, transcript_text
from callforge.local_transcriber import LocalWhisperPipeline
from callforge.quality import build_evidence, file_hash, write_json


def run_comparison(config, references: list[Path], output: Path, models: list[str], *,
                   temperatures: list[float] | None = None, seed: int | None = None,
                   context_prompt: bool = True) -> dict:
    if not references or not models:
        raise ValueError("Provide references and models")
    if temperatures and any(not math.isfinite(t) or not 0 <= t <= 1 for t in temperatures):
        raise ValueError("Temperatures must be finite values between 0 and 1")
    snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in references]
    if len({s["audio_id"] for s in snapshots}) != len(snapshots):
        raise ValueError("Each reference audio must occur once")
    for snapshot in snapshots:
        if snapshot.get("schema_version") != 1 or snapshot.get("split") != "development":
            raise ValueError("ASR experiments require frozen development references")
        if file_hash(Path(snapshot["audio_path"])) != snapshot["audio_sha256"]:
            raise ValueError("Audio no longer matches its frozen reference")
    output.mkdir(parents=True, exist_ok=False)
    pipeline = LocalWhisperPipeline(config)
    scripts = pipeline.skill_directory / "scripts"
    report = {"schema_version": 1, "callforge_version": __version__, "status": "running",
              "helper_sha256": file_hash(scripts / "transcribe_audio.py"),
              "matching_sha256": file_hash(Path(__file__).with_name("quality.py")),
              "reference_hashes": [file_hash(path) for path in references], "results": [],
              "temperatures": temperatures or [0, .2], "seed": seed, "context_prompt": context_prompt,
              "evaluation_scope": "ASR only, not final Codex review or speaker attribution",
              "limitations": "Development set; no generalization claim. Digit-token scores do not equate written numbers with digits. Timings include model loading, exclude download."}
    try:
        # Resolve immutable snapshots once, before timing, in the active cache.
        from huggingface_hub import snapshot_download
        resolved = {}
        for model in models:
            print(f"Resolving model: {model}", flush=True)
            resolved[model] = snapshot_download(repo_id=model,
                cache_dir=str(Path(config.runtime_environment()["HF_HOME"]) / "hub"))
        report["models"] = resolved
        for snapshot in snapshots:
            source = Path(snapshot["audio_path"])
            work = output / str(snapshot["audio_id"])
            work.mkdir()
            stderr = work / "stderr.log"
            print(f"Preparing audio {snapshot['audio_id']}", flush=True)
            metadata = pipeline._run_json(
                [sys.executable, str(scripts / "prepare_audio.py"), str(source), "--output-dir", str(work)],
                work / "audio.json", stderr, "prepare", timeout=60)
            reference = transcript_text(snapshot["reference"]["content"], snapshot["reference"])
            for index, model in enumerate(models):
                passes = {}
                for variant in ("raw", "agc"):
                    print(f"ASR {snapshot['audio_id']} {model} {variant}", flush=True)
                    options = [value for temperature in temperatures or [] for value in ("--temperature", str(temperature))]
                    if seed is not None:
                        options += ["--seed", str(seed)]
                    result = pipeline._run_json(
                        [sys.executable, str(scripts / "transcribe_audio.py"), metadata[f"{variant}_wav"],
                         "--backend", "mlx", "--model", resolved[model], "--language", config.language,
                         "--prompt", pipeline._whisper_prompt() if context_prompt else ""] + options,
                        work / f"model-{index}-{variant}.json", stderr, f"{model} {variant}")
                    passes[variant] = result
                    metrics = score(reference, " ".join(row["text"] for row in result.get("segments", [])))
                    entry = {"audio_id": snapshot["audio_id"], "model": model, "variant": variant,
                             "score": metrics, "elapsed_seconds": result.get("elapsed_seconds")}
                    report["results"].append(entry)
                    print(json.dumps(entry, ensure_ascii=False), flush=True)
                    write_json(output / "report.json", report)
                write_json(work / f"model-{index}-evidence.json",
                           build_evidence(passes["raw"], passes["agc"], metadata["duration_seconds"]))
            if file_hash(source) != snapshot["audio_sha256"]:
                raise ValueError("Source audio changed during experiment; results invalid")
        report["status"] = "completed"
        return report
    except Exception:
        report["status"] = "failed"
        raise
    finally:
        write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--model", action="append", required=True, help="MLX Whisper repo; may download locally")
    parser.add_argument("--temperature", type=float, action="append")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-context-prompt", action="store_true")
    args = parser.parse_args()
    from callforge.registry import get_active_root
    run_comparison(AppConfig.for_root(get_active_root()), args.reference, args.output.resolve(), args.model,
                   temperatures=args.temperature, seed=args.seed, context_prompt=not args.no_context_prompt)


if __name__ == "__main__":
    main()
