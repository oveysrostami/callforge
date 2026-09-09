from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from callforge import __version__
from callforge.codex_runner import CodexRunner
from callforge.config import AppConfig
from callforge.db import Database
from callforge.registry import get_active_root, set_active_root
from callforge.scanner import scan
from callforge.setup_tools import checks, setup, speaker_check, quality_checks
from callforge.worker import run_batch
from callforge.web import serve_ui
from callforge.evaluation import sample_calls, score, transcript_text


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def workspace(directory: str | None = None) -> tuple[AppConfig, Database]:
    root = Path(directory) if directory is not None else get_active_root()
    config = AppConfig.for_root(root)
    config.ensure()
    # ensure() may have migrated an older config; use the effective v2 values
    # immediately rather than waiting for the next command invocation.
    config = AppConfig.for_root(root)
    database = Database(config.database)
    database.initialize()
    return config, database


def print_checks(items) -> bool:
    for item in items:
        marker = "OK" if item.ok else "MISSING"
        print(f"[{marker:7}] {item.name}: {item.detail}")
    return all(item.ok for item in items)


def command_init(args) -> int:
    config, database = workspace(args.directory)
    result = scan(config, database, import_markdown=True)
    set_active_root(config.root)
    print(f"Initialized {config.workspace}")
    print(f"Active audio directory: {config.root}")
    print(
        f"Indexed {result.discovered} audio files; skipped={result.skipped}; collisions={result.collisions}; "
        f"{result.imported_markdown} existing Markdown transcripts imported."
    )
    return 0


def command_scan(args) -> int:
    config, database = workspace(args.directory)
    result = scan(config, database, import_markdown=not args.no_import_markdown)
    set_active_root(config.root)
    print(f"Active audio directory: {config.root}")
    print(
        f"Indexed {result.discovered} audio files; changed={result.changed}, collisions={result.collisions}, "
        f"metadata_errors={result.metadata_errors}, skipped_zero_duration={result.skipped}, "
        f"imported_markdown={result.imported_markdown}."
    )
    return 0


def command_run(args) -> int:
    config, database = workspace()
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    workers = args.workers if args.workers is not None else config.workers
    if args.dry_run:
        counts = database.counts()
        available = counts.get("pending", 0)
        print(f"Dry run: would claim up to {min(batch_size, available)} of {available} pending jobs with {workers} workers.")
        return 0
    result, messages = run_batch(
        config, database, batch_size, workers, runner=CodexRunner(config)
    )
    for message in messages:
        print(message)
    print(f"Batch finished: claimed={result.claimed}, completed={result.completed}, failed={result.failed}.")
    return 1 if result.failed else 0


def command_status(args) -> int:
    _, database = workspace()
    values = database.counts()
    for key in (
        "total_audio_files", "audio_files", "skipped", "pending", "running",
        "completed", "failed", "transcripts", "current_transcripts",
    ):
        print(f"{key}: {values.get(key, 0)}")
    return 0


def command_workspace(args) -> int:
    config, _ = workspace()
    print(f"Audio directory: {config.root}")
    print(f"Database: {config.database}")
    return 0


def command_retry(args) -> int:
    _, database = workspace()
    print(f"Requeued {database.retry_failed()} failed jobs.")
    return 0


def command_reset(args) -> int:
    config, database = workspace()
    target: Path | None = None
    if args.directory:
        supplied = Path(args.directory).expanduser()
        target = (supplied if supplied.is_absolute() else config.root / supplied).resolve()
        root = config.root.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Reset directory must be inside the active audio directory: {root}")
        if target.exists() and not target.is_dir():
            raise ValueError(f"Reset target is not a directory: {target}")
        if target == root:
            target = None
    count = database.reset_count(target)
    if count == 0:
        print("No matching database records found. Source files were not changed.")
        return 0
    scope = f"under {target}" if target is not None else "from the active workspace"
    if not args.yes:
        answer = input(
            f"Delete {count} audio record(s) and all related database data {scope}? "
            "Source MP3/Markdown files will remain. Type RESET to continue: "
        )
        if answer.strip() != "RESET":
            print("Reset cancelled.")
            return 1
    deleted = database.reset(target)
    print(
        f"Deleted {deleted} audio record(s) and their related jobs, runs, transcripts, "
        "and artifacts. Source files were not changed."
    )
    return 0


def command_transcripts(args) -> int:
    _, database = workspace()
    rows = database.recent_transcripts(args.limit)
    for row in rows:
        print(
            f"v{row['version']}\t{row['source']}\t{row['relative_path']}\t{row['markdown_path']}"
        )
    return 0


def command_ui(args) -> int:
    config, database = workspace()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "Warning: UI is being exposed beyond localhost. Audio and transcripts have no authentication.",
            file=sys.stderr,
        )
    serve_ui(
        config,
        database,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
    )
    return 0


def command_evaluation_sample(args) -> int:
    _, database = workspace()
    with database.connect() as connection:
        rows = [dict(row) for row in connection.execute(
            "SELECT a.id, a.filename, a.relative_path, a.duration_seconds, a.direction, a.content_sha256 "
            "FROM audio_files a JOIN jobs j ON j.audio_file_id=a.id AND j.stage='transcribe' "
            "WHERE j.status != 'skipped' AND a.duration_seconds >= 0.5 ORDER BY a.id")]
    value = {"schema_version": 1, "seed": args.seed,
             "instructions": "Review these calls in the UI. Only approve after listening. Keep holdout calls out of prompt/model tuning.",
             "calls": sample_calls(rows, args.count, args.seed)}
    # Never silently overwrite an existing evaluation selection.
    with Path(args.output).expanduser().open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    print(f"Selected {len(value['calls'])} calls for human review: {args.output}")
    return 0


def command_evaluate(args) -> int:
    _, database = workspace()
    selection = json.loads(Path(args.sample).expanduser().read_text(encoding="utf-8")) if args.sample else None
    selected = {row["id"]: row for row in selection["calls"]} if selection else None
    results, excluded = [], []
    with database.connect() as connection:
        references = connection.execute(
            "SELECT t.*, r.data_json FROM transcripts t JOIN transcript_reviews r ON r.transcript_id=t.id "
            "JOIN jobs j ON j.audio_file_id=t.audio_file_id AND j.stage='transcribe' "
            "WHERE t.is_current=1 AND r.status='approved' AND j.status != 'skipped'").fetchall()
        for reference in references:
            audio_id = reference["audio_file_id"]
            if selected is not None and audio_id not in selected:
                continue
            if selected is not None and selected[audio_id]["content_sha256"] != reference["audio_content_hash"]:
                excluded.append({"audio_id": audio_id, "reason": "audio hash changed"}); continue
            baseline = connection.execute(
                "SELECT t.*, r.data_json FROM transcripts t LEFT JOIN transcript_reviews r ON r.transcript_id=t.id "
                "WHERE t.audio_file_id=? AND t.version<? AND t.source='codex_skill' AND t.audio_content_hash=? "
                "ORDER BY t.version DESC LIMIT 1", (audio_id, reference["version"], reference["audio_content_hash"])).fetchone()
            if baseline is None:
                excluded.append({"audio_id": audio_id, "reason": "no machine baseline"}); continue
            reference_text = transcript_text(reference["content"], json.loads(reference["data_json"] or "{}"))
            baseline_text = transcript_text(baseline["content"], json.loads(baseline["data_json"] or "{}"))
            results.append({"audio_id": audio_id, "reference_transcript_id": reference["id"],
                            "baseline_transcript_id": baseline["id"], "audio_sha256": reference["audio_content_hash"],
                            "split": selected[audio_id]["evaluation_split"] if selected else "unspecified",
                            **score(reference_text, baseline_text)})
    words = sum(row["reference_words"] for row in results)
    chars = sum(row["reference_characters"] for row in results)
    report = {"evaluated_calls": len(results), "wer": sum(row["word_errors"] for row in results) / words if words else None,
              "cer": sum(row["character_errors"] for row in results) / chars if chars else None,
              "calls": results, "excluded": excluded,
              "note": "Error rates compare machine text to explicitly human-approved versions; they are not model confidence. No speaker DER is claimed."}
    report["by_split"] = {}
    for split in sorted({row["split"] for row in results}):
        group = [row for row in results if row["split"] == split]
        group_words = sum(row["reference_words"] for row in group)
        group_chars = sum(row["reference_characters"] for row in group)
        report["by_split"][split] = {"calls": len(group),
            "wer": sum(row["word_errors"] for row in group) / group_words if group_words else None,
            "cer": sum(row["character_errors"] for row in group) / group_chars if group_chars else None}
    if selected is not None:
        report["selected_calls"] = len(selected)
        report["not_evaluated_audio_ids"] = sorted(set(selected) - {row["audio_id"] for row in results})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_benchmark_freeze(args) -> int:
    from callforge.benchmark import freeze_reference
    config, database = workspace()
    directory = config.workspace / "benchmarks"
    directory.mkdir(exist_ok=True)
    output = Path(args.output).expanduser().resolve() if args.output else directory / f"audio-{args.audio_id}.json"
    value = freeze_reference(database, args.audio_id, output, split=args.split)
    print(json.dumps({"benchmark": str(output), "reference_version": value["reference"]["version"],
                      "text": value["baseline_text_score"], "speaker": value["baseline_speaker_score"]}, ensure_ascii=False, indent=2))
    return 0


def command_benchmark_asr(args) -> int:
    from callforge.asr_benchmark import run_comparison
    config = AppConfig.for_root(get_active_root())  # No database writes or publication.
    report = run_comparison(config, [Path(p).expanduser().resolve() for p in args.reference],
                            Path(args.output).expanduser().resolve(), args.model,
                            temperatures=args.temperature, seed=args.seed,
                            providers=args.provider,
                            prompt_modes=["off"] if args.no_context_prompt else args.prompt_mode,
                            audio_variants=args.audio_variant, repeats=args.repeats,
                            holdout_references=[Path(p).expanduser().resolve() for p in args.holdout_reference]
                            if args.holdout_reference else None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


def command_benchmark_diarization(args) -> int:
    from callforge.benchmark import run_speaker_experiment
    from datetime import UTC, datetime
    config, _ = workspace()
    benchmark = Path(args.benchmark).expanduser().resolve()
    output = (Path(args.output).expanduser().resolve() if args.output else
              config.workspace / "experiments" / datetime.now(UTC).strftime("speakers-%Y%m%dT%H%M%S%fZ"))
    print(f"Isolated speaker experiment: {output}", flush=True)
    report = run_speaker_experiment(config, benchmark, output, timeout=args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


def command_benchmark_roles(args) -> int:
    from callforge.benchmark import run_role_experiment
    from datetime import UTC, datetime
    config = AppConfig.for_root(get_active_root())  # No DB initialization or writes for experiments.
    output = (Path(args.output).expanduser().resolve() if args.output else
              config.workspace / "experiments" / ("roles-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")))
    report = run_role_experiment(config, Path(args.benchmark).expanduser().resolve(),
                                 Path(args.diarization).expanduser().resolve(), output, timeout=args.timeout,
                                 alignment_directory=Path(args.alignment).expanduser().resolve() if args.alignment else None)
    print(json.dumps({"output": str(output), **report}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


def command_setup_alignment(args) -> int:
    from callforge.alignment_experiment import setup_alignment_runtime
    config = AppConfig.for_root(get_active_root())
    setup_alignment_runtime(config)
    print("Alignment dependencies ready in isolated speaker runtime; normal transcription settings unchanged.")
    return 0


def command_benchmark_alignment(args) -> int:
    from callforge.alignment_experiment import run_alignment_experiment
    from datetime import UTC, datetime
    config = AppConfig.for_root(get_active_root())
    output = (Path(args.output).expanduser().resolve() if args.output else config.workspace / "experiments" /
              ("alignment-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")))
    print(f"Local word alignment experiment: {output}", flush=True)
    report = run_alignment_experiment(config, Path(args.benchmark).expanduser().resolve(),
                                      Path(args.diarization).expanduser().resolve(), output, timeout=args.timeout, backend=args.backend)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


def command_hf_login(args) -> int:
    import subprocess
    from callforge.speaker_runtime import environment, python
    if not sys.stdin.isatty():
        raise ValueError("Run callforge hf-login in your own interactive terminal; do not pass tokens as command arguments")
    # A fresh process ensures the Hub reads this workspace's HF_HOME before import.
    # Hugging Face's own masked prompt verifies and stores the token, never in git.
    interpreter = python()
    if not interpreter.is_file():
        raise ValueError("Run callforge setup first")
    return subprocess.run([str(interpreter), "-m", "callforge.hf_auth"],
                          env=environment(), check=False).returncode


def command_setup(args) -> int:
    install = args.yes and not args.check
    if not args.check and not install:
        if not sys.stdin.isatty():
            raise ValueError("Use callforge setup --yes to install, or --check for a read-only check. Tokens require an interactive terminal.")
        install = input("Install dependencies and download local Whisper/speaker models? [Y/n]: ").strip().lower() not in {"n", "no"}
    return 0 if print_checks(setup(install, args.force_skill, quality_models=args.quality_models)) else 1


def command_setup_diarization(args) -> int:
    from callforge.benchmark import setup_speaker_runtime
    config, _ = workspace()
    setup_speaker_runtime(config)
    print("Isolated speaker runtime is ready. Whisper and normal transcription settings were not changed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="callforge")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Check and install runtime dependencies")
    setup_parser.add_argument("--yes", action="store_true", help="Install missing components")
    setup_parser.add_argument("--force-skill", action="store_true", help="Back up and replace the installed skill")
    setup_parser.add_argument("--diarization", action="store_true", help="Compatibility option: speaker models are now included by default")
    setup_parser.add_argument("--check", action="store_true", help="Read-only readiness check; no installation or login")
    setup_parser.add_argument("--quality-models", action="store_true",
                              help="Explicitly install/cache full Whisper and optional Qwen candidates")
    setup_parser.set_defaults(func=command_setup)

    doctor = subparsers.add_parser("doctor", help="Check the local runtime")
    doctor.set_defaults(func=lambda args: 0 if print_checks(checks() + [speaker_check()] + quality_checks()) else 1)

    init = subparsers.add_parser("init", help="Initialize and index an audio directory")
    init.add_argument("directory")
    init.set_defaults(func=command_init)

    scan_parser = subparsers.add_parser("scan", help="Discover MP3/WAV/M4A/FLAC/OGG files and update metadata")
    scan_parser.add_argument("directory")
    scan_parser.add_argument("--no-import-markdown", action="store_true")
    scan_parser.set_defaults(func=command_scan)

    run = subparsers.add_parser("run", help="Transcribe one configurable batch")
    run.add_argument("--batch-size", type=positive_int)
    run.add_argument("--workers", type=positive_int)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=command_run)

    status = subparsers.add_parser("status", help="Show queue and transcript counts")
    status.set_defaults(func=command_status)

    current = subparsers.add_parser("workspace", help="Show the active audio directory and database")
    current.set_defaults(func=command_workspace)

    sample = subparsers.add_parser("evaluation-sample", help="Select a local stratified human-review benchmark")
    sample.add_argument("--count", type=positive_int, default=40)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--output", required=True)
    sample.set_defaults(func=command_evaluation_sample)
    evaluate = subparsers.add_parser("evaluate", help="Measure machine WER/CER against human-approved revisions")
    evaluate.add_argument("--sample", help="Optional frozen evaluation selection JSON")
    evaluate.set_defaults(func=command_evaluate)

    freeze = subparsers.add_parser("benchmark-freeze", help="Freeze one current human-approved reference and its machine baseline")
    freeze.add_argument("--audio-id", required=True, type=positive_int)
    freeze.add_argument("--output", help="New JSON path; defaults to the active workspace benchmarks folder")
    freeze.add_argument("--split", choices=("development", "holdout"), default="development")
    freeze.set_defaults(func=command_benchmark_freeze)
    asr = subparsers.add_parser("benchmark-asr", help="Compare offline ASR providers against frozen references; never publish")
    asr.add_argument("--reference", action="append", required=True, help="Frozen development JSON; repeat for multiple references")
    asr.add_argument("--holdout-reference", action="append", help="Frozen holdout JSON; only the development winner is evaluated")
    asr.add_argument("--model", action="append", required=True, help="Model name or provider:model; must already be cached")
    asr.add_argument("--provider", action="append", choices=("mlx-whisper", "faster-whisper", "qwen3-asr"))
    asr.add_argument("--output", required=True, help="New experiment directory; never overwritten")
    asr.add_argument("--temperature", type=float, action="append", help="Experimental fallback schedule; repeat")
    asr.add_argument("--seed", type=int, default=42, help="Base seed; each repeat increments it")
    asr.add_argument("--repeats", type=positive_int, default=3)
    asr.add_argument("--prompt-mode", action="append", choices=("off", "domain", "glossary"),
                     help="Repeat to benchmark modes; default is off")
    asr.add_argument("--no-context-prompt", action="store_true", help=argparse.SUPPRESS)
    asr.add_argument("--audio-variant", action="append", choices=("raw", "agc", "denoise"),
                     help="Repeat to benchmark variants; denoise is never promoted implicitly")
    asr.set_defaults(func=command_benchmark_asr)
    speakers = subparsers.add_parser("benchmark-diarization", help="Run local speaker detection only, without publishing any transcript")
    speakers.add_argument("--benchmark", required=True)
    speakers.add_argument("--output", help="New experiment directory; existing directories are never overwritten")
    speakers.add_argument("--timeout", type=positive_int, default=300)
    speakers.set_defaults(func=command_benchmark_diarization)
    roles = subparsers.add_parser("benchmark-roles", help="Infer roles from machine text and existing acoustic evidence; never publish")
    roles.add_argument("--benchmark", required=True)
    roles.add_argument("--diarization", required=True, help="Directory of a completed matching benchmark-diarization run")
    roles.add_argument("--alignment", help="Optional completed benchmark-alignment directory; enables word-local refinement")
    roles.add_argument("--output", help="New experiment directory; existing directories are never overwritten")
    roles.add_argument("--timeout", type=positive_int, default=180)
    roles.set_defaults(func=command_benchmark_roles)
    alignment_setup = subparsers.add_parser("setup-alignment", help="Install experimental alignment dependencies in isolated runtime")
    alignment_setup.set_defaults(func=command_setup_alignment)
    alignment = subparsers.add_parser("benchmark-alignment", help="Align ambiguous machine-text words locally without publishing")
    alignment.add_argument("--benchmark", required=True)
    alignment.add_argument("--diarization", required=True)
    alignment.add_argument("--output")
    alignment.add_argument("--timeout", type=positive_int, default=600)
    alignment.add_argument("--backend", choices=("ctc", "mlx"), default="ctc", help="mlx uses only cached Whisper on Apple Silicon, without downloading")
    alignment.set_defaults(func=command_benchmark_alignment)
    login = subparsers.add_parser("hf-login", help="Secure interactive Hugging Face login for the active workspace model cache")
    login.set_defaults(func=command_hf_login)
    speaker_setup = subparsers.add_parser("setup-diarization", help="Install the experimental speaker detector in a separate runtime")
    speaker_setup.set_defaults(func=command_setup_diarization)

    retry = subparsers.add_parser("retry", help="Requeue terminal failures")
    retry.set_defaults(func=command_retry)

    reset = subparsers.add_parser("reset", help="Delete indexed database data without deleting source files")
    reset.add_argument("directory", nargs="?", help="Optional directory inside the active audio root")
    reset.add_argument("--yes", action="store_true", help="Skip the RESET confirmation prompt")
    reset.set_defaults(func=command_reset)

    transcripts = subparsers.add_parser("transcripts", help="List current transcripts")
    transcripts.add_argument("--limit", type=positive_int, default=20)
    transcripts.set_defaults(func=command_transcripts)

    ui = subparsers.add_parser("ui", help="Open the local audio and transcript browser")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=positive_int, default=8765)
    ui.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    ui.set_defaults(func=command_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
