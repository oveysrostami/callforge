from __future__ import annotations

import argparse
import sys
from pathlib import Path

from callforge import __version__
from callforge.codex_runner import CodexRunner
from callforge.config import AppConfig
from callforge.db import Database
from callforge.registry import get_active_root, set_active_root
from callforge.scanner import scan
from callforge.setup_tools import checks, setup
from callforge.worker import run_batch
from callforge.web import serve_ui


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def workspace(directory: str | None = None) -> tuple[AppConfig, Database]:
    root = Path(directory) if directory is not None else get_active_root()
    config = AppConfig.for_root(root)
    config.ensure()
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
        f"Indexed {result.discovered} MP3 files; skipped_zero_duration={result.skipped}; "
        f"{result.imported_markdown} existing Markdown transcripts imported."
    )
    return 0


def command_scan(args) -> int:
    config, database = workspace(args.directory)
    result = scan(config, database, import_markdown=not args.no_import_markdown)
    set_active_root(config.root)
    print(f"Active audio directory: {config.root}")
    print(
        f"Indexed {result.discovered} MP3 files; changed={result.changed}, "
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="callforge")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Check and install runtime dependencies")
    setup_parser.add_argument("--yes", action="store_true", help="Install missing components")
    setup_parser.add_argument("--force-skill", action="store_true", help="Back up and replace the installed skill")
    setup_parser.set_defaults(func=lambda args: 0 if print_checks(setup(args.yes, args.force_skill)) else 1)

    doctor = subparsers.add_parser("doctor", help="Check the local runtime")
    doctor.set_defaults(func=lambda args: 0 if print_checks(checks()) else 1)

    init = subparsers.add_parser("init", help="Initialize and index an audio directory")
    init.add_argument("directory")
    init.set_defaults(func=command_init)

    scan_parser = subparsers.add_parser("scan", help="Discover MP3 files and update metadata")
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
