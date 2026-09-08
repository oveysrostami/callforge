"""Bounded local alignment experiments, with frozen inputs and no publication."""
from __future__ import annotations

import json
import math
import sys
import subprocess
import tempfile
import time
from pathlib import Path

from callforge import __version__
from callforge.alignment import MODEL, REVISION
from callforge.benchmark import speaker_python
from callforge.quality import file_hash, write_json
from callforge.roles import align_segments


def setup_alignment_runtime(config) -> None:
    interpreter = speaker_python(config)
    if not interpreter.is_file():
        raise ValueError("Run callforge setup-diarization first")
    subprocess.run([str(interpreter), "-m", "pip", "install", "transformers==5.16.1"], check=True)


def run_alignment_experiment(config, benchmark: Path, diarization: Path, output: Path, *, timeout: int = 600, backend: str = "ctc") -> dict:
    import imageio_ffmpeg
    snapshot = json.loads(benchmark.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or snapshot.get("split") != "development":
        raise ValueError("Alignment requires a schema-1 development benchmark")
    source = Path(snapshot["audio_path"])
    benchmark_hash = file_hash(benchmark)
    acoustic_path = diarization / "report.json"
    acoustic = json.loads(acoustic_path.read_text(encoding="utf-8"))
    if (file_hash(source) != snapshot["audio_sha256"] or acoustic.get("audio_sha256") != snapshot["audio_sha256"]
            or acoustic.get("benchmark_sha256") != benchmark_hash or acoustic.get("status") != "completed"
            or acoustic.get("source_unchanged") is not True):
        raise ValueError("Source/diarization does not match the frozen benchmark")
    aligned = align_segments(snapshot["baseline"]["segments"], acoustic["runtime"]["turns"])
    selected = [{key: row[key] for key in ("id", "start", "end", "text")}
                for row in aligned if row["speaker_id"] is None]
    if len(selected) > 40:
        raise ValueError("Development alignment is limited to 40 ambiguous segments per call")
    if backend not in {"ctc", "mlx"}:
        raise ValueError("Unsupported alignment backend")
    worker = "alignment_worker" if backend == "ctc" else "alignment_mlx_worker"
    interpreter = speaker_python(config) if backend == "ctc" else Path(sys.executable)
    if not interpreter.is_file():
        raise ValueError("Run callforge setup-diarization and setup-alignment first")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = {"segments": selected}
    write_json(output / "input.json", data)
    from callforge.codex_runner import CodexRunner
    md_hash = CodexRunner._file_digest(source.with_suffix(".md"))
    started = time.monotonic()
    model_id = MODEL if backend == "ctc" else "mlx-community/whisper-large-v3-turbo-q4"
    report = {"schema_version": 1, "callforge_version": __version__, "status": "failed", "model": model_id,
              "backend": backend, "revision": REVISION if backend == "ctc" else None,
              "benchmark_sha256": benchmark_hash, "audio_sha256": snapshot["audio_sha256"],
              "diarization_report_sha256": file_hash(acoustic_path), "input_sha256": file_hash(output / "input.json"),
              "alignment_code_sha256": file_hash(Path(__file__).with_name("alignment.py")),
              "worker_code_sha256": file_hash(Path(__file__).with_name(worker + ".py")),
              "selected_segments": len(selected), "reference_used_for_inference": False, "published": False}
    try:
        if selected:
            environment = config.runtime_environment()
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
            environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
            with tempfile.TemporaryDirectory(prefix="audio-", dir=output) as scratch:
                wav = Path(scratch) / "raw.wav"
                with (output / "stderr.log").open("w", encoding="utf-8") as errors:
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(source),
                                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                                   check=True, stdout=subprocess.DEVNULL, stderr=errors, timeout=60)
                    with (output / "alignment.json").open("w", encoding="utf-8") as result:
                        subprocess.run([str(interpreter), "-m", "callforge." + worker,
                                        "--audio", str(wav), "--input", str(output / "input.json")],
                                       check=True, env=environment, stdout=result, stderr=errors, timeout=timeout)
            prediction = json.loads((output / "alignment.json").read_text(encoding="utf-8"))
        else:
            prediction = {"model": model_id, "revision": REVISION if backend == "ctc" else None, "segments": []}
            write_json(output / "alignment.json", prediction)
        if (prediction.get("model") != model_id or (backend == "ctc" and prediction.get("revision") != REVISION)
                or len(prediction.get("segments", [])) != len(selected)
                or {r["id"] for r in prediction["segments"]} != {r["id"] for r in selected}):
            raise ValueError("Incomplete or mismatched alignment output")
        refine_segments(aligned, prediction, acoustic["runtime"]["turns"])
        report.update(status="completed", revision=prediction.get("revision"), runtime=prediction)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: local alignment failed; see stderr.log; no transcript published"
        raise RuntimeError(report["error"]) from exc
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["source_unchanged"] = CodexRunner._file_digest(source) == snapshot["audio_sha256"]
        report["markdown_unchanged"] = CodexRunner._file_digest(source.with_suffix(".md")) == md_hash
        report["benchmark_unchanged"] = CodexRunner._file_digest(benchmark) == benchmark_hash
        if not all(report[key] for key in ("source_unchanged", "markdown_unchanged", "benchmark_unchanged")):
            report.update(status="failed", error="Sources changed during alignment; results invalid")
        write_json(output / "report.json", report)
    return report


def refine_segments(aligned: list[dict], alignment: dict, turns: list[dict]) -> list[dict]:
    """Use word-local acoustics, never majority-fill a mixed speaker segment.

    Preserve all literal parent text/timing. Word diagnostics may remain unknown;
    a parent is resolved only with >=60% supported words and no competing voice.
    """
    predictions = {row["id"]: row for row in alignment["segments"]}
    if len(predictions) != len(alignment["segments"]) or not set(predictions) <= {r["id"] for r in aligned}:
        raise ValueError("Unknown or duplicate alignment segment")
    result = []
    for parent in aligned:
        row = dict(parent)
        candidate = predictions.get(row["id"])
        if candidate and candidate.get("status") not in {"completed", "unalignable", "skipped"}:
            raise ValueError("Invalid alignment status")
        if row["speaker_id"] is not None or not candidate or candidate.get("status") != "completed":
            result.append(row)
            continue
        words, last_offset, last_time = [], 0, -1.
        for word in candidate["words"]:
            a, b = word["char_start"], word["char_end"]
            if not isinstance(a, int) or not isinstance(b, int) or a < last_offset or b <= a or row["text"][a:b] != word["text"]:
                raise ValueError("Alignment does not preserve literal text")
            last_offset = b
            item = dict(word, speaker_id=None)
            if word.get("accepted"):
                start, end = float(word["start"]), float(word["end"])
                if (not math.isfinite(start + end) or end <= start or start < row["start"] - .31
                        or end > row["end"] + .31 or start < last_time):
                    raise ValueError("Invalid aligned word bounds/order")
                if not (.3 <= float(word["score"]) <= 1 and
                        (word.get("method") == "whisper_attention" or .5 <= float(word["character_hits"]) <= 1)):
                    raise ValueError("Accepted alignment has insufficient evidence")
                last_time = end
                evidence = align_segments([dict(id="word", start=start, end=end, text=word["text"])], turns)[0]
                item["speaker_id"] = evidence["speaker_id"]
                item["acoustic_evidence"] = evidence["acoustic_evidence"]
            words.append(item)
        # Require every original token exactly once; do not accept selective evidence.
        import re
        expected = [(m.start(), m.end()) for m in re.finditer(r"\S+", row["text"])]
        if [(w["char_start"], w["char_end"]) for w in words] != expected:
            raise ValueError("Alignment omitted or added text units")
        assigned = [word for word in words if word["speaker_id"] is not None]
        identities = {word["speaker_id"] for word in assigned}
        row["aligned_words"] = words
        row["word_supported_fraction"] = len(assigned) / len(words) if words else 0.
        if len(identities) == 1 and row["word_supported_fraction"] >= .6:
            row["speaker_id"] = next(iter(identities))
            row["flags"] = sorted(set(row["flags"] + ["word_aligned", "boundary_review_required"]))
            row["uncertain"] = True  # Alignment confidence is not human approval.
        elif len(identities) > 1:
            row["flags"] = sorted(set(row["flags"] + ["mixed_speakers"]))
        result.append(row)
    return result


def render_word_review(segments: list[dict], roles: dict) -> str:
    """Human-readable diagnostics, separate from the original transcript."""
    import html
    def cell(value):
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("`", "&#96;")
    lines = ["# بازبینی مرز کلمات و گوینده", "",
             "این خروجی آزمایشی است؛ متن و زمان بخش‌های اصلی تغییر نکرده‌اند. زمان کلمات تخمینی است، نه تأیید انسانی.", "",
             "امتیاز هم‌ترازی احتمال صحت نیست؛ کلمات نامفهوم، اعداد و هم‌پوشانی صدا ممکن است بدون برچسب بمانند.", ""]
    for row in segments:
        if "aligned_words" not in row:
            continue
        lines.extend([f"## {cell(row['id'])} — {row['start']:.2f} تا {row['end']:.2f} ثانیه", "",
                      cell(row["text"]), "", f"نقش بخش: {cell(row['speaker'])}", "",
                      "| کلمه | زمان پیشنهادی | شناسهٔ صوتی | نقش پیشنهادی | وضعیت |",
                      "| --- | --- | --- | --- | --- |"])
        for word in row["aligned_words"]:
            label = word["speaker_id"]
            role = roles.get(label, {}).get("role", "گوینده نامشخص")
            timing = f"{word['start']:.2f}–{word['end']:.2f}" if word["start"] is not None else "—"
            status = "نیازمند بررسی" if not word["accepted"] or label is None else "شاهد محلی؛ تأییدنشده"
            lines.append(f"| {cell(word['text'])} | {timing} | {cell(label or '—')} | {cell(role)} | {status} |")
        lines.append("")
    return "\n".join(lines)
