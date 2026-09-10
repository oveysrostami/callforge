---
name: pbx-call-transcriber
description: Transcribe Persian audio locally from MP3, WAV, M4A, FLAC, or OGG with an evidence-grounded ASR cascade and save a reviewed same-name Markdown transcript. PBX filenames are optional metadata. Do not use for general audio editing or summaries without transcription.
---

# PBX Call Transcriber

Transcribe the requested recording locally. In standalone mode save a same-name Markdown transcript beside the audio. In managed mode return structured segment corrections; CallForge preserves evidence, renders Markdown, and stores versions in its database. Machine-reviewed is not human-approved.

## Privacy and accuracy rules

- Run speech recognition locally. Do not upload the recording to a third-party transcription API.
- Never invent a word, number, name, or speaker identity. Write `[نامفهوم]` for unresolved audio.
- Interpret filename prefixes exactly: `external-` is an inbound call, `out-` is an outbound call, and `internal-` is an extension-to-extension call.
- For `external-`, the first field is the support extension and the second is the remote number. For `out-`, the first field is the remote number and the second is the support extension. For `internal-`, the first and second fields are the two extensions.
- Speaker identity and role are separate. Use `گوینده نامشخص` whenever the evidence does not establish a turn or role. Text-only review is not acoustic diarization. Filename direction alone cannot identify a speaker; use `کارشناس پشتیبانی`, `مشتری`, or extension labels only when supported.
- A glossary is only a spelling hint. Do not insert missing entities, resolve an ambiguous amount/date/identifier from context, or copy numbers from filenames. Keep units and digits literal and mark uncertain wording. Two passes of the same model can share an error; agreement is not proof.
- A structured glossary may provide an accepted `entity_resolutions` record for an exact person-name span. This means the observed alias passed local Persian CTC score, character-hit and runner-up checks. It is acoustic evidence only for that span. A rejected `entity_candidates` record is not evidence and must remain unresolved.
- Do not summarize or omit greetings, repeated phrases, hesitations that affect meaning, amounts, identifiers, or closing remarks.

## Runtime

Use the interpreter in `$CALLFORGE_PYTHON` when set; otherwise use `python3` (or `python` on Windows). Use that path exactly: do not resolve its symlink or substitute another interpreter. Helper scripts are under this skill's `scripts` directory.

The helper automatically selects MLX Whisper on Apple Silicon and faster-whisper elsewhere. Preserve the inherited `HF_HOME`; it is the persistent model cache. Never create a virtual environment, install packages, change `HF_HOME`, or manually download or assemble model files. If the configured interpreter or backend is unavailable, stop with a concise instruction to run `callforge setup --yes --force-skill`.

## CallForge-managed review

When the request supplies prepared JSON paths and a canonical evidence timeline, read these files (including contextual coverage recovery and selective-retry evidence when present), compare timed segments, and return the structured response required by the supplied schema. In this mode:

- do not run audio preparation or Whisper again;
- do not inspect, install, repair, download, or benchmark runtimes or models;
- do not write files, create extra artifacts, or claim to have listened to the source audio;
- prefer high-tier consensus, preserve agreed prefix/suffix, and put `[نامفهوم]` only over the unresolved span;
- accept ordinary words with two valid hypotheses or strong alignment. Require two model families or sufficient acoustic evidence for numbers, amounts, dates, names and identifiers;
- use an accepted `entity_resolutions` canonical spelling only in its attached segment; never copy it into another introduction or infer a person from extension/history;
- include every canonical segment id exactly once, retaining complete wording without summarization; put unresolved speech in `[نامفهوم]`, set `uncertain=true`, and explain the unresolved difference in `notes`;
- enhanced alternatives are partitioned between review units; do not repeat one unit's alternative in its neighbors. `alternative_timing_uncertain` marks a joint unit with unavailable word timing, not automatically unintelligible speech;
- do not alter timing or omit a segment because it is repetitive or difficult; describe confirmed non-speech explicitly and flag uncertain non-speech decisions;
- keep numbers and units grounded in the supplied evidence; never make the transcript more specific than that evidence;
- when CallForge says the separate speaker pipeline is enabled, return `گوینده نامشخص` during text review. CallForge subsequently runs local community-1 diarization, aligns ambiguous reviewed words with the Persian CTC model, and infers roles from evidence attached to each acoustic voice. Do not run those stages yourself or rewrite text to fit a proposed role;
- return only the final structured response. CallForge checks completion and coverage, generates Markdown, preserves the raw evidence, and leaves the transcript awaiting human review.

## Standalone workflow

Use this workflow only when CallForge-managed JSON inputs were not supplied.

1. Resolve the supported audio path and exact sibling Markdown path. Refuse same-stem audio collisions.
2. Inspect duration and filename metadata. Reject a missing or empty source.
3. Create an isolated temporary working directory. Do not place intermediate WAV or JSON files beside the source.
4. Run `scripts/prepare_audio.py SOURCE --output-dir TEMP`. It creates a conservative 16 kHz mono decode and an AGC copy and reports signal statistics.
5. Use non-Q4 Turbo on raw audio. Use adaptive AGC only for quiet windows and full large-v3 only for unresolved spans. Context prompt is off by default; glossary terms are spelling normalization only.
6. Quarantine repetition, compression, invalid timestamp and truncation before word allocation. Never expand a zero-duration token loop over the decoder interval. Never download a model during transcription.
7. Reconstruct speaker turns conservatively. Telephone recordings may be mono; do not pretend speaker diarization is certain.
8. Read the final text from start to finish and remove Whisper hallucinations caused by silence, repeated fragments, and impossible continuations.
9. Write the final UTF-8 Markdown atomically to the exact sibling path. Ensure it is non-empty, then delete temporary files.

## Markdown contract

Use this structure:

```markdown
# متن تماس

- فایل صوتی: `source-name.mp3`
- جهت تماس: ورودی|خروجی|داخلی به داخلی|نامشخص
- زبان: فارسی

## مکالمه

**کارشناس پشتیبانی:** ...

**مشتری:** ...
```

For an extension-to-extension call, replace the two example speaker labels with `داخلی اول` and `داخلی دوم` (or reliable names/roles from the audio).

Preserve segment timestamps for replay and review. Mark the document as requiring human review; never call it human-approved unless a human explicitly reviewed that version.

## Standalone completion gate

Before finishing, verify all of these:

- the source audio still exists and was not modified;
- the sibling `.md` has exactly the source stem;
- the Markdown includes `## مکالمه` and actual dialogue;
- uncertain content is marked `[نامفهوم]`;
- no intermediate audio or model JSON remains beside the source.
