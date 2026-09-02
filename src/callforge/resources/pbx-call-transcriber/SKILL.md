---
name: pbx-call-transcriber
description: Transcribe Persian PBX MP3 call recordings locally with Whisper, improve quiet telephone audio, compare independent passes, reconstruct speaker turns, and save a reviewed same-name Markdown transcript beside the source. Use for PBX recordings whose names commonly begin with external-, internal-, or out-. Do not use for general audio editing or summaries without transcription.
---

# PBX Call Transcriber

Transcribe the requested recording locally. The only durable output is a reviewed Markdown file next to the source MP3, with the same stem. Never replace an existing transcript without first reading it and improving it.

## Privacy and accuracy rules

- Run speech recognition locally. Do not upload the recording to a third-party transcription API.
- Never invent a word, number, name, or speaker identity. Write `[نامفهوم]` for unresolved audio.
- Interpret filename prefixes exactly: `external-` is an inbound call, `out-` is an outbound call, and `internal-` is an extension-to-extension call.
- For `external-`, the first field is the support extension and the second is the remote number. For `out-`, the first field is the remote number and the second is the support extension. For `internal-`, the first and second fields are the two extensions.
- For inbound and outbound calls, use functional speaker labels `کارشناس پشتیبانی` and `مشتری`. For extension-to-extension calls, use `داخلی اول` and `داخلی دوم` unless names or roles are clear from the audio. Infer turns from the conversation, filename direction, audio timing, and independent Whisper passes.
- Do not summarize or omit greetings, repeated phrases, hesitations that affect meaning, amounts, identifiers, or closing remarks.

## Runtime

Use the interpreter in `$CALLFORGE_PYTHON` when set; otherwise use `python3` (or `python` on Windows). Helper scripts are under this skill's `scripts` directory.

The helper automatically selects MLX Whisper on Apple Silicon and faster-whisper elsewhere. Use the turbo model first and the full large-v3 model for uncertain regions.

## Workflow

1. Resolve the absolute MP3 path and the exact sibling Markdown path.
2. Inspect duration and filename metadata. Reject a missing or empty source.
3. Create an isolated temporary working directory. Do not place intermediate WAV or JSON files beside the source.
4. Run `scripts/prepare_audio.py SOURCE --output-dir TEMP`. It creates a conservative 16 kHz mono decode and an AGC copy and reports signal statistics.
5. Run at least two independent transcription passes with `scripts/transcribe_audio.py`:
   - turbo model on the raw 16 kHz WAV;
   - turbo model on the AGC WAV.
6. Compare segment timing and wording. For disagreements, low-confidence segments, names, phone numbers, money, or identifiers, rerun the relevant interval with the full model on both raw and AGC audio.
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

Add timestamps only when they help resolve an uncertain passage. The transcript itself, not a prose report about the work, is the required artifact.

## Completion gate

Before finishing, verify all of these:

- the MP3 still exists and was not modified;
- the sibling `.md` has exactly the source stem;
- the Markdown includes `## مکالمه` and actual dialogue;
- uncertain content is marked `[نامفهوم]`;
- no intermediate audio or model JSON remains beside the source.
