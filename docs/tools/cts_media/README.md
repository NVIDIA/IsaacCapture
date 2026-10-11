<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CTS docs media

Generates the step GIFs and the hand rehearsal track used by the Device Conformance
pages (`docs/source/conformance/`). The outputs are committed; nothing here runs in CI.
Re-run it, and commit the outputs with the change, when a suite's `capture/session.py`
(`STEPS`, `COUNTDOWN_BEATS`, `BEAT_S`), the hand `capture_panel.py` `TAIL_S`, the hand
cue WAVs or an oracle's motion changes.

| Output | Made by |
|---|---|
| `docs/source/_static/cts/hand/NN-<label>.gif` (960×600, 10 fps) | `build_pages.py` → `capture.mjs` → `encode.py` |
| `docs/source/_static/cts/full_body/NN-<label>.gif` (640×640, 12.5 fps) | same |
| `docs/source/conformance/hand-rehearsal.mp3` | `rehearsal_audio.py` |

## Prerequisites

The committed files were made on macOS with these versions; other versions can shift
pixels and bytes.

- Python 3.11 or newer, standard library only.
- Node.js v26.8.1 (any release with built-in `fetch` and `WebSocket`; no npm packages).
- Google Chrome 155.0.8059.40. `capture.mjs` looks in `/Applications`; set `CHROME` to
  the binary elsewhere.
- ffmpeg 9.0.1 with `libmp3lame`.
- Git LFS, so `cts/hand/capture/cues/*.wav` are audio rather than pointer files.

## Regenerate

From the repository root:

```bash
OUT=$(mktemp -d)
python3 docs/tools/cts_media/build_pages.py --out "$OUT"
node docs/tools/cts_media/capture.mjs "$OUT/hand.html" "$OUT/hand" 480 300 2
node docs/tools/cts_media/capture.mjs "$OUT/full_body.html" "$OUT/full_body" 320 320 2
python3 docs/tools/cts_media/encode.py "$OUT/hand" docs/source/_static/cts/hand
python3 docs/tools/cts_media/encode.py "$OUT/full_body" docs/source/_static/cts/full_body
python3 docs/tools/cts_media/rehearsal_audio.py docs/source/conformance/hand-rehearsal.mp3
```

Stage the results with a plain `git add`: GIFs and MP3s go through Git LFS.

## What the GIFs show

- The motion is each oracle's synthetic reference take
  (`src/python/isaaccapture/cts/{hand,full_body}/oracle/`), with sway, tremor and noise
  switched off in `build_pages.py` only. Steps, order and durations come from each
  suite's `capture/session.py`; file names follow that order.
- `build_pages.py` also redraws two hand poses to match the cues, without touching the
  oracle: flat on the table with the fingers together and the thumb beside the index,
  and both thumbs closed for palms together.
- Hand: a 1.2 s move into the pose, then the hold from beep to tick at real speed.
- Full body: the oracle's windows are shorter than the real ones, so each window is
  resampled to play for the session's duration. Still steps open with a short
  "getting into the pose" lead-in, which a performer does before pressing.
- Each GIF holds its last frame for 0.6 s, then loops. The canvas font is pinned to
  Helvetica Neue / Helvetica / Arial.
- Keep each suite under 2.5 MB and each GIF under 1 MB; `encode.py` prints the sizes.

`hand.html` and `full_body.html` are render-only templates: `build_pages.py` fills in the
data, `capture.mjs` calls their `setup()` and `render(step, frame)`.
