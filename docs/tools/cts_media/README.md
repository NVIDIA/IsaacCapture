<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CTS docs media

Generates the step GIFs and the hand rehearsal track used by the Device Conformance
pages (`docs/source/conformance/`). The outputs are committed; nothing here runs in CI.
Re-run it, and commit the outputs with the change, when the hand `capture/session.py`
(`STEPS`, `COUNTDOWN_BEATS`, `BEAT_S`), the hand `capture_panel.py` `TAIL_S`, the hand
cue WAVs or the oracle's motion changes.

| Output | Made by |
|---|---|
| `docs/source/_static/cts/hand/NN-<label>.gif` (960×600, 10 fps) | `build_pages.py` → `capture.mjs` → `encode.py` |
| `docs/source/conformance/hand-rehearsal.mp3` | `rehearsal_audio.py` |

Not generated here, so update them by hand:

- `hand-checker-panel.png`: a screenshot (1400×910) of the checker viewer on the
  oracle's golden take. Retake it when the viewer's sidebar changes.
- `hand-pacing.svg`, `verdicts.svg`: hand-drawn. Edit them when `STEPS` durations, the
  pacing or the verdict rules change.

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
python3 docs/tools/cts_media/encode.py "$OUT/hand" docs/source/_static/cts/hand --stats full
python3 docs/tools/cts_media/rehearsal_audio.py docs/source/conformance/hand-rehearsal.mp3
```

Stage the results with a plain `git add`: GIFs and MP3s go through Git LFS.

## What the GIFs show

- The motion is the oracle's synthetic reference take
  (`src/python/isaaccapture/cts/hand/oracle/`), with sway, tremor and noise switched
  off in `build_pages.py` only. Steps, order and durations come from the suite's
  `capture/session.py`; file names follow that order.
- `build_pages.py` also redraws two hand poses to match the cues, without touching the
  oracle: flat on the table with the fingers together and the thumb beside the index,
  and, for palms together, each thumb straight beside its own index, clear of the seam.
- From the oracle's poses and placements, holds play at real speed: the flats are one
  still frame; the pinches close from an open hand and the fingertips come together
  from ~13 cm apart, each over 1 s, then hold; fist and palms hold while the camera
  circles the fists once and swings ±70° from your view of the palms. The rolls are the
  oracle's take: 1.2 s still in the pose, then the turn from beep to tick.
- Each GIF holds its last frame for 0.6 s, then loops. The canvas font is pinned to
  Helvetica Neue / Helvetica / Arial.
- Keep the suite under 3 MB (3,000,000 bytes) and each GIF under 1 MB; `encode.py`
  prints the sizes.

`hand.html` is a render-only template: `build_pages.py` fills in the data, `capture.mjs`
calls its `setup()` and `render(step, frame)`.
