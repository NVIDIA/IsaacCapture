<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hand capture

Reference for this directory. The step-by-step guide — setup, headset, recording, what to
do when it goes wrong — is [`../README.md`](../README.md).

`record.sh` records one take of the hand script through `TeleopSession` — both hand
channels and both controllers — into `~/isaaccapture-captures/<device>_<date>_<time>/`.
The files repeat the directory's name:

| File | What |
|---|---|
| `*-hand.mcap` | the recording |
| `*-hand.labels.json` | one window per step, on the recording's sample clock |
| `*-hand.json` | provenance: device, hand source, plugin versions, wrist source, pacing, notes, commit |
| `*-hand.log` | the panel's terminal output, including the label audit |

## Commands

```bash
./setup_env.sh                 # after ../checker/setup_env.sh
./record.sh DEVICE [options]   # options are capture_panel.py's, below
.venv/bin/python make_labels.py TAKE.mcap   # re-run the label audit on an existing take
```

`DEVICE` is required, comes first, and only names the output directory.

| Option | Effect |
|---|---|
| `--plugin NAME` | Launch a hand plugin, repeatable. `NAME` is the plugin's name in its `plugin.yaml` (`manus_hand_plugin`, `wuji_glove_plugin`), looked up in `plugins/` and `install/plugins/`. Without it the take uses the headset's hands. |
| `--plugin-version NAME=VERSION` | Record this version instead of the one in `plugin.yaml`. |
| `--wrist-source SOURCE` | Record where a glove's wrist came from, e.g. `hand_tracking` or `controller`. Defaults to `$WUJI_GLOVE_WRIST_SOURCE`. It labels the take; the plugin reads the variable itself. |
| `--note TEXT` | Free text kept in the provenance file. |
| `--host HOST` / `--port PORT` | Where the panel listens; `0.0.0.0` and `8081` by default. A busy port moves to the next free one. |
| CloudXR flags | `--accept-eula`, `--cloudxr-install-dir`, `--cloudxr-device-profile`, `--cloudxr-env-config`, `--[no-]launch-cloudxr-runtime`, `--[no-]host-client`, `--[no-]launch-wss-proxy` — the same set the full-body capture takes. |

## The script

[`session.py`](session.py) is the script. The performer sits at a table. Each step is
the spoken cue, a 3-2-1 count (`count.wav`), the beep that opens the window (`beep.wav`),
the hold, and the tick that closes it (`tick.wav`). The move into each pose happens
during the cue and the count, so the hold covers the pose only. The script starts once
both hands have had a valid joint.

| # | Label | Hold | Cue |
|---|---|---|---|
| 1 | `flat_on_table_open` | 4 s | Rest both hands flat on the table, palms down, fingers together. |
| 2 | `fist` | 4 s | Make two tight fists. |
| 3 | `pinch_index` | 3 s | Thumb to index fingertip. |
| 4 | `pinch_middle` | 3 s | Thumb to middle fingertip. |
| 5 | `pinch_ring` | 3 s | Thumb to ring fingertip. |
| 6 | `pinch_little` | 3 s | Thumb to little fingertip. |
| 7 | `palms_together` | 4 s | Press your palms together, fingers up. |
| 8 | `tips_together` | 3 s | Raise both hands, palms facing you. Touch middle fingertips. |
| 9 | `right_tip_roll` | 6 s | Fingertips touching. At the tone, turn the right palm out, then back. |
| 10 | `left_tip_roll` | 6 s | Fingertips touching. At the tone, turn the left palm out, then back. |
| 11 | `flat_on_table_close` | 4 s | Hands flat on the table again. |

A window opens on the timer whether or not the performer is in the pose. The checker
measures the settled end of each held window and asks for a retake when a pose was not
held.

When the script ends the recording closes and `make_labels.py` writes the sidecar and
audits it against the file: both hand channels present, every window inside the recorded
span, and the checker's own "was the step performed" result. A `BAD` row there means a
step was missed.

## Cues

`cues/*.wav` are pre-rendered speech and three tones, played with `aplay`; nothing is
synthesised at run time. `cues/index.json` holds a hash of each cue's wording, and the
panel exits at start-up when a cue in `session.py` no longer matches its WAV. The script
is part of the specification: do not edit it.

A WAV that is a Git LFS pointer (a clone without git-lfs) stops the panel with a message
to run `git lfs pull`.

To re-render a cue after changing its wording, use the voice and length scale recorded in
`cues/index.json` (`en_GB-alba-medium`, `1.25`) and update that cue's `text_sha1`:

```bash
printf '%s' "TEXT" | piper --model en_GB-alba-medium.onnx --length_scale 1.25 \
  --output_file cues/LABEL.wav
printf '%s' "TEXT" | sha1sum | cut -c1-16     # the new text_sha1
```

## Tests

```bash
../checker/.venv/bin/python -m pytest
```

No isaaccapture or viser needed. The sidecar tests use the oracle's fixtures and skip
until `../oracle/generate.sh` has built them.
