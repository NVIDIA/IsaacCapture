<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hand capture

Records one take of the hand script through `TeleopSession` — both hand channels and both
controllers — and writes, into `~/isaacteleop-captures/<device>_<date>_<time>/`:

| File | What |
|---|---|
| `*-hand.mcap` | the recording |
| `*-hand.labels.json` | one window per step, on the recording's sample clock |
| `*-hand.json` | provenance: device, hand source, plugin versions, wrist source, pacing |
| `*-hand.log` | the panel's terminal output, including the label audit |

```bash
./setup_env.sh                                   # after the checker's setup_env.sh
./record.sh DEVICE [--plugin NAME] [--plugin-version NAME=VERSION] \
                   [--wrist-source SOURCE] [--note TEXT] [--accept-eula]
```

`--plugin` names the plugin as its `plugin.yaml` does (`manus_hand_plugin`,
`wuji_glove_plugin`) and is looked up in `plugins/` and `install/plugins/`. Without it the
session uses the headset's hands. `--wrist-source` defaults to `$WUJI_GLOVE_WRIST_SOURCE`,
which is recorded either way.

## The script

`session.py` is the script: label, hold seconds, spoken cue. Each step is cue, 3-2-1
(`count.wav`), window opens (`beep.wav`), hold, window closes (`tick.wav`). The script
starts once both hands have had a valid joint. A window opens on the timer whether or not
the performer is in the pose; the checker measures only the settled end of each window and
asks for a retake when a pose was not held.

After the session closes, the panel writes the sidecar and runs the checker's label
checks against the file. A `BAD` row there means a step was missed: record again.

`python make_labels.py TAKE.mcap` re-runs that audit on an existing take.

## Cues

`cues/*.wav` are rendered once with piper, voice and length scale as in
`cues/index.json`. Changing a cue's wording in `session.py` without re-rendering it stops
the panel at start-up. Re-render with:

```bash
printf '%s' "TEXT" | piper --model en_GB-alba-medium.onnx --length_scale 1.25 \
  --output_file cues/LABEL.wav
```

then update that label's `text_sha1` (the first 16 hex digits of the text's SHA-1).

## Tests

```bash
../checker/.venv/bin/python -m pytest
```

No isaacteleop or viser needed. The sidecar tests use the oracle's fixtures and skip until
`../oracle/generate.sh` has built them.
