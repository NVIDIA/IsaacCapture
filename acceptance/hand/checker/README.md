<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hand checker

```bash
./setup_env.sh
.venv/bin/python -m hand_acceptance.cli TAKE.mcap [--json] [--check NAME] [--labels SIDECAR]
.venv/bin/python -m hand_acceptance.cli --list-checks
```

Exit status: 0 `pass`, 1 `fail`, 2 `retake`, 3 `insufficient_data`.

## Verdict

- `fail` — a hard check attributed to the device failed.
- `retake` — the recording is fine but a step was not performed; record again.
- `insufficient_data` — a required check could not answer, e.g. too few frames.
- `pass` — otherwise.

Window checks are not required: a take without labels is judged on the other three
groups, and the report's note says the windows went unanswered.

`[meas]` rows are measurements with no threshold yet; they never move the verdict.
Advisory rows are reported and never move it either.

## What is checked

Each per-hand check runs on the left and right channels separately and reports both.

| Group | Checks |
|---|---|
| signal | schema, finite values, unit quaternions, timestamps, payload presence, rate, gaps, joint speed |
| shape | required joints valid, stable valid set, radius, metres, bone-length constancy, proportions, joint index assignment, handedness, flex direction, optional joints |
| placement | not at the origin, wrist not frozen, quaternion component order, position and orientation in one frame |
| windows | label structure and alignment, each step performed and in order, open hand, fist, pinch finger identity and contact, drift between open-hand windows, grasp aperture |

The required joints are the wrist and the 20 finger joints. PALM and the four
non-thumb metacarpals are optional: a 21-joint glove passes without them and a
25-joint optical hand without PALM, but when they are marked valid they must sit where
OpenXR puts them.

Handedness, component order and flex direction are read from the OpenXR joint frame
(−Z along the bone toward the tip, +Y dorsal), so they need no threshold.

## Labels

Window checks read `TAKE.labels.json`, written by `../capture/`. Each window is measured
over its trailing 60%; the first 40% may still hold the move into the pose.

## Tests

```bash
.venv/bin/python -m pytest
```

Unit tests build hands in memory (`tests/synth.py`). Oracle tests run over
`../oracle/fixtures_index.json` and skip until `../oracle/generate.sh` has built the set;
`HAND_FIXTURES` overrides where it is looked for.
