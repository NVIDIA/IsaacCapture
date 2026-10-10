<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hand checker

Decides whether a hand or glove integration works, from one MCAP recording of the
prescribed hand script and its label sidecar. It reads the `core.HandPoseRecord` channels
of both hands, runs every check below over them and returns one verdict.

Recording the take is [`../README.md`](../README.md). This page is the reference for
the checker itself.

| Verdict | Exit status | Meaning |
|---|---|---|
| `pass` | 0 | Nothing in the recording argues against the integration. |
| `fail` | 1 | A failure attributable to the device or its plugin. |
| `retake` | 2 | The capture is unusable because of how it was performed — a step was not held. The device is not implicated. |
| `insufficient_data` | 3 | The recording cannot answer the question: too short, no hand channel, or a required check could not answer. |

`fail` and `retake` are kept apart on purpose: a performer who did not press the palms
together has not shown that the glove is broken.

The checker imports no `isaaccapture` package. It needs only `mcap` and `flatbuffers`
and the FlatBuffers schema text under `src/core/schema/fbs/`.

## Set up

```bash
cd src/python/isaaccapture/cts/hand/checker
./setup_env.sh
```

Needs `uv`, `curl` and `unzip` on a Linux or macOS host. The script builds `.venv`,
fetches **flatc v24.3.25** into `toolchain/`, generates the Python bindings into
`generated/`, and verifies that the `.bfbs` it produces is byte-identical to
`src/core/schema/golden/hand.bfbs` — the schema bytes a real recording embeds. It is
idempotent and writes nothing outside this directory.

## Run the checks

```bash
.venv/bin/python -m hand_cts.cli ~/isaaccapture-captures/<take>/<take>-hand.mcap
```

Spell the recording out. The checker reads the labels from `<recording>.labels.json` as
the path is written, so a symlink or a copy elsewhere finds none and the window checks go
unanswered.

| Option | Effect |
|---|---|
| `--json` | Machine-readable report: per-check status, severity, attribution, and every measurement. |
| `--check NAME` | Run only this check. Repeatable. |
| `--labels SIDECAR` | Use these labels instead of `RECORDING.labels.json` beside the file. |
| `--list-checks` | Print the check names with their group, severity and a one-line summary. Needs no recording. |

Labels are optional. Without them the report carries a note, the window checks report
that they could not be answered, and the other three groups are judged as usual.

## Read the report

```text
verdict  RETAKE

  [pass] rate.interval_regularity                      left: 60.0 Hz, jitter 0.24 ms; right: ...
  [meas] pinch.contact_distance                        left: pinch_index 13 mm, ...
  [FAIL] posture.fist_closure                          left: fist straightness 0.70 (needs at most 0.6); right: ...
  [n/a ] table.level                                   left: no window was held flat on the table
```

Per-hand checks run on the left and right channels separately and report both in one
line.

| Mark | What it means |
|---|---|
| `pass` | The check was answered and the answer is acceptable. |
| `FAIL` | The check was answered and the answer is not. Whether that makes the verdict `fail` or `retake` depends on whom the check blames. |
| `meas` | A number was measured and nobody judged it. Read the value; do not read it as approval. |
| `n/a ` | Not answered. Either there was not enough data, or a check this one depends on failed, which makes the measurement describe the wrong frames. It never becomes a pass. Unanswered checks are listed again at the end. |
| `note` | An advisory check failed. It is reported and does not move the verdict. |

How the verdict follows from the marks:

- `fail` — a `FAIL` on a check that blames the device.
- `retake` — no such failure, but a `FAIL` on a check that blames the performance.
- `insufficient_data` — neither, but a required check was unanswered.
- `pass` — otherwise.

The window checks are not required: a take without labels is judged on the other three
groups. Its report carries `no motion-step labels beside the recording`, which is the
sign to check the path before trusting a `pass`.

Tolerances are provisional and deliberately loose. A `pass` says nothing in the take
argued against the integration, not that the integration is good.

## View a take

```bash
./setup_env.sh --panel     # once; installs viser, the viewer's renderer
.venv/bin/python -m hand_cts.panel ~/isaaccapture-captures/<take>/<take>-hand.mcap
```

Prints the same report, then serves a viewer on `http://127.0.0.1:8080`. Options:
`--labels SIDECAR`, `--host` (`0.0.0.0` to reach it from another machine) and `--port`.

| | |
|---|---|
| **Both hands, one playhead** | Each digit in its own colour (legend in the sidebar). A joint whose `is_valid` went false is drawn **red at the position it was last seen**; tick *name held joints* to label them. |
| **The labelled step** | The step under the playhead, or `unlabelled`. |
| **Frame rate and valid joints** | Per hand: rate over the last 8 s, valid joints over the whole take, and the valid count at the playhead. |
| **Grid** | At the table height the flat steps measured, or on the stage floor when no flat step was held. |
| **Results** | The verdict banner, the checks that decide the take, then the four groups below, worst result first. |

`play` runs the take at its own rate; `speed` and the `time s` slider move the playhead.
The list always shows the **final** result of every check, because the window checks need
the whole recording.

The viewer is an optional extra. Without `--panel` the checker runs on `mcap` and
`flatbuffers` alone.

## What is checked

The four groups are what `--list-checks` prints in its group column and what the viewer
shows as its sections.

| Group | Checks |
|---|---|
| signal | schema, finite values, no valid joint at the origin, unit quaternions, timestamps, payload presence, rate regularity, minimum rate (30 Hz), gaps, joint speed |
| shape | required joints valid, stable valid set, radius, metres, bone-length constancy, proportions, joint index assignment, handedness, flex direction, optional joints |
| placement | not at the origin, wrist not frozen, quaternion component order, position and orientation in one frame |
| windows | label structure and alignment, each step performed and in order, flat hand, fist, pinch finger identity and contact, fingers coplanar and level on the table, both hands at one table height, drift between the flat windows, fingertip gap with palms pressed together, fingertip contact while each hand rolls |

The required joints are the wrist and the 20 finger joints. PALM and the four non-thumb
metacarpals are optional: a 21-joint glove passes without them and a 25-joint optical
hand without PALM, but when they are marked valid they must sit where OpenXR puts them.

Handedness, component order and flex direction are read from the OpenXR joint frame (−Z
along the bone toward the tip, +Y dorsal), so they need no threshold.

Finger flexion between open and fist is not measured; the fist check asks only that the
fingers are fully curled.

## Labels

The window checks read `<recording>.labels.json`, written by [`../capture/`](../capture/README.md)
when the take ends. Each step is one window on the recording's sample clock. A held
window is measured over its trailing 60%, since the first 40% may still hold the move into
the pose; the two roll windows are measured whole. Time between windows is reported, not
judged.

The table, the pressed palms and the touching middle fingertips are contact the
performer can feel, so the data is held to them: fingers flat on the table lie in one
level plane, both hands sit at one height, same-named fingertips meet, and a fingertip on
the axis a hand rolls about stays on the other hand's. Each of these first checks that
the pose was struck (palm down, palms facing and close, a roll of at least 90 deg) and
asks for a retake otherwise.

Left and right records are paired when their sample times lie within 8 ms; a take with no such pair leaves these checks unanswered.

## Tests

```bash
.venv/bin/python -m pytest
```

The unit tests build hands in memory. The oracle tests run over
`../oracle/fixtures_index.json` and skip until `../oracle/generate.sh` has built the
recordings; set `HAND_FIXTURES` if you built them elsewhere.
