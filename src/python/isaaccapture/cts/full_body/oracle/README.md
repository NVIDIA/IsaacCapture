<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# oracle/

Recordings whose correct verdict is already known, and the generators that build them.
This is what [the checker](../checker/README.md) is tested against: 55 synthetic MCAP
files, each one either correct or carrying exactly one injected fault, plus
`fixtures_index.json` saying what each of them should make the checker say.

## Building the set

```bash
../checker/setup_env.sh    # once per clone
./generate.sh
```

`generate.sh` writes `fixtures/`, reads every file back and checks it against the
index — about 50 seconds in total. There is no setup step and no venv of its own: the
checker's `setup_env.sh` already pins flatc v24.3.25 and proves its `.bfbs`
byte-identical to `src/core/schema/golden/full_body.bfbs`, and the fixtures embed those
same bytes.

**The recordings are not in git and the index is.** They are 67 MB of derived data that
regenerates byte-identically from the committed generators, so the index is the part
worth versioning. A generator change that moves a byte therefore shows up as a dirty
`fixtures_index.json`, which `generate.sh` refuses to leave unreviewed.

Until you have run it, the checker's oracle tests skip, each naming this script. The rest
of its suite runs from a fresh clone and needs nothing from here.

## The index is the contract

Every fixture carries an `expected_verdict`, and the four values are not
interchangeable:

| Verdict | Meaning for the checker |
|---|---|
| `pass` | Must be accepted. Includes the `benign` category — unusual but correct. |
| `fail` | CTS failure **attributable to the device or integration**. |
| `retake` | Capture unusable because of **human performance**. The device is not implicated. |
| `graded` | A known magnitude is injected. There is no threshold, so the oracle is the *measured value*, not a verdict. |

A checker that collapses `fail` and `retake` into one outcome is wrong even when it
flags the right files. Telling a vendor their device is broken when their operator
simply squatted too shallow is the specific failure this distinction exists to prevent.

Per fixture the index gives `filename`, `batch`, `category`, `expected_verdict`,
`description`, frame count and size; for defects `expected_failing_check`; for posture
fixtures additionally `kind`, `labels_sidecar` and, for the graded series, `injected` with the
exact magnitude and its unit.

## What is in the set

**Batch 1 — envelope (`fixtures/`), 27 files.** Three categories, and the third is the
point of the exercise.

*Golden — must pass.* A clean 14 s @ 60 Hz scripted sequence (A-pose still, T-pose hold,
single-arm raise, single-leg raise, return to T-pose) plus a 6 s static T-pose baseline.

*Benign — unusual but correct, must also pass.* These are the false-positive guards, and
they exist because a naively written checker fails valid output:
`benign_invalid_joints_zero_pose` (joints the vendor does not provide carry an all-zero
pose with `is_valid=false`, exactly what `make_invalid_body_joint_pose()` emits in
`src/plugins/noitom_mocap/`), `benign_backfilled_endpoints` (hands and feet derived from
wrists and ankles and marked valid, so four bones have a constantly zero length), and
`benign_dropout_recovery` (a limb goes invalid for a second and comes back). Without
this category, "does the checker catch things" and "does the checker only catch real
things" are not separable, and a checker that fails everything scores perfectly.

*Defects — exactly one injected fault each.* Geometry and frame faults (Z-up,
centimetres, left/right swap, `w,x,y,z` quaternions, mirrored handedness, positions
rotated but not orientations, permuted indices, drifting bone lengths, impossible
proportions, teleport), record and envelope faults (NaN/Inf, non-unit quaternions,
absent `joints`, non-monotonic timestamps, device clock copying the common clock,
`available` before `sample`, bulk null payloads, `all_joint_poses_tracked` contradicting
the per-joint flags) and rate faults (jitter, dropped frames, degrading validity). Each
changes exactly one thing, so a failure maps to one cause.

**Batch 2 — posture semantics (`fixtures/posture/`), 28 files.** The eleven-step script at
50 Hz for 41 s — 60 Hz would push each file past pre-commit's 2000 KiB limit. Each MCAP
has a `<name>.labels.json` beside it giving every step's window in the same clock domain
as the record timestamps. The sidecar is a separate file: the recording has no
annotation channel.

The posture fixtures are tagged `kind`, because two different things can be wrong:
`device_fault` (good performance, broken device — march ankles in phase, squat reps not
matching, asymmetric knees through an even squat, an arm raise that saturates) and
`performance_fault` (working device, bad execution — a squat too shallow to measure, a
slow irregular march, leaning while raising an arm, a step skipped, steps out of order).
`posture_verify.py` prints, for each performance fixture, the measurement that keeps it
distinguishable from the corresponding device fault.

Four quantities come as graded series with the magnitude recorded numerically: cumulative
drift (0, 0.02, 0.05, 0.15 rad), contralateral cross-talk (0, 2, 10, 25°), T-pose arm droop
(0, 5, 15, 30°) and T-pose left/right asymmetry (0, 3, 10°). Assert that the measured value
tracks the injected one — accurate and monotonic — not pass or fail.

## On the wire

Pinned to `src/core/mcap/cpp/inc/mcap/tracker_channels.hpp` and
`src/core/deviceio_session/cpp/deviceio_session.cpp`, so a checker cannot tell these
apart from a real recording by envelope:

| | |
|---|---|
| MCAP schema name | `core.FullBodyPoseRecord` |
| schema encoding / data | `flatbuffer` / the `.bfbs` bytes |
| topic | `full_body/full_body` |
| message encoding | `flatbuffer` |
| `logTime` = `publishTime` | `available_time_local_common_clock` |
| `sequence` | increments from 0 |
| profile / compression | `teleop` / none |

Conventions in the data: right-handed, **Y-up**, metres, quaternions **(x, y, z, w)**,
subject facing **−Z** so `RIGHT_*` joints sit at +X. Joint indices come from the
`BodyJoint` enum in `full_body.fbs`; each joint's parent comes from the 24-joint table in
`docs/source/device/body_tracking.rst`.

## Files

```text
skeleton.py           joint layout, FK, quaternion maths, envelope-batch animation
posture_script.py      the eleven-step motion script + injectable magnitudes
mcap_io.py            FlatBuffer encode + MCAP write, pinned to the C++ writer
toolchain.py          where the checker's flatc output is
generate_fixtures.py  envelope-batch registry; drives both batches
posture_fixtures.py    posture-batch registry
verify_fixtures.py    reads the envelope batch back and asserts every claim
posture_verify.py      measures the posture batch back against the injected truth
fixtures_index.json   the oracle
```

## Adding a fixture

Regenerate, never hand-edit: the files are derived, and a defect added as a
transformation composes with everything else. Envelope batch, in
`generate_fixtures.py`:

```python
@fixture("defect_my_fault", DEFECT, "one line", verdict="fail", check="group.check_name")
def _my_fault(rec):
    for f, i, j in each_joint(rec):
        ...            # mutate in place
```

Posture batch: add a field with a numeric magnitude to `PostureParams` in
`posture_script.py`, use it in `posture_local_rotations`, and add one
`PostureFixture(...)` entry in `posture_fixtures.py`. A
graded series is a loop over values, not a set of files.

Read [`AGENTS.md`](AGENTS.md) before changing a generator. It records what this set can
and cannot establish, which is the part that is easy to get wrong.
