<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Full-body device CTS checker

Decides whether a third-party full-body motion-capture integration works, from one MCAP
recording of a prescribed motion script and nothing else. It runs 38 checks over the
recording and returns one verdict:

| Verdict | Exit status | Meaning |
|---|---|---|
| `pass` | 0 | Nothing in the recording argues against the integration. |
| `fail` | 1 | A CTS failure attributable to the device or its plugin. |
| `retake` | 2 | The capture is unusable because of how it was performed. The device is not implicated. |
| `insufficient_data` | 3 | The recording cannot answer the question — too short, no skeleton in it, or a prerequisite check failed. |

`fail` and `retake` are deliberately separate. Telling a vendor their device is broken
when the operator simply squatted too shallow is the specific mistake this process exists
to avoid.

The checker imports no `isaaccapture` package; the only thing it reads from the tree is
the FlatBuffers schema text under `src/core/schema/fbs/`. Rules for changing it are in
[`AGENTS.md`](AGENTS.md).

## Set up

```bash
cd src/python/isaaccapture/cts/full_body/checker
./setup_env.sh
```

Needs `uv`, `curl` and `unzip` on a Linux or macOS host. The script builds `.venv`,
fetches **flatc v24.3.25** into `toolchain/`, generates the Python bindings into
`generated/`, and verifies that the `.bfbs` it produces is byte-identical to
`src/core/schema/golden/full_body.bfbs` — the schema bytes a real recording embeds. Run
it again after the schema changes; it is idempotent and writes nothing outside this
directory.

## Record a take

[`../README.md`](../README.md) is the step-by-step: the setup, the headset, the ten
steps, the four files a take leaves behind, and what to do when it goes wrong. Recording
is **Linux only** — it drives the device through Isaac Teleop — while everything above
and below this section runs on macOS too.

The label sidecar (`<recording>.labels.json`) carries the motion windows the posture
checks read. The capture panel writes it once the file is closed: each window is resolved
against `sample_time_local_common_clock` in the finished recording. Windows do not tile —
the performer moving between poses falls outside all of them — and each records whether a
trigger or the keyboard opened it. The labels are a separate file, not a channel in
the MCAP.

The trigger and the keyboard are equivalent: with two people the one at the screen does
the pressing, and a mocap suit has no controllers at all.

If the client reports `body_tracking: false`, every joint arrives invalid and only the
container and envelope checks can run. On PICO that is a browser limitation, not a
hardware or licensing one; see [`../README.md`](../README.md).

## Run the checks

```bash
.venv/bin/python -m full_body_cts.cli \
    ~/isaaccapture-captures/<take>/<take>-body.mcap
```

Spell the recording out. Do not use a symlink or a copy of the `.mcap` elsewhere: it has
no `.labels.json` beside *it*, so every posture check goes unanswered and the verdict can read
`pass` instead of `retake`.

| Option | Effect |
|---|---|
| `--json` | Machine-readable report: per-check status, severity, attribution, and every measurement. |
| `--check NAME` | Run only this check. Repeatable. |
| `--labels SIDECAR` | Use these motion labels instead of `RECORDING.labels.json` beside the file. |
| `--list-checks` | Print the check names with their group, severity and a one-line summary. The one option that needs no recording, so it answers *what does this thing look for* before a take exists. |

Labels are optional. Without them the posture window measurements report that they could not
be answered, and the envelope and geometry checks still run.

## Read the report

```text
verdict  RETAKE

  [pass] rate.interval_regularity                 56.1 Hz, jitter 0.14 ms
  [meas] posture.tpose_arm_droop                  arms -3.8 deg below horizontal ...
  [FAIL] posture.squat_knee_symmetry              the knees differ by 21.4 deg ...
  [n/a ] posture.march_ankle_antiphase            not judged: ... failed, so this ...
```

| Mark | What it means |
|---|---|
| `pass` | The check was answered and the answer is acceptable. |
| `FAIL` | The check was answered and the answer is not. Whether that makes the verdict `fail` or `retake` depends on whom the check blames. |
| `meas` | A number was measured and **nobody judged it.** These checks have no threshold. Read the value; do not read it as approval. |
| `n/a ` | Not answered. Either there was not enough data, or a check this one depends on failed, which makes the measurement describe the wrong frames. A suppressed check never becomes a pass. |
| `note` | An advisory check failed. Reported, but it does not move the verdict — these are cases where a hard failure would reject legitimate hardware. |

## View a take

```bash
./setup_env.sh --panel     # once; installs viser, the panel's renderer
.venv/bin/python -m full_body_cts.panel \
    ~/isaaccapture-captures/<take>/<take>-body.mcap
```

Prints the same report, then serves a panel on `http://127.0.0.1:8080`. It takes
`--labels`, and `--host` / `--port` for reaching it from another machine. It shows the
things a list of 38 lines cannot:

| | |
|---|---|
| **Skeleton, coloured per joint** | A live joint is green. A joint whose `is_valid` went false is drawn **red at the position it was last seen**, and named in the scene — its recorded position is arbitrary, so it is not drawn where the file says. This is the view in which a hand dropping out as the subject turns is obvious. |
| **Frame rate, scrolling** | A dropped block is a spike here. In the text report it is a slightly lower mean. |
| **Valid joints over the whole take** | A decay is one glance, rather than a start-and-end pair of percentages. |
| **Three groups of checks** | Named after what they read — the envelope and signal, the skeleton's geometry, the posture over the labelled windows. Each carries its own answer before you expand it, worst result first, with the few checks that decide the take pulled out above them. |

`play` runs the take at its own rate; `speed` and the `frame` scrubber move the
playhead. The list always shows the **final** result of every check: three of them need
the whole recording by construction, so a result taken mid-playback would be a number
with no meaning.

viser is an optional extra, and only the panel's renderer uses it. Without `--panel` the
checker runs on `mcap` and `flatbuffers` alone.

## Send a take

**`package for submission`** in the panel builds one archive and downloads it through the
browser — so it lands on the machine you will send it from, even when the panel is being
viewed over `--host`. Right-click the link it offers for *Save as…* to choose where.

```text
pico4u_2026-03-04_101530-body.retake.zip           the take and the verdict, so a mailbox
└── pico4u_2026-03-04_101530-body.retake/          of these can be triaged unopened
    ├── report.json                                verdict, checks, groups, per-frame
    │                                              series, input hashes, tool commit
    ├── report.txt                                 the same report, for reading
    ├── pico4u_2026-03-04_101530-body.mcap         the recording, byte for byte
    ├── pico4u_2026-03-04_101530-body.labels.json  motion-step windows
    ├── pico4u_2026-03-04_101530-body.json         capture provenance
    └── pico4u_2026-03-04_101530-body.log          recorder log
```

The name is the take's own, which already carries the device and the moment it was
recorded, so two devices recording at once cannot collide.

`report.json` is a superset of `--json`
and reads out of the archive without decompressing the recording, so a dashboard needs
only that member. Each check in it carries its **`mark`** and its group, so nothing
reading the file has to reimplement how a result is displayed or how the verdict is
reached.

The recording and its sidecars are the evidence; the packed report is a convenience. The
checker is deterministic, so re-running it on the archive reproduces the verdict exactly
— at the same commit, which is why `tool.commit` is in there. **The labels sidecar cannot
be regenerated** from the recording: without it every posture check reports unanswered and the
verdict changes, which is why the archive holds more than one file. A companion that was
never there is not an error; `inputs.missing` names it.

## Check groups

Every check carries a group, which `--json` reports and `--list-checks` prints.

| Group | What the checks cover |
|---|---|
| `signal` | Schema and envelope conformance — the right schema and channel, timestamps, finite values, unit quaternions, a pose on every record — and signal quality: frame rate and jitter, dropouts, validity trend, plausible joint speed. |
| `geometry` | Skeleton geometry — up axis, metres, handedness, bone-length constancy, human proportions, left/right labelling, joint indexing, quaternion component order, position/orientation agreement. |
| `posture` | Posture semantics over the motion windows — was each step performed, in order, and does each joint angle read what the pose implies. |

Not checked from a recording: plugin build and skip behaviour, replay through
retargeting, and human review of the session video.

## Tests

```bash
.venv/bin/python -m pytest
```

Two layers. The unit layer feeds accumulators frames built in memory and runs from a
fresh clone. The oracle layer is parametrised over
[`../oracle/`](../oracle)`fixtures_index.json` and skips until
[`../oracle/generate.sh`](../oracle/generate.sh) has built the recordings that index
describes; point `FULLBODY_FIXTURES` at them if you built them somewhere else.

## Layout

```text
src/full_body_cts/
  frames.py        Frame / JointPose for core.FullBodyPoseRecord
  mcap_source.py   decodes the full-body channel
  labels.py        this script's still windows on the shared sidecar reader
  profile.py       skeleton profile: topology, symmetry, priors, which checks apply
  checks/          one module per check group; envelope.py puts the shared ones in `signal`
  report.py        run(): the checks over one source, plus full-body notes
  cli.py           python -m full_body_cts.cli
  panel/           the viewer; app.py is the only module here that imports viser
../../common/src/cts_common/
                   shared with every schema: file-order MCAP reading, the accumulator
                   contract, the envelope checks, verdicts and Mark, the sidecar reader
tests/
  synth.py               builds MCAPs in memory
  known_deviations.py    where this checker knowingly disagrees with the fixture index
```
