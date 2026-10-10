<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `cts/full_body/checker/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../../../../../AGENTS.md`](../../../../../../AGENTS.md)
before editing here. Read [`../oracle/AGENTS.md`](../oracle/AGENTS.md) too if you touch
anything the fixture set is the oracle for, and [`../../common/AGENTS.md`](../../common/AGENTS.md)
for the shared `cts_common` code. [`README.md`](README.md) is the operator's guide.

Synthetic fixtures prove a measurement is correct; they cannot say what a real recording
looks like. Do not assume the recording is the performance (it starts before and ends
after), that the performer can reach the textbook pose, that every bone length is
measured (on a three-tracker rig the arms are solved), or that a squat is symmetric.

## Check rules

- **`segmentation.label_alignment` requires containment, not coincidence.** Every window
  must be backed by frames and lie inside the recording (`MAX_UNCOVERED_FRACTION` in
  `cts_common/checks/segmentation.py`); real takes carry lead-in and tail, which are
  reported, not judged. It gates the window checks, so tightening it suppresses them all.
  A sidecar shifted but still inside the recording is caught by
  `segmentation.labelled_step_actually_performed` and `segmentation.step_order_matches_labels`.
- **`segmentation.label_windows_wellformed` does not require windows to tile.** The move
  between poses belongs to no window; that time is the measurement `unlabelled_between_s`.
  An overlap is a defect. Do not restore a partition requirement.
- **`posture.arm_raise_range_of_motion` attributes to PERFORMANCE.** One recording cannot
  separate a clipped arm from an arm that never went up, and the share of samples near the
  peak does not either (24% and 23% on the saturating and clean fixtures). A device that
  truly clips fails every retake.
- **The arm raise is the elevation of the shoulder-to-hand line**, never the shoulder
  joint angle, which is solver output without shoulder or elbow tracking.
  `MIN_ELEVATION_DEG = 60.0` sits between the saturating fixture (25 deg) and a clean one (89 deg).
- **`judged = False` checks report a number, never a verdict**:
  `posture.tpose_arm_droop`, `posture.tpose_left_right_asymmetry`,
  `posture.cumulative_drift_between_tpose_windows`,
  `posture.contralateral_crosstalk_single_leg_raise`. They return `PASS` only because
  there is no other way to say "measured". Add a threshold only from real subjects.
- **A varying bone is solved, not broken, only on an otherwise rigid skeleton.**
  `skeleton.bone_length_constancy` excuses it only when at least `MIN_RIGID_FRACTION` of
  measured bones are fixed (`RIGID_CV`); `skeleton.anthropometric_plausibility` drops a
  solved forearm from its proportion prior. Never excuse a skeleton with no fixed bones
  (`defect_bone_length_drift`). Constantly-zero bones (back-filled hands and feet marked
  valid) are derived, never a fault.
- **`posture.squat_knee_symmetry` attributes through `Outcome.attribution`.** A person
  favouring a leg bends the whole leg chain; a device mis-estimating knees leaves hips and
  ankles alone. `CHAIN_ASYMMETRY_DEG = 5.0` sits above the 0 deg the device-fault fixture
  leaves in hips and ankles beside 22 deg of knee difference. A check whose evidence can
  name the culprit must say which one it saw.

## The model

A check is an incremental accumulator (`update(frame)` / `result()`), so one implementation
serves an MCAP file and a live or replay session; `result()` may return `INSUFFICIENT_DATA`
at any time. The accumulator reports status and measurements; the class declares policy:
`severity` (`HARD`, `SOFT`, `ADVISORY` — advisory never moves the verdict), `attribution`
(`DEVICE` or `PERFORMANCE`, overridable per result by `Outcome`), `judged`, `required`
(`False` when the question is about what the subject did, so an unanswered one does not
force `insufficient_data`) and `depends_on`. `suppress_dependents` iterates to a fixed
point; suppression yields "cannot conclude", never a pass.

Verdict: a counted DEVICE failure gives `fail`, else a counted PERFORMANCE failure gives
`retake`, else an unanswered required check (or no answered check) gives
`insufficient_data`, else `pass`. Never collapse `fail` and `retake`. Thresholds are named
constants on the check class.

- **`Mark` in `cts_common/report.py` is the single definition of how a result is shown**,
  derived by `CheckResult.mark`. A new state goes there; a renderer that recomputes it drifts.
- **Every JSON this package writes must survive a strict parser.** Measurements can be
  non-finite (`inf`): use `report.json_safe` and `allow_nan=False` in any new writer.

## Panel

- **viser stays in `panel/app.py`** and `cts_common/panel/skeleton.py`; it is an optional
  extra. `tests/test_panel_boundary.py` asserts nothing else imports it and the checker
  imports without it.
- **Skeleton topology comes from `profile.FULL_BODY.bones()`**, not `examples/mcap_record_replay`'s
  `BODY_BONES`, which imports the `TeleopSession` pipeline.
- **Write the panel to `FrameSource`, not MCAP**; `panel/track.TeeSource` builds the track
  in the same pass as the checks.
- **Draw an invalid joint red at its last valid position**, never where the record says:
  invalid joints carry arbitrary values (`Sample.positions` drops them, `Sample.valid`
  says which is which).
- **Show only final results**: three checks need the whole recording (`coverage.validity_trend`,
  `posture.cumulative_drift_between_tpose_windows`, the window checks).
- **Do not substitute a local clock** for a missing sample time (`_stamp` in
  `cts_common/panel/track.py`); it measures delivery jitter, not the device. Live frames
  carry no `DeviceDataTimestamp`, so time-dependent checks stay unanswerable there.
- **`panel/bundle.py` holds no viser.** The archive carries the inputs that cannot be
  regenerated, above all the labels sidecar. Keep `inputs` (file hashes, labels,
  `tool.commit`) apart from outputs; the packed report is a cache and the recording wins.
  A missing companion goes in `inputs.missing`, not an error. It is a zip so
  `report.json` reads without decompressing the recording.

## Architecture

Modules are organised by what input they need, not by group: envelope (the Record
wrapper only), payload (one descriptor), geometry (a skeleton profile), window (a label
sidecar).

- **Check groups are named, not numbered:** `signal`, `geometry`, `posture`.
- **Check names come from `../oracle/fixtures_index.json`** (each defect fixture's
  `expected_failing_check`); do not rename a check without the index.
- **Locate the channel by declared schema name** (`core.FullBodyPoseRecord`), not by topic.
  Profile string and compression are informational.
- **Read in file order** (`cts_common/mcap_reader.py`), never `mcap.reader.make_reader()`,
  which re-sorts by log time and repairs a non-monotonic-timestamp recording.
- **Topology, symmetry pairs, proportion priors and which checks apply come from the
  profile** (`profile.py`). A profile belongs to a schema, never a device: where devices
  differ, write a rule every device is judged by, never a per-device table.
- **`cts/` is not part of the wheel.** `src/python/CMakeLists.txt` filters it out and
  `pyproject.toml` excludes it; keep both when moving it.

## Hard constraints

1. **The checker imports no `isaaccapture`**; `mcap` and `flatbuffers` are the whole
   dependency. `../capture/` records through `TeleopSession`, so it imports the built
   package and the checker's `Frame`, `TrackBuilder` and profile — never the reverse.
   `tests/test_boundaries.py` asserts it.
2. **`../oracle/` shares no code with the checker**, so a fault cannot cancel itself out
   across the two. The duplicated joint table and quaternion arithmetic are deliberate.
3. **`fixtures_index.json` is never edited to make a test pass.** Declare a disagreement
   in `tests/known_deviations.py` with its reason (`EXPECTED_COLLATERAL` there lists
   where one defect legitimately trips several checks).
4. **New test data belongs here, not in the oracle**: the generator rewrites the index as
   a side effect, so build cases this checker needs with `tests/synth.py` in a temp directory.
5. **Thresholds do not come from fixtures**, which are smooth and caricatured. Real
   subjects choose a number.

## Established constraints — do not re-derive

- **Chirality and left/right labelling derive forward from the ankle-to-foot vectors, not
  the pelvis quaternion**, so an unrelated orientation fault does not fail both checks.
  Trust feet only when both are present, plausibly long against the torso, and within 60°
  of each other (a foot zeroed to the origin passes a bare length test); otherwise fall
  back to the pelvis and record which reference was used.
- **`skeleton.joint_index_assignment` measures how much of a bone's length moves back
  toward the pelvis**, not raw distance-to-root (an A-pose puts every elbow nearer the
  pelvis than its shoulder). `REVERSED = 0.8` sits in an empty gap: a reversed bone reads
  +1, nothing correctly indexed exceeds −0.47.
- **Joint angles are parent-relative, not world coordinates**, so a leaning operator is
  not read as a broken device. The arm raise is the exception;
  `performance.arm_raise_torso_stability` catches the lean.
- **"Stature" is the head *joint* above the lowest joint, about 1.57 m**, not anatomical
  stature (~1.72 m); a proportion prior must say which it tests. `stature_chain` sums
  bone lengths along one leg and the spine, so no posture moves it.
- **Decide whether a step was performed from peak joint angle, not path length**: position
  noise random-walks to ~0.33 m per window, ~12% of the real signal.
- **Measure held poses over the trailing part of the window** (`SETTLE_FRACTION` in
  `checks/posture.py`); the script blends into each pose.
- **flatc v24.3.25 with the `GenerateFlatBuffers.cmake` flags must produce a `.bfbs`
  byte-identical to `src/core/schema/golden/full_body.bfbs`** (checked by `setup_env.sh`);
  a distro or Homebrew flatc (25.x) does not match. The Python `flatbuffers` package has
  no reflection module: decode with the generated bindings, use the `.bfbs` only for
  byte comparison, so payload checks are per-schema.
- **Never offer an alias for a recording path** (no `latest` symlink or copy).
  `StepTimeline.beside` resolves the sidecar from the path as written; an alias finds
  none, and a missing sidecar is supported, so a `retake` becomes a `pass`.
- **Never commit real recordings** or put them in the fixture folder: they are human motion data.
- **Invalid joints carry arbitrary values, not zeros.** Gate every value check on
  `is_valid`. Pico copies the pose and derives `is_valid` afterwards, Noitom zeroes it;
  `tests/test_pico_shapes.py` covers Pico. On a three-tracker rig the upper body is
  solver output.

## Tests

- **Unit** tests feed accumulators frames built in memory by `tests/synth.py`; they need
  no MCAP and are all CI runs. `tests/test_panel_app.py` needs viser and skips without it;
  the checker's suite must not start needing the extra.
- **Oracle** tests are parametrised over `../oracle/fixtures_index.json` and skip until
  `../oracle/generate.sh` has built the recordings (`FULLBODY_FIXTURES` overrides the location).
- Assert graded fixtures by accuracy and monotonicity against the injected magnitude,
  plus a zero rung reading zero — never by pass/fail.
