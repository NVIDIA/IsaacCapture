<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `acceptance/full_body/checker/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md)
before editing here. Read [`../oracle/AGENTS.md`](../oracle/AGENTS.md) too if you touch
anything the fixture set is the oracle for.

This is the design record for the checker: the reasoning, the measurements behind every
threshold, and the questions still open. [`README.md`](README.md) is the operator's guide
and this file does not repeat it.

## The one thing to take away

One real PICO 4 Ultra capture exposed five defects in this checker. Fifty-five
synthetic fixtures had found none of them.

They share a single cause: **a property that the generated fixtures happen to satisfy was
being treated as a property of recordings in general.** Four instances of that mistake,
each of which looked like a reasonable assumption until a person performed the script:

- the recording is the performance (it is not: it starts before and ends after),
- the performer can reach the textbook pose (a generated one always does),
- every bone length is measured (on a three-tracker rig the arms are solved),
- a squat is symmetric (a person favours a leg).

So when you add or change a check, ask: **is this criterion resting on a property of the
device, or on a property the generator happens to produce?** Synthetic fixtures prove a
measurement is correct. They cannot tell you what a recording looks like.

## The five defects, and the evidence

### 1. `segmentation.label_alignment` requires containment, not coincidence

It used to require the labelled span and the recorded span to agree end to end within
250 ms. Real captures do not: one carries 20.8 s of lead-in and 15.2 s of tail, another
16.6 s and 12.4 s. It now requires that every window be backed by frames and lie inside
the recording (`MAX_UNCOVERED_FRACTION = 0.02` of a window's own length may stick out),
and it reports the unlabelled ends rather than judging them.

This is a dependency of thirteen G4 checks, so the old form suppressed every G4
measurement on a perfectly good recording.

It **deliberately no longer detects** a sidecar shifted wholesale but still inside the
recorded span. The evidence for that case belongs to
`segmentation.labelled_step_actually_performed` and `segmentation.step_order_matches_labels`,
which read what the windows contain instead of where they sit.

**`label_windows_wellformed` no longer requires a partition either, for the same
reason.** `StepTimeline.defects()` used to call an unlabelled stretch between two
windows a `gap`, and the check is HARD/DEVICE and a dependency of the same thirteen. Once
the capture panel started opening each window on the performer's press, gaps became
universal — the performer is already in the pose when they press, so the move between
poses belongs to no window. Measured on the 102 s take relabelled that way: 6.4 s
unlabelled between windows, `verdict FAIL`, every G4 check suppressed and the device
blamed. The time between windows is now a measurement (`unlabelled_between_s`). An
**overlap** is still a defect: the frames one window would be measured over belong to
the next. Do not restore the partition requirement — windows that tile end to end are
the signature of a computed schedule, not of a good take.

### 2. `posture.arm_raise_range_of_motion` attributes to PERFORMANCE, not DEVICE

One recording cannot separate "the device clipped the arm" from "the arm never went up."
The plateau-shape idea that would have separated them is dead, measured: the share of
samples within one degree of the peak is 24% on the saturating fixture, 23% on the clean
one — the generated performer holds the pose — and 5% for a real performer. That statistic
separates synthetic from human, not fault from performance.

So the peak angle is the only signal, it is the same number in both cases, and asking for
a retake is the honest verdict. A device that truly clips fails every retake, which is
where that evidence accumulates.

### 3. Four graded measurements never judge, and say so

`posture.tpose_arm_droop`, `posture.tpose_left_right_asymmetry`,
`posture.cumulative_drift_between_tpose_windows` and
`posture.contralateral_crosstalk_single_leg_raise` report a number and no verdict, because
their thresholds have to come from real subjects rather than from the fixtures used to
prove the measurement correct.

They return `PASS` only because there is no other way to say "measured", which reads as an
approval nobody gave: a real capture printed 70.9 deg of contralateral crosstalk beside
the word `pass`. Hence the `judged` class flag, rendered `[meas]`.

### 4. A bone that varies on an otherwise rigid skeleton is solved, not broken

With three trackers (both ankles and the waist) the arms are IK-solved to the controllers,
and the solver puts its reach error in the last segment. On a real capture the two
forearms varied 68% and 46% of their length while the other 21 bones held to within
float32 storage precision.

The discriminating signal is **the contrast, not the variation.** A genuinely rubber
skeleton has no fixed bones to contrast against: `defect_bone_length_drift` varies all 23
of them and holds none. So `skeleton.bone_length_constancy` reports a varying bone as
solved/derived only on a rig where at least `MIN_RIGID_FRACTION = 0.6` of the measured
bones are fixed (`RIGID_CV = 0.002`), and `skeleton.anthropometric_plausibility` drops a
solved forearm from its proportion prior rather than judging a length the solver invented.

Constantly-zero bones are the mirror case: Noitom back-fills hands from wrists and feet
from ankles and marks them valid. Derived, never a fault.

### 5. The arm raise is measured at the tracked endpoint

The measurement is the elevation of the shoulder-to-hand line, not the shoulder joint
angle. On a rig with no shoulder or elbow tracking the shoulder angle is something the
solver made up: on a real overhead raise it read **−2.8 deg while the hand was 30 cm above
the shoulder**, and the shoulder-to-elbow geometric vector agreed at 2.6 deg. Two
independent algorithms reading the same wrong thing is how it is known to be the rig and
not the quaternion arithmetic.

`MIN_ELEVATION_DEG = 60.0` did not move, because the fixtures are driven by forward
kinematics and the new metric still separates them: 25.4 deg on the saturating fixture
against 89.4 deg on a clean one. This was not a loosening — it corrected a wrong reading.
The real capture raised to 72.5 and 79.8 deg; the old metric read 16 and 38 deg and
reported a device fault.

## `squat_knee_symmetry` has a real attribution signal

This is the one place where one recording genuinely separates device from performer, so it
is worth stating the mechanism: a person favouring a leg bends that **whole leg chain**
differently, while a device that mis-estimates the knees leaves the joints it actually
tracks alone.

`g4_device_squat_knee_asymmetry` holds hips and ankles at exactly 0.0 deg beside 22 deg of
knee difference — its own description says the performer squatted evenly. A real uneven
squat measured 21.4 deg of knee difference beside 12.9 deg at the hips and 6.6 deg at the
ankles. `CHAIN_ASYMMETRY_DEG = 5.0` sits in that gap.

The consequence for the model: `Outcome` carries an optional `attribution` that overrides
the class default. A check whose evidence can name the culprit must say which one it saw
rather than declaring one for both cases.

## The model

A check is an incremental accumulator (`update(frame)` / `result()`), so one implementation
serves an MCAP file, a live session and a replay session. `result()` can return
`INSUFFICIENT_DATA` at any time: that is the grey state in a live panel and the correct
offline answer for a recording too short to conclude anything.

Measurement and policy are separate. The accumulator reports a status plus its
measurements; the class declares the policy:

- **`severity`** — `HARD`, `SOFT`, `ADVISORY`. Advisory results never move the verdict;
  they exist for cases where a hard failure would reject legitimate hardware.
- **`attribution`** — `DEVICE` or `PERFORMANCE`, overridable per result by `Outcome`.
- **`judged`** — `False` for a check that measures and never judges. Renders `[meas]`.
- **`required`** — `False` when the question is about what the subject did rather than
  about the recording being adequate. A held T-pose cannot reveal a frame mismatch however
  long it runs, so such a check left unanswered must not drag the whole recording into
  limbo; an unanswered *required* check must.
- **`depends_on`** — names of checks whose failure makes this measurement meaningless.
  `suppress_dependents` iterates to a fixed point, so suppression propagates along a chain.
  **Suppression turns a result into "cannot conclude", never into a pass.**

Verdict aggregation is a pure function over the results: any counted DEVICE failure gives
`fail`; otherwise any counted PERFORMANCE failure gives `retake`; otherwise an unanswered
required check gives `insufficient_data`; otherwise `pass`. Keeping `fail` and `retake`
apart is the whole point of the attribution field — do not collapse them.

Thresholds live on the check class as named constants, so a number can be filled in later
without touching a measurement.

`Mark` in `report.py` is the single definition of how one result is shown — `pass`,
`FAIL`, `meas`, `n/a`, `note` — and `CheckResult.mark` derives it from status, severity
and `judged`. The text report, the panel and `to_dict` all read it, so it travels with
every result we emit rather than being recomputed downstream. A sixth state, or a change
to an existing one, goes there and nowhere else; a renderer that recomputes this drifts,
and the drift is silent.

**Every JSON this package writes must survive a strict parser.** Measurements can be
non-finite — `coordinate_frame.up_axis` divides by the second-largest axis component and
reports `inf` when that is exactly zero — and Python's `json` writes a bare `NaN` that
`JSON.parse` rejects. `report.json_safe` nulls them and `allow_nan=False` makes a leak
loud. A new measurement that can divide needs no new code, but a new writer does need
the flag.

## The panel

`panel/` serves one recording as a skeleton plus the result list. It exists for what the
text cannot express: **which joint, and when.** Moving 38 lines onto a web page would
buy nothing, so what it adds is spatial and temporal — a hand that drops out as the
subject turns, a dropped block that is a spike rather than a slightly lower mean.

- **viser stays in `panel/app.py`.** It is an optional extra
  (`requirements-panel.txt`, `./setup_env.sh --panel`), and the checker keeps running on
  `mcap` and `flatbuffers` alone. `tests/test_panel_boundary.py` asserts both halves:
  that no other module imports it, and that the checker and the panel's own arithmetic
  import with the module blocked. One convenience import is all it would take to lose a
  property that currently lets acceptance work proceed without building the repository.
  For the same reason the skeleton topology comes from `profile.FULL_BODY.bones()`, not
  from `examples/mcap_record_replay`'s `BODY_BONES`, which would drag the whole
  `TeleopSession` pipeline in.
- **It is written to `FrameSource`, not to MCAP.** `panel/track.TeeSource` wraps any
  source and accumulates the playback track while the checks consume the same pass.
  Adding a live panel is then a `LiveFrameSource` and no change here — that is the whole
  reason for the shape.
- **An invalid joint is drawn where it was last valid, in red, never where the record
  says.** Invalid joints carry arbitrary values, and one of those in a point cloud puts
  the camera far enough away that nothing else is visible. `Sample.positions` drops the
  recorded value and holds the last valid one; `Sample.valid` is what says which it is.
  Nothing downstream of the panel should read a garbage position, so it is not kept.
- **Only final results are shown.** Three checks need the whole recording by
  construction (`coverage.validity_trend` compares a head window against a tail window,
  `posture.cumulative_drift_between_tpose_windows` needs both T-poses, and all of G4
  needs the labels), so a `result()` taken mid-playback has no meaning.
- **`panel/bundle.py` packages a submission, and holds no viser.** The archive's job is
  to carry the **inputs that cannot be regenerated** — above all the labels sidecar,
  without which every G4 check reports unanswered and the verdict changes. The packed
  report is a cache: the checker is deterministic, so a re-run at the same commit
  reproduces it, and where the two disagree the recording wins. That is why `inputs`
  (file hashes, labels, `tool.commit`) is kept apart from the outputs, and why a missing
  companion is stated in `inputs.missing` rather than treated as an error. zip rather
  than tar.gz for the central directory: `report.json` reads out in 0.4 ms without
  touching the recording, against 17.4 ms to merely list a tar.gz.
- **Deliberately absent from this version**, each waiting on something undecided:
  jumping to where a check failed (checks cannot report a time range yet), capture
  orchestration, a submitter attestation form (the G0 fields are not settled), and live
  (needs `LiveFrameSource`). Mixing any of them in would have stalled the panel on
  someone else's design question.

## Architecture

The modules are organised by **what input they need**, which deliberately does not mirror
the G-numbering. The G-numbers are the narrative for the submitter; these are the layers:

| Layer | Needs | Per-schema cost |
|---|---|---|
| Envelope — the Record wrapper only | nothing | none |
| Payload — field roles | one descriptor | a descriptor |
| Geometry — a skeleton profile | `profile.py` | a profile |
| Window — the reviewer's motion labels | a label sidecar | a script table |

Other structural decisions that are settled:

- **Check names come from `fixtures_index.json`.** Each defect fixture carries an
  `expected_failing_check`, so the vocabulary is specified rather than invented. Do not
  rename a check without the index.
- **Locate the channel by declared schema name** (`core.FullBodyPoseRecord`), not by topic.
  The topic prefix is whatever `name=` the recording script passed. Same for the profile
  string and compression: unverified container details are informational.
- **Read in file order.** `mcap.reader.make_reader()` re-sorts by log time, which silently
  repairs the non-monotonic-timestamp fixture; use
  `NonSeekingReader(path).iter_messages(log_time_order=False)`, as the C++
  `LinearMessageView` does.
- **`full_body` is one profile, `hand` is the known next one.** The geometry checks read
  topology, symmetry pairs, proportion priors and *which checks apply* from the profile —
  gravity alignment is body-only.
  - **A profile belongs to a schema, never to a device.** Where devices legitimately differ
    (which optional joints they fill, how they source the wrist), write a rule every device
    is judged by — never a per-device table. A new device must be checkable with no edit
    here; the device name is provenance only.
- **This directory sits outside `src/`** because `src/python/CMakeLists.txt` globs `.py`
  recursively: a file placed under it would change the wheel with no edit to any build
  file. `acceptance/` holds one directory per recordable `.fbs`, and each splits three
  ways: `capture/` records a take, `checker/` judges it, `oracle/` generates takes whose
  verdict is already known. The boundaries between them are under Hard constraints.

## Hard constraints

1. **Pure addition, and the checker imports no `isaacteleop`.** Nothing under `src/`,
   `examples/` or `docs/` is modified, and nothing under `checker/` names the package —
   `mcap` and `flatbuffers` are the whole dependency, which is why the work needs no
   schema review and competes for no merge window.

   The ban stops at `checker/`. `../capture/` records through `TeleopSession`
   and therefore **has to** import the built package, so the dependency runs one way:
   capture reads the checker's `Frame`, `TrackBuilder` and profile, never the reverse.
   `tests/test_boundaries.py` asserts all of it mechanically, diffing against the commit
   this branch grew from rather than against wherever `origin/main` has since moved to.

   `../oracle/` runs the other way again: it shares **no** code with this package, so
   that a fault cannot cancel itself out across the oracle and the thing it judges. The
   duplicated joint table and quaternion arithmetic are deliberate; see
   [`../oracle/AGENTS.md`](../oracle/AGENTS.md).
2. **`fixtures_index.json` is the oracle and is never edited to make a test pass.** A
   disagreement is declared in `tests/known_deviations.py` with its reason.
3. **New test data belongs here.** The fixture generator rewrites the index as a side
   effect of adding a fixture, so cases this checker needs are built by `tests/synth.py`
   into a temp directory.
4. **Thresholds do not come from fixtures.** Synthetic motion is smooth by construction and
   injected faults are caricatures. Use the fixtures to prove a measurement; use real
   subjects to choose a number.

### The three declared deviations

All in `tests/known_deviations.py`:

- **`defect_all_tracked_flag_inconsistent`** — `all_joint_poses_tracked` comes from the
  OpenXR runtime's `locations.allJointPosesTracked` while `is_valid` comes independently
  from each joint's `XR_SPACE_LOCATION_*_VALID_BIT`. The two can legitimately disagree, so
  `consistency.all_joint_poses_tracked` is advisory and the fixture reports `pass`.
- **`defect_device_clock_copies_common`** — Pico derives
  `sample_time_raw_device_clock` from `xrConvertTimespecTimeToTimeKHR`, so a runtime
  representing `XrTime` as `CLOCK_MONOTONIC` nanoseconds makes the device clock equal the
  common clock on a good recording. Advisory, unverified without more hardware.
- **`g4_device_arm_raise_saturates`** — `retake` where the index says `fail`, for the
  reason in defect 2 above.

`EXPECTED_COLLATERAL` in the same file records where one injected defect legitimately trips
more than one check — centimetre units really do imply 168 m/s and a 157 m skeleton, and on
a bilaterally symmetric body mirroring the rig and swapping the left/right labels are the
same transform, so those two fixtures each trip both chirality checks.

## Established constraints — do not re-derive these

Each cost real time to establish and is asserted somewhere in the tests.

- **Chirality and left/right labelling derive forward from the ankle-to-foot vectors, not
  from the pelvis quaternion.** A pelvis-derived facing makes both checks fail on any
  recording whose orientations are wrong for an unrelated reason, sending the submitter
  after the wrong defect. The mean over the pair is unchanged by either a mirror or a
  left/right swap. Feet are trusted only when both are present, plausibly long against the
  torso, and pointing within 60° of each other — a foot zeroed to the world origin passes a
  bare length test because its ankle is near the origin too. Recordings without usable feet
  (back-filled endpoints, and Pico, which reports no feet) fall back to the pelvis, and the
  outcome records which reference it used.
- **`skeleton.joint_index_assignment` measures how much of a bone's length is spent moving
  back toward the pelvis.** Raw distance-to-root ordering does not work: an A-pose puts
  every elbow nearer the pelvis than its shoulder, which is what made the check fire on
  `defect_validity_degradation`, whose surviving frames are all in the opening A-pose. A
  reversed bone points straight at the root for a ratio of +1 while nothing correctly
  indexed in the corpus exceeds −0.47, so the 0.8 threshold sits in an empty gap rather
  than being tuned.
- **Range of motion is measured torso-relative, never in world coordinates.** An operator
  who leans while raising an arm reads −83° of shoulder ROM in world coordinates, against
  −95° clean and −25° for a device whose tracking saturates: the leaning operator looks like
  a mildly broken device. Torso-relative they read −95.5°, i.e. intact. A world-frame
  measurement converts a `retake` into a `fail`.
- **"Stature" here is the head *joint* above the lowest joint, about 1.57 m**, not
  anatomical stature (~1.72 m). A proportion prior must state which it tests or it sits
  systematically ~15 cm low. `stature_chain` sums bone lengths along one leg and the spine,
  so no posture can move it — unlike a bounding box.
- **Use peak joint angle, not path length, to decide whether a step was performed.**
  Position noise of 0.3 mm/frame random-walks to roughly 0.33 m over a window, about 12% of
  the real signal, so motion-energy and path-length statistics are noise-dominated at this
  scale. Peak angle is not.
- **Held poses are measured over the trailing 60% of their window** (`SETTLE_FRACTION`),
  because the script blends into each pose over 0.55 s and the window mean otherwise reports
  the transition.
- **flatc v24.3.25 with the `GenerateFlatBuffers.cmake` flag set produces a `.bfbs`
  byte-identical to `src/core/schema/golden/full_body.bfbs`.** That check in `setup_env.sh`
  is what ties this checker to the schema the C++ writer actually embeds. A distro package
  or Homebrew ships 25.x, whose output does not match.
- **The Python `flatbuffers` package has no reflection module.** Payloads cannot be walked
  from the embedded `.bfbs`; decode with the flatc-generated bindings and use the `.bfbs`
  only for byte comparison. Consequence: payload checks are per-schema, one mechanical
  flatc step, not runtime-generic.
- **Never offer an alias for a recording path — no `latest` symlink, no convenience copy.**
  `StepTimeline.beside` resolves the sidecar from the path as written, so an alias finds no
  labels beside itself, and a missing sidecar is a supported case rather than an error: the
  thirteen G4 checks go unanswered, they are `required=False`, and the verdict comes out
  `pass`. Measured on a real take: `retake` by its real path, `pass` through a symlink to
  it. `capture/record.sh` used to maintain such a symlink; that is why it no longer does.

## Real hardware, established facts

The captures are PICO 4 Ultra through the CloudXR web client, ~56 Hz, three trackers (both
ankles and the waist) with a controller in each hand. Real recordings are human motion data:
they stay out of git and out of the fixture folder, and live under `$HOME`.

- **`body_tracking: false` came from the browser** — not from the hardware and not from
  licensing. The browser PICO ships granted the WebXR `body-tracking` feature on the same
  consumer 4 Ultra with no enterprise activation. Earlier notes claimed a consumer headset
  never grants it; that claim is wrong, do not write it back.
- **Invalid joints carry arbitrary values, not zeros.** One held a quaternion component of
  −16363.96. The fixture set records zeros here, so gating on the validity flag is
  load-bearing for a wider reason than that set can show: an ungated finite or unit-norm
  check fails a working device on data it was never meant to read.
- **The two clocks are a rigid offset.** `sample_time_raw_device_clock` runs 1291.369356 s
  ahead of the common clock with zero variance over 3373 frames, and
  `available_time_local_common_clock` equals the sample time exactly, so
  `timestamps.available_not_before_sample` cannot fail on this device.
- **Three trackers means the upper body is inferred** from the headset and controllers.
  Shoulder, elbow and spine angles on such a rig are solver output; see defects 4 and 5.

## Derived data

`../capture/amplify_arm_raise.py` builds a passing take out of the 102 s one. Only the
two arm-raise windows are edited, and inside them only the three joints below each
shoulder: positions and orientations rotate together about the shoulder so the relative
geometry `consistency.position_orientation_same_frame` reads is preserved. Every other
frame and joint is the original capture. The result reaches `pass` with all 38 checks
answered, and the solved bones are still exactly the two forearms.

**Its limits matter as much as its result.** It proves that no check misfires on the shape
of real data — the failure mode that produced five defects in one afternoon and that the
generated fixtures never showed. It does **not** prove a person can reach the gate: that
one reading is synthetic. It is not a substitute for one real qualifying capture.

Building it also found a ceiling: targeting 85 deg of elevation drives a joint to 25 m/s,
past the 20 m/s `continuity.max_joint_velocity` limit. On a real performer's own timing the
arm cannot be raised much higher without making the speed implausible. The file ships at
70 deg, which peaks at 17.4 m/s.

## Open

- **`consistency.all_joint_poses_tracked` is still undecided.** Both real captures have
  every joint valid, so the flag and the per-joint flags agree trivially. Settling it needs
  a capture with partial occlusion or a tracker dropping out.
- **Contralateral crosstalk of 70.9 and 77.6 deg on the standing hip is unexplained.** The
  hip angle is pelvis-relative, so pelvis tilt is the suspicion, unverified.
- **Peak joint speeds of 10–11 m/s are unexplained.** Below the 20 m/s limit, so nothing
  fails, but nothing accounts for them either.
- **No purely real capture has reached `pass`.** The 102 s take squats evenly (4.0 deg)
  but the hands reach only 36 deg; the earlier take reaches 72 deg but squats unevenly
  (21.4 deg). This is the most concrete gap in the evidence.
- **G5 is not a slice away — the thing it would test does not exist.** No retargeter in
  this repository consumes `full_body`. Everything under
  `src/python/isaacteleop/retargeters/` takes `ControllerInput`, `HandInput`,
  `AxisPedalInput`, `JointState` or `ValueInput`; the G1 humanoid retargeter is
  controller-driven like the rest. Every `FullBodyInputIndex` consumer in the tree is a
  source node, a tensor-type definition, or a record/replay/live/viz/ROS 2 example —
  recording and visualisation, not retargeting. **The mapping from a 24-joint human
  skeleton onto a robot has not been written.** Estimate G5 as that mapping, not as a
  wrapper around replay.

  Where G5 should end up — attested by the submitter, redefined, or dropped — is
  undecided, and this entry does not decide it.
- **G6 is not automated by design** (a reviewer watches the video of the same session).
- **The offline panel is built; live still needs a `LiveFrameSource`. When it arrives it
  will show geometry but not timing.** `core.FullBodyPoseRecord` is two layers,
  `data: FullBodyPose` beside `timestamp: DeviceDataTimestamp`, and
  `FullBodyTracker::get_body_pose` returns `Serialized<FullBodyPose>` — the inner payload,
  carrying only `joints` and `all_joint_poses_tracked`. The comment at the top of
  `src/core/schema/python/schema_serialized.h` says why: trackers publish their payload
  table directly and the `Record` wrapper is bound for MCAP, so the three device
  timestamps only exist once a recorder has serialized a `Record`. Nor is there a
  container time to fall back on: the panel's `Frame.log_time_ns` is a required field
  with no live value to put in it.

  So under live every time-dependent check is unavailable, not merely unanswered:
  `rate.interval_regularity` and its jitter, `timestamps.monotonic`,
  `timestamps.available_not_before_sample`, and `continuity.max_joint_velocity`, which
  needs a `dt`. A live panel shows the skeleton, per-joint validity and the valid-joint
  count; it does **not** show frame rate or jitter.

  **Do not substitute a local clock.** The reason is on `panel/track._stamp()`: that
  measures when Python received the sample, across the whole CloudXR path and the GIL, so
  the jitter it reports is the network's rather than the device's. Getting a live frame
  rate means exposing `DeviceDataTimestamp` through the C++ bindings, which is a `src/`
  change and is ruled out by Hard constraint 1.
- **Labels are a sidecar outside the recording**, marked provisional, because MCAP carries
  no annotation channel yet. Moving them in would need a core change, which is why they are
  not in there now.
- **An idea, not scheduled:** apply the fixture set's defect transformations to a real
  capture. That would give a real noise floor and exact ground truth at once — the defect
  transformations apply to any source.

## Tests

Two layers, split by whether a test needs the fixture set.

- **Unit** — accumulators fed frames built in memory by `tests/synth.py`. No MCAP, no
  external data, runs from a fresh clone. `tests/test_pico_shapes.py` covers shapes the
  fixture set (modelled on Noitom) does not: garbage on invalid joints, a registered channel
  with no messages, a file whose schema is something else, a renamed topic.
- **Oracle** — parametrised over `../oracle/fixtures_index.json`, asserting
  `expected_verdict` and `expected_failing_check`. 26 tests, skipped until
  `../oracle/generate.sh` has built the recordings the index describes;
  `FULLBODY_FIXTURES` overrides where they are looked for. The index is committed and
  the recordings are not, so `fixtures_root()` checks for both.

CI can only ever run the unit layer: the recordings are derived and not in git, the
largest is 1.77 MiB against pre-commit's 2000 KiB ceiling, and building them needs flatc
downloaded.

`tests/test_panel_app.py` is a third, smaller layer: it drives the renderer against a
real `ViserServer` and skips wherever viser is absent, which includes CI. Keep it that
way — the checker's suite must not start needing the extra. Everything about the panel
that is arithmetic rather than rendering — the track, the grouping, the bundle — is
tested in the unit layer instead, and that is the reason those modules hold no viser.

Graded fixtures are asserted by accuracy and monotonicity against the injected magnitude,
plus a zero rung reading zero — never by pass/fail.

## Working here

- Run `SKIP=check-copyright-year pre-commit run --all-files` from the repo root and fix
  every failure before treating a change as done; that is the hook set CI uses.
- Commit with `git commit -s` (DCO).
- New documents need the SPDX header pair, as an HTML comment block in Markdown, or the
  REUSE hook fails.
