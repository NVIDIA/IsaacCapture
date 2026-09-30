<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `acceptance/full_body/oracle/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md)
before editing here, and read [`../checker/AGENTS.md`](../checker/AGENTS.md), which is
the design record for the thing this set judges.

[`README.md`](README.md) says what each file is and how to build the set. This file is
about the two ways a change here goes wrong: by making the fixtures lie, and by making
them agree with the checker for the wrong reason.

## The one thing to take away

**Use these fixtures to prove a measurement is correct. Never to choose a threshold.**

Synthetic motion is smooth by construction — still windows are perfectly still, while
real ones carry tremor, soft-tissue artefact, foot sliding and tracker jitter — and
injected faults are caricatures, where a real broken integration produces *strange*
motion rather than cleanly wrong motion. Structural checks (up axis, units, handedness,
quaternion order, joint indexing) need no threshold and are fully settled here.
Everything expressed as a tolerance is not.

One real capture exposed five checker defects that all 55 of these fixtures had missed,
every one of them a property the generator happens to produce being mistaken for a
property of recordings in general. The five are written up in
[`../checker/AGENTS.md`](../checker/AGENTS.md) with their measurements; they are not
repeated here, and the measurement rules they produced — torso-relative range of motion,
peak angle over path length, the stature definition, gating on the validity flag —
belong to the checker and are documented there.

## Share no code with the checker

The 24-joint table, the parent table, the forward kinematics and the quaternion
arithmetic exist twice on purpose: once here in `skeleton.py`, once across
`../checker/profile.py` and `../../common/src/acceptance_common/vectors.py`. A fault in code shared by the
oracle and the thing it judges cancels itself out and neither side can see it.

`../checker/tests/test_boundaries.py::test_the_oracle_shares_no_code_with_the_checker`
asserts it. Do not resolve the duplication — it is the reason the set is worth having.

## The index is committed; the recordings are not

67 MB of derived MCAP against a 33 KB index, so only the index is versioned. That is
only safe because generation is **byte-deterministic**, which is a property to preserve
rather than a happy accident:

- Every seed is fixed (`skeleton.build_clean`, `PostureParams.seed`, and the seven
  explicit `random.Random(...)` in `generate_fixtures.py`).
- Nothing reads a wall clock. `write_mcap` takes every timestamp from the frame.
- `mcap` is pinned in `../checker/requirements.txt`, because the index records
  `size_bytes` and a writer that changed its chunking would move every one of them.

`generate.sh` fails if `fixtures_index.json` comes out dirty. When it does, the question
is which of the above broke, not whether to commit the new index.

## Rules

- **Regenerate, never hand-edit.** Fixtures are derived artefacts. Add a defect as a new
  transformation in the generator so it composes with everything else.
- **The index is never edited to make a checker test pass.** A disagreement between the
  checker and the index is declared in `../checker/tests/known_deviations.py` with its
  reason. Three are on record.
- **Check names in the index are the vocabulary.** Each defect fixture's
  `expected_failing_check` is what specifies a check's name; renaming one without the
  index is how the two drift apart.
- **Never put a real recording in here.** Real captures are human motion data with
  licensing and governance constraints, and are too large for version control.
- **Test data the checker needs for itself does not belong here.** Adding a fixture
  rewrites the index as a side effect, so cases that are about the checker rather than
  about the device are built in memory by `../checker/tests/synth.py`.

## Open

- **The highest-value next step is not more synthetic data.** It is applying these defect
  transformations to a real capture: the transformations apply to any source, so one real
  recording converts the whole set into real noise with exact ground truth. Not scheduled.
- **The label sidecar is provisional** and expected to move into the recording once MCAP
  carries an annotation channel. `message_channel` records inbound client messages only
  and is not usable for this.
