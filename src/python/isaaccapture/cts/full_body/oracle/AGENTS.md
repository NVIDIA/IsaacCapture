<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `cts/full_body/oracle/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../../../../../AGENTS.md`](../../../../../../AGENTS.md)
before editing here, and read [`../checker/AGENTS.md`](../checker/AGENTS.md), which covers
the thing this set judges. [`README.md`](README.md) says what each file is and how to
build the set.

## Purpose and limits

**Use these fixtures to prove a measurement is correct. Never to choose a threshold.**
Synthetic motion is smooth by construction (still windows are perfectly still, real ones
carry tremor, foot sliding and tracker jitter) and injected faults are caricatures.
Structural checks (up axis, units, handedness, quaternion order, joint indexing) need no
threshold and are settled here; everything expressed as a tolerance is not. Do not treat a property the
generator happens to produce as a property of real recordings.

## Share no code with the checker

The 24-joint table, the parent table, the forward kinematics and the quaternion
arithmetic exist twice on purpose: here in `skeleton.py`, and in
`../checker/src/full_body_cts/profile.py` and `../../common/src/cts_common/vectors.py`. A
fault in code shared by the oracle and the thing it judges cancels itself out. Do not
resolve the duplication.
`../checker/tests/test_boundaries.py::test_the_oracle_shares_no_code_with_the_checker`
asserts it.

## The index is committed; the recordings are not

Only the index is versioned, which is safe only because generation is **byte-deterministic**.
Preserve that:

- Every seed is fixed (`skeleton.build_clean`, `PostureParams.seed`, every
  `random.Random(...)` in `generate_fixtures.py`).
- Nothing reads a wall clock; `write_mcap` takes every timestamp from the frame.
- `mcap` is pinned in `../checker/requirements.txt`: the index records `size_bytes`, and
  a writer with different chunking would move every one.

`generate.sh` fails if `fixtures_index.json` comes out dirty; find which of the above
broke rather than committing the new index.

## Rules

- **Regenerate, never hand-edit.** Add a defect as a new transformation in the generator
  so it composes with everything else.
- **The index is never edited to make a checker test pass.** Declare a disagreement in
  `../checker/tests/known_deviations.py` with its reason.
- **Check names in the index are the vocabulary.** Each defect fixture's
  `expected_failing_check` specifies a check's name; do not rename one without the index.
- **Never put a real recording in here**: human motion data, and too large for version control.
- **Test data the checker needs for itself does not belong here.** Adding a fixture
  rewrites the index, so build cases about the checker rather than the device in memory
  with `../checker/tests/synth.py`.
