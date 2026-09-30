<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `acceptance/hand/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../AGENTS.md`](../../AGENTS.md)
before editing here, and read [`../common/AGENTS.md`](../common/AGENTS.md) and
[`checker/AGENTS.md`](checker/AGENTS.md).

- **Gloves are the target.** Headset optical hands are the development source; they
  cannot show 21-joint layouts, glove axis conversions or wrist-fusion faults.
- **No internal gate or milestone numbers** anywhere in this tree, including docs and
  check names. The four check groups are `signal`, `shape`, `placement`, `windows`.
  `checker/tests/test_boundaries.py` enforces it.
- **Dependency direction:** `capture` imports the checker; the checker never imports
  isaacteleop or capture; the oracle imports neither the checker nor
  `acceptance_common`.
- **Thresholds come from real glove recordings only**, never from optical hands or the
  oracle. Until then they are loose placeholders; do not tighten one to make a fixture
  pass.
- **Capture pacing is a timer; nothing is pressed.** A trigger moves a finger and the
  controller may be the wrist source. Windows are stamped with `time.monotonic_ns()`,
  which is the clock `sample_time_local_common_clock` is in, so labels need no record
  counting.
- **Cue WAVs are committed and hash-checked** against `session.py` through
  `cues/index.json`; re-render with piper when wording changes.
