<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `cts/hand/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../../../../AGENTS.md`](../../../../../AGENTS.md)
before editing here, and read [`../common/AGENTS.md`](../common/AGENTS.md) and
[`checker/AGENTS.md`](checker/AGENTS.md).

- **Gloves are the target.** Headset optical hands cannot show 21-joint layouts, glove
  axis conversions or wrist-fusion faults.
- **Check groups are named, not numbered:** `signal`, `shape`, `placement`, `windows`.
- **Dependency direction:** `capture` imports the checker; the checker never imports
  isaaccapture or capture; the oracle imports neither the checker nor
  `cts_common`.
- **Physical contact is the ground truth.** With no motion capture, the table, pressed
  palms and touching fingertips are what the data can be held to; new cross-hand steps
  should be built on contact the performer can feel, not on a pose they approximate.
- **Thresholds come from real glove recordings only**, never from optical hands or the
  oracle. Existing ones are loose; do not tighten one to make a fixture pass.
- **Capture pacing is a timer; nothing is pressed.** A trigger moves a finger and the
  controller may be the wrist source. Windows are stamped with `time.monotonic_ns()`,
  which is the clock `sample_time_local_common_clock` is in, so labels need no record
  counting.
- **Name the sides in full** (`left`, `right`, `l_h`): ruff E741 rejects a bare `l`,
  and paired left/right code reaches for it.
- **Cue WAVs are committed and hash-checked** against `session.py` through
  `cues/index.json`; re-render with piper when wording changes.
