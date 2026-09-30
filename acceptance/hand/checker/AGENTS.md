<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `acceptance/hand/checker/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md)
before editing here, and read [`../AGENTS.md`](../AGENTS.md) and
[`../../common/AGENTS.md`](../../common/AGENTS.md).

- **Per-hand checks subclass `HandCheck(side)` and are registered through `per_hand()`**,
  which copies the policy fields and merges the two sides: any fail fails, else any
  insufficient is insufficient. A check that compares the hands (`grasp.*`) is a plain
  `Check`.
- **Structural questions use the OpenXR joint frame, not a threshold:** −Z along the
  bone toward the tip, +Y dorsal. Handedness depends on `quaternion.component_order`,
  because a wxyz stream makes every axis read wrong.
- **Shape measurements are in the wrist frame** (`Frame.local`), so they survive any
  placement of the hand in the world.
- **Window checks are `required = False`.** Whether a step was performed attributes to
  PERFORMANCE, so a missed pose asks for a retake rather than blaming the device.
- **Unanswered is not failed.** When PALM and the metacarpals are all invalid,
  `optional_joints.geometry_consistent` has nothing to judge and says so.
- **Oracle check names are the vocabulary.** Renaming a check means updating
  `../oracle/fixtures_index.json` through the generator, never by hand.
