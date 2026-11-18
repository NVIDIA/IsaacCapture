<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `cts/hand/checker/`

**CRITICAL:** complete the mandatory `AGENTS.md` preflight in [`../../../../../../AGENTS.md`](../../../../../../AGENTS.md)
before editing here, and read [`../AGENTS.md`](../AGENTS.md) and
[`../../common/AGENTS.md`](../../common/AGENTS.md).

- **Per-hand checks subclass `HandCheck(side)` and are registered through `per_hand()`**,
  which copies the policy fields and merges the two sides: any fail fails, else any
  insufficient is insufficient. A check that compares the hands subclasses
  `_PairCheck`, which pairs opposite-side records within `PAIR_TOLERANCE_NS` and
  leaves a check with no pairs unanswered.
- **A cross-hand check gates on the pose before judging the device.** Not palm down,
  not facing, not rolled far enough is a PERFORMANCE outcome (`Outcome.attribution`),
  so a missed pose asks for a retake instead of failing the glove.
- **Structural questions use the OpenXR joint frame, not a threshold:** −Z along the
  bone toward the tip, +Y dorsal. Handedness depends on `quaternion.component_order`,
  because a wxyz stream makes every axis read wrong.
- **Shape measurements are in the wrist frame** (`Frame.local`), so they survive any
  placement of the hand in the world.
- **A window is not held for its whole length.** Performers lift a hand partway through
  a flat window. Gate each frame on the pose itself (still,
  palm down, at table height) and take the median of per-frame measures; a joint-wise
  mean over a window averages two placements into a hand that never existed.
- **A cross-hand vector compared across windows keeps one sign.** Write it right minus
  left everywhere; a roll reference taken as one difference and judged against the
  other doubles the reference gap, and a small synthetic gap hides it.
- **Nearest-neighbour is only structural when the neighbours are far.** Thumb to
  fingertip works; fingertip to fingertip across pressed palms does not, since the
  neighbours are 2 cm apart and any offset over 1 cm reads as a wrong finger.
- **Window checks are `required = False`.** Whether a step was performed attributes to
  PERFORMANCE, so a missed pose asks for a retake rather than blaming the device.
- **Unanswered is not failed.** When PALM and the metacarpals are all invalid,
  `optional_joints.geometry_consistent` has nothing to judge and says so.
- **Oracle check names are the vocabulary.** Renaming a check means updating
  `../oracle/fixtures_index.json` through the generator, never by hand.
