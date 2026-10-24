# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import synth

from hand_cts.checks.windows import JOINT_ABOVE_TABLE_M
from session import STEPS
from steps import Take
from table import TableEstimate

MS = 1_000_000
TABLE_Y = 0.75


def _frames(tick: int):
    flat = synth.right_local(0.0)
    return {
        side: synth.frame(tick, side, local=flat, wrist=(x, TABLE_Y, -0.3))
        for side, x in (("left", -0.2), ("right", 0.2))
    }


def test_the_table_is_set_on_the_tick_the_first_flat_window_closes():
    take = Take(cue_seconds={label: 0.5 for label, _, _ in STEPS}, end_tone_s=0.1)
    estimate = TableEstimate()
    now, tick = 0, 0
    take.start(now)
    while not estimate.update(take, now, _frames(tick)):
        now += 10 * MS
        tick += 1
        take.advance(now)
        assert now < 60_000 * MS, "no estimate"
    assert [w.label for w in take.windows] == ["flat_on_table_open"]
    finger_floor = min(p[1] for p, _ in synth.right_local(0.0)[7:]) + TABLE_Y
    assert abs(estimate.height_m - (finger_floor - JOINT_ABOVE_TABLE_M)) < 0.02
    assert not estimate.update(take, now, _frames(tick))
