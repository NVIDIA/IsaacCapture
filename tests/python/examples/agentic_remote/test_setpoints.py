# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The setpoints of a move: the same values APAI's own driver would send."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.agent import setpoints
from isaaccapture_examples.agentic_remote.agent.setpoints import (
    largest_step,
    min_jerk,
    sample_move,
    setpoint_count,
)

START = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
GOAL = np.array([0.3, 0.9, 1.4, -0.2, 0.25, 0.1])


def test_the_module_imports_without_apai() -> None:
    package = Path(setpoints.__file__).resolve().parents[3]
    code = (
        "import sys\n"
        "from isaaccapture_examples.agentic_remote.agent import setpoints\n"
        "assert len(setpoints.sample_move([0.0], [1.0], 0.1, 50.0)) == 5\n"
        "assert 'apai' not in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": str(package), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_the_start_is_left_out_and_the_last_setpoint_is_the_goal() -> None:
    rows = sample_move(START, GOAL, 1.5, 50.0)
    assert len(rows) == 75
    assert np.array_equal(rows[-1], GOAL)
    assert not np.array_equal(rows[0], START)
    # Min-jerk: the first step is a tiny part of the travel.
    assert largest_step(START, rows[0]) < 1e-4 * largest_step(START, GOAL)
    assert all(row.shape == (6,) and row.dtype == float for row in rows)


@pytest.mark.parametrize(
    ("duration_s", "count"),
    [(0.0, 2), (0.01, 2), (0.04, 2), (0.05, 2), (0.06, 3), (0.999, 49), (60.0, 3000)],
)
def test_a_move_has_the_whole_periods_of_its_length_and_at_least_two(
    duration_s: float, count: int
) -> None:
    assert setpoint_count(duration_s, 50.0) == count
    assert len(sample_move(START, GOAL, duration_s, 50.0)) == count


def test_the_time_law_runs_from_nought_to_one() -> None:
    assert min_jerk(0.0) == 0.0 and min_jerk(1.0) == 1.0
    assert min_jerk(0.5) == pytest.approx(0.5)
    steps = [min_jerk(i / 100) for i in range(101)]
    assert steps == sorted(steps)


def test_another_time_law_can_be_given() -> None:
    rows = sample_move([0.0, 0.0], [1.0, 2.0], 0.08, 50.0, time_law=lambda s: s)
    assert [row.tolist() for row in rows] == [
        [0.25, 0.5],
        [0.5, 1.0],
        [0.75, 1.5],
        [1.0, 2.0],
    ]


@pytest.mark.parametrize(
    ("start", "goal", "duration_s", "rate_hz"),
    [
        ([0.0], [1.0, 2.0], 1.0, 50.0),
        ([0.0], [1.0], float("nan"), 50.0),
        ([0.0], [1.0], 1.0, 0.0),
        ([0.0], [1.0], -1.0, 50.0),
        ([0.0], [1.0], 1.0, float("inf")),
    ],
)
def test_a_move_that_cannot_be_sampled_is_refused(
    start: list[float], goal: list[float], duration_s: float, rate_hz: float
) -> None:
    with pytest.raises(ValueError):
        sample_move(start, goal, duration_s, rate_hz)


@pytest.mark.apai
def test_the_time_law_is_apais() -> None:
    profile = importlib.import_module("apai.control.motion_profile")
    for i in range(1001):
        assert min_jerk(i / 1000) == profile.min_jerk(i / 1000)


@pytest.mark.apai
@pytest.mark.parametrize(
    "duration_s",
    [0.0, 0.03, 0.04, 0.1, 0.33, 0.5, 0.999, 1.0, 1.5, 1.7321, 2.0, 3.14159, 7.5],
)
@pytest.mark.parametrize("rate_hz", [50.0, 20.0])
def test_the_setpoints_are_those_apais_driver_sends_to_its_mock_arm(
    duration_s: float, rate_hz: float
) -> None:
    config = importlib.import_module("apai.config")
    mock_arm = importlib.import_module("apai.control.mock_arm")
    arm = mock_arm.MockArm(config.Cfg({"n_joints": 6, "home_q": START.tolist()}))
    arm.connect()
    approved: list[float] = []
    assert arm.stream_to(
        GOAL,
        duration_s,
        rate_hz=rate_hz,
        approve=lambda q_prev, q_i, dt: approved.append(dt),
    )
    ours = sample_move(START, GOAL, duration_s, rate_hz)
    assert len(arm.commands) == len(ours) == len(approved)
    for sent, row in zip(arm.commands, ours):
        assert np.array_equal(sent, row)  # value for value, not approximately
    # APAI's driver gives the checker duration / N; the Thor gives each
    # setpoint one period, which is never longer.
    assert 1.0 / rate_hz <= approved[0] + 1e-12 or duration_s * rate_hz < 2
