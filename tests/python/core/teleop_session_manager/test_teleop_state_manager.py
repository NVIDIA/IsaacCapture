# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DefaultTeleopStateManager reset pulses and fail-safe state."""

import pytest

from isaaccapture.retargeting_engine.interface import (
    ExecutionState,
    OptionalTensorGroup,
)
from isaaccapture.teleop_session_manager import DefaultTeleopStateManager
from isaaccapture.teleop_session_manager.teleop_state_manager_types import (
    teleop_control_states,
)

KILL = DefaultTeleopStateManager.INPUT_KILL
RUN = DefaultTeleopStateManager.INPUT_RUN_TOGGLE
RESET = DefaultTeleopStateManager.INPUT_RESET


def _step(manager, kill=False, run=False, reset=False):
    """Run one frame; ``None`` leaves an input absent. Returns (reset, state)."""
    spec = manager.input_spec()
    inputs = {}
    for name, value in ((KILL, kill), (RUN, run), (RESET, reset)):
        group = OptionalTensorGroup(spec[name])
        if value is not None:
            group[0] = value
        inputs[name] = group
    outputs = manager(inputs)
    states = teleop_control_states()
    one_hot = [bool(outputs["teleop_state"][i]) for i in range(len(states))]
    assert sum(one_hot) == 1
    return bool(outputs["reset_event"][0]), states[one_hot.index(True)]


@pytest.fixture
def manager():
    m = DefaultTeleopStateManager(name="teleop_manager")
    _step(m)
    return m


def _run_to_running(manager):
    _step(manager, run=True)
    _step(manager)
    _step(manager, run=True)
    assert _step(manager) == (False, ExecutionState.RUNNING)


def test_kill_held_emits_one_reset(manager):
    _run_to_running(manager)
    frames = [_step(manager, kill=True) for _ in range(5)]
    assert [r for r, _ in frames] == [True, False, False, False, False]
    assert all(s == ExecutionState.STOPPED for _, s in frames)


def test_kill_repress_emits_one_more_reset(manager):
    _step(manager, kill=True)
    _step(manager, kill=True)
    assert _step(manager) == (False, ExecutionState.STOPPED)
    resets = [_step(manager, kill=True)[0] for _ in range(3)]
    assert resets == [True, False, False]


def test_reset_button_rising_edge(manager):
    _run_to_running(manager)
    resets = [_step(manager, reset=True)[0] for _ in range(3)]
    assert resets == [True, False, False]
    assert _step(manager) == (False, ExecutionState.RUNNING)


@pytest.mark.parametrize("reset_input", [None, False])
def test_input_loss_emits_one_reset(manager, reset_input):
    _run_to_running(manager)
    frames = [_step(manager, None, None, reset_input) for _ in range(3)]
    assert (
        frames
        == [(True, ExecutionState.STOPPED)] + [(False, ExecutionState.STOPPED)] * 2
    )


def test_inputs_absent_from_start_emit_no_reset():
    m = DefaultTeleopStateManager(name="teleop_manager")
    frames = [_step(m, None, None, None) for _ in range(3)]
    assert frames == [(False, ExecutionState.STOPPED)] * 3


def test_recovery_with_kill_held_emits_no_extra_reset(manager):
    _run_to_running(manager)
    assert _step(manager, None, None, None)[0] is True
    frames = [_step(manager, kill=True) for _ in range(3)]
    assert frames == [(False, ExecutionState.STOPPED)] * 3
    assert _step(manager) == (False, ExecutionState.STOPPED)
    assert _step(manager, kill=True) == (True, ExecutionState.STOPPED)
