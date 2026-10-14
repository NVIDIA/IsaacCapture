# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gamepad retargeters, fed through GamepadSource from real GamepadSnapshot payloads."""

import numpy as np
import pytest

from isaaccapture.retargeters import (
    GamepadGripperRetargeter,
    GamepadToSe2Retargeter,
    GamepadToSe2RetargeterConfig,
    GamepadToSe3RelRetargeter,
    GamepadToSe3RelRetargeterConfig,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import GamepadSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.execution_events import ExecutionEvents
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
)
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import GamepadInputState, GamepadSnapshot


def _gamepad(absent=False, **fields):
    """GamepadSource's "gamepad" output for a pad in this state, or for no pad when absent."""
    src = GamepadSource(name="gamepad")
    tg = TensorGroup(src.input_spec()["deviceio_gamepad"])
    tg[0] = None if absent else GamepadSnapshot(GamepadInputState(**fields))
    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_gamepad": tg}, outputs)
    return outputs["gamepad"]


def _run(retargeter, output, gamepad, reset=False):
    out = {output: _make_output_group(retargeter.output_spec()[output])}
    context = ComputeContext(execution_events=ExecutionEvents(reset=reset))
    retargeter.compute({"gamepad": gamepad}, out, context)
    return np.asarray(out[output][0])


def _se3():
    return GamepadToSe3RelRetargeter(GamepadToSe3RelRetargeterConfig(), name="se3")


def _se2():
    return GamepadToSe2Retargeter(GamepadToSe2RetargeterConfig(), name="se2")


class TestGamepadToSe3RelRetargeter:
    def test_left_stick_up_produces_forward_delta(self):
        delta = _run(_se3(), "ee_delta", _gamepad(thumbstick_left_y=1.0))

        assert delta[0] == pytest.approx(0.4)  # default pos_sensitivity
        assert np.allclose(delta[1:], 0.0)

    def test_sticks_combine_on_independent_axes(self):
        delta = _run(
            _se3(), "ee_delta", _gamepad(thumbstick_left_y=1.0, thumbstick_right_y=1.0)
        )

        assert delta[0] == pytest.approx(0.4)  # left stick up: +X
        assert delta[2] == pytest.approx(0.4)  # right stick up: +Z
        assert delta[1] == pytest.approx(0.0)
        assert np.allclose(delta[3:], 0.0)

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_stick_right_is_positive_y_and_yaw(self, direction):
        delta = _run(
            _se3(),
            "ee_delta",
            _gamepad(thumbstick_left_x=direction, thumbstick_right_x=direction),
        )

        assert np.sign(delta[1]) == direction
        assert np.sign(delta[5]) == direction
        assert np.allclose(delta[[0, 2, 3, 4]], 0.0)

    def test_dpad_rotates_roll_and_pitch(self):
        delta = _run(
            _se3(), "ee_delta", _gamepad(dpad_left_click=True, dpad_down_click=True)
        )

        assert np.allclose(delta[:3], 0.0)
        assert delta[3] > 0 and delta[4] > 0

    def test_absent_pad_yields_zero_delta(self):
        assert np.allclose(_run(_se3(), "ee_delta", _gamepad(absent=True)), 0.0)

    def test_dead_zone_suppresses_stick_drift(self):
        assert np.allclose(
            _run(_se3(), "ee_delta", _gamepad(thumbstick_left_y=0.1)), 0.0
        )


class TestGamepadGripperRetargeter:
    @staticmethod
    def _step(gripper, reset=False, **fields):
        return float(_run(gripper, "gripper_command", _gamepad(**fields), reset=reset))

    def test_gripper_toggles_on_west_rising_edge_only(self):
        gripper = GamepadGripperRetargeter(name="gripper")

        assert self._step(gripper) == pytest.approx(1.0)  # open (default)
        assert self._step(gripper, west_click=True) == pytest.approx(-1.0)  # close
        assert self._step(gripper, west_click=True) == pytest.approx(-1.0)  # held
        assert self._step(gripper) == pytest.approx(-1.0)  # released: stays closed
        assert self._step(gripper, west_click=True) == pytest.approx(1.0)  # open

    def test_absent_pad_yields_default_open(self):
        gripper = GamepadGripperRetargeter(name="gripper")

        assert self._step(gripper, absent=True) == pytest.approx(1.0)

    def test_reset_does_not_toggle_while_west_is_held(self):
        gripper = GamepadGripperRetargeter(name="gripper")

        assert self._step(gripper, west_click=True) == pytest.approx(-1.0)
        # Reset reopens the gripper; west still held is not a new press.
        assert self._step(gripper, reset=True, west_click=True) == pytest.approx(1.0)
        assert self._step(gripper, west_click=True) == pytest.approx(1.0)
        assert self._step(gripper) == pytest.approx(1.0)
        assert self._step(gripper, west_click=True) == pytest.approx(-1.0)

    def test_reset_with_absent_pad_keeps_the_edge_state(self):
        gripper = GamepadGripperRetargeter(name="gripper")

        assert self._step(gripper, west_click=True) == pytest.approx(-1.0)
        assert self._step(gripper, reset=True, absent=True) == pytest.approx(1.0)
        # West was held throughout, so it is not read as a new press when the pad returns.
        assert self._step(gripper, west_click=True) == pytest.approx(1.0)

    def test_other_buttons_do_not_affect_the_gripper(self):
        gripper = GamepadGripperRetargeter(name="gripper")

        assert self._step(
            gripper, shoulder_right_click=True, south_click=True
        ) == pytest.approx(1.0)


class TestGamepadToSe2Retargeter:
    def test_left_and_right_stick_combine(self):
        velocity = _run(
            _se2(),
            "base_command",
            _gamepad(thumbstick_left_y=1.0, thumbstick_right_x=0.5),
        )

        assert velocity[0] == pytest.approx(1.0)  # default v_x_sensitivity
        assert velocity[1] == pytest.approx(0.0)
        assert velocity[2] == pytest.approx(0.5)  # default omega_z_sensitivity

    def test_dead_zone_suppresses_stick_drift(self):
        assert np.allclose(
            _run(_se2(), "base_command", _gamepad(thumbstick_left_y=0.1)), 0.0
        )

    def test_absent_pad_yields_zero_velocity(self):
        assert np.allclose(_run(_se2(), "base_command", _gamepad(absent=True)), 0.0)
