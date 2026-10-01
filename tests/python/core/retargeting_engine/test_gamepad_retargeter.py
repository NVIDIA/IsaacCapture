# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for GamepadToSe3RelRetargeter, GamepadGripperRetargeter, and
GamepadToSe2Retargeter, exercised through GamepadSource so a regression anywhere in the
schema -> source -> retargeter chain (a field rename, an index drift, a sign flip) fails
here, with no OpenXR device involved.
"""

import numpy as np
import pytest

from isaaccapture.retargeting_engine.deviceio_source_nodes import GamepadSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.execution_events import ExecutionEvents
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
)
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.retargeters import (
    GamepadGripperRetargeter,
    GamepadToSe2Retargeter,
    GamepadToSe2RetargeterConfig,
    GamepadToSe3RelRetargeter,
    GamepadToSe3RelRetargeterConfig,
)
from isaaccapture.schema import GamepadOutput

# Linux joystick-API axis indices, matching live_gamepad_tracker_impl.cpp / GamepadSource / the retargeters.
AXIS_LEFT_X, AXIS_LEFT_Y = 0, 1
AXIS_RIGHT_X, AXIS_RIGHT_Y = 3, 4
AXIS_DPAD_X, AXIS_DPAD_Y = 6, 7
BUTTON_X = 2


def _axes(by_index: dict[int, float]) -> list[float]:
    values = [0.0] * 8
    for index, value in by_index.items():
        values[index] = value
    return values


def _gamepad_source():
    return GamepadSource(name="gamepad")


def _run_source(
    src, pressed_buttons: list[int] | None, axes: list[float] | None = None
):
    """Feed raw button/axis state (None = inactive device) through GamepadSource.compute()."""
    state = (
        None
        if pressed_buttons is None
        else GamepadOutput(pressed_buttons, axes or [0.0] * 8, True)
    )

    input_spec = src.input_spec()
    tg = TensorGroup(input_spec["deviceio_gamepad"])
    tg[0] = state

    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_gamepad": tg}, outputs)
    return outputs


class TestGamepadToSe3RelRetargeter:
    def test_left_stick_up_produces_forward_delta(self):
        """Left stick pushed up (axis Y = -1) -> GamepadSource -> Se3Retargeter -> +X delta."""
        src = _gamepad_source()
        src_outputs = _run_source(src, [], axes=_axes({AXIS_LEFT_Y: -1.0}))
        assert not src_outputs["gamepad_axes"].is_none

        retargeter = GamepadToSe3RelRetargeter(
            GamepadToSe3RelRetargeterConfig(), name="se3"
        )
        out = {"ee_delta": _make_output_group(retargeter.output_spec()["ee_delta"])}
        retargeter.compute({"gamepad_axes": src_outputs["gamepad_axes"]}, out)

        delta = np.asarray(out["ee_delta"][0])
        assert delta[0] == pytest.approx(0.4)  # default pos_sensitivity
        assert np.allclose(delta[1:], 0.0)

    def test_opposing_axes_combine(self):
        """Left stick up (+X) and right stick up (+Z) held together combine on independent axes."""
        src = _gamepad_source()
        axes = _axes({AXIS_LEFT_Y: -1.0, AXIS_RIGHT_Y: -1.0})
        src_outputs = _run_source(src, [], axes=axes)

        retargeter = GamepadToSe3RelRetargeter(
            GamepadToSe3RelRetargeterConfig(), name="se3"
        )
        out = {"ee_delta": _make_output_group(retargeter.output_spec()["ee_delta"])}
        retargeter.compute({"gamepad_axes": src_outputs["gamepad_axes"]}, out)

        delta = np.asarray(out["ee_delta"][0])
        assert delta[0] == pytest.approx(0.4)  # left stick up: +X
        assert delta[2] == pytest.approx(0.4)  # right stick up: +Z
        assert delta[1] == pytest.approx(0.0)
        assert np.allclose(delta[3:], 0.0)  # no rotation axes deflected

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_stick_right_is_positive_y_and_yaw(self, direction):
        """Stick right gives +Y (left stick) and +yaw (right stick), the same direction as SE(2)."""
        src = _gamepad_source()
        axes = _axes({AXIS_LEFT_X: direction, AXIS_RIGHT_X: direction})
        src_outputs = _run_source(src, [], axes=axes)

        retargeter = GamepadToSe3RelRetargeter(
            GamepadToSe3RelRetargeterConfig(), name="se3"
        )
        out = {"ee_delta": _make_output_group(retargeter.output_spec()["ee_delta"])}
        retargeter.compute({"gamepad_axes": src_outputs["gamepad_axes"]}, out)

        delta = np.asarray(out["ee_delta"][0])
        assert np.sign(delta[1]) == direction
        assert np.sign(delta[5]) == direction
        assert np.allclose(delta[[0, 2, 3, 4]], 0.0)

    def test_inactive_device_yields_zero_delta(self):
        src = _gamepad_source()
        src_outputs = _run_source(src, None)

        se3 = GamepadToSe3RelRetargeter(GamepadToSe3RelRetargeterConfig(), name="se3")
        se3_out = {"ee_delta": _make_output_group(se3.output_spec()["ee_delta"])}
        se3.compute({"gamepad_axes": src_outputs["gamepad_axes"]}, se3_out)
        assert np.allclose(np.asarray(se3_out["ee_delta"][0]), 0.0)

    def test_dead_zone_suppresses_small_deflection(self):
        """A deflection smaller than the configured dead zone is treated as zero."""
        src = _gamepad_source()
        axes = _axes({AXIS_LEFT_Y: -0.005})
        src_outputs = _run_source(src, [], axes=axes)

        retargeter = GamepadToSe3RelRetargeter(
            GamepadToSe3RelRetargeterConfig(), name="se3"
        )
        out = {"ee_delta": _make_output_group(retargeter.output_spec()["ee_delta"])}
        retargeter.compute({"gamepad_axes": src_outputs["gamepad_axes"]}, out)

        assert np.allclose(np.asarray(out["ee_delta"][0]), 0.0)


class TestGamepadGripperRetargeter:
    def test_gripper_toggles_on_button_rising_edge_only(self):
        """X press/release/press across three frames toggles exactly on each rising edge."""
        src = _gamepad_source()
        retargeter = GamepadGripperRetargeter(name="gripper")

        def step(pressed_buttons):
            src_outputs = _run_source(src, pressed_buttons)
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            retargeter.compute({"gamepad_buttons": src_outputs["gamepad_buttons"]}, out)
            return float(out["gripper_command"][0])

        assert step([]) == pytest.approx(1.0)  # open (default)
        assert step([BUTTON_X]) == pytest.approx(-1.0)  # rising edge -> close
        assert step([BUTTON_X]) == pytest.approx(
            -1.0
        )  # held -> stays closed, no re-toggle
        assert step([]) == pytest.approx(-1.0)  # release -> stays closed
        assert step([BUTTON_X]) == pytest.approx(1.0)  # rising edge again -> open

    def test_inactive_device_yields_default_open(self):
        src = _gamepad_source()
        src_outputs = _run_source(src, None)

        gripper = GamepadGripperRetargeter(name="gripper")
        gripper_out = {
            "gripper_command": _make_output_group(
                gripper.output_spec()["gripper_command"]
            )
        }
        gripper.compute(
            {"gamepad_buttons": src_outputs["gamepad_buttons"]}, gripper_out
        )
        assert float(gripper_out["gripper_command"][0]) == pytest.approx(
            1.0
        )  # default open

    def test_reset_does_not_toggle_gripper_while_x_is_held(self):
        """X held across a reset frame is not a rising edge and must not toggle the gripper."""
        src = _gamepad_source()
        retargeter = GamepadGripperRetargeter(name="gripper")

        def step(pressed_buttons, reset=False):
            src_outputs = _run_source(src, pressed_buttons)
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            context = ComputeContext(execution_events=ExecutionEvents(reset=reset))
            retargeter.compute(
                {"gamepad_buttons": src_outputs["gamepad_buttons"]}, out, context
            )
            return float(out["gripper_command"][0])

        assert step([BUTTON_X]) == pytest.approx(-1.0)  # rising edge -> close
        # Reset resets the gripper to open, but X is still held -- not a new rising
        # edge, so this must not immediately re-close it.
        assert step([BUTTON_X], reset=True) == pytest.approx(1.0)
        assert step([BUTTON_X]) == pytest.approx(1.0)  # still held -> stays open
        assert step([]) == pytest.approx(1.0)  # release
        assert step([BUTTON_X]) == pytest.approx(-1.0)  # genuine rising edge -> close

    def test_reset_with_inactive_device_preserves_prior_edge_state(self):
        """A reset frame with no gamepad data must not clobber _prev_x_pressed."""
        src = _gamepad_source()
        retargeter = GamepadGripperRetargeter(name="gripper")

        def step(pressed_buttons, reset=False):
            src_outputs = _run_source(src, pressed_buttons)
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            context = ComputeContext(execution_events=ExecutionEvents(reset=reset))
            retargeter.compute(
                {"gamepad_buttons": src_outputs["gamepad_buttons"]}, out, context
            )
            return float(out["gripper_command"][0])

        assert step([BUTTON_X]) == pytest.approx(-1.0)  # rising edge -> close
        # Reset while the device is inactive (gamepad_buttons.is_none) -- must not
        # force _prev_x_pressed to False, or the next frame (X still held) would be
        # misread as a fresh rising edge.
        assert step(None, reset=True) == pytest.approx(1.0)  # gripper still resets
        assert step([BUTTON_X]) == pytest.approx(
            1.0
        )  # still held -> no spurious toggle

    def test_other_button_does_not_affect_gripper(self):
        """A button other than X shows up in gamepad_buttons but does not affect the gripper."""
        src = _gamepad_source()
        src_outputs = _run_source(src, [5])  # RB, not the gripper button

        gripper = GamepadGripperRetargeter(name="gripper")
        gripper_out = {
            "gripper_command": _make_output_group(
                gripper.output_spec()["gripper_command"]
            )
        }
        gripper.compute(
            {"gamepad_buttons": src_outputs["gamepad_buttons"]}, gripper_out
        )
        assert float(gripper_out["gripper_command"][0]) == pytest.approx(
            1.0
        )  # unaffected


class TestGamepadToSe2Retargeter:
    def test_left_and_right_stick_combine(self):
        """Left-stick-up (+v_x) and right-stick-right (+omega_z) held together -> combined base_command."""
        src = _gamepad_source()
        axes = _axes({AXIS_LEFT_Y: -1.0, AXIS_RIGHT_X: 0.5})
        src_outputs = _run_source(src, [], axes=axes)

        retargeter = GamepadToSe2Retargeter(GamepadToSe2RetargeterConfig(), name="se2")
        out = {
            "base_command": _make_output_group(retargeter.output_spec()["base_command"])
        }
        retargeter.compute({"gamepad_axes": src_outputs["gamepad_axes"]}, out)

        velocity = np.asarray(out["base_command"][0])
        assert velocity[0] == pytest.approx(1.0)  # default v_x_sensitivity
        assert velocity[1] == pytest.approx(0.0)
        assert velocity[2] == pytest.approx(0.5)  # default omega_z_sensitivity

    def test_dead_zone_suppresses_small_deflection(self):
        """A deflection smaller than the configured dead zone is treated as zero."""
        src = _gamepad_source()
        axes = _axes({AXIS_LEFT_Y: -0.005})
        src_outputs = _run_source(src, [], axes=axes)

        retargeter = GamepadToSe2Retargeter(GamepadToSe2RetargeterConfig(), name="se2")
        out = {
            "base_command": _make_output_group(retargeter.output_spec()["base_command"])
        }
        retargeter.compute({"gamepad_axes": src_outputs["gamepad_axes"]}, out)

        assert np.allclose(np.asarray(out["base_command"][0]), 0.0)

    def test_inactive_device_yields_zero_velocity(self):
        src = _gamepad_source()
        src_outputs = _run_source(src, None)

        retargeter = GamepadToSe2Retargeter(GamepadToSe2RetargeterConfig(), name="se2")
        out = {
            "base_command": _make_output_group(retargeter.output_spec()["base_command"])
        }
        retargeter.compute({"gamepad_axes": src_outputs["gamepad_axes"]}, out)

        assert np.allclose(np.asarray(out["base_command"][0]), 0.0)
