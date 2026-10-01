# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for SpaceMouseToSe3RelRetargeter, SpaceMouseGripperRetargeter, and
SpaceMouseToSe2Retargeter, exercised through SpaceMouseSource so a regression anywhere in
the schema -> source -> retargeter chain (a field rename, an index drift, a sign flip)
fails here, with no OpenXR device involved.
"""

import numpy as np
import pytest

from isaaccapture.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.execution_events import ExecutionEvents
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
)
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.retargeters import (
    SpaceMouseGripperRetargeter,
    SpaceMouseToSe2Retargeter,
    SpaceMouseToSe2RetargeterConfig,
    SpaceMouseToSe3RelRetargeter,
    SpaceMouseToSe3RelRetargeterConfig,
)
from isaaccapture.schema import SpaceMouseOutput

BUTTON_LEFT = 0


def _spacemouse_source():
    return SpaceMouseSource(name="spacemouse")


def _run_source(
    src,
    translation: list[float] | None,
    rotation: list[float] | None = None,
    pressed_buttons: list[int] | None = None,
):
    """Feed raw axis/button state (None translation = inactive device) through SpaceMouseSource.compute()."""
    state = (
        None
        if translation is None
        else SpaceMouseOutput(
            translation, rotation or [0.0, 0.0, 0.0], pressed_buttons or [], True
        )
    )

    input_spec = src.input_spec()
    tg = TensorGroup(input_spec["deviceio_spacemouse"])
    tg[0] = state

    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_spacemouse": tg}, outputs)
    return outputs


class TestSpaceMouseToSe3RelRetargeter:
    def test_translation_produces_position_delta(self):
        """translation[0] -> +Y, translation[1] -> +X, translation[2] -> -Z (inverted)."""
        src = _spacemouse_source()
        src_outputs = _run_source(src, translation=[1.0, 1.0, 1.0])

        retargeter = SpaceMouseToSe3RelRetargeter(
            SpaceMouseToSe3RelRetargeterConfig(), name="se3"
        )
        out = {"ee_delta": _make_output_group(retargeter.output_spec()["ee_delta"])}
        retargeter.compute(
            {
                "spacemouse_translation": src_outputs["spacemouse_translation"],
                "spacemouse_rotation": src_outputs["spacemouse_rotation"],
            },
            out,
        )

        delta = np.asarray(out["ee_delta"][0])
        assert delta[0] == pytest.approx(0.4)  # +X from translation[1]
        assert delta[1] == pytest.approx(0.4)  # +Y from translation[0]
        assert delta[2] == pytest.approx(-0.4)  # -Z from translation[2] (inverted)

    def test_inactive_device_yields_zero_delta(self):
        src = _spacemouse_source()
        src_outputs = _run_source(src, translation=None)

        se3 = SpaceMouseToSe3RelRetargeter(
            SpaceMouseToSe3RelRetargeterConfig(), name="se3"
        )
        se3_out = {"ee_delta": _make_output_group(se3.output_spec()["ee_delta"])}
        se3.compute(
            {
                "spacemouse_translation": src_outputs["spacemouse_translation"],
                "spacemouse_rotation": src_outputs["spacemouse_rotation"],
            },
            se3_out,
        )
        assert np.allclose(np.asarray(se3_out["ee_delta"][0]), 0.0)


class TestSpaceMouseGripperRetargeter:
    def test_gripper_toggles_on_button_rising_edge_only(self):
        """Left-button press/release/press across three frames toggles on each rising edge."""
        src = _spacemouse_source()
        retargeter = SpaceMouseGripperRetargeter(name="gripper")

        def step(pressed_buttons):
            src_outputs = _run_source(
                src, translation=[0.0, 0.0, 0.0], pressed_buttons=pressed_buttons
            )
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            retargeter.compute(
                {"spacemouse_buttons": src_outputs["spacemouse_buttons"]}, out
            )
            return float(out["gripper_command"][0])

        assert step([]) == pytest.approx(1.0)  # open (default)
        assert step([BUTTON_LEFT]) == pytest.approx(-1.0)  # rising edge -> close
        assert step([BUTTON_LEFT]) == pytest.approx(
            -1.0
        )  # held -> stays closed, no re-toggle
        assert step([]) == pytest.approx(-1.0)  # release -> stays closed
        assert step([BUTTON_LEFT]) == pytest.approx(1.0)  # rising edge again -> open

    def test_inactive_device_yields_default_open(self):
        src = _spacemouse_source()
        src_outputs = _run_source(src, translation=None)

        gripper = SpaceMouseGripperRetargeter(name="gripper")
        gripper_out = {
            "gripper_command": _make_output_group(
                gripper.output_spec()["gripper_command"]
            )
        }
        gripper.compute(
            {"spacemouse_buttons": src_outputs["spacemouse_buttons"]}, gripper_out
        )
        assert float(gripper_out["gripper_command"][0]) == pytest.approx(
            1.0
        )  # default open

    def test_reset_does_not_toggle_gripper_while_left_is_held(self):
        """Left button held across a reset frame is not a rising edge and must not toggle."""
        src = _spacemouse_source()
        retargeter = SpaceMouseGripperRetargeter(name="gripper")

        def step(pressed_buttons, reset=False):
            src_outputs = _run_source(
                src, translation=[0.0, 0.0, 0.0], pressed_buttons=pressed_buttons
            )
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            context = ComputeContext(execution_events=ExecutionEvents(reset=reset))
            retargeter.compute(
                {"spacemouse_buttons": src_outputs["spacemouse_buttons"]}, out, context
            )
            return float(out["gripper_command"][0])

        assert step([BUTTON_LEFT]) == pytest.approx(-1.0)  # rising edge -> close
        # Reset resets the gripper to open, but the left button is still held -- not
        # a new rising edge, so this must not immediately re-close it.
        assert step([BUTTON_LEFT], reset=True) == pytest.approx(1.0)
        assert step([BUTTON_LEFT]) == pytest.approx(1.0)  # still held -> stays open
        assert step([]) == pytest.approx(1.0)  # release
        assert step([BUTTON_LEFT]) == pytest.approx(
            -1.0
        )  # genuine rising edge -> close

    def test_reset_with_inactive_device_preserves_prior_edge_state(self):
        """A reset frame with no spacemouse data must not clobber _prev_left_pressed."""
        src = _spacemouse_source()
        retargeter = SpaceMouseGripperRetargeter(name="gripper")

        def step(translation, pressed_buttons=None, reset=False):
            src_outputs = _run_source(
                src, translation=translation, pressed_buttons=pressed_buttons
            )
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            context = ComputeContext(execution_events=ExecutionEvents(reset=reset))
            retargeter.compute(
                {"spacemouse_buttons": src_outputs["spacemouse_buttons"]}, out, context
            )
            return float(out["gripper_command"][0])

        assert step([0.0, 0.0, 0.0], [BUTTON_LEFT]) == pytest.approx(
            -1.0
        )  # rising edge -> close
        # Reset while the device is inactive (spacemouse_buttons.is_none) -- must
        # not force _prev_left_pressed to False, or the next frame (left button
        # still held) would be misread as a fresh rising edge.
        assert step(None, reset=True) == pytest.approx(1.0)  # gripper still resets
        assert step([0.0, 0.0, 0.0], [BUTTON_LEFT]) == pytest.approx(
            1.0
        )  # still held -> no spurious toggle


class TestSpaceMouseToSe2Retargeter:
    def test_translation_and_rotation_combine(self):
        """translation -> v_x/v_y, rotation[1] -> omega_z."""
        src = _spacemouse_source()
        src_outputs = _run_source(
            src, translation=[1.0, 1.0, 0.0], rotation=[0.0, 1.0, 0.0]
        )

        retargeter = SpaceMouseToSe2Retargeter(
            SpaceMouseToSe2RetargeterConfig(), name="se2"
        )
        out = {
            "base_command": _make_output_group(retargeter.output_spec()["base_command"])
        }
        retargeter.compute(
            {
                "spacemouse_translation": src_outputs["spacemouse_translation"],
                "spacemouse_rotation": src_outputs["spacemouse_rotation"],
            },
            out,
        )

        velocity = np.asarray(out["base_command"][0])
        assert velocity[0] == pytest.approx(1.0)  # v_x from translation[1]
        assert velocity[1] == pytest.approx(1.0)  # v_y from translation[0]
        assert velocity[2] == pytest.approx(1.0)  # omega_z from rotation[1]

    def test_inactive_device_yields_zero_velocity(self):
        src = _spacemouse_source()
        src_outputs = _run_source(src, translation=None)

        retargeter = SpaceMouseToSe2Retargeter(
            SpaceMouseToSe2RetargeterConfig(), name="se2"
        )
        out = {
            "base_command": _make_output_group(retargeter.output_spec()["base_command"])
        }
        retargeter.compute(
            {
                "spacemouse_translation": src_outputs["spacemouse_translation"],
                "spacemouse_rotation": src_outputs["spacemouse_rotation"],
            },
            out,
        )

        assert np.allclose(np.asarray(out["base_command"][0]), 0.0)
