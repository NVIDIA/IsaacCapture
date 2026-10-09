# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for the SpaceMouse retargeters, exercised through SpaceMouseSource so a
regression anywhere in the schema -> source -> retargeter chain (a field rename, an index drift,
a sign flip) fails here.
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
from isaaccapture.schema import Point, SpaceMouseOutput

LEFT = [1]  # button 0 held
NONE_HELD = [0]


def _source_outputs(
    translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), buttons=(), connected=True
):
    """SpaceMouseSource outputs for one SpaceMouse state; ``connected=None`` means no data at all."""
    src = SpaceMouseSource(name="spacemouse")
    state = (
        None
        if connected is None
        else SpaceMouseOutput(
            Point(*translation), Point(*rotation), list(buttons), connected
        )
    )
    tg = TensorGroup(src.input_spec()["deviceio_spacemouse"])
    tg[0] = state
    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_spacemouse": tg}, outputs)
    return outputs


def _run(retargeter, source_outputs, reset=False):
    inputs = {name: source_outputs[name] for name in retargeter.input_spec()}
    out = {
        name: _make_output_group(gt) for name, gt in retargeter.output_spec().items()
    }
    retargeter.compute(
        inputs, out, ComputeContext(execution_events=ExecutionEvents(reset=reset))
    )
    return out


class TestSpaceMouseToSe3RelRetargeter:
    def _delta(self, **state):
        retargeter = SpaceMouseToSe3RelRetargeter(
            SpaceMouseToSe3RelRetargeterConfig(), name="se3"
        )
        return np.asarray(_run(retargeter, _source_outputs(**state))["ee_delta"][0])

    def test_translation_maps_to_position_delta(self):
        delta = self._delta(translation=(1.0, 0.5, 0.25))

        assert delta[:3] == pytest.approx([0.2, 0.4, -0.1])  # [+ty, +tx, -tz] * 0.4
        assert delta[3:] == pytest.approx([0.0, 0.0, 0.0])

    @pytest.mark.parametrize(
        ("rotation", "axis", "sign"),
        [
            ((0.0, 0.5, 0.0), 0, 1.0),
            ((0.5, 0.0, 0.0), 1, 1.0),
            ((0.0, 0.0, 0.5), 2, -1.0),
        ],
        ids=["roll-from-y", "pitch-from-x", "yaw-from-negated-z"],
    )
    def test_rotation_maps_to_rotation_delta(self, rotation, axis, sign):
        delta = self._delta(rotation=rotation)

        expected = np.zeros(3)
        expected[axis] = sign * 0.4  # 0.5 * rot_sensitivity 0.8, single-axis rotvec
        assert delta[3:] == pytest.approx(expected)

    def test_no_device_yields_zero_delta(self):
        assert np.allclose(self._delta(connected=False), 0.0)
        assert np.allclose(self._delta(connected=None), 0.0)


class TestSpaceMouseToSe2Retargeter:
    def _velocity(self, **state):
        retargeter = SpaceMouseToSe2Retargeter(
            SpaceMouseToSe2RetargeterConfig(), name="se2"
        )
        return np.asarray(_run(retargeter, _source_outputs(**state))["base_command"][0])

    def test_axes_map_to_base_velocity(self):
        velocity = self._velocity(
            translation=(0.25, 0.5, 1.0), rotation=(1.0, 0.75, 1.0)
        )

        assert velocity == pytest.approx([0.5, 0.25, 0.75])  # [ty, tx, ry]

    def test_no_device_yields_zero_velocity(self):
        assert np.allclose(self._velocity(connected=False), 0.0)


class TestSpaceMouseGripperRetargeter:
    def _stepper(self):
        retargeter = SpaceMouseGripperRetargeter(name="gripper")

        def step(buttons=NONE_HELD, reset=False, connected=True):
            out = _run(
                retargeter,
                _source_outputs(buttons=buttons, connected=connected),
                reset=reset,
            )
            return float(out["gripper_command"][0])

        return step

    def test_toggles_on_each_press_only(self):
        step = self._stepper()

        assert step() == 1.0  # open
        assert step(LEFT) == -1.0  # press: close
        assert step(LEFT) == -1.0  # held: no toggle
        assert step() == -1.0  # release
        assert step(LEFT) == 1.0  # press: open

    def test_reset_reopens_and_a_held_button_does_not_toggle_again(self):
        step = self._stepper()

        assert step(LEFT) == -1.0
        assert step(LEFT, reset=True) == 1.0
        assert step(LEFT) == 1.0
        assert step() == 1.0
        assert step(LEFT) == -1.0

    def test_a_button_held_while_the_device_was_away_is_not_a_new_press(self):
        step = self._stepper()

        assert step(LEFT) == -1.0
        assert step(connected=False, reset=True) == 1.0  # reset still reopens
        assert step(LEFT) == 1.0  # still held from before: no toggle

    def test_no_device_holds_the_current_state(self):
        step = self._stepper()

        assert step(connected=None) == 1.0
        assert step(LEFT) == -1.0
        assert step(connected=False) == -1.0
