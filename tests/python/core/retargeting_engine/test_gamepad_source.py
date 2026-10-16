# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GamepadSource DeviceIO converter.

Feeds GamepadSnapshot payloads built through the real schema bindings into the converter, with no
plugin or OpenXR runtime involved.
"""

import pytest

from isaaccapture.retargeting_engine.deviceio_source_nodes import GamepadSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.retargeting_engine.tensor_types import GamepadInput, GamepadInputIndex
from isaaccapture.schema import GamepadInputState, GamepadSnapshot


def _run(state):
    src = GamepadSource(name="gamepad")
    tg = TensorGroup(src.input_spec()["deviceio_gamepad"])
    tg[0] = state
    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_gamepad": tg}, outputs)
    return outputs["gamepad"]


def test_source_reads_the_generated_tracker():
    tracker = GamepadSource(name="gamepad").get_tracker()
    assert tracker.get_name() == "GamepadTracker"


def test_the_group_has_one_field_per_schema_field():
    assert len(GamepadInput().types) == 20
    assert len(GamepadInputIndex) == 20


def test_fields_pass_through_by_name():
    inputs = GamepadInputState(
        west_click=True,
        dpad_up_click=True,
        thumbstick_left_y=1.0,
        trigger_right_value=0.5,
    )
    gamepad = _run(GamepadSnapshot(inputs))

    held = {
        i.name
        for i in GamepadInputIndex
        if i.name.endswith("_CLICK") and gamepad[i] > 0.5
    }
    assert held == {"WEST_CLICK", "DPAD_UP_CLICK"}
    assert float(gamepad[GamepadInputIndex.THUMBSTICK_LEFT_Y]) == pytest.approx(1.0)
    assert float(gamepad[GamepadInputIndex.TRIGGER_RIGHT_VALUE]) == pytest.approx(0.5)
    assert float(gamepad[GamepadInputIndex.THUMBSTICK_RIGHT_X]) == 0.0


@pytest.mark.parametrize(
    "state",
    [None, GamepadSnapshot()],
    ids=["plugin-not-pushing", "no-device-open"],
)
def test_no_device_yields_absent_output(state):
    assert _run(state).is_none
