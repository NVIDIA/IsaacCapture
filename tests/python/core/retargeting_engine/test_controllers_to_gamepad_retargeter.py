# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllersToGamepadRetargeter: XR controllers in, one GamepadInput out."""

import numpy as np
import pytest

from isaaccapture.retargeters import (
    ControllersToGamepadRetargeter,
    ControllersToGamepadRetargeterConfig,
    GamepadGripperRetargeter,
)
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import (
    OptionalTensorGroup,
    TensorGroup,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    GamepadInputIndex,
)


def _controller(
    primary=0.0,
    secondary=0.0,
    stick=(0.0, 0.0),
    stick_click=0.0,
    menu=0.0,
    squeeze=0.0,
    trigger=0.0,
):
    tg = TensorGroup(ControllerInput())
    tg[ControllerInputIndex.GRIP_POSITION] = np.zeros(3, dtype=np.float32)
    tg[ControllerInputIndex.GRIP_ORIENTATION] = np.array([0, 0, 0, 1], dtype=np.float32)
    tg[ControllerInputIndex.GRIP_IS_VALID] = True
    tg[ControllerInputIndex.AIM_POSITION] = np.zeros(3, dtype=np.float32)
    tg[ControllerInputIndex.AIM_ORIENTATION] = np.array([0, 0, 0, 1], dtype=np.float32)
    tg[ControllerInputIndex.AIM_IS_VALID] = True
    tg[ControllerInputIndex.PRIMARY_CLICK] = float(primary)
    tg[ControllerInputIndex.SECONDARY_CLICK] = float(secondary)
    tg[ControllerInputIndex.THUMBSTICK_X] = float(stick[0])
    tg[ControllerInputIndex.THUMBSTICK_Y] = float(stick[1])
    tg[ControllerInputIndex.THUMBSTICK_CLICK] = float(stick_click)
    tg[ControllerInputIndex.MENU_CLICK] = float(menu)
    tg[ControllerInputIndex.SQUEEZE_VALUE] = float(squeeze)
    tg[ControllerInputIndex.TRIGGER_VALUE] = float(trigger)
    return tg


def _absent():
    return OptionalTensorGroup(ControllerInput())


def _run(left, right, **config):
    retargeter = ControllersToGamepadRetargeter(
        ControllersToGamepadRetargeterConfig(**config), name="to_gamepad"
    )
    outputs = {
        name: _make_output_group(t) for name, t in retargeter.output_spec().items()
    }
    retargeter.compute({"controller_left": left, "controller_right": right}, outputs)
    return outputs["gamepad"]


def _held(gamepad):
    return {
        i.name.removesuffix("_CLICK").lower()
        for i in GamepadInputIndex
        if i.name.endswith("_CLICK") and float(gamepad[i]) > 0.5
    }


def test_no_controllers_yields_absent_output():
    assert _run(_absent(), _absent()).is_none


def test_sticks_and_triggers_map_by_hand_and_a_missing_hand_reads_centered():
    gamepad = _run(_controller(stick=(0.3, 0.9), trigger=0.7), _absent())

    assert float(gamepad[GamepadInputIndex.THUMBSTICK_LEFT_X]) == pytest.approx(0.3)
    # OpenXR's up-positive Y passes straight through.
    assert float(gamepad[GamepadInputIndex.THUMBSTICK_LEFT_Y]) == pytest.approx(0.9)
    assert float(gamepad[GamepadInputIndex.TRIGGER_LEFT_VALUE]) == pytest.approx(0.7)
    assert float(gamepad[GamepadInputIndex.THUMBSTICK_RIGHT_X]) == 0.0
    assert float(gamepad[GamepadInputIndex.THUMBSTICK_RIGHT_Y]) == 0.0
    assert _held(gamepad) == set()


def test_face_buttons_sit_where_an_xbox_pad_has_them():
    left = _controller(primary=1, secondary=1, stick_click=1, menu=1)
    right = _controller(primary=1, secondary=1, stick_click=1, menu=1)

    assert _held(_run(left, right)) == {
        "south",
        "east",
        "west",
        "north",
        "thumbstick_left",
        "thumbstick_right",
        "menu",
    }
    assert _held(_run(_absent(), _controller(primary=1))) == {"south"}
    assert _held(_run(_controller(primary=1), _absent())) == {"west"}


def test_grips_are_shoulders_above_the_threshold():
    gamepad = _run(_controller(squeeze=0.4), _controller(squeeze=0.6))

    assert _held(gamepad) == {"shoulder_right"}


def test_without_a_dpad_stick_the_dpad_is_never_pressed():
    gamepad = _run(_absent(), _controller(stick=(0.0, 1.0), squeeze=1.0))

    assert _held(gamepad) == {"shoulder_right"}
    assert float(gamepad[GamepadInputIndex.THUMBSTICK_RIGHT_Y]) == 1.0


def test_dpad_stick_is_the_dpad_while_its_grip_is_held():
    gripped = _run(
        _absent(), _controller(stick=(-0.8, 0.9), squeeze=1.0), dpad_stick="right"
    )

    assert _held(gripped) == {"dpad_up", "dpad_left"}
    assert float(gripped[GamepadInputIndex.THUMBSTICK_RIGHT_X]) == 0.0
    assert float(gripped[GamepadInputIndex.THUMBSTICK_RIGHT_Y]) == 0.0

    released = _run(_absent(), _controller(stick=(-0.8, 0.9)), dpad_stick="right")
    assert _held(released) == set()
    assert float(released[GamepadInputIndex.THUMBSTICK_RIGHT_X]) == pytest.approx(-0.8)


def test_an_unknown_dpad_stick_is_rejected():
    with pytest.raises(ValueError, match="dpad_stick"):
        ControllersToGamepadRetargeter(
            ControllersToGamepadRetargeterConfig(dpad_stick="center"), name="bad"
        )


def test_left_x_toggles_the_gamepad_gripper():
    gripper = GamepadGripperRetargeter(name="gripper")

    def step(left):
        out = {
            "gripper_command": _make_output_group(
                gripper.output_spec()["gripper_command"]
            )
        }
        gripper.compute({"gamepad": _run(left, _absent())}, out)
        return float(out["gripper_command"][0])

    assert step(_controller()) == pytest.approx(1.0)
    assert step(_controller(primary=1)) == pytest.approx(-1.0)
