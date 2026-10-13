# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Controllers-to-Gamepad Retargeter Module.

Joins a pair of XR controllers (ControllersSource) into one gamepad, so the gamepad retargeters drive
a robot from controllers unchanged. A gamepad is a generic controller split in two, so this puts the
halves back together.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group_type import OptionalType
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    GamepadInput,
    GamepadInputIndex,
)

C = ControllerInputIndex
G = GamepadInputIndex

# Each controller's buttons, as (gamepad field, controller field). Quest's A/B/X/Y sit where an Xbox
# pad's do: A and B on the right controller, X and Y on the left.
_LEFT_BUTTONS = (
    (G.WEST_CLICK, C.PRIMARY_CLICK),
    (G.NORTH_CLICK, C.SECONDARY_CLICK),
    (G.THUMBSTICK_LEFT_CLICK, C.THUMBSTICK_CLICK),
    # The right controller's menu button is the system button, so only the left one is menu.
    (G.MENU_CLICK, C.MENU_CLICK),
)
_RIGHT_BUTTONS = (
    (G.SOUTH_CLICK, C.PRIMARY_CLICK),
    (G.EAST_CLICK, C.SECONDARY_CLICK),
    (G.THUMBSTICK_RIGHT_CLICK, C.THUMBSTICK_CLICK),
)


@dataclass
class ControllersToGamepadRetargeterConfig:
    """Configuration for the controllers-to-gamepad retargeter."""

    # Grip squeeze above which the grip counts as a held shoulder button.
    grip_threshold: float = 0.5
    # Controllers have no D-pad. "left" or "right" turns that thumbstick into D-pad presses while the
    # same hand's grip is held; the stick then drives no axes and that shoulder reads released.
    dpad_stick: Optional[Literal["left", "right"]] = None
    # Stick deflection beyond which a D-pad direction counts as pressed.
    dpad_threshold: float = 0.5


class ControllersToGamepadRetargeter(BaseRetargeter):
    """
    Maps left and right XR controllers to a GamepadInput ("gamepad").

    Bindings:
        Thumbsticks -> thumbstick_[left|right]_[x|y] (both up-positive, as in OpenXR)
        Triggers -> trigger_[left|right]_value
        Grips -> shoulder_[left|right]_click
        A, B (right) -> south, east;  X, Y (left) -> west, north
        Thumbstick clicks -> thumbstick_[left|right]_click;  left menu -> menu
        D-pad -> one thumbstick while its grip is held, when dpad_stick is set

    back_click is never pressed. A controller that is absent reads as released and centered; the
    output is absent only while both are.
    """

    def __init__(self, config: ControllersToGamepadRetargeterConfig, name: str) -> None:
        if config.dpad_stick not in (None, "left", "right"):
            raise ValueError(
                f"dpad_stick must be None, 'left' or 'right', not {config.dpad_stick!r}"
            )
        self._config = config
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "controller_left": OptionalType(ControllerInput()),
            "controller_right": OptionalType(ControllerInput()),
        }

    def output_spec(self) -> RetargeterIOType:
        return {"gamepad": OptionalType(GamepadInput())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        gamepad = outputs["gamepad"]
        left, right = inputs["controller_left"], inputs["controller_right"]
        if left.is_none and right.is_none:
            gamepad.set_none()
            return

        values = dict.fromkeys(GamepadInputIndex, 0.0)
        hands = (
            (left, _LEFT_BUTTONS, "left", G.SHOULDER_LEFT_CLICK, G.THUMBSTICK_LEFT_X,
             G.THUMBSTICK_LEFT_Y, G.TRIGGER_LEFT_VALUE),
            (right, _RIGHT_BUTTONS, "right", G.SHOULDER_RIGHT_CLICK, G.THUMBSTICK_RIGHT_X,
             G.THUMBSTICK_RIGHT_Y, G.TRIGGER_RIGHT_VALUE),
        )  # fmt: skip
        for controller, buttons, side, shoulder, stick_x, stick_y, trigger in hands:
            if controller.is_none:
                continue
            for field, source in buttons:
                values[field] = float(float(controller[source]) > 0.5)
            x = float(controller[C.THUMBSTICK_X])
            y = float(controller[C.THUMBSTICK_Y])
            gripped = float(controller[C.SQUEEZE_VALUE]) > self._config.grip_threshold
            values[trigger] = float(controller[C.TRIGGER_VALUE])
            if gripped and self._config.dpad_stick == side:
                threshold = self._config.dpad_threshold
                values[G.DPAD_UP_CLICK] = float(y > threshold)
                values[G.DPAD_DOWN_CLICK] = float(y < -threshold)
                values[G.DPAD_RIGHT_CLICK] = float(x > threshold)
                values[G.DPAD_LEFT_CLICK] = float(x < -threshold)
                continue
            values[shoulder] = float(gripped)
            values[stick_x] = x
            values[stick_y] = y

        for field, value in values.items():
            gamepad[field] = value
