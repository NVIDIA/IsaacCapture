# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gamepad Source Node - DeviceIO to Retargeting Engine converter.

Reads a Linux joystick-API gamepad in process (no plugin process, no OpenXR extension, no
``input`` group membership) and converts its raw GamepadOutput (button/axis state) to two standard
outputs: a button-press bitmap and an axis-value array. Carries no semantic mapping -- which
button/axis means what (a stick, a trigger, a toggle) is entirely up to the consuming
retargeter.

Buttons are sampled: each frame reports the buttons held when the device was read, so a press and
release between two reads (a very quick tap, more likely in slow frames) is not seen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType, TensorGroupType
from ..tensor_types import DLDataType, NDArrayType
from .deviceio_tensor_types import DeviceIOGamepadOutputTracked
from .interface import IDeviceIOSource

if TYPE_CHECKING:
    from isaaccapture.deviceio import ITracker
    from isaaccapture.schema import GamepadOutput

# Linux joystick JS_EVENT_BUTTON indices go up to 31 on every driver observed in
# practice (Xbox-style pads report ~11); 32 covers the full range with headroom.
GAMEPAD_BUTTONS_BITMAP_SIZE = 32

# Fixed axis-array size returned to consumers, independent of how many axes the
# connected device actually reports (the tracker queries JSIOCGAXES and reports
# fewer/more; this source pads with 0.0 or truncates to fit).
GAMEPAD_AXES_SIZE = 8


def GamepadButtonsType() -> TensorGroupType:
    """Type for the "gamepad_buttons" output: a 32-entry uint8 bitmap indexed by joystick button number."""
    return TensorGroupType(
        "gamepad_buttons",
        [
            NDArrayType(
                "bitmap",
                shape=(GAMEPAD_BUTTONS_BITMAP_SIZE,),
                dtype=DLDataType.UINT,
                dtype_bits=8,
            )
        ],
    )


def GamepadAxesType() -> TensorGroupType:
    """Type for the "gamepad_axes" output: a fixed-size float32 array of joystick axis values."""
    return TensorGroupType(
        "gamepad_axes",
        [
            NDArrayType(
                "axes",
                shape=(GAMEPAD_AXES_SIZE,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            )
        ],
    )


class GamepadSource(IDeviceIOSource):
    """
    Stateless converter: DeviceIO GamepadOutput → button-bitmap / axis-array tensors.

    Inputs:
        - "deviceio_gamepad": Raw GamepadOutput flatbuffer from GamepadTracker

    Outputs (Optional — absent while no gamepad is connected):
        - "gamepad_buttons": OptionalTensorGroup, a 32-entry uint8 bitmap indexed by
          Linux joystick button number (1 = held, 0 = released).
        - "gamepad_axes": OptionalTensorGroup, a fixed-size float32 array of axis
          values in [-1, 1], padded/truncated to a fixed length independent of the
          connected device's actual axis count.

    Usage:
        # In TeleopSession, the gamepad tracker is discovered from the pipeline;
        # data is polled via poll_tracker. Or manually:
        tracked = gamepad_tracker.get_gamepad_data(session)
        result = gamepad_source_node({
            "deviceio_gamepad": TensorGroup(DeviceIOGamepadOutputTracked(), [tracked])
        })
    """

    def __init__(self, name: str, device_path: str | None = None) -> None:
        """Initialize stateless gamepad source node.

        Creates a GamepadTracker instance for TeleopSession to discover and use.

        Args:
            name: Unique name for this source node
            device_path: Joystick device to read (e.g. ``/dev/input/js0``). ``None`` picks the
                first connected joystick, and picks again after a disconnect.
        """
        import isaaccapture.deviceio as deviceio

        self._gamepad_tracker = deviceio.GamepadTracker(device_path or "")
        super().__init__(name)

    def get_tracker(self) -> ITracker:
        """Get the GamepadTracker instance.

        Returns:
            The GamepadTracker instance for TeleopSession to initialize
        """
        return self._gamepad_tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the gamepad tracker and return input data.

        Args:
            deviceio_session: The active DeviceIO session.

        Returns:
            Dict with "deviceio_gamepad" TensorGroup containing GamepadOutput | None.
        """
        state = self._gamepad_tracker.get_gamepad_data(deviceio_session)
        tg = TensorGroup(DeviceIOGamepadOutputTracked())
        tg[0] = state
        return {"deviceio_gamepad": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare DeviceIO gamepad input."""
        return {
            "deviceio_gamepad": DeviceIOGamepadOutputTracked(),
        }

    def output_spec(self) -> RetargeterIOType:
        """Declare standard gamepad outputs (Optional — may be absent)."""
        return {
            "gamepad_buttons": OptionalType(GamepadButtonsType()),
            "gamepad_axes": OptionalType(GamepadAxesType()),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """
        Convert DeviceIO GamepadOutput to the standard gamepad outputs.

        Calls ``set_none()`` on both outputs while no gamepad is connected.

        Args:
            inputs: Dict with "deviceio_gamepad" containing GamepadOutput | None
            outputs: Dict with "gamepad_buttons" and "gamepad_axes" OptionalTensorGroups
            context: Shared ComputeContext for the current step (carries GraphTime).
        """
        import numpy as np

        state: GamepadOutput | None = inputs["deviceio_gamepad"][0]

        buttons_out = outputs["gamepad_buttons"]
        axes_out = outputs["gamepad_axes"]
        if state is None:
            buttons_out.set_none()
            axes_out.set_none()
            return

        bitmap = np.zeros(GAMEPAD_BUTTONS_BITMAP_SIZE, dtype=np.uint8)
        for code in state.pressed_buttons:
            if code < GAMEPAD_BUTTONS_BITMAP_SIZE:
                bitmap[code] = 1
        buttons_out[0] = bitmap

        axes = np.zeros(GAMEPAD_AXES_SIZE, dtype=np.float32)
        reported = np.asarray(state.axes, dtype=np.float32)
        count = min(reported.shape[0], GAMEPAD_AXES_SIZE)
        axes[:count] = reported[:count]
        axes_out[0] = axes
