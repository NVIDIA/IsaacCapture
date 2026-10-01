# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SpaceMouse Source Node - DeviceIO to Retargeting Engine converter.

Reads a 3Dconnexion SpaceMouse in process from its HID device (no plugin process, no
OpenXR extension) and converts its raw SpaceMouseOutput (axis/button state) to three standard
outputs: translation axes, rotation axes, and a button-press bitmap. Carries no semantic
mapping -- which axis/button means what (a position delta, a rotation delta, a toggle) is
entirely up to the consuming retargeter. Reading ``/dev/hidraw*`` needs a udev rule granting
the user access.

Buttons are sampled: each frame reports the buttons held when the device was read, so a press and
release between two reads (a very quick tap, more likely in slow frames) is not seen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType, TensorGroupType
from ..tensor_types import DLDataType, NDArrayType
from .deviceio_tensor_types import DeviceIOSpaceMouseOutputTracked
from .interface import IDeviceIOSource

if TYPE_CHECKING:
    from isaaccapture.deviceio import ITracker
    from isaaccapture.schema import SpaceMouseOutput

# Fixed axis-array size for both translation and rotation: [x, y, z].
SPACEMOUSE_AXES_SIZE = 3

# Button bitmap size: covers every bit position the device's button-report byte can set.
SPACEMOUSE_BUTTONS_BITMAP_SIZE = 8


def SpaceMouseTranslationType() -> TensorGroupType:
    """Type for the "spacemouse_translation" output: a 3-entry float32 [x, y, z] array."""
    return TensorGroupType(
        "spacemouse_translation",
        [
            NDArrayType(
                "axes",
                shape=(SPACEMOUSE_AXES_SIZE,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            )
        ],
    )


def SpaceMouseRotationType() -> TensorGroupType:
    """Type for the "spacemouse_rotation" output: a 3-entry float32 [x, y, z] array."""
    return TensorGroupType(
        "spacemouse_rotation",
        [
            NDArrayType(
                "axes",
                shape=(SPACEMOUSE_AXES_SIZE,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            )
        ],
    )


def SpaceMouseButtonsType() -> TensorGroupType:
    """Type for the "spacemouse_buttons" output: an 8-entry uint8 bitmap indexed by button index."""
    return TensorGroupType(
        "spacemouse_buttons",
        [
            NDArrayType(
                "bitmap",
                shape=(SPACEMOUSE_BUTTONS_BITMAP_SIZE,),
                dtype=DLDataType.UINT,
                dtype_bits=8,
            )
        ],
    )


class SpaceMouseSource(IDeviceIOSource):
    """
    Stateless converter: DeviceIO SpaceMouseOutput → translation / rotation / button tensors.

    Inputs:
        - "deviceio_spacemouse": Raw SpaceMouseOutput flatbuffer from SpaceMouseTracker

    Outputs (Optional — absent while no SpaceMouse is connected):
        - "spacemouse_translation": OptionalTensorGroup, a 3-entry float32 [x, y, z]
          array in [-1, 1].
        - "spacemouse_rotation": OptionalTensorGroup, a 3-entry float32 [x, y, z]
          array in [-1, 1].
        - "spacemouse_buttons": OptionalTensorGroup, an 8-entry uint8 bitmap indexed
          by button index (1 = held, 0 = released).

    Usage:
        # In TeleopSession, the spacemouse tracker is discovered from the pipeline;
        # data is polled via poll_tracker. Or manually:
        state = spacemouse_tracker.get_spacemouse_data(session)
        result = spacemouse_source_node({
            "deviceio_spacemouse": TensorGroup(DeviceIOSpaceMouseOutputTracked(), [state])
        })
    """

    def __init__(
        self,
        name: str,
        device_path: str | None = None,
        combined_report: bool = False,
    ) -> None:
        """Initialize stateless spacemouse source node.

        Creates a SpaceMouseTracker instance for TeleopSession to discover and use.

        Args:
            name: Unique name for this source node
            device_path: HID device to read (e.g. ``/dev/hidraw3``). ``None`` picks the first
                connected SpaceMouse of a validated model, and picks again after a disconnect.
            combined_report: Only used with an explicit ``device_path``: the device packs
                translation and rotation into one report (3Dconnexion Universal Receiver).
        """
        import isaaccapture.deviceio as deviceio

        self._spacemouse_tracker = deviceio.SpaceMouseTracker(
            device_path or "", combined_report
        )
        super().__init__(name)

    def get_tracker(self) -> ITracker:
        """Get the SpaceMouseTracker instance.

        Returns:
            The SpaceMouseTracker instance for TeleopSession to initialize
        """
        return self._spacemouse_tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the spacemouse tracker and return input data.

        Args:
            deviceio_session: The active DeviceIO session.

        Returns:
            Dict with "deviceio_spacemouse" TensorGroup containing SpaceMouseOutput | None.
        """
        state = self._spacemouse_tracker.get_spacemouse_data(deviceio_session)
        tg = TensorGroup(DeviceIOSpaceMouseOutputTracked())
        tg[0] = state
        return {"deviceio_spacemouse": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare DeviceIO spacemouse input."""
        return {
            "deviceio_spacemouse": DeviceIOSpaceMouseOutputTracked(),
        }

    def output_spec(self) -> RetargeterIOType:
        """Declare standard spacemouse outputs (Optional — may be absent)."""
        return {
            "spacemouse_translation": OptionalType(SpaceMouseTranslationType()),
            "spacemouse_rotation": OptionalType(SpaceMouseRotationType()),
            "spacemouse_buttons": OptionalType(SpaceMouseButtonsType()),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """
        Convert DeviceIO SpaceMouseOutput to the standard spacemouse outputs.

        Calls ``set_none()`` on all three outputs while no SpaceMouse is connected.

        Args:
            inputs: Dict with "deviceio_spacemouse" containing SpaceMouseOutput | None
            outputs: Dict with "spacemouse_translation", "spacemouse_rotation", and
                "spacemouse_buttons" OptionalTensorGroups
            context: Shared ComputeContext for the current step (carries GraphTime).
        """
        import numpy as np

        state: SpaceMouseOutput | None = inputs["deviceio_spacemouse"][0]

        translation_out = outputs["spacemouse_translation"]
        rotation_out = outputs["spacemouse_rotation"]
        buttons_out = outputs["spacemouse_buttons"]
        if state is None:
            translation_out.set_none()
            rotation_out.set_none()
            buttons_out.set_none()
            return

        translation = np.zeros(SPACEMOUSE_AXES_SIZE, dtype=np.float32)
        reported_translation = np.asarray(state.translation, dtype=np.float32)
        count = min(reported_translation.shape[0], SPACEMOUSE_AXES_SIZE)
        translation[:count] = reported_translation[:count]
        translation_out[0] = translation

        rotation = np.zeros(SPACEMOUSE_AXES_SIZE, dtype=np.float32)
        reported_rotation = np.asarray(state.rotation, dtype=np.float32)
        count = min(reported_rotation.shape[0], SPACEMOUSE_AXES_SIZE)
        rotation[:count] = reported_rotation[:count]
        rotation_out[0] = rotation

        bitmap = np.zeros(SPACEMOUSE_BUTTONS_BITMAP_SIZE, dtype=np.uint8)
        for code in state.pressed_buttons:
            if code < SPACEMOUSE_BUTTONS_BITMAP_SIZE:
                bitmap[code] = 1
        buttons_out[0] = bitmap
