# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SpaceMouse Source Node - DeviceIO to Retargeting Engine converter.

Converts the SpaceMouseOutput pushed by the spacemouse plugin to three outputs: translation axes,
rotation axes, and a button bitmap. Carries no semantic mapping: which axis or button means what
is up to the consuming retargeter.

Buttons are sampled: each frame reports the buttons held at the plugin's latest sample, so a press
and release between two samples is not seen.
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
    from isaaccapture.deviceio_trackers import ITracker
    from isaaccapture.schema import SpaceMouseOutput

# Default collection_id of the spacemouse plugin and SpaceMouseTracker.
DEFAULT_SPACEMOUSE_COLLECTION_ID = "spacemouse"

# [x, y, z]
SPACEMOUSE_AXES_SIZE = 3

# Buttons reported in the bitmap; a device reporting more has the rest dropped.
SPACEMOUSE_BUTTONS_BITMAP_SIZE = 8


def _axes_type(name: str) -> TensorGroupType:
    return TensorGroupType(
        name,
        [
            NDArrayType(
                "axes",
                shape=(SPACEMOUSE_AXES_SIZE,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            )
        ],
    )


def SpaceMouseTranslationType() -> TensorGroupType:
    """Type for "spacemouse_translation": a float32 [x, y, z] array in [-1, 1]."""
    return _axes_type("spacemouse_translation")


def SpaceMouseRotationType() -> TensorGroupType:
    """Type for "spacemouse_rotation": a float32 [x, y, z] array in [-1, 1]."""
    return _axes_type("spacemouse_rotation")


def SpaceMouseButtonsType() -> TensorGroupType:
    """Type for "spacemouse_buttons": a uint8 bitmap indexed by button, 1 while held."""
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
        - "deviceio_spacemouse": SpaceMouseOutput from SpaceMouseTracker

    Outputs (Optional -- absent while the plugin is not running or has no SpaceMouse open):
        - "spacemouse_translation": float32 [x, y, z] in [-1, 1]
        - "spacemouse_rotation": float32 [x, y, z] in [-1, 1]
        - "spacemouse_buttons": uint8 bitmap indexed by button, 1 while held

    The data comes from the spacemouse plugin. Start it with the session, using the copy bundled
    with isaaccapture::

        PluginConfig(
            plugin_name="spacemouse",
            plugin_root_id="spacemouse",
            search_paths=[BUNDLED_PLUGINS_DIR],
        )

    and pass ``--collection-id=<collection_id>`` in ``plugin_args`` for a non-default collection.
    """

    def __init__(
        self, name: str, collection_id: str = DEFAULT_SPACEMOUSE_COLLECTION_ID
    ) -> None:
        """Initialize the source node and its SpaceMouseTracker.

        Args:
            name: Unique name for this source node.
            collection_id: Tensor collection the spacemouse plugin pushes to.
        """
        from isaaccapture.deviceio_trackers import SpaceMouseTracker

        self._spacemouse_tracker = SpaceMouseTracker(collection_id)
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Get the SpaceMouseTracker instance for TeleopSession to register."""
        return self._spacemouse_tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the SpaceMouse tracker.

        Returns:
            Dict with "deviceio_spacemouse" TensorGroup containing SpaceMouseOutput | None.
        """
        state = self._spacemouse_tracker.get_spacemouse_data(deviceio_session)
        tg = TensorGroup(DeviceIOSpaceMouseOutputTracked())
        tg[0] = state
        return {"deviceio_spacemouse": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare DeviceIO SpaceMouse input."""
        return {"deviceio_spacemouse": DeviceIOSpaceMouseOutputTracked()}

    def output_spec(self) -> RetargeterIOType:
        """Declare the translation, rotation and button outputs (Optional)."""
        return {
            "spacemouse_translation": OptionalType(SpaceMouseTranslationType()),
            "spacemouse_rotation": OptionalType(SpaceMouseRotationType()),
            "spacemouse_buttons": OptionalType(SpaceMouseButtonsType()),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Convert SpaceMouseOutput to the three outputs, or set them to None without a device."""
        import numpy as np

        state: SpaceMouseOutput | None = inputs["deviceio_spacemouse"][0]

        translation_out = outputs["spacemouse_translation"]
        rotation_out = outputs["spacemouse_rotation"]
        buttons_out = outputs["spacemouse_buttons"]
        if state is None or not state.connected:
            translation_out.set_none()
            rotation_out.set_none()
            buttons_out.set_none()
            return

        t, r = state.translation, state.rotation
        translation_out[0] = np.array([t.x, t.y, t.z], dtype=np.float32)
        rotation_out[0] = np.array([r.x, r.y, r.z], dtype=np.float32)

        bitmap = np.zeros(SPACEMOUSE_BUTTONS_BITMAP_SIZE, dtype=np.uint8)
        held = np.asarray(
            state.buttons[:SPACEMOUSE_BUTTONS_BITMAP_SIZE], dtype=np.uint8
        )
        bitmap[: held.shape[0]] = held
        buttons_out[0] = bitmap
