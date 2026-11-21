# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gamepad Source Node - DeviceIO to Retargeting Engine converter.

Converts the GamepadSnapshot pushed by the gamepad plugin to the standard GamepadInput tensor group:
the buttons and axes every standard gamepad has, by name. The plugin resolves each pad's own
numbering, so west_click is X on an Xbox pad and Square on a PlayStation pad.

Buttons are sampled: each frame reports the buttons held at the plugin's latest sample, so a press
and release between two samples is not seen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType
from ..tensor_types import GamepadInput, GamepadInputIndex
from .deviceio_tensor_types import DeviceIOGamepadSnapshotTracked
from .interface import IDeviceIOSource

if TYPE_CHECKING:
    from isaaccapture.deviceio_trackers import ITracker
    from isaaccapture.schema import GamepadSnapshot

# Default collection_id of the gamepad plugin and GamepadTracker.
DEFAULT_GAMEPAD_COLLECTION_ID = "gamepad"


class GamepadSource(IDeviceIOSource):
    """
    Stateless converter: DeviceIO GamepadSnapshot → GamepadInput tensor group.

    Inputs:
        - "deviceio_gamepad": GamepadSnapshot from GamepadTracker

    Outputs (Optional -- absent while the plugin is not running or has no gamepad open):
        - "gamepad": GamepadInput, indexed by GamepadInputIndex

    The data comes from the gamepad plugin. Start it with the session, using the copy bundled with
    isaaccapture::

        PluginConfig(
            plugin_name="gamepad",
            plugin_root_id="gamepad",
            search_paths=[BUNDLED_PLUGINS_DIR],
        )

    and pass ``--collection-id=<collection_id>`` in ``plugin_args`` for a non-default collection,
    with ``--player=N`` to read the Nth connected pad.
    """

    def __init__(
        self, name: str, collection_id: str = DEFAULT_GAMEPAD_COLLECTION_ID
    ) -> None:
        """Initialize the source node and its GamepadTracker.

        Args:
            name: Unique name for this source node.
            collection_id: Tensor collection the gamepad plugin pushes to.
        """
        from isaaccapture.deviceio_trackers import GamepadTracker

        self._gamepad_tracker = GamepadTracker(collection_id)
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Get the GamepadTracker instance for TeleopSession to register."""
        return self._gamepad_tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the gamepad tracker.

        Returns:
            Dict with "deviceio_gamepad" TensorGroup containing GamepadSnapshot | None.
        """
        state = self._gamepad_tracker.get_gamepad_data(deviceio_session)
        tg = TensorGroup(DeviceIOGamepadSnapshotTracked())
        tg[0] = state
        return {"deviceio_gamepad": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare DeviceIO gamepad input."""
        return {"deviceio_gamepad": DeviceIOGamepadSnapshotTracked()}

    def output_spec(self) -> RetargeterIOType:
        """Declare the gamepad output (Optional)."""
        return {"gamepad": OptionalType(GamepadInput())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Convert GamepadSnapshot to GamepadInput, or set it to None without a pad."""
        snapshot: GamepadSnapshot | None = inputs["deviceio_gamepad"][0]

        group = outputs["gamepad"]
        if snapshot is None or snapshot.inputs is None:
            group.set_none()
            return

        pad = snapshot.inputs
        group[GamepadInputIndex.SOUTH_CLICK] = float(pad.south_click)
        group[GamepadInputIndex.EAST_CLICK] = float(pad.east_click)
        group[GamepadInputIndex.WEST_CLICK] = float(pad.west_click)
        group[GamepadInputIndex.NORTH_CLICK] = float(pad.north_click)
        group[GamepadInputIndex.VIEW_CLICK] = float(pad.view_click)
        group[GamepadInputIndex.MENU_CLICK] = float(pad.menu_click)
        group[GamepadInputIndex.SHOULDER_LEFT_CLICK] = float(pad.shoulder_left_click)
        group[GamepadInputIndex.SHOULDER_RIGHT_CLICK] = float(pad.shoulder_right_click)
        group[GamepadInputIndex.THUMBSTICK_LEFT_CLICK] = float(
            pad.thumbstick_left_click
        )
        group[GamepadInputIndex.THUMBSTICK_RIGHT_CLICK] = float(
            pad.thumbstick_right_click
        )
        group[GamepadInputIndex.DPAD_UP_CLICK] = float(pad.dpad_up_click)
        group[GamepadInputIndex.DPAD_DOWN_CLICK] = float(pad.dpad_down_click)
        group[GamepadInputIndex.DPAD_LEFT_CLICK] = float(pad.dpad_left_click)
        group[GamepadInputIndex.DPAD_RIGHT_CLICK] = float(pad.dpad_right_click)
        group[GamepadInputIndex.THUMBSTICK_LEFT_X] = float(pad.thumbstick_left_x)
        group[GamepadInputIndex.THUMBSTICK_LEFT_Y] = float(pad.thumbstick_left_y)
        group[GamepadInputIndex.THUMBSTICK_RIGHT_X] = float(pad.thumbstick_right_x)
        group[GamepadInputIndex.THUMBSTICK_RIGHT_Y] = float(pad.thumbstick_right_y)
        group[GamepadInputIndex.TRIGGER_LEFT_VALUE] = float(pad.trigger_left_value)
        group[GamepadInputIndex.TRIGGER_RIGHT_VALUE] = float(pad.trigger_right_value)
