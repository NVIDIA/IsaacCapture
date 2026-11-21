# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the gamepad types in isaaccapture.schema.

- GamepadInputState: the buttons and axes every standard gamepad has
- GamepadSnapshot: the input state, absent while no pad is open
- GamepadSnapshotRecord: record wrapper carrying DeviceDataTimestamp
"""

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    GamepadInputState,
    GamepadSnapshot,
    GamepadSnapshotRecord,
)


class TestGamepadInputState:
    def test_default_is_released_and_centered(self):
        inputs = GamepadInputState()

        assert not inputs.south_click
        assert not inputs.dpad_right_click
        assert inputs.thumbstick_left_x == 0.0
        assert inputs.trigger_right_value == 0.0

    def test_fields_by_name(self):
        inputs = GamepadInputState(
            west_click=True, thumbstick_left_y=1.0, trigger_right_value=0.5
        )

        assert inputs.west_click
        assert not inputs.east_click
        assert inputs.thumbstick_left_y == pytest.approx(1.0)
        assert inputs.trigger_right_value == pytest.approx(0.5)

    def test_repr_names_the_held_buttons(self):
        repr_str = repr(GamepadInputState(south_click=True, dpad_up_click=True))

        assert "held=[south, dpad_up]" in repr_str


class TestGamepadSnapshot:
    def test_default_is_no_pad(self):
        snapshot = GamepadSnapshot()

        assert snapshot.inputs is None
        assert "inputs=None" in repr(snapshot)

    def test_fields_round_trip(self):
        snapshot = GamepadSnapshot(GamepadInputState(north_click=True))

        assert snapshot.inputs.north_click


class TestGamepadSnapshotRecord:
    def test_record_carries_data_and_timestamp(self):
        data = GamepadSnapshot(GamepadInputState(south_click=True))
        record = GamepadSnapshotRecord(data, DeviceDataTimestamp(1, 2, 3))

        assert record.timestamp.available_time_local_common_clock == 1
        assert record.timestamp.sample_time_local_common_clock == 2
        assert record.timestamp.sample_time_raw_device_clock == 3
        assert record.data.inputs.south_click

    def test_payload_less_record(self):
        record = GamepadSnapshotRecord(None, DeviceDataTimestamp(1, 2, 3))

        assert record.data is None
