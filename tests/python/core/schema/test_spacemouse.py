# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SpaceMouse types in isaaccapture.schema.

- SpaceMouseOutput: translation and rotation axes, per-button state, and whether the device is open
- SpaceMouseOutputRecord: record wrapper carrying DeviceDataTimestamp
"""

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    Point,
    SpaceMouseOutput,
    SpaceMouseOutputRecord,
)


class TestSpaceMouseOutput:
    def test_default_is_a_disconnected_device_at_rest(self):
        output = SpaceMouseOutput()

        assert not output.connected
        assert (output.translation.x, output.translation.y, output.translation.z) == (
            0,
            0,
            0,
        )
        assert (output.rotation.x, output.rotation.y, output.rotation.z) == (0, 0, 0)
        assert output.buttons == []

    def test_fields_round_trip(self):
        output = SpaceMouseOutput(
            translation=Point(0.5, -0.25, 1.0),
            rotation=Point(-1.0, 0.0, 0.75),
            buttons=[1, 0],
            connected=True,
        )

        assert output.connected
        assert output.translation.x == pytest.approx(0.5)
        assert output.translation.y == pytest.approx(-0.25)
        assert output.translation.z == pytest.approx(1.0)
        assert output.rotation.x == pytest.approx(-1.0)
        assert output.rotation.z == pytest.approx(0.75)
        assert output.buttons == [1, 0]

    def test_repr(self):
        repr_str = repr(SpaceMouseOutput(buttons=[1, 0], connected=True))

        assert "SpaceMouseOutput" in repr_str
        assert "connected=True" in repr_str


class TestSpaceMouseOutputRecord:
    def test_record_carries_data_and_timestamp(self):
        data = SpaceMouseOutput(translation=Point(0.1, 0.2, 0.3), connected=True)
        record = SpaceMouseOutputRecord(data, DeviceDataTimestamp(1, 2, 3))

        assert record.timestamp.available_time_local_common_clock == 1
        assert record.timestamp.sample_time_local_common_clock == 2
        assert record.timestamp.sample_time_raw_device_clock == 3
        assert record.data.translation.z == pytest.approx(0.3)

    def test_payload_less_record(self):
        record = SpaceMouseOutputRecord(None, DeviceDataTimestamp(1, 2, 3))

        assert record.data is None
