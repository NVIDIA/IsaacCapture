# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GamepadOutput type in isaaccapture.schema.

Tests the following FlatBuffers types:
- GamepadOutput: Table with pressed_buttons (joystick button indices) and axes
- GamepadOutputRecord: Record wrapper carrying DeviceDataTimestamp

Timestamps are carried by GamepadOutputRecord, not GamepadOutput.
"""

import pytest

from isaaccapture.schema import DeviceDataTimestamp, GamepadOutput, GamepadOutputRecord


class TestGamepadOutputConstruction:
    """Tests for GamepadOutput table construction."""

    def test_construction(self):
        """Test construction with explicit fields."""
        output = GamepadOutput(pressed_buttons=[2], axes=[0.5, -0.5], is_valid=True)

        assert list(output.pressed_buttons) == [2]
        assert list(output.axes) == pytest.approx([0.5, -0.5])
        assert output.is_valid is True

    def test_repr(self):
        """Test __repr__ returns meaningful string."""
        output = GamepadOutput(pressed_buttons=[], axes=[], is_valid=False)
        repr_str = repr(output)

        assert "GamepadOutput" in repr_str


class TestGamepadOutputFields:
    """Tests that pressed_buttons and axes round-trip through the encoding."""

    def test_empty_fields(self):
        """Test encoding with no buttons held and no axes reported."""
        output = GamepadOutput(pressed_buttons=[], axes=[], is_valid=True)

        assert list(output.pressed_buttons) == []
        assert list(output.axes) == []

    def test_multiple_pressed_buttons(self):
        """Test encoding multiple simultaneously-held buttons."""
        output = GamepadOutput(pressed_buttons=[0, 5], axes=[], is_valid=True)

        assert list(output.pressed_buttons) == [0, 5]

    def test_encodings_are_independent(self):
        """Test each encoding carries its own values, not a shared buffer's."""
        first = GamepadOutput(pressed_buttons=[0], axes=[1.0], is_valid=True)
        second = GamepadOutput(pressed_buttons=[1], axes=[-1.0], is_valid=True)

        assert list(first.pressed_buttons) == [0]
        assert list(second.pressed_buttons) == [1]
        assert list(first.axes) == pytest.approx([1.0])
        assert list(second.axes) == pytest.approx([-1.0])


class TestGamepadOutputEncoding:
    """Tests that an encoded payload reads back.

    A tracker with no gamepad data returns None rather than an empty payload, so
    absence needs no case here; the source-node tests cover feeding None through.
    """

    def test_encoded_payload_reads_back(self):
        """An encoded payload gates as True and its fields read directly."""
        output = GamepadOutput(pressed_buttons=[2], axes=[0.5], is_valid=True)

        assert output
        assert list(output.pressed_buttons) == [2]
        assert list(output.axes) == pytest.approx([0.5])

    def test_repr_present(self):
        """Repr of a present payload names the type."""
        assert "GamepadOutput" in repr(
            GamepadOutput(pressed_buttons=[], axes=[], is_valid=True)
        )


class TestGamepadOutputRecordTimestamp:
    """Tests for GamepadOutputRecord with DeviceDataTimestamp."""

    def test_construction_with_timestamp(self):
        """Test GamepadOutputRecord carries DeviceDataTimestamp."""
        data = GamepadOutput(pressed_buttons=[2], axes=[0.5], is_valid=True)
        ts = DeviceDataTimestamp(1000000000, 2000000000, 3000000000)
        record = GamepadOutputRecord(data, ts)

        assert record.timestamp.available_time_local_common_clock == 1000000000
        assert record.timestamp.sample_time_local_common_clock == 2000000000
        assert record.timestamp.sample_time_raw_device_clock == 3000000000
        assert list(record.data.pressed_buttons) == [2]

    def test_payload_less_record(self):
        """A record may carry a timestamp and no payload: MCAP's frame sentinel."""
        record = GamepadOutputRecord(None, DeviceDataTimestamp(1, 2, 3))
        assert record.data is None
        assert record.timestamp.available_time_local_common_clock == 1
