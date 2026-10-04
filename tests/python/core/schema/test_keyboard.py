# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KeyboardOutput type in isaaccapture.schema.

Tests the following FlatBuffers types:
- KeyboardOutput: Table with pressed_keys (evdev key codes) and is_valid
- KeyboardOutputRecord: Record wrapper carrying DeviceDataTimestamp

Timestamps are carried by KeyboardOutputRecord, not KeyboardOutput.
"""

from isaaccapture.schema import (
    DeviceDataTimestamp,
    KeyboardOutput,
    KeyboardOutputRecord,
)

# Evdev key codes (linux/input-event-codes.h), matching keyboard_plugin.cpp.
KEY_W, KEY_A = 17, 30


class TestKeyboardOutputConstruction:
    """Tests for KeyboardOutput table construction."""

    def test_construction(self):
        """Test construction with explicit fields."""
        output = KeyboardOutput(pressed_keys=[KEY_W], is_valid=True)

        assert list(output.pressed_keys) == [KEY_W]
        assert output.is_valid is True

    def test_repr(self):
        """Test __repr__ returns meaningful string."""
        output = KeyboardOutput(pressed_keys=[], is_valid=False)
        repr_str = repr(output)

        assert "KeyboardOutput" in repr_str


class TestKeyboardOutputPressedKeys:
    """Tests that pressed_keys round-trips through the encoding."""

    def test_empty_pressed_keys(self):
        """Test encoding with no keys held."""
        output = KeyboardOutput(pressed_keys=[], is_valid=True)

        assert list(output.pressed_keys) == []

    def test_multiple_pressed_keys(self):
        """Test encoding multiple simultaneously-held keys."""
        output = KeyboardOutput(pressed_keys=[KEY_W, KEY_A], is_valid=True)

        assert list(output.pressed_keys) == [KEY_W, KEY_A]

    def test_encodings_are_independent(self):
        """Test each encoding carries its own values, not a shared buffer's."""
        first = KeyboardOutput(pressed_keys=[KEY_W], is_valid=True)
        second = KeyboardOutput(pressed_keys=[KEY_A], is_valid=True)

        assert list(first.pressed_keys) == [KEY_W]
        assert list(second.pressed_keys) == [KEY_A]


class TestKeyboardOutputEncoding:
    """Tests that an encoded payload reads back.

    A tracker with no keyboard data returns None rather than an empty payload, so
    absence needs no case here; the source-node tests cover feeding None through.
    """

    def test_encoded_payload_reads_back(self):
        """An encoded payload gates as True and its fields read directly."""
        output = KeyboardOutput(pressed_keys=[KEY_W], is_valid=True)

        assert output
        assert list(output.pressed_keys) == [KEY_W]
        assert output.is_valid is True

    def test_repr_present(self):
        """Repr of a present payload names the type."""
        assert "KeyboardOutput" in repr(KeyboardOutput(pressed_keys=[], is_valid=True))


class TestKeyboardOutputRecordTimestamp:
    """Tests for KeyboardOutputRecord with DeviceDataTimestamp."""

    def test_construction_with_timestamp(self):
        """Test KeyboardOutputRecord carries DeviceDataTimestamp."""
        data = KeyboardOutput(pressed_keys=[KEY_W], is_valid=True)
        ts = DeviceDataTimestamp(1000000000, 2000000000, 3000000000)
        record = KeyboardOutputRecord(data, ts)

        assert record.timestamp.available_time_local_common_clock == 1000000000
        assert record.timestamp.sample_time_local_common_clock == 2000000000
        assert record.timestamp.sample_time_raw_device_clock == 3000000000
        assert list(record.data.pressed_keys) == [KEY_W]

    def test_payload_less_record(self):
        """A record may carry a timestamp and no payload: MCAP's frame sentinel."""
        record = KeyboardOutputRecord(None, DeviceDataTimestamp(1, 2, 3))
        assert record.data is None
        assert record.timestamp.available_time_local_common_clock == 1
