# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the keyboard types in isaaccapture.schema.

Tests the following FlatBuffers types:
- KeyAction / KeyEvent: one key transition (struct)
- KeyboardOutput: Table with pressed_keys (evdev key codes) and ordered events
- KeyboardOutputRecord: Record wrapper carrying DeviceDataTimestamp

Timestamps are carried by KeyboardOutputRecord, not KeyboardOutput.
"""

from isaaccapture.schema import (
    DeviceDataTimestamp,
    KeyAction,
    KeyboardOutput,
    KeyboardOutputRecord,
    KeyEvent,
)

# Evdev key codes (linux/input-event-codes.h).
KEY_W, KEY_A = 17, 30


class TestKeyEvent:
    """Tests for the KeyEvent struct."""

    def test_fields(self):
        event = KeyEvent(timestamp_ns=42, code=KEY_W, action=KeyAction.PRESS)

        assert event.timestamp_ns == 42
        assert event.code == KEY_W
        assert event.action == KeyAction.PRESS

    def test_repr(self):
        repr_str = repr(KeyEvent(1, KEY_A, KeyAction.RELEASE))

        assert "KeyEvent" in repr_str
        assert "RELEASE" in repr_str


class TestKeyboardOutputConstruction:
    """Tests for KeyboardOutput table construction."""

    def test_construction(self):
        output = KeyboardOutput(pressed_keys=[KEY_W])

        assert list(output.pressed_keys) == [KEY_W]
        assert output.events == []

    def test_repr(self):
        repr_str = repr(KeyboardOutput(pressed_keys=[]))

        assert "KeyboardOutput" in repr_str


class TestKeyboardOutputPressedKeys:
    """Tests that pressed_keys round-trips through the encoding."""

    def test_empty_pressed_keys(self):
        assert list(KeyboardOutput(pressed_keys=[]).pressed_keys) == []

    def test_multiple_pressed_keys(self):
        output = KeyboardOutput(pressed_keys=[KEY_W, KEY_A])

        assert list(output.pressed_keys) == [KEY_W, KEY_A]

    def test_encodings_are_independent(self):
        """Each encoding carries its own values, not a shared buffer's."""
        first = KeyboardOutput(pressed_keys=[KEY_W])
        second = KeyboardOutput(pressed_keys=[KEY_A])

        assert list(first.pressed_keys) == [KEY_W]
        assert list(second.pressed_keys) == [KEY_A]


class TestKeyboardOutputEvents:
    """Tests that the ordered event list round-trips through the encoding."""

    def test_events_keep_order(self):
        events = [
            KeyEvent(10, KEY_W, KeyAction.PRESS),
            KeyEvent(20, KEY_W, KeyAction.RELEASE),
            KeyEvent(30, KEY_A, KeyAction.PRESS),
        ]
        output = KeyboardOutput(pressed_keys=[KEY_A], events=events)

        read = output.events
        assert [(e.timestamp_ns, e.code, e.action) for e in read] == [
            (10, KEY_W, KeyAction.PRESS),
            (20, KEY_W, KeyAction.RELEASE),
            (30, KEY_A, KeyAction.PRESS),
        ]

    def test_events_outlive_payload(self):
        """Events are copied out by value, so they stay valid after the payload is gone."""
        events = KeyboardOutput(
            pressed_keys=[], events=[KeyEvent(5, KEY_W, KeyAction.PRESS)]
        ).events

        assert events[0].code == KEY_W


class TestKeyboardOutputRecordTimestamp:
    """Tests for KeyboardOutputRecord with DeviceDataTimestamp."""

    def test_construction_with_timestamp(self):
        data = KeyboardOutput(
            pressed_keys=[KEY_W], events=[KeyEvent(7, KEY_W, KeyAction.PRESS)]
        )
        ts = DeviceDataTimestamp(1000000000, 2000000000, 3000000000)
        record = KeyboardOutputRecord(data, ts)

        assert record.timestamp.available_time_local_common_clock == 1000000000
        assert record.timestamp.sample_time_local_common_clock == 2000000000
        assert record.timestamp.sample_time_raw_device_clock == 3000000000
        assert list(record.data.pressed_keys) == [KEY_W]
        assert record.data.events[0].timestamp_ns == 7

    def test_payload_less_record(self):
        """A record may carry a timestamp and no payload: MCAP's frame sentinel."""
        record = KeyboardOutputRecord(None, DeviceDataTimestamp(1, 2, 3))
        assert record.data is None
        assert record.timestamp.available_time_local_common_clock == 1
