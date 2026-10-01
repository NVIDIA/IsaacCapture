# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SpaceMouseOutput type in isaaccapture.schema.

Tests the following FlatBuffers types:
- SpaceMouseOutput: Table with translation, rotation, pressed_buttons, is_valid
- SpaceMouseOutputRecord: Record wrapper carrying DeviceDataTimestamp

Timestamps are carried by SpaceMouseOutputRecord, not SpaceMouseOutput.
"""

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    SpaceMouseOutput,
    SpaceMouseOutputRecord,
)


class TestSpaceMouseOutputConstruction:
    """Tests for SpaceMouseOutput table construction."""

    def test_construction(self):
        """Test construction with explicit fields."""
        output = SpaceMouseOutput(
            translation=[0.1, 0.2, 0.3],
            rotation=[-0.1, -0.2, -0.3],
            pressed_buttons=[0],
            is_valid=True,
        )

        assert list(output.translation) == pytest.approx([0.1, 0.2, 0.3])
        assert list(output.rotation) == pytest.approx([-0.1, -0.2, -0.3])
        assert list(output.pressed_buttons) == [0]
        assert output.is_valid is True

    def test_repr(self):
        """Test __repr__ returns meaningful string."""
        output = SpaceMouseOutput(
            translation=[], rotation=[], pressed_buttons=[], is_valid=False
        )
        repr_str = repr(output)

        assert "SpaceMouseOutput" in repr_str


class TestSpaceMouseOutputFields:
    """Tests that translation, rotation, and pressed_buttons round-trip through the encoding."""

    def test_empty_fields(self):
        """Test encoding with no axes and no buttons held."""
        output = SpaceMouseOutput(
            translation=[], rotation=[], pressed_buttons=[], is_valid=True
        )

        assert list(output.translation) == []
        assert list(output.rotation) == []
        assert list(output.pressed_buttons) == []

    def test_encodings_are_independent(self):
        """Test each encoding carries its own values, not a shared buffer's."""
        first = SpaceMouseOutput(
            translation=[1.0], rotation=[], pressed_buttons=[0], is_valid=True
        )
        second = SpaceMouseOutput(
            translation=[-1.0], rotation=[], pressed_buttons=[1], is_valid=True
        )

        assert list(first.translation) == pytest.approx([1.0])
        assert list(second.translation) == pytest.approx([-1.0])
        assert list(first.pressed_buttons) == [0]
        assert list(second.pressed_buttons) == [1]


class TestSpaceMouseOutputEncoding:
    """Tests that an encoded payload reads back.

    A tracker with no spacemouse data returns None rather than an empty payload, so
    absence needs no case here; the source-node tests cover feeding None through.
    """

    def test_encoded_payload_reads_back(self):
        """An encoded payload gates as True and its fields read directly."""
        output = SpaceMouseOutput(
            translation=[0.5], rotation=[0.25], pressed_buttons=[0], is_valid=True
        )

        assert output
        assert list(output.translation) == pytest.approx([0.5])
        assert list(output.rotation) == pytest.approx([0.25])

    def test_repr_present(self):
        """Repr of a present payload names the type."""
        assert "SpaceMouseOutput" in repr(
            SpaceMouseOutput(
                translation=[], rotation=[], pressed_buttons=[], is_valid=True
            )
        )


class TestSpaceMouseOutputRecordTimestamp:
    """Tests for SpaceMouseOutputRecord with DeviceDataTimestamp."""

    def test_construction_with_timestamp(self):
        """Test SpaceMouseOutputRecord carries DeviceDataTimestamp."""
        data = SpaceMouseOutput(
            translation=[0.5], rotation=[0.25], pressed_buttons=[0], is_valid=True
        )
        ts = DeviceDataTimestamp(1000000000, 2000000000, 3000000000)
        record = SpaceMouseOutputRecord(data, ts)

        assert record.timestamp.available_time_local_common_clock == 1000000000
        assert record.timestamp.sample_time_local_common_clock == 2000000000
        assert record.timestamp.sample_time_raw_device_clock == 3000000000
        assert list(record.data.translation) == pytest.approx([0.5])

    def test_payload_less_record(self):
        """A record may carry a timestamp and no payload: MCAP's frame sentinel."""
        record = SpaceMouseOutputRecord(None, DeviceDataTimestamp(1, 2, 3))
        assert record.data is None
        assert record.timestamp.available_time_local_common_clock == 1
