# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TactileData / TactileDataRecord in isaaccapture.schema (internal only)."""

import gc

import numpy as np
import pytest

from isaaccapture.schema import DeviceDataTimestamp, TactileData, TactileDataRecord

FORCES = np.array([[0.0, 0.0, 1.0], [0.5, -0.5, 2.0], [0.0, 0.1, 0.0]], np.float32)


class TestTactileData:
    def test_default_construction(self):
        data = TactileData()
        assert data.taxel_count == 0
        assert data.taxel_force.shape == (0, 3)
        assert data.device_id == ""
        assert data.shape_version == 0

    def test_round_trip(self):
        data = TactileData(FORCES, device_id="glove_right", shape_version=3)
        assert data.taxel_count == 3
        assert data.taxel_force.dtype == np.float32
        np.testing.assert_array_equal(data.taxel_force, FORCES)
        assert data.device_id == "glove_right"
        assert data.shape_version == 3

    def test_accepts_nested_lists(self):
        data = TactileData([[1, 2, 3]])
        np.testing.assert_array_equal(data.taxel_force, [[1.0, 2.0, 3.0]])

    def test_view_outlives_its_table(self):
        forces = TactileData(FORCES).taxel_force
        gc.collect()
        np.testing.assert_array_equal(forces, FORCES)

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="taxel_force"):
            TactileData(np.zeros((2, 4), np.float32))

    def test_repr(self):
        assert "taxel_count=3" in repr(TactileData(FORCES))


class TestTactileDataRecord:
    def test_record_carries_payload_and_timestamp(self):
        record = TactileDataRecord(TactileData(FORCES), DeviceDataTimestamp(1, 2, 3))
        np.testing.assert_array_equal(record.data.taxel_force, FORCES)
        assert record.timestamp.sample_time_raw_device_clock == 3

    def test_payload_less_record(self):
        record = TactileDataRecord(None, DeviceDataTimestamp(1, 2, 3))
        assert record.data is None
