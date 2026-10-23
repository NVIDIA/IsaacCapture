# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TactileDataArray / TactileDataArrayRecord in isaaccapture.schema."""

import gc

import numpy as np
import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    TactileDataArray,
    TactileDataArrayRecord,
)

FORCES = np.array([[0.0, 0.0, 1.0], [0.5, -0.5, 2.0], [0.0, 0.1, 0.0]], np.float32)


class TestTactileDataArray:
    @pytest.mark.parametrize("args", [(), (None,)])
    def test_requires_force_array(self, args):
        with pytest.raises(TypeError, match="taxel_forces"):
            TactileDataArray(*args)

    def test_explicit_empty_force_array(self):
        data = TactileDataArray(np.empty((0, 3), np.float32))
        assert data.taxel_forces.shape == (0, 3)
        record = TactileDataArrayRecord(data, DeviceDataTimestamp(1, 2, 3))
        assert record.data.taxel_forces.shape == (0, 3)

    def test_round_trip(self):
        data = TactileDataArray(FORCES)
        assert data.taxel_forces.dtype == np.float32
        np.testing.assert_array_equal(data.taxel_forces, FORCES)

    def test_accepts_nested_lists(self):
        data = TactileDataArray([[1, 2, 3]])
        np.testing.assert_array_equal(data.taxel_forces, [[1.0, 2.0, 3.0]])

    def test_view_outlives_its_table(self):
        forces = TactileDataArray(FORCES).taxel_forces
        gc.collect()
        np.testing.assert_array_equal(forces, FORCES)

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="taxel_forces"):
            TactileDataArray(np.zeros((2, 4), np.float32))

    def test_repr(self):
        assert "taxels=3" in repr(TactileDataArray(FORCES))


class TestTactileDataArrayRecord:
    def test_record_carries_payload_and_timestamp(self):
        record = TactileDataArrayRecord(
            TactileDataArray(FORCES), DeviceDataTimestamp(1, 2, 3)
        )
        np.testing.assert_array_equal(record.data.taxel_forces, FORCES)
        assert record.timestamp.sample_time_raw_device_clock == 3

    def test_rejects_missing_payload(self):
        with pytest.raises(TypeError, match="data"):
            TactileDataArrayRecord(None, DeviceDataTimestamp(1, 2, 3))
