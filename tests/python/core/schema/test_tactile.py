# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TactileMetadata and TactileData in isaaccapture.schema."""

import numpy as np
import pytest

from isaaccapture.schema import (
    BodySide,
    DeviceDataTimestamp,
    SkeletonBinding,
    SkeletonProfileRef,
    TactileChannels,
    TactileData,
    TactileDataRecord,
    TactileDevice,
    TactileExtension,
    TactileMetadata,
    TactileShape,
    TaxelState,
    validate_tactile_data,
    validate_tactile_metadata,
)

IDENTITY = [0.0, 0.0, 0.0, 1.0]


def make_shape(taxel_segments=((1, 0, 0, 0), (0, 1, 0, 0)), **overrides):
    """Wrist and index tip; one taxel on the tip, one skinned across both."""
    arguments = dict(
        segment_bind_positions=[[0, 0, 0], [0, 0.15, 0]],
        segment_bind_orientations=[IDENTITY, IDENTITY],
        taxel_positions=[[0, 0.16, 0.01], [0, 0.08, 0.01]],
        taxel_orientations=[IDENTITY, IDENTITY],
        taxel_segments=taxel_segments,
        taxel_weights=[[1, 0, 0, 0], [0.5, 0.5, 0, 0]],
        taxel_areas=[1e-5, 2e-5],
    )
    arguments.update(overrides)
    return TactileShape(**arguments)


def make_metadata(**overrides):
    arguments = dict(
        shape=make_shape(),
        channels=TactileChannels(normal_force=True, shear_force=True),
        device=TactileDevice(
            vendor="Example", side=BodySide.LEFT, nominal_rate_hz=100.0
        ),
        binding=SkeletonBinding(
            ["WRIST", "INDEX_TIP"], profile=SkeletonProfileRef("openxr.hand", 1)
        ),
        extensions=[TactileExtension("com.example.glove", 1, b"\x01\x02")],
    )
    arguments.update(overrides)
    return TactileMetadata(**arguments)


def test_shape_reads_back_as_views():
    shape = make_shape()

    assert shape.segment_count == 2
    assert shape.taxel_count == 2
    np.testing.assert_allclose(shape.segment_bind_positions[1], [0, 0.15, 0])
    np.testing.assert_allclose(shape.taxel_orientations[0], IDENTITY)
    assert shape.taxel_segments.dtype == np.uint16
    assert shape.taxel_segments.tolist() == [[1, 0, 0, 0], [0, 1, 0, 0]]
    np.testing.assert_allclose(shape.taxel_weights[1], [0.5, 0.5, 0, 0])
    np.testing.assert_allclose(shape.taxel_areas, [1e-5, 2e-5])


def test_shape_rejects_mismatched_or_out_of_range_input():
    with pytest.raises(ValueError, match="one row per taxel"):
        make_shape(taxel_weights=[[1, 0, 0, 0]])
    with pytest.raises(ValueError, match="taxel_segments"):
        make_shape(taxel_segments=[[70000, 0, 0, 0], [0, 1, 0, 0]])
    with pytest.raises(ValueError, match="shape"):
        make_shape(taxel_positions=[0, 0.16, 0.01])


def test_metadata_round_trips():
    metadata = make_metadata()

    assert metadata.device.vendor == "Example"
    assert metadata.device.side == BodySide.LEFT
    assert metadata.device.nominal_rate_hz == pytest.approx(100.0)
    assert TactileDevice().nominal_rate_hz is None
    assert metadata.shape.taxel_count == 2
    assert metadata.binding.segment_names == ["WRIST", "INDEX_TIP"]
    assert metadata.channels.shear_force
    assert metadata.extensions[0].data == b"\x01\x02"
    assert validate_tactile_metadata(metadata) == ""


def test_metadata_constructor_validates():
    with pytest.raises(ValueError, match="sum to"):
        make_metadata(shape=make_shape(taxel_weights=[[1, 0, 0, 0], [0.5, 0.4, 0, 0]]))
    with pytest.raises(ValueError, match="past the 2 segments"):
        make_metadata(shape=make_shape(taxel_segments=[[2, 0, 0, 0], [0, 1, 0, 0]]))
    with pytest.raises(ValueError, match=r"binding: .*side"):
        make_metadata(device=TactileDevice(side=BodySide.UNSPECIFIED))
    with pytest.raises(ValueError, match="no force"):
        make_metadata(channels=TactileChannels(taxel_state=True))


def test_data_reads_back_as_views():
    data = TactileData(
        normal_force=[1.5, 0.25],
        shear_force=[[0.1, 0.0], [0.0, -0.1]],
        taxel_state=[TaxelState.VALID, int(TaxelState.SATURATED)],
        sequence=7,
    )

    np.testing.assert_allclose(data.normal_force, [1.5, 0.25])
    assert data.shear_force.shape == (2, 2)
    np.testing.assert_allclose(data.shear_force[1], [0.0, -0.1])
    assert data.taxel_state.tolist() == [0, 1]
    assert data.raw is None
    assert data.sequence == 7
    assert TactileData().sequence is None


def test_data_matches_declared_channels():
    metadata = make_metadata()

    full = TactileData(normal_force=[1.0, 2.0], shear_force=np.zeros((2, 2)))
    assert validate_tactile_data(full, metadata) == ""
    assert "shear_force" in validate_tactile_data(
        TactileData(normal_force=[1.0, 2.0]), metadata
    )
    assert "normal_force has 1" in validate_tactile_data(
        TactileData(normal_force=[1.0], shear_force=np.zeros((2, 2))), metadata
    )
    with pytest.raises(ValueError, match="taxel_state"):
        TactileData(taxel_state=[4])


def test_record_wraps_data():
    record = TactileDataRecord(
        TactileData(normal_force=[1.0, 2.0]), DeviceDataTimestamp(1, 2, 3)
    )

    np.testing.assert_allclose(record.data.normal_force, [1.0, 2.0])
    assert record.timestamp.sample_time_local_common_clock == 2
