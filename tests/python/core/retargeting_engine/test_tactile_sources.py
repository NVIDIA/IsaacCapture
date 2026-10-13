# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TactileSource DeviceIO converter.

Payloads are built through the real schema bindings; no OpenXR device is involved.
"""

import numpy as np
import pytest

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    TactileShapeIndex,
    TactileShapeLayout,
    TactileSource,
)
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import TactileData, TactileShape

NUM_JOINTS = 2
NUM_TAXELS = 3


def _source(**kwargs) -> TactileSource:
    args = dict(
        collection_id="tactile/right", num_joints=NUM_JOINTS, num_taxels=NUM_TAXELS
    )
    args.update(kwargs)
    return TactileSource("tactile", **args)


def _inputs(source, data, shape) -> dict:
    tg = TensorGroup(source.input_spec()["deviceio_tactile"])
    tg[0] = data
    tg[1] = shape
    return {"deviceio_tactile": tg}


def _outputs(source) -> dict:
    return {name: _make_output_group(gt) for name, gt in source.output_spec().items()}


def _shape(num_joints: int = NUM_JOINTS, num_taxels: int = NUM_TAXELS) -> TactileShape:
    indices = np.zeros((num_taxels, 4), np.uint32)
    weights = np.zeros((num_taxels, 4), np.float32)
    weights[:, 0] = 1.0
    return TactileShape(
        joint_bind_positions=np.zeros((num_joints, 3), np.float32),
        joint_bind_orientations=np.tile([0.0, 0.0, 0.0, 1.0], (num_joints, 1)),
        joint_parents=[TactileShape.ROOT_PARENT] + list(range(num_joints - 1)),
        taxel_positions=np.full((num_taxels, 3), 0.01, np.float32),
        taxel_normals=np.tile([0.0, 0.0, 1.0], (num_taxels, 1)),
        taxel_tangents=np.tile([1.0, 0.0, 0.0], (num_taxels, 1)),
        taxel_joint_indices=indices,
        taxel_joint_weights=weights,
    )


class _FakeTracker:
    """Stands in for a TactileTracker: fixed data and metadata."""

    def __init__(self, data, metadata):
        self._data = data
        self._metadata = metadata

    def get_data(self, _session):
        return self._data

    def get_metadata(self, _session):
        return self._metadata


class TestTactileSource:
    def test_creation_and_tracker(self):
        assert _source().get_tracker().get_name() == "TactileTracker"

    def test_specs(self):
        src = _source()
        assert list(src.input_spec()) == ["deviceio_tactile"]
        forces = src.output_spec()[TactileSource.FORCES]
        assert forces.is_optional
        assert forces.inner_type.types[0].shape == (NUM_TAXELS, 3)
        assert src.output_spec()[TactileSource.SHAPE].is_optional

    def test_index_matches_layout_order(self):
        names = [t.name for t in TactileShapeLayout(NUM_JOINTS, NUM_TAXELS).types]
        assert names == [member.name.lower() for member in TactileShapeIndex]

    def test_poll_reads_data_and_metadata_from_one_tracker(self):
        src = _source()
        data = TactileData(np.zeros((NUM_TAXELS, 3), np.float32))
        shape = _shape()
        src._tracker = _FakeTracker(data, shape)

        polled = src.poll_tracker(None)["deviceio_tactile"]

        assert polled[0] is data
        assert polled[1] is shape

    def test_active_conversion(self):
        src = _source()
        forces = np.arange(9, dtype=np.float32).reshape(3, 3)
        outputs = _outputs(src)
        src.compute(_inputs(src, TactileData(forces), _shape()), outputs)

        np.testing.assert_array_equal(
            np.asarray(outputs[TactileSource.FORCES][0]), forces
        )
        group = outputs[TactileSource.SHAPE]
        assert not group.is_none
        parents = np.asarray(group[TactileShapeIndex.JOINT_PARENTS])
        assert parents.dtype == np.int32
        assert parents.tolist() == [TactileShape.ROOT_PARENT, 0]
        assert (
            np.asarray(group[TactileShapeIndex.TAXEL_JOINT_INDICES]).dtype == np.int32
        )
        np.testing.assert_array_equal(
            np.asarray(group[TactileShapeIndex.TAXEL_NORMALS]),
            np.tile([0.0, 0.0, 1.0], (NUM_TAXELS, 1)),
        )

    def test_shape_without_data(self):
        src = _source()
        outputs = _outputs(src)
        src.compute(_inputs(src, None, _shape()), outputs)
        assert outputs[TactileSource.FORCES].is_none
        assert not outputs[TactileSource.SHAPE].is_none

    def test_inactive_sets_none(self):
        src = _source()
        outputs = _outputs(src)
        src.compute(_inputs(src, None, None), outputs)
        assert outputs[TactileSource.FORCES].is_none
        assert outputs[TactileSource.SHAPE].is_none

    def test_taxel_count_mismatch_raises(self):
        src = _source(num_taxels=4)
        frame = TactileData(np.zeros((3, 3), np.float32))
        with pytest.raises(ValueError, match="3 taxels"):
            src.compute(_inputs(src, frame, None), _outputs(src))

    def test_shape_count_mismatch_raises(self):
        src = _source(num_joints=NUM_JOINTS + 1)
        with pytest.raises(ValueError, match="2 joints"):
            src.compute(_inputs(src, None, _shape()), _outputs(src))
