# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TactileSource DeviceIO converter.

Payloads are built through the real schema bindings; no OpenXR device is involved.
Session-backed tracker reads are covered in tests/cpp/core/replay_deviceio_session.
"""

import numpy as np
import pytest

from isaaccapture.retargeting_engine.deviceio_source_nodes import TactileSource
from isaaccapture.retargeting_engine.tensor_types import (
    TactileLBSLayout,
    TactileLBSLayoutIndex,
)
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import (
    HandJoint,
    HandSkeletonRef,
    LBS,
    TactileDataArray,
    TactileMetadata,
)

NUM_SEGMENTS = 2
NUM_TAXELS = 3


def _source(**kwargs) -> TactileSource:
    args = dict(
        collection_id="tactile/right", num_segments=NUM_SEGMENTS, num_taxels=NUM_TAXELS
    )
    args.update(kwargs)
    return TactileSource("tactile", **args)


def _inputs(source, data, metadata) -> dict:
    tg = TensorGroup(source.input_spec()["deviceio_tactile"])
    tg[0] = data
    tg[1] = metadata
    return {"deviceio_tactile": tg}


def _outputs(source) -> dict:
    return {name: _make_output_group(gt) for name, gt in source.output_spec().items()}


def _metadata(
    num_segments: int = NUM_SEGMENTS, num_taxels: int = NUM_TAXELS
) -> TactileMetadata:
    indices = np.zeros((num_taxels, 4), np.uint32)
    weights = np.zeros((num_taxels, 4), np.float32)
    weights[:, 0] = 1.0
    segment_poses = np.zeros((num_segments, 7), np.float32)
    segment_poses[:, 6] = 1
    taxel_poses = np.zeros((num_taxels, 7), np.float32)
    taxel_poses[:, :3] = 0.01
    taxel_poses[:, 6] = 1
    return TactileMetadata(
        LBS(segment_poses, taxel_poses, indices, weights),
        HandSkeletonRef([HandJoint(i) for i in range(num_segments)]),
    )


class TestTactileSource:
    def test_creation_and_tracker(self):
        assert _source().get_tracker().get_name() == "TactileTracker"

    def test_specs(self):
        src = _source()
        assert list(src.input_spec()) == ["deviceio_tactile"]
        forces = src.output_spec()[TactileSource.FORCES]
        assert forces.is_optional
        assert forces.inner_type.types[0].shape == (NUM_TAXELS, 3)
        assert src.output_spec()[TactileSource.LAYOUT].is_optional

    def test_index_matches_layout_order(self):
        names = [t.name for t in TactileLBSLayout(NUM_SEGMENTS, NUM_TAXELS).types]
        assert names == [member.name.lower() for member in TactileLBSLayoutIndex]

    def test_active_conversion(self):
        src = _source()
        forces = np.arange(9, dtype=np.float32).reshape(3, 3)
        outputs = _outputs(src)
        src.compute(_inputs(src, TactileDataArray(forces), _metadata()), outputs)

        np.testing.assert_array_equal(
            np.asarray(outputs[TactileSource.FORCES][0]), forces
        )
        group = outputs[TactileSource.LAYOUT]
        assert not group.is_none
        joints = np.asarray(group[TactileLBSLayoutIndex.SEGMENT_JOINT_IDS])
        assert joints.dtype == np.int32
        assert joints.tolist() == [0, 1]
        assert (
            np.asarray(group[TactileLBSLayoutIndex.TAXEL_SEGMENT_INDICES]).dtype
            == np.int32
        )
        np.testing.assert_array_equal(
            np.asarray(group[TactileLBSLayoutIndex.TAXEL_BIND_POSES])[:, 3:],
            np.tile([0.0, 0.0, 0.0, 1.0], (NUM_TAXELS, 1)),
        )

    def test_layout_outputs_do_not_alias_the_metadata(self):
        src = _source()
        metadata = _metadata()
        encoded = metadata.to_bytes()
        outputs = _outputs(src)
        src.compute(_inputs(src, None, metadata), outputs)

        group = outputs[TactileSource.LAYOUT]
        for index in TactileLBSLayoutIndex:
            array = np.asarray(group[index])
            array[...] = 7
        assert metadata.to_bytes() == encoded

    def test_metadata_without_data(self):
        src = _source()
        outputs = _outputs(src)
        src.compute(_inputs(src, None, _metadata()), outputs)
        assert outputs[TactileSource.FORCES].is_none
        assert not outputs[TactileSource.LAYOUT].is_none

    def test_inactive_sets_none(self):
        src = _source()
        outputs = _outputs(src)
        src.compute(_inputs(src, None, None), outputs)
        assert outputs[TactileSource.FORCES].is_none
        assert outputs[TactileSource.LAYOUT].is_none

    def test_taxel_count_mismatch_raises(self):
        src = _source(num_taxels=4)
        frame = TactileDataArray(np.zeros((3, 3), np.float32))
        with pytest.raises(ValueError, match="3 taxels"):
            src.compute(_inputs(src, frame, None), _outputs(src))

    def test_metadata_count_mismatch_raises(self):
        src = _source(num_segments=NUM_SEGMENTS + 1)
        with pytest.raises(ValueError, match="2 segments"):
            src.compute(_inputs(src, None, _metadata()), _outputs(src))

    def test_layout_arrays_are_reused_while_forces_update(self):
        src = _source()
        metadata = _metadata()
        first = src(
            _inputs(
                src, TactileDataArray(np.zeros((NUM_TAXELS, 3), np.float32)), metadata
            )
        )
        second = src(
            _inputs(
                src, TactileDataArray(np.ones((NUM_TAXELS, 3), np.float32)), metadata
            )
        )

        assert first[TactileSource.LAYOUT] is not second[TactileSource.LAYOUT]
        for index in TactileLBSLayoutIndex:
            array = first[TactileSource.LAYOUT][index]
            assert second[TactileSource.LAYOUT][index] is array
            np.testing.assert_array_equal(np.from_dlpack(array), array)
        np.testing.assert_array_equal(first[TactileSource.FORCES][0], 0)
        np.testing.assert_array_equal(second[TactileSource.FORCES][0], 1)

    def test_layout_cache_follows_the_metadata_buffer(self):
        src = _source()
        metadata = _metadata()
        first = src(_inputs(src, None, metadata))
        replacement = _metadata()
        replacement.geometry_mapping.taxel_bind_poses[:, 0] = 0.02
        second = src(_inputs(src, None, replacement))

        for index in TactileLBSLayoutIndex:
            assert (
                second[TactileSource.LAYOUT][index]
                is not first[TactileSource.LAYOUT][index]
            )
        taxel_poses = TactileLBSLayoutIndex.TAXEL_BIND_POSES
        np.testing.assert_array_equal(
            first[TactileSource.LAYOUT][taxel_poses][:, 0], np.float32(0.01)
        )
        np.testing.assert_array_equal(
            second[TactileSource.LAYOUT][taxel_poses][:, 0], np.float32(0.02)
        )

        with pytest.raises(ValueError, match="4 taxels"):
            src(_inputs(src, None, _metadata(num_taxels=NUM_TAXELS + 1)))

        outputs = _outputs(src)
        src.compute(_inputs(src, None, None), outputs)
        assert outputs[TactileSource.LAYOUT].is_none
        src.compute(_inputs(src, None, replacement), outputs)
        for index in TactileLBSLayoutIndex:
            assert (
                outputs[TactileSource.LAYOUT][index]
                is second[TactileSource.LAYOUT][index]
            )
