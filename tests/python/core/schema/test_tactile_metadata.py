# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tactile metadata union composition, binding invariants, and nested view lifetime."""

import gc

import numpy as np
import pytest

from isaaccapture.schema import (
    AlgorithmSource,
    HandJoint,
    HandSkeletonRef,
    LBS,
    SegmentSource,
    TactileMetadata,
)


def lbs_arrays(num_joints: int = 2, num_taxels: int = 3) -> dict:
    segments = np.zeros((num_joints, 7), np.float32)
    segments[:, 6] = 1
    taxels = np.zeros((num_taxels, 7), np.float32)
    taxels[:, :3] = 0.01
    taxels[:, 6] = 1
    indices = np.zeros((num_taxels, 4), np.uint32)
    indices[:, 0] = num_joints - 1
    weights = np.zeros((num_taxels, 4), np.float32)
    weights[:, 0] = 1
    return dict(
        segment_bind_pose=segments,
        taxel_bind_pose=taxels,
        taxel_segment_indices=indices,
        taxel_segment_weights=weights,
    )


def test_metadata_round_trip():
    arrays = lbs_arrays()
    metadata = TactileMetadata(
        LBS(**arrays), HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
    )
    assert metadata.algorithm_type == AlgorithmSource.LBS
    assert metadata.segments_type == SegmentSource.HandSkeletonRef
    assert metadata.segments.joints == [HandJoint.WRIST, HandJoint.INDEX_TIP]
    for name, expected in arrays.items():
        actual = getattr(metadata.algorithm, name)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    assert metadata.algorithm.taxel_segment_indices[0, 0] == 1


def test_nested_views_and_arrays_retain_the_buffer():
    metadata = TactileMetadata(
        LBS(**lbs_arrays()), HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
    )
    algorithm = metadata.algorithm
    joints = metadata.segments
    rows = algorithm.taxel_bind_pose
    encoded = algorithm.to_bytes()
    del metadata, algorithm
    gc.collect()
    assert joints.joints == [HandJoint.WRIST, HandJoint.INDEX_TIP]
    np.testing.assert_array_equal(rows, lbs_arrays()["taxel_bind_pose"])
    assert encoded


def test_rejects_mismatched_taxel_lengths():
    arrays = lbs_arrays()
    arrays["taxel_bind_pose"] = arrays["taxel_bind_pose"][:2]
    with pytest.raises(ValueError, match="same length"):
        LBS(**arrays)


def test_rejects_mismatched_segment_mapping():
    with pytest.raises(ValueError, match="same length"):
        TactileMetadata(LBS(**lbs_arrays()), HandSkeletonRef([HandJoint.WRIST]))


def test_rejects_out_of_range_zero_weight_slot():
    arrays = lbs_arrays()
    arrays["taxel_segment_indices"][0, 3] = 2
    with pytest.raises(ValueError, match="taxel_segment_indices"):
        LBS(**arrays)


@pytest.mark.parametrize("weight", [-1.0, np.inf, np.nan, 0.5])
def test_rejects_invalid_weights(weight):
    arrays = lbs_arrays()
    arrays["taxel_segment_weights"][0, 0] = weight
    with pytest.raises(ValueError, match="taxel_segment_weights"):
        LBS(**arrays)


@pytest.mark.parametrize("field", ["segment_bind_pose", "taxel_bind_pose"])
def test_rejects_nonunit_quaternions(field):
    arrays = lbs_arrays()
    arrays[field][0, 6] = 0
    with pytest.raises(ValueError, match="unit quaternions"):
        LBS(**arrays)


def test_rejects_nonfinite_bind_positions():
    arrays = lbs_arrays()
    arrays["segment_bind_pose"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        LBS(**arrays)


def test_rejects_wrong_influence_width():
    arrays = lbs_arrays()
    arrays["taxel_segment_weights"] = np.zeros((3, 3), np.float32)
    with pytest.raises(ValueError, match="taxel_segment_weights"):
        LBS(**arrays)


@pytest.mark.parametrize(
    "joints", [[HandJoint.WRIST, HandJoint.WRIST], [HandJoint.NUM_JOINTS]]
)
def test_rejects_invalid_joint_mapping(joints):
    with pytest.raises(ValueError, match="joints"):
        HandSkeletonRef(joints)
