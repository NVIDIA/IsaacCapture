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
    segment_bind_poses = np.zeros((num_joints, 7), np.float32)
    segment_bind_poses[:, 6] = 1
    taxels = np.zeros((num_taxels, 7), np.float32)
    taxels[:, :3] = 0.01
    taxels[:, 6] = 1
    indices = np.zeros((num_taxels, 4), np.uint32)
    indices[:, 0] = num_joints - 1
    weights = np.zeros((num_taxels, 4), np.float32)
    weights[:, 0] = 1
    return dict(
        segment_bind_poses=segment_bind_poses,
        taxel_bind_poses=taxels,
        taxel_segment_indices=indices,
        taxel_segment_weights=weights,
    )


def test_metadata_round_trip():
    arrays = lbs_arrays()
    metadata = TactileMetadata(
        LBS(**arrays), HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
    )
    assert metadata.geometry_mapping_type == AlgorithmSource.LBS
    assert metadata.segment_source_type == SegmentSource.HandSkeletonRef
    assert metadata.segment_source.joint_ids == [HandJoint.WRIST, HandJoint.INDEX_TIP]
    for name, expected in arrays.items():
        actual = getattr(metadata.geometry_mapping, name)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    assert metadata.geometry_mapping.taxel_segment_indices[0, 0] == 1


@pytest.mark.parametrize("joint_ids", [[], [HandJoint.WRIST, HandJoint.INDEX_TIP]])
def test_joint_ids_can_reconstruct_hand_reference(joint_ids):
    hand = HandSkeletonRef(joint_ids)
    rebuilt = HandSkeletonRef(hand.joint_ids)
    assert all(isinstance(joint, HandJoint) for joint in hand.joint_ids)
    assert rebuilt.joint_ids == joint_ids
    assert rebuilt.to_bytes() == hand.to_bytes()


def test_nested_views_and_arrays_retain_the_buffer():
    metadata = TactileMetadata(
        LBS(**lbs_arrays()), HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
    )
    geometry_mapping = metadata.geometry_mapping
    segment_source = metadata.segment_source
    rows = geometry_mapping.taxel_bind_poses
    encoded = geometry_mapping.to_bytes()
    del metadata, geometry_mapping
    gc.collect()
    assert segment_source.joint_ids == [HandJoint.WRIST, HandJoint.INDEX_TIP]
    np.testing.assert_array_equal(rows, lbs_arrays()["taxel_bind_poses"])
    assert encoded


def test_metadata_revalidates_existing_geometry_mapping():
    geometry_mapping = LBS(**lbs_arrays())
    geometry_mapping.taxel_segment_weights[0] = 0
    with pytest.raises(ValueError, match="taxel_segment_weights"):
        TactileMetadata(
            geometry_mapping, HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
        )


@pytest.mark.parametrize(
    "field", ["taxel_bind_poses", "taxel_segment_indices", "taxel_segment_weights"]
)
def test_rejects_mismatched_taxel_lengths(field):
    arrays = lbs_arrays()
    arrays[field] = arrays[field][:2]
    geometry_mapping = LBS(**arrays)
    with pytest.raises(ValueError, match=r"taxel_segment_(indices|weights): expected"):
        TactileMetadata(
            geometry_mapping, HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
        )


def test_rejects_mismatched_segment_mapping():
    with pytest.raises(ValueError, match=r"segment_bind_poses.*joint_ids"):
        TactileMetadata(LBS(**lbs_arrays()), HandSkeletonRef([HandJoint.WRIST]))


def test_rejects_out_of_range_zero_weight_slot():
    arrays = lbs_arrays()
    arrays["taxel_segment_indices"][0, 3] = 2
    geometry_mapping = LBS(**arrays)
    with pytest.raises(ValueError, match="taxel_segment_indices"):
        TactileMetadata(
            geometry_mapping, HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
        )


@pytest.mark.parametrize("weight", [-1.0, np.inf, np.nan, 0.5])
def test_rejects_invalid_weights(weight):
    arrays = lbs_arrays()
    arrays["taxel_segment_weights"][0, 0] = weight
    geometry_mapping = LBS(**arrays)
    with pytest.raises(ValueError, match="taxel_segment_weights"):
        TactileMetadata(
            geometry_mapping, HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
        )


@pytest.mark.parametrize("field", ["segment_bind_poses", "taxel_bind_poses"])
@pytest.mark.parametrize(
    "qw", [0, 2, np.finfo(np.float32).max, np.sqrt(np.float32(1 + 2e-4))]
)
def test_rejects_nonunit_quaternions(field, qw):
    arrays = lbs_arrays()
    arrays[field][-1, 6] = qw
    geometry_mapping = LBS(**arrays)
    with pytest.raises(ValueError, match="unit quaternions"):
        TactileMetadata(
            geometry_mapping, HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
        )


@pytest.mark.parametrize("field", ["segment_bind_poses", "taxel_bind_poses"])
@pytest.mark.parametrize(
    "quaternion",
    [
        [0.5, 0.5, 0.5, 0.5],
        [0, 0, 0, -1],
        [0, 0, 0, np.sqrt(np.float32(1 + 0.5e-4))],
    ],
)
def test_accepts_unit_quaternions_with_rounding_tolerance(field, quaternion):
    arrays = lbs_arrays()
    arrays[field][-1, 3:] = quaternion
    TactileMetadata(
        LBS(**arrays), HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
    )


@pytest.mark.parametrize("field", ["segment_bind_poses", "taxel_bind_poses"])
@pytest.mark.parametrize("component", range(7))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_bind_poses(field, component, value):
    arrays = lbs_arrays()
    arrays[field][-1, component] = value
    geometry_mapping = LBS(**arrays)
    with pytest.raises(ValueError, match="finite"):
        TactileMetadata(
            geometry_mapping, HandSkeletonRef([HandJoint.WRIST, HandJoint.INDEX_TIP])
        )


@pytest.mark.parametrize(
    "field",
    [
        "segment_bind_poses",
        "taxel_bind_poses",
        "taxel_segment_indices",
        "taxel_segment_weights",
    ],
)
def test_lbs_rejects_wrong_array_width(field):
    arrays = lbs_arrays()
    arrays[field] = arrays[field][:, :-1]
    with pytest.raises(ValueError, match=field):
        LBS(**arrays)


@pytest.mark.parametrize(
    "joint_ids", [[HandJoint.WRIST, HandJoint.WRIST], [HandJoint.NUM_JOINTS]]
)
def test_rejects_invalid_joint_mapping(joint_ids):
    segment_source = HandSkeletonRef(joint_ids)
    geometry_mapping = LBS(**lbs_arrays(num_joints=len(joint_ids)))
    with pytest.raises(ValueError, match="joint_ids"):
        TactileMetadata(geometry_mapping, segment_source)
