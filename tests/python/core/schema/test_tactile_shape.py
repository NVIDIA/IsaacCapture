# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TactileShape in isaaccapture.schema."""

import numpy as np
import pytest

from isaaccapture.schema import TactileShape

ROOT = TactileShape.ROOT_PARENT


def shape_arrays(num_joints: int = 2, num_taxels: int = 3) -> dict:
    """A wrist-rooted chain with every taxel skinned to its last two joints."""
    indices = np.zeros((num_taxels, 4), np.uint32)
    indices[:, 0] = num_joints - 1
    weights = np.zeros((num_taxels, 4), np.float32)
    weights[:, 0] = 1.0
    return dict(
        joint_bind_positions=np.arange(num_joints * 3, dtype=np.float32).reshape(-1, 3),
        joint_bind_orientations=np.tile([0.0, 0.0, 0.0, 1.0], (num_joints, 1)).astype(
            np.float32
        ),
        joint_parents=np.array([ROOT] + list(range(num_joints - 1)), np.uint32),
        taxel_positions=np.full((num_taxels, 3), 0.01, np.float32),
        taxel_normals=np.tile([0.0, 0.0, 1.0], (num_taxels, 1)).astype(np.float32),
        taxel_tangents=np.tile([1.0, 0.0, 0.0], (num_taxels, 1)).astype(np.float32),
        taxel_joint_indices=indices,
        taxel_joint_weights=weights,
    )


class TestTactileShape:
    def test_root_parent_is_the_openxr_max_enum(self):
        assert ROOT == 0x7FFFFFFF

    def test_round_trip(self):
        arrays = shape_arrays()
        shape = TactileShape(**arrays)

        for name, expected in arrays.items():
            value = getattr(shape, name)
            assert value.dtype == expected.dtype, name
            np.testing.assert_array_equal(value, expected, err_msg=name)

    def test_bytes_round_trip(self):
        shape = TactileShape(**shape_arrays())

        restored = TactileShape.from_bytes(shape.to_bytes())

        np.testing.assert_array_equal(
            restored.taxel_joint_weights, shape.taxel_joint_weights
        )
        assert restored.to_bytes() == shape.to_bytes()

    @pytest.mark.parametrize("payload", [b"", b"not a flatbuffer"])
    def test_from_bytes_rejects_invalid_payload(self, payload):
        with pytest.raises(ValueError, match="valid FlatBuffer"):
            TactileShape.from_bytes(payload)

    def test_rejects_mismatched_joint_lengths(self):
        arrays = shape_arrays()
        arrays["joint_parents"] = arrays["joint_parents"][:1]
        with pytest.raises(ValueError, match="same length"):
            TactileShape(**arrays)

    def test_rejects_mismatched_taxel_lengths(self):
        arrays = shape_arrays()
        arrays["taxel_normals"] = arrays["taxel_normals"][:2]
        with pytest.raises(ValueError, match="same length"):
            TactileShape(**arrays)

    def test_rejects_parent_out_of_range(self):
        arrays = shape_arrays()
        arrays["joint_parents"][1] = 5
        with pytest.raises(ValueError, match="joint_parents"):
            TactileShape(**arrays)

    def test_rejects_skinning_joint_out_of_range(self):
        arrays = shape_arrays()
        arrays["taxel_joint_indices"][0, 3] = 2
        with pytest.raises(ValueError, match="taxel_joint_indices"):
            TactileShape(**arrays)

    def test_rejects_wrong_width(self):
        arrays = shape_arrays()
        arrays["taxel_joint_weights"] = np.zeros((3, 3), np.float32)
        with pytest.raises(ValueError, match="taxel_joint_weights"):
            TactileShape(**arrays)

    def test_repr(self):
        assert (
            repr(TactileShape(**shape_arrays())) == "TactileShape(joints=2, taxels=3)"
        )
