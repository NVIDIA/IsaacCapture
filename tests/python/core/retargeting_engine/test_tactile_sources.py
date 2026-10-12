# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TactileSource / TactileShapeSource DeviceIO converters (internal only).

Payloads are built through the real schema bindings; no OpenXR device is involved.
"""

import numpy as np
import pytest

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    TactileShapeIndex,
    TactileShapeLayout,
    TactileShapeSource,
    TactileSource,
)
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import TactileData, TactileShape

NUM_JOINTS = 2
NUM_TAXELS = 3


def _inputs(source, name: str, payload) -> dict:
    tg = TensorGroup(source.input_spec()[name])
    tg[0] = payload
    return {name: tg}


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


class _ScriptedTracker:
    """Stands in for a tracker: hands out the queued payloads, then None."""

    def __init__(self, payloads):
        self._payloads = list(payloads)

    def get_data(self, _session):
        return self._payloads.pop(0) if self._payloads else None


class TestTactileSource:
    def test_creation_and_tracker(self):
        src = TactileSource("tactile", collection_id="tactile/right", num_taxels=3)
        assert src.get_tracker().get_name() == "TactileTracker"

    def test_specs(self):
        src = TactileSource("tactile", collection_id="tactile/right", num_taxels=3)
        assert list(src.input_spec()) == ["deviceio_tactile"]
        out = src.output_spec()[TactileSource.FORCES]
        assert out.is_optional
        assert out.inner_type.types[0].shape == (NUM_TAXELS, 3)

    def test_active_conversion(self):
        src = TactileSource("tactile", collection_id="tactile/right", num_taxels=3)
        forces = np.arange(9, dtype=np.float32).reshape(3, 3)
        outputs = _outputs(src)
        src.compute(_inputs(src, "deviceio_tactile", TactileData(forces)), outputs)

        group = outputs[TactileSource.FORCES]
        assert not group.is_none
        np.testing.assert_array_equal(np.asarray(group[0]), forces)

    def test_inactive_sets_none(self):
        src = TactileSource("tactile", collection_id="tactile/right", num_taxels=3)
        outputs = _outputs(src)
        src.compute(_inputs(src, "deviceio_tactile", None), outputs)
        assert outputs[TactileSource.FORCES].is_none

    def test_taxel_count_mismatch_raises(self):
        src = TactileSource("tactile", collection_id="tactile/right", num_taxels=4)
        frame = TactileData(np.zeros((3, 3), np.float32))
        with pytest.raises(ValueError, match="3 taxels"):
            src.compute(_inputs(src, "deviceio_tactile", frame), _outputs(src))


class TestTactileShapeSource:
    def _source(self, **kwargs) -> TactileShapeSource:
        args = dict(collection_id="tactile_shape/right", num_joints=NUM_JOINTS)
        args["num_taxels"] = NUM_TAXELS
        args.update(kwargs)
        return TactileShapeSource("shape", **args)

    def test_creation_and_tracker(self):
        assert self._source().get_tracker().get_name() == "TactileShapeTracker"

    def test_index_matches_layout_order(self):
        names = [t.name for t in TactileShapeLayout(NUM_JOINTS, NUM_TAXELS).types]
        assert names == [member.name.lower() for member in TactileShapeIndex]

    def test_conversion(self):
        src = self._source()
        outputs = _outputs(src)
        src.compute(_inputs(src, "deviceio_tactile_shape", _shape()), outputs)

        group = outputs[TactileShapeSource.SHAPE]
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

    def test_none_before_first_shape(self):
        src = self._source()
        outputs = _outputs(src)
        src.compute(_inputs(src, "deviceio_tactile_shape", None), outputs)
        assert outputs[TactileShapeSource.SHAPE].is_none

    def test_keeps_last_shape_when_tracker_goes_empty(self):
        src = self._source()
        src._tracker = _ScriptedTracker([None, _shape(), None, None])

        polled = [src.poll_tracker(None) for _ in range(4)]

        payloads = [p["deviceio_tactile_shape"][0] for p in polled]
        assert payloads[0] is None
        assert payloads[1] is not None
        assert payloads[2] is payloads[1] and payloads[3] is payloads[1]

    def test_count_mismatch_raises(self):
        src = self._source(num_taxels=NUM_TAXELS + 1)
        with pytest.raises(ValueError, match="3 taxels"):
            src.compute(_inputs(src, "deviceio_tactile_shape", _shape()), _outputs(src))
