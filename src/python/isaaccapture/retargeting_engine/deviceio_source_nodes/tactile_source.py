# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tactile Source Node - DeviceIO to Retargeting Engine converter.

``TactileSource`` turns per-frame ``TactileDataArray`` into a force array, and the tracker's
``TactileMetadata`` metadata into the arrays that lay those forces out. Taxel and joint counts are
fixed at construction so the retargeting graph has a static spec.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from .interface import IDeviceIOSource
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType, TensorGroupType
from ..tensor_types import DLDataType, NDArrayType
from .deviceio_tensor_types import DeviceIOTactileTracked

if TYPE_CHECKING:
    from isaaccapture.deviceio_trackers import ITracker
    from isaaccapture.schema import TactileDataArray, TactileMetadata


def _float_array(name: str, rows: int, width: int) -> NDArrayType:
    return NDArrayType(name, shape=(rows, width), dtype=DLDataType.FLOAT, dtype_bits=32)


def _int_array(name: str, shape: tuple[int, ...]) -> NDArrayType:
    return NDArrayType(name, shape=shape, dtype=DLDataType.INT, dtype_bits=32)


def TactileForces(num_taxels: int) -> TensorGroupType:
    """Per-taxel force vectors: one ``(num_taxels, 3)`` float32 array."""
    return TensorGroupType(
        f"tactile_forces_{num_taxels}",
        [_float_array("taxel_force", num_taxels, 3)],
    )


class TactileShapeIndex(IntEnum):
    """Positions of the arrays in a :func:`TactileShapeLayout` group."""

    SEGMENT_JOINTS = 0
    SEGMENT_BIND_POSE = 1
    TAXEL_BIND_POSE = 2
    TAXEL_SEGMENT_INDICES = 3
    TAXEL_SEGMENT_WEIGHTS = 4


def TactileShapeLayout(num_joints: int, num_taxels: int) -> TensorGroupType:
    """``TactileMetadata`` as arrays, ordered by :class:`TactileShapeIndex`.

    HandJoint identifiers and segment indices are int32. Pose rows contain
    position XYZ followed by quaternion XYZW, all in hand-root space.
    """
    return TensorGroupType(
        f"tactile_shape_{num_joints}x{num_taxels}",
        [
            _int_array("segment_joints", (num_joints,)),
            _float_array("segment_bind_pose", num_joints, 7),
            _float_array("taxel_bind_pose", num_taxels, 7),
            _int_array("taxel_segment_indices", (num_taxels, 4)),
            _float_array("taxel_segment_weights", num_taxels, 4),
        ],
    )


class TactileSource(IDeviceIOSource):
    """Converter: DeviceIO ``TactileTracker`` -> per-taxel forces and the shape that lays them out.

    Outputs (Optional):
        - :data:`FORCES`: :func:`TactileForces` of ``num_taxels``, in ``TactileMetadata`` taxel
          order; absent when the device is inactive.
        - :data:`SHAPE`: :func:`TactileShapeLayout` of ``num_joints`` x ``num_taxels``; absent
          until the tracker holds the device's shape. The tracker publishes no forces before then.

    A frame or shape whose counts differ from the constructor's raises ``ValueError``.
    """

    FORCES = "taxel_forces"
    SHAPE = "tactile_shape"

    def __init__(
        self,
        name: str,
        collection_id: str,
        num_joints: int,
        num_taxels: int,
        max_flatbuffer_size: Optional[int] = None,
    ) -> None:
        """Initialize the tactile source.

        Args:
            name: Unique name for this source node.
            collection_id: Tensor collection ID of the producer's ``TactileDataArray`` stream.
            num_joints: Joint count of the device skeleton; fixes the shape output spec.
            num_taxels: Taxel count of the device; fixes the output specs.
            max_flatbuffer_size: Override for ``TactileTracker``'s default collection size.
        """
        from isaaccapture.deviceio_trackers import TactileTracker

        self._tracker = (
            TactileTracker(collection_id)
            if max_flatbuffer_size is None
            else TactileTracker(collection_id, max_flatbuffer_size)
        )
        self._num_joints = int(num_joints)
        self._num_taxels = int(num_taxels)
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Return the ``TactileTracker`` instance for ``TeleopSession`` to initialize."""
        return self._tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the tracker's frame and metadata for the compute step."""
        tg = TensorGroup(DeviceIOTactileTracked())
        tg[0] = self._tracker.get_data(deviceio_session)
        tg[1] = self._tracker.get_metadata(deviceio_session)
        return {"deviceio_tactile": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare the raw DeviceIO tactile input."""
        return {"deviceio_tactile": DeviceIOTactileTracked()}

    def output_spec(self) -> RetargeterIOType:
        """Declare the force and shape outputs (Optional -- may be absent)."""
        return {
            self.FORCES: OptionalType(TactileForces(self._num_taxels)),
            self.SHAPE: OptionalType(
                TactileShapeLayout(self._num_joints, self._num_taxels)
            ),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Publish the frame's forces and a copy of the shape; ``set_none()`` whichever is absent."""
        data: "TactileDataArray | None" = inputs["deviceio_tactile"][0]
        shape: "TactileMetadata | None" = inputs["deviceio_tactile"][1]

        forces_out = outputs[self.FORCES]
        if data is None:
            forces_out.set_none()
        else:
            forces = data.taxel_force
            if len(forces) != self._num_taxels:
                raise ValueError(
                    f"{self.name}: frame has {len(forces)} taxels, source expects {self._num_taxels}"
                )
            forces_out[0] = forces

        shape_out = outputs[self.SHAPE]
        if shape is None:
            shape_out.set_none()
            return
        algorithm = shape.algorithm
        segments = shape.segments
        if algorithm is None or segments is None:
            raise ValueError(
                f"{self.name}: metadata must specify an algorithm and segments"
            )
        joints, taxels = len(segments.joints), len(algorithm.taxel_bind_pose)
        if (joints, taxels) != (self._num_joints, self._num_taxels):
            raise ValueError(
                f"{self.name}: shape has {joints} joints and {taxels} taxels, "
                f"source expects {self._num_joints} and {self._num_taxels}"
            )
        values = (
            np.asarray(segments.joints, dtype=np.int32),
            algorithm.segment_bind_pose,
            algorithm.taxel_bind_pose,
            algorithm.taxel_segment_indices,
            algorithm.taxel_segment_weights,
        )
        for index, value in enumerate(values):
            # Copied: the views alias the metadata the tracker holds and compares later pushes to.
            shape_out[index] = value.astype(
                np.int32 if value.dtype == np.uint32 else value.dtype
            )
