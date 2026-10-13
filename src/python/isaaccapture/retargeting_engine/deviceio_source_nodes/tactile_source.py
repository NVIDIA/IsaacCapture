# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tactile Source Nodes - DeviceIO to Retargeting Engine converters.

``TactileSource`` turns per-frame ``TactileData`` into a force array; ``TactileShapeSource``
turns the device's static ``TactileShape`` into the arrays that lay those forces out. Taxel and
joint counts are fixed at construction so the retargeting graph has a static spec.
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
from .deviceio_tensor_types import (
    DeviceIOTactileDataTracked,
    DeviceIOTactileShapeTracked,
)

if TYPE_CHECKING:
    from isaaccapture.deviceio_trackers import ITracker
    from isaaccapture.schema import TactileData, TactileShape


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

    JOINT_BIND_POSITIONS = 0
    JOINT_BIND_ORIENTATIONS = 1
    JOINT_PARENTS = 2
    TAXEL_POSITIONS = 3
    TAXEL_NORMALS = 4
    TAXEL_TANGENTS = 5
    TAXEL_JOINT_INDICES = 6
    TAXEL_JOINT_WEIGHTS = 7


def TactileShapeLayout(num_joints: int, num_taxels: int) -> TensorGroupType:
    """``TactileShape`` as arrays, ordered by :class:`TactileShapeIndex`.

    Each array is named after the ``TactileShape`` property it is read from. Joint parents and
    joint indices are int32; the root parent stays ``TactileShape.ROOT_PARENT``.
    """
    return TensorGroupType(
        f"tactile_shape_{num_joints}x{num_taxels}",
        [
            _float_array("joint_bind_positions", num_joints, 3),
            _float_array("joint_bind_orientations", num_joints, 4),
            _int_array("joint_parents", (num_joints,)),
            _float_array("taxel_positions", num_taxels, 3),
            _float_array("taxel_normals", num_taxels, 3),
            _float_array("taxel_tangents", num_taxels, 3),
            _int_array("taxel_joint_indices", (num_taxels, 4)),
            _float_array("taxel_joint_weights", num_taxels, 4),
        ],
    )


class TactileSource(IDeviceIOSource):
    """Converter: DeviceIO ``TactileData`` -> per-taxel force array.

    Outputs (Optional -- absent when the device is inactive):
        - :data:`FORCES`: :func:`TactileForces` of ``num_taxels``, in ``TactileShape`` taxel order.

    A frame whose taxel count differs from ``num_taxels`` raises ``ValueError``.
    """

    FORCES = "taxel_forces"

    def __init__(
        self,
        name: str,
        collection_id: str,
        num_taxels: int,
        max_flatbuffer_size: Optional[int] = None,
    ) -> None:
        """Initialize the tactile source.

        Args:
            name: Unique name for this source node.
            collection_id: Tensor collection ID of the producer's ``TactileData`` stream.
            num_taxels: Taxel count of the device; fixes the output spec.
            max_flatbuffer_size: Override for ``TactileTracker``'s default collection size.
        """
        from isaaccapture.deviceio_trackers import TactileTracker

        self._tracker = (
            TactileTracker(collection_id)
            if max_flatbuffer_size is None
            else TactileTracker(collection_id, max_flatbuffer_size)
        )
        self._num_taxels = int(num_taxels)
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Return the ``TactileTracker`` instance for ``TeleopSession`` to initialize."""
        return self._tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the tracker and wrap the raw tracked data for the compute step."""
        tg = TensorGroup(DeviceIOTactileDataTracked())
        tg[0] = self._tracker.get_data(deviceio_session)
        return {"deviceio_tactile": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare the raw DeviceIO tactile input."""
        return {"deviceio_tactile": DeviceIOTactileDataTracked()}

    def output_spec(self) -> RetargeterIOType:
        """Declare the force output (Optional -- may be absent)."""
        return {self.FORCES: OptionalType(TactileForces(self._num_taxels))}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Copy the frame's forces out, or call ``set_none()`` when the device is inactive."""
        data: "TactileData | None" = inputs["deviceio_tactile"][0]

        out = outputs[self.FORCES]
        if data is None:
            out.set_none()
            return

        forces = data.taxel_force
        if len(forces) != self._num_taxels:
            raise ValueError(
                f"{self.name}: frame has {len(forces)} taxels, source expects {self._num_taxels}"
            )
        out[0] = forces


class TactileShapeSource(IDeviceIOSource):
    """Converter: DeviceIO ``TactileShape`` -> layout arrays.

    The last shape received is kept and re-emitted every frame. Producers publish the shape
    only occasionally, and replay reads one record per update, so it runs out of recorded
    shapes after the first few frames.

    Outputs (Optional -- absent until the first shape arrives):
        - :data:`SHAPE`: :func:`TactileShapeLayout` of ``num_joints`` x ``num_taxels``.

    A shape with other counts raises ``ValueError``.
    """

    SHAPE = "tactile_shape"

    def __init__(
        self,
        name: str,
        collection_id: str,
        num_joints: int,
        num_taxels: int,
        max_flatbuffer_size: Optional[int] = None,
    ) -> None:
        """Initialize the tactile shape source.

        Args:
            name: Unique name for this source node.
            collection_id: Tensor collection ID of the producer's ``TactileShape`` stream.
            num_joints: Joint count of the device skeleton; fixes the output spec.
            num_taxels: Taxel count of the device; fixes the output spec.
            max_flatbuffer_size: Override for ``TactileShapeTracker``'s default collection size.
        """
        from isaaccapture.deviceio_trackers import TactileShapeTracker

        self._tracker = (
            TactileShapeTracker(collection_id)
            if max_flatbuffer_size is None
            else TactileShapeTracker(collection_id, max_flatbuffer_size)
        )
        self._num_joints = int(num_joints)
        self._num_taxels = int(num_taxels)
        self._latest: "TactileShape | None" = None
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Return the ``TactileShapeTracker`` instance for ``TeleopSession`` to initialize."""
        return self._tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the tracker, keeping the last shape it delivered."""
        shape = self._tracker.get_data(deviceio_session)
        if shape is not None:
            self._latest = shape
        tg = TensorGroup(DeviceIOTactileShapeTracked())
        tg[0] = self._latest
        return {"deviceio_tactile_shape": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare the raw DeviceIO tactile shape input."""
        return {"deviceio_tactile_shape": DeviceIOTactileShapeTracked()}

    def output_spec(self) -> RetargeterIOType:
        """Declare the layout output (Optional -- may be absent)."""
        return {
            self.SHAPE: OptionalType(
                TactileShapeLayout(self._num_joints, self._num_taxels)
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Copy the shape's arrays out, or call ``set_none()`` before the first shape."""
        shape: "TactileShape | None" = inputs["deviceio_tactile_shape"][0]

        out = outputs[self.SHAPE]
        if shape is None:
            out.set_none()
            return

        joints, taxels = len(shape.joint_parents), len(shape.taxel_positions)
        if (joints, taxels) != (self._num_joints, self._num_taxels):
            raise ValueError(
                f"{self.name}: shape has {joints} joints and {taxels} taxels, "
                f"source expects {self._num_joints} and {self._num_taxels}"
            )
        for index, tensor_type in enumerate(out.group_type.types):
            value = getattr(shape, tensor_type.name)
            if value.dtype == np.uint32:
                value = value.astype(np.int32)
            out[index] = value
