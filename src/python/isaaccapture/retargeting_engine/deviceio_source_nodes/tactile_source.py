# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tactile Source Node - DeviceIO to Retargeting Engine converter.

``TactileSource`` turns per-frame ``TactileDataArray`` into a force array, and the tracker's
``TactileMetadata`` metadata into the arrays that lay those forces out. Taxel and joint counts are
fixed at construction so the retargeting graph has a static spec.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from .interface import IDeviceIOSource
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType
from ..tensor_types import (
    DeviceIOTactileIndex,
    TactileForces,
    TactileLBSLayout,
    TactileLBSLayoutIndex,
)
from .deviceio_tensor_types import DeviceIOTactileTracked

if TYPE_CHECKING:
    from isaaccapture.deviceio_trackers import ITracker
    from isaaccapture.schema import TactileDataArray, TactileMetadata


class TactileSource(IDeviceIOSource):
    """Convert tactile frames and metadata into per-taxel forces and an LBS layout.

    Outputs (Optional):
        - :data:`FORCES`: :func:`TactileForces` of ``num_taxels``, in ``TactileMetadata`` taxel
          order; absent when the device is inactive.
        - :data:`LAYOUT`: :func:`TactileLBSLayout` of ``num_segments`` x ``num_taxels``; absent
          until the tracker holds the device's metadata. The tracker publishes no forces before then.

    A frame or metadata whose counts differ from the constructor's raises ``ValueError``.
    Layout arrays are reused across frames; copy them before modifying them.
    """

    FORCES = "taxel_forces"
    LAYOUT = "tactile_layout"

    def __init__(
        self,
        name: str,
        collection_id: str,
        num_segments: int,
        num_taxels: int,
        max_flatbuffer_size: Optional[int] = None,
    ) -> None:
        """Initialize the tactile source.

        Args:
            name: Unique name for this source node.
            collection_id: Tensor collection ID of the producer's ``TactileDataArray`` stream.
            num_segments: Referenced joint count; fixes the LBS layout output spec.
            num_taxels: Taxel count of the device; fixes the output specs.
            max_flatbuffer_size: Override for ``TactileTracker``'s default collection size.
        """
        from isaaccapture.deviceio_trackers import TactileTracker

        self._tracker = (
            TactileTracker(collection_id)
            if max_flatbuffer_size is None
            else TactileTracker(collection_id, max_flatbuffer_size)
        )
        self._num_segments = int(num_segments)
        self._num_taxels = int(num_taxels)
        self._layout: dict[TactileLBSLayoutIndex, np.ndarray] | None = None
        self._layout_bind_poses: np.ndarray | None = None
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Return the ``TactileTracker`` instance for ``TeleopSession`` to initialize."""
        return self._tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the tracker's frame and metadata for the compute step."""
        tg = TensorGroup(DeviceIOTactileTracked())
        tg[DeviceIOTactileIndex.DATA] = self._tracker.get_data(deviceio_session)
        tg[DeviceIOTactileIndex.METADATA] = self._tracker.get_metadata(deviceio_session)
        return {"deviceio_tactile": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare the raw DeviceIO tactile input."""
        return {"deviceio_tactile": DeviceIOTactileTracked()}

    def output_spec(self) -> RetargeterIOType:
        """Declare the force and LBS layout outputs (Optional -- may be absent)."""
        return {
            self.FORCES: OptionalType(TactileForces(self._num_taxels)),
            self.LAYOUT: OptionalType(
                TactileLBSLayout(self._num_segments, self._num_taxels)
            ),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Publish forces and a cached LBS layout; mark absent outputs with ``set_none()``."""
        data: "TactileDataArray | None" = inputs["deviceio_tactile"][
            DeviceIOTactileIndex.DATA
        ]
        metadata: "TactileMetadata | None" = inputs["deviceio_tactile"][
            DeviceIOTactileIndex.METADATA
        ]

        forces_out = outputs[self.FORCES]
        if data is None:
            forces_out.set_none()
        else:
            forces = data.taxel_forces
            if len(forces) != self._num_taxels:
                raise ValueError(
                    f"{self.name}: frame has {len(forces)} taxels, source expects {self._num_taxels}"
                )
            forces_out[0] = forces

        layout_out = outputs[self.LAYOUT]
        if metadata is None:
            layout_out.set_none()
            return
        geometry_mapping = metadata.geometry_mapping
        segment_source = metadata.segment_source
        if geometry_mapping is None or segment_source is None:
            raise ValueError(
                f"{self.name}: metadata must specify geometry_mapping and segment_source"
            )
        # Normalize the schema stub's array type without copying its storage.
        segment_bind_poses = np.asarray(geometry_mapping.segment_bind_poses)
        if (
            self._layout is None
            or self._layout_bind_poses is None
            or not np.shares_memory(segment_bind_poses, self._layout_bind_poses)
        ):
            num_segments, num_taxels = (
                len(segment_source.joint_ids),
                len(geometry_mapping.taxel_bind_poses),
            )
            if (num_segments, num_taxels) != (self._num_segments, self._num_taxels):
                raise ValueError(
                    f"{self.name}: metadata has {num_segments} segments and {num_taxels} taxels, "
                    f"source expects {self._num_segments} and {self._num_taxels}"
                )
            values = {
                TactileLBSLayoutIndex.SEGMENT_JOINT_IDS: np.asarray(
                    segment_source.joint_ids, dtype=np.int32
                ),
                TactileLBSLayoutIndex.SEGMENT_BIND_POSES: segment_bind_poses,
                TactileLBSLayoutIndex.TAXEL_BIND_POSES: geometry_mapping.taxel_bind_poses,
                TactileLBSLayoutIndex.TAXEL_SEGMENT_INDICES: geometry_mapping.taxel_segment_indices,
                TactileLBSLayoutIndex.TAXEL_SEGMENT_WEIGHTS: geometry_mapping.taxel_segment_weights,
            }
            arrays = {}
            for index, value in values.items():
                # Copy once: graph consumers must not alter the tracker's metadata baseline.
                array = np.asarray(value)
                arrays[index] = array.astype(
                    np.int32 if array.dtype == np.uint32 else array.dtype
                )
            self._layout = arrays
            # Views keep the buffer alive and identify it across fresh Python wrappers.
            self._layout_bind_poses = segment_bind_poses
        for index, array in self._layout.items():
            layout_out[index] = array
