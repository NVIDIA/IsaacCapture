# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeviceIO SOMA body source for rotation and evaluated-pose transports."""

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from isaaccapture.deviceio_trackers import (
    SomaBodyJointPosesV0Tracker,
    SomaBodyJointRotationsV0Tracker,
)

from .deviceio_tensor_types import (
    DeviceIOSomaBodyJointPosesV0Tracked,
    DeviceIOSomaBodyJointRotationsV0Tracked,
)
from .interface import IDeviceIOSource
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType
from ..tensor_types import SomaBodyInput, SomaBodyInputIndex
from ..utilities.soma_body_evaluator import _SomaBodyEvaluator

if TYPE_CHECKING:
    from isaaccapture.deviceio import ITracker


class SomaBodyRepresentation(str, Enum):
    JOINT_ROTATIONS = "joint-rotations"
    JOINT_POSES = "joint-poses"


class SomaBodySource(IDeviceIOSource):
    """Normalize either SOMA V0 wire profile into evaluated joint poses.

    The generated tracker remains the raw transport and recording boundary. The
    rotation profile runs FK once per graph step. The evaluated-pose profile maps
    directly into the same downstream ``SomaBodyInput``.
    """

    BODY = "soma_body"

    def __init__(
        self,
        name: str,
        collection_id: str,
        layer: Any | None = None,
        *,
        representation: SomaBodyRepresentation
        | str = SomaBodyRepresentation.JOINT_ROTATIONS,
    ) -> None:
        if not collection_id:
            raise ValueError("SOMA body collection_id must not be empty")
        self.representation = SomaBodyRepresentation(representation)
        if self.representation is SomaBodyRepresentation.JOINT_ROTATIONS:
            if layer is None:
                raise ValueError("joint-rotations requires a prepared SOMA layer")
            self._tracker = SomaBodyJointRotationsV0Tracker(collection_id)
            self._input_type = DeviceIOSomaBodyJointRotationsV0Tracked()
        else:
            self._tracker = SomaBodyJointPosesV0Tracker(collection_id)
            self._input_type = DeviceIOSomaBodyJointPosesV0Tracked()
        self._evaluator = _SomaBodyEvaluator(layer) if layer is not None else None
        self.joint_names = self._evaluator.joint_names if self._evaluator else ()
        self.bones = self._evaluator.bones if self._evaluator else ()
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        return self._tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        group = TensorGroup(self.input_spec()["deviceio_soma_body"])
        group[0] = self._tracker.get_data(deviceio_session)
        return {"deviceio_soma_body": group}

    def input_spec(self) -> RetargeterIOType:
        return {"deviceio_soma_body": self._input_type}

    def output_spec(self) -> RetargeterIOType:
        return {self.BODY: OptionalType(SomaBodyInput())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        data = inputs["deviceio_soma_body"][0]
        group = outputs[self.BODY]
        if data is None:
            group.set_none()
            return

        if self.representation is SomaBodyRepresentation.JOINT_ROTATIONS:
            assert self._evaluator is not None
            positions, orientations, valid = self._evaluator.evaluate(data)
        elif data.joint_poses is None:
            positions = np.zeros((77, 3), dtype=np.float32)
            orientations = np.zeros((77, 4), dtype=np.float32)
            orientations[:, 3] = 1.0
            valid = np.zeros(77, dtype=np.uint8)
        else:
            positions = np.asarray(data.joint_poses.positions, dtype=np.float32)
            orientations = np.asarray(data.joint_poses.orientations, dtype=np.float32)
            valid = np.asarray(data.joint_poses.is_valid, dtype=np.uint8)
        group[SomaBodyInputIndex.JOINT_POSITIONS] = positions
        group[SomaBodyInputIndex.JOINT_ORIENTATIONS] = orientations
        group[SomaBodyInputIndex.JOINT_VALID] = valid
