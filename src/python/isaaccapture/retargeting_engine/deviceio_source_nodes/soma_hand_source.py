# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeviceIO SOMA hand source for rotation and evaluated-pose transports."""

from enum import Enum
from typing import TYPE_CHECKING, Any


from isaaccapture.deviceio_trackers import (
    SomaHandJointPosesTracker,
    SomaHandJointRotationsTracker,
)
from isaaccapture.schema import SomaHandedness

from .deviceio_tensor_types import (
    DeviceIOSomaHandJointPosesTracked,
    DeviceIOSomaHandJointRotationsTracked,
)
from .interface import IDeviceIOSource
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType
from ..tensor_types import SomaHandInput, SomaHandInputIndex
from ..utilities.soma_hand_evaluator import _SomaHandEvaluator
from ..utilities.soma_body_evaluator import _dense_joint_poses

if TYPE_CHECKING:
    from isaaccapture.deviceio import ITracker


class SomaHandRepresentation(str, Enum):
    JOINT_ROTATIONS = "joint-rotations"
    JOINT_POSES = "joint-poses"


class SomaHandSource(IDeviceIOSource):
    """Normalize one SOMA hand collection into evaluated joint poses."""

    HAND = "soma_hand"

    def __init__(
        self,
        name: str,
        collection_id: str,
        handedness: SomaHandedness,
        layer: Any | None = None,
        *,
        representation: SomaHandRepresentation
        | str = SomaHandRepresentation.JOINT_ROTATIONS,
    ) -> None:
        if not collection_id:
            raise ValueError("SOMA hand collection_id must not be empty")
        if handedness not in (SomaHandedness.LEFT, SomaHandedness.RIGHT):
            raise ValueError("SOMA hand source requires LEFT or RIGHT handedness")
        self.handedness = handedness
        self.representation = SomaHandRepresentation(representation)
        if self.representation is SomaHandRepresentation.JOINT_ROTATIONS:
            if layer is None:
                raise ValueError("joint-rotations requires a prepared SOMA hand layer")
            self._tracker = SomaHandJointRotationsTracker(collection_id)
            self._input_type = DeviceIOSomaHandJointRotationsTracked()
        else:
            self._tracker = SomaHandJointPosesTracker(collection_id)
            self._input_type = DeviceIOSomaHandJointPosesTracked()
        # Pybind enum values read from buffers are not singleton objects.
        expected_hand_type = "left" if handedness == SomaHandedness.LEFT else "right"
        self._evaluator = (
            _SomaHandEvaluator(layer, expected_hand_type) if layer is not None else None
        )
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        return self._tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        group = TensorGroup(self.input_spec()["deviceio_soma_hand"])
        group[0] = self._tracker.get_data(deviceio_session)
        return {"deviceio_soma_hand": group}

    def input_spec(self) -> RetargeterIOType:
        return {"deviceio_soma_hand": self._input_type}

    def output_spec(self) -> RetargeterIOType:
        return {self.HAND: OptionalType(SomaHandInput())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        data = inputs["deviceio_soma_hand"][0]
        group = outputs[self.HAND]
        if data is None:
            group.set_none()
            return
        if data.handedness != self.handedness:
            raise ValueError(
                f"SOMA hand payload handedness {data.handedness} does not match "
                f"source handedness {self.handedness}"
            )

        if self.representation is SomaHandRepresentation.JOINT_ROTATIONS:
            assert self._evaluator is not None
            positions, orientations, valid = self._evaluator.evaluate(data)
        else:
            positions, orientations, valid = _dense_joint_poses(data.joint_poses, 25)
        group[SomaHandInputIndex.JOINT_POSITIONS] = positions
        group[SomaHandInputIndex.JOINT_ORIENTATIONS] = orientations
        group[SomaHandInputIndex.JOINT_VALID] = valid
