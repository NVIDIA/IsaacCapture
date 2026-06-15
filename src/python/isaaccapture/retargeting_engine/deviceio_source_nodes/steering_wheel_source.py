# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steering wheel source node - DeviceIO to retargeting engine converter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType
from ..tensor_types import SteeringWheelInput, SteeringWheelInputIndex
from .deviceio_tensor_types import DeviceIOSteeringWheelOutputTracked
from .interface import IDeviceIOSource

if TYPE_CHECKING:
    from isaaccapture.deviceio import ITracker
    from isaaccapture.schema import SteeringWheelOutput


DEFAULT_STEERING_WHEEL_COLLECTION_ID = "steering_wheel"


class SteeringWheelSource(IDeviceIOSource):
    """Stateless converter: DeviceIO SteeringWheelOutput -> SteeringWheelInput tensors."""

    def __init__(
        self, name: str, collection_id: str = DEFAULT_STEERING_WHEEL_COLLECTION_ID
    ) -> None:
        import isaaccapture.deviceio_trackers as deviceio

        self._steering_wheel_tracker = deviceio.SteeringWheelTracker(collection_id)
        self._collection_id = collection_id
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        return self._steering_wheel_tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        tracked = self._steering_wheel_tracker.get_wheel_data(deviceio_session)
        tg = TensorGroup(DeviceIOSteeringWheelOutputTracked())
        tg[0] = tracked
        return {"deviceio_steering_wheel": tg}

    def input_spec(self) -> RetargeterIOType:
        return {
            "deviceio_steering_wheel": DeviceIOSteeringWheelOutputTracked(),
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "steering_wheel": OptionalType(SteeringWheelInput()),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        wheel: SteeringWheelOutput | None = inputs["deviceio_steering_wheel"][0]

        out = outputs["steering_wheel"]
        if wheel is None:
            out.set_none()
            return

        out[SteeringWheelInputIndex.STEERING] = float(wheel.steering)
        out[SteeringWheelInputIndex.THROTTLE] = float(wheel.throttle)
        out[SteeringWheelInputIndex.BRAKE] = float(wheel.brake)
        out[SteeringWheelInputIndex.CLUTCH] = float(wheel.clutch)
        out[SteeringWheelInputIndex.HAT_X] = float(wheel.hat_x)
        out[SteeringWheelInputIndex.HAT_Y] = float(wheel.hat_y)
