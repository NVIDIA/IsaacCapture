# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpolate a named hand posture from the controller's trigger value."""

import numpy as np

from isaaccapture.retargeting_engine.interface import BaseRetargeter
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    FloatType,
)


class ControllerPostureRetargeter(BaseRetargeter):
    """Produce trigger-scaled joint targets; rate limiting belongs downstream."""

    def __init__(self, name, input_device, joint_names, closed_posture):
        self._input_device = input_device
        self._joint_names = list(joint_names)
        self._closed = np.asarray(closed_posture, dtype=float)
        if (
            self._closed.shape != (len(self._joint_names),)
            or not np.isfinite(self._closed).all()
        ):
            raise ValueError("closed_posture must contain one finite value per joint")
        self._last = np.zeros_like(self._closed)
        super().__init__(name=name)

    def input_spec(self):
        return {self._input_device: OptionalType(ControllerInput())}

    def output_spec(self):
        return {
            "joint_targets": TensorGroupType(
                "joint_targets", [FloatType(name) for name in self._joint_names]
            )
        }

    def _compute_fn(self, inputs, outputs, context):
        if context.execution_events.reset:
            self._last[:] = 0.0
        controller = inputs[self._input_device]
        # Buttons remain usable when grip-pose tracking is occluded.
        if not controller.is_none:
            trigger = float(controller[ControllerInputIndex.TRIGGER_VALUE])
            if np.isfinite(trigger):
                self._last = np.clip(trigger, 0.0, 1.0) * self._closed
        for index, value in enumerate(self._last):
            outputs["joint_targets"][index] = float(value)
