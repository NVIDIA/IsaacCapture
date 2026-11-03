# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-trigger commands for RealHand simulation models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from isaaccapture.retargeting_engine.interface import BaseRetargeter
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    RetargeterIO,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    FloatType,
)

from .profiles import get_realhand_profile

_LOGGER = logging.getLogger(__name__)


@dataclass
class ControllerTriggerRealHandRetargeterConfig:
    input_device: str
    joint_names: list[str]
    side: str
    hand_model: str = "l6"
    close_at_trigger_one: bool = True
    smoothing_alpha: float = 0.8
    trigger_source: str = "max"
    max_close_step: float = 1.0
    max_open_step: float = 1.0


class ControllerTriggerRealHandRetargeter(BaseRetargeter):
    """Map one controller's trigger to its matching RealHand posture."""

    def __init__(
        self, config: ControllerTriggerRealHandRetargeterConfig, name: str
    ) -> None:
        if config.side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {config.side!r}")
        if config.trigger_source not in ("trigger", "squeeze", "max"):
            raise ValueError(
                f"trigger_source must be trigger, squeeze, or max, got {config.trigger_source!r}"
            )
        if config.max_close_step <= 0.0 or config.max_open_step <= 0.0:
            raise ValueError("max_close_step and max_open_step must both be positive")
        self._config = config
        self._profile = get_realhand_profile(config.hand_model)
        self._specs = [
            self._profile.joint_spec(config.side, name) for name in config.joint_names
        ]
        self._last_output = np.zeros(len(config.joint_names), dtype=np.float64)
        self._last_close_amount = 0.0
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {self._config.input_device: OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                f"{self._profile.model}_{self._config.side}_trigger_joints",
                [FloatType(name) for name in self._config.joint_names],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        output = outputs["hand_joints"]
        controller_group = inputs[self._config.input_device]
        if controller_group.is_none or not bool(
            controller_group[ControllerInputIndex.GRIP_IS_VALID]
        ):
            self._last_output[:] = 0.0
            self._last_close_amount = 0.0
            self._write_output(output)
            return

        trigger_value = float(
            np.clip(controller_group[ControllerInputIndex.TRIGGER_VALUE], 0.0, 1.0)
        )
        squeeze_value = float(
            np.clip(controller_group[ControllerInputIndex.SQUEEZE_VALUE], 0.0, 1.0)
        )
        if self._config.trigger_source == "trigger":
            close_input = trigger_value
        elif self._config.trigger_source == "squeeze":
            close_input = squeeze_value
        else:
            close_input = max(trigger_value, squeeze_value)
        requested = (
            close_input if self._config.close_at_trigger_one else 1.0 - close_input
        )

        if context.execution_events.reset:
            self._last_output[:] = 0.0
            self._last_close_amount = 0.0
        delta = requested - self._last_close_amount
        max_step = (
            self._config.max_close_step if delta >= 0.0 else self._config.max_open_step
        )
        self._last_close_amount = float(
            np.clip(
                self._last_close_amount + np.clip(delta, -max_step, max_step), 0.0, 1.0
            )
        )
        target = np.array(
            [self._last_close_amount * spec.trigger_closed for spec in self._specs],
            dtype=np.float64,
        )
        alpha = float(np.clip(self._config.smoothing_alpha, 0.0, 1.0))
        self._last_output = alpha * target + (1.0 - alpha) * self._last_output
        self._write_output(output)

        _LOGGER.debug(
            "%s controller side=%s trigger=%.3f squeeze=%.3f close=%.3f",
            self._profile.display_name,
            self._config.side,
            trigger_value,
            squeeze_value,
            self._last_close_amount,
        )

    def _write_output(self, output) -> None:
        for index, value in enumerate(self._last_output):
            output[index] = float(value)


__all__ = [
    "ControllerTriggerRealHandRetargeter",
    "ControllerTriggerRealHandRetargeterConfig",
]
