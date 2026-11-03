# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retargeter adapter that relabels joint output names without changing values."""

from typing import Sequence

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.tensor_types import FloatType


class JointNameAliasRetargeter(BaseRetargeter):
    """Copy hand joint values while exposing output tensors with alias names."""

    def __init__(
        self,
        input_joint_names: Sequence[str],
        output_joint_names: Sequence[str],
        name: str,
    ) -> None:
        self._input_joint_names = list(input_joint_names)
        self._output_joint_names = list(output_joint_names)
        if len(self._input_joint_names) != len(self._output_joint_names):
            raise ValueError(
                "input_joint_names and output_joint_names must have the same "
                f"length, got {len(self._input_joint_names)} and "
                f"{len(self._output_joint_names)}"
            )

        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                "hand_joints_input",
                [FloatType(name) for name in self._input_joint_names],
            )
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                "hand_joints",
                [FloatType(name) for name in self._output_joint_names],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        source = inputs["hand_joints"]
        target = outputs["hand_joints"]
        for index in range(len(self._output_joint_names)):
            target[index] = float(source[index])
