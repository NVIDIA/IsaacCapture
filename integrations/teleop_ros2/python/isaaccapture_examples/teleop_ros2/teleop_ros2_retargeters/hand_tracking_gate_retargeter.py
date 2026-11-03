# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retargeter adapter that withholds hand joints while tracking is invalid."""

from typing import Sequence

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    OptionalType,
    RetargeterIOType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.tensor_types import FloatType


class HandTrackingGateRetargeter(BaseRetargeter):
    """Pass hand joints through while tracked, emit an absent group otherwise.

    Retargeters that can tell tracked from untracked report it separately,
    because an all-zero pose is also a legitimate flat-open hand and the joint
    values alone cannot distinguish the two. Downstream consumers already treat
    an absent group as "no data for this side", so converting the flag into
    absence here keeps that knowledge out of the publisher: the message builder
    omits the side, and consumers holding their last command hold through a
    dropout instead of snapping the hand open.

    Insert only for retargeters whose output spec includes ``hand_valid``.
    """

    def __init__(self, joint_names: Sequence[str], name: str) -> None:
        self._joint_names = list(joint_names)
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                "hand_joints_input",
                [FloatType(name) for name in self._joint_names],
            ),
            "hand_valid": TensorGroupType("hand_valid_input", [FloatType("valid")]),
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": OptionalType(
                TensorGroupType(
                    "hand_joints",
                    [FloatType(name) for name in self._joint_names],
                )
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        target = outputs["hand_joints"]
        valid = inputs["hand_valid"]
        if valid.is_none or float(valid[0]) <= 0.5:
            target.set_none()
            return
        source = inputs["hand_joints"]
        # Writing to an absent group marks it present again, so a resumed frame
        # needs no explicit clear.
        for index in range(len(self._joint_names)):
            target[index] = float(source[index])
