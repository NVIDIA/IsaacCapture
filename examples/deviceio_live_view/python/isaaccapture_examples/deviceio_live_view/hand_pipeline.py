# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-schema selection contracts for the DeviceIO live viewer."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SomaHandRepresentation,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    OutputSelector,
)


class HandSchema(str, Enum):
    OPENXR_HAND_POSE = "openxr-hand-pose"
    SOMA = "soma"


@dataclass(frozen=True)
class HandViewLayout:
    joint_names: tuple[str, ...]
    bones: tuple[tuple[int, int], ...]
    positions_index: int
    valid_index: int


@dataclass(frozen=True)
class HandViewPipeline:
    left: OutputSelector
    right: OutputSelector
    layout: HandViewLayout


def create_hand_view_pipeline(
    soma_data_root: Path | None = None,
    *,
    hand_schema: HandSchema | str = HandSchema.OPENXR_HAND_POSE,
    soma_left_collection_id: str = "soma_hand_left_demo",
    soma_right_collection_id: str = "soma_hand_right_demo",
    soma_hand_representation: SomaHandRepresentation
    | str = SomaHandRepresentation.JOINT_ROTATIONS,
) -> HandViewPipeline:
    selected = HandSchema(hand_schema)
    if selected is HandSchema.OPENXR_HAND_POSE:
        from .openxr_hand_pose import create_openxr_hand_pipeline

        return create_openxr_hand_pipeline()

    representation = SomaHandRepresentation(soma_hand_representation)
    if (
        representation is SomaHandRepresentation.JOINT_ROTATIONS
        and soma_data_root is None
    ):
        raise ValueError(
            "--hand-schema soma with joint-rotations requires --soma-data-root"
        )
    from .soma_hand import create_soma_hand_pipeline

    return create_soma_hand_pipeline(
        soma_data_root,
        soma_left_collection_id,
        soma_right_collection_id,
        representation,
    )
