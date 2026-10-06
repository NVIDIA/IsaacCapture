# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Body-schema selection contracts for the DeviceIO live viewer."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    OutputSelector,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import SomaBodyRepresentation


class BodySchema(str, Enum):
    FULL_BODY_POSE = "full-body-pose"
    SOMA = "soma"


@dataclass(frozen=True)
class BodyViewLayout:
    joint_names: tuple[str, ...]
    bones: tuple[tuple[int, int], ...]
    positions_index: int
    valid_index: int


@dataclass(frozen=True)
class BodyViewPipeline:
    output: OutputSelector
    layout: BodyViewLayout


def resolve_body_schema(
    body_schema: BodySchema | str | None, soma_data_root: Path | None
) -> BodySchema:
    selected = (
        BodySchema(body_schema)
        if body_schema is not None
        else (
            BodySchema.SOMA if soma_data_root is not None else BodySchema.FULL_BODY_POSE
        )
    )
    if selected is BodySchema.SOMA and soma_data_root is None:
        raise ValueError("--body-schema soma requires --soma-data-root")
    return selected


def create_body_view_pipeline(
    soma_data_root: Path | None = None,
    *,
    body_schema: BodySchema | str | None = None,
    soma_body_collection_id: str = "soma_body_demo",
    soma_body_representation: SomaBodyRepresentation
    | str = SomaBodyRepresentation.JOINT_ROTATIONS,
) -> BodyViewPipeline:
    selected = resolve_body_schema(body_schema, soma_data_root)
    if selected is BodySchema.FULL_BODY_POSE:
        from .full_body_pose import create_full_body_pose_pipeline

        return create_full_body_pose_pipeline()

    from .soma_body import create_soma_body_pipeline

    assert soma_data_root is not None
    return create_soma_body_pipeline(
        soma_data_root, soma_body_collection_id, soma_body_representation
    )
