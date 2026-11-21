# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA body source and layout for the DeviceIO live viewer."""

from pathlib import Path

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SomaBodyRepresentation,
    SomaBodySource,
)
from isaaccapture.retargeting_engine.tensor_types import SomaBodyInputIndex
from isaaccapture.schema import SomaBodyJoint

from .body_pipeline import BodyViewLayout, BodyViewPipeline


# SOMA-X v0.3.1 public hierarchy with virtual Root omitted. Hips uses -1.
SOMA_BODY_PARENT_IDS: tuple[int, ...] = (
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    3,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    14,
    19,
    20,
    21,
    22,
    14,
    24,
    25,
    26,
    27,
    14,
    29,
    30,
    31,
    32,
    14,
    34,
    35,
    36,
    37,
    3,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    42,
    47,
    48,
    49,
    50,
    42,
    52,
    53,
    54,
    55,
    42,
    57,
    58,
    59,
    60,
    42,
    62,
    63,
    64,
    65,
    0,
    67,
    68,
    69,
    70,
    0,
    72,
    73,
    74,
    75,
)

SOMA_BODY_LAYOUT = BodyViewLayout(
    joint_names=tuple(
        name
        for name, joint in sorted(
            SomaBodyJoint.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    ),
    bones=tuple(
        (parent, child)
        for child, parent in enumerate(SOMA_BODY_PARENT_IDS)
        if parent >= 0
    ),
    positions_index=int(SomaBodyInputIndex.JOINT_POSITIONS),
    valid_index=int(SomaBodyInputIndex.JOINT_VALID),
)


def create_layer(data_root: Path | None = None):
    import soma
    import torch

    # Resolve through the public helper because py-soma-x 0.3.1's body-layer
    # fallback imports a nonexistent soma.body.assets module.
    if data_root is None:
        data_root = soma.get_assets_dir()
    layer = soma.SOMALayer(
        data_root=str(data_root),
        identity_model_type="soma",
        device="cpu",
        lod="low",
        output_unit=soma.Unit.METERS,
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    expected = [
        name.replace("_", "")
        for name, joint in sorted(
            SomaBodyJoint.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    ]
    if [name.upper() for name in layer.public_joint_names] != ["ROOT", *expected]:
        raise ValueError("SOMA layer joint order differs from the FBS")
    parent_ids = tuple(
        -1 if parent == 0 else int(parent) - 1
        for parent in layer.output_joint_parent_ids[1:]
    )
    if parent_ids != SOMA_BODY_PARENT_IDS:
        raise ValueError("SOMA layer hierarchy differs from the viewer layout")
    with torch.no_grad():
        layer.prepare_identity(
            torch.zeros((1, layer.identity_model.num_identity_coeffs))
        )
    return layer


def create_soma_body_pipeline(
    collection_id: str,
    representation: SomaBodyRepresentation
    | str = SomaBodyRepresentation.JOINT_ROTATIONS,
) -> BodyViewPipeline:
    representation = SomaBodyRepresentation(representation)
    layer = (
        create_layer()
        if representation is SomaBodyRepresentation.JOINT_ROTATIONS
        else None
    )
    source = SomaBodySource(
        "body",
        collection_id,
        layer,
        representation=representation,
    )
    return BodyViewPipeline(
        output=source.output(SomaBodySource.BODY),
        layout=SOMA_BODY_LAYOUT,
    )
