# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA body source and layout for the DeviceIO live viewer."""

from pathlib import Path

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SomaBodyRepresentation,
    SomaBodySource,
)
from isaaccapture.retargeting_engine.tensor_types import SomaBodyInputIndex
from isaaccapture.schema import SomaBodyJointV0

from .body_pipeline import BodyViewLayout, BodyViewPipeline


def create_layer(data_root: Path):
    import soma
    import torch

    if soma.__version__ != "0.3.1":
        raise RuntimeError(f"The POC requires SOMA-X 0.3.1, got {soma.__version__}")
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
            SomaBodyJointV0.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    ]
    if [name.upper() for name in layer.public_joint_names] != ["ROOT", *expected]:
        raise ValueError("SOMA layer joint order differs from the V0 FBS")
    with torch.no_grad():
        layer.prepare_identity(
            torch.zeros((1, layer.identity_model.num_identity_coeffs))
        )
    return layer


def create_soma_body_pipeline(
    data_root: Path,
    collection_id: str,
    representation: SomaBodyRepresentation
    | str = SomaBodyRepresentation.JOINT_ROTATIONS,
) -> BodyViewPipeline:
    source = SomaBodySource(
        "body",
        collection_id,
        create_layer(data_root),
        representation=representation,
    )
    return BodyViewPipeline(
        output=source.output(SomaBodySource.BODY),
        layout=BodyViewLayout(
            joint_names=source.joint_names,
            bones=source.bones,
            positions_index=int(SomaBodyInputIndex.JOINT_POSITIONS),
            valid_index=int(SomaBodyInputIndex.JOINT_VALID),
        ),
    )
