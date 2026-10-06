# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA hand sources and layout for the DeviceIO live viewer."""

from pathlib import Path

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SomaHandRepresentation,
    SomaHandSource,
)
from isaaccapture.retargeting_engine.tensor_types import SomaHandInputIndex
from isaaccapture.schema import SomaHandedness, SomaHandJoint

from .hand_pipeline import HandViewLayout, HandViewPipeline


SOMA_HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (0, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (0, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
)

SOMA_HAND_LAYOUT = HandViewLayout(
    joint_names=tuple(
        name
        for name, joint in sorted(
            SomaHandJoint.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    ),
    bones=SOMA_HAND_BONES,
    positions_index=int(SomaHandInputIndex.JOINT_POSITIONS),
    valid_index=int(SomaHandInputIndex.JOINT_VALID),
)


def create_layer(hand_type: str, data_root: Path | None = None):
    import soma
    import torch

    if data_root is None:
        data_root = soma.get_assets_dir()
    layer = soma.SOMAHandLayer(
        data_root=str(data_root),
        hand_type=hand_type,
        device="cpu",
        identity_model_type="soma",
        lod="low",
        output_unit=soma.Unit.METERS,
        correctives_model_path=None,
    )
    expected = [name.replace("_", "") for name in SOMA_HAND_LAYOUT.joint_names]
    upstream_prefix = f"{hand_type.capitalize()}Hand"
    actual = [
        name.removeprefix(upstream_prefix).upper()
        for name in layer.rig_data["joint_names"]
    ]
    actual[0] = "WRIST"
    if actual != [name.upper() for name in expected]:
        raise ValueError("SOMA hand layer joint order differs from the FBS")
    with torch.no_grad():
        layer.prepare_identity(
            torch.zeros((1, layer.identity_model.num_identity_coeffs))
        )
    return layer


def create_soma_hand_pipeline(
    left_collection_id: str,
    right_collection_id: str,
    representation: SomaHandRepresentation,
) -> HandViewPipeline:
    layers = (None, None)
    if representation is SomaHandRepresentation.JOINT_ROTATIONS:
        layers = (create_layer("left"), create_layer("right"))
    left = SomaHandSource(
        "hand_left",
        left_collection_id,
        SomaHandedness.LEFT,
        layers[0],
        representation=representation,
    )
    right = SomaHandSource(
        "hand_right",
        right_collection_id,
        SomaHandedness.RIGHT,
        layers[1],
        representation=representation,
    )
    return HandViewPipeline(
        left=left.output(SomaHandSource.HAND),
        right=right.output(SomaHandSource.HAND),
        layout=SOMA_HAND_LAYOUT,
    )
