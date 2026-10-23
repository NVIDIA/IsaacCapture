# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone SOMA hand sources and native layout for this example."""

from isaaccapture.deviceio_trackers import (
    SomaHandJointPosesTracker,
    SomaHandJointRotationsTracker,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SomaHandRepresentation,
    SomaHandSource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.tensor_types import SomaHandInputIndex
from isaaccapture.schema import SomaHandSide, SomaHandJoint

from .common import LEFT_COLOR, RIGHT_COLOR, JointInputViz, channel_info

DEFAULT_LEFT_COLLECTION = "soma_hand_left_demo"
DEFAULT_RIGHT_COLLECTION = "soma_hand_right_demo"
HAND_SIDES = {"hand_left": SomaHandSide.LEFT, "hand_right": SomaHandSide.RIGHT}
JOINT_NAMES = tuple(
    name
    for name, joint in sorted(
        SomaHandJoint.__members__.items(), key=lambda item: int(item[1])
    )
    if name != "NUM_JOINTS"
)
# Matches the DeviceIO viewer's 25-joint Wrist/finger layout, without an OpenXR Palm.
BONES = (
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


def add_hand_arguments(parser):
    parser.add_argument(
        "--soma-hand-representation",
        choices=[profile.value for profile in SomaHandRepresentation],
        default=SomaHandRepresentation.JOINT_ROTATIONS.value,
        help="SOMA wire representation (default: joint-rotations)",
    )
    parser.add_argument(
        "--soma-left-hand-collection-id",
        default=DEFAULT_LEFT_COLLECTION,
        help="Left publisher collection and MCAP channel base",
    )
    parser.add_argument(
        "--soma-right-hand-collection-id",
        default=DEFAULT_RIGHT_COLLECTION,
        help="Right publisher collection and MCAP channel base",
    )


def collections(left, right):
    if not left or not right or left == right:
        raise ValueError("left and right collection IDs must be nonempty and distinct")
    return {"hand_left": left, "hand_right": right}


def create_trackers(collection_ids, representation):
    representation = SomaHandRepresentation(representation)
    tracker_type = (
        SomaHandJointRotationsTracker
        if representation is SomaHandRepresentation.JOINT_ROTATIONS
        else SomaHandJointPosesTracker
    )
    return {
        key: tracker_type(collection_id)
        for key, collection_id in collection_ids.items()
    }


def create_layer(hand_side):
    import soma
    import torch

    hand_type = "left" if hand_side == SomaHandSide.LEFT else "right"
    layer = soma.SOMAHandLayer(
        data_root=str(soma.get_assets_dir()),
        hand_type=hand_type,
        device="cpu",
        identity_model_type="soma",
        lod="low",
        output_unit=soma.Unit.METERS,
        correctives_model_path=None,
    )
    with torch.no_grad():
        layer.prepare_identity(
            torch.zeros((1, layer.identity_model.num_identity_coeffs))
        )
    return layer


def build_pipeline(collection_ids, representation):
    representation = SomaHandRepresentation(representation)
    outputs = {}
    for key, collection_id in collection_ids.items():
        hand_side = HAND_SIDES[key]
        layer = (
            create_layer(hand_side)
            if representation is SomaHandRepresentation.JOINT_ROTATIONS
            else None
        )
        source = SomaHandSource(
            collection_id,
            collection_id,
            hand_side,
            layer,
            representation=representation,
        )
        outputs[key] = source.output(SomaHandSource.HAND)
    return OutputCombiner(outputs)


def create_visualizers(server, ground=None):
    return {
        key: JointInputViz(
            server,
            key,
            len(JOINT_NAMES),
            BONES,
            SomaHandInputIndex.JOINT_POSITIONS,
            SomaHandInputIndex.JOINT_VALID,
            LEFT_COLOR if hand_side == SomaHandSide.LEFT else RIGHT_COLOR,
            ground,
        )
        for key, hand_side in HAND_SIDES.items()
    }


def replay_info(path, collection_ids, representation):
    suffix = "soma_hand_" + representation.replace("-", "_")
    record_name = (
        "core.SomaHandJointRotationsRecord"
        if representation == "joint-rotations"
        else "core.SomaHandJointPosesRecord"
    )
    info = {
        key: channel_info(path, f"{collection_id}/{suffix}_tracked", record_name)
        for key, collection_id in collection_ids.items()
    }
    busiest = max(info.values(), key=lambda item: item[0])
    if busiest[0] == 0:
        raise ValueError(
            f"{path}: no SOMA hand samples; check collection IDs and soma-hand-representation"
        )
    return {key: count for key, (count, _) in info.items()}, busiest[1]
