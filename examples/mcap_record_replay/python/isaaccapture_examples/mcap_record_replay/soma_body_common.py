# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA body pipeline and native skeleton rendering for this example."""

from isaaccapture.deviceio_trackers import (
    SomaBodyJointPosesTracker,
    SomaBodyJointRotationsTracker,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SomaBodyRepresentation,
    SomaBodySource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.tensor_types import SomaBodyInputIndex
from isaaccapture.schema import SomaBodyJoint

from .common import TRACKED_COLOR, JointInputViz, channel_info

DEFAULT_COLLECTION = "soma_body_demo"
JOINT_NAMES = tuple(
    name
    for name, joint in sorted(
        SomaBodyJoint.__members__.items(), key=lambda item: int(item[1])
    )
    if name != "NUM_JOINTS"
)

# Matches the DeviceIO viewer's SOMA-X v0.3.1 hierarchy, with virtual Root omitted.
# Each example is independently installable, so keep the layout local.
PARENT_IDS = (
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
BONES = tuple((parent, child) for child, parent in enumerate(PARENT_IDS) if parent >= 0)


def add_body_arguments(parser):
    parser.add_argument(
        "--soma-body-representation",
        choices=[representation.value for representation in SomaBodyRepresentation],
        default=SomaBodyRepresentation.JOINT_ROTATIONS.value,
        help="SOMA wire representation (default: joint-rotations)",
    )
    parser.add_argument(
        "--soma-body-collection-id",
        default=DEFAULT_COLLECTION,
        help="Publisher collection and MCAP channel base (default: soma_body_demo)",
    )


def create_tracker(collection_id, representation):
    representation = SomaBodyRepresentation(representation)
    if not collection_id:
        raise ValueError("collection-id must not be empty")
    tracker_type = (
        SomaBodyJointRotationsTracker
        if representation is SomaBodyRepresentation.JOINT_ROTATIONS
        else SomaBodyJointPosesTracker
    )
    return tracker_type(collection_id)


def create_layer():
    import soma
    import torch

    # py-soma-x 0.3.1's body-layer fallback imports nonexistent soma.body.assets.
    layer = soma.SOMALayer(
        data_root=str(soma.get_assets_dir()),
        identity_model_type="soma",
        device="cpu",
        lod="low",
        output_unit=soma.Unit.METERS,
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    parent_ids = tuple(
        -1 if parent == 0 else int(parent) - 1
        for parent in layer.output_joint_parent_ids[1:]
    )
    if parent_ids != PARENT_IDS:
        raise ValueError("SOMA layer hierarchy differs from the viewer layout")
    with torch.no_grad():
        layer.prepare_identity(
            torch.zeros((1, layer.identity_model.num_identity_coeffs))
        )
    return layer


def build_pipeline(collection_id, representation):
    representation = SomaBodyRepresentation(representation)
    layer = (
        create_layer()
        if representation is SomaBodyRepresentation.JOINT_ROTATIONS
        else None
    )
    source = SomaBodySource(
        collection_id, collection_id, layer, representation=representation
    )
    return OutputCombiner({SomaBodySource.BODY: source.output(SomaBodySource.BODY)})


def replay_info(path, collection_id, representation):
    """Count the selected tracked channel and estimate its average capture rate."""
    suffix = "soma_body_" + representation.replace("-", "_")
    topic = f"{collection_id}/{suffix}_tracked"
    record_name = (
        "core.SomaBodyJointRotationsRecord"
        if representation == "joint-rotations"
        else "core.SomaBodyJointPosesRecord"
    )
    count, rate = channel_info(path, topic, record_name)
    if count == 0:
        raise ValueError(
            f"{path}: no samples on {topic}; check soma-body-collection-id and soma-body-representation"
        )
    return count, rate


class SomaBodyViz(JointInputViz):
    """Render available joints in SOMA's 77-joint order, including fingers."""

    def __init__(self, server, ground=None):
        super().__init__(
            server,
            "soma_body",
            len(JOINT_NAMES),
            BONES,
            SomaBodyInputIndex.JOINT_POSITIONS,
            SomaBodyInputIndex.JOINT_VALID,
            TRACKED_COLOR,
            ground,
        )
