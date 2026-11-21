# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FullBodyPose source and skeleton layout for the DeviceIO live viewer."""

from isaaccapture.retargeting_engine.deviceio_source_nodes import FullBodySource
from isaaccapture.retargeting_engine.tensor_types import (
    BodyJointIndex,
    FullBodyInputIndex,
)

from .body_pipeline import BodyViewLayout, BodyViewPipeline


# PICO body-joint connectivity (parent → child) for skeleton rendering.
# Indices follow BodyJointIndex: 0=PELVIS, 1/2=LEFT/RIGHT_HIP, 3/6/9=SPINE1/2/3,
# 4/5=LEFT/RIGHT_KNEE, 7/8=LEFT/RIGHT_ANKLE, 10/11=LEFT/RIGHT_FOOT, 12=NECK,
# 13/14=LEFT/RIGHT_COLLAR, 15=HEAD, 16/17=LEFT/RIGHT_SHOULDER,
# 18/19=LEFT/RIGHT_ELBOW, 20/21=LEFT/RIGHT_WRIST, 22/23=LEFT/RIGHT_HAND (24 total).
FULL_BODY_POSE_BONES: tuple[tuple[int, int], ...] = (
    # Trunk and spine
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    # Left leg
    (1, 4),
    (4, 7),
    (7, 10),
    # Right leg
    (2, 5),
    (5, 8),
    (8, 11),
    # Left arm
    (12, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    # Right arm
    (12, 14),
    (14, 17),
    (17, 19),
    (19, 21),
    (21, 23),
)

FULL_BODY_POSE_LAYOUT = BodyViewLayout(
    joint_names=tuple(joint.name for joint in BodyJointIndex),
    bones=FULL_BODY_POSE_BONES,
    positions_index=int(FullBodyInputIndex.JOINT_POSITIONS),
    valid_index=int(FullBodyInputIndex.JOINT_VALID),
)


def create_full_body_pose_pipeline() -> BodyViewPipeline:
    source = FullBodySource("body")
    return BodyViewPipeline(
        output=source.output(FullBodySource.FULL_BODY),
        layout=FULL_BODY_POSE_LAYOUT,
    )
