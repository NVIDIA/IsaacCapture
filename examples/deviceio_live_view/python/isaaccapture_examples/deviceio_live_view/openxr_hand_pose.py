# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenXR HandPose source and layout for the DeviceIO live viewer."""

from isaaccapture.retargeting_engine.deviceio_source_nodes import HandsSource
from isaaccapture.retargeting_engine.tensor_types import HandInputIndex
from isaaccapture.schema import HandJoint

from .hand_pipeline import HandViewLayout, HandViewPipeline


OPENXR_HAND_BONES: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (1, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (1, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
)

OPENXR_HAND_LAYOUT = HandViewLayout(
    joint_names=tuple(
        name
        for name, joint in sorted(
            HandJoint.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    ),
    bones=OPENXR_HAND_BONES,
    positions_index=int(HandInputIndex.JOINT_POSITIONS),
    valid_index=int(HandInputIndex.JOINT_VALID),
)


def create_openxr_hand_pipeline() -> HandViewPipeline:
    source = HandsSource(name="hands")
    return HandViewPipeline(
        left=source.output(HandsSource.LEFT),
        right=source.output(HandsSource.RIGHT),
        layout=OPENXR_HAND_LAYOUT,
    )
