# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RealHand robot data for the official DexHandRetargeter."""

from pathlib import Path

from isaaccapture.retargeters import DexHandRetargeterConfig

from ._profiles import get_realhand_profile

HAND_FRAME = (0, -1, 0, -1, 0, 0, 0, 0, -1)


def get_hand_config(model: str, side: str) -> DexHandRetargeterConfig:
    profile = get_realhand_profile(model)
    names = profile.joint_names(side)
    directory = Path(__file__).parent / "config" / "realhand" / profile.model
    return DexHandRetargeterConfig(
        hand_retargeting_config=str(directory / f"{side}.yml"),
        hand_urdf=str(directory / f"{side}.urdf"),
        hand_joint_names=names,
        hand_side=side,
        handtracking_to_baselink_frame_transform=HAND_FRAME,
    )
