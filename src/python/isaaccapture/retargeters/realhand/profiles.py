# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kinematic and command profiles for RealHand simulation models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RealHandJointSpec:
    """Description of one actively controlled hand joint."""

    short_name: str
    lower: float
    upper: float
    trigger_closed: float
    semantic: str
    ffg_motor_index: int
    ffg_command_at_lower: float = 255.0
    ffg_command_at_upper: float = 0.0


@dataclass(frozen=True)
class RealHandProfile:
    """Model-specific joints while keeping retargeting code model agnostic."""

    model: str
    display_name: str
    asset_dir_name: str
    urdf_name: str
    left_joints: tuple[RealHandJointSpec, ...]
    right_joints: tuple[RealHandJointSpec, ...]
    action_joint_names: tuple[str, ...]
    mimic_joint_names: tuple[str, ...] = ()

    def joints(self, side: str) -> tuple[RealHandJointSpec, ...]:
        if side == "left":
            return self.left_joints
        if side == "right":
            return self.right_joints
        raise ValueError(f"side must be left or right, got {side!r}")

    def joint_names(self, side: str) -> list[str]:
        return [f"{side}_hand_{joint.short_name}" for joint in self.joints(side)]

    def joint_spec(self, side: str, joint_name: str) -> RealHandJointSpec:
        short_name = joint_name.removeprefix(f"{side}_hand_")
        for joint in self.joints(side):
            if joint.short_name == short_name:
                return joint
        raise KeyError(
            f"{joint_name!r} is not an active {self.display_name} {side}-hand joint"
        )

    def resolve_urdf(self, asset_root: str | Path) -> Path:
        """Resolve this profile's downloaded bimanual URDF."""
        path = Path(asset_root).expanduser() / self.asset_dir_name / self.urdf_name
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.display_name} URDF not found at {path}. Run "
                "examples/teleop/python/scripts/fetch_realhand_assets.py first."
            )
        return path.resolve()

    @property
    def motor_count(self) -> int:
        return 1 + max(
            joint.ffg_motor_index for joint in self.left_joints + self.right_joints
        )


def _joint(
    name: str,
    upper: float,
    semantic: str,
    motor: int,
    *,
    lower: float = 0.0,
    closed: float | None = None,
    command_lower: float = 255.0,
    command_upper: float = 0.0,
) -> RealHandJointSpec:
    return RealHandJointSpec(
        short_name=name,
        lower=lower,
        upper=upper,
        trigger_closed=upper if closed is None else closed,
        semantic=semantic,
        ffg_motor_index=motor,
        ffg_command_at_lower=command_lower,
        ffg_command_at_upper=command_upper,
    )


_L6_LEFT = (
    _joint("index_mcp_pitch", 1.26, "index_curl", 2),
    _joint("index_dip", 1.14, "index_curl", 2),
    _joint("middle_mcp_pitch", 1.26, "middle_curl", 3),
    _joint("middle_dip", 1.14, "middle_curl", 3),
    _joint("pinky_mcp_pitch", 1.26, "pinky_curl", 5),
    _joint("pinky_dip", 1.14, "pinky_curl", 5),
    _joint("ring_mcp_pitch", 1.26, "ring_curl", 4),
    _joint("ring_dip", 1.14, "ring_curl", 4),
    _joint("thunb_cmc_roll", 1.39, "thumb_opposition", 1),
    _joint("thumb_cmc_pitch", 0.99, "thumb_flex", 0),
    _joint("thumb_dip", 1.22, "thumb_flex", 0),
)

_L6_RIGHT = (
    _joint("index_mcp_pitch", 1.26, "index_curl", 2),
    _joint("index_dip", 1.14, "index_curl", 2),
    _joint("middle_mcp_pitch", 1.26, "middle_curl", 3),
    _joint("middle_dip", 1.14, "middle_curl", 3),
    _joint("pinky_mcp_pitch", 1.26, "pinky_curl", 5),
    _joint("pinky_dip", 1.14, "pinky_curl", 5),
    _joint("ring_mcp_pitch", 1.26, "ring_curl", 4),
    _joint("ring_dip", 1.14, "ring_curl", 4),
    _joint("thunb_cmc_roll", 1.39, "thumb_opposition", 1),
    _joint("thumb_cmc_pitch", 0.99, "thumb_flex", 0),
    _joint("thumb_ip", 1.22, "thumb_flex", 0),
)

_L6_ACTION_JOINTS = (
    "left_hand_thunb_cmc_roll",
    "left_hand_index_mcp_pitch",
    "left_hand_middle_mcp_pitch",
    "left_hand_ring_mcp_pitch",
    "left_hand_pinky_mcp_pitch",
    "right_hand_thunb_cmc_roll",
    "right_hand_index_mcp_pitch",
    "right_hand_middle_mcp_pitch",
    "right_hand_ring_mcp_pitch",
    "right_hand_pinky_mcp_pitch",
    "left_hand_thumb_cmc_pitch",
    "left_hand_index_dip",
    "left_hand_middle_dip",
    "left_hand_ring_dip",
    "left_hand_pinky_dip",
    "right_hand_thumb_cmc_pitch",
    "right_hand_index_dip",
    "right_hand_middle_dip",
    "right_hand_ring_dip",
    "right_hand_pinky_dip",
    "left_hand_thumb_dip",
    "right_hand_thumb_ip",
)


def _o6_joints(side: str) -> tuple[RealHandJointSpec, ...]:
    thumb_yaw_upper = 1.30 if side == "left" else 1.36
    return (
        _joint("thumb_cmc_yaw", thumb_yaw_upper, "thumb_opposition", 1),
        _joint("thumb_cmc_pitch", 0.58, "thumb_flex", 0),
        _joint("index_mcp_pitch", 1.60, "index_curl", 2),
        _joint("middle_mcp_pitch", 1.60, "middle_curl", 3),
        _joint("ring_mcp_pitch", 1.60, "ring_curl", 4),
        _joint("pinky_mcp_pitch", 1.60, "pinky_curl", 5),
    )


def _l20_joints(side: str) -> tuple[RealHandJointSpec, ...]:
    thumb_roll_upper = 1.40 if side == "left" else 1.39
    thumb_pitch_upper = 0.84 if side == "left" else 0.83
    thumb_mcp_upper = 1.26 if side == "left" else 1.25
    pip_upper = 1.74 if side == "left" else 1.75
    joints = [
        _joint("thumb_cmc_roll", thumb_roll_upper, "thumb_rotation", 5, closed=0.50),
        _joint("thumb_cmc_yaw", 1.57, "thumb_opposition", 10, closed=1.50),
        _joint("thumb_cmc_pitch", thumb_pitch_upper, "thumb_base_flex", 0, closed=0.60),
        _joint("thumb_mcp", thumb_mcp_upper, "thumb_distal_flex", 15, closed=1.20),
    ]
    for finger, root_motor, spread_motor, tip_motor in (
        ("index", 1, 6, 16),
        ("middle", 2, 7, 17),
        ("ring", 3, 8, 18),
        ("pinky", 4, 9, 19),
    ):
        joints.extend(
            (
                _joint(
                    f"{finger}_mcp_roll",
                    0.23,
                    f"{finger}_spread",
                    spread_motor,
                    lower=-0.23,
                    closed=0.0,
                    command_lower=0.0,
                    command_upper=255.0,
                ),
                _joint(
                    f"{finger}_mcp_pitch",
                    1.22,
                    f"{finger}_base_flex",
                    root_motor,
                    closed=1.20,
                ),
                _joint(
                    f"{finger}_pip",
                    pip_upper,
                    f"{finger}_distal_flex",
                    tip_motor,
                    closed=1.70,
                ),
            )
        )
    return tuple(joints)


def _o6_action_joints() -> tuple[str, ...]:
    first_level = (
        "thumb_cmc_yaw",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    )
    return tuple(
        [f"left_hand_{name}" for name in first_level]
        + [f"right_hand_{name}" for name in first_level]
        + ["left_hand_thumb_cmc_pitch", "right_hand_thumb_cmc_pitch"]
    )


def _l20_action_joints() -> tuple[str, ...]:
    levels = (
        (
            "thumb_cmc_roll",
            "index_mcp_roll",
            "middle_mcp_roll",
            "ring_mcp_roll",
            "pinky_mcp_roll",
        ),
        (
            "thumb_cmc_yaw",
            "index_mcp_pitch",
            "middle_mcp_pitch",
            "ring_mcp_pitch",
            "pinky_mcp_pitch",
        ),
        ("thumb_cmc_pitch", "index_pip", "middle_pip", "ring_pip", "pinky_pip"),
        ("thumb_mcp",),
    )
    return tuple(
        f"{side}_hand_{name}"
        for level in levels
        for side in ("left", "right")
        for name in level
    )


def _mimic_joint_names() -> tuple[str, ...]:
    return tuple(
        f"{side}_hand_{name}"
        for side in ("left", "right")
        for name in ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip")
    )


L6_PROFILE = RealHandProfile(
    model="l6",
    display_name="L6",
    asset_dir_name="p7_l6",
    urdf_name="P7_l6_bimanual.urdf",
    left_joints=_L6_LEFT,
    right_joints=_L6_RIGHT,
    action_joint_names=_L6_ACTION_JOINTS,
)

O6_PROFILE = RealHandProfile(
    model="o6",
    display_name="O6",
    asset_dir_name="p7_o6",
    urdf_name="P7_o6_bimanual.urdf",
    left_joints=_o6_joints("left"),
    right_joints=_o6_joints("right"),
    action_joint_names=_o6_action_joints(),
    mimic_joint_names=_mimic_joint_names(),
)

L20_PROFILE = RealHandProfile(
    model="l20",
    display_name="L20",
    asset_dir_name="p7_l20",
    urdf_name="P7_L20_bimanual.urdf",
    left_joints=_l20_joints("left"),
    right_joints=_l20_joints("right"),
    action_joint_names=_l20_action_joints(),
    mimic_joint_names=_mimic_joint_names(),
)

REALHAND_PROFILES = {
    L6_PROFILE.model: L6_PROFILE,
    O6_PROFILE.model: O6_PROFILE,
    L20_PROFILE.model: L20_PROFILE,
}


def get_realhand_profile(model: str) -> RealHandProfile:
    key = model.strip().lower()
    try:
        return REALHAND_PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(REALHAND_PROFILES))
        raise ValueError(
            f"unsupported RealHand model {model!r}; expected one of: {supported}"
        ) from exc


__all__ = [
    "L6_PROFILE",
    "L20_PROFILE",
    "O6_PROFILE",
    "REALHAND_PROFILES",
    "RealHandJointSpec",
    "RealHandProfile",
    "get_realhand_profile",
]
