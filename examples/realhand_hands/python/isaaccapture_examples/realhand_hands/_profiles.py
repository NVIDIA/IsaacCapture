# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read active joint names and limits from the packaged hand models."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

_CONFIG = Path(__file__).parent / "config" / "realhand"


@dataclass(frozen=True)
class RealHandJointSpec:
    short_name: str
    lower: float
    upper: float
    trigger_closed: float


@dataclass(frozen=True)
class RealHandProfile:
    model: str

    def joints(self, side: str) -> tuple[RealHandJointSpec, ...]:
        return _model_joints(self.model, side)

    def joint_names(self, side: str) -> list[str]:
        return [f"{side}_hand_{joint.short_name}" for joint in self.joints(side)]


@lru_cache(maxsize=6)
def _model_joints(model: str, side: str) -> tuple[RealHandJointSpec, ...]:
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {side!r}")
    directory = _CONFIG / model
    names = yaml.safe_load((directory / f"{side}.yml").read_text())["retargeting"][
        "target_joint_names"
    ]
    joints = {
        joint.get("name"): joint
        for joint in ET.parse(directory / f"{side}.urdf").getroot().findall("joint")
    }
    result = []
    for name in names:
        short = name.removeprefix(f"{side}_hand_")
        limits = joints[name].find("limit")
        lower, upper = float(limits.get("lower")), float(limits.get("upper"))
        closed = upper
        if model == "l20":
            if short.endswith("mcp_roll"):
                closed = 0.0
            elif short == "thumb_cmc_roll":
                closed = 0.50
            elif short == "thumb_cmc_yaw":
                closed = 1.50
            elif short == "thumb_cmc_pitch":
                closed = 0.60
            elif short.endswith(("mcp", "mcp_pitch")):
                closed = 1.20
            elif short.endswith("pip"):
                closed = 1.70
        result.append(
            RealHandJointSpec(short, lower, upper, max(lower, min(closed, upper)))
        )
    return tuple(result)


def get_realhand_profile(model: str) -> RealHandProfile:
    model = model.strip().lower()
    if model not in ("l6", "o6", "l20"):
        raise ValueError(
            f"unsupported RealHand model {model!r}; expected l6, o6, or l20"
        )
    return RealHandProfile(model)
