# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URDF placement and Sharpa raw-joint mapping configuration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import normalize


@dataclass(frozen=True)
class UrdfConfig:
    scale: float = 1.0
    offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    joint_map: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("URDF scale must be finite and positive")
        if len(self.offset_cm) != 3 or not np.isfinite(self.offset_cm).all():
            raise ValueError("URDF offset_cm must contain three finite numbers")
        if len(self.rotation_xyzw) != 4:
            raise ValueError("URDF rotation_xyzw must contain four numbers")
        normalize(np.asarray(self.rotation_xyzw), label="URDF rotation_xyzw")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.joint_map.items()
        ):
            raise ValueError("URDF joint_map must map URDF names to raw joint names")


def load_urdf_config(path: Path | None) -> UrdfConfig:
    if path is None:
        config = UrdfConfig()
        config.validate()
        return config
    if not path.is_file():
        raise ValueError(f"URDF config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("URDF config root must be an object")
    unknown = set(raw) - {"scale", "offset_cm", "rotation_xyzw", "joint_map"}
    if unknown:
        raise ValueError(f"unknown URDF config keys: {sorted(unknown)}")
    config = UrdfConfig(
        scale=float(raw.get("scale", 1.0)),
        offset_cm=tuple(float(v) for v in raw.get("offset_cm", (0, 0, 0))),
        rotation_xyzw=tuple(float(v) for v in raw.get("rotation_xyzw", (0, 0, 0, 1))),
        joint_map=dict(raw.get("joint_map", {})),
    )
    config.validate()
    return config


def map_raw_joints(
    urdf_joint_names: list[str],
    raw_values: dict[str, float] | None,
    joint_map: dict[str, str],
) -> np.ndarray:
    raw = raw_values or {}
    return np.asarray(
        [raw.get(joint_map.get(name, name), 0.0) for name in urdf_joint_names],
        dtype=np.float64,
    )
