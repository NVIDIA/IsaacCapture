# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset file path resolution for teleop_ros2_node."""

from pathlib import Path


def resolve_dex_sharpa_config(root: Path, filename: str) -> str:
    return resolve_teleop_ros2_file(
        root,
        "DexPilot Sharpa Wave retargeting config",
        "configs",
        filename,
    )


def resolve_dex_sharpa_urdf(root: Path, filename: str) -> str:
    try:
        return resolve_teleop_ros2_file(
            root,
            "Standalone Sharpa Wave URDF",
            "assets",
            "urdf",
            "sharpa_standalone",
            filename,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Populate the official Sharpa Wave URDFs with "
            "examples/teleop_ros2/scripts/fetch_sharpa_wave_urdfs.py before using "
            "hand_retargeter:=dexpilot, or use a Docker image built from "
            "examples/teleop_ros2/Dockerfile."
        ) from exc


def resolve_sharpa_mjcf(filename: str) -> str:
    try:
        from importlib.resources import files
    except ImportError as exc:  # pragma: no cover - Python 3.10+ has this.
        raise ModuleNotFoundError(
            "Sharpa hand retargeting requires importlib.resources support"
        ) from exc

    try:
        return str(
            files("robotic_grounding") / "assets" / "xmls" / "sharpawave" / filename
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Sharpa hand retargeting requires robotic_grounding assets. "
            "Install/use a build with isaaccapture[grounding] and bundled robotic_grounding."
        ) from exc


def resolve_teleop_ros2_file(root: Path, description: str, *parts: str) -> str:
    path = root.joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found at: {path}")
    return str(path)
