#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch official Sharpa Wave URDFs used by the teleop ROS 2 DexPilot configs."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


_SHARPA_URDF_REF = "3e953f588ba9954cebaa720aaa4cee06a43a068e"
_SHARPA_URDF_BASE_URL = (
    "https://raw.githubusercontent.com/"
    f"sharpa-robotics/sharpa-urdf-usd-xml/{_SHARPA_URDF_REF}/wave_01"
)
_URDF_SHA256 = {
    "left_sharpa_wave.urdf": (
        "4cf9fcf07a4545b538995c6c17aaf1a5768e61541b11b1cc385026da3626bad8"
    ),
    "right_sharpa_wave.urdf": (
        "7a9ab7f824482d23765b2da40b7e96fc605e7e70eda4615a5ca51fea88afb845"
    ),
}
_URDF_SOURCES = {
    "left": {
        "url": f"{_SHARPA_URDF_BASE_URL}/left_sharpa_wave/left_sharpa_wave.urdf",
        "filename": "left_sharpa_wave.urdf",
        "config": "sharpa_wave_left_dexpilot.yml",
    },
    "right": {
        "url": f"{_SHARPA_URDF_BASE_URL}/right_sharpa_wave/right_sharpa_wave.urdf",
        "filename": "right_sharpa_wave.urdf",
        "config": "sharpa_wave_right_dexpilot.yml",
    },
}


def _tree_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    return _tree_root() / "assets" / "urdf" / "sharpa_standalone"


def _default_config_dir() -> Path:
    return _tree_root() / "configs"


def _parse_retargeting_config(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"DexPilot config not found at: {path}")

    links: set[str] = set()
    joints: set[str] = set()
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "finger_tip_link_names:":
            section = "links"
            continue
        if line == "target_joint_names:":
            section = "joints"
            continue
        if line.startswith("wrist_link_name:"):
            links.add(line.split(":", maxsplit=1)[1].strip())
            section = None
            continue
        if line.startswith("- ") and section == "links":
            links.add(line[2:].strip())
            continue
        if line.startswith("- ") and section == "joints":
            joints.add(line[2:].strip())
            continue
        if not line.startswith("- "):
            section = None

    if not links or not joints:
        raise ValueError(
            f"DexPilot config at {path} does not contain fingertip links, "
            "a wrist link, and target joints"
        )
    return links, joints


def _parse_urdf_names(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Sharpa Wave URDF not found at: {path}")

    root = ET.parse(path).getroot()
    links = {
        link.attrib["name"] for link in root.findall("link") if "name" in link.attrib
    }
    joints = {
        joint.attrib["name"]
        for joint in root.findall("joint")
        if "name" in joint.attrib
    }
    return links, joints


def _verify_urdf(config_path: Path, urdf_path: Path) -> None:
    required_links, required_joints = _parse_retargeting_config(config_path)
    urdf_links, urdf_joints = _parse_urdf_names(urdf_path)

    missing_links = sorted(required_links - urdf_links)
    missing_joints = sorted(required_joints - urdf_joints)
    if missing_links or missing_joints:
        details = []
        if missing_links:
            details.append(f"missing links: {', '.join(missing_links)}")
        if missing_joints:
            details.append(f"missing joints: {', '.join(missing_joints)}")
        raise ValueError(
            f"{urdf_path} does not match {config_path}: {'; '.join(details)}"
        )


def _download(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url} to {destination}: {exc}") from exc

    expected_sha256 = _URDF_SHA256[destination.name]
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Downloaded {url} has SHA-256 {actual_sha256}, "
            f"expected {expected_sha256}; not writing {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download pinned official Sharpa Wave URDFs and verify them against "
            "the teleop ROS 2 DexPilot configs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory to write left_sharpa_wave.urdf and right_sharpa_wave.urdf.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_default_config_dir(),
        help="Directory containing sharpa_wave_{left,right}_dexpilot.yml.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Download URDFs without checking link and joint names against DexPilot YAMLs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    for side, source in _URDF_SOURCES.items():
        urdf_path = args.output_dir / source["filename"]
        config_path = args.config_dir / source["config"]
        _download(source["url"], urdf_path)
        if not args.skip_verification:
            _verify_urdf(config_path, urdf_path)
        print(f"Fetched {side} Sharpa Wave URDF: {urdf_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
