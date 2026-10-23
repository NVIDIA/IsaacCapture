# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Runtime capabilities shared by EgoHand and the standalone video reader.

EGOHAND_RUNTIME_PROFILE describes installed capabilities, rather than a per-video
preference. Container builds set it to release; source installs default to full.
"""

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    video_backend: str
    rendering: bool
    compilation: bool


_PROFILES = {
    "full": RuntimeProfile("full", "opencv", rendering=True, compilation=True),
    "release": RuntimeProfile("release", "ffmpeg", rendering=False, compilation=False),
}


def get_runtime_profile() -> RuntimeProfile:
    name = os.environ.get("EGOHAND_RUNTIME_PROFILE", "full")
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown EGOHAND_RUNTIME_PROFILE={name!r}; choose full or release"
        ) from exc


def configure_parser(parser: argparse.ArgumentParser, profile: RuntimeProfile) -> None:
    parser.description += f" (runtime profile: {profile.name})"
    unavailable = set()
    if not profile.compilation:
        unavailable.add("dino_compile")
    if not profile.rendering:
        unavailable.update(("save_images", "save_crop_overlays", "draw_skeleton"))
    for action in parser._actions:
        if action.dest in unavailable:
            action.help = argparse.SUPPRESS
        elif not profile.rendering and action.dest == "no_mesh":
            action.help = (
                "Accepted for compatibility; mesh rendering is already disabled."
            )
    if not profile.rendering:
        parser.set_defaults(no_save_videos=True, no_mesh=True, draw_skeleton=False)


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, profile: RuntimeProfile
) -> None:
    if not profile.compilation and args.dino_compile:
        parser.error(f"--dino_compile is unavailable in the {profile.name} runtime")
    if not profile.rendering:
        for option in ("save_images", "save_crop_overlays", "draw_skeleton"):
            if getattr(args, option):
                parser.error(f"--{option} is unavailable in the {profile.name} runtime")
