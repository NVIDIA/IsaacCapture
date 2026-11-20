# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce L6, O6, or L20 finger commands from standard OpenXR input."""

import argparse
import time

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import DexHandRetargeter, JointRateLimiter
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

from ._controller_posture import ControllerPostureRetargeter
from ._profiles import get_realhand_profile
from .hand_config import get_hand_config


def build_pipeline(mode: str, model: str):
    if mode not in ("controller", "handtracking"):
        raise ValueError(f"Unknown input mode: {mode}")
    profile = get_realhand_profile(model)
    source = (
        ControllersSource(name="controllers")
        if mode == "controller"
        else HandsSource(name="hands")
    )
    outputs = {}
    for side in ("left", "right"):
        device = source.LEFT if side == "left" else source.RIGHT
        names = profile.joint_names(side)
        if mode == "controller":
            hand = ControllerPostureRetargeter(
                name=f"{model}_{side}_trigger",
                input_device=device,
                joint_names=names,
                closed_posture=[joint.trigger_closed for joint in profile.joints(side)],
            ).connect({device: source.output(device)})
            output = hand.output("joint_targets")
        else:
            hand = DexHandRetargeter(
                get_hand_config(model, side), name=f"{model}_{side}_hand"
            ).connect({device: source.output(device)})
            output = hand.output("hand_joints")
        limited = JointRateLimiter(
            name=f"{model}_{side}_rate", joint_names=names
        ).connect({"joint_targets": output})
        outputs[f"{side}_hand"] = limited.output("joint_targets")
    return OutputCombiner(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-model", choices=("l6", "o6", "l20"), default="l6")
    parser.add_argument(
        "--input-mode", choices=("controller", "handtracking"), default="handtracking"
    )
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="Build the graph without OpenXR"
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()
    pipeline = build_pipeline(args.input_mode, args.hand_model)
    profile = get_realhand_profile(args.hand_model)
    print({side: profile.joint_names(side) for side in ("left", "right")})
    if args.dry_run:
        return
    config = TeleopSessionConfig(
        app_name="RealHandHands", trackers=[], pipeline=pipeline
    )
    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            result = session.step()
            if session.frame_count % 60 == 0:
                print(
                    {
                        side: np.round(
                            [float(value) for value in result[f"{side}_hand"]], 3
                        ).tolist()
                        for side in ("left", "right")
                    }
                )
            time.sleep(0.016)


if __name__ == "__main__":
    main()
