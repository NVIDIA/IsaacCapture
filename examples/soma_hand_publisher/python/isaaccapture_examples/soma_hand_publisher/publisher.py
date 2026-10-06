# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish both hands from the bundled SOMA motion through SOMA hand transport."""

import argparse
import logging
import math
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from isaaccapture.schema import (
    Point,
    SomaHandednessV0,
    SomaHandJointPoseArrayV0,
    SomaHandJointPosesV0,
    SomaHandJointRotationArrayV0,
    SomaHandJointRotationsV0,
)

logger = logging.getLogger("isaaccapture.examples.soma_hand_publisher")


def create_layers(data_root: Path):
    """Create the neutral full-body and hand layers used by the bundled motion."""
    import soma
    import torch

    if soma.__version__ != "0.3.1":
        raise RuntimeError(f"The POC requires SOMA-X 0.3.1, got {soma.__version__}")
    body = soma.SOMALayer(
        data_root=str(data_root),
        identity_model_type="soma",
        device="cpu",
        lod="low",
        output_unit=soma.Unit.METERS,
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    hands = {
        side: soma.SOMAHandLayer(
            data_root=str(data_root),
            hand_type=side,
            device="cpu",
            identity_model_type="soma",
            lod="low",
            output_unit=soma.Unit.METERS,
            correctives_model_path=None,
        )
        for side in ("left", "right")
    }
    with torch.no_grad():
        body.prepare_identity(torch.zeros((1, body.identity_model.num_identity_coeffs)))
        for hand in hands.values():
            hand.prepare_identity(
                torch.zeros((1, hand.identity_model.num_identity_coeffs))
            )
    return body, hands


def demo_hand_frames(path: Path, body, hands):
    """Extract both hand subsets and express them through SOMAHandLayer."""
    import torch
    from soma.geometry.rig_utils import (
        joint_local_to_world,
        joint_world_to_local,
        precompute_joint_orient,
        remove_joint_orient_local,
    )
    from soma.geometry.transforms import matrix_to_quaternion_xyzw

    motion = np.load(path, allow_pickle=False)
    if (
        motion.ndim != 4
        or motion.shape[1] not in (78, 94)
        or motion.shape[2:] != (4, 4)
        or not len(motion)
    ):
        raise ValueError("Expected bundled SOMA motion shaped (T, 78 or 94, 4, 4)")
    local = torch.as_tensor(motion[..., :3, :3], dtype=torch.float32)
    if local.shape[1] == 94:
        twists = (*range(40, 44), *range(72, 76), *range(81, 85), *range(90, 94))
        local = local[:, [index for index in range(94) if index not in twists]]
    orient = body.t_pose_world[body.public_transform_joint_indices, :3, :3]
    world = joint_local_to_world(local, body.public_joint_parent_ids)
    relative = joint_world_to_local(
        world @ orient.transpose(-2, -1), body.public_joint_parent_ids
    )[:, 1:]
    with torch.no_grad():
        body_output = body.pose(
            relative,
            transl=torch.as_tensor(motion[:, 1, :3, 3], dtype=torch.float32),
            pose2rot=False,
            fk_only=True,
            apply_correctives=False,
        )

    frames = {}
    for side, start in (("left", 14), ("right", 42)):
        hand = hands[side]
        target = body_output["transforms"][:, start + 1 : start + 26]
        absolute_local = joint_world_to_local(
            target[..., :3, :3], hand.joint_parent_ids
        )
        hand_orient, hand_parent_orient_t = precompute_joint_orient(
            hand.t_pose_world, hand.joint_parent_ids
        )
        controls = remove_joint_orient_local(
            absolute_local, hand_orient, hand_parent_orient_t
        )
        translation = target[:, 0, :3, 3]
        with torch.no_grad():
            evaluated = hand.pose(
                controls,
                global_translation=translation,
                pose2rot=False,
                fk_only=True,
                apply_correctives=False,
            )
        frames[side] = {
            "rotations": matrix_to_quaternion_xyzw(controls).numpy(),
            "translation": translation.numpy(),
            "positions": evaluated["joints"].numpy(),
            "orientations": matrix_to_quaternion_xyzw(
                evaluated["transforms"][..., :3, :3]
            ).numpy(),
        }
    return frames


def soma_joint_rotations(rotations, translation, handedness):
    joints = SomaHandJointRotationArrayV0()
    joints.rotations[:] = rotations
    joints.is_valid[:] = 1
    return SomaHandJointRotationsV0(joints, Point(*translation), True, handedness)


def soma_joint_poses(positions, orientations, handedness):
    joints = SomaHandJointPoseArrayV0()
    joints.positions[:] = positions
    joints.orientations[:] = orientations
    joints.is_valid[:] = 1
    return SomaHandJointPosesV0(joints, handedness)


def _payload(frame, index: int, representation: str, handedness):
    if representation == "joint-poses":
        return soma_joint_poses(
            frame["positions"][index], frame["orientations"][index], handedness
        ).to_bytes()
    return soma_joint_rotations(
        frame["rotations"][index], frame["translation"][index], handedness
    ).to_bytes()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--pusher", required=True, type=Path, help="Built soma_hand_pusher executable"
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--hand-representation",
        choices=("joint-rotations", "joint-poses"),
        default="joint-rotations",
        help="SOMA FlatBuffer profile to publish (default: joint-rotations)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate FBS packets without OpenXR",
    )
    args = parser.parse_args(argv[1:])
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("rate must be finite and positive")

    body, hand_layers = create_layers(args.data_root)
    frames = demo_hand_frames(
        args.data_root / "example_animation.npy", body, hand_layers
    )
    command = [
        str(args.pusher.resolve()),
        "--hand-representation",
        args.hand_representation,
    ] + (["--validate-only"] if args.validate_only else [])
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        interrupted = False
        try:
            start = time.monotonic()
            frame_number = 0
            while True:
                for index in range(len(frames["left"]["positions"])):
                    remaining = start + frame_number / args.rate - time.monotonic()
                    if remaining > 0 and not args.validate_only:
                        time.sleep(remaining)
                    left = _payload(
                        frames["left"],
                        index,
                        args.hand_representation,
                        SomaHandednessV0.LEFT,
                    )
                    right = _payload(
                        frames["right"],
                        index,
                        args.hand_representation,
                        SomaHandednessV0.RIGHT,
                    )
                    process.stdin.write(
                        struct.pack(
                            "<IIQ",
                            len(left),
                            len(right),
                            round(frame_number * 1e9 / args.rate),
                        )
                    )
                    process.stdin.write(left)
                    process.stdin.write(right)
                    process.stdin.flush()
                    frame_number += 1
                if not args.loop or args.validate_only:
                    break
        except KeyboardInterrupt:
            interrupted = True
        finally:
            process.stdin.close()
        exit_code = process.wait()
        if exit_code != 0 and not (interrupted and exit_code == -signal.SIGINT):
            raise RuntimeError("SOMA hand pusher failed; see its runtime diagnostics")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
