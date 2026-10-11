# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish both hands from the bundled SOMA motion through SOMA hand transport."""

import argparse
import logging
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from isaaccapture.deviceio_session import DeviceIOSession
from isaaccapture.deviceio_trackers import TensorPushTracker
from isaaccapture.oxr import OpenXRSession
from isaaccapture.schema import (
    Point,
    Pose,
    Quaternion,
    SomaHandedness,
    SomaHandJoint,
    SomaHandJointPose,
    SomaHandJointPoses,
    SomaHandJointRotation,
    SomaHandJointRotations,
)

logger = logging.getLogger("isaaccapture.examples.soma_hand_publisher")

HAND_JOINTS = tuple(
    joint
    for name, joint in sorted(
        SomaHandJoint.__members__.items(), key=lambda item: int(item[1])
    )
    if name != "NUM_JOINTS"
)


def create_layers(data_root: Path | None = None):
    """Create the neutral full-body and hand layers used by the bundled motion."""
    import soma
    import torch

    # The v0.3.1 body-layer fallback imports a nonexistent soma.body.assets.
    if data_root is None:
        logger.info(
            "Loading SOMA assets. First use may download files and take a while."
        )
        data_root = soma.get_assets_dir()
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
    joints = [
        SomaHandJointRotation(joint, Quaternion(*rotation))
        for joint, rotation in zip(HAND_JOINTS, rotations, strict=True)
    ]
    return SomaHandJointRotations(joints, Point(*translation), True, handedness)


def soma_joint_poses(positions, orientations, handedness):
    joints = [
        SomaHandJointPose(joint, Pose(Point(*position), Quaternion(*orientation)))
        for joint, position, orientation in zip(
            HAND_JOINTS, positions, orientations, strict=True
        )
    ]
    return SomaHandJointPoses(joints, handedness)


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
        help="Encode demo frames and check transport capacity without OpenXR",
    )
    args = parser.parse_args(argv[1:])
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("rate must be finite and positive")

    body, hand_layers = create_layers()
    frames = demo_hand_frames(
        body.data_root / "example_animation.npy", body, hand_layers
    )
    joint_poses = args.hand_representation == "joint-poses"
    max_payload_size = 2048 if joint_poses else 1024
    frame_number = 0
    try:
        with ExitStack() as stack:
            session = None
            if not args.validate_only:
                tensor_identifier = (
                    "soma_hand_joint_poses"
                    if joint_poses
                    else "soma_hand_joint_rotations"
                )
                left_tracker = TensorPushTracker(
                    "soma_hand_left_demo", tensor_identifier, max_payload_size
                )
                right_tracker = TensorPushTracker(
                    "soma_hand_right_demo", tensor_identifier, max_payload_size
                )
                trackers = [left_tracker, right_tracker]
                extensions = DeviceIOSession.get_required_extensions(trackers)
                oxr_session = stack.enter_context(
                    OpenXRSession("SomaHandDemoPublisher", extensions)
                )
                session = stack.enter_context(
                    DeviceIOSession.run(trackers, oxr_session.get_handles())
                )
            start = time.monotonic()
            while True:
                for index in range(len(frames["left"]["positions"])):
                    remaining = start + frame_number / args.rate - time.monotonic()
                    if remaining > 0 and not args.validate_only:
                        time.sleep(remaining)
                    left = _payload(
                        frames["left"],
                        index,
                        args.hand_representation,
                        SomaHandedness.LEFT,
                    )
                    right = _payload(
                        frames["right"],
                        index,
                        args.hand_representation,
                        SomaHandedness.RIGHT,
                    )
                    if len(left) > max_payload_size or len(right) > max_payload_size:
                        raise ValueError("SOMA hand payload exceeds transport capacity")
                    if session is not None:
                        session.update()
                        left_tracker.push(session, left)
                        right_tracker.push(session, right)
                    frame_number += 1
                if not args.loop or args.validate_only:
                    break
    except KeyboardInterrupt:
        pass
    logger.info(
        "%s %d paired SOMA hand demo frames",
        "Encoded" if args.validate_only else "Published",
        frame_number,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
