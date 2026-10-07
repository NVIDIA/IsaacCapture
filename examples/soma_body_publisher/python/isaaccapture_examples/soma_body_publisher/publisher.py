# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the bundled SOMA motion through the SOMA body transport."""

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
    SomaBodyJointPose,
    SomaBodyJointPoses,
    SomaBodyJoint,
    SomaBodyJointRotation,
    SomaBodyJointRotations,
)

logger = logging.getLogger("isaaccapture.examples.soma_body_publisher")

BODY_JOINTS = tuple(
    joint
    for name, joint in sorted(
        SomaBodyJoint.__members__.items(), key=lambda item: int(item[1])
    )
    if name != "NUM_JOINTS"
)


def create_layer(data_root: Path | None = None):
    """Create the neutral SOMA-X v0.3.1 model used by the bundled motion."""
    import soma
    import torch

    # The v0.3.1 body-layer fallback imports a nonexistent soma.body.assets.
    if data_root is None:
        logger.info(
            "Loading SOMA assets. First use may download files and take a while."
        )
        data_root = soma.get_assets_dir()
    layer = soma.SOMALayer(
        data_root=str(data_root),
        identity_model_type="soma",
        device="cpu",
        lod="low",
        output_unit=soma.Unit.METERS,
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    expected = [
        name.replace("_", "")
        for name, joint in sorted(
            SomaBodyJoint.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    ]
    if [name.upper() for name in layer.public_joint_names] != ["ROOT", *expected]:
        raise ValueError("SOMA layer joint order differs from the FBS")
    with torch.no_grad():
        layer.prepare_identity(
            torch.zeros((1, layer.identity_model.num_identity_coeffs))
        )
    return layer


def demo_controls(path: Path, layer):
    """Normalize the bundled motion as in v0.3.1 tools/demo_soma_vis.py."""
    import torch
    from soma.geometry.rig_utils import joint_local_to_world, joint_world_to_local
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
        local = local[:, [i for i in range(94) if i not in twists]]
    orient = layer.t_pose_world[layer.public_transform_joint_indices, :3, :3]
    world = joint_local_to_world(local, layer.public_joint_parent_ids)
    relative = joint_world_to_local(
        world @ orient.transpose(-2, -1), layer.public_joint_parent_ids
    )[:, 1:]
    return matrix_to_quaternion_xyzw(relative).numpy(), motion[:, 1, :3, 3], relative


def soma_joint_rotations(quaternions, translation):
    joints = [
        SomaBodyJointRotation(joint, Quaternion(*quaternion))
        for joint, quaternion in zip(BODY_JOINTS, quaternions, strict=True)
    ]
    return SomaBodyJointRotations(joints, Point(*translation), True)


def evaluate_demo_controls(layer, rotations, translations):
    """Evaluate a batch of SOMA controls into global public joint poses."""
    import torch
    from soma.geometry.transforms import matrix_to_quaternion_xyzw

    with torch.no_grad():
        output = layer.pose(
            rotations,
            transl=torch.as_tensor(translations, dtype=torch.float32),
            pose2rot=False,
            fk_only=True,
            apply_correctives=False,
        )
        orientations = matrix_to_quaternion_xyzw(output["transforms"][:, 1:, :3, :3])
    return output["joints"].numpy(), orientations.numpy()


def soma_joint_poses(positions, orientations):
    joints = [
        SomaBodyJointPose(
            joint,
            Pose(Point(*position), Quaternion(*orientation)),
        )
        for joint, position, orientation in zip(
            BODY_JOINTS, positions, orientations, strict=True
        )
    ]
    return SomaBodyJointPoses(joints)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--body-representation",
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
    layer = create_layer()
    q, t, rotations = demo_controls(layer.data_root / "example_animation.npy", layer)
    if args.body_representation == "joint-poses":
        positions, orientations = evaluate_demo_controls(layer, rotations, t)
        frame_values = positions, orientations
    else:
        frame_values = q, t
    joint_poses = args.body_representation == "joint-poses"
    max_payload_size = 4096 if joint_poses else 2048
    frame = 0
    try:
        with ExitStack() as stack:
            session = None
            if not args.validate_only:
                tracker = TensorPushTracker(
                    "soma_body_demo",
                    "soma_body_joint_poses"
                    if joint_poses
                    else "soma_body_joint_rotations",
                    max_payload_size,
                )
                trackers = [tracker]
                extensions = DeviceIOSession.get_required_extensions(trackers)
                oxr_session = stack.enter_context(
                    OpenXRSession("SomaBodyDemoPublisher", extensions)
                )
                session = stack.enter_context(
                    DeviceIOSession.run(trackers, oxr_session.get_handles())
                )
            start = time.monotonic()
            while True:
                for first, second in zip(*frame_values):
                    remaining = start + frame / args.rate - time.monotonic()
                    if remaining > 0 and not args.validate_only:
                        time.sleep(remaining)
                    if args.body_representation == "joint-poses":
                        payload = soma_joint_poses(first, second).to_bytes()
                    else:
                        payload = soma_joint_rotations(first, second).to_bytes()
                    if len(payload) > max_payload_size:
                        raise ValueError("SOMA body payload exceeds transport capacity")
                    if session is not None:
                        session.update()
                        tracker.push(session, payload)
                    frame += 1
                if not args.loop or args.validate_only:
                    break
    except KeyboardInterrupt:
        pass
    logger.info(
        "%s %d SOMA body demo frames",
        "Encoded" if args.validate_only else "Published",
        frame,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
