# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct
import subprocess

import numpy as np

from isaaccapture.retargeting_engine.utilities.soma_hand_evaluator import (
    _SomaHandEvaluator,
)
from isaaccapture.schema import SomaHandednessV0
from isaaccapture_examples.soma_hand_publisher.publisher import (
    create_layers,
    demo_hand_frames,
    soma_joint_poses,
    soma_joint_rotations,
)


def test_hand_payload_constructors_preserve_side_and_arrays():
    rotations = np.tile([0.0, 0.0, 0.0, 1.0], (25, 1))
    rotation_payload = soma_joint_rotations(rotations, [1, 2, 3], SomaHandednessV0.LEFT)
    assert rotation_payload.handedness == SomaHandednessV0.LEFT
    np.testing.assert_array_equal(rotation_payload.joint_rotations.rotations, rotations)

    positions = np.arange(75, dtype=np.float32).reshape(25, 3)
    pose_payload = soma_joint_poses(positions, rotations, SomaHandednessV0.RIGHT)
    assert pose_payload.handedness == SomaHandednessV0.RIGHT
    np.testing.assert_array_equal(pose_payload.joint_poses.positions, positions)


def test_pusher_rejects_collection_handedness_mismatch(request):
    executable = request.config.getoption("--soma-hand-pusher")
    if executable is None:
        return
    rotations = np.tile([0.0, 0.0, 0.0, 1.0], (25, 1))
    wrong_left = soma_joint_rotations(
        rotations, [0, 0, 0], SomaHandednessV0.RIGHT
    ).to_bytes()
    right = soma_joint_rotations(
        rotations, [0, 0, 0], SomaHandednessV0.RIGHT
    ).to_bytes()
    packet = struct.pack("<IIQ", len(wrong_left), len(right), 0) + wrong_left + right

    result = subprocess.run(
        [str(executable), "--validate-only"],
        input=packet,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1


def test_demo_profiles_match_upstream_hand_fk(soma_assets, request):
    body, hands = create_layers(soma_assets)
    frames = demo_hand_frames(soma_assets / "example_animation.npy", body, hands)

    for side, handedness in (
        ("left", SomaHandednessV0.LEFT),
        ("right", SomaHandednessV0.RIGHT),
    ):
        frame = frames[side]
        payload = soma_joint_rotations(
            frame["rotations"][0], frame["translation"][0], handedness
        )
        positions, _, valid = _SomaHandEvaluator(hands[side]).evaluate(payload)
        np.testing.assert_allclose(positions, frame["positions"][0], atol=1e-5)
        np.testing.assert_array_equal(valid, np.ones(25, dtype=np.uint8))
        evaluated = soma_joint_poses(
            frame["positions"][0], frame["orientations"][0], handedness
        )
        np.testing.assert_allclose(
            evaluated.joint_poses.positions, positions, atol=1e-5
        )

    executable = request.config.getoption("--soma-hand-pusher")
    if executable is None:
        return
    for representation in ("joint-rotations", "joint-poses"):
        packets = []
        for index in range(len(frames["left"]["positions"])):
            encoded = []
            for side, handedness in (
                ("left", SomaHandednessV0.LEFT),
                ("right", SomaHandednessV0.RIGHT),
            ):
                frame = frames[side]
                if representation == "joint-poses":
                    payload = soma_joint_poses(
                        frame["positions"][index],
                        frame["orientations"][index],
                        handedness,
                    )
                else:
                    payload = soma_joint_rotations(
                        frame["rotations"][index],
                        frame["translation"][index],
                        handedness,
                    )
                encoded.append(payload.to_bytes())
            packets.append(
                struct.pack("<IIQ", len(encoded[0]), len(encoded[1]), index)
                + encoded[0]
                + encoded[1]
            )
        result = subprocess.run(
            [
                str(executable),
                "--hand-representation",
                representation,
                "--validate-only",
            ],
            input=b"".join(packets),
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
