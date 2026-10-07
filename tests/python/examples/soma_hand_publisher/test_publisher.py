# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from isaaccapture.retargeting_engine.utilities.soma_body_evaluator import (
    _dense_joint_poses,
    _dense_joint_rotations,
)
from isaaccapture.retargeting_engine.utilities.soma_hand_evaluator import (
    _SomaHandEvaluator,
)
from isaaccapture.schema import SomaHandedness
from isaaccapture_examples.soma_hand_publisher import publisher
from isaaccapture_examples.soma_hand_publisher.publisher import (
    create_layers,
    demo_hand_frames,
    soma_joint_poses,
    soma_joint_rotations,
)


def test_layers_resolve_assets_without_explicit_path(monkeypatch, tmp_path):
    soma = MagicMock()
    soma.get_assets_dir.return_value = tmp_path
    monkeypatch.setitem(sys.modules, "soma", soma)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())

    body, hands = create_layers()

    assert soma.SOMALayer.call_args.kwargs["data_root"] == str(tmp_path)
    assert [call.kwargs["hand_type"] for call in soma.SOMAHandLayer.call_args_list] == [
        "left",
        "right",
    ]
    assert all(
        call.kwargs["data_root"] == str(tmp_path)
        for call in soma.SOMAHandLayer.call_args_list
    )
    body.prepare_identity.assert_called_once()
    assert set(hands) == {"left", "right"}


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
def test_publisher_uses_resolved_demo_without_asset_argument(
    monkeypatch, tmp_path, representation
):
    body = SimpleNamespace(data_root=tmp_path)
    layers = MagicMock(return_value=(body, {}))
    frame = {
        "rotations": np.tile([0, 0, 0, 1], (1, 25, 1)),
        "translation": np.zeros((1, 3)),
        "positions": np.zeros((1, 25, 3)),
        "orientations": np.tile([0, 0, 0, 1], (1, 25, 1)),
    }
    demo = MagicMock(return_value={"left": frame, "right": frame})
    process = MagicMock()
    process.__enter__.return_value = process
    process.wait.return_value = 0
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(publisher, "create_layers", layers)
    monkeypatch.setattr(publisher, "demo_hand_frames", demo)
    monkeypatch.setattr(publisher.subprocess, "Popen", popen)

    assert (
        publisher.main(
            [
                "soma_hand_publisher",
                "--pusher",
                "pusher",
                "--hand-representation",
                representation,
                "--validate-only",
            ]
        )
        == 0
    )

    layers.assert_called_once_with()
    demo.assert_called_once_with(tmp_path / "example_animation.npy", body, {})
    assert popen.call_args.args[0][1:] == [
        "--hand-representation",
        representation,
        "--validate-only",
    ]
    assert process.stdin.write.call_count == 3
    process.stdin.close.assert_called_once()


def test_hand_payload_constructors_preserve_side_and_arrays():
    rotations = np.tile([0.0, 0.0, 0.0, 1.0], (25, 1))
    rotation_payload = soma_joint_rotations(rotations, [1, 2, 3], SomaHandedness.LEFT)
    assert rotation_payload.handedness == SomaHandedness.LEFT
    dense_rotations, rotation_valid = _dense_joint_rotations(
        rotation_payload.joint_rotations, 25
    )
    np.testing.assert_array_equal(dense_rotations, rotations)
    assert rotation_valid.all()

    positions = np.arange(75, dtype=np.float32).reshape(25, 3)
    pose_payload = soma_joint_poses(positions, rotations, SomaHandedness.RIGHT)
    assert pose_payload.handedness == SomaHandedness.RIGHT
    dense_positions, _, pose_valid = _dense_joint_poses(pose_payload.joint_poses, 25)
    np.testing.assert_array_equal(dense_positions, positions)
    assert pose_valid.all()


def test_pusher_rejects_collection_handedness_mismatch(request):
    executable = request.config.getoption("--soma-hand-pusher")
    if executable is None:
        return
    rotations = np.tile([0.0, 0.0, 0.0, 1.0], (25, 1))
    wrong_left = soma_joint_rotations(
        rotations, [0, 0, 0], SomaHandedness.RIGHT
    ).to_bytes()
    right = soma_joint_rotations(rotations, [0, 0, 0], SomaHandedness.RIGHT).to_bytes()
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
        ("left", SomaHandedness.LEFT),
        ("right", SomaHandedness.RIGHT),
    ):
        frame = frames[side]
        payload = soma_joint_rotations(
            frame["rotations"][0], frame["translation"][0], handedness
        )
        positions, _, valid = _SomaHandEvaluator(hands[side], side).evaluate(payload)
        np.testing.assert_allclose(positions, frame["positions"][0], atol=1e-5)
        np.testing.assert_array_equal(valid, np.ones(25, dtype=np.uint8))
        evaluated = soma_joint_poses(
            frame["positions"][0], frame["orientations"][0], handedness
        )
        evaluated_positions, _, evaluated_valid = _dense_joint_poses(
            evaluated.joint_poses, 25
        )
        np.testing.assert_allclose(evaluated_positions, positions, atol=1e-5)
        assert evaluated_valid.all()

    executable = request.config.getoption("--soma-hand-pusher")
    if executable is None:
        return
    for representation in ("joint-rotations", "joint-poses"):
        packets = []
        for index in range(len(frames["left"]["positions"])):
            encoded = []
            for side, handedness in (
                ("left", SomaHandedness.LEFT),
                ("right", SomaHandedness.RIGHT),
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
