# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


@pytest.fixture
def demo(monkeypatch, tmp_path):
    body = SimpleNamespace(data_root=tmp_path)
    layers = MagicMock(return_value=(body, {}))
    frame = {
        "rotations": np.tile([0, 0, 0, 1], (2, 25, 1)),
        "translation": np.zeros((2, 3)),
        "positions": np.zeros((2, 25, 3)),
        "orientations": np.tile([0, 0, 0, 1], (2, 25, 1)),
    }
    frames = MagicMock(return_value={"left": frame, "right": frame})
    session = MagicMock()
    session.__enter__.return_value = session
    oxr_session = MagicMock()
    oxr_session.__enter__.return_value = oxr_session
    deviceio = MagicMock()
    deviceio.run.return_value = session
    oxr = MagicMock(return_value=oxr_session)
    left, right = MagicMock(), MagicMock()
    tracker_factory = MagicMock(side_effect=[left, right])
    monkeypatch.setattr(publisher, "create_layers", layers)
    monkeypatch.setattr(publisher, "demo_hand_frames", frames)
    monkeypatch.setattr(publisher, "DeviceIOSession", deviceio)
    monkeypatch.setattr(publisher, "OpenXRSession", oxr)
    monkeypatch.setattr(publisher, "TensorPushTracker", tracker_factory)
    monkeypatch.setattr(publisher.time, "monotonic", lambda: 0.0)
    sleep = MagicMock()
    monkeypatch.setattr(publisher.time, "sleep", sleep)
    return SimpleNamespace(
        session=session,
        oxr_session=oxr_session,
        deviceio=deviceio,
        oxr=oxr,
        left=left,
        right=right,
        tracker_factory=tracker_factory,
        sleep=sleep,
        layers=layers,
        frames=frames,
        body=body,
        frame=frame,
    )


@pytest.mark.parametrize(
    "representation,capacity", [("joint-rotations", 1024), ("joint-poses", 2048)]
)
def test_publisher_pushes_each_hand_through_one_session(demo, representation, capacity):
    assert (
        publisher.main(
            [
                "soma_hand_publisher",
                "--hand-representation",
                representation,
            ]
        )
        == 0
    )

    demo.layers.assert_called_once_with()
    demo.frames.assert_called_once_with(
        demo.body.data_root / "example_animation.npy", demo.body, {}
    )
    identifier = "soma_hand_" + representation.replace("-", "_")
    assert [call.args for call in demo.tracker_factory.call_args_list] == [
        ("soma_hand_left_demo", identifier, capacity),
        ("soma_hand_right_demo", identifier, capacity),
    ]
    demo.deviceio.get_required_extensions.assert_called_once_with(
        [demo.left, demo.right]
    )
    demo.oxr.assert_called_once_with(
        "SomaHandDemoPublisher", demo.deviceio.get_required_extensions.return_value
    )
    demo.deviceio.run.assert_called_once_with(
        [demo.left, demo.right], demo.oxr_session.get_handles.return_value
    )
    assert demo.session.update.call_count == 2
    for tracker, handedness in (
        (demo.left, SomaHandedness.LEFT),
        (demo.right, SomaHandedness.RIGHT),
    ):
        assert tracker.push.call_count == 2
        tracker.push.assert_called_with(
            demo.session, publisher._payload(demo.frame, 1, representation, handedness)
        )
    demo.sleep.assert_called_once_with(1 / 30)
    demo.session.__exit__.assert_called_once()
    demo.oxr_session.__exit__.assert_called_once()


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
def test_validate_only_does_not_open_runtime_or_loop(demo, representation):
    assert (
        publisher.main(
            [
                "publisher",
                "--hand-representation",
                representation,
                "--validate-only",
                "--loop",
            ]
        )
        == 0
    )
    demo.tracker_factory.assert_not_called()
    demo.deviceio.run.assert_not_called()
    demo.oxr.assert_not_called()
    demo.sleep.assert_not_called()


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_publisher_closes_sessions_and_preserves_push_failures(demo, error):
    demo.right.push.side_effect = error("push stopped")
    if error is KeyboardInterrupt:
        assert publisher.main(["publisher"]) == 0
    else:
        with pytest.raises(RuntimeError, match="push stopped"):
            publisher.main(["publisher"])
    demo.session.__exit__.assert_called_once()
    demo.oxr_session.__exit__.assert_called_once()


def test_session_initialization_failure_closes_openxr(demo):
    demo.deviceio.run.side_effect = RuntimeError("session failed")
    with pytest.raises(RuntimeError, match="session failed"):
        publisher.main(["publisher"])
    demo.oxr_session.__exit__.assert_called_once()
    demo.left.push.assert_not_called()
    demo.right.push.assert_not_called()


def test_publisher_loop_repeats_clip_until_interrupted(demo):
    demo.right.push.side_effect = [None, None, KeyboardInterrupt]
    assert publisher.main(["publisher", "--loop"]) == 0
    assert demo.right.push.call_count == 3
    assert (
        demo.right.push.call_args_list[0].args[1]
        == demo.right.push.call_args_list[2].args[1]
    )


@pytest.mark.parametrize(
    "representation,capacity", [("joint-rotations", 1024), ("joint-poses", 2048)]
)
def test_validate_only_rejects_oversized_payload(
    demo, monkeypatch, representation, capacity
):
    monkeypatch.setattr(publisher, "_payload", lambda *args: bytes(capacity + 1))
    with pytest.raises(ValueError, match="exceeds transport capacity"):
        publisher.main(
            ["publisher", "--hand-representation", representation, "--validate-only"]
        )


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf"])
def test_invalid_rate_is_rejected_before_loading_model(monkeypatch, rate):
    layers = MagicMock()
    monkeypatch.setattr(publisher, "create_layers", layers)
    with pytest.raises(SystemExit) as error:
        publisher.main(["publisher", "--rate", rate])
    assert error.value.code == 2
    layers.assert_not_called()


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


def test_demo_profiles_match_upstream_hand_fk(soma_assets):
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

    for representation in ("joint-rotations", "joint-poses"):
        capacity = 2048 if representation == "joint-poses" else 1024
        for index in range(len(frames["left"]["positions"])):
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
                assert len(payload.to_bytes()) <= capacity
