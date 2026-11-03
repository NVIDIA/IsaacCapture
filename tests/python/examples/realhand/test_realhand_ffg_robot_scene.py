# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check all three robot scenes without requiring Isaac Sim in the test environment."""

import json
from copy import deepcopy

import numpy as np
import pytest

from test_realhand_ffg_hand_demo import _BRIDGE, _VIEWER


@pytest.fixture
def scene():
    from isaaccapture_examples.teleop import realhand_ffg_robot_scene as module

    return module


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
def test_robot_live_worker_preserves_model_and_human_calibration(model, tmp_path):
    interpreter = tmp_path / "python"
    interpreter.touch()
    args = _VIEWER.parse_args(
        [
            "--hand-model",
            model,
            "--capture-python",
            str(interpreter),
            "--capture-package",
            str(tmp_path),
            "--plugin-path",
            str(tmp_path),
            "--calibration",
            str(tmp_path / "human.yml"),
        ]
    )
    command = _VIEWER.worker_command(args, tmp_path)
    assert command[command.index("--hand-model") + 1] == model
    assert command[command.index("--calibration") + 1] == str(tmp_path / "human.yml")
    assert "--ports" not in command
    pipeline = _BRIDGE.build_pipeline(model)
    assert set(pipeline.output_types()) == {
        "left",
        "right",
        "raw_left",
        "raw_right",
        "robot_left",
        "robot_right",
    }


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
def test_existing_human_calibration_drives_every_model_without_refit(
    model, request, calibrated_skeleton_samples, skeleton_samples, monkeypatch
):
    from pathlib import Path
    from isaaccapture_examples.teleop import (
        realhand_ffg_glove_calibration as calibration,
    )
    from isaaccapture_examples.teleop import realhand_ffg_thumb_calibration as fitter
    from test_realhand_ffg_glove_hand_input import _feed_samples

    path = Path(request.config.getoption("--ffg-human-calibration"))
    before = path.read_bytes()

    def unexpected_fit(*_):
        pytest.fail("Changing robot models must not refit the human calibration")

    monkeypatch.setattr(fitter, "fit_thumb_mapping", unexpected_fit)
    _BRIDGE.validate_saved_calibration(path, calibration)
    retargeters = {}
    _BRIDGE.build_pipeline(model, retargeters)
    for side, retargeter in retargeters.items():
        samples = deepcopy(calibrated_skeleton_samples["poses"][side])
        # Calibrated fixtures omit radii; use the same native model's neutral radii.
        for sample in samples:
            for joint, neutral in zip(
                sample["joints"], skeleton_samples[side][0]["joints"]
            ):
                joint["radius"] = neutral["radius"]
        results = _feed_samples(retargeter, side, samples)
        assert np.linalg.norm(results[1] - results[0]) > 0.1
    assert path.read_bytes() == before


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
def test_robot_fixture_replay_is_offline_and_matches_named_output(
    model, tmp_path, skeleton_samples, caplog
):
    interpreter = tmp_path / "python"
    interpreter.touch()
    args = _VIEWER.parse_args(
        [
            "--hand-model",
            model,
            "--test-frames",
            str(tmp_path / "fixture.yml"),
            "--capture-python",
            str(interpreter),
            "--capture-package",
            str(tmp_path),
        ]
    )
    command = _VIEWER.worker_command(args, tmp_path)
    assert "--plugin-path" not in command
    assert "--ipc-dir" not in command
    packets = _BRIDGE.replay_robot_frames(model, skeleton_samples)
    assert len(packets) == 60
    for packet in packets:
        assert len(json.dumps(packet).encode()) < 65536
        for side in ("left", "right"):
            robot = packet["robots"][side]
            assert len(robot["positions"]) == {"l6": 11, "o6": 6, "l20": 16}[model]
            assert all(
                name.startswith(f"{side}_hand_") for name in robot["joint_names"]
            )
            assert f"{side}_hand_hand_base_link" in robot["link_poses"]
            for matrix in robot["link_poses"].values():
                matrix = np.asarray(matrix)
                assert matrix.shape == (4, 4)
                np.testing.assert_allclose(matrix[3], (0, 0, 0, 1), atol=1e-6)
                np.testing.assert_allclose(
                    matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6
                )
    for side in ("left", "right"):
        first = packets[0]["robots"][side]["positions"]
        last = packets[40]["robots"][side]["positions"]
        assert np.linalg.norm(np.asarray(first) - last) > 0.5
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
def test_disconnected_glove_has_neutral_robot_output(model):
    retargeters = {}
    _BRIDGE.build_pipeline(model, retargeters)
    for retargeter in retargeters.values():
        frame = _BRIDGE.robot_frame(retargeter, None)
        np.testing.assert_allclose(frame["positions"], 0.0)
        assert frame["link_poses"]


def test_robot_display_is_palm_up_without_reflection(scene):
    for side in ("left", "right"):
        matrix = scene.robot_display_transform(side)
        assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)
        np.testing.assert_allclose(matrix[:3, :3] @ (1, 0, 0), (0, 0, 1))
        np.testing.assert_allclose(matrix[:3, :3] @ (0, 0, 1), (0, 1, 0))
        assert np.sign(matrix[0, 3]) == (-1 if side == "left" else 1)


def test_robot_visual_loading_uses_mesh_scale_origin_and_only_selected_hand(
    scene, monkeypatch, tmp_path
):
    urdf = tmp_path / "assembly.urdf"
    urdf.write_text(
        """
<robot name="fixture">
  <link name="arm"><visual><geometry><mesh filename="arm.stl"/></geometry></visual></link>
  <link name="right_hand_hand_base_link"><visual><geometry><mesh filename="right.stl"/></geometry></visual></link>
  <link name="left_hand_hand_base_link">
    <visual><origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>
      <geometry><mesh filename="left.stl" scale="2 3 4"/></geometry>
      <material><color rgba="0.1 0.2 0.3 1"/></material>
    </visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(scene, "_validate_model", lambda root, model: urdf)
    visuals = scene.visual_links(tmp_path, "l6", "left")
    assert list(visuals) == ["left_hand_hand_base_link"]
    filename, matrix, color = visuals["left_hand_hand_base_link"][0]
    assert filename == tmp_path / "left.stl"
    np.testing.assert_allclose(matrix[:3, 3], (0.1, 0.2, 0.3))
    np.testing.assert_allclose(matrix[:3, :3], np.diag((2, 3, 4)))
    assert color == pytest.approx((0.1, 0.2, 0.3, 1))
