# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-construction checks for every supported P7 and RealHand combination."""

import importlib.util
import sys
from copy import deepcopy
from contextlib import nullcontext
from repo_paths import repo_root

import pytest

_REPO_ROOT = repo_root()
_EXAMPLE_PATH = (
    _REPO_ROOT
    / "examples/teleop/python/isaaccapture_examples/teleop/p7_realhand_bimanual_example.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "isaaccapture_examples.teleop.p7_realhand_bimanual_example", _EXAMPLE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_EXAMPLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXAMPLE)


@pytest.mark.parametrize(
    ("model", "expected_width"), (("l6", 36), ("o6", 26), ("l20", 46))
)
@pytest.mark.parametrize("mode", ("controller", "handtracking", "ffg"))
def test_pipeline_builds_for_every_input_mode(mode, model, expected_width):
    pipeline, action_order = _EXAMPLE.build_pipeline(mode, model)

    assert len(action_order) == expected_width
    assert list(pipeline.output_types()) == ["action"]
    assert pipeline.output_types()["action"].types[0].shape == (expected_width,)


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
def test_ffg_uses_the_same_hand_retargeter_as_optical_tracking(model, monkeypatch):
    configs = []
    import isaaccapture.retargeters as official

    original = official.DexHandRetargeter

    def capture(config, **kwargs):
        configs.append(deepcopy(config))
        return original(config, **kwargs)

    monkeypatch.setattr(official, "DexHandRetargeter", capture)
    _EXAMPLE.build_pipeline("ffg", model)
    glove_configs = list(configs)
    configs.clear()
    _EXAMPLE.build_pipeline("handtracking", model)
    assert configs == glove_configs
    assert [config.hand_side for config in configs] == ["left", "right"]
    assert all(f"/{model}/" in config.hand_urdf for config in configs)
    assert all(not hasattr(config, "calibration_path") for config in configs)


def test_combined_example_has_no_raw_sensor_source():
    assert not hasattr(_EXAMPLE, "JointStateSource")
    assert not hasattr(_EXAMPLE, "RealHandFFGGloveRetargeter")
    assert not hasattr(_EXAMPLE, "RealHandHandTrackingRetargeter")


def test_starting_ffg_plugin_requires_human_calibration(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_EXAMPLE_PATH), "--mode", "ffg", "--plugin-path", "install/plugins"],
    )
    with pytest.raises(SystemExit) as error:
        _EXAMPLE.main()
    assert error.value.code == 2


def test_external_calibrated_glove_plugin_does_not_require_robot_calibration(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_EXAMPLE_PATH), "--mode", "ffg", "--hand-model", "l20", "--dry-run"],
    )
    _EXAMPLE.main()


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_home_pose_has_hand_back_forward_and_fingers_down(model, side):
    rotation = _EXAMPLE.Rotation.from_quat(_EXAMPLE._HOME_POSES[model][side][1])

    hand_back = rotation.apply((-1.0, 0.0, 0.0))
    finger_direction = rotation.apply((0.0, 0.0, 1.0))

    assert hand_back[1] > 0.99
    assert finger_direction[2] < -0.99


def test_each_handtracking_model_uses_its_own_home_rotation():
    for side in ("left", "right"):
        actual = _EXAMPLE._pose_config("l20", side, "handtracking")
        expected = _EXAMPLE._home_rpy("l20", side)

        assert actual.rotation_offset_rpy_deg == pytest.approx(expected)


@pytest.mark.parametrize("calibration_example", (False, True))
@pytest.mark.parametrize("ports", (None, "/dev/ttyUSB0,/dev/ttyUSB1"))
def test_example_forwards_verified_glove_ports(
    calibration_example, ports, monkeypatch, human_calibration, tmp_path
):
    module = _EXAMPLE
    if calibration_example:
        path = _EXAMPLE_PATH.with_name("realhand_ffg_glove_calibration.py")
        spec = importlib.util.spec_from_file_location(
            "isaaccapture_examples.teleop.realhand_ffg_glove_calibration", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arguments = ["--output", str(tmp_path / "new.yml")]
    else:
        arguments = [
            "--mode",
            "ffg",
            "--hand-model",
            "l6",
            "--realhand-ffg-glove-calibration",
            str(human_calibration),
        ]
    arguments.extend(["--plugin-path", str(_REPO_ROOT / "install" / "plugins")])
    if ports:
        arguments.append(f"--realhand-ffg-glove-ports={ports}")
    monkeypatch.setattr(sys, "argv", [str(_EXAMPLE_PATH), *arguments])
    captured = []

    class StopBeforeOpenXR(Exception):
        pass

    def capture_config(config):
        captured.append(config)
        raise StopBeforeOpenXR

    monkeypatch.setattr(module, "TeleopSession", capture_config)
    monkeypatch.setattr(
        module.CloudXRLauncher, "launch_context", lambda args: nullcontext()
    )
    with pytest.raises(StopBeforeOpenXR):
        module.main()
    assert len(captured) == 1
    expected = (
        ["--calibration="]
        if calibration_example
        else [
            "--wrist-source=controller",
            f"--calibration={human_calibration.resolve()}",
        ]
    )
    if ports:
        expected.append(f"--ports={ports}")
    assert captured[0].plugins[0].plugin_args == expected
