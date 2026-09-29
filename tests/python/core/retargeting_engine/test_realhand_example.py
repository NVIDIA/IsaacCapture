# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-construction checks for every supported P7 and RealHand combination."""

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_PATH = (
    _REPO_ROOT / "examples" / "teleop" / "python" / "p7_realhand_bimanual_example.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "p7_realhand_bimanual_example", _EXAMPLE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_EXAMPLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXAMPLE)


@pytest.mark.parametrize(
    ("model", "expected_width"), (("l6", 36), ("o6", 26), ("l20", 46))
)
@pytest.mark.parametrize("mode", ("controller", "handtracking", "ffg"))
def test_pipeline_builds_for_every_input_mode(
    mode, model, expected_width, realhand_calibration_dir
):
    calibration = realhand_calibration_dir / f"{model}.yml"
    pipeline, action_order = _EXAMPLE.build_pipeline(mode, model, calibration)

    assert len(action_order) == expected_width
    assert list(pipeline.output_types()) == ["action"]
    assert pipeline.output_types()["action"].types[0].shape == (expected_width,)


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
def test_ffg_pipeline_requires_user_calibration(model):
    with pytest.raises(ValueError, match="requires a calibration YAML"):
        _EXAMPLE.build_pipeline("ffg", model, None)


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
