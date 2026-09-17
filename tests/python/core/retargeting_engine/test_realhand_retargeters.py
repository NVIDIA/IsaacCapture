# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulation-free checks for P7 and RealHand retargeters."""

import numpy as np
import pytest

from isaaccapture.retargeters.realhand.arm import (
    P7ControllerPoseRetargeter,
    P7WorkspacePoseConfig,
)
from isaaccapture.retargeters.realhand.realhand_ffg_glove import (
    RealHandFFGGloveRetargeterConfig,
    _NaturalRealHandFFGGloveMapper,
    load_realhand_ffg_glove_calibration,
)
from isaaccapture.retargeters.realhand.hand import (
    ControllerTriggerRealHandRetargeter,
    ControllerTriggerRealHandRetargeterConfig,
    RealHandHandTrackingRetargeter,
    RealHandHandTrackingRetargeterConfig,
)
from isaaccapture.retargeters.realhand.profiles import get_realhand_profile
from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalTensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
)

_DIGIT_SENSOR_CHANNELS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


def _context(*, reset: bool = False) -> ComputeContext:
    return ComputeContext(
        graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
        execution_events=ExecutionEvents(
            reset=reset, execution_state=ExecutionState.RUNNING
        ),
    )


def _build_io(retargeter):
    def make_group(group_type):
        if isinstance(group_type, OptionalTensorGroupType):
            return OptionalTensorGroup(group_type)
        return TensorGroup(group_type)

    return (
        {name: make_group(spec) for name, spec in retargeter.input_spec().items()},
        {name: make_group(spec) for name, spec in retargeter.output_spec().items()},
    )


def _fill_controller(group, *, position=(0.0, 0.0, 0.0), trigger=0.0, squeeze=0.0):
    group[ControllerInputIndex.GRIP_POSITION] = np.asarray(position, dtype=np.float32)
    group[ControllerInputIndex.GRIP_ORIENTATION] = np.asarray(
        (0.0, 0.0, 0.0, 1.0), dtype=np.float32
    )
    group[ControllerInputIndex.GRIP_IS_VALID] = True
    group[ControllerInputIndex.TRIGGER_VALUE] = trigger
    group[ControllerInputIndex.SQUEEZE_VALUE] = squeeze


_FINGER_INDICES = {
    "index": (
        HandJointIndex.INDEX_METACARPAL,
        HandJointIndex.INDEX_PROXIMAL,
        HandJointIndex.INDEX_INTERMEDIATE,
        HandJointIndex.INDEX_DISTAL,
        HandJointIndex.INDEX_TIP,
    ),
    "middle": (
        HandJointIndex.MIDDLE_METACARPAL,
        HandJointIndex.MIDDLE_PROXIMAL,
        HandJointIndex.MIDDLE_INTERMEDIATE,
        HandJointIndex.MIDDLE_DISTAL,
        HandJointIndex.MIDDLE_TIP,
    ),
    "ring": (
        HandJointIndex.RING_METACARPAL,
        HandJointIndex.RING_PROXIMAL,
        HandJointIndex.RING_INTERMEDIATE,
        HandJointIndex.RING_DISTAL,
        HandJointIndex.RING_TIP,
    ),
    "pinky": (
        HandJointIndex.LITTLE_METACARPAL,
        HandJointIndex.LITTLE_PROXIMAL,
        HandJointIndex.LITTLE_INTERMEDIATE,
        HandJointIndex.LITTLE_DISTAL,
        HandJointIndex.LITTLE_TIP,
    ),
}


def _direction(flexion: float) -> np.ndarray:
    return np.asarray((0.0, np.cos(flexion), -np.sin(flexion)))


def _synthetic_hand(close: float, *, mirror_x: bool) -> np.ndarray:
    positions = np.zeros((26, 3), dtype=np.float32)
    positions[HandJointIndex.PALM] = (0.0, 0.04, 0.0)
    bases = {
        "index": (0.031, 0.031, 0.0),
        "middle": (0.010, 0.036, 0.0),
        "ring": (-0.012, 0.033, 0.0),
        "pinky": (-0.034, 0.026, 0.0),
    }
    lengths = (0.026, 0.035, 0.025, 0.019)
    for finger, indices in _FINGER_INDICES.items():
        point = np.asarray(bases[finger], dtype=np.float64)
        positions[indices[0]] = point
        for segment, (index, length) in enumerate(zip(indices[1:], lengths)):
            point = point + length * _direction(close * (1.15 + 0.65 * segment))
            positions[index] = point

    thumb_open = np.asarray(
        (
            (0.045, 0.020, -0.002),
            (0.065, 0.030, -0.001),
            (0.083, 0.038, 0.000),
            (0.099, 0.044, 0.001),
        )
    )
    index_tip = positions[HandJointIndex.INDEX_TIP]
    thumb_closed = np.asarray(
        (
            thumb_open[0],
            0.70 * thumb_open[0] + 0.30 * index_tip + (0.008, -0.004, 0.014),
            0.35 * thumb_open[0] + 0.65 * index_tip + (0.004, -0.009, 0.010),
            index_tip + (0.002, -0.001, 0.001),
        )
    )
    thumb = (1.0 - close) * thumb_open + close * thumb_closed
    for index, point in zip(
        (
            HandJointIndex.THUMB_METACARPAL,
            HandJointIndex.THUMB_PROXIMAL,
            HandJointIndex.THUMB_DISTAL,
            HandJointIndex.THUMB_TIP,
        ),
        thumb,
    ):
        positions[index] = point
    if mirror_x:
        positions[:, 0] *= -1.0
    return positions


def _fill_hand(group, positions: np.ndarray) -> None:
    group[HandInputIndex.JOINT_POSITIONS] = positions
    group[HandInputIndex.JOINT_ORIENTATIONS] = np.tile(
        np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (26, 1)
    )
    group[HandInputIndex.JOINT_RADII] = np.full(26, 0.008, dtype=np.float32)
    group[HandInputIndex.JOINT_VALID] = np.ones(26, dtype=np.uint8)


def _mapper(model: str, calibration_dir, side: str = "left"):
    profile = get_realhand_profile(model)
    config = RealHandFFGGloveRetargeterConfig(
        input_device="glove",
        joint_names=profile.joint_names(side),
        side=side,
        hand_model=model,
        calibration_path=str(calibration_dir / f"{model}.yml"),
    )
    return profile, _NaturalRealHandFFGGloveMapper(config)


@pytest.mark.parametrize("model", ["l6", "o6", "l20"])
def test_profile_resolves_downloaded_urdf(model, tmp_path):
    profile = get_realhand_profile(model)
    urdf = tmp_path / profile.asset_dir_name / profile.urdf_name
    urdf.parent.mkdir(parents=True)
    urdf.touch()

    assert profile.resolve_urdf(tmp_path) == urdf.resolve()


def test_profile_reports_asset_fetch_command_when_urdf_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"fetch_realhand_assets\.py"):
        get_realhand_profile("l6").resolve_urdf(tmp_path)


@pytest.mark.parametrize("model", ["l6", "o6", "l20"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_open_and_fist_cover_joint_ranges(model, side, realhand_calibration_dir):
    profile, mapper = _mapper(model, realhand_calibration_dir, side)
    names = profile.joint_names(side)
    calibration = load_realhand_ffg_glove_calibration(
        realhand_calibration_dir / f"{model}.yml", side
    )
    opened = mapper.map(calibration["open"], names)
    closed = mapper.map(calibration["fist"], names)

    assert np.all(np.isfinite(opened))
    assert np.all(np.isfinite(closed))
    assert np.max(np.abs(opened)) < 1.0e-8
    for index, spec in enumerate(profile.joints(side)):
        if not spec.semantic.endswith("_spread"):
            assert closed[index] >= 0.85 * min(spec.trigger_closed, spec.upper)


@pytest.mark.parametrize("model", ["l6", "o6", "l20"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_motion_beyond_fist_does_not_reopen_flexion_joints(
    model, side, realhand_calibration_dir
):
    profile, mapper = _mapper(model, realhand_calibration_dir, side)
    names = profile.joint_names(side)
    calibration = load_realhand_ffg_glove_calibration(
        realhand_calibration_dir / f"{model}.yml", side
    )
    opened = np.asarray(calibration["open"], dtype=np.float64)
    fist = np.asarray(calibration["fist"], dtype=np.float64)
    beyond_fist = opened + 1.5 * (fist - opened)

    closed = mapper.map(fist, names)
    beyond = mapper.map(beyond_fist, names)
    flexion = [
        index
        for index, spec in enumerate(profile.joints(side))
        if not spec.semantic.endswith("_spread")
    ]
    assert np.all(beyond[flexion] >= closed[flexion] - 1.0e-8)


@pytest.mark.parametrize("model", ["l6", "o6", "l20"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("digit", ["thumb", "index", "middle", "ring", "pinky"])
def test_each_digit_ignores_other_digit_sensors(
    model, side, digit, realhand_calibration_dir
):
    profile, mapper = _mapper(model, realhand_calibration_dir, side)
    names = profile.joint_names(side)
    calibration = load_realhand_ffg_glove_calibration(
        realhand_calibration_dir / f"{model}.yml", side
    )
    baseline = np.asarray(calibration["open"], dtype=np.float64)
    changed = baseline.copy()
    for other_digit, channels in _DIGIT_SENSOR_CHANNELS.items():
        if other_digit != digit:
            changed[list(channels)] += 0.75

    baseline_target = mapper.map(baseline, names)
    changed_target = mapper.map(changed, names)
    own_joints = [
        index
        for index, spec in enumerate(profile.joints(side))
        if spec.semantic.startswith(digit)
    ]
    np.testing.assert_allclose(
        changed_target[own_joints], baseline_target[own_joints], atol=1.0e-12
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_l20_thumb_is_independent_of_non_thumb_sensors(side, realhand_calibration_dir):
    profile, mapper = _mapper("l20", realhand_calibration_dir, side)
    names = profile.joint_names(side)
    calibration = load_realhand_ffg_glove_calibration(
        realhand_calibration_dir / "l20.yml", side
    )
    baseline = np.asarray(calibration["open"], dtype=np.float64)
    changed = baseline.copy()
    changed[5:] += np.linspace(-1.0, 1.0, len(changed) - 5)

    baseline_target = mapper.map(baseline, names)
    changed_target = mapper.map(changed, names)
    thumb = [
        index
        for index, spec in enumerate(profile.joints(side))
        if spec.semantic.startswith("thumb")
    ]
    np.testing.assert_allclose(
        changed_target[thumb], baseline_target[thumb], atol=1.0e-12
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_l20_each_finger_ignores_other_finger_sensors(side, realhand_calibration_dir):
    profile, mapper = _mapper("l20", realhand_calibration_dir, side)
    names = profile.joint_names(side)
    calibration = load_realhand_ffg_glove_calibration(
        realhand_calibration_dir / "l20.yml", side
    )
    baseline = np.asarray(calibration["open"], dtype=np.float64)
    changed = baseline.copy()
    changed[14] += 1.0
    changed[16] += 1.0

    baseline_target = mapper.map(baseline, names)
    changed_target = mapper.map(changed, names)
    index_joints = [
        index
        for index, spec in enumerate(profile.joints(side))
        if spec.semantic.startswith("index_")
    ]
    np.testing.assert_allclose(
        changed_target[index_joints], baseline_target[index_joints], atol=1.0e-12
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("finger", ["index", "middle", "ring", "pinky"])
def test_l20_thumb_calibration_pose_is_an_exact_curve_anchor(
    side, finger, realhand_calibration_dir
):
    profile, mapper = _mapper("l20", realhand_calibration_dir, side)
    names = profile.joint_names(side)
    calibration = load_realhand_ffg_glove_calibration(
        realhand_calibration_dir / "l20.yml", side
    )
    pose = calibration["oposes"][finger]
    target = mapper.map(pose, names)
    assert np.all(np.isfinite(target))
    assert np.linalg.norm(target[:4]) > 0.2


def test_profiles_have_unique_action_joint_names():
    for model in ("l6", "o6", "l20"):
        profile = get_realhand_profile(model)
        assert len(profile.action_joint_names) == len(set(profile.action_joint_names))


def test_p7_controller_pose_maps_absolute_workspace_and_clamps():
    retargeter = P7ControllerPoseRetargeter(
        P7WorkspacePoseConfig(
            input_device="controller_left",
            fallback_position=(0.0, 0.0, 0.0),
            fallback_rotation=(0.0, 0.0, 0.0, 1.0),
            input_center=(0.0, 0.0, 0.0),
            workspace_center=(1.0, 2.0, 3.0),
            position_scale=(2.0, 2.0, 2.0),
            max_delta=(0.25, 0.25, 0.25),
        ),
        name="p7_left",
    )
    inputs, outputs = _build_io(retargeter)
    _fill_controller(inputs["controller_left"], position=(0.1, -0.2, 0.5))
    retargeter.compute(inputs, outputs, _context(reset=True))

    pose = np.from_dlpack(outputs["ee_pose"][0])
    np.testing.assert_allclose(pose[:3], (1.2, 1.75, 3.25), atol=1.0e-6)
    np.testing.assert_allclose(pose[3:], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6)


@pytest.mark.parametrize("model", ["l6", "o6", "l20"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_controller_trigger_closes_only_configured_hand(model, side):
    profile = get_realhand_profile(model)
    joint_names = profile.joint_names(side)
    input_name = f"controller_{side}"
    retargeter = ControllerTriggerRealHandRetargeter(
        ControllerTriggerRealHandRetargeterConfig(
            input_device=input_name,
            joint_names=joint_names,
            side=side,
            hand_model=model,
            smoothing_alpha=1.0,
        ),
        name=f"{model}_{side}",
    )
    inputs, outputs = _build_io(retargeter)
    _fill_controller(inputs[input_name], trigger=1.0)
    retargeter.compute(inputs, outputs, _context(reset=False))

    actual = np.asarray([float(value) for value in outputs["hand_joints"]])
    expected = np.asarray([joint.trigger_closed for joint in profile.joints(side)])
    np.testing.assert_allclose(actual, expected, atol=1.0e-6)


@pytest.mark.parametrize("model", ["l6", "o6", "l20"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_handtracking_open_to_fist_drives_every_digit(model, side):
    profile = get_realhand_profile(model)
    input_name = f"hand_{side}"
    retargeter = RealHandHandTrackingRetargeter(
        RealHandHandTrackingRetargeterConfig(
            input_device=input_name,
            joint_names=profile.joint_names(side),
            side=side,
            hand_model=model,
            smoothing_alpha=1.0,
        ),
        name=f"{model}_{side}_tracking",
    )
    inputs, outputs = _build_io(retargeter)
    _fill_hand(inputs[input_name], _synthetic_hand(0.0, mirror_x=side == "right"))
    retargeter.compute(inputs, outputs, _context(reset=True))
    opened = np.asarray([float(value) for value in outputs["hand_joints"]])

    _fill_hand(inputs[input_name], _synthetic_hand(1.0, mirror_x=side == "right"))
    retargeter.compute(inputs, outputs, _context())
    closed = np.asarray([float(value) for value in outputs["hand_joints"]])

    np.testing.assert_allclose(opened, 0.0, atol=1.0e-6)
    for digit in ("thumb", "index", "middle", "ring", "pinky"):
        indices = [
            index
            for index, spec in enumerate(profile.joints(side))
            if spec.semantic.startswith(digit) and not spec.semantic.endswith("_spread")
        ]
        assert indices
        assert np.max(closed[indices]) > 0.20
