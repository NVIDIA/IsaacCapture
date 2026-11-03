# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate robot data and reachable targets using the unmodified official solver."""

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from isaaccapture.retargeters import DexHandRetargeter
from isaaccapture.retargeters.realhand.profiles import get_realhand_profile
from isaaccapture.retargeting_engine.tensor_types import HandJointIndex
from test_realhand_example import _EXAMPLE
from test_realhand_ffg_glove_hand_input import _feed_samples


def _robot_positions(retargeter, qpos, names):
    robot = retargeter._dex_hand.optimizer.robot
    robot.compute_forward_kinematics(qpos)
    return np.asarray(
        [robot.get_link_pose(robot.get_link_index(name))[:3, 3] for name in names]
    )


def _full_qpos(retargeter, active):
    optimizer = retargeter._dex_hand.optimizer
    qpos = np.zeros(optimizer.robot.dof)
    for name, value in zip(retargeter._config.hand_joint_names, active):
        qpos[optimizer.robot.get_joint_index(name)] = value
    if optimizer.adaptor is not None:
        qpos = optimizer.adaptor.forward_qpos(qpos)
    return qpos


def _reachable_sample(retargeter, data, qpos, rotation, translation):
    origins = _robot_positions(retargeter, qpos, data["target_origin_link_names"])
    tasks = _robot_positions(retargeter, qpos, data["target_task_link_names"])
    points = np.zeros((21, 3))
    for indices, positions in zip(data["target_link_human_indices"], (origins, tasks)):
        for index, position in zip(indices, positions):
            points[index] = position
    transform = np.asarray(
        retargeter._config.handtracking_to_baselink_frame_transform
    ).reshape(3, 3)
    points = rotation.apply(points @ transform.T / data["scaling_factor"]) + translation
    joints = [
        {"position": translation, "orientation": rotation.as_quat(), "radius": 0.008}
        for _ in range(26)
    ]
    for index, point in zip(retargeter._hand_joints_index, points):
        joints[index] = {
            "position": point,
            "orientation": rotation.as_quat(),
            "radius": 0.008,
        }
    return {"joints": joints}, tasks - origins


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_model_config_contains_only_robot_data(model, side):
    config = _EXAMPLE.dex_hand_config(model, side)
    data = yaml.safe_load(Path(config.hand_retargeting_config).read_text())[
        "retargeting"
    ]
    root = ET.parse(config.hand_urdf).getroot()
    active = [
        j.get("name")
        for j in root.findall("joint")
        if j.get("type") == "revolute" and j.find("mimic") is None
    ]
    assert (
        set(active) == set(config.hand_joint_names) == set(data["target_joint_names"])
    )
    assert data["type"] == "vector"
    assert data["scaling_factor"] == (0.9 if model == "l20" else 0.75)
    assert data["normal_delta"] == (0.016 if model == "l20" else 0.001)
    assert data["huber_delta"] == (0.005 if model == "l20" else 0.02)
    assert data["low_pass_alpha"] == 0.4
    assert not {"project_dist", "escape_dist", "gamma"}.intersection(data)
    if model == "l20":
        assert f"{side}_hand_hand_base_link" not in data["target_origin_link_names"]
        pairs = set(
            zip(data["target_origin_link_names"], data["target_task_link_names"])
        )
        for finger in ("index", "middle", "ring", "pinky"):
            assert (
                f"{side}_hand_{finger}_proximal",
                f"{side}_hand_{finger}_middle",
            ) in pairs
        assert (
            f"{side}_hand_thumb_metacarpals",
            f"{side}_hand_thumb_proximal",
        ) in pairs
    assert root.find(".//mesh") is None
    assert all(
        j.find("parent").get("link").startswith(f"{side}_hand_")
        for j in root.findall("joint")
    )
    assert len(root.findall("joint/mimic")) == (0 if model == "l6" else 5)
    transform = np.asarray(config.handtracking_to_baselink_frame_transform).reshape(
        3, 3
    )
    np.testing.assert_allclose(transform.T @ transform, np.eye(3))
    assert np.linalg.det(transform) == pytest.approx(1.0)
    np.testing.assert_allclose(np.asarray((0, 1, 0)) @ transform, (-1, 0, 0))
    np.testing.assert_allclose(np.asarray((0, 0, -1)) @ transform, (0, 0, 1))


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_official_solver_round_trip_and_world_pose_invariance(model, side, caplog):
    config = _EXAMPLE.dex_hand_config(model, side)
    data = yaml.safe_load(Path(config.hand_retargeting_config).read_text())[
        "retargeting"
    ]
    specs = get_realhand_profile(model).joints(side)
    all_errors = []
    for rotated in (False, True):
        retargeter = DexHandRetargeter(
            _EXAMPLE.dex_hand_config(model, side), name=f"{model}_{side}_{rotated}"
        )
        rotation = (
            Rotation.from_euler("xyz", (0.4, -0.7, 1.2))
            if rotated
            else Rotation.identity()
        )
        translation = np.asarray((0.6, -0.3, 1.1)) if rotated else np.zeros(3)
        errors = []
        for fraction in (0.15, 0.35, 0.60, 0.80, 0.50, 0.20):
            active = np.asarray(
                [s.lower + fraction * (s.upper - s.lower) for s in specs]
            )
            qpos = _full_qpos(retargeter, active)
            sample, expected_vectors = _reachable_sample(
                retargeter, data, qpos, rotation, translation
            )
            result = _feed_samples(retargeter, side, [sample] * 15)[-1]
            actual_qpos = _full_qpos(retargeter, result)
            origins = _robot_positions(
                retargeter, actual_qpos, data["target_origin_link_names"]
            )
            tasks = _robot_positions(
                retargeter, actual_qpos, data["target_task_link_names"]
            )
            error = np.linalg.norm(tasks - origins - expected_vectors, axis=1)
            # This checks reachable robot geometry, not measured glove accuracy.
            assert np.max(error) < 0.005, (model, side, fraction, error)
            assert np.mean(error) < 0.002
            errors.append(error)
        all_errors.append(errors)
    np.testing.assert_allclose(all_errors[0], all_errors[1], atol=0.001)
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_official_filter_reduces_stationary_noise_without_changing_solver(model, side):
    filtered = DexHandRetargeter(
        _EXAMPLE.dex_hand_config(model, side), name=f"filtered_{model}_{side}"
    )
    direct = DexHandRetargeter(
        _EXAMPLE.dex_hand_config(model, side), name=f"direct_{model}_{side}"
    )
    direct._dex_hand.filter.alpha = 1.0
    data = yaml.safe_load(
        Path(_EXAMPLE.dex_hand_config(model, side).hand_retargeting_config).read_text()
    )["retargeting"]
    specs = get_realhand_profile(model).joints(side)
    active = np.asarray([s.lower + 0.45 * (s.upper - s.lower) for s in specs])
    sample, _ = _reachable_sample(
        filtered, data, _full_qpos(filtered, active), Rotation.identity(), np.zeros(3)
    )
    rng = np.random.default_rng(73)
    samples = [sample] * 35
    for _ in range(100):
        noisy = deepcopy(sample)
        for index, joint in enumerate(noisy["joints"]):
            if index != HandJointIndex.WRIST:
                joint["position"] = joint["position"] + rng.normal(0, 0.0005, 3)
        samples.append(noisy)
    raw = np.asarray(_feed_samples(direct, side, samples))
    smooth = np.asarray(_feed_samples(filtered, side, samples))
    expected = raw[0].copy()
    for target, actual in zip(raw[1:], smooth[1:]):
        expected += 0.4 * (target - expected)
        np.testing.assert_allclose(actual, expected, atol=2e-5)
    raw_step = np.sqrt(np.mean(np.diff(raw[55:], axis=0) ** 2))
    smooth_step = np.sqrt(np.mean(np.diff(smooth[55:], axis=0) ** 2))
    assert smooth_step < raw_step * 0.65


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_official_solver_step_response_and_static_hold(model, side):
    retargeter = DexHandRetargeter(
        _EXAMPLE.dex_hand_config(model, side), name=f"step_{model}_{side}"
    )
    data = yaml.safe_load(
        Path(_EXAMPLE.dex_hand_config(model, side).hand_retargeting_config).read_text()
    )["retargeting"]
    specs = get_realhand_profile(model).joints(side)
    endpoints = []
    for fraction in (0.2, 0.7):
        active = np.asarray([s.lower + fraction * (s.upper - s.lower) for s in specs])
        endpoints.append(
            _reachable_sample(
                retargeter,
                data,
                _full_qpos(retargeter, active),
                Rotation.identity(),
                np.zeros(3),
            )
        )
    _feed_samples(retargeter, side, [endpoints[0][0]] * 35)
    outputs = _feed_samples(retargeter, side, [endpoints[1][0]] * 35)
    errors = []
    for output in outputs:
        qpos = _full_qpos(retargeter, output)
        vectors = _robot_positions(
            retargeter, qpos, data["target_task_link_names"]
        ) - _robot_positions(retargeter, qpos, data["target_origin_link_names"])
        errors.append(np.mean(np.linalg.norm(vectors - endpoints[1][1], axis=1)))
    # Frame bounds, not a latency claim: the acquisition rate determines milliseconds.
    assert errors[9] < 0.003
    assert errors[-1] < 0.002
    assert np.max(np.ptp(np.asarray(outputs[-10:]), axis=0)) < np.deg2rad(1)


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_smooth_reachable_curl_sweep_respects_limits(model, side):
    retargeter = DexHandRetargeter(
        _EXAMPLE.dex_hand_config(model, side), name=f"sweep_{model}_{side}"
    )
    data = yaml.safe_load(
        Path(_EXAMPLE.dex_hand_config(model, side).hand_retargeting_config).read_text()
    )["retargeting"]
    specs = get_realhand_profile(model).joints(side)
    fractions = np.concatenate(
        (np.linspace(0.15, 0.85, 40), np.linspace(0.85, 0.15, 40))
    )
    samples = []
    for fraction in fractions:
        active = np.asarray([s.lower + fraction * (s.upper - s.lower) for s in specs])
        sample, _ = _reachable_sample(
            retargeter,
            data,
            _full_qpos(retargeter, active),
            Rotation.identity(),
            np.zeros(3),
        )
        samples.append(sample)
    _feed_samples(retargeter, side, [samples[0]] * 35)
    outputs = np.asarray(_feed_samples(retargeter, side, samples))
    assert np.all(outputs >= np.asarray([s.lower for s in specs]) - 1.01e-3)
    assert np.all(outputs <= np.asarray([s.upper for s in specs]) + 1.01e-3)
    assert np.max(np.abs(np.diff(outputs, axis=0))) < np.deg2rad(15)
    assert np.max(np.ptp(outputs, axis=0)) > 0.5


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_official_solver_invalid_wrist_returns_safe_zero(model, side, skeleton_samples):
    from isaaccapture.retargeting_engine.interface import (
        ComputeContext,
        ExecutionEvents,
        ExecutionState,
        OptionalTensorGroup,
        TensorGroup,
    )
    from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
        GraphTime,
    )
    from isaaccapture.retargeting_engine.tensor_types import HandInputIndex

    retargeter = DexHandRetargeter(
        _EXAMPLE.dex_hand_config(model, side), name=f"invalid_{model}_{side}"
    )
    _feed_samples(retargeter, side, skeleton_samples[side])
    name, spec = next(iter(retargeter.input_spec().items()))
    inputs = {name: OptionalTensorGroup(spec)}
    outputs = {
        key: TensorGroup(value) for key, value in retargeter.output_spec().items()
    }
    context = ComputeContext(
        graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
        execution_events=ExecutionEvents(
            reset=False, execution_state=ExecutionState.RUNNING
        ),
    )
    retargeter.compute(inputs, outputs, context)
    np.testing.assert_allclose([float(v) for v in outputs["hand_joints"]], 0)
    joints = skeleton_samples[side][0]["joints"]
    inputs[name][HandInputIndex.JOINT_POSITIONS] = np.asarray(
        [j["position"] for j in joints], dtype=np.float32
    )
    inputs[name][HandInputIndex.JOINT_ORIENTATIONS] = np.asarray(
        [j["orientation"] for j in joints], dtype=np.float32
    )
    valid = np.ones(26, dtype=np.uint8)
    valid[HandJointIndex.WRIST] = 0
    inputs[name][HandInputIndex.JOINT_VALID] = valid
    retargeter.compute(inputs, outputs, context)
    np.testing.assert_allclose([float(v) for v in outputs["hand_joints"]], 0)


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize(
    ("subdivisions", "noise"), ((1, 0), (2, 0), (4, 0), (2, 0.00015))
)
def test_l20_slow_calibrated_thumb_sweeps_without_filter(
    side, subdivisions, noise, calibrated_skeleton_samples
):
    source = calibrated_skeleton_samples.get("sweeps", {}).get(side)
    if source is None:
        pytest.skip(f"No {side} human calibration supplied")
    groups = {}
    for sample in source:
        name, _ = sample["name"].rsplit("_", 1)
        groups.setdefault(name, []).append(sample)
    specs = get_realhand_profile("l20").joints(side)
    lower = np.asarray([s.lower for s in specs]) - 1.01e-3
    upper = np.asarray([s.upper for s in specs]) + 1.01e-3
    rng = np.random.default_rng(116)
    for name, anchors in groups.items():
        retargeter = DexHandRetargeter(
            _EXAMPLE.dex_hand_config("l20", side), name=f"slow_{side}_{name}"
        )
        # Reachable robot targets and output smoothing can hide human-space branch changes.
        retargeter._dex_hand.filter.alpha = 1.0
        samples = []
        for first, second in zip(anchors, anchors[1:]):
            for alpha in np.linspace(0, 1, subdivisions, endpoint=False):
                joints = []
                for index, (a, b) in enumerate(zip(first["joints"], second["joints"])):
                    position = (1 - alpha) * np.asarray(
                        a["position"]
                    ) + alpha * np.asarray(b["position"])
                    if index != HandJointIndex.WRIST:
                        position += rng.normal(0, noise, 3)
                    joints.append({**a, "position": position, "radius": 0.008})
                samples.append({"joints": joints})
        final = deepcopy(anchors[-1])
        for joint in final["joints"]:
            joint["radius"] = 0.008
        samples.append(final)
        samples += samples[::-1]
        _feed_samples(retargeter, side, [samples[0]] * 35)
        outputs = np.asarray(_feed_samples(retargeter, side, samples))
        assert np.all(outputs >= lower)
        assert np.all(outputs <= upper)
        mask = np.asarray(
            ["thumb" in joint for joint in retargeter._config.hand_joint_names]
        )
        steps = np.abs(np.diff(outputs[:, mask], axis=0))
        assert np.max(steps) < np.deg2rad(8), (
            side,
            name,
            subdivisions,
            np.rad2deg(np.max(steps)),
        )
