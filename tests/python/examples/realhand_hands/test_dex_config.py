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
from isaaccapture_examples.realhand_hands._profiles import get_realhand_profile
from isaaccapture.retargeting_engine.tensor_types import HandJointIndex
from isaaccapture_examples.realhand_hands.hand_config import get_hand_config
from _samples import _feed_samples


def _kinematics(config):
    from _kinematics import hand_kinematics

    return hand_kinematics(config.hand_urdf, tuple(config.hand_joint_names))


def _robot_positions(config, qpos, names):
    return _kinematics(config).positions(qpos, names)


def _full_qpos(config, active):
    return _kinematics(config).full_qpos(active)


def _unfiltered_config(model, side, tmp_path):
    config = get_hand_config(model, side)
    data = yaml.safe_load(Path(config.hand_retargeting_config).read_text())
    data["retargeting"]["low_pass_alpha"] = 1.0
    path = tmp_path / f"{model}_{side}_unfiltered.yml"
    path.write_text(yaml.safe_dump(data))
    config.hand_retargeting_config = str(path)
    return config


def _reachable_sample(config, data, qpos, rotation, translation):
    origins = _robot_positions(config, qpos, data["target_origin_link_names"])
    tasks = _robot_positions(config, qpos, data["target_task_link_names"])
    points = np.zeros((21, 3))
    for indices, positions in zip(data["target_link_human_indices"], (origins, tasks)):
        for index, position in zip(indices, positions):
            points[index] = position
    transform = np.asarray(config.handtracking_to_baselink_frame_transform).reshape(
        3, 3
    )
    points = rotation.apply(points @ transform.T / data["scaling_factor"]) + translation
    joints = [
        {"position": translation, "orientation": rotation.as_quat(), "radius": 0.008}
        for _ in range(26)
    ]
    for index, point in zip(
        (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25),
        points,
    ):
        joints[index] = {
            "position": point,
            "orientation": rotation.as_quat(),
            "radius": 0.008,
        }
    return {"joints": joints}, tasks - origins


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_model_config_contains_only_robot_data(model, side):
    config = get_hand_config(model, side)
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
def test_mimic_relations_match_between_hands(model):
    relations = {}
    for side in ("left", "right"):
        root = ET.parse(get_hand_config(model, side).hand_urdf).getroot()
        relations[side] = {
            joint.get("name").removeprefix(f"{side}_hand_"): (
                mimic.get("joint").removeprefix(f"{side}_hand_"),
                float(mimic.get("multiplier", "1")),
                float(mimic.get("offset", "0")),
            )
            for joint in root.findall("joint")
            if (mimic := joint.find("mimic")) is not None
        }
    # A round-trip through one URDF cannot detect side-specific coupling errors.
    assert relations["left"] == relations["right"]


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_mimic_respects_limits_at_full_travel(model, side):
    root = ET.parse(get_hand_config(model, side).hand_urdf).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    # A solver can apply mimic coupling without clamping the follower's limits.
    for name, follower in joints.items():
        mimic = follower.find("mimic")
        if mimic is None:
            continue
        source_limit = joints[mimic.get("joint")].find("limit")
        follower_limit = follower.find("limit")
        for endpoint in ("lower", "upper"):
            angle = float(source_limit.get(endpoint)) * float(
                mimic.get("multiplier", "1")
            ) + float(mimic.get("offset", "0"))
            assert float(follower_limit.get("lower")) - 1e-12 <= angle, name
            assert angle <= float(follower_limit.get("upper")) + 1e-12, name


@pytest.mark.parametrize(
    "joint_name", ("index_pip", "middle_pip", "ring_pip", "pinky_pip", "thumb_mcp")
)
def test_l20_linked_joint_limits_match_between_hands(joint_name):
    limits = {}
    for side in ("left", "right"):
        root = ET.parse(get_hand_config("l20", side).hand_urdf).getroot()
        joint = root.find(f"joint[@name='{side}_hand_{joint_name}']")
        limit = joint.find("limit")
        limits[side] = tuple(float(limit.get(bound)) for bound in ("lower", "upper"))
    assert limits["left"] == limits["right"]


def test_o6_thumb_full_travel_is_mirrored():
    profile = get_realhand_profile("o6")
    configs = {side: get_hand_config("o6", side) for side in ("left", "right")}
    for yaw in np.linspace(0.0, 1.3, 7):
        for pitch in np.linspace(0.0, 0.58, 7):
            targets = {"thumb_cmc_yaw": yaw, "thumb_cmc_pitch": pitch}
            tips = {}
            for side, config in configs.items():
                active = [
                    targets.get(joint.short_name, 0.0) for joint in profile.joints(side)
                ]
                tips[side] = _robot_positions(
                    config, _full_qpos(config, active), [f"{side}_hand_thumb_tip"]
                )[0]
            # The packaged left/right fingertip markers differ by under 0.2 mm.
            np.testing.assert_allclose(
                tips["left"] * (1, -1, 1), tips["right"], atol=0.0002
            )


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_official_solver_round_trip_and_world_pose_invariance(model, side, caplog):
    config = get_hand_config(model, side)
    data = yaml.safe_load(Path(config.hand_retargeting_config).read_text())[
        "retargeting"
    ]
    specs = get_realhand_profile(model).joints(side)
    all_errors = []
    for rotated in (False, True):
        retargeter = DexHandRetargeter(
            get_hand_config(model, side), name=f"{model}_{side}_{rotated}"
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
            qpos = _full_qpos(config, active)
            sample, expected_vectors = _reachable_sample(
                config, data, qpos, rotation, translation
            )
            result = _feed_samples(retargeter, side, [sample] * 15)[-1]
            actual_qpos = _full_qpos(config, result)
            origins = _robot_positions(
                config, actual_qpos, data["target_origin_link_names"]
            )
            tasks = _robot_positions(
                config, actual_qpos, data["target_task_link_names"]
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
def test_official_filter_reduces_stationary_noise_without_changing_solver(
    model, side, tmp_path
):
    config = get_hand_config(model, side)
    filtered = DexHandRetargeter(
        get_hand_config(model, side), name=f"filtered_{model}_{side}"
    )
    direct = DexHandRetargeter(
        _unfiltered_config(model, side, tmp_path), name=f"direct_{model}_{side}"
    )
    data = yaml.safe_load(
        Path(get_hand_config(model, side).hand_retargeting_config).read_text()
    )["retargeting"]
    specs = get_realhand_profile(model).joints(side)
    active = np.asarray([s.lower + 0.45 * (s.upper - s.lower) for s in specs])
    sample, _ = _reachable_sample(
        config, data, _full_qpos(config, active), Rotation.identity(), np.zeros(3)
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
    config = get_hand_config(model, side)
    retargeter = DexHandRetargeter(
        get_hand_config(model, side), name=f"step_{model}_{side}"
    )
    data = yaml.safe_load(
        Path(get_hand_config(model, side).hand_retargeting_config).read_text()
    )["retargeting"]
    specs = get_realhand_profile(model).joints(side)
    endpoints = []
    for fraction in (0.2, 0.7):
        active = np.asarray([s.lower + fraction * (s.upper - s.lower) for s in specs])
        endpoints.append(
            _reachable_sample(
                config,
                data,
                _full_qpos(config, active),
                Rotation.identity(),
                np.zeros(3),
            )
        )
    _feed_samples(retargeter, side, [endpoints[0][0]] * 35)
    outputs = _feed_samples(retargeter, side, [endpoints[1][0]] * 35)
    errors = []
    for output in outputs:
        qpos = _full_qpos(config, output)
        vectors = _robot_positions(
            config, qpos, data["target_task_link_names"]
        ) - _robot_positions(config, qpos, data["target_origin_link_names"])
        errors.append(np.mean(np.linalg.norm(vectors - endpoints[1][1], axis=1)))
    # Frame bounds, not a latency claim: the acquisition rate determines milliseconds.
    assert errors[9] < 0.003
    assert errors[-1] < 0.002
    assert np.max(np.ptp(np.asarray(outputs[-10:]), axis=0)) < np.deg2rad(1)


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("side", ("left", "right"))
def test_smooth_reachable_curl_sweep_respects_limits(model, side):
    config = get_hand_config(model, side)
    retargeter = DexHandRetargeter(
        get_hand_config(model, side), name=f"sweep_{model}_{side}"
    )
    data = yaml.safe_load(
        Path(get_hand_config(model, side).hand_retargeting_config).read_text()
    )["retargeting"]
    specs = get_realhand_profile(model).joints(side)
    fractions = np.concatenate(
        (np.linspace(0.15, 0.85, 40), np.linspace(0.85, 0.15, 40))
    )
    samples = []
    for fraction in fractions:
        active = np.asarray([s.lower + fraction * (s.upper - s.lower) for s in specs])
        sample, _ = _reachable_sample(
            config,
            data,
            _full_qpos(config, active),
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
        get_hand_config(model, side), name=f"invalid_{model}_{side}"
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
