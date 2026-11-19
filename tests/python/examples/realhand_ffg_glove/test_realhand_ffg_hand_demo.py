# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the PC calibration state machine without Isaac Sim or a connected glove."""

import importlib
import subprocess
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


def load_example(name):
    return importlib.import_module(f"isaaccapture_examples.realhand_ffg_glove.{name}")


_BRIDGE = load_example("realhand_ffg_hand_demo_bridge")
_VIEWER = load_example("realhand_ffg_hand_demo")
_THUMB = load_example("_thumb_calibration")
_CALIBRATION = load_example("realhand_ffg_glove_calibration")


@pytest.mark.parametrize("ports", (None, "/dev/ttyUSB0", "/dev/ttyUSB0,/dev/ttyUSB1"))
def test_demo_forwards_optional_ports_without_an_implicit_single_glove(tmp_path, ports):
    interpreter = tmp_path / "python"
    interpreter.touch()
    argv = [
        "--capture-python",
        str(interpreter),
        "--capture-package",
        str(tmp_path),
        "--plugin-path",
        str(tmp_path),
    ]
    if ports is not None:
        argv.extend(("--ports", ports))
    args = _VIEWER.parse_args(argv)
    assert args.ports == ports
    worker = _VIEWER.worker_command(args, tmp_path)
    native = _BRIDGE.plugin_arguments(args.ports, args.calibration)
    assert "--wrist-source=auto" in native
    assert "--wrist-source=controller" not in native
    if ports is None:
        assert "--ports" not in worker
        assert not any(value.startswith("--ports=") for value in native)
    else:
        assert worker[worker.index("--ports") + 1] == ports
        assert f"--ports={ports}" in native


def test_demo_worker_preserves_calibration_and_runtime_flags(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.touch()
    calibration = tmp_path / "human.yml"
    calibration.touch()
    args = _VIEWER.parse_args(
        [
            "--capture-python",
            str(interpreter),
            "--capture-package",
            str(tmp_path),
            "--plugin-path",
            str(tmp_path),
            "--calibration",
            str(calibration),
            "--no-launch-cloudxr-runtime",
        ]
    )
    command = _VIEWER.worker_command(args, tmp_path)
    assert command[1:3] == [
        "-m",
        "isaaccapture_examples.realhand_ffg_glove.realhand_ffg_hand_demo_bridge",
    ]
    assert command[command.index("--calibration") + 1] == str(calibration)
    assert "--no-launch-cloudxr-runtime" in command
    assert f"--calibration={calibration.resolve()}" in _BRIDGE.plugin_arguments(
        None, calibration
    )


@pytest.fixture(scope="module")
def fitted_thumb():
    poses = calibration_pose_samples()
    poses["thumb_mapping"] = _THUMB.fit_thumb_mapping(poses)
    return poses


@pytest.fixture(scope="module")
def fitted_native_samples(request, tmp_path_factory, fitted_thumb):
    executable = request.config.getoption("--ffg-skeleton-executable")
    if executable is None:
        pytest.skip("Pass --ffg-skeleton-executable for native thumb-map verification")
    directory = tmp_path_factory.mktemp("fitted_thumb")
    calibration = directory / "calibration.yml"
    calibration.write_text(
        yaml.safe_dump(
            {
                "format": "realhand-ffg-glove-human-calibration-v1",
                "hands": dict.fromkeys(("left", "right"), fitted_thumb),
            }
        ),
        encoding="utf-8",
    )
    output = directory / "samples.yml"
    subprocess.run(
        [
            executable,
            "--write-calibrated-skeleton-fixtures",
            str(calibration),
            str(output),
        ],
        check=True,
    )
    return yaml.safe_load(output.read_text(encoding="utf-8"))


def test_thumb_contacts_improve_without_losing_open_and_curl(fitted_thumb):
    mapping = fitted_thumb["thumb_mapping"]
    assert mapping["base_position"] == pytest.approx(_THUMB.BASE)
    assert mapping["fit_model"] == _THUMB.FIT_MODEL
    assert mapping["fit_report"]["training_only"]
    report = mapping["fit_report"]
    assert max(report["contact_surface_error_mm"]) < 6
    assert (
        np.mean(report["contact_surface_error_mm"])
        < np.mean(report["contact_surface_error_before_mm"]) / 3
    )
    assert _THUMB.mapped_angles(mapping, fitted_thumb["open"]) == pytest.approx(
        _THUMB.OPEN_ANGLES
    )
    full = _THUMB.mapped_angles(mapping, fitted_thumb["thumb_curl"])
    assert full[2] > 0.9
    assert full[3] > 1.0
    opening = np.asarray(fitted_thumb["open"])
    slight_curl = opening + 0.1 * _THUMB.angle_delta(
        fitted_thumb["thumb_curl"], opening
    )
    assert (
        np.linalg.norm(_THUMB.mapped_angles(mapping, slight_curl) - _THUMB.OPEN_ANGLES)
        < 0.4
    )


def test_quadratic_sensor_jacobian_matches_finite_differences():
    values = np.random.default_rng(44).uniform(-1, 1, (9, 4))
    actual = _THUMB.quadratic_feature_jacobian(values)
    expected = np.empty_like(actual)
    for sensor in range(4):
        step = np.eye(4)[sensor] * 1e-6
        expected[..., sensor] = (
            _THUMB.quadratic_features(values + step)
            - _THUMB.quadratic_features(values - step)
        ) / 2e-6
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_contact_quality_is_not_traded_away_for_strong_smoothness_priors(monkeypatch):
    poses = calibration_pose_samples()
    monkeypatch.setattr(_THUMB, "SENSOR_GAIN_SCALE", 0.5)
    mapping = _THUMB.fit_thumb_mapping(poses)
    assert mapping["fit_report"]["solver"] == "noise_regularized_contact_surface"
    assert max(mapping["fit_report"]["contact_surface_error_mm"]) < 6
    assert mapping["base_position"] == pytest.approx(_THUMB.BASE)
    assert _THUMB.mapped_angles(mapping, poses["open"]) == pytest.approx(
        _THUMB.OPEN_ANGLES
    )


def test_failed_constrained_fit_reports_side_without_accepting_it(monkeypatch):
    poses = calibration_pose_samples()
    monkeypatch.setattr(_THUMB, "SENSOR_GAIN_SCALE", 0.5)
    invalid = np.r_[np.zeros(56), 1.0]
    monkeypatch.setattr(
        _THUMB,
        "least_squares",
        lambda *args, **kwargs: SimpleNamespace(
            x=invalid, success=False, message="No fit"
        ),
    )
    monkeypatch.setattr(
        _THUMB,
        "minimize",
        lambda objective, initial, **kwargs: SimpleNamespace(
            x=initial, success=False, message="No feasible contact fit"
        ),
    )
    with pytest.raises(_THUMB.ThumbFitError, match=r"RIGHT.*quality checks") as caught:
        _THUMB.fit_thumb_mapping(poses, "right")
    assert caught.value.report["side"] == "right"
    assert caught.value.report["solver_message"] == "No feasible contact fit"


def test_repeated_open_is_an_additional_training_observation(fitted_thumb):
    poses = deepcopy(fitted_thumb)
    repeated = np.asarray(poses["open"]).copy()
    repeated[list(_THUMB.CHANNELS)] += (0.01, 0.025, 0.015, 0.02)
    poses["open_repeat"] = repeated.tolist()
    assert _THUMB.calibration_fingerprint(poses) != _THUMB.calibration_fingerprint(
        fitted_thumb
    )
    mapping = _THUMB.fit_thumb_mapping(poses)
    assert mapping["fit_report"]["repeat_open_training_tip_mm"] < 15
    assert max(mapping["fit_report"]["contact_surface_error_mm"]) < 6
    assert mapping["fit_report"]["training_only"]


def test_malformed_repeated_open_is_rejected(fitted_thumb):
    poses = dict(fitted_thumb, open_repeat=[0.0] * 20)
    with pytest.raises(ValueError, match="21 finite"):
        _THUMB.fit_thumb_mapping(poses)


@pytest.mark.parametrize("scale", (0.85, 1.0, 1.15))
def test_opposed_thumb_preserves_endpoints_and_bone_lengths_throughout_joint_range(
    scale,
):
    angles = np.random.default_rng(29).uniform(_THUMB.LOWER, _THUMB.UPPER, (1000, 4))
    angles = np.vstack((angles, _THUMB.LOWER, _THUMB.UPPER, _THUMB.OPEN_ANGLES))
    lengths = _THUMB.LENGTHS * scale
    points = np.asarray([_THUMB.thumb_joints(a, lengths=lengths) for a in angles])
    np.testing.assert_allclose(
        points[:, -1], _THUMB.thumb_tip_positions(angles, lengths), atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(points, axis=1), axis=2),
        np.broadcast_to(lengths, (len(points), 3)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "mcp,ip", ((0.2, 0.3), (0.9, 0.0), (0.0, 0.6), (1.1, 1.2), (1.8, 1.8))
)
def test_thumb_bone_posture_uses_its_chord_and_radial_plane(mcp, ip):
    for a, b in np.random.default_rng(30).uniform(
        _THUMB.LOWER[:2], _THUMB.UPPER[:2], (20, 2)
    ):
        points = _THUMB.thumb_joints((a, b, mcp, ip))
        bones = np.diff(points, axis=0)
        normal = np.cross(bones[0], bones[1]) + np.cross(bones[1], bones[2])
        along = points[-1] - points[0]
        along /= np.linalg.norm(along)
        radial = np.array((1.0, 0, 0)) - along[0] * along
        radial /= np.linalg.norm(radial)
        expected = np.cross(radial, along)
        np.testing.assert_allclose(
            normal / np.linalg.norm(normal), expected, atol=1e-12
        )
        for point in points[1:3]:
            v = point - points[0]
            bow = v - np.dot(v, along) * along
            assert np.dot(bow, radial) > 0
            np.testing.assert_allclose(bow / np.linalg.norm(bow), radial, atol=1e-12)


def test_opposed_thumb_preserves_straight_pose_and_is_continuous_near_zero_curl():
    for a in np.linspace(_THUMB.LOWER[0], _THUMB.UPPER[0], 20):
        rotation = _THUMB._rotation(a, 1) @ _THUMB._rotation(0.25, 2)
        straight = _THUMB.BASE + np.r_[0, np.cumsum(_THUMB.LENGTHS)][:, None] * (
            rotation @ (0, 0, -1)
        )
        np.testing.assert_allclose(
            _THUMB.thumb_joints((a, 0.25, 0, 0)), straight, atol=1e-12
        )
        for epsilon in (1e-3, 1e-5, 1e-8):
            curled = _THUMB.thumb_joints((a, 0.25, epsilon, epsilon))
            assert np.max(np.linalg.norm(curled - straight, axis=1)) < epsilon * 0.3


@pytest.mark.parametrize("mcp,ip", ((73, 13), (83, 0), (94, 0), (100, 1.4)))
def test_mcp_heavy_thumb_flexion_is_shared_without_moving_endpoint(mcp, ip):
    angles = np.r_[0.2, -0.4, np.deg2rad((mcp, ip))]
    points = _THUMB.thumb_joints(angles)
    bones = np.diff(points, axis=0)
    turns = np.arccos(
        np.clip(
            np.sum(bones[:-1] * bones[1:], axis=1)
            / np.linalg.norm(bones[:-1], axis=1)
            / np.linalg.norm(bones[1:], axis=1),
            -1,
            1,
        )
    )
    assert np.rad2deg(turns[0]) < 65
    assert 20 < np.rad2deg(turns[1]) < 65
    root = bones[0] / np.linalg.norm(bones[0])
    assert abs(root[1]) < np.sin(np.deg2rad(60))
    np.testing.assert_allclose(
        points[-1], _THUMB.thumb_tip_positions(angles[None])[0], atol=1e-12
    )


def test_thumb_posture_regularization_preserves_ip_only_motion_and_stays_continuous():
    for ip in np.linspace(0, 1.8, 50):
        assert _THUMB.regularized_thumb_flexion(0, ip) == (0, ip)
    for ip in (0.0, 0.5, 1.8):
        points = np.array(
            [
                _THUMB.thumb_joints((0.2, -0.4, mcp, ip))
                for mcp in np.linspace(0, 1.8, 1801)
            ]
        )
        assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=-1)) < 0.0002


def test_sensor_gain_constraint_invalidates_cached_coefficients(
    monkeypatch, fitted_thumb
):
    before = _THUMB.calibration_fingerprint(fitted_thumb)
    monkeypatch.setattr(_THUMB, "SENSOR_GAIN_SCALE", _THUMB.SENSOR_GAIN_SCALE * 2)
    assert _THUMB.calibration_fingerprint(fitted_thumb) != before


def test_indistinguishable_thumb_contacts_are_rejected():
    poses = calibration_pose_samples()
    for key in _THUMB.POSES[4:]:
        poses[key][:5] = poses["thumb_index"][:5]
    with pytest.raises(ValueError, match="indistinguishable"):
        _THUMB.fit_thumb_mapping(poses)


def test_invalid_saved_calibration_is_not_rewritten(human_calibration):
    calibration = yaml.safe_load(human_calibration.read_text())
    calibration["hands"]["left"]["fist"] = [0.0] * 21
    human_calibration.write_text(yaml.safe_dump(calibration), encoding="utf-8")
    before = human_calibration.read_bytes()
    with pytest.raises(ValueError, match="insufficient"):
        _BRIDGE.validate_saved_calibration(human_calibration, _CALIBRATION)
    assert human_calibration.read_bytes() == before


@pytest.mark.parametrize("boundary,direction", (("input_min", -1), ("input_max", 1)))
def test_thumb_saturates_beyond_captured_ranges(fitted_thumb, boundary, direction):
    mapping = fitted_thumb["thumb_mapping"]
    endpoint = np.asarray(fitted_thumb["open"]).copy()
    endpoint[list(_THUMB.CHANNELS)] = (
        np.asarray(mapping["input_origin"]) + mapping[boundary]
    )
    overshoot = endpoint.copy()
    overshoot[list(_THUMB.CHANNELS)] += direction * 0.15
    assert _THUMB.mapped_angles(mapping, overshoot) == pytest.approx(
        _THUMB.mapped_angles(mapping, endpoint)
    )


def test_preview_preserves_saved_calibration_without_fitting(
    human_calibration, fitted_thumb, monkeypatch
):
    calibration = yaml.safe_load(human_calibration.read_text())
    calibration["hands"]["left"] = fitted_thumb
    human_calibration.write_text(yaml.safe_dump(calibration), encoding="utf-8")
    before = human_calibration.read_bytes()

    def unexpected_fit(*_):
        pytest.fail("Loading a calibration must not refit it")

    monkeypatch.setattr(_THUMB, "fit_thumb_mapping", unexpected_fit)
    for _ in range(2):
        assert (
            _BRIDGE.validate_saved_calibration(human_calibration, _CALIBRATION)
            == calibration
        )
    assert human_calibration.read_bytes() == before
    assert list(human_calibration.parent.iterdir()) == [human_calibration]


@pytest.mark.parametrize("side, mirror", (("left", 1), ("right", -1)))
def test_native_thumb_matches_fitter_for_every_pose_and_trajectory(
    fitted_thumb, fitted_native_samples, side, mirror
):
    mapping = fitted_thumb["thumb_mapping"]
    for sample in fitted_native_samples["sweeps"][side]:
        expected = _THUMB.thumb_joints(
            _THUMB.mapped_angles(mapping, sample["raw"]),
            mapping["base_position"],
            mapping["lengths"],
        )
        expected[:, 0] *= mirror
        points = np.asarray([joint["position"] for joint in sample["joints"]])
        assert points[2:6] == pytest.approx(expected, abs=2e-6)
        quaternions = np.asarray([joint["orientation"] for joint in sample["joints"]])
        assert np.linalg.norm(quaternions, axis=1) == pytest.approx(
            np.ones(26), abs=2e-6
        )
    samples = fitted_native_samples["sweeps"][side]
    for pose in range(7):
        tips = np.asarray(
            [
                sample["joints"][5]["position"]
                for sample in samples[pose * 101 : (pose + 1) * 101]
            ]
        )
        assert np.max(np.linalg.norm(np.diff(tips, axis=0), axis=1)) < 0.003


def test_multivariate_mapping_accepts_a_small_legacy_opposition_span(fitted_thumb):
    poses = deepcopy(fitted_thumb)
    poses["thumb_opposition"][0] = poses["open"][0] + 0.049
    example = _BRIDGE.load_calibration_example()
    example._validate_calibration({"hands": {"left": poses}}, ("left",))
    del poses["thumb_mapping"]
    with pytest.raises(ValueError, match="insufficient calibration motion"):
        example._validate_calibration({"hands": {"left": poses}}, ("left",))


def test_multivariate_mapping_still_rejects_motionless_inputs(fitted_thumb):
    poses = deepcopy(fitted_thumb)
    poses["thumb_mapping"]["input_min"] = [0.0] * 4
    poses["thumb_mapping"]["input_max"] = [0.0] * 4
    with pytest.raises(ValueError, match="invalid thumb mapping input ranges"):
        _BRIDGE.load_calibration_example()._validate_calibration(
            {"hands": {"left": poses}}, ("left",)
        )


@pytest.mark.parametrize("side", ("left", "right"))
def test_native_other_fingers_cannot_move_thumb_and_phase_wrap_is_safe(
    fitted_native_samples, side
):
    poses = fitted_native_samples["poses"][side]
    for group in ("independence", "phase_wrap"):
        for index, sample in enumerate(fitted_native_samples[group][side]):
            expected = np.asarray(
                [joint["position"] for joint in poses[index // 2]["joints"]]
            )
            points = np.asarray([joint["position"] for joint in sample["joints"]])
            assert points[2:6] == pytest.approx(expected[2:6], abs=2e-6)


def test_native_right_sway_and_offline_finger_geometry_use_the_same_sensor_convention(
    fitted_thumb, fitted_native_samples
):
    for finger in range(4):
        for direction in range(2):
            index = 2 * finger + direction
            left = fitted_native_samples["lateral"]["left"][index]
            right = fitted_native_samples["lateral"]["right"][index]
            left_points = np.asarray([joint["position"] for joint in left["joints"]])
            right_points = np.asarray([joint["position"] for joint in right["joints"]])
            assert right_points[:, 0] == pytest.approx(-left_points[:, 0], abs=2e-6)
            assert right_points[:, 1:] == pytest.approx(left_points[:, 1:], abs=2e-6)
            for side, sample, points in (
                ("left", left, left_points),
                ("right", right, right_points),
            ):
                expected = _THUMB.finger_tip(fitted_thumb, sample["raw"], finger, side)
                expected[0] *= 1 if side == "left" else -1
                assert points[10 + 5 * finger] == pytest.approx(expected, abs=2e-6)


def test_explicit_right_calibration_preserves_left(human_calibration, fitted_thumb):
    calibration = _CALIBRATION._load_output(human_calibration)
    before = deepcopy(calibration["hands"]["left"])
    saved = _CALIBRATION._fit_and_save(human_calibration, {"right": fitted_thumb})
    assert saved["hands"]["left"] == before
    mapping = saved["hands"]["right"]["thumb_mapping"]
    assert mapping["calibration_sha256"] == _THUMB.calibration_fingerprint(
        fitted_thumb, "right"
    )
    assert (
        mapping["calibration_sha256"]
        != fitted_thumb["thumb_mapping"]["calibration_sha256"]
    )


@pytest.mark.parametrize(
    "field, bad",
    (
        ("input_scale", [0] * 4),
        ("coefficients", [[0] * 15] * 3),
        ("base_position", [1] * 3),
        ("format", "unknown"),
    ),
)
def test_native_rejects_malformed_thumb_mapping(
    request, tmp_path, fitted_thumb, field, bad
):
    executable = request.config.getoption("--ffg-skeleton-executable")
    if executable is None:
        pytest.skip("Pass --ffg-skeleton-executable")
    poses = dict(
        fitted_thumb, thumb_mapping=dict(fitted_thumb["thumb_mapping"], **{field: bad})
    )
    path = tmp_path / "invalid.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "format": "realhand-ffg-glove-human-calibration-v1",
                "hands": {"left": poses},
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            executable,
            "--write-calibrated-skeleton-fixtures",
            str(path),
            str(tmp_path / "out.yml"),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0


def calibration_pose_samples():
    poses = {key: np.zeros(21) for key in _THUMB.POSES}
    for finger in range(4):
        poses["fist"][6 + 4 * finger] = -1.2
        poses["fist"][8 + 4 * finger] = 0.9
    poses["fist"][[0, 1, 2, 4]] = (-0.04, -0.3, -0.5, 0.1)
    poses["thumb_curl"][[0, 1, 2, 4]] = (-0.1, -0.5, -1.6, 0.6)
    thumb_values = (
        (-0.12, -0.85, -0.9, 0.4),
        (-0.18, -0.98, -0.9, 0.4),
        (-0.22, -0.97, -1, 0.4),
        (-0.25, -0.99, -1.3, 0.46),
    )
    for finger, key in enumerate(_THUMB.POSES[3:]):
        poses[key][list(_THUMB.CHANNELS)] = thumb_values[finger]
        poses[key][6 + 4 * finger] = -0.8
        poses[key][8 + 4 * finger] = 0.6
    poses["thumb_opposition"] = np.zeros(21)
    poses["thumb_opposition"][[0, 1]] = (-0.25, -0.99)
    poses["spread"] = np.zeros(21)
    poses["spread"][[5, 9, 13, 17]] = 0.3
    for finger in _CALIBRATION._CONTACTS:
        poses[f"repeat_thumb_{finger}"] = poses[f"thumb_{finger}"].copy()
    poses["open_repeat"] = poses["open"].copy()
    return {key: value.tolist() for key, value in poses.items()}


def capture_step(capture, values, connected, now):
    capture.command({"action": "capture"}, connected, now)
    assert capture.state == "preparing"
    assert not capture.update(values, now + capture.prepare_seconds - 0.1)
    assert not any(capture.samples.values())
    return [
        capture.update(values, now + capture.prepare_seconds + elapsed)
        for elapsed in np.linspace(0, capture.hold_seconds, capture.sample_count + 1)
    ]


def capture_all(capture, *, connected=None, identical=False, sides=("left",)):
    connected = connected or {"left": True, "right": True}
    capture.command({"action": "begin", "sides": list(sides)}, connected, 0.0)
    reloads = []
    poses = calibration_pose_samples()
    for step, (_, key) in enumerate(capture.poses):
        now = step * (capture.prepare_seconds + capture.hold_seconds + 1)
        value = np.zeros(21) if identical else poses[key]
        reloads.extend(
            capture_step(capture, dict.fromkeys(sides, value), connected, now)
        )
    return reloads


def test_ui_uses_full_shared_calibration_and_preserves_other_side(human_calibration):
    before = yaml.safe_load(human_calibration.read_text())["hands"]["right"]
    capture = _BRIDGE.CalibrationCapture(
        human_calibration, _BRIDGE.load_calibration_example()
    )
    reloads = capture_all(capture)
    assert sum(reloads) == 1
    assert capture.state == "complete"
    data = yaml.safe_load(human_calibration.read_text())
    assert data["hands"]["right"] == before
    assert set(data["hands"]["left"]) == {key for _, key in capture.poses} | {
        "thumb_mapping",
        "thumb_observations",
    }
    assert capture.status(100)["total"] == 14
    assert len(data["hands"]["left"]["thumb_observations"]) == 12 * 16
    assert data["hands"]["left"]["spread"] == calibration_pose_samples()["spread"]


def test_invalid_captures_do_not_overwrite_working_calibration(human_calibration):
    before = human_calibration.read_bytes()
    capture = _BRIDGE.CalibrationCapture(
        human_calibration, _BRIDGE.load_calibration_example()
    )
    assert not any(capture_all(capture, identical=True))
    assert capture.state == "error"
    assert capture.message
    assert human_calibration.read_bytes() == before


def test_disconnection_and_cancel_keep_existing_calibration(human_calibration):
    before = human_calibration.read_bytes()
    capture = _BRIDGE.CalibrationCapture(
        human_calibration, _BRIDGE.load_calibration_example()
    )
    connected = {"left": True, "right": False}
    capture.command({"action": "begin", "sides": ["right"]}, connected, 0.0)
    assert capture.state == "idle"
    capture.command({"action": "begin", "sides": ["left"]}, connected, 0.0)
    capture.command({"action": "capture"}, connected, 1.0)
    capture.update({"left": None}, 1.0 + capture.prepare_seconds)
    assert capture.state == "error"
    capture.command({"action": "cancel"}, connected, 3.0)
    assert capture.state == "idle"
    assert human_calibration.read_bytes() == before


def test_cached_raw_values_are_not_used_after_device_disconnect():
    report = SimpleNamespace(
        schema_version=1,
        report_time_ns=time.monotonic_ns(),
        devices=[
            SimpleNamespace(
                path="/glove/realhand_ffg_glove_left",
                state=SimpleNamespace(name="DISCONNECTED"),
                error="USB disconnected",
            ),
            SimpleNamespace(
                path="/glove/realhand_ffg_glove_right",
                state=SimpleNamespace(name="CONNECTED"),
                error="",
            ),
        ],
    )
    tracker = SimpleNamespace(get_device_status_snapshot=lambda session: report)
    session = SimpleNamespace(deviceio_session=object())
    example = SimpleNamespace(_sample_group=lambda result, name: np.ones(21))
    raw, errors = _BRIDGE.sample_glove_inputs(session, {}, example, tracker)
    assert raw["left"] is None
    assert raw["right"].shape == (21,)
    assert errors["left"] == "USB disconnected"
    report.report_time_ns -= 3_000_000_000
    raw, _ = _BRIDGE.sample_glove_inputs(session, {}, example, tracker)
    assert raw == {"left": None, "right": None}


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize(
    "reason",
    (
        "missing human calibration for this hand",
        "controller wrist unavailable or discontinuous; awaiting stable tracking",
    ),
)
def test_connected_glove_reports_unavailable_hand_stream(side, reason):
    report = SimpleNamespace(
        schema_version=1,
        report_time_ns=time.monotonic_ns(),
        devices=[
            SimpleNamespace(
                path=f"/glove/realhand_ffg_glove_{side}",
                state=SimpleNamespace(name="CONNECTED"),
                error="",
            ),
            SimpleNamespace(
                path=f"/hand/{side}",
                state=SimpleNamespace(name="DEGRADED"),
                error=reason,
            ),
        ],
    )
    tracker = SimpleNamespace(get_device_status_snapshot=lambda session: report)
    session = SimpleNamespace(deviceio_session=object())
    example = SimpleNamespace(_sample_group=lambda result, name: np.ones(21))
    raw, errors = _BRIDGE.sample_glove_inputs(session, {}, example, tracker)
    assert raw[side].shape == (21,)
    assert errors[side] == reason
    report.devices[1].state.name = "CONNECTED"
    report.devices[1].error = ""
    _, errors = _BRIDGE.sample_glove_inputs(session, {}, example, tracker)
    assert errors[side] == ""


@pytest.mark.parametrize("side, wrist_x", (("left", -0.15), ("right", 0.15)))
def test_viewer_is_a_rigid_axis_change_not_a_gesture_retarget(
    skeleton_samples, side, wrist_x
):
    points = np.asarray(
        [joint["position"] for joint in skeleton_samples[side][1]["joints"]]
    )
    before = points.copy()
    displayed = _VIEWER.display_positions(points, side)
    for first, second in _VIEWER._EDGES:
        assert np.linalg.norm(displayed[first] - displayed[second]) == pytest.approx(
            np.linalg.norm(points[first] - points[second])
        )
    assert displayed[1] == pytest.approx((wrist_x, -0.08, 0.20))
    assert points == pytest.approx(before)


@pytest.mark.parametrize("side", ("left", "right"))
def test_viewer_faces_palms_up_without_mirroring_or_reversing_fingers(side):
    points = np.zeros((26, 3))
    points[6] = (0, 0.03, 0)
    points[7] = (0.02, 0, 0)
    points[8] = (0, 0, -0.04)
    displayed = _VIEWER.display_positions(points, side)
    local = displayed - displayed[1]
    assert local[6] == pytest.approx((0, 0, -0.03))
    assert local[8] == pytest.approx((0, 0.04, 0))
    assert np.linalg.det(local[[7, 6, 8]]) == pytest.approx(
        np.linalg.det(points[[7, 6, 8]])
    )


@pytest.mark.parametrize("bad", (np.zeros((25, 3)), np.full((26, 3), np.nan)))
def test_viewer_rejects_invalid_geometry(bad):
    with pytest.raises(ValueError, match="26 finite"):
        _VIEWER.display_positions(bad, "left")


@pytest.mark.parametrize("side, mirror", (("left", 1), ("right", -1)))
@pytest.mark.parametrize("finger", range(4))
def test_palm_up_view_preserves_sideways_motion_relative_to_anatomy(
    skeleton_samples, side, mirror, finger
):
    points = np.asarray(
        [joint["position"] for joint in skeleton_samples[side][0]["joints"]]
    )
    before = _VIEWER.display_positions(points, side)
    first = 6 + 5 * finger
    for direction in (-1, 1):
        moved = points.copy()
        moved[first : first + 5, 0] -= mirror * direction * 0.01
        delta = _VIEWER.display_positions(moved, side) - before
        assert delta[first : first + 5, 0] == pytest.approx(
            [mirror * direction * 0.01] * 5
        )
        assert delta[:, 1:] == pytest.approx(np.zeros((26, 2)))
        assert delta[:first] == pytest.approx(np.zeros((first, 3)))
        assert delta[first + 5 :] == pytest.approx(np.zeros((26 - first - 5, 3)))


def test_pc_and_cli_share_pose_order_timing_and_sampling(tmp_path):
    capture = _BRIDGE.CalibrationCapture(tmp_path / "human.yml", _CALIBRATION)
    assert capture.poses is _CALIBRATION._CAPTURE_POSES
    assert capture.prepare_seconds == _CALIBRATION._DEFAULT_PREPARE_SECONDS == 8
    assert capture.hold_seconds == _CALIBRATION._DEFAULT_HOLD_SECONDS == 6
    assert capture.sample_count == _CALIBRATION._DEFAULT_SAMPLES == 120


def test_both_hands_complete_in_one_workflow(human_calibration):
    capture = _BRIDGE.CalibrationCapture(
        human_calibration, _BRIDGE.load_calibration_example()
    )
    assert sum(capture_all(capture, sides=("left", "right"))) == 1
    data = yaml.safe_load(human_calibration.read_text())
    for side in ("left", "right"):
        assert all(key in data["hands"][side] for _, key in capture.poses)


@pytest.mark.parametrize("bad", (np.zeros(20), np.full(21, np.nan)))
def test_invalid_live_samples_do_not_replace_calibration(human_calibration, bad):
    before = human_calibration.read_bytes()
    capture = _BRIDGE.CalibrationCapture(
        human_calibration, _BRIDGE.load_calibration_example()
    )
    capture.command({"action": "begin", "sides": ["left"]}, {"left": True}, 0)
    capture.command({"action": "capture"}, {"left": True}, 0)
    assert not capture.update({"left": bad}, capture.prepare_seconds)
    assert capture.state == "error"
    assert human_calibration.read_bytes() == before


def test_cancel_during_contacts_does_not_save_partial_calibration(human_calibration):
    before = human_calibration.read_bytes()
    capture = _BRIDGE.CalibrationCapture(
        human_calibration, _BRIDGE.load_calibration_example()
    )
    capture.command({"action": "begin", "sides": ["left"]}, {"left": True}, 0)
    for step in range(6):
        now = step * 20
        capture_step(capture, {"left": np.full(21, float(step))}, {"left": True}, now)
    assert capture.status(120)["instruction"].startswith("touch")
    capture.command({"action": "cancel"}, {}, 120)
    assert human_calibration.read_bytes() == before
    assert not human_calibration.with_suffix(".validation.yml").exists()


def test_pc_uses_cli_sample_reduction_and_save(tmp_path, monkeypatch):
    captured = []
    original = _CALIBRATION._store_capture

    def store(poses, key, samples):
        captured.append(key)
        original(poses, key, samples)

    saved = []
    monkeypatch.setattr(_CALIBRATION, "_store_capture", store)
    monkeypatch.setattr(
        _CALIBRATION, "_fit_and_save", lambda path, poses: saved.append(deepcopy(poses))
    )
    capture = _BRIDGE.CalibrationCapture(tmp_path / "human.yml", _CALIBRATION)
    assert sum(capture_all(capture)) == 1
    assert captured == [key for _, key in _CALIBRATION._CAPTURE_POSES]
    assert len(saved) == 1
    expected = {}
    values = calibration_pose_samples()
    for _, key in _CALIBRATION._CAPTURE_POSES:
        original(expected, key, np.tile(values[key], (121, 1)))
    assert saved[0]["left"] == expected


def test_preparation_can_be_cancelled_without_sampling_or_saving(human_calibration):
    before = human_calibration.read_bytes()
    capture = _BRIDGE.CalibrationCapture(human_calibration, _CALIBRATION)
    capture.command({"action": "begin", "sides": ["left"]}, {"left": True}, 0)
    capture.command({"action": "capture"}, {"left": True}, 0)
    assert capture.status(1)["state"] == "preparing"
    assert capture.status(1)["remaining"] == 7
    capture.command({"action": "cancel"}, {}, 2)
    assert not capture.update({"left": np.ones(21)}, 20)
    assert capture.state == "idle"
    assert human_calibration.read_bytes() == before


@pytest.mark.parametrize("side", ("left", "right"))
def test_solid_bones_connect_every_joint_for_every_pose(skeleton_samples, side):
    for pose in skeleton_samples[side]:
        points = _VIEWER.display_positions(
            [joint["position"] for joint in pose["joints"]], side
        )
        centers, vectors, lengths = _VIEWER.bone_segments(points)
        assert len(lengths) == 25
        assert np.all(lengths > 1e-6)
        for index, (first, second) in enumerate(_VIEWER._EDGES):
            assert centers[index] - vectors[index] * 0.5 == pytest.approx(points[first])
            assert centers[index] + vectors[index] * 0.5 == pytest.approx(
                points[second]
            )
            assert lengths[index] == pytest.approx(
                np.linalg.norm(points[first] - points[second])
            )
