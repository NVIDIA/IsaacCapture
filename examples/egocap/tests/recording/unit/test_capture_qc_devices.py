"""Device-specific capture checks (Avatar hand liveness, Pico body joints, camera parity) and the advisories.

Explicit arrays with known answers, written to real Zarr groups and read back by the production checker. The
measurement functions are also exercised directly for their edge cases.
"""

import numpy as np
import pytest
import zarr

from egocap.recording.capture_measure import (
    pico_body_problem,
    stream_liveness,
    total_frame_drops,
    windowed_frame_drops,
)
from egocap.recording.capture_qc import (
    CaptureQcConfig,
    PicoBodyConfig,
    QcDecision,
    check_capture_quality,
    decide,
)
from egocap.recording.dataset_check import Outcome
from egocap.recording.episode_store import FieldSpec, NumericSourceSpec, VideoSourceSpec

ROWS = 20
AVATAR = NumericSourceSpec(
    "glove",
    tuple(
        FieldSpec(name, "float64", (25, 4, 4))
        for name in (
            "left_hand_keypoints",
            "left_hand_joints_robot",
            "right_hand_keypoints",
            "right_hand_joints_robot",
        )
    ),
)
PICO = NumericSourceSpec(
    "pico",
    (FieldSpec("body_joints_pose", "float32", (24, 4, 4)), FieldSpec("body_joints_valid", "bool")),
)
CAMERAS = (VideoSourceSpec("cam_a", 64, 48, 30.0), VideoSourceSpec("cam_b", 64, 48, 30.0))


def _moving(rows=ROWS, shape=(25, 4, 4)):
    """A stream that changes every frame."""
    return np.random.default_rng(0).normal(size=(rows, *shape))


def _frozen(rows=ROWS, shape=(25, 4, 4)):
    """A stream pinned at the identity matrix (the canonical sensor dropout)."""
    return np.broadcast_to(np.eye(4), (rows, *shape)).copy()


def _episode(tmp_path, sources: dict[str, dict[str, np.ndarray]], name="ep"):
    """A staged episode: each source gets ``capture_time`` at 10 Hz (or its own) plus the given arrays."""
    root = zarr.open_group(str(tmp_path / name), mode="w")
    for source, arrays in sources.items():
        group = root.create_group(source)
        rows = len(next(iter(arrays.values()))) if arrays else 0
        if "capture_time" not in arrays:
            group.create_array("capture_time", data=np.arange(rows) / 10.0)
        for field, data in arrays.items():
            group.create_array(field, data=data)
    return tmp_path / name


def _glove(left=None, right=None, rows=ROWS):
    arrays = {}
    for side, streams in (("left", left), ("right", right)):
        for stream in ("hand_keypoints", "hand_joints_robot"):
            arrays[f"{side}_{stream}"] = _moving(rows) if streams is None else streams
    return arrays


def _by_name(report):
    return {r.name: r for r in report.check_results}


def _config(**overrides):
    return CaptureQcConfig(**{"avatar_hand_sources": ("glove",), **overrides})


# --- Avatar hand liveness --------------------------------------------------------------------


def test_two_live_hands_pass_with_their_stream_evidence(tmp_path):
    report = check_capture_quality(_episode(tmp_path, {"glove": _glove()}), _config())
    left, right = (
        _by_name(report)["avatar_hand_liveness:glove/left"],
        _by_name(report)["avatar_hand_liveness:glove/right"],
    )
    assert left.outcome is right.outcome is Outcome.PASSED
    assert "20 rows, live True" in left.evidence[0] and left.measurements == {"rows": 20}
    assert decide(report) is QcDecision.PASSED


def test_a_hand_frozen_for_the_whole_episode_flags_the_take_bad(tmp_path):
    report = check_capture_quality(_episode(tmp_path, {"glove": _glove(left=_frozen())}), _config())
    left = _by_name(report)["avatar_hand_liveness:glove/left"]
    assert left.outcome is Outcome.FAILED and "left hand: 0 live states" in left.issues[0]
    assert _by_name(report)["avatar_hand_liveness:glove/right"].outcome is Outcome.PASSED
    assert decide(report) is QcDecision.FLAGGED_BAD


def test_one_live_stream_is_enough_for_a_hand(tmp_path):
    glove = _glove(right=_moving())
    glove["left_hand_keypoints"] = _frozen()  # pinned, but the joints stream still moves
    report = check_capture_quality(_episode(tmp_path, {"glove": glove}), _config())
    assert _by_name(report)["avatar_hand_liveness:glove/left"].outcome is Outcome.PASSED


def test_a_hand_with_too_few_rows_is_dead(tmp_path):
    glove = _glove(left=_moving(rows=1))  # one row: below avatar_min_states
    path = _episode(tmp_path, {"glove": {**glove, **{k: v for k, v in _glove().items() if k.startswith("right")}}})
    report = check_capture_quality(path, _config())
    assert _by_name(report)["avatar_hand_liveness:glove/left"].outcome is Outcome.FAILED


def test_a_hand_with_no_recorded_streams_is_dead(tmp_path):
    only_left = {k: v for k, v in _glove().items() if k.startswith("left")}
    report = check_capture_quality(_episode(tmp_path, {"glove": only_left}), _config())
    right = _by_name(report)["avatar_hand_liveness:glove/right"]
    assert right.outcome is Outcome.FAILED and "none recorded" in right.evidence[0]


# --- Pico body joints ------------------------------------------------------------------------


def _pico(tmp_path, poses=None, valid=None, **config):
    poses = np.random.default_rng(1).normal(size=(ROWS, 24, 4, 4)).astype(np.float32) if poses is None else poses
    valid = np.ones(len(poses), dtype=bool) if valid is None else valid
    path = _episode(tmp_path, {"pico": {"body_joints_pose": poses, "body_joints_valid": valid}})
    return check_capture_quality(path, CaptureQcConfig(pico_body=PicoBodyConfig(source="pico", **config)))


def test_valid_pico_body_tracking_passes(tmp_path):
    report = _pico(tmp_path)
    assert _by_name(report)["pico_body_joints:pico"].outcome is Outcome.PASSED
    assert decide(report) is QcDecision.PASSED


def test_pico_body_problems_flag_the_take_bad(tmp_path):
    invalid = np.ones(ROWS, dtype=bool)
    invalid[3] = False
    report = _pico(tmp_path, valid=invalid)
    assert "1 invalid body-joint frame(s) > allowed 0" in _by_name(report)["pico_body_joints:pico"].issues[0]
    assert decide(report) is QcDecision.FLAGGED_BAD


def test_pico_validity_field_is_optional(tmp_path):
    path = _episode(
        tmp_path,
        {"pico": {"body_joints_pose": np.random.default_rng(2).normal(size=(ROWS, 24, 4, 4)).astype(np.float32)}},
    )
    config = CaptureQcConfig(pico_body=PicoBodyConfig(source="pico", valid_field=None))
    assert _by_name(check_capture_quality(path, config))["pico_body_joints:pico"].outcome is Outcome.PASSED


@pytest.mark.parametrize(
    ("poses", "valid", "limits", "problem"),
    [
        (np.zeros((5, 3, 4, 4)), None, (0, 0), "expected shape (T, 24, 4, 4)"),
        (np.zeros((0, 24, 4, 4)), None, (0, 0), "no body-joint frames recorded"),
        (np.full((2, 24, 4, 4), np.nan), None, (0, 0), "non-finite values"),
        (np.ones((3, 24, 4, 4)), np.ones(2, dtype=bool), (0, 0), "validity length 2 does not match 3"),
        (
            np.ones((3, 24, 4, 4)),
            np.array([True, False, False]),
            (1, 0),
            "2 invalid body-joint frame(s) > allowed 1",
        ),
        (
            np.broadcast_to(np.eye(4), (3, 24, 4, 4)).copy(),
            None,
            (0, 1),
            "3 all-identity body-joint frame(s) > allowed 1",
        ),
        (np.broadcast_to(np.eye(4), (3, 24, 4, 4)).copy(), None, (0, 3), None),
    ],
)
def test_pico_body_measurement_reports_the_first_problem(poses, valid, limits, problem):
    found = pico_body_problem(poses, valid, limits[0], limits[1], 1e-6)
    assert (problem in found) if problem else found is None


# --- camera frame-count parity (opt-in) ------------------------------------------------------


def _cameras(tmp_path, a, b):
    return _episode(
        tmp_path,
        {"cam_a": {"capture_time": np.arange(a) / 30.0}, "cam_b": {"capture_time": np.arange(b) / 30.0}},
    )


def test_cameras_within_the_allowed_difference_pass(tmp_path):
    config = CaptureQcConfig(camera_parity=("cam_a", "cam_b"), camera_parity_max_diff=5)
    report = check_capture_quality(_cameras(tmp_path, 100, 104), config)
    parity = _by_name(report)["camera_frame_parity"]
    assert parity.outcome is Outcome.PASSED and parity.measurements == {"cam_a": 100, "cam_b": 104}
    assert decide(report) is QcDecision.PASSED


def test_a_camera_that_lost_frames_rejects_the_take(tmp_path):
    config = CaptureQcConfig(camera_parity=("cam_a", "cam_b"), camera_parity_max_diff=5)
    report = check_capture_quality(_cameras(tmp_path, 100, 60), config)
    assert "spread 40 exceeds 5" in _by_name(report)["camera_frame_parity"].issues[0]
    assert decide(report) is QcDecision.REJECTED_CAMERA


# --- advisories ------------------------------------------------------------------------------


def test_a_pose_stream_frozen_for_the_whole_episode_is_an_advisory_dead_stream(tmp_path):
    """Advisory here (freeze limit above the episode length); pinned at identity reads as sensor dropout."""
    path = _episode(tmp_path, {"tracker": {"pose": _frozen(shape=(4, 4))}})
    config = CaptureQcConfig(pose_streams=(("tracker", "pose"),), freeze_seconds=100.0, flat_seconds=100.0)
    report = check_capture_quality(path, config)
    dead = _by_name(report)["dead_stream:tracker/pose"]
    assert dead.outcome is Outcome.FAILED and "at identity (sensor dropout)" in dead.issues[0]
    assert report.warnings and decide(report) is QcDecision.PASSED


def test_frame_drops_are_measured_and_only_advise(tmp_path):
    stamps = np.concatenate([np.arange(100) / 30.0, np.arange(100) / 30.0 + 100 / 30.0 + 0.3])  # a 0.3 s stall
    path = _episode(tmp_path, {"cam_a": {"capture_time": stamps}})
    config = CaptureQcConfig(camera_expected_fps=(("cam_a", 30.0),))
    report = check_capture_quality(path, config)
    drops = _by_name(report)["frame_drops:cam_a"]
    assert drops.outcome is Outcome.FAILED and drops.measurements["window_max"] == drops.measurements["total"] == 9
    assert report.warnings == drops.issues and decide(report) is QcDecision.PASSED
    steady = _episode(tmp_path, {"cam_a": {"capture_time": np.arange(60) / 30.0}}, name="steady")
    assert _by_name(check_capture_quality(steady, config))["frame_drops:cam_a"].measurements["total"] == 0


def test_measurement_edge_cases():
    assert stream_liveness(np.zeros((1, 3)), 1e-6, 0.95)["frozen"] is False  # one frame cannot be judged
    assert stream_liveness(np.array([]), 1e-6, 0.95)["frames"] == 0  # not even an array of frames
    assert stream_liveness(np.array([[np.nan], [np.inf]]), 1e-6, 0.95)["frozen"] is True  # non-finite counts as 0
    assert windowed_frame_drops(np.array([0.0]), 30.0, 2.0) == 0 and total_frame_drops(np.array([0.0]), 30.0) == 0
    assert (
        windowed_frame_drops(np.arange(5) / 30.0, 0.0, 2.0) == 0 and total_frame_drops(np.arange(5) / 30.0, 0) == 0
    )
    # the window forgets old gaps: two single missed frames 10 s apart never add up inside a 2 s window
    stamps = np.delete(np.arange(0, 12, 1 / 30), [30, 330])
    assert windowed_frame_drops(stamps, 30.0, 2.0) == 1 and total_frame_drops(stamps, 30.0) == 2


# --- decision precedence, errors and configuration -------------------------------------------


def test_camera_failure_outranks_every_tracking_failure_and_tracking_failure_outranks_errors(tmp_path):
    path = _episode(
        tmp_path,
        {
            "glove": _glove(left=_frozen()),
            "cam_a": {"capture_time": np.arange(100) / 30.0},
            "cam_b": {"capture_time": np.arange(10) / 30.0},
        },
    )
    both = _config(camera_parity=("cam_a", "cam_b"), camera_parity_max_diff=5)
    assert decide(check_capture_quality(path, both)) is QcDecision.REJECTED_CAMERA
    ghost = _config(pico_body=PicoBodyConfig(source="ghost"))  # a body check that cannot run
    report = check_capture_quality(path, ghost)
    assert _by_name(report)["pico_body_joints:ghost"].outcome is Outcome.ERROR
    assert decide(report) is QcDecision.FLAGGED_BAD  # the measured hand failure is not hidden by the error
    only_ghost = check_capture_quality(path, CaptureQcConfig(pico_body=PicoBodyConfig(source="ghost")))
    assert decide(only_ghost) is QcDecision.ERROR


def test_an_unreadable_episode_makes_every_configured_device_gate_an_error(tmp_path):
    config = _config(camera_parity=("cam_a", "cam_b"), pico_body=PicoBodyConfig(source="pico"))
    report = check_capture_quality(tmp_path / "missing", config)
    assert set(_by_name(report)) == {
        "camera_frame_parity",
        "avatar_hand_liveness:glove/left",
        "avatar_hand_liveness:glove/right",
        "pico_body_joints:pico",
    }
    assert all(r.outcome is Outcome.ERROR for r in report.check_results)


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"avatar_hand_sources": ("pico",)}, "not a numeric source with Avatar hand fields"),
        ({"avatar_hand_sources": ("nope",)}, "not a numeric source with Avatar hand fields"),
        ({"avatar_min_states": 0}, "avatar_min_states must be an integer >= 1"),
        ({"dead_frac_identical": 1.5}, "dead_frac_identical"),
        ({"dead_std_eps": 0.0}, "dead_std_eps must be a finite number > 0"),
        ({"drop_window_s": float("inf")}, "drop_window_s"),
        ({"max_dropped_frames": -1}, "max_dropped_frames must be an integer >= 0"),
        ({"camera_parity": ("cam_a",)}, "camera_parity needs two or more distinct video sources"),
        ({"camera_parity": ("cam_a", "nope")}, "unknown: ['nope']"),
        ({"camera_parity": ("cam_a", "cam_a")}, "distinct"),
        ({"camera_parity": ("cam_a", "cam_b"), "camera_parity_max_diff": -1}, "camera_parity_max_diff"),
        ({"pico_body": PicoBodyConfig(source="nope")}, "pico_body names unknown numeric source"),
        ({"pico_body": PicoBodyConfig(source="pico", field="nope")}, "must be a float field of shape (24, 4, 4)"),
        (
            {"pico_body": PicoBodyConfig(source="pico", valid_field="nope")},
            "validity pico/nope must be a per-frame flag",
        ),
        ({"pico_body": PicoBodyConfig(source="pico", max_invalid_frames=-1)}, "max_invalid_frames"),
        ({"pico_body": PicoBodyConfig(source="pico", max_identity_frames=-1)}, "max_identity_frames"),
        ({"pico_body": PicoBodyConfig(source="pico", identity_atol=-1.0)}, "identity_atol"),
    ],
)
def test_device_check_configuration_problems_are_named(overrides, problem):
    base = {"avatar_hand_sources": ()} if "avatar_hand_sources" in overrides else {}
    config = CaptureQcConfig(**{"camera_expected_fps": (("cam_a", 30.0),), **base, **overrides})
    problems = config.problems((AVATAR, PICO), CAMERAS)
    assert any(problem in p for p in problems), problems


def test_an_avatar_source_that_records_only_one_hand_is_refused_because_it_would_flag_every_take():
    one_hand = NumericSourceSpec("glove", tuple(f for f in AVATAR.fields if f.name.startswith("left_")))
    config = CaptureQcConfig(camera_expected_fps=(("cam_a", 30.0),), avatar_hand_sources=("glove",))
    problems = config.problems((one_hand,), CAMERAS)
    assert any("records only the ['left'] hand; both are needed" in p for p in problems), problems
    assert config.problems((AVATAR,), CAMERAS) == []


def test_a_valid_device_configuration_has_no_problems():
    config = CaptureQcConfig(
        avatar_hand_sources=("glove",),
        pico_body=PicoBodyConfig(source="pico"),
        camera_parity=("cam_a", "cam_b"),
    )
    assert config.problems((AVATAR, PICO), CAMERAS) == []


def test_a_parity_only_configuration_is_a_valid_enabled_check():
    assert CaptureQcConfig(camera_parity=("cam_a", "cam_b")).problems((), CAMERAS) == []


def test_pico_declarations_with_the_wrong_schema_are_refused_before_recording():
    """A pose field that is not (24, 4, 4), or a validity field that is not a per-frame flag, cannot work."""
    wrong = NumericSourceSpec(
        "pico",
        (FieldSpec("body_joints_pose", "float32", (4, 4)), FieldSpec("body_joints_valid", "bool", (3,))),
    )
    flags = NumericSourceSpec(
        "pico", (FieldSpec("body_joints_pose", "float32", (24, 4, 4)), FieldSpec("body_joints_valid", "float32"))
    )
    config = CaptureQcConfig(pico_body=PicoBodyConfig(source="pico"))
    assert any("shape (24, 4, 4)" in p for p in config.problems((wrong,), ()))
    assert any("per-frame flag" in p for p in config.problems((wrong,), ()))
    assert any("per-frame flag" in p for p in config.problems((flags,), ()))


def test_a_pico_declaration_without_a_validity_field_needs_none():
    config = CaptureQcConfig(pico_body=PicoBodyConfig(source="pico", valid_field=None))
    assert config.problems((PICO,), ()) == []
