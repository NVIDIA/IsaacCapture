# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture-quality checks and the keep/flag/reject decision, on explicit numeric arrays with known answers.

The arrays are written to real Zarr groups on disk and read back by the production checker; durations and
motions are chosen so every expected rate or still-window is exact. Real recorded data goes through the
service in tests/recording_service/test_capture_qc_service.py.
"""

import numpy as np
import pytest
import zarr

from egocap.recording.capture_qc import (
    ALGORITHM_VERSION,
    CaptureQcConfig,
    QcDecision,
    check_capture_quality,
    decide,
    finalization_for,
    policy_to_dict,
    quality_attributes,
)
from egocap.recording.dataset_check import CheckResult, Outcome, QualityReport, report_from_dict, report_to_dict
from egocap.recording.episode_layout import FINALIZATION_KEEP, FINALIZATION_REMOVE, FINALIZATION_RETAIN
from egocap.recording.episode_store import FieldSpec, NumericSourceSpec, VideoSourceSpec

NUMERIC = NumericSourceSpec("tracker", (FieldSpec("pose", "float64", (3,)), FieldSpec("flag", "bool")))
CAMERA = VideoSourceSpec("cam", 64, 48, 60.0)


def _episode(tmp_path, *, cam_fps=60.0, cam_frames=61, pose_rate=10.0, pose=None, name="ep"):
    """A staged episode: camera times at ``cam_fps``; tracker rows at ``pose_rate`` Hz with ``pose`` values."""
    root = zarr.open_group(str(tmp_path / name), mode="w")
    root.create_group("cam").create_array("capture_time", data=np.arange(cam_frames) / cam_fps)
    rows = 31 if pose is None else len(pose)
    tracker = root.create_group("tracker")
    tracker.create_array("capture_time", data=np.arange(rows) / pose_rate)
    values = np.arange(rows * 3, dtype=np.float64).reshape(rows, 3) if pose is None else pose
    tracker.create_array("pose", data=values)
    tracker.create_array("flag", data=np.zeros(rows, dtype=bool))
    return tmp_path / name


def _config(**overrides):
    base = dict(camera_expected_fps=(("cam", 60.0),), pose_streams=(("tracker", "pose"),))
    return CaptureQcConfig(**{**base, **overrides})


def _by_name(report):
    return {r.name: r for r in report.check_results}


def test_healthy_capture_passes_with_measured_evidence(tmp_path):
    """A camera at its expected rate and a moving pose pass, and passing items still carry their measurements."""
    report = check_capture_quality(_episode(tmp_path), _config())
    assert report.outcome is Outcome.PASSED and not report.issues and not report.warnings
    results = _by_name(report)
    assert set(results) == {
        "camera_min_fps:cam",
        "sustained_freeze:tracker/pose",
        "flat_motion:tracker/pose",
        "dead_stream:tracker/pose",
        "frame_drops:cam",
    }
    assert results["camera_min_fps:cam"].evidence == ("60.00 fps average (minimum 51.60)",)
    assert results["sustained_freeze:tracker/pose"].evidence == ("longest still window 0.00s (limit 1.0s)",)
    assert decide(report) is QcDecision.PASSED


@pytest.mark.parametrize(("fps", "passes"), [(52.0, True), (51.0, False)])
def test_camera_minimum_fps_threshold_is_the_native_ratio_of_expected(tmp_path, fps, passes):
    """The native 0.86 x 60 = 51.6 fps is the default floor: above passes, below fails with the measured rate."""
    report = check_capture_quality(_episode(tmp_path, cam_fps=fps), _config())
    camera = _by_name(report)["camera_min_fps:cam"]
    assert (camera.outcome is Outcome.PASSED) is passes
    assert passes or f"{fps:.2f} fps average is below 0.86 x 60.0 = 51.60" in camera.issues[0]
    assert decide(report) is (QcDecision.PASSED if passes else QcDecision.REJECTED_CAMERA)


def test_a_camera_with_fewer_than_two_frames_counts_as_zero_fps(tmp_path):
    """One frame has no duration, so its rate is 0 and the gate fails (native convention)."""
    report = check_capture_quality(_episode(tmp_path, cam_frames=1), _config())
    assert _by_name(report)["camera_min_fps:cam"].outcome is Outcome.FAILED
    assert decide(report) is QcDecision.REJECTED_CAMERA


def test_sustained_freeze_flags_the_kept_data_bad(tmp_path):
    """Two identical seconds (limit 1 s) fail the freeze gate; the take is flagged bad, not rejected."""
    pose = np.arange(31 * 3, dtype=np.float64).reshape(31, 3)
    pose[5:26] = pose[5]  # rows 5..25 identical: 20 intervals at 10 Hz = 2.0 s still
    report = check_capture_quality(_episode(tmp_path, pose=pose), _config())
    freeze = _by_name(report)["sustained_freeze:tracker/pose"]
    assert freeze.outcome is Outcome.FAILED and "no motion for 2.00s" in freeze.issues[0]
    assert report.outcome is Outcome.FAILED
    assert decide(report) is QcDecision.FLAGGED_BAD


def test_a_still_window_at_the_end_of_the_recording_counts(tmp_path):
    """A freeze that runs to the last row is measured to the last timestamp."""
    pose = np.arange(31 * 3, dtype=np.float64).reshape(31, 3)
    pose[20:] = pose[20]
    report = check_capture_quality(_episode(tmp_path, pose=pose), _config())
    assert "no motion for 1.00s" in _by_name(report)["sustained_freeze:tracker/pose"].issues[0]


def test_flat_motion_is_advisory_and_never_gates(tmp_path):
    """0.5 s still is under both limits; 2 s is under a raised freeze limit but over the flat limit."""
    pose = np.arange(31 * 3, dtype=np.float64).reshape(31, 3)
    pose[10:16] = pose[10]  # 0.5 s still: below the 1 s freeze limit and the 2 s flat limit
    ok = check_capture_quality(_episode(tmp_path, pose=pose, name="short"), _config())
    assert decide(ok) is QcDecision.PASSED and not ok.warnings
    # Freeze limit raised above the flat limit: the same 2 s window is now only an advisory warning.
    long_still = np.arange(31 * 3, dtype=np.float64).reshape(31, 3)
    long_still[5:26] = long_still[5]
    report = check_capture_quality(_episode(tmp_path, pose=long_still, name="long"), _config(freeze_seconds=5.0))
    flat = _by_name(report)["flat_motion:tracker/pose"]
    assert flat.outcome is Outcome.FAILED and report.warnings == flat.issues
    assert report.outcome is Outcome.PASSED and decide(report) is QcDecision.PASSED


def test_each_camera_is_measured_separately_so_a_failure_survives_a_later_read_error(tmp_path):
    """Camera 'cam' fails measurably; camera 'ghost' has no data and errors. The rejection must stand."""
    config = _config(camera_expected_fps=(("cam", 600.0), ("ghost", 60.0)))
    report = check_capture_quality(_episode(tmp_path), config)
    results = _by_name(report)
    assert results["camera_min_fps:cam"].outcome is Outcome.FAILED
    assert results["camera_min_fps:ghost"].outcome is Outcome.ERROR
    assert report.outcome is Outcome.ERROR
    assert decide(report) is QcDecision.REJECTED_CAMERA


def test_a_camera_gate_that_cannot_run_is_error_and_never_passed(tmp_path):
    """No measured camera failure and one unreadable camera: ERROR (the take is retained), tracking fine."""
    report = check_capture_quality(_episode(tmp_path), _config(camera_expected_fps=(("ghost", 60.0),)))
    assert decide(report) is QcDecision.ERROR


def test_a_pose_stream_that_cannot_be_read_is_error_and_a_measured_freeze_still_flags(tmp_path):
    """One stream errors, another is measurably frozen: the freeze verdict is kept; with no freeze it is ERROR."""
    frozen = np.zeros((31, 3))
    path = _episode(tmp_path, pose=frozen)
    root = zarr.open_group(str(path), mode="r+")
    other = root.create_group("other")
    other.create_array("capture_time", data=np.arange(31) / 10.0)  # no 'pose' array: unreadable stream
    both = _config(pose_streams=(("tracker", "pose"), ("other", "pose")))
    flagged = check_capture_quality(path, both)
    assert _by_name(flagged)["sustained_freeze:other/pose"].outcome is Outcome.ERROR
    assert decide(flagged) is QcDecision.FLAGGED_BAD
    only_other = check_capture_quality(path, _config(pose_streams=(("other", "pose"),)))
    assert decide(only_other) is QcDecision.ERROR


def test_camera_rejection_takes_precedence_over_a_freeze(tmp_path):
    """Both a camera failure and a frozen pose were measured: the take is rejected, not just flagged."""
    path = _episode(tmp_path, cam_fps=10.0, pose=np.zeros((31, 3)))
    report = check_capture_quality(path, _config())
    assert decide(report) is QcDecision.REJECTED_CAMERA


def test_an_unreadable_episode_makes_every_gate_an_error(tmp_path):
    """If the sealed episode cannot be opened nothing was measured: every configured gate reports ERROR."""
    report = check_capture_quality(tmp_path / "missing", _config())
    assert report.outcome is Outcome.ERROR
    assert {r.name for r in report.check_results} == {"camera_min_fps:cam", "sustained_freeze:tracker/pose"}
    assert all(r.outcome is Outcome.ERROR for r in report.check_results)
    assert decide(report) is QcDecision.ERROR


def test_decide_without_a_report_is_skipped():
    """Capture QC not configured means nothing was measured: SKIPPED, never PASSED."""
    assert decide(None) is QcDecision.SKIPPED


def test_evidence_round_trips_and_an_older_report_without_it_still_loads(tmp_path):
    """Per-check evidence survives the manifest round trip; a report persisted before it existed loads empty."""
    report = check_capture_quality(_episode(tmp_path), _config())
    assert report_from_dict(report_to_dict(report)) == report
    old = report_to_dict(report)
    for result in old["check_results"]:
        del result["evidence"]
    assert all(r.evidence == () for r in report_from_dict(old).check_results)


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"camera_expected_fps": (), "pose_streams": ()}, "names no camera, parity, pose stream"),
        ({"camera_expected_fps": (("nope", 60.0),)}, "unknown video source 'nope'"),
        ({"camera_expected_fps": (("cam", 0.0),)}, "'cam' must be a finite number > 0"),
        ({"camera_expected_fps": (("cam", float("inf")),)}, "'cam' must be a finite number > 0"),
        ({"pose_streams": (("nope", "pose"),)}, "unknown numeric source 'nope'"),
        ({"pose_streams": (("tracker", "flag"),)}, "tracker/flag is not a float field"),
        ({"pose_streams": (("tracker", "missing"),)}, "tracker/missing is not a float field"),
        ({"camera_min_fps_ratio": 1.5}, "camera_min_fps_ratio must be > 0 and <= 1"),
        ({"freeze_seconds": 0.0}, "freeze_seconds must be a finite number > 0"),
        ({"flat_eps": float("nan")}, "flat_eps must be a finite number > 0"),
    ],
)
def test_configuration_problems_are_reported_before_recording(overrides, problem):
    """An unusable configuration is named precisely, so it can be refused before any take starts."""
    problems = _config(**overrides).problems((NUMERIC,), (CAMERA,))
    assert any(problem in p for p in problems), problems


def test_a_valid_configuration_has_no_problems():
    """The default thresholds with declared, existing sources are accepted."""
    assert _config().problems((NUMERIC,), (CAMERA,)) == []


def test_a_single_pose_row_has_no_still_window(tmp_path):
    """One row has no interval to measure, so it cannot be frozen."""
    report = check_capture_quality(_episode(tmp_path, pose=np.zeros((1, 3))), _config())
    assert _by_name(report)["sustained_freeze:tracker/pose"].outcome is Outcome.PASSED


def test_the_stored_policy_names_the_algorithm_version_and_every_threshold():
    """What is persisted with a take is enough to interpret its decision without the launch configuration."""
    policy = policy_to_dict(_config())
    assert policy["algorithm_version"] == ALGORITHM_VERSION
    assert policy["camera_min_fps_ratio"] == 0.86 and policy["freeze_seconds"] == 1.0
    assert policy["camera_expected_fps"] == (("cam", 60.0),) and policy["pose_streams"] == (("tracker", "pose"),)


def test_a_camera_that_cannot_be_measured_never_hides_known_frozen_tracking(tmp_path):
    """Only a measured camera failure outranks bad tracking: camera ERROR plus a measured freeze is FLAGGED_BAD."""
    config = _config(camera_expected_fps=(("ghost", 60.0),))
    report = check_capture_quality(_episode(tmp_path, pose=np.zeros((31, 3))), config)
    assert _by_name(report)["camera_min_fps:ghost"].outcome is Outcome.ERROR  # still reported per check
    assert decide(report) is QcDecision.FLAGGED_BAD


def test_finalization_follows_integrity_first_then_the_capture_decision():
    """Failed or unreadable integrity retains the data; only a camera rejection removes it; the rest keep."""
    for integrity in (Outcome.FAILED, Outcome.ERROR):
        for decision in QcDecision:
            assert finalization_for(integrity, decision) == FINALIZATION_RETAIN
    assert finalization_for(Outcome.PASSED, QcDecision.REJECTED_CAMERA) == FINALIZATION_REMOVE
    for decision in (QcDecision.PASSED, QcDecision.SKIPPED, QcDecision.ERROR, QcDecision.FLAGGED_BAD):
        assert finalization_for(Outcome.PASSED, decision) == FINALIZATION_KEEP


def _report(*results, warnings=()):
    return QualityReport(Outcome.PASSED, (), tuple(warnings), tuple(results))


def test_quality_attributes_when_capture_qc_did_not_run():
    attributes = quality_attributes(QcDecision.SKIPPED, None, None)
    assert attributes["domain_checks"] == "SKIPPED" and attributes["capture_qc_decision"] == "SKIPPED"
    assert attributes["bad_episode"] is False and attributes["capture_qc_results"] == []
    assert "freeze_check_passed" not in attributes and "quality_warnings" not in attributes


def test_the_native_tracking_verdict_covers_every_tracking_gate_that_established_it():
    freeze_ok = CheckResult("sustained_freeze:t/pose", Outcome.PASSED)
    freeze_bad = CheckResult("sustained_freeze:t/pose", Outcome.FAILED, ("frozen",))
    freeze_err = CheckResult("sustained_freeze:t/pose", Outcome.ERROR, ("could not run",))
    hand_ok = CheckResult("avatar_hand_liveness:t/left", Outcome.PASSED)
    hand_bad = CheckResult("avatar_hand_liveness:t/left", Outcome.FAILED, ("dead glove",))
    body_bad = CheckResult("pico_body_joints:t", Outcome.FAILED, ("non-finite body joints",))
    camera = CheckResult("camera_min_fps:cam", Outcome.PASSED)
    ok = quality_attributes(QcDecision.PASSED, _report(camera, freeze_ok, hand_ok), {"algorithm_version": 2})
    assert (
        ok["freeze_check_passed"] is True and ok["freeze_check_issues"] == [] and ok["domain_checks"] == "EXECUTED"
    )
    assert (
        ok["capture_qc_policy"] == {"algorithm_version": 2} and ok["capture_qc_results"][0]["name"] == camera.name
    )
    bad = quality_attributes(QcDecision.FLAGGED_BAD, _report(camera, freeze_bad), None)
    assert bad["freeze_check_passed"] is False and bad["freeze_check_issues"] == ["frozen"] and bad["bad_episode"]
    # A dead Avatar hand or a Pico body problem fails the native verdict even when the pose freeze check passed.
    hand = quality_attributes(QcDecision.FLAGGED_BAD, _report(camera, freeze_ok, hand_bad), None)
    assert hand["freeze_check_passed"] is False and hand["freeze_check_issues"] == ["dead glove"]
    both = quality_attributes(QcDecision.FLAGGED_BAD, _report(freeze_bad, hand_bad, body_bad), None)
    assert both["freeze_check_issues"] == ["frozen", "dead glove", "non-finite body joints"]
    # Not established: no tracking gate ran (camera-only policy), or the only tracking gate could not run.
    for report in (_report(camera), _report(camera, freeze_err)):
        assert "freeze_check_passed" not in quality_attributes(QcDecision.PASSED, report, None)
    # One gate passed and another could not run: not "passed" either, since the second says nothing.
    hand_err = CheckResult("avatar_hand_liveness:t/left", Outcome.ERROR)
    mixed = quality_attributes(QcDecision.ERROR, _report(freeze_ok, hand_err), None)
    assert "freeze_check_passed" not in mixed


def test_the_native_drop_evidence_is_kept_per_camera_and_warnings_are_listed():
    drops = CheckResult("frame_drops:cam", Outcome.FAILED, ("cam: 9 dropped",), (), {"window_max": 9, "total": 9})
    errored = CheckResult("frame_drops:other", Outcome.ERROR, ("could not run",))
    attributes = quality_attributes(QcDecision.PASSED, _report(drops, errored, warnings=("cam: 9 dropped",)), None)
    assert attributes["frame_drops"] == {"cam": {"window_max": 9, "total": 9}}
    assert attributes["quality_warnings"] == ["cam: 9 dropped"]
    assert "frame_drops" not in quality_attributes(QcDecision.PASSED, _report(errored), None)


def test_a_default_camera_rate_resolves_to_the_sources_own_fps_and_an_explicit_rate_wins():
    video = (VideoSourceSpec("cam", 64, 48, 30.0), VideoSourceSpec("other", 64, 48, 24.0))
    config = _config(camera_expected_fps=(("cam", None), ("other", 60.0)))
    assert config.problems((NUMERIC,), video) == []
    assert config.resolved(video).camera_expected_fps == (("cam", 30.0), ("other", 60.0))


def test_an_unknown_camera_with_a_default_rate_stays_unresolved_and_is_reported():
    config = _config(camera_expected_fps=(("ghost", None),))
    assert config.resolved((CAMERA,)).camera_expected_fps == (("ghost", None),)
    assert any("unknown video source 'ghost'" in p for p in config.problems((NUMERIC,), (CAMERA,)))


def test_checking_with_an_unresolved_default_rate_is_an_error_for_that_camera_not_a_crash(tmp_path):
    report = check_capture_quality(_episode(tmp_path), _config(camera_expected_fps=(("cam", None),)))
    camera = _by_name(report)["camera_min_fps:cam"]
    assert camera.outcome is Outcome.ERROR and "unresolved" in camera.issues[0]
    assert decide(report) is QcDecision.ERROR
