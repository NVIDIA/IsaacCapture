# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit layer: accumulators fed frames built in memory, no MCAP and no fixture set."""

from __future__ import annotations

import math

import pytest
import synth

from full_body_acceptance.checks import (
    CHECKS,
    Attribution,
    Severity,
    Status,
    build,
    build_all,
)
from full_body_acceptance.checks.envelope import (
    AvailableNotBeforeSample,
    DeviceClockDistinct,
    Finite,
    Monotonic,
    PayloadPresenceRate,
    UnitNormOnValidJoints,
    ZeroPoseOnValidJoint,
)
from full_body_acceptance.report import Verdict, aggregate, run


def drive(check, frame_list):
    for item in frame_list:
        check.update(item)
    return check.result()


def status_of(check_cls, frame_list) -> Status:
    return drive(check_cls(), frame_list).status


# --- the contract itself -----------------------------------------------------------


def test_check_names_are_unique():
    names = [cls.name for cls in CHECKS]
    assert len(names) == len(set(names))


def test_result_reports_insufficient_data_before_any_frame():
    for check in build_all():
        assert check.result().status is Status.INSUFFICIENT_DATA


def test_monotonic_needs_two_frames_to_conclude():
    assert status_of(Monotonic, synth.frames(1)) is Status.INSUFFICIENT_DATA
    assert status_of(Monotonic, synth.frames(2)) is Status.PASS


def test_build_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        build(["values.does_not_exist"])


# --- values ------------------------------------------------------------------------


def test_finite_passes_a_clean_recording():
    assert status_of(Finite, synth.frames(10)) is Status.PASS


def test_finite_catches_nan_and_inf_on_a_valid_joint():
    base = synth.frame(0)
    spoiled = [
        synth.with_joint(base, 3, synth.joint(position=(float("nan"), 1.0, 0.0))),
        synth.with_joint(base, 7, synth.joint(position=(float("inf"), 1.0, 0.0))),
    ]
    outcome = drive(Finite(), spoiled)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["nan_components"] == 1
    assert outcome.measurements["inf_components"] == 1
    assert outcome.measurements["joints"] == [3, 7]


def test_zero_position_fails_only_when_the_joint_claims_to_be_valid():
    base = synth.frame(0)
    zeroed = synth.joint(position=(0.0, 0.0, 0.0), orientation=synth.IDENTITY)
    valid_zero = [synth.with_joint(base, i, zeroed) for i in range(6)]
    assert status_of(ZeroPoseOnValidJoint, valid_zero) is Status.FAIL

    invalid_zero = [
        synth.with_joint(
            base,
            i,
            synth.joint(
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 0.0),
                is_valid=False,
            ),
        )
        for i in range(6)
    ]
    assert status_of(ZeroPoseOnValidJoint, invalid_zero) is Status.PASS


# --- quaternion --------------------------------------------------------------------


def test_unit_norm_accepts_a_rotated_but_normalised_orientation():
    rotated = synth.unit_quaternion(math.radians(37), (0.0, 1.0, 0.0))
    frame_list = [
        synth.with_joint(synth.frame(i), 5, synth.joint(orientation=rotated))
        for i in range(4)
    ]
    assert status_of(UnitNormOnValidJoints, frame_list) is Status.PASS


def test_unit_norm_catches_a_scaled_orientation():
    scaled = (0.0, 0.0, 0.0, 1.8)
    frame_list = [synth.with_joint(synth.frame(0), 2, synth.joint(orientation=scaled))]
    outcome = drive(UnitNormOnValidJoints(), frame_list)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["worst_norm"] == pytest.approx(1.8)


def test_unit_norm_ignores_the_all_zero_quaternion_of_an_invalid_joint():
    """make_invalid_body_joint_pose() emits exactly this, so it must not fail."""
    zero_quat = synth.joint(orientation=(0.0, 0.0, 0.0, 0.0), is_valid=False)
    frame_list = [synth.with_joint(synth.frame(i), 9, zero_quat) for i in range(4)]
    assert status_of(UnitNormOnValidJoints, frame_list) is Status.PASS


# --- timestamps --------------------------------------------------------------------


def test_monotonic_catches_a_backward_step():
    frame_list = synth.frames(6)
    frame_list[4] = synth.frame(4, sample_ns=frame_list[3].sample_time_ns - 40_000_000)
    outcome = drive(Monotonic(), frame_list)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["sample_regressions"] == 1
    assert outcome.measurements["worst_step_ms"] == pytest.approx(40.0)


def test_available_equal_to_sample_is_accepted():
    """The Pico tracker passes one monotonic reading as both, so zero latency is normal."""
    sample = synth.CLOCK_BASE_NS
    frame_list = [
        synth.frame(
            i,
            sample_ns=sample + i * synth.PERIOD_NS,
            available_ns=sample + i * synth.PERIOD_NS,
        )
        for i in range(5)
    ]
    assert status_of(AvailableNotBeforeSample, frame_list) is Status.PASS


def test_available_before_sample_fails():
    frame_list = [
        synth.frame(
            i,
            sample_ns=synth.CLOCK_BASE_NS + i * synth.PERIOD_NS,
            available_ns=synth.CLOCK_BASE_NS + i * synth.PERIOD_NS - 5_000_000,
        )
        for i in range(5)
    ]
    assert status_of(AvailableNotBeforeSample, frame_list) is Status.FAIL


def test_device_clock_check_is_advisory_and_never_blocks():
    assert DeviceClockDistinct.severity is Severity.ADVISORY
    frame_list = [
        synth.frame(i, device_ns=synth.CLOCK_BASE_NS + i * synth.PERIOD_NS)
        for i in range(5)
    ]
    outcome = drive(DeviceClockDistinct(), frame_list)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["rate"] == pytest.approx(1.0)

    results = run(synth.StubSource(frame_list)).results
    advisory = next(r for r in results if r.name == DeviceClockDistinct.name)
    assert not advisory.counts_toward_verdict


# --- coverage ----------------------------------------------------------------------


def test_payload_presence_rate():
    present = synth.frames(10)
    assert status_of(PayloadPresenceRate, present) is Status.PASS

    mostly_empty = [synth.frame(i, has_payload=i < 3) for i in range(10)]
    outcome = drive(PayloadPresenceRate(), mostly_empty)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["rate"] == pytest.approx(0.3)


# --- aggregation -------------------------------------------------------------------


def test_a_clean_run_passes_end_to_end():
    report = run(synth.StubSource(synth.frames(300)))
    assert report.verdict is Verdict.PASS
    assert report.failures == ()
    assert report.advisories == ()


def test_advisory_failure_does_not_change_the_verdict():
    frame_list = [
        synth.frame(i, device_ns=synth.CLOCK_BASE_NS + i * synth.PERIOD_NS)
        for i in range(300)
    ]
    report = run(synth.StubSource(frame_list))
    assert report.verdict is Verdict.PASS
    assert [r.name for r in report.advisories] == ["timestamps.device_clock_distinct"]


def test_performance_attribution_yields_retake_not_fail():
    from full_body_acceptance.report import CheckResult

    def result(name, status, attribution):
        return CheckResult(
            name=name,
            gate="G4",
            severity=Severity.SOFT,
            attribution=attribution,
            summary="",
            status=status,
            detail="",
            measurements={},
        )

    assert aggregate([result("a", Status.PASS, Attribution.DEVICE)]) is Verdict.PASS
    assert (
        aggregate([result("a", Status.FAIL, Attribution.PERFORMANCE)]) is Verdict.RETAKE
    )
    assert (
        aggregate(
            [
                result("a", Status.FAIL, Attribution.PERFORMANCE),
                result("b", Status.FAIL, Attribution.DEVICE),
            ]
        )
        is Verdict.FAIL
    )
    assert (
        aggregate([result("a", Status.INSUFFICIENT_DATA, Attribution.DEVICE)])
        is Verdict.INSUFFICIENT_DATA
    )


# --- schema ------------------------------------------------------------------------


def test_joints_field_absence_is_a_schema_violation_not_a_rate():
    from full_body_acceptance.checks.envelope import JointsFieldPresent

    assert status_of(JointsFieldPresent, synth.frames(50)) is Status.PASS

    one_bad = synth.frames(50)
    one_bad[20] = synth.frame(20, include_joints=False)
    outcome = drive(JointsFieldPresent(), one_bad)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["missing_joints"] == 1


def test_a_null_payload_is_not_a_missing_joints_field():
    """The data table being absent is coverage.payload_presence_rate's business."""
    from full_body_acceptance.checks.envelope import JointsFieldPresent

    frame_list = [synth.frame(i, has_payload=i % 2 == 0) for i in range(50)]
    outcome = drive(JointsFieldPresent(), frame_list)
    assert outcome.status is Status.PASS
    assert outcome.measurements["records_with_payload"] == 25


# --- consistency -------------------------------------------------------------------


def test_all_tracked_flag_disagreement_is_advisory():
    from full_body_acceptance.checks.consistency import AllJointPosesTracked

    assert AllJointPosesTracked.severity is Severity.ADVISORY

    lying = [synth.frame(i, valid_joints=20, all_tracked=True) for i in range(50)]
    outcome = drive(AllJointPosesTracked(), lying)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["flag_true_but_joints_invalid"] == 50


def test_a_transient_dropout_whose_flag_follows_is_consistent():
    from full_body_acceptance.checks.consistency import AllJointPosesTracked

    frame_list = [
        synth.frame(
            i, valid_joints=20 if 20 <= i < 30 else 24, all_tracked=not (20 <= i < 30)
        )
        for i in range(60)
    ]
    assert status_of(AllJointPosesTracked, frame_list) is Status.PASS


# --- validity trend ----------------------------------------------------------------


def test_validity_trend_separates_decay_from_a_transient_dropout():
    from full_body_acceptance.checks.envelope import ValidityTrend

    decaying = [
        synth.frame(i, valid_joints=24 - round(17 * i / 499)) for i in range(500)
    ]
    outcome = drive(ValidityTrend(), decaying)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["drop"] > 0.25

    transient = [
        synth.frame(i, valid_joints=20 if 240 <= i < 300 else 24) for i in range(500)
    ]
    assert status_of(ValidityTrend, transient) is Status.PASS


def test_validity_trend_will_not_conclude_from_a_short_recording():
    from full_body_acceptance.checks.envelope import ValidityTrend

    assert status_of(ValidityTrend, synth.frames(100)) is Status.INSUFFICIENT_DATA


def test_permanently_invalid_joints_are_not_a_decaying_trend():
    """A vendor that never provides four joints has constant, not falling, coverage."""
    from full_body_acceptance.checks.envelope import ValidityTrend

    assert status_of(ValidityTrend, synth.frames(400, valid_joints=20)) is Status.PASS


# --- rate --------------------------------------------------------------------------


def test_interval_regularity_accepts_a_steady_stream_and_rejects_jitter():
    import random

    from full_body_acceptance.checks.envelope import IntervalRegularity

    assert status_of(IntervalRegularity, synth.frames(300)) is Status.PASS

    rng = random.Random(7)
    jittered = []
    time_ns = synth.CLOCK_BASE_NS
    for i in range(300):
        time_ns += synth.PERIOD_NS + rng.randint(-9_000_000, 9_000_000)
        jittered.append(synth.frame(i, sample_ns=time_ns))
    jittered.sort(key=lambda f: f.sample_time_ns)
    assert status_of(IntervalRegularity, jittered) is Status.FAIL


def test_a_few_dropped_blocks_do_not_read_as_jitter():
    """Median absolute deviation is chosen precisely so these two stay separable."""
    from full_body_acceptance.checks.envelope import FrameGaps, IntervalRegularity

    kept = [i for i in range(300) if not (100 <= i < 120 or 200 <= i < 225)]
    frame_list = [
        synth.frame(i, sample_ns=synth.CLOCK_BASE_NS + i * synth.PERIOD_NS)
        for i in kept
    ]

    assert status_of(IntervalRegularity, frame_list) is Status.PASS
    outcome = drive(FrameGaps(), frame_list)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["gaps"] == 2


# --- continuity --------------------------------------------------------------------


def test_human_speed_passes_and_a_teleport_fails():
    from full_body_acceptance.checks.envelope import MaxJointVelocity

    assert (
        status_of(MaxJointVelocity, synth.moving_frames(60, speed_mps=1.7))
        is Status.PASS
    )

    teleporting = synth.moving_frames(60, speed_mps=1.0)
    jumped = synth.with_joint(
        teleporting[30], 0, synth.joint(position=(50.0, 1.0, 0.0))
    )
    teleporting[30] = jumped
    outcome = drive(MaxJointVelocity(), teleporting)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["peak_speed_mps"] > 20.0


def test_velocity_over_an_implausibly_short_interval_is_discarded():
    """A sub-4 ms interval describes a broken clock, which rate.interval_regularity owns."""
    from full_body_acceptance.checks.envelope import MaxJointVelocity

    frame_list = synth.moving_frames(20, speed_mps=1.0)
    squeezed = (
        frame_list[:10]
        + [
            synth.frame(
                10,
                joints=frame_list[10].joints,
                sample_ns=frame_list[9].sample_time_ns + 200_000,
            )
        ]
        + frame_list[11:]
    )
    outcome = drive(MaxJointVelocity(), squeezed)
    assert outcome.status is Status.PASS
    assert outcome.measurements["discarded_short_intervals"] == 1


def test_a_short_recording_cannot_conclude_overall():
    report = run(synth.StubSource(synth.frames(5)))
    assert report.verdict is Verdict.INSUFFICIENT_DATA


# --- geometry ----------------------------------------------------------------------


def test_up_axis_accepts_y_up_and_rejects_z_up():

    from full_body_acceptance.checks.geometry import UpAxis

    upright = synth.frames(60)
    assert status_of(UpAxis, upright) is Status.PASS

    def rotate_x90(position):
        x, y, z = position
        return (x, -z, y)

    tipped = []
    for item in upright:
        assert item.joints is not None
        tipped.append(
            synth.replace(
                item,
                joints=tuple(
                    synth.joint(rotate_x90(j.position), j.orientation, j.is_valid)
                    for j in item.joints
                ),
            )
        )
    outcome = drive(UpAxis(), tipped)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["dominant_axis"] == "Z"
    assert outcome.measurements["torso_length_m"] == pytest.approx(0.65, abs=0.02)


def test_up_axis_will_not_guess_from_a_degenerate_torso():
    flattened = [
        synth.replace(
            item,
            joints=tuple(
                synth.joint(
                    (j.position[0], 1.0, j.position[2]), j.orientation, j.is_valid
                )
                for j in item.joints
            ),
        )
        for item in synth.frames(60)
    ]
    from full_body_acceptance.checks.geometry import UpAxis

    assert status_of(UpAxis, flattened) is Status.INSUFFICIENT_DATA


def test_a_zero_length_bone_is_reported_as_derived_not_broken():
    """Back-filled hands and feet keep a constant zero length; that is not a fault."""
    from full_body_acceptance.checks.geometry import BoneLengthConstancy
    from full_body_acceptance.profile import FULL_BODY

    wrist = FULL_BODY.index("LEFT_WRIST")
    hand = FULL_BODY.index("LEFT_HAND")
    frame_list = []
    for item in synth.frames(60):
        assert item.joints is not None
        joints = list(item.joints)
        joints[hand] = synth.joint(
            joints[wrist].position, joints[wrist].orientation, True
        )
        frame_list.append(synth.replace(item, joints=tuple(joints)))

    outcome = drive(BoneLengthConstancy(), frame_list)
    assert outcome.status is Status.PASS
    assert "LEFT_WRIST->LEFT_HAND" in outcome.measurements["derived_bones"]


def test_stature_declines_rather_than_guessing_when_a_chain_joint_is_never_valid():
    """A vendor that never reports the neck leaves the size unknowable, not wrong."""
    from full_body_acceptance.checks.geometry import (
        AnthropometricPlausibility,
        PositionScaleMetres,
    )
    from full_body_acceptance.profile import FULL_BODY

    neck = FULL_BODY.index("NECK")
    frame_list = [
        synth.with_joint(item, neck, synth.joint(is_valid=False))
        for item in synth.frames(60)
    ]
    assert status_of(PositionScaleMetres, frame_list) is Status.INSUFFICIENT_DATA
    assert status_of(AnthropometricPlausibility, frame_list) is Status.INSUFFICIENT_DATA


def test_a_plausible_synthetic_skeleton_reads_as_human():
    from full_body_acceptance.checks.geometry import (
        AnthropometricPlausibility,
        PositionScaleMetres,
    )

    outcome = drive(PositionScaleMetres(), synth.frames(60))
    assert outcome.status is Status.PASS
    assert 1.2 < outcome.measurements["skeletal_height"] < 2.2

    outcome = drive(AnthropometricPlausibility(), synth.frames(60))
    assert outcome.status is Status.PASS
    ratios = outcome.measurements["forearm_over_upper_arm"]
    assert set(ratios) == {"left", "right"}
    assert all(0.6 < r < 1.15 for r in ratios.values())


# --- orientation vs position -------------------------------------------------------


def test_a_consistent_moving_skeleton_passes_both_frame_checks():
    from full_body_acceptance.checks.orientation import (
        ComponentOrder,
        PositionOrientationSameFrame,
    )

    waving = synth.waving_frames()
    assert status_of(PositionOrientationSameFrame, waving) is Status.PASS
    assert status_of(ComponentOrder, waving) is Status.PASS


def test_rotating_positions_without_orientations_breaks_the_frame_check():
    from full_body_acceptance.checks.orientation import PositionOrientationSameFrame

    turn = synth.unit_quaternion(math.radians(90), (0.0, 1.0, 0.0))
    half_converted = [
        synth.replace(
            item,
            joints=tuple(
                synth.joint(synth.qrot(turn, j.position), j.orientation, j.is_valid)
                for j in item.joints
            ),
        )
        for item in synth.waving_frames()
    ]
    outcome = drive(PositionOrientationSameFrame(), half_converted)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["median_relative_spread"] > 0.1


def test_components_written_wxyz_are_identified_as_such():
    from full_body_acceptance.checks.orientation import (
        ComponentOrder,
        PositionOrientationSameFrame,
    )

    def to_wxyz(q):
        x, y, z, w = q
        return (w, x, y, z)

    misordered = [
        synth.replace(
            item,
            joints=tuple(
                synth.joint(j.position, to_wxyz(j.orientation), j.is_valid)
                for j in item.joints
            ),
        )
        for item in synth.waving_frames()
    ]
    outcome = drive(ComponentOrder(), misordered)
    assert outcome.status is Status.FAIL
    assert (
        outcome.measurements["median_relative_spread_if_wxyz"]
        < outcome.measurements["median_relative_spread"]
    )

    # The same symptom must be claimed once, by the check that explains it.
    assert status_of(PositionOrientationSameFrame, misordered) is Status.PASS


def test_a_held_pose_cannot_answer_the_frame_questions():
    """The fixture set's own principle: a T-pose alone reveals nothing about frames."""
    from full_body_acceptance.checks.orientation import (
        ComponentOrder,
        PositionOrientationSameFrame,
    )

    still = synth.frames(300)
    assert status_of(PositionOrientationSameFrame, still) is Status.INSUFFICIENT_DATA
    assert status_of(ComponentOrder, still) is Status.INSUFFICIENT_DATA


def test_an_unanswerable_conditional_check_does_not_block_the_verdict():
    """A held pose without motion labels leaves the frame and G4 questions open."""
    report = run(synth.StubSource(synth.frames(300)))
    assert report.verdict is Verdict.PASS
    unanswered = {r.name for r in report.unanswered}
    assert {
        "consistency.position_orientation_same_frame",
        "quaternion.component_order",
    } <= unanswered
    # The verdict only survives because none of them is a required check.
    assert not [r for r in report.unanswered if r.required]


def test_a_whole_rig_expressed_z_up_is_still_internally_consistent():
    """Z-up rotates positions and orientations together; coordinate_frame owns it."""
    from full_body_acceptance.checks.orientation import PositionOrientationSameFrame

    turn = synth.unit_quaternion(math.radians(90), (1.0, 0.0, 0.0))
    tipped = [
        synth.replace(
            item,
            joints=tuple(
                synth.joint(
                    synth.qrot(turn, j.position),
                    synth.qmul(turn, j.orientation),
                    j.is_valid,
                )
                for j in item.joints
            ),
        )
        for item in synth.waving_frames()
    ]
    assert status_of(PositionOrientationSameFrame, tipped) is Status.PASS


def test_a_clean_rig_is_right_handed_and_correctly_labelled():
    from full_body_acceptance.checks.geometry import (
        Handedness,
        JointIndexAssignment,
        LeftRightLabelling,
    )

    clean = synth.waving_frames()
    assert status_of(Handedness, clean) is Status.PASS
    assert status_of(LeftRightLabelling, clean) is Status.PASS
    assert status_of(JointIndexAssignment, clean) is Status.PASS


def test_a_mirrored_rig_is_left_handed():
    from full_body_acceptance.checks.geometry import Handedness

    assert status_of(Handedness, synth.mirrored(synth.waving_frames())) is Status.FAIL


def test_labels_on_the_wrong_side_of_the_body_are_caught():
    from full_body_acceptance.checks.geometry import LeftRightLabelling

    swapped = synth.swap_left_right(synth.waving_frames())
    assert status_of(LeftRightLabelling, swapped) is Status.FAIL


def test_the_labelling_check_ignores_orientation_faults():
    """Its facing reference is the feet, so a quaternion bug cannot implicate labels.

    Without this the check blames the labels on every recording whose orientations are
    wrong for an unrelated reason, and the submitter chases the wrong defect.
    """
    from full_body_acceptance.checks.geometry import LeftRightLabelling

    def to_wxyz(q):
        x, y, z, w = q
        return (w, x, y, z)

    garbled = [
        synth.replace(
            item,
            joints=tuple(
                synth.joint(j.position, to_wxyz(j.orientation), j.is_valid)
                for j in item.joints
            ),
        )
        for item in synth.waving_frames()
    ]
    assert status_of(LeftRightLabelling, garbled) is Status.PASS


def test_feet_pinned_to_the_origin_do_not_become_the_facing_reference():
    """Two feet at the world origin point at each other, not forward."""
    from full_body_acceptance.checks.geometry import LeftRightLabelling

    feet = (synth.FULL_BODY.index("LEFT_FOOT"), synth.FULL_BODY.index("RIGHT_FOOT"))
    pinned = []
    for item in synth.waving_frames():
        joints = list(item.joints)
        for index in feet:
            joints[index] = synth.joint(
                (0.0, 0.0, 0.0), joints[index].orientation, True
            )
        pinned.append(synth.replace(item, joints=tuple(joints)))

    result = LeftRightLabelling()
    for item in pinned:
        result.update(item)
    outcome = result.result()
    assert outcome.status is Status.PASS
    assert outcome.measurements["forward_from"] == ["pelvis_orientation"]


def test_two_joints_in_each_others_indices_are_caught():
    from full_body_acceptance.checks.geometry import JointIndexAssignment

    permuted = synth.swap_indices(synth.waving_frames(), "SPINE1", "SPINE2")
    assert status_of(JointIndexAssignment, permuted) is Status.FAIL


def test_a_limb_folded_for_the_whole_session_is_not_a_swapped_index():
    """An A-pose puts every elbow nearer the pelvis than its shoulder, legitimately.

    Judging joint order by distance to the root alone reads this as a permanent
    inversion, so the check measures the bone's own direction instead.
    """
    from full_body_acceptance.checks.geometry import JointIndexAssignment

    assert status_of(JointIndexAssignment, synth.arms_down_frames()) is Status.PASS


def test_a_bone_seen_only_briefly_is_not_judged():
    """A short burst of validity cannot separate a reversal from a held posture.

    Guards the whole-session case: validity that decays away leaves some bones with a
    handful of samples, all drawn from whatever pose the subject happened to hold.
    """
    from full_body_acceptance.checks.geometry import JointIndexAssignment

    spine2 = synth.FULL_BODY.index("SPINE2")
    permuted = synth.swap_indices(synth.waving_frames(300), "SPINE1", "SPINE2")
    briefly_valid = []
    for position, item in enumerate(permuted):
        joints = list(item.joints)
        joints[spine2] = synth.joint(
            joints[spine2].position, joints[spine2].orientation, is_valid=position < 100
        )
        briefly_valid.append(synth.replace(item, joints=tuple(joints)))

    outcome = drive(JointIndexAssignment(), briefly_valid)
    assert outcome.status is Status.PASS
    assert "SPINE1->SPINE2" not in outcome.measurements["reversed_bones"]


# --- solved endpoints on an otherwise rigid rig ------------------------------------


def _stretch_forearm(frames_, joint="LEFT_WRIST"):
    """Lengthens one bone progressively, leaving every other bone alone.

    This is what a three-tracker PICO does to the forearm: the solver has to reach the
    controller, so the reach error lands in the last segment.
    """
    from full_body_acceptance.profile import FULL_BODY

    index = FULL_BODY.index(joint)
    out = []
    for step, base in enumerate(frames_):
        assert base.joints is not None
        pose = base.joints[index]
        moved = synth.joint(
            position=(
                pose.position[0],
                pose.position[1] - 0.004 * step,
                pose.position[2],
            ),
            orientation=pose.orientation,
        )
        out.append(synth.with_joint(base, index, moved))
    return out


def test_a_bone_that_stretches_on_a_rigid_rig_is_reported_as_derived():
    from full_body_acceptance.checks.geometry import BoneLengthConstancy

    outcome = drive(BoneLengthConstancy(), _stretch_forearm(synth.frames(60)))
    assert outcome.status is Status.PASS
    # Moving the wrist moves the hand hanging off it, so both bones read as solved.
    assert outcome.measurements["solved_bones"] == [
        "LEFT_ELBOW->LEFT_WRIST",
        "LEFT_WRIST->LEFT_HAND",
    ]
    assert "derived rather than measured" in outcome.detail


def test_a_solved_forearm_is_not_judged_for_its_proportions():
    from full_body_acceptance.checks.geometry import AnthropometricPlausibility

    outcome = drive(AnthropometricPlausibility(), _stretch_forearm(synth.frames(60)))
    assert outcome.status is Status.PASS
    assert outcome.measurements["derived_arms"] == ["left"]
    assert "right" in outcome.measurements["forearm_over_upper_arm"]
