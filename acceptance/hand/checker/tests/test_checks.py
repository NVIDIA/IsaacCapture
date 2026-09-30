# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One measurement per test, on the analytic hand in ``synth.py``."""

from __future__ import annotations

import math

import flatbuffers
import synth

from hand_acceptance._schema import GENERATED_DIR  # noqa: F401  (puts core on sys.path)
from hand_acceptance.checks import build
from hand_acceptance.checks.base import Status
from hand_acceptance.labels import Step, StepTimeline
from hand_acceptance.mcap_source import decode_record, side_of


def _run(name: str, frames, timeline=None):
    (check,) = build([name], timeline)
    for frame in frames:
        check.update(frame)
    return check.result()


def both(count: int = 120, **kwargs):
    return synth.take(count, "left", **kwargs) + synth.take(count, "right", **kwargs)


def test_a_missing_hand_leaves_a_per_hand_check_unanswered():
    outcome = _run("units.position_scale_metres", synth.take(60, "right"))
    assert outcome.status is Status.INSUFFICIENT_DATA
    assert outcome.detail.startswith("left:")


def test_one_failing_hand_fails_the_check_and_is_named():
    frames = synth.take(60, "left") + synth.take(60, "right", radius=0.0)
    outcome = _run("joints.radius_plausible", frames)
    assert outcome.status is Status.FAIL
    assert outcome.detail.startswith("right:")
    assert outcome.measurements["left"]["bad_rate"] == 0.0


def test_handedness_reads_each_side():
    assert _run("coordinate_frame.handedness", both()).status is Status.PASS


def test_handedness_fails_a_left_hand_on_the_right_channel():
    frames = synth.take(120, "left") + synth.take(
        120, "right", transform=synth.mirrored
    )
    outcome = _run("coordinate_frame.handedness", frames)
    assert outcome.status is Status.FAIL
    assert "right channel carries a left hand" in outcome.detail


def test_component_order_catches_wxyz_on_a_turned_hand():
    turned = synth.axis_angle((1.0, 0.7, 0.2), math.radians(70))
    frames = both(orientation=turned)
    assert _run("quaternion.component_order", frames).status is Status.PASS

    def scramble(frame):
        joints = tuple(
            type(j)(
                j.position, (j.orientation[3], *j.orientation[:3]), j.is_valid, j.radius
            )
            for j in frame.joints
        )
        return type(frame)(**{**frame.__dict__, "joints": joints})

    outcome = _run("quaternion.component_order", [scramble(f) for f in frames])
    assert outcome.status is Status.FAIL


def test_flex_direction_needs_the_tips_to_go_palmward():
    assert _run("shape.flex_direction", both()).status is Status.PASS

    def reverse(local):
        return [((p[0], -p[1], p[2]), q) for p, q in local]

    outcome = _run("shape.flex_direction", both(transform=reverse))
    assert outcome.status is Status.FAIL


def test_a_brief_valid_set_flicker_is_not_a_source_switch():
    frames = synth.take(300, "right") + synth.take(300, "left")
    required = set(range(26)) - {0}
    for i in range(100, 110):
        frames[i] = synth.frame(i, "right", valid=required)
    assert _run("consistency.valid_set_stable", frames).status is Status.PASS

    for i in range(150, 300):
        frames[i] = synth.frame(i, "right", valid=required)
    outcome = _run("consistency.valid_set_stable", frames)
    assert outcome.status is Status.FAIL
    assert outcome.measurements["right"]["switches"][0]["lost"] == ["PALM"]


def test_centimetres_fail_the_scale_check():
    def cm(local):
        return [(tuple(100 * c for c in p), q) for p, q in local]

    outcome = _run("units.position_scale_metres", both(transform=cm))
    assert outcome.status is Status.FAIL


def test_swapped_fingers_fail_index_assignment():
    def swap(local):
        local = list(local)
        for a, b in zip(range(7, 11), range(12, 16)):
            local[a], local[b] = local[b], local[a]
        return local

    assert _run("skeleton.joint_index_assignment", both()).status is Status.PASS
    outcome = _run("skeleton.joint_index_assignment", both(transform=swap))
    assert outcome.status is Status.FAIL


def test_window_measurements_skip_the_unsettled_lead():
    """A fist reached only after 40% of the window is still measured as a fist."""
    closed = synth.right_local(math.radians(80))
    frames = []
    for tick in range(100):
        local = synth.right_local(0.0) if tick < 35 else closed
        frames += [
            synth.frame(tick, "left", local=local),
            synth.frame(tick, "right", local=local),
        ]
    start = frames[0].sample_time_ns
    timeline = StepTimeline(
        steps=(Step(0, "fist", start, start + 100 * synth.PERIOD_NS, True),)
    )
    assert _run("posture.fist_closure", frames, timeline).status is Status.PASS


def test_the_decoder_reads_the_struct_layout_flatc_writes():
    from core import DeviceDataTimestamp, HandJoints, HandPose, HandPoseRecord

    b = flatbuffers.Builder(1100)
    HandPose.Start(b)
    joints = HandJoints.CreateHandJoints(
        b,
        [float(i) for i in range(26)],
        [0.5] * 26,
        [-1.0] * 26,
        [0.0] * 26,
        [0.0] * 26,
        [0.0] * 26,
        [1.0] * 26,
        [i % 2 == 0 for i in range(26)],
        [0.001 * i for i in range(26)],
    )
    HandPose.AddJoints(b, joints)
    data = HandPose.End(b)
    HandPoseRecord.Start(b)
    HandPoseRecord.AddTimestamp(
        b, DeviceDataTimestamp.CreateDeviceDataTimestamp(b, 3, 2, 1)
    )
    HandPoseRecord.AddData(b, data)
    b.Finish(HandPoseRecord.End(b))

    frame = decode_record(bytes(b.Output()), 0, 3, 3, side_of("hands/right_hand"))
    assert frame.side == "right"
    assert frame.joints[7].position == (7.0, 0.5, -1.0)
    assert [j.is_valid for j in frame.joints[:3]] == [True, False, True]
    assert math.isclose(frame.joints[25].radius, 0.025, rel_tol=1e-6)
    assert (frame.available_time_ns, frame.sample_time_ns, frame.device_time_ns) == (
        3,
        2,
        1,
    )


def test_side_comes_from_the_topic_suffix():
    assert side_of("anything/left_hand") == "left"
    assert side_of("hands/right_hand") == "right"
    assert side_of("hands/head") is None
