# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One measurement per test, on the analytic hand in ``synth.py``."""

from __future__ import annotations

import math
import random
from dataclasses import replace

import flatbuffers
import pytest
import synth

from hand_cts._schema import GENERATED_DIR  # noqa: F401  (puts core on sys.path)
from hand_cts.checks import build
from hand_cts.checks.base import Status, fit_plane
from hand_cts.labels import Step, StepTimeline
from hand_cts.mcap_source import decode_record, side_of


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


def test_flex_direction_ignores_a_bend_at_the_knuckle():
    """The straightest frames here are straight fingers folded 60 deg at the knuckle."""
    frames = []
    for side in ("left", "right"):
        for tick in range(120):
            closed = (1 - math.cos(2 * math.pi * tick / 120)) / 2
            local = synth.right_local(
                curl=math.radians(70) * closed, knuckle=math.radians(60) * (1 - closed)
            )
            frames.append(synth.frame(tick, side, local=local))
    outcome = _run("shape.flex_direction", frames)
    assert outcome.status is Status.PASS, outcome.detail


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


def _thumb_proximal_swung(local):
    local = list(local)
    local[3] = ((0.012, -0.02, -0.05), local[3][1])
    return local


def _thumb_chain_on_little_side(local):
    local = list(local)
    for joint in (2, 3, 4, 5):
        (x, y, z), q = local[joint]
        local[joint] = ((x + 0.07, y, z), q)
    return local


def _in_some_frames(transform, swung_from: int):
    frames = []
    for side in ("left", "right"):
        for tick in range(100):
            local = synth.right_local(0.3)
            frames.append(
                synth.frame(
                    tick, side, local=transform(local) if tick >= swung_from else local
                )
            )
    return frames


def test_a_thumb_proximal_swinging_toward_the_little_knuckle_is_not_misassigned():
    frames = _in_some_frames(_thumb_proximal_swung, swung_from=85)
    outcome = _run("skeleton.joint_index_assignment", frames)
    assert outcome.status is Status.PASS, outcome.detail


def test_a_thumb_chain_on_the_little_side_fails_index_assignment():
    outcome = _run(
        "skeleton.joint_index_assignment",
        both(transform=_thumb_chain_on_little_side),
    )
    assert outcome.status is Status.FAIL
    assert "thumb" in outcome.detail


def test_swapped_knuckles_are_named_as_knuckle_order():
    def swap(local):
        local = list(local)
        for a, b in zip(range(7, 11), range(12, 16)):
            local[a], local[b] = local[b], local[a]
        return local

    outcome = _run("skeleton.joint_index_assignment", both(transform=swap))
    assert outcome.status is Status.FAIL
    assert "knuckles are in" in outcome.detail
    assert "thumb" not in outcome.detail


def test_a_valid_fingertip_stuck_at_the_origin_fails():
    def stuck(frame):
        joints = list(frame.joints)
        joints[10] = replace(joints[10], position=(0.0, 0.0, 0.0))
        return replace(frame, joints=tuple(joints))

    name = "values.zero_pose_on_valid_joint"
    assert _run(name, both()).status is Status.PASS
    outcome = _run(name, [stuck(f) for f in both()])
    assert outcome.status is Status.FAIL
    assert outcome.detail.startswith("left:")


def _noisy(sigma_m: float):
    rng = random.Random(7)

    def jitter(local, _tick=None):
        return [(tuple(c + rng.gauss(0.0, sigma_m) for c in p), q) for p, q in local]

    return jitter


def test_bone_length_constancy_tolerates_two_millimetres_of_noise():
    name = "skeleton.bone_length_constancy"
    outcome = _run(name, both(transform=_noisy(0.002)))
    assert outcome.status is Status.PASS, outcome.detail
    assert _run(name, both(transform=_noisy(0.008))).status is Status.FAIL


def test_a_slow_stream_fails_the_minimum_rate(monkeypatch):
    name = "rate.minimum_rate"
    assert _run(name, both(60)).status is Status.PASS
    monkeypatch.setattr(synth, "PERIOD_NS", synth.PERIOD_NS * 8)
    outcome = _run(name, both(60))
    assert outcome.status is Status.FAIL
    assert "7.5 Hz" in outcome.detail


def test_too_few_frames_leave_the_minimum_rate_unanswered():
    outcome = _run("rate.minimum_rate", both(5))
    assert outcome.status is Status.INSUFFICIENT_DATA


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


def _scripted_take(fist_twist_deg: float, rolling: dict[str, str]):
    """Flat, fist, right roll, left roll windows of 120 frames; each side turns 150 deg
    in the window ``rolling[side]`` and the fist window twists ``fist_twist_deg``."""
    labels = ("flat_on_table_open", "fist", "right_tip_roll", "left_tip_roll")
    start = 1_000_000_000
    steps = tuple(
        Step(
            i,
            label,
            start + i * 120 * synth.PERIOD_NS,
            start + (i + 1) * 120 * synth.PERIOD_NS,
            label in ("flat_on_table_open", "fist"),
        )
        for i, label in enumerate(labels)
    )
    frames = []
    for side in ("left", "right"):
        for tick in range(4 * 120):
            window, k = divmod(tick, 120)
            label, progress = labels[window], k / 120
            curl = 1.3 if label == "fist" else 0.0
            if label == "fist" and progress >= 0.4:
                angle = fist_twist_deg * math.sin(math.pi * (progress - 0.4) / 0.6) ** 2
            elif label == rolling[side]:
                angle = 150.0 * math.sin(math.pi * progress) ** 2
            else:
                angle = 0.0
            frames.append(
                synth.frame(
                    tick,
                    side,
                    local=synth.right_local(curl),
                    orientation=synth.axis_angle((0, 0, 1), math.radians(angle)),
                )
            )
    return frames, StepTimeline(steps=steps)


OWN_ROLLS = {"left": "left_tip_roll", "right": "right_tip_roll"}


def test_a_wrist_twist_during_the_fist_does_not_reorder_the_script():
    frames, timeline = _scripted_take(155.0, OWN_ROLLS)
    outcome = _run("segmentation.step_order_matches_labels", frames, timeline)
    assert outcome.status is Status.PASS, outcome.detail


def test_rolls_performed_in_swapped_windows_fail_the_order_check():
    frames, timeline = _scripted_take(
        0.0, {"left": "right_tip_roll", "right": "left_tip_roll"}
    )
    outcome = _run("segmentation.step_order_matches_labels", frames, timeline)
    assert outcome.status is Status.FAIL
    assert "turns more than" in outcome.detail


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


def test_plane_fit_recovers_a_known_tilt():
    tilt = math.radians(7.0)
    points = [
        (x, 0.75 + x * math.tan(tilt), z)
        for x in (-0.04, -0.02, 0.0, 0.02, 0.04)
        for z in (-0.08, -0.04, 0.0, 0.04)
    ]
    _, normal, residuals = fit_plane(points)
    assert math.degrees(math.acos(normal[1])) == pytest.approx(7.0, abs=0.01)
    assert max(abs(r) for r in residuals) < 1e-9


def _flat_pair(ticks: int, right_lift: float = 0.0, right_offset_ns: int = 0):
    flat = synth.right_local(0.0)
    frames = []
    for tick in range(ticks):
        frames.append(synth.frame(tick, "left", local=flat, wrist=(-0.2, 0.75, -0.3)))
        right = synth.frame(
            tick, "right", local=flat, wrist=(0.2, 0.75 + right_lift, -0.3)
        )
        frames.append(
            replace(right, sample_time_ns=right.sample_time_ns + right_offset_ns)
        )
    start = 1_000_000_000
    timeline = StepTimeline(
        steps=(
            Step(0, "flat_on_table_open", start, start + ticks * synth.PERIOD_NS, True),
        )
    )
    return frames, timeline


def test_hands_on_one_table_read_one_height():
    frames, timeline = _flat_pair(60)
    assert _run("table.hands_same_height", frames, timeline).status is Status.PASS
    frames, timeline = _flat_pair(60, right_lift=0.05)
    assert _run("table.hands_same_height", frames, timeline).status is Status.FAIL


def test_cross_hand_checks_pair_only_records_close_in_time():
    frames, timeline = _flat_pair(60, right_offset_ns=40_000_000)
    outcome = _run("table.hands_same_height", frames, timeline)
    assert outcome.status is Status.INSUFFICIENT_DATA
    assert "close enough in time" in outcome.detail


def _palms_pair(ticks: int, right_offset_ns: int = 0):
    """Palms facing, 2 cm apart, over a ``palms_together`` window."""
    upright = synth.axis_angle((1.0, 0.0, 0.0), math.pi / 2)
    flipped = synth.multiply(synth.axis_angle((0.0, 1.0, 0.0), math.pi), upright)
    frames = []
    for tick in range(ticks):
        frames.append(
            synth.frame(tick, "left", wrist=(0.0, 1.1, -0.32), orientation=flipped)
        )
        right = synth.frame(tick, "right", wrist=(0.0, 1.1, -0.3), orientation=upright)
        frames.append(
            replace(right, sample_time_ns=right.sample_time_ns + right_offset_ns)
        )
    start = 1_000_000_000
    timeline = StepTimeline(
        steps=(Step(0, "palms_together", start, start + ticks * synth.PERIOD_NS, True),)
    )
    return frames, timeline


def test_a_device_stamping_each_hand_separately_still_pairs():
    frames, timeline = _flat_pair(60, right_offset_ns=6_000_000)
    assert _run("table.hands_same_height", frames, timeline).status is Status.PASS

    frames, timeline = _palms_pair(60)
    same = _run("palms.fingertip_gap", frames, timeline)
    assert same.status is Status.PASS, same.detail
    frames, timeline = _palms_pair(60, right_offset_ns=6_000_000)
    assert _run("palms.fingertip_gap", frames, timeline).status is Status.PASS

    frames, timeline = _roll_take(right_offset_ns=6_000_000)
    outcome = _run("tips.roll_contact", frames, timeline)
    assert outcome.status is Status.PASS, outcome.detail


def test_palms_that_cannot_be_paired_are_unanswered_not_failed():
    frames, timeline = _palms_pair(60, right_offset_ns=40_000_000)
    outcome = _run("palms.fingertip_gap", frames, timeline)
    assert outcome.status is Status.INSUFFICIENT_DATA, outcome.detail
    assert "close enough in time" in outcome.detail


def _roll_take(right_offset_ns: int = 0):
    """The left hand turns 150 deg about its middle fingertip, held against the right's."""
    tip = 15
    left_tip_local = synth.mirrored(synth.right_local())[tip][0]
    held_tip = (-0.02, 1.2, -0.3)
    right_wrist = (0.18, 1.2, -0.3)
    right_tip = synth.frame(0, "right", wrist=right_wrist).joints[tip].position
    offset = tuple(a - b for a, b in zip(held_tip, right_tip))
    right_wrist = tuple(
        w + o + g for w, o, g in zip(right_wrist, offset, (0.04, 0.0, 0.0))
    )
    frames = []
    for tick in range(180):
        angle = 0.0 if tick < 60 else math.radians(150.0) * (tick - 60) / 119
        q = synth.axis_angle((1.0, 0.0, 0.0), angle)
        wrist = tuple(h - r for h, r in zip(held_tip, synth.rotate(q, left_tip_local)))
        frames.append(synth.frame(tick, "left", wrist=wrist, orientation=q))
        right = synth.frame(tick, "right", wrist=right_wrist)
        frames.append(
            replace(right, sample_time_ns=right.sample_time_ns + right_offset_ns)
        )
    start = 1_000_000_000
    timeline = StepTimeline(
        steps=(
            Step(0, "tips_together", start, start + 60 * synth.PERIOD_NS, True),
            Step(
                1,
                "left_tip_roll",
                start + 60 * synth.PERIOD_NS,
                start + 180 * synth.PERIOD_NS,
                False,
            ),
        )
    )
    return frames, timeline


def test_a_left_roll_is_measured_against_the_same_reference_as_a_right_roll():
    """The still tips_together gap must cancel, whichever hand turns."""
    frames, timeline = _roll_take()
    outcome = _run("tips.roll_contact", frames, timeline)
    assert outcome.status is Status.PASS, outcome.detail
    rolls = outcome.measurements["rolls"]
    assert rolls["left_tip_roll"]["excursion_p95_m"] < 0.002
    assert outcome.measurements["tips_together_gap_m"] > 0.03
