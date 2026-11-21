# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generates every hand fixture, its label sidecar, and ``fixtures_index.json``.

Deterministic to the byte: fixed seeds, no wall clock, no compression.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Callable

import hand_model as hm
import mcap_io
import script
from script import HandSample, Joint, Options, Window

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

Transform = Callable[[HandSample], None]

OPTIONAL = (hm.PALM, 6, 11, 16, 21)
INVALID = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _invalidate(sample: HandSample, indices) -> None:
    if sample.joints is None:
        return
    for i in indices:
        sample.joints[i] = Joint(*INVALID, valid=False, radius=0.0)


def glove_joint_set(sample: HandSample) -> None:
    _invalidate(sample, OPTIONAL)


def palm_invalid(sample: HandSample) -> None:
    _invalidate(sample, (hm.PALM,))


def palm_on_little_tip(sample: HandSample) -> None:
    tip = sample.joints[25]
    sample.joints[hm.PALM] = replace(tip, radius=hm.RADIUS["palm"])


def swap_index_middle(sample: HandSample) -> None:
    js = sample.joints
    for a, b in zip(hm.FINGER_JOINTS["index"][1:], hm.FINGER_JOINTS["middle"][1:]):
        js[a], js[b] = js[b], js[a]


def centimetres(sample: HandSample) -> None:
    for j in sample.joints:
        j.position = hm.scale(j.position, 100.0)


def wxyz(sample: HandSample) -> None:
    for j in sample.joints:
        x, y, z, w = j.orientation
        j.orientation = (w, x, y, z)


def at_origin(sample: HandSample) -> None:
    """No wrist source: the hand is placed by the identity transform."""
    wrist = sample.joints[hm.WRIST]
    inverse = (
        -wrist.orientation[0],
        -wrist.orientation[1],
        -wrist.orientation[2],
        wrist.orientation[3],
    )
    origin = wrist.position
    for j in sample.joints:
        j.position = hm.quat_rotate(inverse, hm.sub(j.position, origin))
        j.orientation = hm.quat_mul(inverse, j.orientation)


def _stale_wrist(hold_s: float, after_s: float) -> Transform:
    """Places each hand by a wrist pose refreshed only every ``hold_s`` seconds."""
    held: dict[str, tuple] = {}

    def apply(sample: HandSample) -> None:
        wrist = sample.joints[hm.WRIST]
        key = sample.side
        slot = int(sample.t // hold_s)
        if sample.t < after_s:
            return
        if key not in held or held[key][0] != slot:
            held[key] = (slot, wrist.position, wrist.orientation)
        _, position, orientation = held[key]
        # Copied out first: the loop below rewrites the WRIST joint itself.
        here = wrist.position
        inverse = hm.quat_conj(wrist.orientation)
        for j in sample.joints:
            local_p = hm.quat_rotate(inverse, hm.sub(j.position, here))
            local_q = hm.quat_mul(inverse, j.orientation)
            j.position = hm.add(position, hm.quat_rotate(orientation, local_p))
            j.orientation = hm.quat_mul(orientation, local_q)
        sample.joints[hm.WRIST] = Joint(position, orientation, True, wrist.radius)

    return apply


def _source_switch(at_s: float) -> Transform:
    def apply(sample: HandSample) -> None:
        if sample.t < at_s:
            glove_joint_set(sample)

    return apply


def _required_dropout(joint: int) -> Transform:
    def apply(sample: HandSample) -> None:
        if int(sample.t * 2) % 3 == 0:
            _invalidate(sample, (joint,))

    return apply


def unrolled_wrist(sample: HandSample) -> None:
    """The wrist quaternion misses the forearm roll its own joint positions show."""
    if sample.wrist_roll == 0.0:
        return
    sense = 1.0 if sample.side == "right" else -1.0
    undo = hm.quat_axis_angle((0.0, 0.0, 1.0), -sense * sample.wrist_roll)
    wrist = sample.joints[hm.WRIST]
    wrist.orientation = hm.quat_mul(wrist.orientation, undo)


def _right_only(apply: Transform) -> Transform:
    def wrapped(sample: HandSample) -> None:
        if sample.side == "right":
            apply(sample)

    return wrapped


def _hand_frame_offset(offset: hm.Vec) -> Transform:
    """Every joint displaced by a fixed offset in the wrist's frame, so it turns with
    the hand: a tracker-to-wrist offset the plugin has wrong."""

    def apply(sample: HandSample) -> None:
        shift = hm.quat_rotate(sample.joints[hm.WRIST].orientation, offset)
        for j in sample.joints:
            j.position = hm.add(j.position, shift)

    return apply


def _world_shift(offset: hm.Vec) -> Transform:
    def apply(sample: HandSample) -> None:
        for j in sample.joints:
            j.position = hm.add(j.position, offset)

    return apply


def _scaled_about_wrist(factor: float) -> Transform:
    """A hand model larger than the performer's hand."""

    def apply(sample: HandSample) -> None:
        wrist = sample.joints[hm.WRIST].position
        for j in sample.joints:
            j.position = hm.add(wrist, hm.scale(hm.sub(j.position, wrist), factor))

    return apply


def _world_turned(axis: hm.Vec, degrees: float) -> Transform:
    """The whole stage turned about the origin, as a wrong world up would read."""
    turn = hm.quat_axis_angle(axis, math.radians(degrees))

    def apply(sample: HandSample) -> None:
        for j in sample.joints:
            j.position = hm.quat_rotate(turn, j.position)
            j.orientation = hm.quat_mul(turn, j.orientation)

    return apply


@dataclass
class Fixture:
    name: str
    category: str
    description: str
    expected_verdict: str
    options: Options = field(default_factory=Options)
    transforms: tuple[Transform, ...] = ()
    sides: tuple[str, ...] = ("left", "right")
    expected_failing_check: str | None = None
    expected_advisory: str | None = None


FIXTURE_SET: tuple[Fixture, ...] = (
    Fixture(
        "golden_full_script",
        "golden",
        "The full two-hand script at 60 Hz, all 26 joints valid, optical-hand style.",
        "pass",
    ),
    Fixture(
        "golden_glove_21",
        "golden",
        "The full script with PALM and the four finger METACARPALs invalid throughout, "
        "as a glove that only drives the 21 judged joints publishes it.",
        "pass",
        transforms=(glove_joint_set,),
    ),
    Fixture(
        "benign_optical_25",
        "benign",
        "PALM invalid throughout, the 25-joint layout a WebXR hand arrives in.",
        "pass",
        transforms=(palm_invalid,),
    ),
    Fixture(
        "benign_palm_on_little_tip",
        "benign",
        "PALM is valid but sits on LITTLE_TIP. An advisory note, never a failure: "
        "PALM is not judged.",
        "pass",
        transforms=(palm_on_little_tip,),
        expected_advisory="optional_joints.geometry_consistent",
    ),
    Fixture(
        "benign_late_arrival",
        "benign",
        "The performer reaches each pose 1.0 s after its window opens, inside the "
        "unsettled lead of every window.",
        "pass",
        options=Options(arrival_delay_s=2.0),
    ),
    Fixture(
        "defect_finger_swap_index_middle",
        "defect",
        "INDEX and MIDDLE PROXIMAL..TIP joints exchanged on both hands.",
        "fail",
        transforms=(swap_index_middle,),
        expected_failing_check="skeleton.joint_index_assignment",
    ),
    Fixture(
        "defect_flex_reversed",
        "defect",
        "Every finger flexion angle negated: fingers curl toward the back of the hand.",
        "fail",
        options=Options(flex_sign=-1.0),
        expected_failing_check="shape.flex_direction",
    ),
    Fixture(
        "defect_right_channel_mirrored",
        "defect",
        "The right channel carries a left hand's geometry at the right wrist.",
        "fail",
        options=Options(mirrored=frozenset({"right"})),
        expected_failing_check="coordinate_frame.handedness",
    ),
    Fixture(
        "defect_wrist_frozen",
        "defect",
        "From 8 s on, both hands are placed by a wrist pose refreshed only every 2 s "
        "while the fingers keep moving.",
        "fail",
        transforms=(_stale_wrist(2.0, 8.0),),
        expected_failing_check="placement.wrist_not_frozen",
    ),
    Fixture(
        "defect_hand_at_origin",
        "defect",
        "No wrist source: both hands placed by the identity, wrist at the STAGE origin.",
        "fail",
        transforms=(at_origin,),
        expected_failing_check="placement.not_at_origin",
    ),
    Fixture(
        "defect_centimetres",
        "defect",
        "Every position written in centimetres.",
        "fail",
        transforms=(centimetres,),
        expected_failing_check="units.position_scale_metres",
    ),
    Fixture(
        "defect_quaternion_wxyz",
        "defect",
        "Every orientation written w,x,y,z into the x,y,z,w fields.",
        "fail",
        transforms=(wxyz,),
        expected_failing_check="quaternion.component_order",
    ),
    Fixture(
        "defect_source_switch",
        "defect",
        "21 joints valid (glove) until 30 s, then all 26 (headset optical fallback).",
        "fail",
        transforms=(_source_switch(30.0),),
        expected_failing_check="consistency.valid_set_stable",
    ),
    Fixture(
        "defect_required_joint_invalid",
        "defect",
        "INDEX_DISTAL invalid for 0.5 s out of every 1.5 s.",
        "fail",
        transforms=(_required_dropout(9),),
        expected_failing_check="coverage.required_joints_valid",
    ),
    Fixture(
        "defect_wrist_orientation_unrolled",
        "defect",
        "During each hand's fingertip roll the wrist quaternion misses the roll while "
        "the joint positions turn palm-out and back.",
        "fail",
        transforms=(unrolled_wrist,),
        expected_failing_check="consistency.position_orientation_same_frame",
    ),
    Fixture(
        "defect_wrist_offset_in_hand_frame",
        "defect",
        "Right hand displaced 18 mm dorsally in its own wrist frame: invisible while it "
        "holds still, an orbit of the touching fingertip while it rolls.",
        "fail",
        transforms=(_right_only(_hand_frame_offset((0.0, 0.018, 0.0))),),
        expected_failing_check="tips.roll_contact",
    ),
    Fixture(
        "defect_right_hand_raised",
        "defect",
        "Right hand shifted 35 mm up throughout.",
        "fail",
        transforms=(_right_only(_world_shift((0.0, 0.035, 0.0))),),
        expected_failing_check="table.hands_same_height",
    ),
    Fixture(
        "defect_right_hand_oversized",
        "defect",
        "Right hand scaled 30% larger than the performer's, about the wrist.",
        "fail",
        transforms=(_right_only(_scaled_about_wrist(1.3)),),
        expected_failing_check="palms.fingertip_gap",
    ),
    Fixture(
        "defect_world_tilted",
        "defect",
        "Both hands turned 20 deg about the stage X axis: the table reads tilted.",
        "fail",
        transforms=(_world_turned((1.0, 0.0, 0.0), 20.0),),
        expected_failing_check="table.level",
    ),
    Fixture(
        "defect_ring_finger_bent",
        "defect",
        "Every pose reports the ring finger 30 deg more flexed at the knuckle than the performer's hand.",
        "fail",
        options=Options(finger_bias={"ring": math.radians(30.0)}),
        expected_failing_check="table.fingers_coplanar",
    ),
    Fixture(
        "retake_shallow_fist",
        "retake",
        "The fist is only half closed: a performance miss, not a device fault.",
        "retake",
        options=Options(fist=hm.SHALLOW_FIST),
        expected_failing_check="posture.fist_closure",
    ),
)


def labels_sidecar(windows: list[Window], mcap_name: str) -> dict:
    return {
        "note": "synthetic fixture; windows are the generator's own schedule",
        "clock_domain": "sample_time_local_common_clock",
        "nominal_rate_hz": script.RATE_HZ,
        "mcap": mcap_name,
        "steps": [
            {
                "index": w.index,
                "label": w.label,
                "start_ns": w.start_ns,
                "end_ns": w.end_ns,
                "is_still_window": w.label not in script.ROLLS,
                "boundary_source": "timer",
            }
            for w in windows
        ],
    }


def build(fixture: Fixture, bfbs: bytes) -> dict:
    ticks, windows = script.synthesise(fixture.options)
    for _, left, right in ticks:
        for sample in (left, right):
            for transform in fixture.transforms:
                transform(sample)
    path = os.path.join(FIXTURES, f"{fixture.name}.mcap")
    records = mcap_io.write_mcap(path, ticks, bfbs)
    sidecar = os.path.join(FIXTURES, f"{fixture.name}.labels.json")
    with open(sidecar, "w") as fh:
        json.dump(labels_sidecar(windows, os.path.basename(path)), fh, indent=2)
        fh.write("\n")
    entry = {
        "filename": f"fixtures/{fixture.name}.mcap",
        "labels": f"fixtures/{fixture.name}.labels.json",
        "category": fixture.category,
        "description": fixture.description,
        "expected_verdict": fixture.expected_verdict,
        "records": records,
        "size_bytes": os.path.getsize(path),
    }
    if fixture.expected_failing_check:
        entry["expected_failing_check"] = fixture.expected_failing_check
    if fixture.expected_advisory:
        entry["expected_advisory"] = fixture.expected_advisory
    return entry


def main() -> None:
    os.makedirs(FIXTURES, exist_ok=True)
    bfbs = mcap_io.load_bfbs()
    entries = []
    for fixture in FIXTURE_SET:
        entries.append(build(fixture, bfbs))
        print(f"  {entries[-1]['filename']:<52} {entries[-1]['records']:>6} records")
    index = {
        "schema": {
            "mcap_schema_name": mcap_io.SCHEMA_NAME,
            "schema_encoding": mcap_io.ENCODING,
            "message_encoding": mcap_io.ENCODING,
            "topics": list(mcap_io.TOPICS.values()),
            "mcap_profile": mcap_io.PROFILE,
            "log_time": "available_time_local_common_clock",
        },
        "conventions": {
            "stage": "+Y up, the performer faces -Z",
            "joint_frame": "OpenXR: -Z along the bone toward the tip, +Y dorsal, X = Y x Z",
            "position_units": "metres",
            "quaternion_order": "x, y, z, w",
            "nominal_rate_hz": script.RATE_HZ,
        },
        "categories": {
            "golden": "clean and correct; the checker must pass",
            "benign": "unusual but correct; the checker must also pass",
            "defect": "one injected fault; the checker must fail the named check",
            "retake": "a performance miss; the checker must ask for a retake",
        },
        "verdicts": {
            "pass": "checker must accept",
            "fail": "a fault attributable to the device or integration",
            "retake": "a performance miss; the device is not implicated",
        },
        "motion_script": [
            {"index": i, "label": label, "hold_s": hold}
            for i, (label, hold) in enumerate(script.STEPS)
        ],
        "fixtures": entries,
    }
    with open(os.path.join(HERE, "fixtures_index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
