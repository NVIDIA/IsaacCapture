# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shapes the Pico tracker produces that the shared fixture set does not cover."""

from __future__ import annotations

import synth

from full_body_cts import McapFrameSource, run
from full_body_cts.report import Verdict


def test_invalid_joints_may_carry_arbitrary_values(tmp_path):
    """Pico copies the OpenXR pose through whatever the location flags say."""
    garbage = [
        synth.joint(
            position=(float("nan"), float("inf"), -0.0),
            orientation=(9.0, -3.0, 0.5, 0.0),
            is_valid=False,
        ),
        synth.joint(
            position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 0.0), is_valid=False
        ),
        synth.joint(
            position=(1e9, -1e9, 7.0), orientation=(0.0, 0.0, 0.0, 12.0), is_valid=False
        ),
    ]
    # LEFT_FOOT, RIGHT_FOOT and LEFT_HAND: endpoints a vendor may not provide, and
    # outside the chain the stature measurements need.
    targets = (10, 11, 22)
    frame_list = []
    for i in range(300):
        item = synth.frame(i)
        for target, bad in zip(targets, garbage):
            item = synth.with_joint(item, target, bad)
        frame_list.append(item)

    path = synth.write_recording(tmp_path / "pico_invalid_garbage.mcap", frame_list)
    report = run(McapFrameSource(path))

    assert report.verdict is Verdict.PASS, report.to_text()
    assert report.failures == ()


def test_registered_channel_with_no_messages_cannot_conclude(tmp_path):
    """A tracker that publishes nothing leaves the channel registered and empty."""
    path = synth.write_recording(tmp_path / "pico_no_messages.mcap", [])
    source = McapFrameSource(path)

    assert source.metadata.channel_found is True
    assert list(source) == []

    report = run(source)
    assert report.verdict is Verdict.INSUFFICIENT_DATA
    assert any("publishes no body data" in note for note in report.notes)


def test_a_file_without_the_full_body_schema_is_not_a_pass(tmp_path):
    path = synth.write_recording(
        tmp_path / "other_schema.mcap",
        synth.frames(5),
        schema_name="core.HandPoseRecord",
        topic="hand/hand",
    )
    report = run(McapFrameSource(path))

    assert report.metadata.channel_found is False
    assert report.verdict is Verdict.INSUFFICIENT_DATA
    assert any("no channel declares" in note for note in report.notes)


def test_channel_is_located_by_schema_name_not_topic(tmp_path):
    """The topic prefix is whatever ``name=`` the recording script passed."""
    path = synth.write_recording(
        tmp_path / "renamed_topic.mcap",
        synth.frames(300),
        topic="vendor_xyz_body/full_body",
    )
    report = run(McapFrameSource(path))

    assert report.metadata.topic == "vendor_xyz_body/full_body"
    assert report.frames == 300
    assert report.verdict is Verdict.PASS
