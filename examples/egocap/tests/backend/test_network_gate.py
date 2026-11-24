# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""network_check_block: when CaptureServer's reported delivery check lets a take start."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capture_server import protocol as p
from egocap_web.network_gate import network_check_block
from egocap_web.network_gate import tested_tracks as measured_tracks
from egocap_web.preview_protocol import NativeFacts
from tests.support.network_check import (
    capture_status,
    check_status,
    livekit_stream,
    native_facts,
    status_snapshot,
)

State = p.NetworkCheckState


def test_a_ready_check_for_the_offered_cameras_allows_a_take():
    assert network_check_block(native_facts(check_status(State.READY))) is None


@pytest.mark.parametrize(
    "state,text",
    [
        (State.CHECKING, "still running"),
        (State.NOT_TESTED, "has not run yet"),
        (State.STALE, "out of date"),
    ],
)
def test_a_check_that_is_not_ready_refuses_with_its_state(state, text):
    assert text in network_check_block(native_facts(check_status(state)))


def test_an_unsuitable_check_refuses_with_the_checks_own_reason():
    block = network_check_block(native_facts(check_status(State.UNSUITABLE, reason="cam0 drops 40% of frames")))
    assert block == "cam0 drops 40% of frames"


def test_an_unsuitable_check_without_a_reason_still_refuses():
    assert network_check_block(native_facts(check_status(State.UNSUITABLE))) == (
        "the connection is unsuitable for recording"
    )


def test_a_ready_check_measured_on_other_cameras_refuses():
    measured = [livekit_stream(1, "cam0")]
    offered = [livekit_stream(1, "cam0"), livekit_stream(2, "cam1")]
    block = network_check_block(native_facts(check_status(State.READY, streams=measured), streams=offered))
    assert "cameras changed" in block


def test_a_ready_check_with_a_changed_camera_setting_refuses():
    measured = [livekit_stream(1, fps=30.0)]
    offered = [livekit_stream(1, fps=60.0)]
    assert network_check_block(native_facts(check_status(State.READY, streams=measured), streams=offered))


def test_a_ready_check_with_an_unrelated_generation_refuses():
    assert network_check_block(native_facts(check_status(State.READY, generation="0" * 16))) is not None


def test_the_generation_ignores_stream_order():
    streams = [livekit_stream(1, "cam0"), livekit_stream(2, "cam1")]
    check = check_status(State.READY, streams=streams)
    assert network_check_block(native_facts(check, streams=list(reversed(streams)))) is None


def test_a_server_that_reports_no_check_never_refuses():
    assert network_check_block(native_facts(None)) is None


def test_no_started_session_never_refuses():
    native = NativeFacts(status=capture_status(status_snapshot(check_status(State.CHECKING))))
    assert network_check_block(native) is None


def test_a_stopped_session_never_refuses():
    stopped = SimpleNamespace(session_id="s1")
    assert network_check_block(native_facts(check_status(State.CHECKING), stopped=stopped)) is None


def test_no_status_yet_never_refuses():
    native = NativeFacts(started=SimpleNamespace(streams=(livekit_stream(),), session_id="s1"))
    assert network_check_block(native) is None


def test_a_session_with_no_livekit_streams_never_refuses():
    assert network_check_block(native_facts(check_status(State.CHECKING), streams=[])) is None


def test_non_livekit_streams_do_not_count_as_offered_cameras():
    other = SimpleNamespace(stream_id=3, name="usb")
    assert network_check_block(native_facts(check_status(State.CHECKING), streams=[other])) is None


def test_a_verdict_from_another_capture_session_refuses_even_when_ready():
    block = network_check_block(native_facts(check_status(State.READY, session_id="old"), session_id="new"))
    assert "another capture session" in block


def test_a_verdict_with_no_session_refuses():
    assert network_check_block(native_facts(check_status(State.READY, session_id=None)))


def _camera(stream_id: int, track_id: str | None) -> p.NetworkCheckCamera:
    return p.NetworkCheckCamera(
        stream_id=stream_id, name=f"cam{stream_id}", passed=True, expected_fps=30.0, track_id=track_id
    )


def test_measured_tracks_maps_each_measured_camera_to_its_track():
    cameras = (_camera(1, "t1"), _camera(2, "t2"), _camera(3, None))
    assert measured_tracks(native_facts(check_status(cameras=cameras))) == {1: "t1", 2: "t2"}


def test_measured_tracks_is_empty_when_no_check_or_no_status_is_reported():
    assert measured_tracks(native_facts(None)) == {}
    assert measured_tracks(NativeFacts()) == {}
