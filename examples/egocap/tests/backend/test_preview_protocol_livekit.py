# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The snapshot's capture.started member lists offered LiveKit cameras inside ``streams`` (kind 3)."""

from __future__ import annotations

from capture_server import protocol as p
from capture_server.client import CaptureStarted, _to_client_started
from egocap_web.preview_protocol import _capture_started_to_json

_STREAM = p.LiveKitCameraStream(
    stream_id=16,
    name="Head left",
    url="wss://lk.example:7881",
    room="sensing-preview",
    publisher="orin",
    track="sensing-left",
    width=2560,
    height=1984,
    fps=30.0,
)


def _started(streams: list[p.LiveKitCameraStream]) -> CaptureStarted:
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    init = p.session_init_event(session_id="s1", mode=p.CaptureMode.MOCK, begin=begin, streams=streams)
    return _to_client_started(init)


def test_capture_started_json_lists_livekit_cameras_in_streams():
    wire = _capture_started_to_json(_started([_STREAM]))
    assert wire["streams"] == [_STREAM.to_json()]
    assert wire["streams"][0]["kind"] == 3


def test_capture_started_json_has_no_separate_livekit_list():
    assert "livekit_streams" not in _capture_started_to_json(_started([_STREAM]))
    assert "livekit_streams" not in _capture_started_to_json(_started([]))
