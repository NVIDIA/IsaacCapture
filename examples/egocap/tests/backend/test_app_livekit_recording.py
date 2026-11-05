# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The app's wiring of recordable LiveKit streams: the recorder's token and the recording config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capture_server import protocol as p
from egocap_web import app as app_module


def _stream(stream_id: int, basis, width=6) -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream(
        stream_id=stream_id,
        name=f"cam{stream_id}",
        url="wss://lk",
        room="r",
        publisher="o",
        track=f"t{stream_id}",
        timestamp_basis=basis,
        width=width,
        height=4,
        fps=30.0,
    )


def _monitor(streams):
    return SimpleNamespace(camera_descriptors=(), native=SimpleNamespace(started=SimpleNamespace(streams=streams)))


def test_config_factory_declares_and_requires_each_recordable_livekit_stream(tmp_path):
    capture = p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
    monitor = _monitor(
        [
            _stream(16, capture),
            _stream(17, p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US),
            _stream(18, capture, width=None),
        ]
    )
    config = app_module._recording_config_factory(monitor, None, tmp_path)()
    assert [(v.name, v.width, v.height, v.fps, v.intrinsics) for v in config.video_sources] == [
        ("cam16", 6, 4, 30.0, False)
    ]
    assert "cam16" in config.qc.required_sources


def test_config_factory_without_a_started_session_has_no_livekit_sources(tmp_path):
    monitor = SimpleNamespace(camera_descriptors=(), native=SimpleNamespace(started=None))
    assert app_module._recording_config_factory(monitor, None, tmp_path)().video_sources == ()


def test_recorder_token_is_signed_when_the_key_pair_is_set(monkeypatch):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    assert app_module._recorder_livekit_token("room").count(".") == 2


@pytest.mark.parametrize("missing", ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"])
def test_recorder_token_names_the_missing_variables(monkeypatch, missing):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match="LIVEKIT_API_KEY and LIVEKIT_API_SECRET"):
        app_module._recorder_livekit_token("room")
