# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LiveKit stream offers: wire protocol, YAML loader, server SessionInit, client, and the CLI flag."""

from __future__ import annotations

import asyncio
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest
import websockets

from capture_server import protocol as p
from capture_server.__main__ import main
from capture_server.client import CaptureClient, _StreamRegistry
from capture_server.livekit_streams import load_livekit_streams
from capture_server.server import ServerConfig, StationCamera
from tests.support.capture_server import poll_until, start_capture_server, stop_capture_server


def _stream_json(**overrides) -> dict:
    base = {
        "stream_id": 16,
        "kind": int(p.StreamKind.LIVEKIT_IMAGE),
        "name": "Head left",
        "url": "wss://lk.example:7881",
        "room": "sensing-preview",
        "publisher": "orin",
        "track": "sensing-left",
        "timestamp_basis": 1,
        "width": 2560,
        "height": 1984,
        "fps": 30,
    }
    return {**base, **overrides}


def _stream(**overrides) -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream.from_json(_stream_json(**overrides))


def _two_streams() -> tuple[p.LiveKitCameraStream, ...]:
    return (
        _stream(),
        _stream(stream_id=17, name="Head right", track="sensing-right", width=None, height=None, fps=None),
    )


def _begin() -> p.BeginRequest:
    return p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))


def _init(streams) -> p.SessionInit:
    return p.session_init_event(session_id="s1", mode=p.CaptureMode.MOCK, begin=_begin(), streams=list(streams))


# ---------------------------------------------------------------------------
# Protocol.
# ---------------------------------------------------------------------------


def test_livekit_camera_stream_round_trips_through_json():
    stream = _stream()
    assert p.LiveKitCameraStream.from_json(stream.to_json()) == stream
    assert stream.kind == p.StreamKind.LIVEKIT_IMAGE
    assert stream.to_json()["kind"] == 3
    assert stream.timestamp_basis == p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US
    assert stream.fps == 30.0


def test_livekit_camera_stream_unknown_dimensions_are_null():
    stream = _stream(width=None, height=None, fps=None)
    assert (stream.width, stream.height, stream.fps) == (None, None, None)
    assert stream.to_json()["fps"] is None


def test_session_init_carries_livekit_streams_in_streams_and_round_trips():
    streams = _two_streams()
    wire = _init(streams).to_json()
    assert "livekit_streams" not in wire
    assert [s["kind"] for s in wire["streams"]] == [3, 3]
    assert p.SessionInit.from_json(wire).streams == streams


def test_session_init_mixes_livekit_with_tracking_and_jpeg_streams():
    tracking = p.tracking_stream_descriptor((p.SourceType.HEAD,), 60)
    init = _init([tracking, *_two_streams()])
    assert p.SessionInit.from_json(init.to_json()) == init


def test_session_init_without_livekit_streams_is_unchanged_on_the_wire():
    wire = _init([]).to_json()
    assert set(wire) == {"type", "version", "session_id", "mode", "effective_config", "clock", "streams"}
    assert wire["streams"] == []


def test_session_init_still_rejects_unknown_keys():
    with pytest.raises(p.ProtocolError):
        p.SessionInit.from_json({**_init([]).to_json(), "surprise": 1})


@pytest.mark.parametrize(
    "mutation",
    [
        {"surprise": 1},
        {"kind": 2},
        {"stream_id": 15},
        {"stream_id": 24},
        {"stream_id": "16"},
        {"stream_id": True},
        {"name": ""},
        {"name": None},
        {"room": ""},
        {"publisher": 7},
        {"track": ""},
        {"url": "https://lk.example"},
        {"url": "lk.example:7881"},
        {"url": "x" * 600},
        {"timestamp_basis": 4},
        {"timestamp_basis": 0},
        {"timestamp_basis": "1"},
        {"timestamp_basis": True},
        {"width": 0},
        {"width": -1},
        {"width": 1.5},
        {"width": "100"},
        {"height": True},
        {"fps": 0},
        {"fps": -5},
        {"fps": "30"},
        {"fps": True},
        {"fps": float("nan")},
        {"fps": float("inf")},
        {"fps": float("-inf")},
    ],
)
def test_livekit_camera_stream_rejects_bad_fields(mutation):
    with pytest.raises(p.ProtocolError):
        p.LiveKitCameraStream.from_json(_stream_json(**mutation))


@pytest.mark.parametrize(
    "missing", ["stream_id", "kind", "name", "url", "room", "publisher", "track", "timestamp_basis"]
)
def test_livekit_camera_stream_rejects_a_missing_required_key(missing):
    data = _stream_json()
    del data[missing]
    with pytest.raises(p.ProtocolError):
        p.LiveKitCameraStream.from_json(data)


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), float("-inf")])
def test_livekit_fps_rejection_says_finite_positive(fps):
    with pytest.raises(p.ProtocolError, match="finite positive number or null"):
        p.LiveKitCameraStream.from_json(_stream_json(fps=fps))


def test_livekit_camera_stream_treats_missing_optional_dimensions_as_unknown():
    data = _stream_json()
    for key in ("width", "height", "fps"):
        del data[key]
    stream = p.LiveKitCameraStream.from_json(data)
    assert (stream.width, stream.height, stream.fps) == (None, None, None)


def test_livekit_camera_stream_accepts_ws_and_wss():
    assert _stream(url="ws://192.0.2.5:7880").url == "ws://192.0.2.5:7880"
    assert _stream(url="wss://lk").url == "wss://lk"


def _wire_with(entries: list[dict]) -> dict:
    return {**_init([]).to_json(), "streams": entries}


def test_session_init_livekit_limit_is_eight():
    def many(n):
        return [_stream_json(stream_id=16 + i, name=f"n{i}", track=f"t{i}") for i in range(n)]

    assert len(p.SessionInit.from_json(_wire_with(many(8))).streams) == 8
    with pytest.raises(p.ProtocolError):
        p.SessionInit.from_json(_wire_with(many(9)))


def test_check_livekit_streams_limit_is_eight():
    streams = [_stream(stream_id=16 + i % 8, name=f"n{i}", track=f"t{i}") for i in range(9)]
    with pytest.raises(p.ProtocolError, match="at most 8"):
        p.check_livekit_streams(streams)


@pytest.mark.parametrize(
    ("second", "match"),
    [
        ({"stream_id": 16, "name": "other", "track": "other"}, "duplicate stream_id"),
        ({"name": "Head left", "track": "other"}, "names must be unique"),
        ({"name": "other", "track": "sensing-left"}, "same publisher and track"),
    ],
)
def test_session_init_rejects_duplicate_livekit_entries(second, match):
    first = _stream_json()
    other = _stream_json(stream_id=17, name="b", track="u")
    with pytest.raises(p.ProtocolError, match=match):
        p.SessionInit.from_json(_wire_with([first, {**other, **second}]))


def test_livekit_name_may_not_collide_with_a_jpeg_camera_name():
    camera = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="Head left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    with pytest.raises(p.ProtocolError, match="names must be unique"):
        p.check_livekit_streams([camera, _stream()])


@pytest.mark.parametrize(
    "different",
    [{"publisher": "other-host"}, {"room": "other-room"}, {"url": "wss://other.example"}],
)
def test_livekit_streams_must_share_one_url_room_and_publisher(different):
    with pytest.raises(p.ProtocolError, match="same url, room and publisher"):
        p.check_livekit_streams([_stream(), _stream(stream_id=17, name="b", track="u", **different)])


def test_server_config_rejects_livekit_streams_in_several_rooms():
    other = _stream(stream_id=17, name="b", track="u", room="other-room")
    with pytest.raises(ValueError, match="same url, room and publisher"):
        ServerConfig(livekit_streams=(_stream(), other))


@pytest.mark.parametrize("field", ["cameras", "known_cameras"])
def test_server_config_rejects_a_livekit_name_equal_to_a_camera_name(field):
    value = {"Head left": StationCamera(device_id="dev")} if field == "cameras" else ("Head left",)
    with pytest.raises(ValueError, match="equal to a camera name"):
        ServerConfig(livekit_streams=(_stream(),), **{field: value})


# ---------------------------------------------------------------------------
# YAML loader.
# ---------------------------------------------------------------------------

_GOOD_YAML = """
livekit:
  url: wss://lk.example:7881
  room: sensing-preview
  publisher: orin
  streams:
    - {name: Head left, track: sensing-left, width: 2560, height: 1984, fps: 30}
    - {name: Head right, track: sensing-right}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "livekit.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loader_reads_a_valid_file(tmp_path):
    streams = load_livekit_streams(_write(tmp_path, _GOOD_YAML))
    assert [s.stream_id for s in streams] == [16, 17]
    assert [s.name for s in streams] == ["Head left", "Head right"]
    assert streams[0].url == "wss://lk.example:7881"
    assert streams[0].room == "sensing-preview"
    assert streams[0].publisher == "orin"
    assert (streams[0].width, streams[0].height, streams[0].fps) == (2560, 1984, 30.0)
    assert streams[1].width is None
    assert streams[1].timestamp_basis == p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US


def test_loader_accepts_an_empty_streams_list(tmp_path):
    text = "livekit: {url: wss://lk, room: r, publisher: o, streams: []}\n"
    assert load_livekit_streams(_write(tmp_path, text)) == ()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "- a\n- b\n",
        "other: 1\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [], extra: 1}\n",
        "livekit: {url: wss://lk, room: r, publisher: o}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: x}\n",
        "livekit: {room: r, publisher: o, streams: [{name: A, track: t}]}\n",
        "livekit: {url: wss://lk, publisher: o, streams: [{name: A, track: t}]}\n",
        "livekit: {url: wss://lk, room: r, streams: [{name: A, track: t}]}\n",
        "livekit: {url: http://lk, room: r, publisher: o, streams: [{name: A, track: t}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{name: A}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{id: a, name: A, track: t}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{name: A, track: t, bogus: 1}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [x]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{name: A, track: t, fps: .nan}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{name: A, track: t, fps: .inf}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{name: A, track: t, fps: -.inf}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{name: A, track: t, fps: 0}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{name: A, track: t}, {name: A, track: u}]}\n",
        "livekit: {url: wss://lk, room: r, publisher: o, streams: [{name: A, track: t}, {name: B, track: t}]}\n",
        "livekit: [unterminated\n",
    ],
)
def test_loader_rejects_unusable_files_with_value_error(tmp_path, text):
    with pytest.raises(ValueError):
        load_livekit_streams(_write(tmp_path, text))


def test_loader_reports_an_unreadable_file_as_value_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        load_livekit_streams(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# Server SessionInit and client.
# ---------------------------------------------------------------------------


async def _begin_and_read_init(url: str) -> p.SessionInit:
    begin = _begin()
    async with websockets.connect(url) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": int(p.MessageType.BEGIN),
                    "version": p.PROTOCOL_VERSION,
                    "session_id": "livekit-session",
                    **p.effective_config(begin).to_json(),
                }
            )
        )
        while True:
            message = p.parse_server_message(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))
            if isinstance(message, p.SessionInit):
                return message


def _livekit_in(init) -> tuple[p.LiveKitCameraStream, ...]:
    return tuple(s for s in init.streams if isinstance(s, p.LiveKitCameraStream))


@pytest.mark.asyncio
async def test_server_session_init_streams_include_the_configured_livekit_cameras(unused_tcp_port):
    streams = _two_streams()
    running = await start_capture_server(unused_tcp_port, config=ServerConfig(livekit_streams=streams))
    try:
        init = await _begin_and_read_init(f"ws://127.0.0.1:{unused_tcp_port}")
    finally:
        await stop_capture_server(running)
    assert _livekit_in(init) == streams
    assert any(isinstance(s, p.TrackingStream) for s in init.streams)


@pytest.mark.asyncio
async def test_server_without_livekit_streams_offers_none(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        init = await _begin_and_read_init(f"ws://127.0.0.1:{unused_tcp_port}")
    finally:
        await stop_capture_server(running)
    assert _livekit_in(init) == ()


@pytest.mark.asyncio
async def test_client_started_streams_include_livekit_cameras_and_capture_still_runs(unused_tcp_port):
    streams = _two_streams()
    running = await start_capture_server(unused_tcp_port, config=ServerConfig(livekit_streams=streams))
    started = []
    frames = []
    client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        await client.start(sources=(p.SourceType.HEAD,), on_started=started.append, on_capture_frame=frames.append)
        await poll_until(lambda: len(frames) >= 1)
        await client.stop()
    finally:
        await stop_capture_server(running)
    assert _livekit_in(started[0]) == streams
    assert not hasattr(started[0], "livekit_streams")


def test_client_validate_layout_ignores_livekit_entries_when_checking_requested_cameras():
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
        cameras=(p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),),
    )
    tracking = p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)
    camera = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="camera_left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    init = p.session_init_event(
        session_id="s", mode=p.CaptureMode.MOCK, begin=begin, streams=[tracking, camera, *_two_streams()]
    )
    CaptureClient("ws://127.0.0.1:1")._validate_layout(begin, init)
    registry = _StreamRegistry.from_session_init(init)
    assert set(registry.cameras) == {1}


# ---------------------------------------------------------------------------
# CLI flag.
# ---------------------------------------------------------------------------


def test_cli_rejects_a_bad_streams_file_with_exit_code_2(tmp_path, capsys):
    bad = _write(tmp_path, "livekit: {url: http://lk, room: r, publisher: o, streams: [{name: A, track: t}]}\n")
    assert main(["--mock", "--livekit-streams", str(bad)]) == 2
    assert "ws:// or wss://" in capsys.readouterr().err


def test_cli_rejects_a_missing_streams_file_with_exit_code_2(tmp_path, capsys):
    assert main(["--mock", "--livekit-streams", str(tmp_path / "nope.yaml")]) == 2
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_cli_serves_the_streams_from_the_file(tmp_path, unused_tcp_port):
    path = _write(tmp_path, _GOOD_YAML)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "capture_server",
            "--mock",
            "--host",
            "127.0.0.1",
            "--port",
            str(unused_tcp_port),
            "--livekit-streams",
            str(path),
        ],  # fmt: skip
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for the startup banner so the connection below cannot race the listener.
        deadline = time.monotonic() + 10
        seen = ""
        while "listening on" not in seen and time.monotonic() < deadline:
            ready, _, _ = select.select([proc.stderr], [], [], 0.5)
            if ready:
                chunk = os.read(proc.stderr.fileno(), 4096).decode(errors="replace")
                if not chunk:
                    break
                seen += chunk
        assert "listening on" in seen
        init = await _begin_and_read_init(f"ws://127.0.0.1:{unused_tcp_port}")
        assert [s.stream_id for s in _livekit_in(init)] == [16, 17]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Timestamp basis.
# ---------------------------------------------------------------------------


def test_timestamp_basis_values_are_wire_stable():
    assert int(p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US) == 1
    assert int(p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US) == 2
    assert int(p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US) == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2, p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US),
        (3, p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US),
    ],
)
def test_unix_bases_round_trip_through_json(value, expected):
    stream = _stream(timestamp_basis=value)
    assert stream.timestamp_basis == expected
    assert p.LiveKitCameraStream.from_json(stream.to_json()) == stream


@pytest.mark.parametrize(
    ("basis_line", "expected"),
    [
        ("", p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US),
        ("timestamp_basis: monotonic_us,", p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US),
        ("timestamp_basis: processing_unix_us,", p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US),
        ("timestamp_basis: capture_unix_us,", p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US),
    ],
)
def test_loader_puts_the_room_basis_in_every_stream(tmp_path, basis_line, expected):
    text = (
        f"livekit: {{url: wss://lk, room: r, publisher: o, {basis_line} "
        "streams: [{name: A, track: t}, {name: B, track: u}]}\n"
    )
    streams = load_livekit_streams(_write(tmp_path, text))
    assert [s.timestamp_basis for s in streams] == [expected, expected]


@pytest.mark.parametrize(
    "value",
    [
        "wall_clock",
        "unix_us",
        "UNIX_US",
        "CAPTURE_UNIX_US",
        "2",
        "''",
        "null",
        "[capture_unix_us]",
        "{a: 1}",
    ],
)
def test_loader_rejects_an_unknown_basis(tmp_path, value):
    text = f"livekit: {{url: wss://lk, room: r, publisher: o, timestamp_basis: {value}, streams: []}}\n"
    with pytest.raises(ValueError, match="timestamp_basis"):
        load_livekit_streams(_write(tmp_path, text))
