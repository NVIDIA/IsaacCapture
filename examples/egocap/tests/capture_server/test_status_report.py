# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

import pytest

from capture_server import protocol as p
from capture_server import status_report
from capture_server.client import CaptureClient, ServerStatusQueryError
from capture_server.network_check import NetworkCheckPolicy
from capture_server.server import ServerConfig
from tests.support.capture_server import start_capture_server, stop_capture_server

State = p.NetworkCheckState


def _camera(**overrides) -> p.NetworkCheckCamera:
    fields = dict(
        stream_id=16,
        name="Head left",
        passed=True,
        expected_fps=30.0,
        usable_frames=900,
        delivered_fps=29.9,
        missing_frames=0,
        max_arrival_gap_ms=61.0,
        age_median_ms=80.0,
        age_max_ms=120.0,
    )
    fields.update(overrides)
    return p.NetworkCheckCamera(**fields)


def _check(state=State.READY, **overrides) -> p.NetworkCheckStatus:
    fields = dict(
        state=state,
        session_id="sess-1",
        generation="g",
        endpoint="LiveKit wss://lk, room r, publisher o",
        legs_tested=("sensor capture", "decode"),
        legs_not_tested=("browser preview", "disk"),
        window_s=30.0,
        valid_for_s=120.0,
        cameras=(_camera(),),
    )
    fields.update(overrides)
    return p.NetworkCheckStatus(**fields)


def _snapshot() -> p.StatusSnapshot:
    return p.StatusSnapshot(
        updated_at_ns=1,
        providers=(
            p.ProviderStatus(
                id="prov-1",
                type=p.ProviderType.OPENXR_RUNTIME,
                plugin=None,
                name="OpenXR",
                status=p.ProviderState.FAILED,
                reason=p.StatusReason.PROVIDER_FAILED,
                error="runtime crashed",
            ),
        ),
        devices=(
            p.DeviceStatus(
                id="dev-1",
                path="/dev/x",
                type="hid",
                description="Left glove",
                provider_id="prov-1",
                status=p.DeviceState.CONNECTED,
                reason=p.StatusReason.NONE,
                error="",
            ),
        ),
        cameras=(
            p.CameraStatus(
                name="cam-a",
                stream_id=16,
                status=p.DeviceState.DISCONNECTED,
                reason=p.StatusReason.NONE,
                error="unplugged",
            ),
        ),
    )


def _reply(**overrides) -> p.ServerStatusReply:
    fields = dict(
        request_id="q-1",
        mode=p.CaptureMode.MOCK,
        session_id="sess-1",
        status=_snapshot(),
        network_check=_check(),
    )
    fields.update(overrides)
    return p.ServerStatusReply(**fields)


# -- format_report ---------------------------------------------------------------------------------------------


def test_report_names_mode_session_sources_and_check():
    text = status_report.format_report(_reply())
    lines = text.splitlines()
    assert lines[0] == "capture server: mock mode"
    assert "session: sess-1" in lines
    assert "provider OpenXR: failed (runtime crashed)" in lines
    assert "device Left glove: connected" in lines
    assert "network check: passed" in lines
    assert any(
        line.startswith(
            "  Head left: passed; delivery rate: 29.9 fps (configured: 30 fps); usable frames "
            "received: 900; missing frames: 0;"
        )
        for line in lines
    )
    assert "  measured path: sensor capture, decode" in lines
    assert "  outside this test: browser preview; disk" in lines
    assert "  media endpoint: LiveKit wss://lk, room r, publisher o" in lines


def test_report_lists_each_session_camera_once_from_the_inventory():
    inventory = p.ServerCamera(
        name="cam-a",
        transport="capture_server",
        selected=True,
        device_id="/dev/video0",
        state=p.DeviceState.DISCONNECTED,
        error="unplugged",
    )
    lines = status_report.format_report(_reply(cameras=(inventory,))).splitlines()
    camera_lines = [line for line in lines if line.startswith("camera cam-a")]
    assert len(camera_lines) == 1
    assert "in session; disconnected (unplugged)" in camera_lines[0]
    assert "provider OpenXR: failed (runtime crashed)" in lines
    assert "device Left glove: connected" in lines


def test_report_without_a_session_says_so_and_lists_no_sources():
    text = status_report.format_report(
        _reply(session_id=None, status=None, network_check=_check(State.NOT_TESTED))
    )
    assert "session: none running" in text
    assert "provider" not in text and "camera cam" not in text
    assert "network check: not tested" in text


def test_report_without_a_check_says_not_configured_and_keeps_the_rest():
    text = status_report.format_report(_reply(network_check=None))
    assert "network check: not configured" in text
    assert "provider OpenXR: failed" in text


@pytest.mark.parametrize(
    ("state", "label"),
    [
        (State.NOT_TESTED, "not tested"),
        (State.CHECKING, "checking"),
        (State.READY, "passed"),
        (State.UNSUITABLE, "failed"),
        (State.STALE, "expired; run again before recording"),
    ],
)
def test_report_labels_every_check_state(state, label):
    assert f"network check: {label}" in status_report.format_report(_reply(network_check=_check(state)))


def test_report_shows_progress_while_checking_and_the_reason_when_unsuitable():
    checking = _check(State.CHECKING, phase="measuring", progress_s=7, cameras=())
    assert "  progress: measuring, 7 of 30 seconds" in status_report.format_report(_reply(network_check=checking))
    bad = _check(
        State.UNSUITABLE,
        reason="the connection is unsuitable for recording",
        cameras=(
            _camera(
                passed=False, reasons=("a 400 ms gap", "No usable frames received in 2 of 30 one-second samples")
            ),
        ),
    )
    text = status_report.format_report(_reply(network_check=bad))
    assert "  reason: the connection is unsuitable for recording" in text
    assert "  Head left: failed;" in text
    assert "    - a 400 ms gap" in text and "    - No usable frames received in 2 of 30 one-second samples" in text


def test_report_shows_unknown_measurements_as_unknown_not_zero():
    camera = p.NetworkCheckCamera(stream_id=16, name="Head left", passed=None, expected_fps=30.0)
    text = status_report.format_report(_reply(network_check=_check(State.CHECKING, cameras=(camera,))))
    line = next(x for x in text.splitlines() if x.startswith("  Head left:"))
    assert line.startswith(
        "  Head left: not measured; delivery rate: unknown fps (configured: 30 fps); usable "
        "frames received: unknown; missing frames: unknown; longest gap without usable frames: unknown;"
    )
    assert "capture-to-receive delay: median unknown, maximum unknown" in line


# -- main against a real server --------------------------------------------------------------------------------


async def _main(*argv: str) -> int:
    return await asyncio.to_thread(status_report.main, list(argv))


async def test_main_prints_the_report_of_a_running_server_and_exits_zero(unused_tcp_port, capsys):
    config = ServerConfig(
        livekit_streams=(
            p.LiveKitCameraStream(
                stream_id=16,
                name="Head left",
                url="wss://lk",
                room="r",
                publisher="o",
                track="t",
                width=6,
                height=4,
                fps=10,
            ),
        ),
        network_check=NetworkCheckPolicy(startup_s=1.0, window_s=1.0),
    )
    running = await start_capture_server(unused_tcp_port, config=config)
    try:
        assert await _main("--url", f"ws://127.0.0.1:{unused_tcp_port}") == 0
        out = capsys.readouterr().out
        assert "capture server: mock mode" in out
        assert "session: none running" in out
        assert "network check: not tested" in out
    finally:
        await stop_capture_server(running)


async def test_main_json_prints_the_wire_reply(unused_tcp_port, capsys):
    running = await start_capture_server(unused_tcp_port)
    try:
        assert await _main("--url", f"ws://127.0.0.1:{unused_tcp_port}", "--json") == 0
        data = json.loads(capsys.readouterr().out)
        assert data["type"] == int(p.MessageType.SERVER_STATUS)
        assert (data["session_id"], data["status"], data["network_check"]) == (None, None, None)
        assert p.ServerStatusReply.from_json(data).mode is p.CaptureMode.MOCK
    finally:
        await stop_capture_server(running)


async def test_main_exits_one_with_a_message_when_the_server_does_not_answer(unused_tcp_port, capsys):
    assert await _main("--url", f"ws://127.0.0.1:{unused_tcp_port}") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("status_report: the status query failed")


# -- the client ------------------------------------------------------------------------------------------------


async def test_query_status_returns_the_reply_of_a_running_server(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        reply = await CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}").query_status()
        assert reply.mode is p.CaptureMode.MOCK
        assert (reply.session_id, reply.status, reply.network_check) == (None, None, None)
    finally:
        await stop_capture_server(running)


async def test_query_status_raises_when_nothing_listens(unused_tcp_port):
    with pytest.raises(ServerStatusQueryError, match="status query failed"):
        await CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}").query_status()
