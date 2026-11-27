# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from capture_server import protocol as p
from capture_server import server as server_module
from capture_server import status_report
from capture_server.client import CaptureClient, ServerStatusQueryError
from capture_server.network_check import NetworkCheckPolicy, NetworkCheckService
from capture_server.server import ServerConfig, StationCamera
from tests.support.capture_server import start_capture_server, stop_capture_server

HEAD = p.LiveKitCameraStream(
    stream_id=16,
    name="head",
    url="wss://lk",
    room="room-a",
    publisher="orin",
    track="head-track",
    width=1280,
    height=720,
    fps=30.0,
)
POLICY = NetworkCheckPolicy(startup_s=1.0, window_s=1.0, valid_for_s=100.0)
State = p.NetworkCheckState


class _Factory:
    """Builds the real check service on a receiver factory that records every use."""

    def __init__(self) -> None:
        self.receivers_built = 0
        self.services: list[NetworkCheckService] = []

    def _receiver(self, *_args, **_kwargs):
        self.receivers_built += 1
        raise AssertionError("a status query must not build a LiveKit receiver")

    def __call__(self, streams, policy, make_token, **_kwargs):
        service = NetworkCheckService(streams, policy, lambda room: "token", receiver_factory=self._receiver)
        self.services.append(service)
        return service


@pytest.fixture
def factory(monkeypatch) -> _Factory:
    built = _Factory()
    monkeypatch.setattr(server_module, "NetworkCheckService", built)
    return built


def _config(**overrides) -> ServerConfig:
    fields = dict(
        cameras={"wrist": StationCamera(device_id="MX1", width=640, height=480, fps=15.0)},
        known_cameras=("spare",),
        livekit_streams=(HEAD,),
    )
    fields.update(overrides)
    return ServerConfig(**fields)


def _mock_config() -> ServerConfig:
    return ServerConfig(known_cameras=("wrist", "spare"), livekit_streams=(HEAD,))


def _begin(session_id: str = "s1", *cameras: str) -> dict:
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
        cameras=tuple(p.CameraRequest(name=n, type=p.CameraType.OAK_D_COLOR) for n in cameras),
    )
    return {
        "type": int(p.MessageType.BEGIN),
        "version": p.PROTOCOL_VERSION,
        "session_id": session_id,
        **p.effective_config(begin).to_json(),
    }


async def _query(port: int) -> p.ServerStatusReply:
    async with websockets.connect(f"ws://127.0.0.1:{port}/query") as ws:
        await ws.send(json.dumps(p.QueryStatusRequest(request_id="q-1").to_json()))
        return p.ServerStatusReply.from_json(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))


async def _recv_until(ws, kind, timeout: float = 5.0):
    async def read():
        while True:
            raw = await ws.recv()
            if isinstance(raw, str):
                event = p.parse_server_message(json.loads(raw))
                if isinstance(event, kind):
                    return event

    return await asyncio.wait_for(read(), timeout)


def _by_name(reply: p.ServerStatusReply) -> dict[str, p.ServerCamera]:
    return {c.name: c for c in reply.cameras}


def _session_status(*cameras: p.CameraStatus) -> p.StatusSnapshot:
    return p.StatusSnapshot(updated_at_ns=1, cameras=cameras)


def _observed(name: str, state: p.DeviceState, error: str = "", stream_id: int = 3) -> p.CameraStatus:
    return p.CameraStatus(name=name, stream_id=stream_id, status=state, reason=p.StatusReason(0), error=error)


# -- inventory ----------------------------------------------------------------------------------------------


async def test_an_idle_server_lists_every_configured_camera_without_a_session(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        reply = await _query(unused_tcp_port)
        assert (reply.session_id, reply.status) == (None, None)
        cameras = _by_name(reply)
        assert set(cameras) == {"wrist", "spare", "head"}
        assert len(reply.cameras) == 3
        wrist = cameras["wrist"]
        assert (wrist.transport, wrist.device_id, wrist.selected, wrist.state) == (
            "capture_server",
            "MX1",
            False,
            None,
        )
        assert (wrist.width, wrist.height, wrist.fps) == (640, 480, 15.0)
        spare = cameras["spare"]
        assert (spare.transport, spare.device_id, spare.width, spare.selected) == (
            "capture_server",
            None,
            None,
            False,
        )
        head = cameras["head"]
        assert (head.transport, head.room, head.track, head.publisher, head.stream_id) == (
            "livekit",
            "room-a",
            "head-track",
            "orin",
            16,
        )
        assert (head.width, head.height, head.fps, head.selected, head.state) == (1280, 720, 30.0, False, None)
    finally:
        await stop_capture_server(running)


async def test_an_active_session_enriches_by_name_without_duplicates(unused_tcp_port, monkeypatch):
    status = _session_status(_observed("wrist", p.DeviceState.FAILED, error="usb reset", stream_id=2))
    monkeypatch.setattr(server_module._Session, "last_status", property(lambda self: status))
    running = await start_capture_server(unused_tcp_port, config=_mock_config())
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(json.dumps(_begin("s1", "wrist")))
            await _recv_until(ws, p.SessionBegun)
            reply = await _query(unused_tcp_port)
        assert reply.session_id == "s1"
        assert len(reply.cameras) == len({c.name for c in reply.cameras}) == 3
        cameras = _by_name(reply)
        wrist = cameras["wrist"]
        assert (wrist.selected, wrist.state, wrist.error, wrist.stream_id) == (
            True,
            p.DeviceState.FAILED,
            "usb reset",
            2,
        )
        assert cameras["head"].selected is True and cameras["head"].state is None
        assert cameras["head"].stream_id == 16
        assert cameras["spare"].selected is False and cameras["spare"].state is None
    finally:
        await stop_capture_server(running)


@pytest.mark.parametrize(
    "state, expected",
    [(p.DeviceState.UNKNOWN, p.DeviceState.UNKNOWN), (p.DeviceState.CONNECTED, p.DeviceState.CONNECTED)],
)
async def test_unknown_and_connected_are_distinct_observed_states(unused_tcp_port, monkeypatch, state, expected):
    status = _session_status(_observed("wrist", state))
    monkeypatch.setattr(server_module._Session, "last_status", property(lambda self: status))
    running = await start_capture_server(unused_tcp_port, config=_mock_config())
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(json.dumps(_begin("s1", "wrist")))
            await _recv_until(ws, p.SessionBegun)
            wrist = _by_name(await _query(unused_tcp_port))["wrist"]
        assert wrist.state is expected and wrist.error is None
    finally:
        await stop_capture_server(running)


async def test_a_session_camera_not_yet_observed_has_no_state(unused_tcp_port, monkeypatch):
    monkeypatch.setattr(server_module._Session, "last_status", property(lambda self: _session_status()))
    running = await start_capture_server(unused_tcp_port, config=_mock_config())
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(json.dumps(_begin("s1", "wrist")))
            await _recv_until(ws, p.SessionBegun)
            wrist = _by_name(await _query(unused_tcp_port))["wrist"]
        assert (wrist.selected, wrist.state) == (True, None)
    finally:
        await stop_capture_server(running)


@pytest.mark.parametrize("check_state", [State.NOT_TESTED, State.STALE, State.UNSUITABLE])
def test_the_network_check_state_never_erases_the_inventory(factory, monkeypatch, check_state):
    server = server_module.CaptureServer(_config(network_check=POLICY))
    service = factory.services[0]
    check = p.NetworkCheckStatus(
        state=check_state,
        session_id=None,
        generation="g",
        endpoint="e",
        legs_tested=("a",),
        legs_not_tested=("b",),
        window_s=1.0,
        valid_for_s=100.0,
    )
    monkeypatch.setattr(service, "snapshot", lambda: check)
    reply = server._server_status("q-1")
    assert reply.network_check.state is check_state
    assert {c.name for c in reply.cameras} == {"wrist", "spare", "head"}


def test_a_server_with_the_check_disabled_still_lists_its_cameras():
    reply = server_module.CaptureServer(_config())._server_status("q-1")
    assert reply.network_check is None
    assert {c.name for c in reply.cameras} == {"wrist", "spare", "head"}


def test_a_server_with_no_cameras_lists_none():
    reply = server_module.CaptureServer(ServerConfig())._server_status("q-1")
    assert reply.cameras == ()


def test_the_query_has_no_side_effects(factory):
    server = server_module.CaptureServer(_config(network_check=POLICY))
    for _ in range(3):
        server._server_status("q-1")
    assert server._active_session is None
    assert factory.receivers_built == 0
    service = factory.services[0]
    assert service.snapshot().state is State.NOT_TESTED
    assert service.snapshot().session_id is None


async def test_a_status_socket_query_starts_neither_a_session_nor_a_check(unused_tcp_port, factory):
    running = await start_capture_server(unused_tcp_port, config=_config(network_check=POLICY))
    try:
        reply = await _query(unused_tcp_port)
        assert reply.network_check.state is State.NOT_TESTED
        assert running.server._active_session is None
        assert factory.receivers_built == 0
    finally:
        await stop_capture_server(running)


# -- status_report ------------------------------------------------------------------------------------------


def _reply(*, session: bool, check: p.NetworkCheckStatus | None = None, cameras=None) -> p.ServerStatusReply:
    wrist = p.ServerCamera(
        name="wrist",
        transport="capture_server",
        selected=session,
        device_id="MX1",
        width=640,
        height=480,
        fps=15.0,
        state=p.DeviceState.CONNECTED if session else None,
    )
    head = p.ServerCamera(
        name="head",
        transport="livekit",
        selected=session,
        room="room-a",
        track="head-track",
        width=1280,
        height=720,
        fps=30.0,
    )
    return p.ServerStatusReply(
        request_id="r-1",
        mode=p.CaptureMode.MOCK,
        session_id="s1" if session else None,
        status=_session_status(_observed("wrist", p.DeviceState.CONNECTED)) if session else None,
        network_check=check,
        cameras=(wrist, head) if cameras is None else cameras,
    )


def test_format_report_with_no_session_and_check_off():
    text = status_report.format_report(_reply(session=False))
    assert "session: none running" in text
    assert "camera wrist: CaptureServer camera MX1, 640x480 @ 15 fps; configured; not observed" in text
    assert (
        "camera head: LiveKit room room-a, track head-track, 1280x720 @ 30 fps; configured; not observed" in text
    )
    assert "network check: not configured" in text


def test_format_report_with_an_active_session():
    text = status_report.format_report(_reply(session=True))
    assert "session: s1" in text
    assert "camera wrist: CaptureServer camera MX1, 640x480 @ 15 fps; in session; connected" in text
    assert "camera head: LiveKit room room-a" in text and "; in session; not observed" in text


def test_format_report_includes_the_check_when_present():
    check = p.NetworkCheckStatus(
        state=State.READY,
        session_id="s1",
        generation="g",
        endpoint="wss://lk",
        legs_tested=("a",),
        legs_not_tested=("b",),
        window_s=10.0,
        valid_for_s=60.0,
    )
    text = status_report.format_report(_reply(session=True, check=check))
    assert "network check: passed" in text and "network check: not configured" not in text


def test_format_report_for_a_camera_error():
    camera = p.ServerCamera(
        name="wrist", transport="capture_server", selected=True, state=p.DeviceState.FAILED, error="usb reset"
    )
    assert "camera wrist: CaptureServer camera; in session; failed (usb reset)" in status_report.format_report(
        _reply(session=True, cameras=(camera,))
    )


def _patch_query(monkeypatch, outcome) -> None:
    async def query_status(self):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(CaptureClient, "query_status", query_status)


@pytest.mark.parametrize("session", [False, True])
def test_main_prints_text_and_exits_zero(monkeypatch, capsys, session):
    _patch_query(monkeypatch, _reply(session=session))
    assert status_report.main(["--url", "ws://x:1"]) == 0
    out = capsys.readouterr().out
    assert "capture server: mock mode" in out and "camera wrist" in out


@pytest.mark.parametrize("session", [False, True])
def test_main_json_is_the_reply_wire_form(monkeypatch, capsys, session):
    reply = _reply(session=session)
    _patch_query(monkeypatch, reply)
    assert status_report.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == reply.to_json()


def test_main_reports_the_check_being_off(monkeypatch, capsys):
    _patch_query(monkeypatch, _reply(session=False, check=None))
    assert status_report.main([]) == 0
    assert "network check: not configured" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [[], ["--json"]])
def test_main_exits_one_when_the_server_is_unreachable(monkeypatch, capsys, extra):
    _patch_query(monkeypatch, ServerStatusQueryError("the status query failed: refused"))
    assert status_report.main(extra) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "the status query failed: refused" in captured.err
