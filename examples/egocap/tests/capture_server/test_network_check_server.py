# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The per-session network check over real sockets; the LiveKit receiver is a fake, so nothing joins a room."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict

import numpy as np
import pytest
import websockets
from websockets.exceptions import ConnectionClosedError

from capture_server import protocol as p
from capture_server import server as server_module
from capture_server.clock_exchange import ClockMapping
from capture_server.frame_timing import BUDGET_ENV
from capture_server.livekit_receiver import ReceivedFrame, StreamCounters
from capture_server.network_check import NetworkCheckPolicy, NetworkCheckService
from capture_server.server import ServerConfig
from tests.support.capture_server import poll_until, start_capture_server, stop_capture_server

STREAMS = (
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
)
POLICY = NetworkCheckPolicy(startup_s=1.0, window_s=1.0, valid_for_s=100.0)
State = p.NetworkCheckState


class FakeService:
    """Records the lifecycle calls the server makes and reports the status a real service would."""

    instances: list[FakeService] = []

    def __init__(self, streams, policy, make_token, **_kwargs) -> None:
        self.streams, self.policy, self.make_token = streams, policy, make_token
        self.begun: list[str] = []
        self.ended: list[str] = []
        self.starts = 0
        self.stops = 0
        self.accept_start = True
        self.session_id: str | None = None
        FakeService.instances.append(self)

    def snapshot(self) -> p.NetworkCheckStatus:
        state = State.NOT_TESTED if self.session_id is None else State.CHECKING
        return p.NetworkCheckStatus(
            state=state,
            session_id=self.session_id,
            generation="gen-1",
            endpoint="LiveKit wss://lk, room r, publisher o",
            legs_tested=("a",),
            legs_not_tested=("b",),
            window_s=1.0,
            valid_for_s=100.0,
        )

    def begin_session(self, session_id: str) -> None:
        self.begun.append(session_id)
        self.session_id = session_id

    async def end_session(self, session_id: str) -> None:
        self.ended.append(session_id)
        self.session_id = None

    def start(self) -> bool:
        if self.session_id is None or not self.accept_start:
            return False
        self.starts += 1
        return True

    async def stop(self) -> None:
        self.stops += 1


@pytest.fixture
def fake_service(monkeypatch):
    FakeService.instances = []
    monkeypatch.setattr(server_module, "NetworkCheckService", FakeService)
    return FakeService


class _ClockSync:
    discontinuity = False

    def mapping(self):
        return ClockMapping(epoch=1, offset_ns=0, rtt_ns=2_000, uncertainty_ns=1_000, accepted_at_ns=0)


class FakeReceiver:
    """A receiver whose camera delivers real-time frames, or is lost, or whose join waits on a gate."""

    def __init__(self, streams, token, clock_sync=False, *, mode="healthy", gate=None):
        self.streams, self.token, self.mode, self.gate = streams, token, mode, gate
        self.clock_sync = _ClockSync() if clock_sync else None
        self.counters = defaultdict(StreamCounters)
        self.starts = self.stops = 0
        self._next_id = 0

    async def start(self):
        self.starts += 1
        if self.gate is not None:
            await self.gate.wait()

    async def stop(self):
        self.stops += 1

    def clear_queued(self):
        pass

    def pending(self, stream_id):
        return 0

    def track_ids(self):
        return {s.stream_id: "TR_16" for s in self.streams}

    async def next_frame_or_lost(self, stream_id, timeout=None):
        if self.mode == "lost":
            return None
        await asyncio.sleep(0.01)
        now = time.monotonic_ns()
        self._next_id += 1
        return ReceivedFrame(
            stream_id=stream_id,
            image=np.zeros((4, 6, 3), dtype=np.uint8),
            receive_time_ns=now,
            transport_timestamp_us=1,
            user_timestamp_us=(now - 50_000_000) // 1000,
            frame_id=self._next_id,
            user_data=None,
        )


class RealServiceFactory:
    """Stands in for ``NetworkCheckService``: the real service, built on a per-check scripted fake receiver."""

    def __init__(self, modes: list[str], gates: dict[int, asyncio.Event] | None = None) -> None:
        self.modes = modes
        self.gates = gates or {}
        self.receivers: list[FakeReceiver] = []
        self.service: NetworkCheckService | None = None

    def _receiver(self, streams, token, clock_sync=False):
        index = len(self.receivers)
        mode = self.modes[min(index, len(self.modes) - 1)]
        receiver = FakeReceiver(streams, token, clock_sync, mode=mode, gate=self.gates.get(index))
        self.receivers.append(receiver)
        return receiver

    def __call__(self, streams, policy, make_token, **_kwargs):
        self.service = NetworkCheckService(streams, policy, lambda room: "token", receiver_factory=self._receiver)
        return self.service


@pytest.fixture(autouse=True)
def _default_budget(monkeypatch):
    monkeypatch.delenv(BUDGET_ENV, raising=False)


def _config(**overrides) -> ServerConfig:
    fields = dict(livekit_streams=STREAMS, network_check=POLICY)
    fields.update(overrides)
    return ServerConfig(**fields)


def _begin(session_id: str = "nc-session") -> dict:
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    return {
        "type": int(p.MessageType.BEGIN),
        "version": p.PROTOCOL_VERSION,
        "session_id": session_id,
        **p.effective_config(begin).to_json(),
    }


def _stop(session_id: str) -> str:
    return json.dumps({"type": int(p.MessageType.STOP), "version": p.PROTOCOL_VERSION, "session_id": session_id})


def _run_request(session_id: str = "nc-1") -> str:
    return json.dumps(p.RunNetworkCheckRequest(session_id=session_id).to_json())


def _connect(port: int, path: str = ""):
    # A short close timeout: the close handshake of a client that leaves a running session otherwise waits 10 s.
    return websockets.connect(f"ws://127.0.0.1:{port}{path}", close_timeout=0.5)


async def _recv(ws, timeout: float = 5.0):
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout)
        if isinstance(raw, str):
            return p.parse_server_message(json.loads(raw))


async def _recv_until(ws, kind, timeout: float = 5.0):
    async def read():
        while True:
            event = await _recv(ws, timeout)
            if isinstance(event, kind):
                return event

    return await asyncio.wait_for(read(), timeout)


async def _query_status(port: int) -> p.ServerStatusReply:
    async with _connect(port, "/query") as ws:
        await ws.send(json.dumps(p.QueryStatusRequest(request_id="q-1").to_json()))
        return p.ServerStatusReply.from_json(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))


async def _stop_session(ws, session_id: str) -> None:
    await ws.send(_stop(session_id))
    await _recv_until(ws, p.SessionStopped)


async def _check_state(port: int, state: State, session_id: str | None) -> None:
    for _ in range(300):
        check = (await _query_status(port)).network_check
        if check.state is state and check.session_id == session_id:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"network check never reached {state.name} for {session_id}: {check}")


# -- lifecycle calls ------------------------------------------------------------------------------------------


async def test_nothing_starts_at_server_start(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        service = fake_service.instances[0]
        assert service.streams == STREAMS
        assert service.policy == POLICY
        assert service.make_token is server_module._viewer_token
        await asyncio.sleep(0.1)
        assert (service.begun, service.starts) == ([], 0)
    finally:
        await stop_capture_server(running)


async def test_a_new_session_starts_the_check_once_and_its_begun_status_is_already_checking(
    unused_tcp_port, fake_service
):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            begun = await _recv_until(ws, p.SessionBegun)
            check = begun.status.network_check
            assert (check.state, check.session_id) == (State.CHECKING, "s1")
            assert fake_service.instances[0].begun == ["s1"]
    finally:
        await stop_capture_server(running)


async def test_an_identical_begin_replay_does_not_rerun_the_check(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            assert fake_service.instances[0].begun == ["s1"]
    finally:
        await stop_capture_server(running)


async def test_a_second_owner_connection_does_not_rerun_the_check(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        service = fake_service.instances[0]
        async with _connect(unused_tcp_port) as first:
            await first.send(json.dumps(_begin("s1")))
            await _recv_until(first, p.SessionBegun)
            async with _connect(unused_tcp_port) as second:
                with pytest.raises(ConnectionClosedError) as excinfo:
                    await second.send(json.dumps(_begin("s1")))
                    await asyncio.wait_for(second.recv(), timeout=2)
                assert excinfo.value.rcvd.code == 1013
            assert service.begun == ["s1"] and service.ended == []
    finally:
        await stop_capture_server(running)


async def test_a_session_ending_ends_its_check_and_a_second_session_begins_afresh(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            await _stop_session(ws, "s1")
            service = fake_service.instances[0]
            await poll_until(lambda: service.ended == ["s1"])
            await ws.send(json.dumps(_begin("s2")))
            begun = await _recv_until(ws, p.SessionBegun)
            assert begun.status.network_check.session_id == "s2"
            assert service.begun == ["s1", "s2"]
    finally:
        await stop_capture_server(running)


async def test_a_rejected_begin_starts_no_check(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            await ws.send(json.dumps(_begin("s2")))
            rejected = await _recv_until(ws, p.RequestRejected)
            assert rejected.reason is p.RejectionReason.BUSY
            assert fake_service.instances[0].begun == ["s1"]
    finally:
        await stop_capture_server(running)


async def test_a_state_change_produces_a_new_session_status(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin()))
            await _recv_until(ws, p.SessionBegun)
            fake_service.instances[0].session_id = None
            update = await _recv_until(ws, p.SessionStatus)
            assert update.status.network_check.state is State.NOT_TESTED
    finally:
        await stop_capture_server(running)


async def test_a_session_without_a_policy_carries_no_network_check(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin()))
            begun = await _recv_until(ws, p.SessionBegun)
            assert begun.status.network_check is None
    finally:
        await stop_capture_server(running)


# -- RUN_NETWORK_CHECK ----------------------------------------------------------------------------------------


async def test_the_owner_can_retry_the_check_in_its_session(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            service = fake_service.instances[0]
            await ws.send(_run_request())
            await poll_until(lambda: service.starts == 1)
    finally:
        await stop_capture_server(running)


async def test_retry_without_an_active_session_is_rejected_unknown_session(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(_run_request("nc-none"))
            rejected = await _recv_until(ws, p.RequestRejected)
            assert rejected.session_id == "nc-none"
            assert rejected.request_type is p.MessageType.RUN_NETWORK_CHECK
            assert rejected.reason is p.RejectionReason.UNKNOWN_SESSION
        assert fake_service.instances[0].starts == 0
    finally:
        await stop_capture_server(running)


async def test_retry_after_the_session_ended_is_rejected_unknown_session(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            await _stop_session(ws, "s1")
            await poll_until(lambda: fake_service.instances[0].ended == ["s1"])
            await ws.send(_run_request("nc-late"))
            rejected = await _recv_until(ws, p.RequestRejected)
            assert rejected.reason is p.RejectionReason.UNKNOWN_SESSION
    finally:
        await stop_capture_server(running)


async def test_retry_while_a_check_runs_is_rejected_busy(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            fake_service.instances[0].accept_start = False
            await ws.send(_run_request("nc-busy"))
            rejected = await _recv_until(ws, p.RequestRejected)
            assert rejected.session_id == "nc-busy"
            assert rejected.reason is p.RejectionReason.BUSY
    finally:
        await stop_capture_server(running)


async def test_a_server_without_a_policy_rejects_the_request_invalid_config(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(_run_request("nc-none"))
            rejected = await _recv_until(ws, p.RequestRejected)
            assert rejected.request_type is p.MessageType.RUN_NETWORK_CHECK
            assert rejected.reason is p.RejectionReason.INVALID_CONFIG
            assert "no network check" in rejected.error_message
    finally:
        await stop_capture_server(running)


@pytest.mark.parametrize(
    "message",
    [
        {"type": int(p.MessageType.RUN_NETWORK_CHECK), "version": p.PROTOCOL_VERSION},
        {"type": int(p.MessageType.RUN_NETWORK_CHECK), "version": p.PROTOCOL_VERSION, "session_id": ""},
        {
            "type": int(p.MessageType.RUN_NETWORK_CHECK),
            "version": p.PROTOCOL_VERSION,
            "session_id": "x",
            "extra": 1,
        },
        {"type": int(p.MessageType.RUN_NETWORK_CHECK), "version": 999, "session_id": "x"},
    ],
    ids=["no-session-id", "empty-session-id", "extra-key", "bad-version"],
)
async def test_a_malformed_envelope_closes_the_connection_with_1002(unused_tcp_port, fake_service, message):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(message))
            with pytest.raises(ConnectionClosedError) as excinfo:
                await asyncio.wait_for(ws.recv(), timeout=2)
            assert excinfo.value.rcvd.code == 1002
        assert fake_service.instances[0].starts == 0
    finally:
        await stop_capture_server(running)


async def test_a_query_client_cannot_start_a_check_but_the_owner_can(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        service = fake_service.instances[0]
        async with _connect(unused_tcp_port) as owner:
            await owner.send(json.dumps(_begin("s1")))
            await _recv_until(owner, p.SessionBegun)
            async with _connect(unused_tcp_port, "/query") as query:
                await query.send(_run_request("nc-query"))
                with pytest.raises(ConnectionClosedError) as excinfo:
                    await asyncio.wait_for(query.recv(), timeout=2)
                assert excinfo.value.rcvd.code == 1002
            assert service.starts == 0
            await owner.send(_run_request("nc-owner"))
            await poll_until(lambda: service.starts == 1)
    finally:
        await stop_capture_server(running)


# -- configuration and shutdown -------------------------------------------------------------------------------


def test_a_policy_without_livekit_streams_is_a_configuration_error():
    with pytest.raises(ValueError, match="needs livekit_streams"):
        ServerConfig(network_check=NetworkCheckPolicy())


def test_a_policy_with_livekit_streams_is_accepted():
    assert ServerConfig(livekit_streams=STREAMS, network_check=NetworkCheckPolicy()).network_check is not None


async def test_close_stops_the_check_service(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    service = fake_service.instances[0]
    assert service.stops == 0
    await stop_capture_server(running)
    assert service.stops == 1


async def test_a_server_without_a_policy_builds_no_service(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port)
    try:
        assert fake_service.instances == []
    finally:
        await stop_capture_server(running)


def test_viewer_token_requires_the_livekit_credentials(monkeypatch):
    from capture_server.livekit_token import API_KEY_ENV, API_SECRET_ENV

    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_SECRET_ENV, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        server_module._viewer_token("room")
    monkeypatch.setenv(API_KEY_ENV, "key")
    with pytest.raises(RuntimeError, match=API_SECRET_ENV):
        server_module._viewer_token("room")


def test_viewer_token_is_a_signed_token_when_credentials_are_set(monkeypatch):
    from capture_server.livekit_token import API_KEY_ENV, API_SECRET_ENV

    monkeypatch.setenv(API_KEY_ENV, "key")
    monkeypatch.setenv(API_SECRET_ENV, "s" * 40)
    assert server_module._viewer_token("room").count(".") == 2


# -- /query status --------------------------------------------------------------------------------------------


async def test_a_status_query_needs_no_session_and_reports_not_tested(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        reply = await _query_status(unused_tcp_port)
        assert reply.request_id == "q-1"
        assert (reply.session_id, reply.status) == (None, None)
        assert (reply.network_check.state, reply.network_check.session_id) == (State.NOT_TESTED, None)
        assert fake_service.instances[0].begun == []
    finally:
        await stop_capture_server(running)


async def test_a_status_query_without_a_policy_still_answers_with_no_check(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        reply = await _query_status(unused_tcp_port)
        assert reply.network_check is None
        assert (reply.session_id, reply.status) == (None, None)
        assert reply.mode is p.CaptureMode.MOCK
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            running_reply = await _query_status(unused_tcp_port)
            assert running_reply.session_id == "s1"
            assert running_reply.status is not None and running_reply.network_check is None
    finally:
        await stop_capture_server(running)


async def test_a_status_query_during_a_session_reports_the_session_and_its_check(unused_tcp_port, fake_service):
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("s1")))
            await _recv_until(ws, p.SessionBegun)
            reply = await _query_status(unused_tcp_port)
            assert reply.session_id == "s1" and reply.status is not None
            assert (reply.network_check.state, reply.network_check.session_id) == (State.CHECKING, "s1")
            # The query never disturbs the owner's session.
            assert not running.server._active_session.done.is_set()
    finally:
        await stop_capture_server(running)


async def test_an_unknown_query_type_closes_with_1002_but_query_devices_still_works(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        async with _connect(unused_tcp_port, "/query") as ws:
            await ws.send(json.dumps({"type": int(p.MessageType.STOP), "version": p.PROTOCOL_VERSION}))
            with pytest.raises(ConnectionClosedError) as excinfo:
                await asyncio.wait_for(ws.recv(), timeout=2)
            assert excinfo.value.rcvd.code == 1002
        async with _connect(unused_tcp_port, "/query") as ws:
            await ws.send(json.dumps(p.QueryDevicesRequest(request_id="d-1").to_json()))
            reply = p.DevicesReply.from_json(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))
            assert reply.request_id == "d-1"
    finally:
        await stop_capture_server(running)


async def test_a_malformed_status_query_closes_with_1002(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        bad = {"type": int(p.MessageType.QUERY_STATUS), "version": p.PROTOCOL_VERSION, "request_id": ""}
        async with _connect(unused_tcp_port, "/query") as ws:
            await ws.send(json.dumps(bad))
            with pytest.raises(ConnectionClosedError) as excinfo:
                await asyncio.wait_for(ws.recv(), timeout=2)
            assert excinfo.value.rcvd.code == 1002
    finally:
        await stop_capture_server(running)


# -- the real service on fake receivers: one session's verdict never serves another ---------------------------


async def test_a_ready_session_does_not_admit_the_next_and_its_end_leaves_no_ready(unused_tcp_port, monkeypatch):
    gate = asyncio.Event()
    factory = RealServiceFactory(["healthy", "healthy"], gates={1: gate})
    monkeypatch.setattr(server_module, "NetworkCheckService", factory)
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("A")))
            begun = await _recv_until(ws, p.SessionBegun)
            assert (begun.status.network_check.state, begun.status.network_check.session_id) == (
                State.CHECKING,
                "A",
            )
            await _check_state(unused_tcp_port, State.READY, "A")
            await _stop_session(ws, "A")
            await poll_until(lambda: factory.service.snapshot().session_id is None)
            after_end = (await _query_status(unused_tcp_port)).network_check
            assert (after_end.state, after_end.session_id) == (State.NOT_TESTED, None)
            assert after_end.cameras[0].passed is None

            await ws.send(json.dumps(_begin("B")))
            begun_b = await _recv_until(ws, p.SessionBegun)
            check_b = begun_b.status.network_check
            assert (check_b.state, check_b.session_id) == (State.CHECKING, "B")
            # B's own check is held at its join: it is CHECKING for B, never A's READY.
            await asyncio.sleep(0.2)
            held = (await _query_status(unused_tcp_port)).network_check
            assert (held.state, held.session_id) == (State.CHECKING, "B")
            gate.set()
            await _check_state(unused_tcp_port, State.READY, "B")
            assert len(factory.receivers) == 2
    finally:
        await stop_capture_server(running)


async def test_a_failing_check_for_one_session_is_not_shown_for_the_next(unused_tcp_port, monkeypatch):
    gate = asyncio.Event()
    factory = RealServiceFactory(["lost", "healthy"], gates={1: gate})
    monkeypatch.setattr(server_module, "NetworkCheckService", factory)
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("A")))
            await _recv_until(ws, p.SessionBegun)
            await _check_state(unused_tcp_port, State.UNSUITABLE, "A")
            failed = (await _query_status(unused_tcp_port)).network_check
            assert failed.reason
            await _stop_session(ws, "A")
            await ws.send(json.dumps(_begin("B")))
            await _recv_until(ws, p.SessionBegun)
            check_b = (await _query_status(unused_tcp_port)).network_check
            assert (check_b.state, check_b.session_id, check_b.reason) == (State.CHECKING, "B", None)
            gate.set()
            await _check_state(unused_tcp_port, State.READY, "B")
    finally:
        await stop_capture_server(running)


async def test_a_replay_and_a_retry_run_the_real_check_the_expected_number_of_times(unused_tcp_port, monkeypatch):
    factory = RealServiceFactory(["healthy"])
    monkeypatch.setattr(server_module, "NetworkCheckService", factory)
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("A")))
            await _recv_until(ws, p.SessionBegun)
            await ws.send(json.dumps(_begin("A")))
            await _recv_until(ws, p.SessionBegun)
            await _check_state(unused_tcp_port, State.READY, "A")
            assert len(factory.receivers) == 1
            await ws.send(_run_request("again"))
            await poll_until(lambda: len(factory.receivers) == 2)
            await _check_state(unused_tcp_port, State.READY, "A")
    finally:
        await stop_capture_server(running)


async def test_ending_a_session_during_its_check_cancels_it_and_leaves_the_room_once(unused_tcp_port, monkeypatch):
    factory = RealServiceFactory(["healthy"])
    monkeypatch.setattr(server_module, "NetworkCheckService", factory)
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("A")))
            await _recv_until(ws, p.SessionBegun)
            await poll_until(lambda: factory.receivers and factory.receivers[0].starts == 1)
            await _stop_session(ws, "A")
            await poll_until(lambda: factory.receivers[0].stops == 1)
            await asyncio.sleep(0.1)
            assert factory.receivers[0].stops == 1
            check = (await _query_status(unused_tcp_port)).network_check
            assert (check.state, check.session_id) == (State.NOT_TESTED, None)
    finally:
        await stop_capture_server(running)


async def test_a_slow_room_leave_for_one_session_does_not_block_the_next_begin_or_reset_its_check(
    unused_tcp_port, monkeypatch
):
    factory = RealServiceFactory(["healthy"])
    release = asyncio.Event()
    entered = asyncio.Event()
    begun: list[str] = []

    def build(streams, policy, make_token, **kwargs):
        service = factory(streams, policy, make_token, **kwargs)
        real_end, real_begin = service.end_session, service.begin_session

        async def slow_end(session_id):
            if session_id == "A":
                entered.set()
                await release.wait()
            await real_end(session_id)

        def recording_begin(session_id):
            begun.append(session_id)
            real_begin(session_id)

        service.end_session, service.begin_session = slow_end, recording_begin
        return service

    monkeypatch.setattr(server_module, "NetworkCheckService", build)
    running = await start_capture_server(unused_tcp_port, config=_config())
    try:
        async with _connect(unused_tcp_port) as ws:
            await ws.send(json.dumps(_begin("A")))
            await _recv_until(ws, p.SessionBegun)
            await _check_state(unused_tcp_port, State.READY, "A")
            await ws.send(_stop("A"))
            await _recv_until(ws, p.SessionStopped)
            await asyncio.wait_for(entered.wait(), 5)

            await ws.send(json.dumps(_begin("B")))
            begun_b = await _recv_until(ws, p.SessionBegun)
            assert begun_b.session_id == "B"
            assert begun == ["A", "B"]
            assert "A" in running.server._completed

            await ws.send(_stop("A"))
            replay = await _recv_until(ws, p.SessionStopped)
            assert replay.session_id == "A"

            release.set()
            await asyncio.sleep(0.2)
            check = (await _query_status(unused_tcp_port)).network_check
            assert check.session_id == "B"
            assert check.state is not State.NOT_TESTED
    finally:
        release.set()
        await stop_capture_server(running)
