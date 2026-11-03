# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CaptureClient's calibration API against small scripted servers, so each test controls exactly what the
server sends: acceptance, rejection, statuses, acknowledgements, terminal messages, transport loss and
protocol violations."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
import websockets

from capture_server import protocol as p
from capture_server.client import (
    CalibrationEndedError,
    CalibrationStopped,
    CaptureClient,
    RequestRejectedError,
    StoppedBeforeAcceptedError,
)
from tests.capture_server.calibration_helpers import BOTH, LEFT, RIGHT, CalibrationRun
from tests.support.capture_server import poll_until

Script = Callable[[Any, p.StartCalibrationRequest], Awaitable[None]]


@contextlib.asynccontextmanager
async def scripted_server(port: int, script: Script) -> AsyncIterator[str]:
    """A server that reads one StartCalibration and hands the connection to ``script``."""

    async def handler(ws) -> None:
        request = p.parse_start_calibration(json.loads(await ws.recv()))
        await script(ws, request)

    server = await websockets.serve(handler, "127.0.0.1", port)
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


async def push(ws, event) -> None:
    await ws.send(json.dumps(event.to_json()))


def step(token: str = "left-0", **overrides) -> p.CalibrationStep:
    fields = dict(
        token=token,
        side=LEFT,
        index=0,
        count=2,
        title="Make a fist",
        instruction="Lift up your hand and make a fist.",
        mode=p.CalibrationStepMode.HOLD,
        duration_seconds=0.0,
        continuous=False,
        estimated_seconds=0.0,
    )
    fields.update(overrides)
    return p.CalibrationStep(**fields)


def status(session_id: str, offered: p.CalibrationStep | None = None, phase=None) -> p.CalibrationStatus:
    phase = phase or (p.CalibrationPhase.AWAITING_STEP if offered else p.CalibrationPhase.CONNECTING)
    return p.CalibrationStatus(session_id, phase, (p.CalibrationGlove(LEFT, 11),), offered, None)


def stopped(session_id: str, reason=p.CalibrationStopReason.CANCELLED, result=None, message=None):
    return p.CalibrationStopped(session_id, reason, result, message)


class Recorder:
    """Collects callback values and whether callbacks ever overlapped."""

    def __init__(self) -> None:
        self.started: list[p.CalibrationStarted] = []
        self.statuses: list[p.CalibrationStatus] = []
        self.stopped: list[CalibrationStopped] = []
        self.done = asyncio.Event()
        self.running = 0
        self.max_running = 0

    def on_started(self, event) -> None:
        self.started.append(event)

    async def on_status(self, event) -> None:
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        await asyncio.sleep(0.01)
        self.statuses.append(event)
        self.running -= 1

    def on_stopped(self, event: CalibrationStopped) -> None:
        self.stopped.append(event)
        self.done.set()


async def start(client: CaptureClient, recorder: Recorder, sides=(LEFT,)) -> None:
    await client.start_calibration(
        sides=sides, on_started=recorder.on_started, on_status=recorder.on_status, on_stopped=recorder.on_stopped
    )


# ---------------------------------------------------------------------------
# Acceptance and rejection
# ---------------------------------------------------------------------------


async def test_start_calibration_returns_after_acceptance_not_after_completion(unused_tcp_port):
    release = asyncio.Event()

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await release.wait()
        await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        assert len(recorder.started) == 1
        assert recorder.stopped == []
        release.set()
        await asyncio.wait_for(recorder.done.wait(), 5)


async def test_a_rejected_start_raises_and_calls_no_callback_and_leaves_the_client_free(unused_tcp_port):
    async def script(ws, request):
        await push(
            ws,
            p.RequestRejected(request.session_id, p.MessageType.START_CALIBRATION, p.RejectionReason.BUSY, "busy"),
        )

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        with pytest.raises(RequestRejectedError) as excinfo:
            await start(client, recorder)
        assert excinfo.value.reason is p.RejectionReason.BUSY
        assert str(excinfo.value) == "busy"
        assert (recorder.started, recorder.statuses, recorder.stopped) == ([], [], [])
        assert client._calibration is None
        with pytest.raises(RequestRejectedError):
            await start(client, recorder)


async def test_a_calibration_started_for_other_sides_is_a_protocol_error_before_acceptance(unused_tcp_port):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, (RIGHT,)))

    async with scripted_server(unused_tcp_port, script) as url:
        with pytest.raises(p.ProtocolError):
            await start(CaptureClient(url), Recorder())


async def test_a_wrong_first_message_is_a_protocol_error_before_acceptance(unused_tcp_port):
    async def script(ws, request):
        await push(ws, status(request.session_id))

    async with scripted_server(unused_tcp_port, script) as url:
        recorder = Recorder()
        with pytest.raises(p.ProtocolError):
            await start(CaptureClient(url), recorder)
        assert recorder.stopped == []


async def test_a_stop_before_acceptance_raises_stopped_before_accepted_and_calls_no_stopped_callback(
    unused_tcp_port,
):
    never = asyncio.Event()

    async def script(ws, request):
        await never.wait()

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        pending = asyncio.ensure_future(start(client, recorder))
        await asyncio.sleep(0.1)
        await client.stop_calibration()
        with pytest.raises(StoppedBeforeAcceptedError):
            await asyncio.wait_for(pending, 5)
        assert recorder.stopped == []
        never.set()


# ---------------------------------------------------------------------------
# Argument validation and local exclusivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sides", [(), (LEFT, LEFT), (1,), ("left",)])
async def test_start_calibration_rejects_invalid_sides_before_connecting(sides):
    client = CaptureClient("ws://127.0.0.1:1")
    with pytest.raises(ValueError, match="start_calibration"):
        await client.start_calibration(sides=sides)  # type: ignore[arg-type]


async def test_start_calibration_rejects_a_non_callable_callback():
    client = CaptureClient("ws://127.0.0.1:1")
    with pytest.raises(ValueError, match="on_status"):
        await client.start_calibration(sides=(LEFT,), on_status="nope")  # type: ignore[arg-type]


async def test_a_second_start_calibration_while_one_is_active_raises_runtime_error(unused_tcp_port):
    hold = asyncio.Event()

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await hold.wait()
        await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        with pytest.raises(RuntimeError, match="active"):
            await start(client, Recorder())
        with pytest.raises(RuntimeError, match="active"):
            await client.start(sources=(p.SourceType.HEAD,))
        hold.set()
        await asyncio.wait_for(recorder.done.wait(), 5)
        assert client._calibration is None


async def test_start_calibration_while_a_capture_is_active_raises_runtime_error(make_station):
    station = await make_station()
    done = asyncio.Event()
    client = CaptureClient(station.url)
    await client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda _r: done.set())
    try:
        with pytest.raises(RuntimeError, match="active"):
            await client.start_calibration(sides=(LEFT,))
    finally:
        await client.stop()
        await asyncio.wait_for(done.wait(), 10)


async def test_confirm_without_an_accepted_calibration_raises_runtime_error():
    client = CaptureClient("ws://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="no accepted calibration"):
        await client.confirm_calibration_step(step_token="left-0")


async def test_confirm_rejects_an_empty_token(unused_tcp_port):
    hold = asyncio.Event()

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await hold.wait()
        await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        with pytest.raises(ValueError):
            await client.confirm_calibration_step(step_token="")
        hold.set()
        await asyncio.wait_for(recorder.done.wait(), 5)


async def test_stop_calibration_without_an_attempt_is_a_no_op():
    await CaptureClient("ws://127.0.0.1:1").stop_calibration()


# ---------------------------------------------------------------------------
# Statuses, acknowledgements and rejections of confirmations
# ---------------------------------------------------------------------------


async def test_statuses_are_delivered_in_order_and_callbacks_never_overlap(unused_tcp_port):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        for index in range(4):
            await push(ws, status(request.session_id, step(f"left-{index}", index=index, count=4)))
        await asyncio.sleep(0.3)
        await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        await start(CaptureClient(url), recorder)
        await asyncio.wait_for(recorder.done.wait(), 5)
    assert [s.step.token for s in recorder.statuses] == ["left-0", "left-1", "left-2", "left-3"]
    assert recorder.max_running == 1


async def test_confirm_returns_only_after_the_acceptance_and_a_retry_sends_no_second_request(unused_tcp_port):
    received: list[p.ConfirmCalibrationStepRequest] = []
    accept = asyncio.Event()
    finish = asyncio.Event()

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, status(request.session_id, step("left-0")))
        async for raw in ws:
            message = json.loads(raw)
            if message["type"] == int(p.MessageType.CONFIRM_CALIBRATION_STEP):
                received.append(p.parse_confirm_calibration_step(message))
                await accept.wait()
                await push(ws, p.CalibrationStepAccepted(request.session_id, "left-0"))
                await finish.wait()
                await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        confirming = asyncio.ensure_future(client.confirm_calibration_step(step_token="left-0"))
        await poll_until(lambda: received, timeout_s=5)
        assert not confirming.done()
        accept.set()
        await asyncio.wait_for(confirming, 5)
        await client.confirm_calibration_step(step_token="left-0")
        await asyncio.sleep(0.1)
        assert len(received) == 1
        finish.set()
        await asyncio.wait_for(recorder.done.wait(), 5)


async def test_two_concurrent_confirms_of_one_token_send_one_request_and_both_return(unused_tcp_port):
    received: list[str] = []

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        async for raw in ws:
            message = json.loads(raw)
            if message["type"] == int(p.MessageType.CONFIRM_CALIBRATION_STEP):
                received.append(message["step_token"])
                await asyncio.sleep(0.1)
                await push(ws, p.CalibrationStepAccepted(request.session_id, message["step_token"]))
                await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        await asyncio.wait_for(
            asyncio.gather(
                client.confirm_calibration_step(step_token="left-0"),
                client.confirm_calibration_step(step_token="left-0"),
            ),
            5,
        )
        assert received == ["left-0"]
        await asyncio.wait_for(recorder.done.wait(), 5)


async def test_a_rejected_confirm_raises_with_the_reason_and_the_attempt_continues(unused_tcp_port):
    finish = asyncio.Event()

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        async for raw in ws:
            message = json.loads(raw)
            if message["type"] == int(p.MessageType.CONFIRM_CALIBRATION_STEP):
                await push(
                    ws,
                    p.RequestRejected(
                        request.session_id,
                        p.MessageType.CONFIRM_CALIBRATION_STEP,
                        p.RejectionReason.INVALID_CONFIG,
                        "that step is not awaiting confirmation",
                        step_token=message["step_token"],
                    ),
                )
                await finish.wait()
                await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        with pytest.raises(RequestRejectedError) as excinfo:
            await client.confirm_calibration_step(step_token="stale")
        assert excinfo.value.reason is p.RejectionReason.INVALID_CONFIG
        assert recorder.stopped == []
        assert client._calibration is not None
        finish.set()
        await asyncio.wait_for(recorder.done.wait(), 5)
        assert len(recorder.stopped) == 1


# ---------------------------------------------------------------------------
# Terminal outcomes
# ---------------------------------------------------------------------------


async def test_a_completed_terminal_delivers_the_files_and_exactly_one_stopped_callback(unused_tcp_port):
    result = p.CalibrationResult((p.McalFile(LEFT, "set_left.mcal"), p.McalFile(RIGHT, "set_right.mcal")))

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, stopped(request.session_id, p.CalibrationStopReason.COMPLETED, result, None))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        await start(CaptureClient(url), recorder, BOTH)
        await asyncio.wait_for(recorder.done.wait(), 5)
        await asyncio.sleep(0.1)
    assert len(recorder.stopped) == 1
    assert recorder.stopped[0].reason is p.CalibrationStopReason.COMPLETED
    assert recorder.stopped[0].result == result
    assert recorder.stopped[0].error_message is None


@pytest.mark.parametrize(
    "reason",
    [p.CalibrationStopReason.CANCELLED, p.CalibrationStopReason.FAILED, p.CalibrationStopReason.UNCERTAIN],
)
async def test_non_completed_terminals_carry_the_servers_reason_and_no_files(unused_tcp_port, reason):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, stopped(request.session_id, reason, None, "the server says why"))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        await start(CaptureClient(url), recorder)
        await asyncio.wait_for(recorder.done.wait(), 5)
    assert recorder.stopped[0].reason is reason
    assert recorder.stopped[0].result is None
    assert recorder.stopped[0].error_message == "the server says why"


async def test_the_client_is_free_again_after_the_terminal(unused_tcp_port):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, stopped(request.session_id))

    first, second = Recorder(), Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, first)
        await asyncio.wait_for(first.done.wait(), 5)
        await poll_until(lambda: client._calibration is None, timeout_s=5)
        await start(client, second)
        await asyncio.wait_for(second.done.wait(), 5)


# ---------------------------------------------------------------------------
# Transport loss and protocol violations: an honest, unconfirmed ending
# ---------------------------------------------------------------------------


async def test_transport_loss_ends_with_one_unconfirmed_stopped_and_no_files(unused_tcp_port):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, status(request.session_id, step("left-0")))
        await ws.close()

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        await start(CaptureClient(url), recorder)
        await asyncio.wait_for(recorder.done.wait(), 5)
        await asyncio.sleep(0.1)
    assert len(recorder.stopped) == 1
    ending = recorder.stopped[0]
    assert ending.reason is None
    assert ending.result is None
    assert "outcome is unknown" in ending.error_message


async def test_a_pending_confirm_is_released_with_calibration_ended_when_the_transport_is_lost(unused_tcp_port):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        async for raw in ws:
            if json.loads(raw)["type"] == int(p.MessageType.CONFIRM_CALIBRATION_STEP):
                await ws.close()

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        with pytest.raises(CalibrationEndedError):
            await asyncio.wait_for(client.confirm_calibration_step(step_token="left-0"), 5)
        await asyncio.wait_for(recorder.done.wait(), 5)
    assert recorder.stopped[0].reason is None


async def test_a_pending_confirm_is_released_when_the_server_reports_the_terminal_first(unused_tcp_port):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        async for raw in ws:
            if json.loads(raw)["type"] == int(p.MessageType.CONFIRM_CALIBRATION_STEP):
                await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        with pytest.raises(CalibrationEndedError):
            await asyncio.wait_for(client.confirm_calibration_step(step_token="left-0"), 5)
        await asyncio.wait_for(recorder.done.wait(), 5)
    assert recorder.stopped[0].reason is p.CalibrationStopReason.CANCELLED


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda sid: b"binary", id="binary"),
        pytest.param(lambda sid: "not json", id="not-json"),
        pytest.param(lambda sid: json.dumps({"type": 99, "version": 1}), id="unknown-type"),
        pytest.param(
            lambda sid: json.dumps(
                p.CalibrationStatus("other-id", p.CalibrationPhase.CONNECTING, (), None, None).to_json()
            ),
            id="other-session",
        ),
        pytest.param(
            lambda sid: json.dumps({"type": int(p.MessageType.SESSION_STATUS), "version": 1, "session_id": sid}),
            id="malformed-capture-message",
        ),
    ],
)
async def test_a_protocol_violation_ends_with_an_unconfirmed_stopped(unused_tcp_port, bad):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await ws.send(bad(request.session_id))
        await asyncio.sleep(1)

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        await start(CaptureClient(url), recorder)
        await asyncio.wait_for(recorder.done.wait(), 5)
    assert len(recorder.stopped) == 1
    assert recorder.stopped[0].reason is None
    assert "protocol error" in recorder.stopped[0].error_message
    assert recorder.stopped[0].result is None


# ---------------------------------------------------------------------------
# Callbacks and stop
# ---------------------------------------------------------------------------


async def test_a_callback_exception_requests_cancellation_and_appears_in_the_terminal_diagnostic(unused_tcp_port):
    stops = 0

    async def script(ws, request):
        nonlocal stops
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, status(request.session_id))
        async for raw in ws:
            if json.loads(raw)["type"] == int(p.MessageType.STOP_CALIBRATION):
                stops += 1
                await push(ws, stopped(request.session_id))

    recorder = Recorder()

    def explode(_event) -> None:
        raise ValueError("boom")

    async with scripted_server(unused_tcp_port, script) as url:
        await CaptureClient(url).start_calibration(
            sides=(LEFT,), on_status=explode, on_stopped=recorder.on_stopped
        )
        await asyncio.wait_for(recorder.done.wait(), 5)
    assert stops == 1
    assert recorder.stopped[0].reason is p.CalibrationStopReason.CANCELLED
    assert "a callback raised" in recorder.stopped[0].error_message
    assert "boom" in recorder.stopped[0].error_message


async def test_an_on_started_exception_also_cancels_and_is_reported_once(unused_tcp_port):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        async for raw in ws:
            if json.loads(raw)["type"] == int(p.MessageType.STOP_CALIBRATION):
                await push(ws, stopped(request.session_id))

    recorder = Recorder()

    def explode(_event) -> None:
        raise RuntimeError("bad started handler")

    async with scripted_server(unused_tcp_port, script) as url:
        await CaptureClient(url).start_calibration(
            sides=(LEFT,), on_started=explode, on_stopped=recorder.on_stopped
        )
        await asyncio.wait_for(recorder.done.wait(), 5)
    assert len(recorder.stopped) == 1
    assert "bad started handler" in recorder.stopped[0].error_message


async def test_an_on_stopped_exception_is_reported_to_the_loop_once_and_not_repeated(unused_tcp_port):
    reports: list[dict] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: reports.append(context))
    calls = 0

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, stopped(request.session_id))

    def explode(_event) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("terminal handler failed")

    try:
        async with scripted_server(unused_tcp_port, script) as url:
            client = CaptureClient(url)
            await client.start_calibration(sides=(LEFT,), on_stopped=explode)
            await poll_until(lambda: reports, timeout_s=5)
            await poll_until(lambda: client._calibration is None, timeout_s=5)
    finally:
        loop.set_exception_handler(None)
    assert calls == 1
    assert len(reports) == 1
    assert "on_stopped" in reports[0]["message"]


async def test_stop_calibration_is_idempotent_and_safe_to_call_from_a_callback(unused_tcp_port):
    stops = 0

    async def script(ws, request):
        nonlocal stops
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, status(request.session_id))
        async for raw in ws:
            if json.loads(raw)["type"] == int(p.MessageType.STOP_CALIBRATION):
                stops += 1
                await push(ws, stopped(request.session_id))

    recorder = Recorder()
    client = CaptureClient("ws://placeholder")

    async def stopping_status(event) -> None:
        await client.stop_calibration()
        await client.stop_calibration()

    async with scripted_server(unused_tcp_port, script) as url:
        client._url = url
        await client.start_calibration(sides=(LEFT,), on_status=stopping_status, on_stopped=recorder.on_stopped)
        await asyncio.wait_for(recorder.done.wait(), 5)
        await client.stop_calibration()
    assert stops == 1
    assert recorder.stopped[0].reason is p.CalibrationStopReason.CANCELLED


async def test_confirm_can_be_awaited_from_inside_a_status_callback_without_deadlock(unused_tcp_port):
    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await push(ws, status(request.session_id, step("left-0")))
        async for raw in ws:
            message = json.loads(raw)
            if message["type"] == int(p.MessageType.CONFIRM_CALIBRATION_STEP):
                await push(ws, p.CalibrationStepAccepted(request.session_id, message["step_token"]))
                await push(ws, stopped(request.session_id))

    recorder = Recorder()
    confirmed: list[str] = []
    client = CaptureClient("ws://placeholder")

    async def confirming_status(event) -> None:
        if event.step is not None:
            await client.confirm_calibration_step(step_token=event.step.token)
            confirmed.append(event.step.token)

    async with scripted_server(unused_tcp_port, script) as url:
        client._url = url
        await client.start_calibration(sides=(LEFT,), on_status=confirming_status, on_stopped=recorder.on_stopped)
        await asyncio.wait_for(recorder.done.wait(), 5)
    assert confirmed == ["left-0"]


async def test_an_accepted_attempt_stays_active_until_the_server_ends_it(unused_tcp_port):
    hold = asyncio.Event()

    async def script(ws, request):
        await push(ws, p.CalibrationStarted(request.session_id, request.sides))
        await hold.wait()
        await push(ws, stopped(request.session_id))

    recorder = Recorder()
    async with scripted_server(unused_tcp_port, script) as url:
        client = CaptureClient(url)
        await start(client, recorder)
        await asyncio.sleep(0.1)
        assert client._calibration is not None
        assert recorder.stopped == []
        hold.set()
        await asyncio.wait_for(recorder.done.wait(), 5)


# ---------------------------------------------------------------------------
# Through the real server
# ---------------------------------------------------------------------------


async def test_the_real_client_and_server_complete_a_calibration_end_to_end(make_station):
    station = await make_station()
    run = CalibrationRun(station.url)
    await run.start(BOTH)
    ending = await run.wait()
    assert ending.reason is p.CalibrationStopReason.COMPLETED
    assert sorted(f.filename for f in ending.result.files) == sorted(
        path.name for path in station.directory.glob("*.mcal")
    )
