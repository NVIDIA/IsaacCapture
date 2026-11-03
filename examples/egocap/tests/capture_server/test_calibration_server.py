# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CaptureServer calibration sessions over real sockets, with the scripted fake worker as a real process.

Admission, ownership and message-ordering rules are asserted at the wire level on one raw connection (the
server admits a single owner connection, so capture/calibration exclusivity is observed on that one socket);
the typed client API is exercised end to end through CalibrationRun.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from capture_server import protocol as p
from capture_server.calibration import CalibrationConfig, CalibrationSession
from capture_server.client import CaptureClient
from capture_server.server import ServerConfig
from tests.capture_server.calibration_helpers import (
    BOTH,
    CONTINUOUS_STEPS,
    LEFT,
    RIGHT,
    CalibrationRun,
    connect,
    is_type,
    next_prompt,
    recv_event,
    recv_until,
    send,
    start_over_raw,
    terminal_of,
)
from tests.support.capture_server import poll_until


def _begin(session_id: str = "cap-1") -> dict:
    return {
        "type": int(p.MessageType.BEGIN),
        "version": 1,
        "session_id": session_id,
        "tracking": {"sources": [1], "rate_hz": 60, "headset": {"device_index": 0}},
        "plugins": [],
        "cameras": [],
    }


def _capture_stop(session_id: str) -> dict:
    return {"type": int(p.MessageType.STOP), "version": 1, "session_id": session_id}


async def _stop_calibration_and_wait(ws, session_id: str = "cal-1") -> p.CalibrationStopped:
    await send(ws, p.StopCalibrationRequest(session_id))
    return await terminal_of(ws)


async def _rejection(ws) -> p.RequestRejected:
    event, _ = await recv_until(ws, is_type(p.RequestRejected))
    return event


# ---------------------------------------------------------------------------
# Happy paths through the typed client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sides", [(LEFT,), (RIGHT,), BOTH], ids=["left", "right", "both"])
async def test_a_calibration_completes_with_one_saved_file_per_requested_side(make_station, sides):
    station = await make_station()
    run = CalibrationRun(station.url)
    await run.start(sides)
    stopped = await run.wait()

    assert run.started[0].sides == sides
    assert stopped.reason is p.CalibrationStopReason.COMPLETED
    assert stopped.error_message is None
    assert [f.side for f in stopped.result.files] == list(sides)
    for file in stopped.result.files:
        assert (station.directory / file.filename).is_file()
        assert "/" not in file.filename
    assert len(list(station.directory.glob("*.mcal"))) == len(sides)
    assert not run.errors


async def test_every_prompt_needs_its_own_confirmation_including_the_second_gloves_first(make_station):
    station = await make_station()
    run = CalibrationRun(station.url)
    await run.start(BOTH)
    await run.wait()

    assert [step.token for step in run.prompts()] == ["left-0", "left-1", "right-0", "right-1"]
    assert run.confirmed == ["left-0", "left-1", "right-0", "right-1"]
    assert station.commands().count("start") == 1
    assert station.commands().count("confirm") == 4
    assert all(step.count == 2 for step in run.prompts())


async def test_a_timed_step_reports_elapsed_progress_in_snapshots(make_station):
    station = await make_station()
    run = CalibrationRun(station.url)
    await run.start((LEFT,))
    await run.wait()

    capturing = [s for s in run.statuses if s.phase is p.CalibrationPhase.CAPTURING_STEP]
    assert capturing
    assert all(s.step.mode is p.CalibrationStepMode.MOTION for s in capturing)
    assert any(s.step_elapsed_seconds for s in capturing)
    assert all(
        s.step_elapsed_seconds <= s.step.duration_seconds + 1e-6 for s in capturing if s.step_elapsed_seconds
    )


async def test_continuous_step_metadata_survives_worker_to_public_status_to_client_callback(make_station):
    station = await make_station(scenario={"steps": CONTINUOUS_STEPS})
    run = CalibrationRun(station.url)
    await run.start((LEFT,))
    stopped = await run.wait()

    first, second = run.prompts()
    assert (first.continuous, first.estimated_seconds, first.duration_seconds) == (True, 3.0, 0.0)
    assert (second.continuous, second.estimated_seconds, second.duration_seconds) == (True, 2.0, 0.0)
    assert first.mode is p.CalibrationStepMode.HOLD
    assert (first.title, first.instruction) == ("Make a fist", "Lift up your hand and make a fist.")
    assert stopped.reason is p.CalibrationStopReason.COMPLETED


async def test_the_station_is_free_again_after_a_calibration(make_station):
    station = await make_station()
    first = CalibrationRun(station.url)
    await first.start((LEFT,))
    await first.wait()
    second = CalibrationRun(station.url)
    await second.start((RIGHT,))
    assert (await second.wait()).reason is p.CalibrationStopReason.COMPLETED


# ---------------------------------------------------------------------------
# Exclusive admission
# ---------------------------------------------------------------------------


async def test_a_capture_begin_during_a_calibration_is_rejected_busy(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        assert isinstance(await start_over_raw(ws), p.CalibrationStarted)
        await send(ws, _begin())
        rejected = await _rejection(ws)
        assert rejected.request_type is p.MessageType.BEGIN
        assert rejected.reason is p.RejectionReason.BUSY
        assert (await _stop_calibration_and_wait(ws)).reason is p.CalibrationStopReason.CANCELLED
    finally:
        await ws.close()


async def test_a_calibration_start_during_a_capture_is_rejected_busy(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await send(ws, _begin())
        await recv_until(ws, is_type(p.SessionInit))
        await send(ws, p.StartCalibrationRequest("cal-1", (LEFT,)))
        rejected = await _rejection(ws)
        assert rejected.request_type is p.MessageType.START_CALIBRATION
        assert rejected.reason is p.RejectionReason.BUSY
        await send(ws, _capture_stop("cap-1"))
        await recv_until(ws, is_type(p.SessionStopped))
    finally:
        await ws.close()


async def test_two_near_simultaneous_starts_admit_exactly_one(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        # Sent back to back: the first accepted request must already occupy the station when the second is read.
        await send(ws, _begin())
        await send(ws, p.StartCalibrationRequest("cal-1", (LEFT,)))
        rejected, seen = await recv_until(ws, is_type(p.RequestRejected))
        assert rejected.request_type is p.MessageType.START_CALIBRATION
        assert rejected.reason is p.RejectionReason.BUSY
        # The accepted session's own first message may be written after the rejection of the later request.
        if not any(isinstance(event, p.SessionInit) for event in seen):
            _, later = await recv_until(ws, is_type(p.SessionInit))
            seen += later
        assert any(isinstance(event, p.SessionInit) for event in seen)
        assert not any(isinstance(event, p.CalibrationStarted) for event in seen)
        await send(ws, _capture_stop("cap-1"))
        await recv_until(ws, is_type(p.SessionStopped))
    finally:
        await ws.close()


async def test_a_calibration_first_then_a_capture_admits_only_the_calibration(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await send(ws, p.StartCalibrationRequest("cal-1", (LEFT,)))
        await send(ws, _begin())
        rejected, seen = await recv_until(ws, is_type(p.RequestRejected))
        assert rejected.request_type is p.MessageType.BEGIN
        # The accepted session's own first message may be written after the rejection of the later request.
        if not any(isinstance(event, p.CalibrationStarted) for event in seen):
            _, later = await recv_until(ws, is_type(p.CalibrationStarted))
            seen += later
        assert any(isinstance(event, p.CalibrationStarted) for event in seen)
        assert not any(isinstance(event, p.SessionInit) for event in seen)
        assert (await _stop_calibration_and_wait(ws)).reason is p.CalibrationStopReason.CANCELLED
    finally:
        await ws.close()


async def test_a_capture_stop_addressed_at_a_calibration_id_does_not_stop_it(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        await send(ws, _capture_stop("cal-1"))
        rejected = await _rejection(ws)
        assert rejected.request_type is p.MessageType.STOP
        assert rejected.reason is p.RejectionReason.UNKNOWN_SESSION
        # Still alive: the worker keeps offering its steps.
        assert (await next_prompt(ws)).step.token == "left-0"
        assert (await _stop_calibration_and_wait(ws)).reason is p.CalibrationStopReason.CANCELLED
    finally:
        await ws.close()


async def test_a_stop_calibration_addressed_at_a_capture_id_does_not_stop_the_capture(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await send(ws, _begin("cap-1"))
        await recv_until(ws, is_type(p.SessionInit))
        await send(ws, p.StopCalibrationRequest("cap-1"))
        rejected = await _rejection(ws)
        assert rejected.request_type is p.MessageType.STOP_CALIBRATION
        assert rejected.reason is p.RejectionReason.UNKNOWN_SESSION
        # The capture is still running: its own Stop ends it with the requested reason.
        await send(ws, _capture_stop("cap-1"))
        stopped, _ = await recv_until(ws, is_type(p.SessionStopped))
        assert stopped.reason is p.StopReason.REQUESTED
    finally:
        await ws.close()


async def test_a_session_id_cannot_be_reused_after_a_calibration_finished(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        await _stop_calibration_and_wait(ws, "cal-1")
        await poll_until(lambda: station.server._active_session is None, timeout_s=10)
        await send(ws, p.StartCalibrationRequest("cal-1", (LEFT,)))
        rejected = await _rejection(ws)
        assert rejected.reason is p.RejectionReason.INVALID_CONFIG
        assert "already used" in rejected.error_message
    finally:
        await ws.close()


async def test_a_stop_retry_after_the_terminal_replays_it(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        first = await _stop_calibration_and_wait(ws, "cal-1")
        await poll_until(lambda: station.server._active_session is None, timeout_s=10)
        await send(ws, p.StopCalibrationRequest("cal-1"))
        assert await terminal_of(ws) == first
    finally:
        await ws.close()


async def test_a_stop_for_an_unknown_calibration_is_rejected(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await send(ws, p.StopCalibrationRequest("never-started"))
        rejected = await _rejection(ws)
        assert rejected.request_type is p.MessageType.STOP_CALIBRATION
        assert rejected.reason is p.RejectionReason.UNKNOWN_SESSION
    finally:
        await ws.close()


# ---------------------------------------------------------------------------
# Failed cleanup keeps the station unavailable
# ---------------------------------------------------------------------------


async def test_failed_cleanup_keeps_both_capture_and_calibration_rejected(make_station, monkeypatch):
    station = await make_station()
    # The supervisor's unreleased latch is modelled directly: a worker that could not be reaped.
    monkeypatch.setattr(CalibrationSession, "cleanup_ok", property(lambda self: False))
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        terminal = await _stop_calibration_and_wait(ws, "cal-1")
        await asyncio.sleep(0.3)
        assert station.server._active_session is not None

        await send(ws, _begin("cap-2"))
        capture = await _rejection(ws)
        assert (capture.request_type, capture.reason) == (p.MessageType.BEGIN, p.RejectionReason.BUSY)

        await send(ws, p.StartCalibrationRequest("cal-2", (LEFT,)))
        calibration = await _rejection(ws)
        assert (calibration.request_type, calibration.reason) == (
            p.MessageType.START_CALIBRATION,
            p.RejectionReason.BUSY,
        )

        # The held session answers a Stop retry with the terminal it already produced.
        await send(ws, p.StopCalibrationRequest("cal-1"))
        assert await terminal_of(ws) == terminal
    finally:
        await ws.close()


async def test_a_worker_that_reports_the_sdk_unreleased_ends_uncertain_and_keeps_the_station_unavailable(
    make_station,
):
    station = await make_station(scenario={"unreleased": True})
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        # Stopping before the worker has started never touches the SDK and is a plain cancellation.
        await next_prompt(ws)
        terminal = await _stop_calibration_and_wait(ws, "cal-1")
        assert terminal.reason is p.CalibrationStopReason.UNCERTAIN
        assert terminal.result is None
        assert "release" in terminal.error_message
        assert not list(station.directory.glob("*"))

        await send(ws, p.StartCalibrationRequest("cal-2", (LEFT,)))
        calibration = await _rejection(ws)
        assert (calibration.request_type, calibration.reason) == (
            p.MessageType.START_CALIBRATION,
            p.RejectionReason.BUSY,
        )

        await send(ws, _begin("cap-2"))
        capture = await _rejection(ws)
        assert (capture.request_type, capture.reason) == (p.MessageType.BEGIN, p.RejectionReason.BUSY)

        assert station.commands().count("start") == 1
    finally:
        await ws.close()


async def test_a_worker_that_may_still_hold_the_gloves_refuses_a_new_calibration_with_its_reason(make_station):
    station = await make_station()
    station.server._calibration_worker.unreleased = "the worker (pid 1) could not be stopped"
    ws = await connect(station.url)
    try:
        await send(ws, p.StartCalibrationRequest("cal-1", (LEFT,)))
        rejected = await _rejection(ws)
        assert rejected.reason is p.RejectionReason.BUSY
        assert "could not be stopped" in rejected.error_message
        assert station.server._active_session is None
    finally:
        await ws.close()


# ---------------------------------------------------------------------------
# Confirmations
# ---------------------------------------------------------------------------


async def test_a_duplicate_confirm_of_an_accepted_token_starts_only_one_step(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        token = (await next_prompt(ws)).step.token
        await send(ws, p.ConfirmCalibrationStepRequest("cal-1", token))
        first, _ = await recv_until(ws, is_type(p.CalibrationStepAccepted))
        await send(ws, p.ConfirmCalibrationStepRequest("cal-1", token))
        second, _ = await recv_until(ws, is_type(p.CalibrationStepAccepted))
        assert first.step_token == second.step_token == token
        assert station.commands().count("confirm") == 1
        await _stop_calibration_and_wait(ws)
    finally:
        await ws.close()


async def test_a_stale_token_is_rejected_with_the_token_echoed_and_the_attempt_unchanged(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        offered = (await next_prompt(ws)).step.token
        await send(ws, p.ConfirmCalibrationStepRequest("cal-1", "left-7"))
        rejected = await _rejection(ws)
        assert rejected.request_type is p.MessageType.CONFIRM_CALIBRATION_STEP
        assert rejected.step_token == "left-7"
        assert rejected.reason is p.RejectionReason.INVALID_CONFIG
        # The worker never saw it, and the offered prompt can still be confirmed.
        assert station.commands().count("confirm") == 0
        await send(ws, p.ConfirmCalibrationStepRequest("cal-1", offered))
        accepted, _ = await recv_until(ws, is_type(p.CalibrationStepAccepted))
        assert accepted.step_token == offered
        await _stop_calibration_and_wait(ws)
    finally:
        await ws.close()


async def test_a_confirm_for_an_unknown_calibration_is_rejected_with_the_token_echoed(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await send(ws, p.ConfirmCalibrationStepRequest("nope", "left-0"))
        rejected = await _rejection(ws)
        assert rejected.reason is p.RejectionReason.UNKNOWN_SESSION
        assert rejected.step_token == "left-0"
    finally:
        await ws.close()


async def test_a_slow_confirm_does_not_hold_up_a_stop_calibration_behind_it(make_station):
    station = await make_station(scenario={"slow_confirm": 5.0})
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        token = (await next_prompt(ws)).step.token
        await send(ws, p.ConfirmCalibrationStepRequest("cal-1", token))
        began = time.monotonic()
        await send(ws, p.StopCalibrationRequest("cal-1"))
        terminal = await terminal_of(ws)
        assert terminal.reason is p.CalibrationStopReason.CANCELLED
        assert time.monotonic() - began < 4.0
    finally:
        await ws.close()


async def test_a_confirm_after_the_calibration_ended_is_rejected(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        token = (await next_prompt(ws)).step.token
        await _stop_calibration_and_wait(ws)
        await poll_until(lambda: station.server._active_session is None, timeout_s=10)
        await send(ws, p.ConfirmCalibrationStepRequest("cal-1", token))
        rejected = await _rejection(ws)
        assert rejected.reason is p.RejectionReason.UNKNOWN_SESSION
    finally:
        await ws.close()


# ---------------------------------------------------------------------------
# Start replay and conflicts
# ---------------------------------------------------------------------------


async def test_an_identical_repeated_start_replays_acceptance_and_never_starts_a_second_worker(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        first = await start_over_raw(ws, "cal-1", (LEFT,))
        await send(ws, p.StartCalibrationRequest("cal-1", (LEFT,)))
        replay, _ = await recv_until(ws, is_type(p.CalibrationStarted))
        assert replay == first
        await next_prompt(ws)
        assert station.commands().count("start") == 1
        await _stop_calibration_and_wait(ws)
    finally:
        await ws.close()


async def test_a_start_with_the_same_id_but_different_sides_is_rejected(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1", (LEFT,))
        await send(ws, p.StartCalibrationRequest("cal-1", BOTH))
        rejected = await _rejection(ws)
        assert rejected.reason is p.RejectionReason.INVALID_CONFIG
        assert "differs" in rejected.error_message
        await _stop_calibration_and_wait(ws)
    finally:
        await ws.close()


# ---------------------------------------------------------------------------
# Cancellation and cleanup
# ---------------------------------------------------------------------------


async def test_a_stop_calibration_cancels_and_reports_cancelled_without_files(make_station):
    station = await make_station()
    run = CalibrationRun(station.url, auto_confirm=False)
    await run.start((LEFT,))
    await poll_until(lambda: run.prompts(), timeout_s=10)
    await run.client.stop_calibration()
    await run.client.stop_calibration()  # idempotent
    stopped = await run.wait()
    assert stopped.reason is p.CalibrationStopReason.CANCELLED
    assert stopped.result is None
    assert not station.directory.exists() or list(station.directory.iterdir()) == []


async def test_the_owner_disconnecting_cancels_the_calibration_and_reaps_the_worker(make_station):
    station = await make_station()
    ws = await connect(station.url)
    await start_over_raw(ws, "cal-1")
    await next_prompt(ws)
    await ws.close()

    await poll_until(lambda: station.server._active_session is None, timeout_s=20)
    assert station.server._calibration_worker.unreleased is None
    staging = station.directory / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


async def test_closing_the_server_cancels_an_active_calibration(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        await next_prompt(ws)
        closing = asyncio.ensure_future(station.server.close())
        terminal = await terminal_of(ws)
        assert terminal.reason is p.CalibrationStopReason.CANCELLED
        await asyncio.wait_for(closing, timeout=15)
        await poll_until(lambda: station.server._active_session is None, timeout_s=10)
    finally:
        await ws.close()


async def test_a_calibration_never_publishes_files_when_the_worker_fails(make_station):
    station = await make_station(scenario={"fail_at_step": ["left", 1]})
    run = CalibrationRun(station.url)
    await run.start((LEFT,))
    stopped = await run.wait()
    assert stopped.reason is p.CalibrationStopReason.FAILED
    assert stopped.result is None
    assert "refused" in stopped.error_message
    assert not station.directory.exists() or list(station.directory.iterdir()) == []


# ---------------------------------------------------------------------------
# Unavailable calibration leaves capture alone
# ---------------------------------------------------------------------------


async def test_without_a_worker_and_directory_a_start_is_rejected_with_a_useful_message(make_station):
    station = await make_station(calibration=False)
    ws = await connect(station.url)
    try:
        await send(ws, p.StartCalibrationRequest("cal-1", (LEFT,)))
        rejected = await _rejection(ws)
        assert rejected.request_type is p.MessageType.START_CALIBRATION
        assert rejected.reason is p.RejectionReason.INVALID_CONFIG
        assert "not available" in rejected.error_message
        assert "worker" in rejected.error_message and "directory" in rejected.error_message
    finally:
        await ws.close()


async def test_ordinary_capture_still_works_when_calibration_is_unavailable(make_station):
    station = await make_station(calibration=False)
    stopped: list = []
    done = asyncio.Event()
    client = CaptureClient(station.url)
    await client.start(
        sources=(p.SourceType.HEAD,),
        on_stopped=lambda r: (stopped.append(r), done.set()),
    )
    await asyncio.sleep(0.2)
    await client.stop()
    await asyncio.wait_for(done.wait(), timeout=10)
    assert stopped[0].reason is p.StopReason.REQUESTED


async def test_ordinary_capture_still_works_when_calibration_is_configured(make_station):
    station = await make_station()
    stopped: list = []
    done = asyncio.Event()
    client = CaptureClient(station.url)
    await client.start(
        sources=(p.SourceType.HEAD,),
        on_stopped=lambda r: (stopped.append(r), done.set()),
    )
    await asyncio.sleep(0.2)
    await client.stop()
    await asyncio.wait_for(done.wait(), timeout=10)
    assert stopped[0].reason is p.StopReason.REQUESTED


async def test_a_configured_worker_that_cannot_be_launched_ends_the_calibration_failed(make_station, tmp_path):
    config = ServerConfig(
        calibration=CalibrationConfig(
            worker_path=Path("/nonexistent/manus_glove_calibrator"), directory=tmp_path / "m"
        )
    )
    station = await make_station(calibration=False, config=config)
    ws = await connect(station.url)
    try:
        assert isinstance(await start_over_raw(ws, "cal-1"), p.CalibrationStarted)
        terminal = await terminal_of(ws)
        assert terminal.reason is p.CalibrationStopReason.FAILED
        assert terminal.result is None
        assert "cannot launch" in terminal.error_message
        await poll_until(lambda: station.server._active_session is None, timeout_s=10)
    finally:
        await ws.close()


# ---------------------------------------------------------------------------
# Wire-level message ordering
# ---------------------------------------------------------------------------


async def test_started_precedes_every_status_and_the_terminal_message_is_last(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await send(ws, p.StartCalibrationRequest("cal-1", (LEFT,)))
        events = []
        stop_sent = False
        while True:
            event = await recv_event(ws, timeout=15)
            events.append(event)
            if prompt_ready(event) and not stop_sent:
                stop_sent = True
                await send(ws, p.StopCalibrationRequest("cal-1"))
            if isinstance(event, p.CalibrationStopped):
                break
        assert isinstance(events[0], p.CalibrationStarted)
        assert sum(isinstance(e, p.CalibrationStopped) for e in events) == 1
        assert isinstance(events[-1], p.CalibrationStopped)
        assert all(e.session_id == "cal-1" for e in events)
    finally:
        await ws.close()


def prompt_ready(event) -> bool:
    return (
        isinstance(event, p.CalibrationStatus)
        and event.phase is p.CalibrationPhase.AWAITING_STEP
        and event.step is not None
    )


async def test_the_accepted_acknowledgement_arrives_before_the_next_prompt(make_station):
    station = await make_station(scenario={"steps": CONTINUOUS_STEPS})
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        first = (await next_prompt(ws)).step.token
        await send(ws, p.ConfirmCalibrationStepRequest("cal-1", first))
        _, seen = await recv_until(ws, lambda e: prompt_ready(e) and e.step.token != first)
        accepted_at = next(i for i, e in enumerate(seen) if isinstance(e, p.CalibrationStepAccepted))
        assert accepted_at < len(seen) - 1
        await _stop_calibration_and_wait(ws)
    finally:
        await ws.close()


async def test_calibration_messages_are_json_text_only_and_carry_no_capture_frames(make_station):
    station = await make_station()
    ws = await connect(station.url)
    try:
        await start_over_raw(ws, "cal-1")
        await next_prompt(ws)
        await send(ws, p.StopCalibrationRequest("cal-1"))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            assert isinstance(raw, str)
            message = json.loads(raw)
            assert message["type"] not in (int(p.MessageType.CAPTURE_FRAME), int(p.MessageType.CAPTURE_IMAGE))
            if message["type"] == int(p.MessageType.CALIBRATION_STOPPED):
                break
    finally:
        await ws.close()
