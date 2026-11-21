# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""capture_server.calibration without processes: a stub worker and outbox stand in for the supervisor and
the connection's single writer, so the translation of worker events into calibration messages, the terminal
ordering and the file rules are checked directly."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from isaaccapture_manus_calibration import protocol as wp
from isaaccapture_manus_calibration.worker import AttemptError, CalibrationOutcome, OutcomeKind

from capture_server import calibration as cal
from capture_server import protocol as p

ATTEMPT = "attempt-1"
LEFT, RIGHT = wp.Side.LEFT, wp.Side.RIGHT


# ---------------------------------------------------------------------------
# find_worker
# ---------------------------------------------------------------------------


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_find_worker_returns_an_executable_override(tmp_path):
    worker = _executable(tmp_path / "bin" / "custom-calibrator")
    assert cal.find_worker(worker) == worker


def test_find_worker_rejects_a_non_executable_or_missing_override(tmp_path):
    plain = tmp_path / "plain"
    plain.write_text("x")
    assert cal.find_worker(plain) is None
    assert cal.find_worker(tmp_path / "absent") is None


def test_find_worker_looks_under_the_manus_install_of_isaac_root(tmp_path, monkeypatch):
    worker = _executable(tmp_path / "install" / "plugins" / "manus" / cal.WORKER_NAME)
    monkeypatch.setenv("ISAAC_ROOT", str(tmp_path))
    assert cal.find_worker(None) == worker


def test_find_worker_is_none_without_isaac_root_or_override(monkeypatch):
    monkeypatch.delenv("ISAAC_ROOT", raising=False)
    assert cal.find_worker(None) is None


def test_find_worker_is_none_when_the_install_has_no_worker(tmp_path, monkeypatch):
    (tmp_path / "install" / "plugins" / "manus").mkdir(parents=True)
    monkeypatch.setenv("ISAAC_ROOT", str(tmp_path))
    assert cal.find_worker(None) is None


# ---------------------------------------------------------------------------
# CalibrationSession with a stub worker and outbox
# ---------------------------------------------------------------------------


class StubOutbox:
    """Records every call in one ordered log so cross-channel ordering can be asserted."""

    def __init__(self) -> None:
        self.closed = False
        self.log: list[tuple[str, Any]] = []

    def bind_session(self, stream_ids, *, on_sent, on_fatal) -> None:
        self.log.append(("bind", tuple(stream_ids)))
        self.on_fatal = on_fatal

    def publish_control(self, data) -> bool:
        self.log.append(("control", data))
        return True

    def publish_status(self, data) -> None:
        self.log.append(("status", data))

    async def quiesce(self) -> None:
        self.log.append(("quiesce", None))

    def controls(self, message_type: p.MessageType) -> list[dict]:
        return [d for kind, d in self.log if kind == "control" and d["type"] == int(message_type)]

    def statuses(self) -> list[dict]:
        return [d for kind, d in self.log if kind == "status"]


class StubAttempt:
    def __init__(self, on_event, script) -> None:
        self.on_event = on_event
        self.script = script
        self.prompt: wp.Prompt | None = None
        self.started_with = None
        self.cancel_calls = 0
        self.confirm_calls: list[str] = []
        self.confirm_error: Exception | None = None
        self._outcome: asyncio.Future[CalibrationOutcome] = asyncio.get_running_loop().create_future()

    async def start(self, sides) -> None:
        self.started_with = tuple(sides)
        if self.script is not None:
            await self.script(self)

    async def outcome(self) -> CalibrationOutcome:
        return await asyncio.shield(self._outcome)

    def finish(self, outcome: CalibrationOutcome) -> None:
        if not self._outcome.done():
            self._outcome.set_result(outcome)

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self.finish(_outcome(OutcomeKind.CANCELLED, "cancelled"))

    async def confirm(self, token: str) -> None:
        self.confirm_calls.append(token)
        if self.confirm_error is not None:
            raise self.confirm_error


class StubWorker:
    def __init__(self, script=None, unreleased: str | None = None) -> None:
        self.script = script
        self.unreleased = unreleased
        self.attempts: list[StubAttempt] = []

    def new_attempt(self, on_event) -> StubAttempt:
        attempt = StubAttempt(on_event, self.script)
        self.attempts.append(attempt)
        return attempt


def _outcome(kind: OutcomeKind, message: str = "", files: tuple[wp.ExportedFile, ...] = ()) -> CalibrationOutcome:
    return CalibrationOutcome(kind, ATTEMPT, (LEFT,), message, files=files)


def _files(*sides: wp.Side) -> tuple[wp.ExportedFile, ...]:
    return tuple(wp.ExportedFile(side, f"set_{side.value}.mcal", 10, "0" * 64) for side in sides)


def _gloves():
    return (wp.Glove(LEFT, 11), wp.Glove(RIGHT, 22))


def _prompt(token="left-0", *, side=LEFT, mode=wp.StepMode.HOLD, duration=0.0, continuous=False, estimated=0.0):
    return wp.Prompt(
        ATTEMPT,
        token,
        side,
        11 if side is LEFT else 22,
        0,
        8,
        "Make a fist",
        "Lift up your hand and make a fist.",
        mode,
        duration,
        continuous,
        estimated,
    )


def _session(worker: StubWorker, outbox: StubOutbox | None = None, sides=(p.HandSide.LEFT,)):
    outbox = outbox or StubOutbox()
    request = p.StartCalibrationRequest("cal-1", tuple(sides))
    return cal.CalibrationSession(request, worker, outbox, owner_ws=object()), outbox


async def _run_with(script, *, outcome: CalibrationOutcome, unreleased=None, sides=(p.HandSide.LEFT,)):
    async def scripted(attempt: StubAttempt) -> None:
        await script(attempt)
        attempt.finish(outcome)

    session, outbox = _session(StubWorker(scripted, unreleased), sides=sides)
    await session.run()
    return session, outbox


async def _no_events(_attempt) -> None:
    return None


async def test_run_publishes_started_and_an_initial_connecting_snapshot_first():
    session, outbox = await _run_with(_no_events, outcome=_outcome(OutcomeKind.CANCELLED, "cancelled"))
    kinds = [kind for kind, _ in outbox.log]
    assert kinds[0] == "bind"
    assert outbox.log[0][1] == ()
    assert outbox.log[1][1]["type"] == int(p.MessageType.CALIBRATION_STARTED)
    assert outbox.log[2][0] == "status"
    assert outbox.statuses()[0]["phase"] == int(p.CalibrationPhase.CONNECTING)


async def test_run_starts_the_attempt_with_the_requested_worker_sides():
    worker = StubWorker(None)
    session, outbox = _session(worker, sides=(p.HandSide.LEFT, p.HandSide.RIGHT))
    worker_attempt_ready = asyncio.ensure_future(session.run())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    worker.attempts[0].finish(_outcome(OutcomeKind.CANCELLED))
    _ = await worker_attempt_ready
    assert worker.attempts[0].started_with == (LEFT, RIGHT)


async def test_a_continuous_prompt_carries_its_estimate_distinct_from_the_timed_duration():
    async def script(attempt: StubAttempt) -> None:
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.AWAITING_STEP, _gloves()))
        attempt.on_event(_prompt(continuous=True, estimated=3.0))

    _, outbox = await _run_with(script, outcome=_outcome(OutcomeKind.CANCELLED))
    step = outbox.statuses()[-1]["step"]
    assert step["continuous"] is True
    assert step["estimated_seconds"] == 3.0
    assert step["duration_seconds"] == 0.0
    assert step["mode"] == int(p.CalibrationStepMode.HOLD)
    assert (step["title"], step["instruction"]) == ("Make a fist", "Lift up your hand and make a fist.")
    assert (step["index"], step["count"], step["token"]) == (0, 8, "left-0")


async def test_a_motion_prompt_keeps_its_timed_window_and_no_estimate():
    async def script(attempt: StubAttempt) -> None:
        attempt.on_event(_prompt("m-0", mode=wp.StepMode.MOTION, duration=2.0))

    _, outbox = await _run_with(script, outcome=_outcome(OutcomeKind.CANCELLED))
    step = outbox.statuses()[-1]["step"]
    assert step["mode"] == int(p.CalibrationStepMode.MOTION)
    assert (step["duration_seconds"], step["estimated_seconds"], step["continuous"]) == (2.0, 0.0, False)


async def test_an_awaiting_status_alone_publishes_no_snapshot():
    async def script(attempt: StubAttempt) -> None:
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.AWAITING_STEP, _gloves()))

    _, outbox = await _run_with(script, outcome=_outcome(OutcomeKind.CANCELLED))
    # Only the initial CONNECTING snapshot: the prompt that follows carries the step, so no prompt-less wait shows.
    assert [s["phase"] for s in outbox.statuses()] == [int(p.CalibrationPhase.CONNECTING)]


async def test_step_acceptance_is_published_in_worker_order_before_the_next_prompt_snapshot():
    async def script(attempt: StubAttempt) -> None:
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.AWAITING_STEP, _gloves()))
        attempt.on_event(_prompt("left-0", continuous=True, estimated=2.0))
        attempt.on_event(wp.StepAccepted(ATTEMPT, "left-0"))
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.AWAITING_STEP, _gloves()))
        attempt.on_event(_prompt("left-1", continuous=True, estimated=3.0))

    _, outbox = await _run_with(script, outcome=_outcome(OutcomeKind.CANCELLED))
    accepted_at = next(
        i
        for i, (kind, data) in enumerate(outbox.log)
        if kind == "control" and data["type"] == int(p.MessageType.CALIBRATION_STEP_ACCEPTED)
    )
    next_prompt_at = next(
        i
        for i, (kind, data) in enumerate(outbox.log)
        if kind == "status" and data["step"] and data["step"]["token"] == "left-1"
    )
    assert accepted_at < next_prompt_at
    assert outbox.log[accepted_at][1]["step_token"] == "left-0"


async def test_every_snapshot_is_a_complete_replacement():
    async def script(attempt: StubAttempt) -> None:
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.DISCOVERING, ()))
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.DISCOVERING, _gloves()))
        attempt.on_event(_prompt("m-0", mode=wp.StepMode.MOTION, duration=2.0))
        attempt.on_event(wp.StepAccepted(ATTEMPT, "m-0"))
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.CAPTURING_STEP, _gloves()))
        attempt.on_event(wp.StepProgress(ATTEMPT, "m-0", 0.5))
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.SAVING, _gloves()))

    _, outbox = await _run_with(script, outcome=_outcome(OutcomeKind.CANCELLED))
    keys = {"type", "version", "session_id", "phase", "gloves", "step", "step_elapsed_seconds"}
    assert all(set(status) == keys for status in outbox.statuses())
    capturing = [s for s in outbox.statuses() if s["phase"] == int(p.CalibrationPhase.CAPTURING_STEP)][-1]
    assert capturing["step"]["token"] == "m-0"
    progress = [s for s in outbox.statuses() if s["step_elapsed_seconds"] == 0.5][-1]
    assert progress["phase"] == int(p.CalibrationPhase.CAPTURING_STEP)
    saving = outbox.statuses()[-1]
    assert saving["phase"] == int(p.CalibrationPhase.SAVING)
    assert saving["step"] is None and saving["step_elapsed_seconds"] is None


async def test_gloves_are_reported_with_wire_sides():
    async def script(attempt: StubAttempt) -> None:
        attempt.on_event(wp.Status(ATTEMPT, wp.Phase.DISCOVERING, _gloves()))

    _, outbox = await _run_with(script, outcome=_outcome(OutcomeKind.CANCELLED))
    assert outbox.statuses()[-1]["gloves"] == [{"side": 1, "glove_id": 11}, {"side": 2, "glove_id": 22}]


async def test_the_terminal_message_is_published_last_and_only_after_quiesce():
    session, outbox = await _run_with(_no_events, outcome=_outcome(OutcomeKind.CANCELLED))
    names = [kind for kind, _ in outbox.log]
    assert names.index("quiesce") < len(names) - 1
    kind, last = outbox.log[-1]
    assert kind == "control" and last["type"] == int(p.MessageType.CALIBRATION_STOPPED)
    assert outbox.controls(p.MessageType.CALIBRATION_STOPPED) == [last]
    assert session.terminal is not None and session.terminal.to_json() == last
    assert session.done.is_set()


async def test_a_completed_outcome_offers_one_basename_per_published_side(tmp_path):
    published = _files(LEFT, RIGHT)
    session, _ = await _run_with(
        _no_events,
        outcome=_outcome(OutcomeKind.COMPLETED, "ok", published),
        sides=(p.HandSide.LEFT, p.HandSide.RIGHT),
    )
    terminal = session.terminal
    assert terminal.reason is p.CalibrationStopReason.COMPLETED
    assert terminal.error_message is None
    assert [(f.side, f.filename) for f in terminal.result.files] == [
        (p.HandSide.LEFT, "set_left.mcal"),
        (p.HandSide.RIGHT, "set_right.mcal"),
    ]


@pytest.mark.parametrize(
    "kind,reason",
    [
        (OutcomeKind.FAILED, p.CalibrationStopReason.FAILED),
        (OutcomeKind.UNCERTAIN, p.CalibrationStopReason.UNCERTAIN),
        (OutcomeKind.CANCELLED, p.CalibrationStopReason.CANCELLED),
    ],
)
async def test_files_are_never_offered_for_a_non_completed_outcome(tmp_path, kind, reason):
    session, _ = await _run_with(_no_events, outcome=_outcome(kind, "no", _files(LEFT)))
    assert session.terminal.reason is reason
    assert session.terminal.result is None


async def test_failed_and_uncertain_outcomes_carry_a_diagnostic_and_cancelled_does_not():
    failed, _ = await _run_with(_no_events, outcome=_outcome(OutcomeKind.FAILED, "the SDK refused"))
    uncertain, _ = await _run_with(
        _no_events, outcome=_outcome(OutcomeKind.UNCERTAIN, "no response; state unknown")
    )
    cancelled, _ = await _run_with(_no_events, outcome=_outcome(OutcomeKind.CANCELLED, "cancelled"))
    assert failed.terminal.error_message == "the SDK refused"
    assert uncertain.terminal.error_message == "no response; state unknown"
    assert cancelled.terminal.error_message is None


async def test_a_completed_outcome_without_published_files_degrades_instead_of_raising():
    session, _ = await _run_with(_no_events, outcome=_outcome(OutcomeKind.COMPLETED, "ok", None))
    assert session.terminal.reason is not p.CalibrationStopReason.COMPLETED
    assert session.terminal.result is None


async def test_cleanup_ok_follows_the_supervisors_unreleased_latch():
    ok, _ = await _run_with(_no_events, outcome=_outcome(OutcomeKind.CANCELLED), unreleased=None)
    stuck, _ = await _run_with(
        _no_events, outcome=_outcome(OutcomeKind.UNCERTAIN, "x"), unreleased="worker still running"
    )
    assert ok.cleanup_ok is True
    assert stuck.cleanup_ok is False


async def test_a_worker_that_cannot_be_admitted_raises_before_the_session_exists():
    class Busy:
        unreleased = "a worker may still hold the gloves"

        def new_attempt(self, on_event):
            raise AttemptError(self.unreleased)

    with pytest.raises(AttemptError):
        cal.CalibrationSession(
            p.StartCalibrationRequest("cal-1", (p.HandSide.LEFT,)), Busy(), StubOutbox(), object()
        )


async def test_request_stop_cancels_the_attempt_once():
    worker = StubWorker(None)
    session, outbox = _session(worker)
    run = asyncio.ensure_future(session.run())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    session.request_stop(p.StopReason.OWNER_DISCONNECTED)
    session.request_stop(p.StopReason.SERVER_SHUTDOWN)
    session.request_stop()
    _ = await run
    assert worker.attempts[0].cancel_calls == 1
    assert session.terminal.reason is p.CalibrationStopReason.CANCELLED


async def test_a_stop_before_the_worker_starts_is_settled_quietly():
    class AlreadyCancelled(StubAttempt):
        async def start(self, sides) -> None:
            raise AttemptError("an attempt can be started only once")

    worker = StubWorker(None)
    session, _ = _session(worker)
    attempt = AlreadyCancelled(session._on_event, None)
    session._attempt = attempt
    attempt.finish(_outcome(OutcomeKind.CANCELLED, "cancelled before start"))
    await session.run()
    assert session.terminal.reason is p.CalibrationStopReason.CANCELLED


async def test_the_connection_dying_cancels_the_attempt():
    worker = StubWorker(None)
    session, outbox = _session(worker)
    run = asyncio.ensure_future(session.run())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    outbox.on_fatal("control queue overflow")
    _ = await run
    assert worker.attempts[0].cancel_calls == 1


async def test_confirming_an_accepted_token_acknowledges_again_without_a_second_step():
    worker = StubWorker(None)
    session, outbox = _session(worker)
    session._accepted.add("left-0")
    await session.handle_confirm("left-0")
    assert worker.attempts[0].confirm_calls == []
    assert outbox.controls(p.MessageType.CALIBRATION_STEP_ACCEPTED)[0]["step_token"] == "left-0"


async def test_confirming_a_token_that_is_not_offered_is_refused_without_reaching_the_worker():
    worker = StubWorker(None)
    session, _ = _session(worker)
    with pytest.raises(cal.ConfirmRefused):
        await session.handle_confirm("never-offered")
    worker.attempts[0].prompt = _prompt("left-0")
    with pytest.raises(cal.ConfirmRefused):
        await session.handle_confirm("left-1")
    assert worker.attempts[0].confirm_calls == []


async def test_confirming_the_offered_token_reaches_the_worker_once():
    worker = StubWorker(None)
    session, _ = _session(worker)
    worker.attempts[0].prompt = _prompt("left-0")
    await session.handle_confirm("left-0")
    assert worker.attempts[0].confirm_calls == ["left-0"]


async def test_a_worker_side_confirm_refusal_becomes_confirm_refused_with_its_message():
    worker = StubWorker(None)
    session, _ = _session(worker)
    worker.attempts[0].prompt = _prompt("left-0")
    worker.attempts[0].confirm_error = AttemptError("the attempt is not running")
    with pytest.raises(cal.ConfirmRefused, match="not running"):
        await session.handle_confirm("left-0")


async def test_spawn_confirm_reports_a_refusal_through_the_callback_without_raising():
    worker = StubWorker(None)
    session, _ = _session(worker)
    refusals: list[str] = []
    session.spawn_confirm("nope", refusals.append)
    await asyncio.sleep(0.01)
    assert refusals and "not awaiting confirmation" in refusals[0]


async def test_replay_resends_started_and_the_current_snapshot():
    worker = StubWorker(None)
    session, outbox = _session(worker)
    session.replay()
    assert outbox.controls(p.MessageType.CALIBRATION_STARTED)
    assert outbox.statuses()
    assert outbox.controls(p.MessageType.CALIBRATION_STARTED)[0]["session_id"] == "cal-1"


async def test_a_missing_outcome_is_reported_as_uncertain():
    session, _ = _session(StubWorker(None))
    terminal = session._terminal_of(None)
    assert terminal.reason is p.CalibrationStopReason.UNCERTAIN
    assert terminal.result is None
