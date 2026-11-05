# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The worker adapter against the scripted fake worker, through the real subprocess supervisor."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import pytest
from conftest import BOTH, Folder, commands, drive, wait_for
from isaaccapture_examples.manus_calibration import protocol as proto
from isaaccapture_examples.manus_calibration.worker import (
    AttemptEnded,
    AttemptError,
    AttemptState,
    CalibrationWorker,
    ConfirmError,
    OutcomeKind,
)

# Hold-only steps keep scenarios fast; motion tests override them.
HOLD = {
    "steps": [
        {"title": "A", "instruction": "a", "time": 0.0},
        {"title": "B", "instruction": "b", "time": 0.0},
    ]
}
LEFT = (proto.Side.LEFT,)
RIGHT = (proto.Side.RIGHT,)
SETTLE_S = 20.0


class Run:
    """One attempt plus the events its listener recorded. Create inside a running loop."""

    def __init__(self, make_worker, scenario: dict | None = None, **settings) -> None:
        self.worker, self.store = make_worker({**HOLD, **(scenario or {})}, **settings)
        self.events: list[proto.WorkerEvent] = []
        self.attempt = self.worker.new_attempt(self.events.append)

    def seen(self, kind: type) -> bool:
        return any(isinstance(e, kind) for e in self.events)

    async def settle(self):
        async with asyncio.timeout(SETTLE_S):
            return await self.attempt.outcome()


def confirming(attempt) -> asyncio.Task:
    """Confirm every prompt as it appears, until the attempt ends."""

    async def loop() -> None:
        done: set[str] = set()
        while attempt.state is not AttemptState.DONE:
            prompt = attempt.prompt
            if prompt is not None and prompt.token not in done:
                done.add(prompt.token)
                try:
                    await attempt.confirm(prompt.token)
                except AttemptError:
                    # The attempt may end or refuse mid-confirm; the driver only keeps answering prompts.
                    pass
            await asyncio.sleep(0.01)

    return asyncio.create_task(loop())


def assert_nothing_saved(store) -> None:
    """The folder holds no finished file and no half-written one."""
    assert store.saved_attempts() == []
    assert store.leftovers() == []


def assert_reaped(attempt) -> None:
    # The supervisor's process handle is the only record of the exit status.
    proc = attempt._proc
    assert proc is not None and proc.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(proc.pid, 0)


def run(coro):
    return asyncio.run(coro)


# --- Happy paths -----------------------------------------------------------------------------


def test_both_sides_complete_in_order_and_save_into_the_folder(make_worker):
    async def scenario():
        r = Run(make_worker)
        outcome = await drive(r.attempt, BOTH)
        assert outcome.kind is OutcomeKind.COMPLETED
        assert outcome.exit_code == 0 and not outcome.forced_termination
        assert [f.side for f in outcome.files] == [proto.Side.LEFT, proto.Side.RIGHT]
        assert r.store.saved_attempts() == [outcome.attempt]
        for f in outcome.files:
            data = (r.store.directory / f.file).read_bytes()
            assert data == (b"MCAL-" + f.side.value.encode()) * 64
        assert r.store.leftovers() == []
        assert_reaped(r.attempt)

    run(scenario())


@pytest.mark.parametrize("sides", [LEFT, RIGHT], ids=["left", "right"])
def test_a_single_side_completes_with_only_that_file(make_worker, sides):
    async def scenario():
        r = Run(make_worker)
        outcome = await drive(r.attempt, sides)
        assert outcome.kind is OutcomeKind.COMPLETED
        assert [f.side for f in outcome.files] == list(sides)
        assert len(list(r.store.directory.glob("*.mcal"))) == 1

    run(scenario())


def test_events_arrive_in_contract_order_with_exactly_one_result(make_worker):
    async def scenario():
        r = Run(make_worker)
        outcome = await drive(r.attempt, BOTH)
        assert outcome.kind is OutcomeKind.COMPLETED
        events = r.events
        assert isinstance(events[0], proto.Ready)
        assert isinstance(events[-1], proto.Result)
        assert sum(isinstance(e, proto.Result) for e in events) == 1
        prompts = [e.token for e in events if isinstance(e, proto.Prompt)]
        assert prompts == ["left-0", "left-1", "right-0", "right-1"]
        exported = [e.side for e in events if isinstance(e, proto.Exported)]
        assert exported == [proto.Side.LEFT, proto.Side.RIGHT]
        result_at = len(events) - 1
        assert all(
            events.index(e) < result_at for e in events if isinstance(e, proto.Exported)
        )
        # Exactly one terminal outcome: asking again returns the same object.
        assert await r.attempt.outcome() is outcome
        assert r.attempt.state is AttemptState.DONE and r.attempt.prompt is None

    run(scenario())


def test_a_listener_that_raises_does_not_stop_supervision(make_worker):
    async def scenario():
        r = Run(make_worker)
        r.attempt._on_event = lambda event: 1 / 0
        outcome = await drive(r.attempt, LEFT)
        assert outcome.kind is OutcomeKind.COMPLETED

    run(scenario())


def test_motion_steps_report_progress_within_the_step_duration(make_worker):
    steps = {"steps": [{"title": "Wave", "instruction": "w", "time": 0.3}]}

    async def scenario():
        r = Run(make_worker, steps)
        outcome = await drive(r.attempt, LEFT)
        assert outcome.kind is OutcomeKind.COMPLETED
        progress = [e.elapsed_s for e in r.events if isinstance(e, proto.StepProgress)]
        assert progress and progress == sorted(progress) and progress[-1] <= 0.3 + 1e-6
        prompt = next(e for e in r.events if isinstance(e, proto.Prompt))
        assert (
            prompt.mode is proto.StepMode.MOTION
            and prompt.duration_s == pytest.approx(0.3)
        )

    run(scenario())


# --- Confirmation semantics --------------------------------------------------------------------


def test_every_prompt_needs_its_own_confirmation(make_worker, commands_log):
    async def scenario():
        r = Run(make_worker)
        a = r.attempt
        await a.start(BOTH)
        await wait_for(lambda: a.prompt is not None)
        for token, expected_confirms in (("left-0", 1), ("left-1", 2), ("right-0", 3)):
            assert a.prompt.token == token
            # Nothing advances without the operator: the same prompt stays offered.
            await asyncio.sleep(0.25)
            assert a.prompt.token == token and a.state is AttemptState.RUNNING
            assert commands(commands_log).count("confirm") == expected_confirms - 1
            await a.confirm(token)
            await wait_for(lambda token=token: a.prompt.token != token)
        # The first prompt of the second glove was reached only through its own confirmation.
        assert a.prompt.token == "right-1"
        await asyncio.sleep(0.25)
        assert a.state is AttemptState.RUNNING
        await a.confirm("right-1")
        assert (await r.settle()).kind is OutcomeKind.COMPLETED
        assert commands(commands_log).count("confirm") == 4

    run(scenario())


def test_duplicate_confirmation_reaches_the_worker_once(make_worker, commands_log):
    async def scenario():
        r = Run(make_worker)
        a = r.attempt
        await a.start(LEFT)
        await wait_for(lambda: a.prompt is not None)
        await asyncio.gather(*(a.confirm("left-0") for _ in range(3)))
        await a.confirm("left-0")  # after acceptance
        await wait_for(lambda: a.prompt.token == "left-1")
        await a.confirm("left-0")  # after the next prompt is offered
        assert commands(commands_log).count("confirm") == 1
        await a.confirm("left-1")
        await a.confirm("left-1")
        assert (await r.settle()).kind is OutcomeKind.COMPLETED
        assert commands(commands_log).count("confirm") == 2

    run(scenario())


def test_stale_and_unknown_tokens_never_reach_the_worker(make_worker, commands_log):
    async def scenario():
        r = Run(make_worker)
        a = r.attempt
        await a.start(BOTH)
        await wait_for(lambda: a.prompt is not None)
        for token in ("right-0", "left-1", "nonsense", ""):
            with pytest.raises(ConfirmError):
                await a.confirm(token)
        assert commands(commands_log).count("confirm") == 0
        assert a.prompt.token == "left-0"
        confirming(a)
        assert (await r.settle()).kind is OutcomeKind.COMPLETED

    run(scenario())


def test_confirm_with_no_prompt_pending_is_refused(make_worker):
    async def scenario():
        r = Run(make_worker, {"discover_delay": 30})
        a = r.attempt
        with pytest.raises(AttemptEnded):  # not started
            await a.confirm("left-0")
        await a.start(LEFT)
        await wait_for(lambda: r.seen(proto.Status))
        with pytest.raises(ConfirmError):  # discovering, nothing offered
            await a.confirm("left-0")
        await a.cancel()
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.CANCELLED
        with pytest.raises(AttemptEnded):  # finished
            await a.confirm("left-0")

    run(scenario())


def test_a_worker_refusal_surfaces_and_the_prompt_can_be_confirmed_again(
    make_worker, commands_log
):
    async def scenario():
        r = Run(make_worker, {"reject_first_confirm": True})
        a = r.attempt
        await a.start(LEFT)
        await wait_for(lambda: a.prompt is not None)
        with pytest.raises(ConfirmError, match="unexpected"):
            await a.confirm("left-0")
        assert r.seen(proto.Rejected)
        await a.confirm("left-0")
        confirming(a)
        assert (await r.settle()).kind is OutcomeKind.COMPLETED
        assert (
            commands(commands_log).count("confirm") == 3
        )  # refused, retried, second prompt

    run(scenario())


# --- One attempt ------------------------------------------------------------------------------


def test_an_attempt_starts_once_and_only_one_is_active_per_worker(make_worker):
    async def scenario():
        r = Run(make_worker)
        a = r.attempt
        await a.start(LEFT)
        with pytest.raises(AttemptError, match="only once"):
            await a.start(LEFT)
        with pytest.raises(AttemptError, match="already active"):
            r.worker.new_attempt()
        await a.cancel()
        assert (await r.settle()).kind is OutcomeKind.CANCELLED
        again = r.worker.new_attempt()
        assert again is not a and again.state is AttemptState.NEW
        await again.cancel()

    run(scenario())


def test_an_unusable_calibration_directory_raises_and_ends_the_attempt_failed(
    make_worker, tmp_path, commands_log
):
    async def scenario():
        r = Run(make_worker)
        (tmp_path / "calibrations").write_text("a file where the directory should be")
        with pytest.raises(AttemptError):
            await r.attempt.start(LEFT)
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.FAILED
        assert r.attempt.state is AttemptState.DONE and r.attempt._proc is None
        assert commands(commands_log) == []
        assert r.worker.new_attempt().state is AttemptState.NEW

    run(scenario())


def test_invalid_sides_are_rejected_before_anything_is_launched(
    make_worker, commands_log
):
    async def scenario():
        r = Run(make_worker)
        for sides in ((), (proto.Side.LEFT, proto.Side.LEFT)):
            with pytest.raises(proto.ProtocolError):
                await r.attempt.start(sides)
        assert r.attempt.state is AttemptState.NEW
        assert commands(commands_log) == []

    run(scenario())


# --- Protocol violations --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_keys",
    [
        {"malformed_line": True},
        {"oversized": True},
        {"unknown_event": True},
        {"wrong_attempt": True},
    ],
    ids=["malformed", "oversized", "unknown-type", "wrong-attempt"],
)
def test_a_bad_worker_line_is_a_protocol_failure_never_a_completion(
    make_worker, scenario_keys
):
    async def scenario():
        r = Run(make_worker, scenario_keys)
        await r.attempt.start(LEFT)
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.error is not None and outcome.error.code == "protocol_error"
        assert outcome.files == ()
        assert_nothing_saved(r.store)
        assert_reaped(r.attempt)

    run(scenario())


def test_garbage_before_ready_fails_without_ever_starting_the_worker(
    make_worker, commands_log
):
    async def scenario():
        r = Run(make_worker, {"garbage_ready": True})
        await r.attempt.start(LEFT)
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.FAILED
        assert outcome.error.code == "protocol_error"
        assert commands(commands_log) == []
        assert_reaped(r.attempt)

    run(scenario())


def test_a_message_after_the_result_voids_the_completion(make_worker):
    async def scenario():
        r = Run(make_worker, {"extra_after_result": True})
        outcome = await drive(r.attempt, BOTH)
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.files == ()
        assert_nothing_saved(r.store)

    run(scenario())


def test_events_the_supervisor_cannot_justify_are_protocol_failures(make_worker):
    cases = {
        "spurious_accept": {"spurious_accept": True},
        "unrequested_side": {"unrequested_side_prompt": True},
        "progress_overrun": {
            "progress_overrun": True,
            "steps": [{"title": "M", "instruction": "m", "time": 0.3}],
        },
    }

    async def scenario(keys):
        r = Run(make_worker, keys)
        await r.attempt.start(LEFT)
        if "progress_overrun" in keys:
            await wait_for(lambda: r.attempt.prompt is not None)
            await r.attempt.confirm("left-0")
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.UNCERTAIN, keys
        assert outcome.error.code == "protocol_error", keys
        assert_nothing_saved(r.store)

    for keys in cases.values():
        run(scenario(keys))


def test_sides_are_processed_left_then_right_whatever_order_they_are_requested(
    make_worker,
):
    async def scenario():
        r = Run(make_worker)
        outcome = await drive(r.attempt, (proto.Side.RIGHT, proto.Side.LEFT))
        assert outcome.kind is OutcomeKind.COMPLETED
        assert (
            next(e.token for e in r.events if isinstance(e, proto.Prompt)) == "left-0"
        )
        assert [f.side for f in outcome.files] == [
            proto.Side.LEFT,
            proto.Side.RIGHT,
        ]

    run(scenario())


# --- Worker death and missing evidence --------------------------------------------------------------


def test_a_worker_crash_mid_step_is_uncertain(make_worker):
    async def scenario():
        r = Run(make_worker, {"crash_after_prompt": True})
        await r.attempt.start(BOTH)
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.exit_code == 9
        assert outcome.files == ()
        assert r.attempt.prompt is None
        assert_nothing_saved(r.store)

    run(scenario())


def test_exiting_without_a_result_is_never_a_completion(make_worker):
    async def scenario():
        r = Run(make_worker, {"no_result": True})
        outcome = await drive(r.attempt, BOTH)
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.exit_code == 0
        assert_nothing_saved(r.store)

    run(scenario())


def test_a_completed_result_with_a_nonzero_exit_is_not_trusted(make_worker):
    async def scenario():
        r = Run(make_worker, {"exit_code_after_result": 3})
        outcome = await drive(r.attempt, BOTH)
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.exit_code == 3
        assert_nothing_saved(r.store)

    run(scenario())


@pytest.mark.parametrize(
    "scenario_keys",
    [{"result_mismatch": True}, {"partial_result": True}, {"no_export_events": True}],
    ids=["size-differs", "one-side-missing", "no-exported-events"],
)
def test_a_result_that_disagrees_with_the_exports_publishes_nothing(
    make_worker, scenario_keys
):
    async def scenario():
        r = Run(make_worker, scenario_keys)
        outcome = await drive(r.attempt, BOTH)
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.error.code == "protocol_error"
        assert_nothing_saved(r.store)

    run(scenario())


@pytest.mark.parametrize(
    ("scenario_keys", "code"),
    [
        ({"fail_at_step": ["right", 1]}, "step_failed"),
        ({"fail_after_first_export": True}, "export_failed"),
    ],
    ids=["step-failure", "second-glove-export-failure"],
)
def test_a_failed_result_never_publishes_a_partial_pair(
    make_worker, scenario_keys, code
):
    async def scenario():
        r = Run(make_worker, scenario_keys)
        outcome = await drive(r.attempt, BOTH)
        assert outcome.kind is OutcomeKind.FAILED
        assert outcome.error.code == code
        assert outcome.exit_code == 1 and not outcome.forced_termination
        assert_nothing_saved(r.store)

    run(scenario())


def test_a_step_failure_names_the_glove_and_step(make_worker):
    async def scenario():
        r = Run(make_worker, {"fail_at_step": ["right", 1]})
        outcome = await drive(r.attempt, BOTH)
        assert (outcome.error.side, outcome.error.step) == (proto.Side.RIGHT, 1)

    run(scenario())


# --- Cancellation ---------------------------------------------------------------------------------


def _cancel_case(name: str):
    motion = {"steps": [{"title": "M", "instruction": "m", "time": 30.0}]}
    if name == "discovery":
        return {"discover_delay": 30}, lambda r: r.seen(proto.Status), False
    if name == "awaiting":
        return {}, lambda r: r.attempt.prompt is not None, False
    if name == "sampling":
        return motion, lambda r: r.seen(proto.StepProgress), True
    return (
        {"save_delay": 30},
        lambda r: any(
            isinstance(e, proto.Status) and e.phase is proto.Phase.SAVING
            for e in r.events
        ),
        True,
    )


@pytest.mark.parametrize("phase", ["discovery", "awaiting", "sampling", "saving"])
def test_cancel_in_every_phase_ends_cancelled_and_publishes_nothing(
    make_worker, commands_log, phase
):
    scenario_keys, reached, needs_confirms = _cancel_case(phase)

    async def scenario():
        r = Run(make_worker, scenario_keys)
        await r.attempt.start(BOTH)
        if needs_confirms:
            confirming(r.attempt)
        await wait_for(lambda: reached(r))
        await r.attempt.cancel()
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.CANCELLED
        assert outcome.exit_code == 0 and not outcome.forced_termination
        assert "cancel" in commands(commands_log)
        assert sum(isinstance(e, proto.Result) for e in r.events) == 1
        assert_nothing_saved(r.store)
        assert_reaped(r.attempt)

    run(scenario())


def test_cancel_is_idempotent_and_a_late_cancel_changes_nothing(
    make_worker, commands_log
):
    async def scenario():
        r = Run(make_worker, {"discover_delay": 30})
        await r.attempt.start(LEFT)
        await wait_for(lambda: r.seen(proto.Status))
        await asyncio.gather(r.attempt.cancel(), r.attempt.cancel())
        await r.attempt.cancel()
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.CANCELLED
        assert commands(commands_log).count("cancel") == 1
        await r.attempt.cancel()
        assert await r.attempt.outcome() is outcome

    run(scenario())


def test_a_late_cancel_does_not_undo_a_completed_attempt(make_worker):
    async def scenario():
        r = Run(make_worker)
        outcome = await drive(r.attempt, LEFT)
        await r.attempt.cancel()
        assert await r.attempt.outcome() is outcome
        assert (
            outcome.kind is OutcomeKind.COMPLETED and len(r.store.saved_attempts()) == 1
        )

    run(scenario())


def test_cancel_while_waiting_for_ready_is_cancelled_and_the_worker_reaped(
    make_worker, commands_log
):
    async def scenario():
        r = Run(make_worker, {"never_ready": True}, ready_timeout_s=30, term_grace_s=1)
        starting = asyncio.create_task(r.attempt.start(LEFT))
        await wait_for(
            lambda: (
                r.attempt.state is AttemptState.STARTING and r.attempt._proc is not None
            )
        )
        await r.attempt.cancel()
        _ = await starting
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.CANCELLED
        assert commands(commands_log) == []
        assert_reaped(r.attempt)

    run(scenario())


def test_an_idle_workers_cancelled_result_is_accepted_only_after_a_user_cancel(
    make_worker,
):
    result = proto.Result(attempt=None, outcome=proto.Outcome.CANCELLED)

    async def scenario():
        # The window is between the worker's ready and our start command; drive the state directly.
        r = Run(make_worker)
        r.attempt._ready.set()
        with pytest.raises(proto.ProtocolError):
            r.attempt._handle(result)
        r.attempt._user_cancel = True
        r.attempt._handle(result)
        assert r.attempt._result is result
        other = proto.Result(attempt=None, outcome=proto.Outcome.FAILED)
        with pytest.raises(proto.ProtocolError):
            r.attempt._handle(other)

    run(scenario())


def test_cancel_before_start_with_a_worker_that_reports_its_idle_cancel(make_worker):
    async def scenario():
        r = Run(make_worker, {"idle_cancel_result": True, "never_ready": True})
        starting = asyncio.create_task(r.attempt.start(LEFT))
        await wait_for(lambda: r.attempt._proc is not None)
        await r.attempt.cancel()
        _ = await asyncio.gather(starting, return_exceptions=True)
        outcome = await r.settle()
        # Cancelled while waiting for ready: the worker's SIGTERM result must not read as a protocol error.
        assert outcome.kind in (OutcomeKind.CANCELLED, OutcomeKind.FAILED)
        assert "broke the protocol" not in outcome.message
        assert_reaped(r.attempt)

    run(scenario())


def test_cancel_before_start_never_launches_a_worker(make_worker, commands_log):
    async def scenario():
        r = Run(make_worker)
        await r.attempt.cancel()
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.CANCELLED
        with pytest.raises(AttemptError):
            await r.attempt.start(LEFT)
        assert commands(commands_log) == [] and r.attempt._proc is None

    run(scenario())


@pytest.mark.parametrize(
    ("scenario_keys", "expected_signal"),
    [
        ({"ignore_cancel": True}, signal.SIGTERM),
        ({"ignore_cancel": True, "ignore_term": True}, signal.SIGKILL),
    ],
    ids=["ignores-cancel", "ignores-cancel-and-term"],
)
def test_a_worker_that_will_not_stop_is_terminated_reaped_and_not_trusted(
    make_worker, scenario_keys, expected_signal
):
    async def scenario():
        r = Run(
            make_worker,
            scenario_keys,
            cancel_grace_s=0.5,
            term_grace_s=0.5,
            kill_wait_s=3,
        )
        await r.attempt.start(LEFT)
        await wait_for(lambda: r.attempt.prompt is not None)
        started = time.monotonic()
        await r.attempt.cancel()
        outcome = await r.settle()
        assert time.monotonic() - started < 8
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.forced_termination
        assert outcome.exit_code == -int(expected_signal)
        assert_reaped(r.attempt)
        assert_nothing_saved(r.store)

    run(scenario())


def test_a_worker_that_hangs_after_its_result_is_forced_and_not_trusted(make_worker):
    async def scenario():
        r = Run(
            make_worker, {"hang_after_result": True}, exit_grace_s=0.5, term_grace_s=0.5
        )
        outcome = await drive(r.attempt, BOTH)
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.forced_termination
        assert_reaped(r.attempt)
        assert_nothing_saved(r.store)

    run(scenario())


def test_a_forced_exit_code_of_zero_does_not_make_a_completion_trustworthy(make_worker):
    async def scenario():
        keys = {"hang_after_result": True, "term_exits_zero": True}
        r = Run(make_worker, keys, exit_grace_s=0.5, term_grace_s=1)
        outcome = await drive(r.attempt, BOTH)
        assert outcome.exit_code == 0 and outcome.forced_termination
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert_nothing_saved(r.store)

    run(scenario())


# --- Worker that never starts or never answers ------------------------------------------------------


def test_a_worker_that_never_reports_ready_fails_and_is_reaped(
    make_worker, commands_log
):
    async def scenario():
        r = Run(make_worker, {"never_ready": True}, ready_timeout_s=0.5, term_grace_s=1)
        await r.attempt.start(LEFT)
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.FAILED
        assert "ready" in outcome.message
        assert commands(commands_log) == []
        assert_reaped(r.attempt)

    run(scenario())


def test_a_worker_that_exits_before_ready_fails_with_its_exit_code(make_worker):
    async def scenario():
        r = Run(make_worker, {"exit_before_ready": True})
        await r.attempt.start(LEFT)
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.FAILED
        assert outcome.exit_code == 3 and "3" in outcome.message

    run(scenario())


def test_a_relative_worker_path_is_spawned_by_path_not_looked_up_on_PATH(
    tmp_path, monkeypatch
):
    script = tmp_path / "relative-worker"
    script.write_text("#!/bin/sh\nexit 3\n")
    script.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "/nonexistent")

    async def scenario():
        attempt = CalibrationWorker(
            Path("./relative-worker"), tmp_path / "calibrations"
        ).new_attempt()
        await attempt.start(LEFT)
        return await attempt.outcome()

    outcome = run(scenario())
    # It ran (exited 3 before ready) instead of failing to spawn.
    assert outcome.error.code != "spawn_failed" and outcome.exit_code == 3


def test_a_rejected_start_fails_with_the_workers_reason(make_worker):
    async def scenario():
        r = Run(make_worker, {"reject_start": True}, ready_timeout_s=30)
        await r.attempt.start(LEFT)
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.FAILED
        assert "staging_dir is not a writable directory" in outcome.message
        assert outcome.error.code == "start_rejected"
        assert_reaped(r.attempt)

    run(scenario())


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a read-only directory")
def test_an_unwritable_output_folder_fails_before_the_worker_is_launched(
    make_worker, commands_log
):
    async def scenario():
        worker, folder = make_worker()
        folder.directory.mkdir(parents=True)
        folder.directory.chmod(0o500)
        try:
            attempt = worker.new_attempt()
            with pytest.raises(AttemptError):
                await attempt.start(LEFT)
            outcome = await attempt.outcome()
            assert outcome.kind is OutcomeKind.FAILED
            assert "not a writable directory" in outcome.message
            assert attempt._proc is None
        finally:
            folder.directory.chmod(0o700)

    run(scenario())


def test_a_missing_or_non_executable_worker_fails_cleanly(tmp_path):
    plain = tmp_path / "not-executable"
    plain.write_text("#!/bin/sh\n")
    plain.chmod(0o644)

    async def scenario(executable: Path):
        store = Folder(tmp_path / "calibrations")
        attempt = CalibrationWorker(executable, store.directory).new_attempt()
        await attempt.start(LEFT)
        outcome = await attempt.outcome()
        assert outcome.kind is OutcomeKind.FAILED
        assert outcome.error.code == "spawn_failed"
        assert attempt.state is AttemptState.DONE
        assert_nothing_saved(store)

    run(scenario(tmp_path / "does-not-exist"))
    run(scenario(plain))


# --- Worker output ------------------------------------------------------------------------------------


def test_worker_output_never_corrupts_the_ipc_stream_and_the_tail_is_bounded(
    make_worker,
):
    async def scenario():
        r = Run(make_worker, {"noise_bytes": 300_000}, log_tail_lines=10)
        outcome = await drive(r.attempt, LEFT)
        # A decoy IPC-looking line on stdout was not parsed, so the real result stands.
        assert outcome.kind is OutcomeKind.COMPLETED
        assert len(outcome.log_tail) <= 10
        assert all(len(line) <= len("stdout: ") + 1000 for line in outcome.log_tail)

    run(scenario())


def test_the_log_tail_keeps_stdout_and_stderr_lines(make_worker):
    async def scenario():
        r = Run(make_worker)
        outcome = await drive(r.attempt, LEFT)
        assert "stdout: fake worker log line on stdout" in outcome.log_tail
        assert "stderr: fake worker log line on stderr" in outcome.log_tail

    run(scenario())


# --- Files in the folder ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["../escape.mcal", "/etc/escape.mcal", "sub/dir.mcal", ".hidden.mcal", "a\\b.mcal"],
)
def test_a_traversing_export_name_is_a_protocol_violation(make_worker, tmp_path, name):
    async def scenario():
        r = Run(make_worker, {"reported_name": name, "name_side": "left"})
        outcome = await drive(r.attempt, LEFT)
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.error.code == "protocol_error"
        assert_nothing_saved(r.store)

    run(scenario())
    assert not (tmp_path / "escape.mcal").exists()
    assert not (tmp_path / "calibrations" / "escape.mcal").exists()


def test_new_attempts_leave_earlier_files_alone_and_a_failed_one_leaves_none(
    make_worker,
):
    async def scenario():
        r = Run(make_worker)
        folder = r.store.directory
        folder.mkdir(parents=True, exist_ok=True)
        foreign = folder / "legacy.mcal"
        foreign.write_bytes(b"keep me")
        first = await drive(r.attempt, BOTH)
        assert first.kind is OutcomeKind.COMPLETED
        snapshot = {
            p: (p.read_bytes(), p.stat().st_ino)
            for p in [foreign, *(folder / f.file for f in first.files)]
        }

        second = await drive(Run(make_worker).attempt, BOTH)
        assert second.kind is OutcomeKind.COMPLETED
        assert second.attempt != first.attempt
        failed = Run(make_worker, {"fail_after_first_export": True})
        assert (await drive(failed.attempt, BOTH)).kind is OutcomeKind.FAILED

        # The failed attempt's own first file is gone; every earlier file is exactly as it was.
        for path, (data, inode) in snapshot.items():
            assert path.read_bytes() == data and path.stat().st_ino == inode
        assert sorted(r.store.saved_attempts()) == sorted(
            ["legacy.mcal".split("_")[0], first.attempt, second.attempt]
        )
        assert r.store.leftovers() == []

    run(scenario())


# --- Release evidence -------------------------------------------------------------------------


def assert_unreleased(outcome, r) -> None:
    """The shared contract of a result that reports the SDK not released."""
    assert outcome.kind is OutcomeKind.UNCERTAIN
    assert outcome.error is not None and outcome.error.code == "worker_unreleased"
    assert "could not release the SDK" in outcome.message
    assert outcome.files == ()
    assert r.worker.unreleased == outcome.message
    assert_nothing_saved(r.store)
    assert_reaped(r.attempt)
    with pytest.raises(AttemptError, match="could not release the SDK"):
        r.worker.new_attempt()


def test_a_completed_result_that_reports_the_sdk_unreleased_is_uncertain_and_publishes_nothing(
    make_worker,
):
    async def scenario():
        r = Run(make_worker, {"not_released": True})
        outcome = await drive(r.attempt, BOTH)
        # The files the worker reported would verify, so only the release flag stops publication.
        assert r.seen(proto.Result) and r.seen(proto.Exported)
        assert_unreleased(outcome, r)

    run(scenario())


def test_a_cancelled_result_that_reports_the_sdk_unreleased_is_uncertain(make_worker):
    async def scenario():
        r = Run(make_worker, {"not_released": True})
        await r.attempt.start(LEFT)
        await wait_for(lambda: r.attempt.prompt is not None)
        await r.attempt.cancel()
        outcome = await r.settle()
        assert_unreleased(outcome, r)

    run(scenario())


def test_a_failed_result_that_reports_the_sdk_unreleased_is_uncertain_and_names_the_failure(
    make_worker,
):
    async def scenario():
        r = Run(make_worker, {"not_released": True, "fail_at_step": ["left", 1]})
        outcome = await drive(r.attempt, LEFT)
        assert_unreleased(outcome, r)
        # The worker's own failure text is kept in the message for the operator.
        assert "the SDK refused the step" in outcome.message

    run(scenario())


def test_a_released_result_leaves_the_worker_free_for_the_next_attempt(make_worker):
    async def scenario():
        r = Run(make_worker)
        outcome = await drive(r.attempt, LEFT)
        assert outcome.kind is OutcomeKind.COMPLETED
        assert r.worker.unreleased is None
        assert r.worker.new_attempt().state is AttemptState.NEW

    run(scenario())


def test_a_failure_that_released_the_sdk_does_not_latch_the_worker(make_worker):
    async def scenario():
        r = Run(make_worker, {"fail_at_step": ["left", 1]})
        outcome = await drive(r.attempt, LEFT)
        assert outcome.kind is OutcomeKind.FAILED
        assert r.worker.unreleased is None
        assert r.worker.new_attempt().state is AttemptState.NEW

    run(scenario())
