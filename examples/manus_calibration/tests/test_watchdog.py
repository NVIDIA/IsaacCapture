# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parent-side watchdog: a silent worker is stopped, an operator who thinks is never rushed."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import pytest
from conftest import BOTH, commands, wait_for
from isaaccapture_examples.manus_calibration import protocol as proto
from isaaccapture_examples.manus_calibration.store import McalStore
from isaaccapture_examples.manus_calibration.worker import (
    AttemptEnded,
    AttemptState,
    ConfirmError,
    OutcomeKind,
)

LEFT = (proto.Side.LEFT,)
# One hold step per glove: a single prompt each, so a scenario reaches its hang point quickly.
ONE_STEP = {"steps": [{"title": "A", "instruction": "a", "time": 0.0}]}
# Short deadlines keep the suite fast; the supervisor polls its deadline every 0.25 s.
DEADLINE_S = 1.0
FAST = {
    "startup_timeout_s": DEADLINE_S,
    "confirm_timeout_s": DEADLINE_S,
    "finish_timeout_s": DEADLINE_S,
    "term_grace_s": 1,
    "kill_wait_s": 2,
}
SETTLE_S = 20.0
# Deadline, one poll interval, the SIGTERM grace, the kill wait, and slack for a slow machine.
BOUND_S = DEADLINE_S + 0.25 + 1 + 2 + 4


class Run:
    """One attempt plus the events its listener recorded. Create inside a running loop."""

    def __init__(self, make_worker, scenario: dict | None = None, **settings) -> None:
        self.worker, self.store = make_worker(
            {**ONE_STEP, **(scenario or {})}, **{**FAST, **settings}
        )
        self.events: list[proto.WorkerEvent] = []
        self.attempt = self.worker.new_attempt(self.events.append)

    def seen(self, kind: type) -> bool:
        return any(isinstance(e, kind) for e in self.events)

    async def settle(self):
        async with asyncio.timeout(SETTLE_S):
            return await self.attempt.outcome()


def run(coro):
    return asyncio.run(coro)


def published_files(store: McalStore) -> list[Path]:
    """Every MCAL file or manifest visible in the calibration directory, staging excluded."""
    found = list(store.directory.glob("*.mcal"))
    manifests = store.directory / "manifests"
    if manifests.is_dir():
        found += list(manifests.iterdir())
    return found


def assert_nothing_published(store: McalStore) -> None:
    assert store.list_published() == []
    assert published_files(store) == []


def assert_staging_gone(store: McalStore, attempt_id: str) -> None:
    staging = store.directory / ".staging"
    assert not (staging / attempt_id).exists()
    assert not staging.exists() or list(staging.iterdir()) == []


def assert_reaped(attempt) -> None:
    proc = attempt._proc
    assert proc is not None and proc.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(proc.pid, 0)


async def assert_no_tasks_left() -> None:
    """Supervision, drains and the watchdog must all be finished once the outcome is settled."""
    await asyncio.sleep(0.1)
    current = asyncio.current_task()
    left = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    assert left == []


# (scenario flag, what the supervisor says it was waiting for, commands the worker received)
HANGS = [
    pytest.param(
        "hang_after_start",
        "SDK initialisation, connection and discovery",
        ["start"],
        id="after-start",
    ),
    pytest.param(
        "hang_after_discovering",
        "SDK initialisation, connection and discovery",
        ["start"],
        id="after-discovering",
    ),
    pytest.param(
        "hang_on_confirm",
        "the SDK to start the step",
        ["start", "confirm"],
        id="on-confirm",
    ),
    pytest.param(
        "hang_after_last_accept",
        "the next step or the result",
        ["start", "confirm"],
        id="after-last-accept",
    ),
]


# --- A silent worker is stopped -----------------------------------------------------------------


@pytest.mark.parametrize("ignore_term", [False, True], ids=["term", "kill"])
@pytest.mark.parametrize(("flag", "waiting_for", "received"), HANGS)
def test_a_worker_that_goes_silent_is_stopped_without_any_cancel(
    make_worker, commands_log, flag, waiting_for, received, ignore_term
):
    scenario = {flag: True, **({"ignore_term": True} if ignore_term else {})}
    expected_signal = signal.SIGKILL if ignore_term else signal.SIGTERM

    async def body():
        r = Run(make_worker, scenario)
        await r.attempt.start(LEFT)
        started = time.monotonic()
        confirm = None
        if flag in ("hang_on_confirm", "hang_after_last_accept"):
            await wait_for(lambda: r.attempt.prompt is not None)
            started = time.monotonic()
            confirm = asyncio.create_task(r.attempt.confirm(r.attempt.prompt.token))

        outcome = await r.settle()
        elapsed = time.monotonic() - started

        # Bounded: it waited for the deadline, and the whole stop fits the escalation budget.
        assert DEADLINE_S * 0.8 <= elapsed < BOUND_S
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert f"did not respond within {DEADLINE_S:g} s" in outcome.message
        assert f"while waiting for {waiting_for}" in outcome.message
        assert "SDK state is unknown" in outcome.message
        assert outcome.forced_termination
        assert outcome.exit_code == -int(expected_signal)
        assert outcome.published is None
        assert r.attempt.state is AttemptState.DONE and r.attempt.prompt is None

        # No browser cancel and no cancel command: the supervisor acted alone.
        assert commands(commands_log) == received
        assert not r.seen(proto.Result)

        assert_reaped(r.attempt)
        assert_nothing_published(r.store)
        assert_staging_gone(r.store, outcome.attempt)

        # A confirmation the worker never answered is released, not left waiting forever.
        if flag == "hang_on_confirm":
            with pytest.raises(AttemptEnded):
                await confirm
        elif confirm is not None:
            await confirm
        await assert_no_tasks_left()

    run(body())


def test_a_hang_after_the_last_step_is_not_mistaken_for_completion(make_worker):
    async def body():
        r = Run(make_worker, {"hang_after_last_accept": True})
        await r.attempt.start(BOTH)
        confirmed: set[str] = set()
        async with asyncio.timeout(SETTLE_S):
            while r.attempt.state is not AttemptState.DONE:
                prompt = r.attempt.prompt
                if prompt is not None and prompt.token not in confirmed:
                    confirmed.add(prompt.token)
                    try:
                        await r.attempt.confirm(prompt.token)
                    except AttemptEnded:
                        pass
                await asyncio.sleep(0.01)
        outcome = await r.attempt.outcome()
        # Both prompts were answered and the first glove may have exported, yet nothing is published.
        assert len(confirmed) == 2
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.published is None
        assert_nothing_published(r.store)

    run(body())


# --- A live or thinking operator is never rushed -------------------------------------------------


def test_operator_think_time_is_never_bounded(make_worker):
    think_s = 2.5

    async def body():
        r = Run(make_worker, {}, cancel_grace_s=2)
        await r.attempt.start(BOTH)
        for _ in range(2):
            await wait_for(lambda: r.attempt.prompt is not None)
            token = r.attempt.prompt.token
            # Far longer than every deadline: the operator is simply looking at the prompt.
            await asyncio.sleep(think_s)
            assert r.attempt.state is AttemptState.RUNNING
            assert r.attempt._abort_reason is None
            assert r.attempt.prompt is not None and r.attempt.prompt.token == token
            await r.attempt.confirm(token)
            await wait_for(
                lambda token=token: (
                    r.attempt.prompt is None or r.attempt.prompt.token != token
                )
            )
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.COMPLETED
        assert {f.side for f in outcome.published.files} == set(BOTH)

    run(body())


def test_a_refused_confirmation_returns_to_unbounded_waiting(make_worker):
    async def body():
        r = Run(make_worker, {"reject_first_confirm": True})
        await r.attempt.start(LEFT)
        await wait_for(lambda: r.attempt.prompt is not None)
        token = r.attempt.prompt.token
        with pytest.raises(ConfirmError):
            await r.attempt.confirm(token)
        # The confirm deadline must be gone: the prompt is still offered and the operator may take time.
        await asyncio.sleep(2.5)
        assert r.attempt.state is AttemptState.RUNNING
        assert r.attempt._abort_reason is None
        await r.attempt.confirm(token)
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.COMPLETED

    run(body())


@pytest.mark.parametrize(
    ("scenario", "case"),
    [
        ({"chatty_discovery": 2.5}, "startup"),
        ({"chatty_finish": 2.5}, "finish"),
        (
            {"steps": [{"title": "M", "instruction": "m", "time": 2.5}]},
            "motion-window",
        ),
    ],
    ids=["chatty-discovery", "chatty-finish", "motion-window"],
)
def test_a_worker_that_keeps_reporting_outlives_a_deadline_and_completes(
    make_worker, scenario, case
):
    async def body():
        r = Run(make_worker, scenario)
        await r.attempt.start(LEFT)
        began = time.monotonic()
        confirmed = False
        async with asyncio.timeout(SETTLE_S):
            while r.attempt.state is not AttemptState.DONE:
                prompt = r.attempt.prompt
                if prompt is not None and not confirmed:
                    confirmed = True
                    await r.attempt.confirm(prompt.token)
                await asyncio.sleep(0.01)
        outcome = await r.attempt.outcome()
        # The run lasted longer than the deadline it would have hit if events did not restart it.
        assert time.monotonic() - began > DEADLINE_S * 1.5
        assert outcome.kind is OutcomeKind.COMPLETED, outcome.message
        assert r.attempt._abort_reason is None
        assert len(r.store.list_published()) == 1

    run(body())


# --- Cancel and the watchdog do not interfere ----------------------------------------------------


def test_a_user_cancel_of_a_silent_worker_is_not_blamed_on_the_watchdog(make_worker):
    async def body():
        # A long startup deadline: only the user's cancel ends this attempt.
        r = Run(
            make_worker,
            {"hang_after_start": True},
            startup_timeout_s=30,
            cancel_grace_s=0.5,
        )
        await r.attempt.start(LEFT)
        await asyncio.sleep(0.2)
        await r.attempt.cancel()
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert outcome.forced_termination
        assert "did not respond" not in outcome.message
        assert r.attempt._abort_reason is None
        assert_reaped(r.attempt)
        assert_nothing_published(r.store)
        await assert_no_tasks_left()

    run(body())


def test_a_user_cancel_during_a_pending_confirm_is_not_blamed_on_the_watchdog(
    make_worker,
):
    async def body():
        r = Run(
            make_worker,
            {"hang_on_confirm": True},
            confirm_timeout_s=30,
            cancel_grace_s=0.5,
        )
        await r.attempt.start(LEFT)
        await wait_for(lambda: r.attempt.prompt is not None)
        confirm = asyncio.create_task(r.attempt.confirm(r.attempt.prompt.token))
        await asyncio.sleep(0.3)
        await r.attempt.cancel()
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.UNCERTAIN
        assert "did not respond" not in outcome.message
        with pytest.raises(AttemptEnded):
            await confirm
        assert_nothing_published(r.store)

    run(body())


def test_cancel_disarms_the_deadline_while_the_worker_winds_down(make_worker):
    async def body():
        # The worker needs 1.5 s to stop its SDK after the cancel, longer than the 0.6 s startup
        # deadline that was armed when the attempt started. A live deadline would kill it mid-cleanup.
        r = Run(
            make_worker,
            {"discover_delay": 30, "slow_cancel": 1.5},
            startup_timeout_s=0.6,
            cancel_grace_s=4,
        )
        await r.attempt.start(LEFT)
        await wait_for(lambda: r.seen(proto.Status))
        await r.attempt.cancel()
        outcome = await r.settle()
        assert outcome.kind is OutcomeKind.CANCELLED, outcome.message
        assert outcome.exit_code == 0 and not outcome.forced_termination
        assert r.attempt._abort_reason is None
        assert_nothing_published(r.store)

    run(body())


# --- Settings -----------------------------------------------------------------------------------


def test_deadline_defaults_follow_the_discovery_timeout_and_can_be_overridden(
    make_worker,
):
    worker, _ = make_worker(
        {},
        discovery_timeout_s=7,
        startup_timeout_s=None,
        confirm_timeout_s=30.0,
        finish_timeout_s=60.0,
    )
    # Initialisation and connection get a fixed margin on top of the discovery wait.
    assert worker.startup_timeout_s == 7 + 45
    assert worker.confirm_timeout_s == 30.0
    assert worker.finish_timeout_s == 60.0
    custom, _ = make_worker(
        {}, startup_timeout_s=3, confirm_timeout_s=4, finish_timeout_s=5
    )
    assert (
        custom.startup_timeout_s,
        custom.confirm_timeout_s,
        custom.finish_timeout_s,
    ) == (
        3,
        4,
        5,
    )
