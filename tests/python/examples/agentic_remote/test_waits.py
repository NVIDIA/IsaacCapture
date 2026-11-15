# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the assistance wait in
isaaccapture_examples.agentic_remote.robot_app.waits."""

from __future__ import annotations

import logging
import math
import threading

import pytest
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import (
    RobotAppConfig,
    WaitsConfig,
)
from isaaccapture_examples.agentic_remote.robot_app.waits import (
    BELL,
    BELL_TEXT_HOLD,
    BELL_TEXT_REST,
    COUNTDOWN,
    EXPIRED,
    MAX_STEP_S,
    AssistanceWait,
    WaitEvent,
)

DEFAULT_BELLS = (30, 10, 5, 4, 3, 2, 1)
REST_START = "assistance wait: 120 s left, then the arm moves to rest (agent stuck)"
REST_END = "assistance wait over: timed return to rest requested"
HOLD_START = "assistance wait: 120 s left, then the arm keeps holding (agent stuck)"
HOLD_END = "assistance wait over: the arm keeps holding"
JOIN_S = 5.0  # bound on a thread hand-over; reached only by a failing test


@pytest.fixture
def expired() -> list[bool]:
    """The arguments of every on_expire call."""
    return []


@pytest.fixture
def wait(
    config: RobotAppConfig, clock: FakeClock, expired: list[bool]
) -> AssistanceWait:
    return AssistanceWait(config.waits, clock, expired.append)


def run(
    wait: AssistanceWait, clock: FakeClock, seconds: float, dt: float = 1.0
) -> list[WaitEvent]:
    """Step the wait through `seconds` of fake time, dt at a time."""
    events: list[WaitEvent] = []
    for _ in range(round(seconds / dt)):
        clock.advance(dt)
        events += wait.step(clock.monotonic())
    return events


def started(wait: AssistanceWait, clock: FakeClock, **kwargs: bool) -> list[WaitEvent]:
    """Start the wait and take the step that reports the start."""
    wait.start("agent stuck", **kwargs)
    return wait.step(clock.monotonic())


def full_run(start: str, bell: str, end: str) -> list[WaitEvent]:
    """Every event of a default 120 s wait, in order."""
    events = [(COUNTDOWN, start)]
    for left in range(119, 0, -1):
        if left % 10 == 0:
            events.append((COUNTDOWN, f"assistance wait: {left} s left"))
        if left in DEFAULT_BELLS:
            events.append((BELL, bell.format(left)))
    return events + [(EXPIRED, end)]


REST_RUN = full_run(REST_START, "ARM WILL MOVE TO REST in {} s", REST_END)
HOLD_RUN = full_run(
    HOLD_START, "ARM WILL KEEP HOLDING, assistance wait ends in {} s", HOLD_END
)


def test_no_wait_runs_before_start(wait, clock, expired):
    assert not wait.active()
    assert wait.remaining(clock.monotonic()) is None
    assert run(wait, clock, 300.0) == []
    assert expired == []


def test_start_begins_at_the_full_value_and_reports_it(wait, clock, expired):
    wait.start("agent stuck")
    assert wait.active()
    assert wait.remaining(clock.monotonic()) == 120.0
    assert wait.step(clock.monotonic()) == [(COUNTDOWN, REST_START)]
    assert wait.step(clock.monotonic()) == []
    assert expired == []


def test_the_start_line_leaves_out_an_empty_reason(wait, clock):
    wait.start("")
    assert wait.step(clock.monotonic()) == [
        (COUNTDOWN, "assistance wait: 120 s left, then the arm moves to rest")
    ]


def test_events_are_pairs_of_kind_and_text(wait, clock):
    events = started(wait, clock) + run(wait, clock, 120.0)
    assert {kind for kind, _ in events} == {COUNTDOWN, BELL, EXPIRED}
    assert (COUNTDOWN, BELL, EXPIRED) == ("countdown", "bell", "expired")
    assert all(type(text) is str and text for _, text in events)


def test_full_countdown_of_the_default_120_s(wait, clock, expired):
    events = started(wait, clock)
    for left in range(119, 0, -1):
        clock.advance(1.0)
        due = wait.step(clock.monotonic())
        assert wait.remaining(clock.monotonic()) == float(left)
        assert expired == []
        expected = []
        if left % 10 == 0:
            expected.append((COUNTDOWN, f"assistance wait: {left} s left"))
        if left in DEFAULT_BELLS:
            expected.append((BELL, f"ARM WILL MOVE TO REST in {left} s"))
        assert due == expected
        events += due
    clock.advance(1.0)
    events += wait.step(clock.monotonic())
    assert events == REST_RUN
    assert expired == [True]
    assert not wait.active()
    assert wait.remaining(clock.monotonic()) is None


def test_countdown_lines_come_at_start_and_every_10_s(wait, clock):
    events = started(wait, clock) + run(wait, clock, 120.0)
    lines = [text for kind, text in events if kind == COUNTDOWN]
    assert lines == [REST_START] + [
        f"assistance wait: {left} s left" for left in range(110, 0, -10)
    ]


def test_full_countdown_in_10_ms_steps_gives_the_same_events(wait, clock, expired):
    """The robot app steps the wait every 10 ms."""
    t0 = clock.monotonic()
    events = started(wait, clock)
    bells_left = []
    while wait.active():
        clock.advance(0.01)
        before = wait.remaining(clock.monotonic())
        due = wait.step(clock.monotonic())
        bells_left += [before for kind, _ in due if kind == BELL]
        events += due
        assert clock.monotonic() - t0 < 121.0
    assert events == REST_RUN
    assert expired == [True]
    assert clock.monotonic() - t0 == pytest.approx(120.0, abs=0.011)
    for mark, left in zip(DEFAULT_BELLS, bells_left, strict=True):
        assert mark - 0.011 < left <= mark


def test_a_step_counts_at_most_one_second(wait, clock, expired):
    started(wait, clock)
    clock.advance(3600.0)
    now = clock.monotonic()
    assert wait.remaining(now) == 120.0 - MAX_STEP_S == 119.0
    assert wait.step(now) == []
    assert wait.remaining(now) == 119.0
    assert wait.active()
    assert expired == []


def test_a_clock_that_jumps_still_needs_every_step_and_rings_every_bell(
    wait, clock, expired
):
    events = started(wait, clock)
    for step in range(1, 121):
        assert expired == []
        clock.advance(1e6)
        events += wait.step(clock.monotonic())
        assert wait.active() == (step < 120)
    assert events == REST_RUN
    assert expired == [True]


def test_a_stall_near_the_end_moves_the_wait_on_by_one_second_only(
    wait, clock, expired
):
    started(wait, clock)
    run(wait, clock, 88.5, dt=0.5)
    assert wait.remaining(clock.monotonic()) == 31.5
    clock.advance(600.0)
    assert wait.step(clock.monotonic()) == []
    assert wait.remaining(clock.monotonic()) == 30.5
    clock.advance(0.5)
    assert wait.step(clock.monotonic()) == [
        (COUNTDOWN, "assistance wait: 30 s left"),
        (BELL, "ARM WILL MOVE TO REST in 30 s"),
    ]
    assert expired == []


def test_time_between_start_and_the_first_step_counts(wait, clock):
    wait.start("agent stuck")
    clock.advance(0.5)
    assert wait.remaining(clock.monotonic()) == 119.5
    wait.step(clock.monotonic())
    assert wait.remaining(clock.monotonic()) == 119.5


def test_remaining_does_not_move_the_count(wait, clock):
    started(wait, clock)
    t0 = clock.monotonic()
    assert wait.remaining(t0 + 0.25) == 119.75
    assert wait.remaining(t0 + 0.5) == 119.5
    assert wait.remaining(t0 + 50.0) == 119.0
    assert wait.remaining(t0) == 120.0
    assert wait.step(t0 + 0.25) == []
    assert wait.remaining(t0 + 0.25) == 119.75


def test_a_now_older_than_the_last_one_counts_no_time(wait, clock):
    started(wait, clock)
    t0 = clock.monotonic()
    wait.step(t0 + 0.5)
    wait.step(t0 + 0.25)
    assert wait.remaining(t0 + 0.25) == 119.5
    wait.step(t0 + 0.75)
    assert wait.remaining(t0 + 0.75) == 119.25


@pytest.mark.parametrize("bad_now", [math.nan, math.inf, -math.inf])
def test_a_now_that_is_not_finite_counts_no_time(wait, clock, expired, bad_now):
    started(wait, clock)
    t0 = clock.monotonic()
    for _ in range(200):
        assert wait.step(bad_now) == []
    assert wait.remaining(bad_now) == 120.0
    assert wait.remaining(t0) == 120.0
    wait.step(t0 + 0.5)
    assert wait.remaining(t0 + 0.5) == 119.5
    assert expired == []


def test_bells_ring_once_at_each_configured_mark(clock, expired):
    waits = WaitsConfig(assistance_s=90.0, bell_at_s=(60.0, 2.5, 0.5))
    wait = AssistanceWait(waits, clock, expired.append)
    started(wait, clock)
    bells = []
    for _ in range(180):
        clock.advance(0.5)
        left = wait.remaining(clock.monotonic())
        bells += [
            (left, text) for kind, text in wait.step(clock.monotonic()) if kind == BELL
        ]
    assert bells == [
        (60.0, "ARM WILL MOVE TO REST in 60 s"),
        (2.5, "ARM WILL MOVE TO REST in 2.5 s"),
        (0.5, "ARM WILL MOVE TO REST in 0.5 s"),
    ]
    assert expired == [True]


def test_every_default_bell_carries_the_warning_and_the_seconds_left(wait, clock):
    events = started(wait, clock) + run(wait, clock, 120.0)
    bells = [text for kind, text in events if kind == BELL]
    assert bells == [f"{BELL_TEXT_REST} in {left} s" for left in DEFAULT_BELLS]
    assert BELL_TEXT_REST == "ARM WILL MOVE TO REST"


def test_marks_passed_in_one_step_all_ring(clock, expired):
    waits = WaitsConfig(assistance_s=90.0, bell_at_s=(1.0, 0.75, 0.5))
    wait = AssistanceWait(waits, clock, expired.append)
    started(wait, clock)
    assert run(wait, clock, 88.5, dt=0.5)[-1] == (
        COUNTDOWN,
        "assistance wait: 10 s left",
    )
    clock.advance(1.0)
    assert wait.step(clock.monotonic()) == [
        (BELL, "ARM WILL MOVE TO REST in 1 s"),
        (BELL, "ARM WILL MOVE TO REST in 0.75 s"),
        (BELL, "ARM WILL MOVE TO REST in 0.5 s"),
    ]
    assert expired == []


def test_a_mark_at_the_full_value_rings_with_the_start(clock, expired):
    waits = WaitsConfig(assistance_s=90.0, bell_at_s=(90.0, 1.0))
    wait = AssistanceWait(waits, clock, expired.append)
    assert started(wait, clock) == [
        (
            COUNTDOWN,
            "assistance wait: 90 s left, then the arm moves to rest (agent stuck)",
        ),
        (BELL, "ARM WILL MOVE TO REST in 90 s"),
    ]


def test_countdown_lines_of_a_wait_that_is_no_multiple_of_10_s(clock, expired):
    waits = WaitsConfig(assistance_s=95.0, bell_at_s=(1.0,))
    wait = AssistanceWait(waits, clock, expired.append)
    events = started(wait, clock) + run(wait, clock, 95.0)
    lines = [text for kind, text in events if kind == COUNTDOWN]
    assert lines == [
        "assistance wait: 95 s left, then the arm moves to rest (agent stuck)"
    ] + [f"assistance wait: {left} s left" for left in range(90, 0, -10)]
    assert expired == [True]


def test_the_wait_expires_once(wait, clock, expired):
    events = started(wait, clock) + run(wait, clock, 120.0)
    assert events[-1] == (EXPIRED, REST_END)
    assert expired == [True]
    assert run(wait, clock, 600.0) == []
    assert run(wait, clock, 5.0, dt=0.01) == []
    assert [kind for kind, _ in events].count(EXPIRED) == 1
    assert expired == [True]
    assert not wait.active()


def test_on_expire_is_not_called_before_the_last_second_is_over(wait, clock, expired):
    started(wait, clock)
    run(wait, clock, 119.75, dt=0.25)
    assert wait.remaining(clock.monotonic()) == 0.25
    assert expired == []
    assert run(wait, clock, 0.25, dt=0.25) == [(EXPIRED, REST_END)]
    assert expired == [True]


def test_steps_from_several_threads_expire_the_wait_once(wait, clock, expired):
    started(wait, clock)
    run(wait, clock, 119.0)
    clock.advance(1.0)
    now = clock.monotonic()
    barrier = threading.Barrier(8)
    seen: list[WaitEvent] = []

    def step() -> None:
        barrier.wait(JOIN_S)
        seen.extend(wait.step(now))

    threads = [threading.Thread(target=step) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_S)
    assert seen == [(EXPIRED, REST_END)]
    assert expired == [True]


def test_on_expire_may_call_back_into_the_wait(config, clock):
    seen = []

    def on_expire(timed_return: bool) -> None:
        seen.append((timed_return, wait.active(), wait.remaining(clock.monotonic())))
        wait.start("second wait")

    wait = AssistanceWait(config.waits, clock, on_expire)
    started(wait, clock)
    run(wait, clock, 119.0)
    clock.advance(1.0)
    events: list[WaitEvent] = []
    # A lock held across on_expire would hang here: bound it with a thread.
    stepper = threading.Thread(
        target=lambda: events.extend(wait.step(clock.monotonic())), daemon=True
    )
    stepper.start()
    stepper.join(JOIN_S)
    assert not stepper.is_alive()
    assert events == [(EXPIRED, REST_END)]
    assert seen == [(True, False, None)]
    assert wait.remaining(clock.monotonic()) == 120.0
    assert wait.step(clock.monotonic()) == [
        (
            COUNTDOWN,
            "assistance wait: 120 s left, then the arm moves to rest (second wait)",
        )
    ]


def test_an_error_in_on_expire_leaves_the_wait_ended(config, clock):
    calls = []

    def on_expire(timed_return: bool) -> None:
        calls.append(timed_return)
        raise RuntimeError("return to rest could not be queued")

    wait = AssistanceWait(config.waits, clock, on_expire)
    started(wait, clock)
    run(wait, clock, 119.0)
    clock.advance(1.0)
    with pytest.raises(RuntimeError, match=r"could not be queued"):
        wait.step(clock.monotonic())
    assert not wait.active()
    assert run(wait, clock, 300.0) == []
    assert calls == [True]


def test_cancel_ends_the_wait_without_an_expiry(wait, clock, expired):
    started(wait, clock)
    run(wait, clock, 40.0)
    assert wait.cancel("STOP from the console") is True
    assert not wait.active()
    assert wait.remaining(clock.monotonic()) is None
    assert run(wait, clock, 300.0) == []
    assert expired == []


def test_cancel_in_the_last_second_still_prevents_the_expiry(wait, clock, expired):
    started(wait, clock)
    run(wait, clock, 119.75, dt=0.25)
    assert wait.cancel("takeover accepted") is True
    assert run(wait, clock, 10.0) == []
    assert expired == []


def test_cancel_before_the_first_step_reports_nothing(wait, clock, expired):
    wait.start("agent stuck")
    assert wait.cancel("fault-stop") is True
    assert run(wait, clock, 300.0) == []
    assert expired == []


def test_cancel_without_a_running_wait_does_nothing(wait, clock, expired, caplog):
    with caplog.at_level(logging.INFO, logger="isaaccapture"):
        assert wait.cancel("STOP from the console") is False
        started(wait, clock)
        run(wait, clock, 120.0)
        records = len(caplog.records)
        assert wait.cancel("STOP from the console") is False
        assert len(caplog.records) == records
    assert not wait.active()
    assert wait.step(clock.monotonic()) == []
    assert expired == [True]


def test_start_and_cancel_log_their_reason(wait, clock, caplog):
    name = "isaaccapture.examples.agentic_remote.waits"
    with caplog.at_level(logging.INFO, logger=name):
        started(wait, clock)
        run(wait, clock, 20.0)
        wait.cancel("robot connection lost")
    lines = [r.getMessage() for r in caplog.records if r.name == name]
    assert len(lines) == 2
    assert "agent stuck" in lines[0]
    assert "robot connection lost" in lines[1]


def test_cancel_racing_the_last_step_gives_exactly_one_outcome(wait, clock, expired):
    started(wait, clock)
    run(wait, clock, 119.0)
    clock.advance(1.0)
    now = clock.monotonic()
    barrier = threading.Barrier(2)
    cancelled: list[bool] = []
    seen: list[WaitEvent] = []

    def cancel() -> None:
        barrier.wait(JOIN_S)
        cancelled.append(wait.cancel("STOP from the console"))

    def step() -> None:
        barrier.wait(JOIN_S)
        seen.extend(wait.step(now))

    threads = [threading.Thread(target=cancel), threading.Thread(target=step)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_S)
    assert not wait.active()
    if cancelled == [True]:
        assert (seen, expired) == ([], [])
    else:
        assert (cancelled, seen, expired) == ([False], [(EXPIRED, REST_END)], [True])


def test_start_while_running_starts_over_with_the_new_reason(wait, clock, expired):
    started(wait, clock)
    first = run(wait, clock, 95.0)
    assert (BELL, "ARM WILL MOVE TO REST in 30 s") in first
    assert wait.remaining(clock.monotonic()) == 25.0

    wait.start("step budget ran out")
    assert wait.remaining(clock.monotonic()) == 120.0
    again = wait.step(clock.monotonic()) + run(wait, clock, 120.0)
    assert again == full_run(
        "assistance wait: 120 s left, then the arm moves to rest (step budget ran out)",
        "ARM WILL MOVE TO REST in {} s",
        REST_END,
    )
    assert expired == [True]


def test_a_restart_takes_the_new_timed_return(wait, clock, expired):
    started(wait, clock, timed_return=True)
    run(wait, clock, 60.0)
    wait.start("agent stuck", timed_return=False)
    events = wait.step(clock.monotonic()) + run(wait, clock, 120.0)
    assert events == HOLD_RUN
    assert expired == [False]


def test_a_restart_before_the_first_step_reports_one_start(wait, clock):
    wait.start("agent stuck")
    wait.start("model error")
    assert wait.step(clock.monotonic()) == [
        (
            COUNTDOWN,
            "assistance wait: 120 s left, then the arm moves to rest (model error)",
        )
    ]
    assert wait.step(clock.monotonic()) == []


def test_start_after_an_expiry_or_a_cancel_runs_a_new_wait(wait, clock, expired):
    assert started(wait, clock) + run(wait, clock, 120.0) == REST_RUN
    assert started(wait, clock, timed_return=False) + run(wait, clock, 120.0) == (
        HOLD_RUN
    )
    assert expired == [True, False]
    started(wait, clock)
    wait.cancel("new task")
    assert started(wait, clock) + run(wait, clock, 120.0) == REST_RUN
    assert expired == [True, False, True]


def test_without_timed_return_the_wait_expires_and_the_arm_keeps_holding(
    wait, clock, expired
):
    events = started(wait, clock, timed_return=False)
    assert events == [(COUNTDOWN, HOLD_START)]
    events += run(wait, clock, 119.0)
    assert wait.remaining(clock.monotonic()) == 1.0
    assert expired == []
    events += run(wait, clock, 1.0)
    assert events == HOLD_RUN
    assert expired == [False]
    assert not wait.active()
    assert run(wait, clock, 300.0) == []
    assert expired == [False]


def test_without_timed_return_no_text_announces_a_move(wait, clock):
    events = started(wait, clock, timed_return=False) + run(wait, clock, 120.0)
    bells = [text for kind, text in events if kind == BELL]
    assert len(bells) == len(DEFAULT_BELLS)
    assert all(text.startswith(BELL_TEXT_HOLD) for text in bells)
    assert BELL_TEXT_HOLD == "ARM WILL KEEP HOLDING"
    assert not any("REST" in text.upper() for _, text in events)


def test_the_wait_cannot_be_paused():
    assert not hasattr(AssistanceWait, "pause")
    assert not hasattr(AssistanceWait, "resume")


def test_the_wait_never_sleeps_and_never_moves_the_clock(config, expired):
    class NoSleepClock(FakeClock):
        def sleep(self, s: float) -> None:
            raise AssertionError(f"sleep({s}) was called")

    clock = NoSleepClock()
    wait = AssistanceWait(config.waits, clock, expired.append)
    wait.start("agent stuck")
    t0 = clock.monotonic()
    for i in range(1, 121):
        wait.step(t0 + i)
    assert expired == [True]
    assert clock.monotonic() == t0
