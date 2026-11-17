# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the permit gate in
isaaccapture_examples.agentic_remote.robot_app.permits."""

from __future__ import annotations

import dataclasses
import math
import threading

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import Permit
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import (
    FakeClock,
    SystemClock,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import Owner
from isaaccapture_examples.agentic_remote.robot_app.permits import (
    FUTURE_TOLERANCE_S,
    PermitGate,
)

MOVE = "move-1"
WAIT_S = 5.0  # longest a test waits for another thread
NO_PERMIT = "no permit for this move"
EXPIRED = "permit expired"
TOO_OLD = "permit evidence is too old"

# Times in these tests are sums of powers of two, so the clock arithmetic is exact.


class FakeChecker:
    """Stands in for the agent checker: records what its watchdog is fed."""

    def __init__(self) -> None:
        self.fed: list[float | None] = []

    def feed_watchdog(self, t_mono: float | None = None) -> None:
        self.fed.append(t_mono)


@pytest.fixture
def authority() -> Authority:
    """An authority whose control owner is the agent: owner number 1, stop number 0."""
    authority = Authority()
    authority.set_owner(Owner.AGENT, "new task")
    return authority


@pytest.fixture
def checker() -> FakeChecker:
    return FakeChecker()


@pytest.fixture
def gate(authority, checker, config, clock) -> PermitGate:
    return PermitGate(authority, checker, config.permits, clock)


def make_permit(
    authority: Authority,
    clock: FakeClock,
    *,
    move_id: str = MOVE,
    through_index: int = 10,
    evidence_age_s: float = 0.0,
    expires_in_s: float = 1.0,
) -> Permit:
    """A permit quoting the authority's current values, timed from the clock's now."""
    state = authority.snapshot()
    return Permit(
        move_id=move_id,
        session_id=state.session_id,
        owner_number=state.owner_number,
        stop_number=state.stop_number,
        through_index=through_index,
        evidence_wall_s=clock.wall() - evidence_age_s,
        expires_mono_s=clock.monotonic() + expires_in_s,
    )


def update(gate: PermitGate, clock: FakeClock, permit: Permit) -> str | None:
    return gate.update(permit, clock.monotonic(), clock.wall() - clock.monotonic())


def covers(gate: PermitGate, clock: FakeClock, index: int, move_id: str = MOVE):
    return gate.covers(move_id, index, clock.monotonic())


def test_the_default_age_limit_is_three_seconds(config):
    # The times in the tests below are chosen around this value.
    assert config.permits.evidence_age_limit_s == 3.0


def test_nothing_is_covered_before_a_permit_arrives(gate, clock):
    assert covers(gate, clock, 0) == NO_PERMIT


def test_an_accepted_permit_covers_its_move_up_to_through_index(gate, authority, clock):
    assert update(gate, clock, make_permit(authority, clock, through_index=10)) is None
    assert covers(gate, clock, 0) is None
    assert covers(gate, clock, 10) is None
    assert covers(gate, clock, 11) == "permit reaches setpoint 10, not 11"
    assert covers(gate, clock, 0, move_id="move-2") == NO_PERMIT


def test_an_accepted_permit_feeds_the_watchdog_with_the_evidence_time(
    gate, authority, checker, clock
):
    update(gate, clock, make_permit(authority, clock, evidence_age_s=0.5))
    assert checker.fed == [clock.monotonic() - 0.5]


def test_the_evidence_time_is_converted_with_the_given_clock_difference(
    gate, authority, checker, clock
):
    permit = dataclasses.replace(
        make_permit(authority, clock), evidence_wall_s=149.0, expires_mono_s=60.0
    )
    assert gate.update(permit, now_mono=50.0, wall_minus_mono=100.0) is None
    assert checker.fed == [49.0]
    assert gate.covers(MOVE, 10, now_mono=51.75) is None
    # Evidence taken at 49.0 is three seconds old at 52.0.
    assert gate.covers(MOVE, 10, now_mono=52.0) == TOO_OLD


def test_the_times_default_to_the_gate_clock(gate, authority, checker, clock):
    assert gate.update(make_permit(authority, clock, evidence_age_s=0.25)) is None
    assert checker.fed == [clock.monotonic() - 0.25]
    assert gate.covers(MOVE, 10) is None
    clock.advance(1.0)
    assert gate.covers(MOVE, 10) == EXPIRED


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"session_id": "0" * 32}, "permit: session id does not match"),
        ({"owner_number": 0}, "permit: owner number 0 does not match the current 1"),
        ({"stop_number": 3}, "permit: stop number 3 does not match the current 0"),
    ],
)
def test_update_refuses_a_permit_that_quotes_other_values(
    gate, authority, checker, clock, change, reason
):
    permit = dataclasses.replace(make_permit(authority, clock), **change)
    assert update(gate, clock, permit) == reason
    assert covers(gate, clock, 0) == NO_PERMIT
    assert checker.fed == []


def test_update_refuses_a_permit_made_before_a_stop(gate, authority, checker, clock):
    permit = make_permit(authority, clock)
    authority.raise_stop("typed stop", "console")
    authority.set_owner(Owner.AGENT, "new task")
    assert (
        update(gate, clock, permit)
        == "permit: stop number 0 does not match the current 1"
    )
    assert checker.fed == []


def test_update_refuses_a_permit_while_a_fault_is_latched(
    gate, authority, checker, clock
):
    authority.raise_fault("motor fault")
    permit = make_permit(authority, clock)
    assert update(gate, clock, permit) == "permit: fault-stop latched: motor fault"
    assert checker.fed == []


@pytest.mark.parametrize("owner", [Owner.NOBODY, Owner.HUMAN, Owner.RECOVERY])
def test_update_refuses_a_permit_when_the_agent_is_not_the_control_owner(
    gate, authority, checker, clock, owner
):
    authority.set_owner(owner, "test")
    permit = make_permit(authority, clock)
    assert update(gate, clock, permit) == f"permit: control owner is {owner}, not agent"
    assert covers(gate, clock, 0) == NO_PERMIT
    assert checker.fed == []


@pytest.mark.parametrize("expires_in_s", [0.0, -0.5])
def test_update_refuses_an_expired_permit(
    gate, authority, checker, clock, expires_in_s
):
    clock.advance(1.0)
    permit = make_permit(authority, clock, expires_in_s=expires_in_s)
    assert update(gate, clock, permit) == EXPIRED
    assert covers(gate, clock, 0) == NO_PERMIT
    assert checker.fed == []


def test_update_refuses_evidence_from_the_future(gate, authority, checker, clock):
    assert FUTURE_TOLERANCE_S == 0.05
    permit = make_permit(authority, clock, evidence_age_s=-0.0625)
    assert update(gate, clock, permit) == "permit evidence is in the future"
    assert covers(gate, clock, 0) == NO_PERMIT
    assert checker.fed == []


def test_evidence_just_ahead_of_the_clock_counts_as_taken_now(
    gate, authority, checker, clock
):
    now = clock.monotonic()
    permit = make_permit(authority, clock, evidence_age_s=-0.03125, expires_in_s=10.0)
    assert update(gate, clock, permit) is None
    assert checker.fed == [now]
    clock.advance(3.0)
    assert covers(gate, clock, 0) == TOO_OLD


def test_update_refuses_evidence_at_or_over_the_age_limit(
    gate, authority, checker, clock
):
    for age_s in (3.0, 3.5, 3600.0):
        permit = make_permit(authority, clock, evidence_age_s=age_s)
        assert update(gate, clock, permit) == TOO_OLD
    assert covers(gate, clock, 0) == NO_PERMIT
    assert checker.fed == []


def test_update_accepts_evidence_just_under_the_age_limit(
    gate, authority, checker, clock
):
    permit = make_permit(authority, clock, evidence_age_s=2.75)
    assert update(gate, clock, permit) is None
    assert covers(gate, clock, 10) is None
    assert checker.fed == [clock.monotonic() - 2.75]


def test_the_age_limit_comes_from_the_config(authority, checker, config, clock):
    permits = dataclasses.replace(config.permits, evidence_age_limit_s=0.5)
    gate = PermitGate(authority, checker, permits, clock)
    half_a_second_old = make_permit(authority, clock, evidence_age_s=0.5)
    assert update(gate, clock, half_a_second_old) == TOO_OLD
    assert update(gate, clock, make_permit(authority, clock, expires_in_s=9.0)) is None
    clock.advance(0.25)
    assert covers(gate, clock, 0) is None
    clock.advance(0.25)
    assert covers(gate, clock, 0) == TOO_OLD


@pytest.mark.parametrize(
    "change",
    [
        {"expires_mono_s": math.inf},
        {"expires_mono_s": math.nan},
        {"evidence_wall_s": math.nan},
        {"evidence_wall_s": -math.inf},
    ],
)
def test_update_refuses_times_that_are_not_finite(
    gate, authority, checker, clock, change
):
    permit = dataclasses.replace(make_permit(authority, clock), **change)
    assert update(gate, clock, permit) == "permit times are not finite"
    assert covers(gate, clock, 0) == NO_PERMIT
    assert checker.fed == []


def test_a_refused_permit_leaves_the_held_one_in_place(gate, authority, checker, clock):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    refused = [
        make_permit(authority, clock, through_index=50, expires_in_s=0.0),
        make_permit(authority, clock, through_index=50, evidence_age_s=3.0),
        make_permit(authority, clock, through_index=50, evidence_age_s=-1.0),
        dataclasses.replace(
            make_permit(authority, clock, through_index=50), stop_number=9
        ),
        make_permit(authority, clock, move_id="move-2", expires_in_s=0.0),
    ]
    for permit in refused:
        assert update(gate, clock, permit) is not None
    assert covers(gate, clock, 10) is None
    assert covers(gate, clock, 11) == "permit reaches setpoint 10, not 11"
    assert len(checker.fed) == 1


def test_a_permit_that_reaches_further_replaces_the_held_one(
    gate, authority, checker, clock
):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    clock.advance(0.25)
    assert update(gate, clock, make_permit(authority, clock, through_index=25)) is None
    assert covers(gate, clock, 25) is None
    assert covers(gate, clock, 26) == "permit reaches setpoint 25, not 26"
    assert checker.fed == [clock.monotonic() - 0.25, clock.monotonic()]


def test_a_permit_with_newer_evidence_and_the_same_reach_replaces_the_held_one(
    gate, authority, checker, clock
):
    start = clock.monotonic()
    first = make_permit(authority, clock, evidence_age_s=2.0, expires_in_s=10.0)
    update(gate, clock, first)
    clock.advance(0.5)
    update(gate, clock, make_permit(authority, clock, expires_in_s=10.0))
    assert checker.fed == [start - 2.0, start + 0.5]
    # The first permit's evidence is over the age limit by now; the second's is not.
    clock.advance(1.0)
    assert covers(gate, clock, 10) is None


def test_a_later_permit_brings_its_own_expiry(gate, authority, clock):
    update(gate, clock, make_permit(authority, clock, expires_in_s=1.0))
    clock.advance(0.5)
    update(gate, clock, make_permit(authority, clock, expires_in_s=1.0))
    clock.advance(0.75)
    assert covers(gate, clock, 10) is None
    clock.advance(0.25)
    assert covers(gate, clock, 10) == EXPIRED


def test_a_permit_that_reaches_less_far_shortens_the_reach(
    gate, authority, checker, clock
):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    clock.advance(0.25)
    smaller = make_permit(authority, clock, through_index=5, expires_in_s=10.0)
    assert update(gate, clock, smaller) is None
    assert covers(gate, clock, 5) is None
    assert covers(gate, clock, 6) == "permit reaches setpoint 5, not 6"
    assert len(checker.fed) == 2
    # It brings its own expiry too.
    clock.advance(0.75)
    assert covers(gate, clock, 5) is None


def test_a_permit_with_the_same_evidence_that_reaches_less_far_shortens_the_reach(
    gate, authority, clock
):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    assert update(gate, clock, make_permit(authority, clock, through_index=5)) is None
    assert covers(gate, clock, 6) == "permit reaches setpoint 5, not 6"


def test_an_expired_held_permit_does_not_block_a_shorter_one(gate, authority, clock):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    clock.advance(1.0)
    assert covers(gate, clock, 0) == EXPIRED
    assert update(gate, clock, make_permit(authority, clock, through_index=5)) is None
    assert covers(gate, clock, 5) is None


def test_a_permit_with_older_evidence_is_ignored_without_a_reason(
    gate, authority, checker, clock
):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    older = make_permit(authority, clock, through_index=20, evidence_age_s=1.0)
    assert update(gate, clock, older) is None
    assert covers(gate, clock, 10) is None
    assert covers(gate, clock, 11) == "permit reaches setpoint 10, not 11"
    assert checker.fed == [clock.monotonic()]


def test_a_permit_with_older_evidence_is_ignored_however_the_clocks_read(
    gate, authority, checker, clock
):
    # The rule compares the quoted times, not what each arrival made of them.
    first = make_permit(authority, clock, through_index=10, evidence_age_s=0.5)
    update(gate, clock, first)
    older = make_permit(authority, clock, through_index=20, evidence_age_s=0.75)
    # With a wall clock that stepped back by 0.5 s this evidence converts to a
    # later time than the held permit's did.
    wall_minus_mono = clock.wall() - clock.monotonic() - 0.5
    assert gate.update(older, clock.monotonic(), wall_minus_mono) is None
    assert covers(gate, clock, 11) == "permit reaches setpoint 10, not 11"
    assert len(checker.fed) == 1


# Less cover needs no fresh evidence: a permit of the held move that reaches
# less far and fails a time check lowers the held reach and nothing else.

REACHES_MINUS_1 = "permit reaches setpoint -1, not 3"


@pytest.mark.parametrize("step_s", [-0.5, 4.0])
def test_a_cancel_is_taken_after_a_step_of_the_wall_clock(
    gate, authority, checker, clock, step_s
):
    held = make_permit(authority, clock, through_index=25, evidence_age_s=0.25)
    offset = clock.wall() - clock.monotonic()
    assert gate.update(held, clock.monotonic(), offset) is None
    assert covers(gate, clock, 3) is None
    gate.arrived.clear()
    # The wall clock steps; the cancel quotes the evidence of the permit before.
    clock.advance(0.125)
    cancel = dataclasses.replace(
        make_permit(authority, clock, through_index=-1),
        evidence_wall_s=held.evidence_wall_s,
    )
    assert gate.update(cancel, clock.monotonic(), offset + step_s) is None
    assert covers(gate, clock, 3) == REACHES_MINUS_1
    assert gate.last_refusal(MOVE) is None
    assert gate.arrived.is_set()  # the move looks again
    assert len(checker.fed) == 1  # no new evidence, so the watchdog is not fed


@pytest.mark.parametrize(
    "fields",
    [
        {"expires_in_s": 0.0},
        {"evidence_age_s": 3.5},
        {"evidence_age_s": -1.0},
    ],
)
def test_a_cancel_is_taken_whatever_its_times_say(gate, authority, clock, fields):
    update(gate, clock, make_permit(authority, clock, through_index=25))
    cancel = make_permit(authority, clock, through_index=-1, **fields)
    assert update(gate, clock, cancel) is None
    assert covers(gate, clock, 3) == REACHES_MINUS_1
    assert gate.last_refusal(MOVE) is None
    # A second one finds nothing to lower and is refused like any other permit.
    assert update(gate, clock, cancel) is not None
    assert covers(gate, clock, 3) == REACHES_MINUS_1


def test_a_permit_taken_without_its_times_keeps_the_held_expiry_and_evidence(
    gate, authority, checker, clock
):
    held = make_permit(
        authority, clock, through_index=10, evidence_age_s=2.0, expires_in_s=1.5
    )
    update(gate, clock, held)
    # Its evidence is from the future, so its own times count for nothing.
    shorter = make_permit(
        authority, clock, through_index=5, evidence_age_s=-1.0, expires_in_s=60.0
    )
    assert update(gate, clock, shorter) is None
    assert covers(gate, clock, 5) is None
    assert covers(gate, clock, 6) == "permit reaches setpoint 5, not 6"
    assert checker.fed == [clock.monotonic() - 2.0]
    clock.advance(0.75)
    assert covers(gate, clock, 5) is None
    clock.advance(0.25)
    assert covers(gate, clock, 5) == TOO_OLD  # the held evidence: 2 s old then
    gate.clear()
    update(gate, clock, make_permit(authority, clock, expires_in_s=1.0))
    late = make_permit(authority, clock, through_index=5, evidence_age_s=-1.0)
    late = dataclasses.replace(late, expires_mono_s=late.expires_mono_s + 60.0)
    assert update(gate, clock, late) is None
    clock.advance(1.0)
    assert covers(gate, clock, 5) == EXPIRED  # the held expiry, not its own


def test_a_permit_on_older_evidence_that_reaches_less_far_lowers_the_reach(
    gate, authority, checker, clock
):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    older = make_permit(authority, clock, through_index=5, evidence_age_s=1.0)
    assert update(gate, clock, older) is None
    assert covers(gate, clock, 5) is None
    assert covers(gate, clock, 6) == "permit reaches setpoint 5, not 6"
    assert checker.fed == [clock.monotonic()]  # the held evidence stays


def test_only_a_shorter_permit_of_the_held_move_with_current_numbers_is_taken_so(
    gate, authority, checker, clock
):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    stale = make_permit(authority, clock, through_index=-1, expires_in_s=0.0)
    refused_ones = [
        dataclasses.replace(stale, stop_number=9),  # other numbers
        dataclasses.replace(stale, move_id="move-2"),  # another move
        dataclasses.replace(stale, through_index=-2),  # no cancel
        dataclasses.replace(stale, through_index=10),  # the same reach
        dataclasses.replace(stale, through_index=11),  # further
    ]
    for permit in refused_ones:
        assert update(gate, clock, permit) is not None, permit
        assert covers(gate, clock, 10) is None
    assert gate.last_refusal(MOVE) == EXPIRED
    assert len(checker.fed) == 1
    # A late permit of the held move that is refused never ends the cover.
    assert covers(gate, clock, 11) == "permit reaches setpoint 10, not 11"


# The same evidence quoted again, on the system clock: two reads of its wall and
# monotonic clocks never give the same difference twice.

REPEATS = 200


def system_permit(authority: Authority, evidence_wall_s: float, through_index: int):
    state = authority.snapshot()
    return Permit(
        move_id=MOVE,
        session_id=state.session_id,
        owner_number=state.owner_number,
        stop_number=state.stop_number,
        through_index=through_index,
        evidence_wall_s=evidence_wall_s,
        expires_mono_s=SystemClock().monotonic() + 60.0,
    )


@pytest.fixture
def system_gate(authority, checker, config) -> PermitGate:
    return PermitGate(authority, checker, config.permits, SystemClock())


def test_on_the_system_clock_every_permit_on_the_same_evidence_is_taken(
    system_gate, authority, checker
):
    evidence = SystemClock().wall() - 0.1
    for through_index in range(REPEATS):
        permit = system_permit(authority, evidence, through_index)
        assert system_gate.update(permit) is None
        assert system_gate.covers(MOVE, through_index) is None, through_index
        assert system_gate.covers(MOVE, through_index + 1) == (
            f"permit reaches setpoint {through_index}, not {through_index + 1}"
        )
    assert len(checker.fed) == REPEATS


def test_on_the_system_clock_a_cancel_on_the_same_evidence_always_ends_the_cover(
    system_gate, authority
):
    # The agent cancels a move with a permit through setpoint -1 that quotes
    # the evidence of the permit before it.
    for _ in range(REPEATS):
        evidence = SystemClock().wall() - 0.1
        assert system_gate.update(system_permit(authority, evidence, 25)) is None
        assert system_gate.covers(MOVE, 3) is None
        assert system_gate.update(system_permit(authority, evidence, -1)) is None
        assert system_gate.covers(MOVE, 3) == "permit reaches setpoint -1, not 3"
        system_gate.clear()


def test_on_the_system_clock_older_evidence_is_still_ignored(system_gate, authority):
    evidence = SystemClock().wall() - 0.1
    assert system_gate.update(system_permit(authority, evidence, 10)) is None
    for _ in range(REPEATS):
        older = system_permit(authority, evidence - 1e-6, 20)
        assert system_gate.update(older) is None
    assert system_gate.covers(MOVE, 11) == "permit reaches setpoint 10, not 11"


def test_arrived_is_set_by_a_stored_permit_only(gate, authority, clock):
    assert not gate.arrived.is_set()
    update(gate, clock, make_permit(authority, clock, expires_in_s=0.0))
    assert not gate.arrived.is_set()
    update(gate, clock, make_permit(authority, clock))
    assert gate.arrived.is_set()
    gate.arrived.clear()
    update(gate, clock, make_permit(authority, clock, evidence_age_s=1.0))  # ignored
    assert not gate.arrived.is_set()
    update(gate, clock, make_permit(authority, clock, move_id="move-2"))
    assert gate.arrived.is_set()


def test_a_permit_never_covers_before_the_watchdog_is_fed(authority, config, clock):
    feeding, go_on = threading.Event(), threading.Event()

    class SlowChecker(FakeChecker):
        def feed_watchdog(self, t_mono: float | None = None) -> None:
            feeding.set()
            assert go_on.wait(WAIT_S)
            super().feed_watchdog(t_mono)

    checker = SlowChecker()
    gate = PermitGate(authority, checker, config.permits, clock)
    permit = make_permit(authority, clock)
    answers: list[tuple[str | None, int]] = []
    updater = threading.Thread(target=lambda: update(gate, clock, permit))
    asker = threading.Thread(
        target=lambda: answers.append((covers(gate, clock, 0), len(checker.fed)))
    )
    updater.start()
    assert feeding.wait(WAIT_S)
    asker.start()
    asker.join(0.005)
    # While the watchdog is being fed the permit must not cover yet.
    assert all(answer is not None for answer, _ in answers)
    go_on.set()
    updater.join(WAIT_S)
    asker.join(WAIT_S)
    assert not updater.is_alive() and not asker.is_alive()
    assert all(answer is not None or fed == 1 for answer, fed in answers)
    assert covers(gate, clock, 0) is None
    assert len(checker.fed) == 1


def test_the_same_permit_twice_is_accepted_twice(gate, authority, checker, clock):
    permit = make_permit(authority, clock)
    assert update(gate, clock, permit) is None
    assert update(gate, clock, permit) is None
    assert covers(gate, clock, 10) is None
    assert checker.fed == [clock.monotonic()] * 2


def test_a_permit_for_another_move_replaces_the_held_one(
    gate, authority, checker, clock
):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    other = make_permit(
        authority, clock, move_id="move-2", through_index=3, evidence_age_s=1.0
    )
    assert update(gate, clock, other) is None
    assert covers(gate, clock, 0) == NO_PERMIT
    assert covers(gate, clock, 3, move_id="move-2") is None
    assert covers(gate, clock, 4, move_id="move-2") is not None
    assert checker.fed == [clock.monotonic(), clock.monotonic() - 1.0]


def test_expiry_passing_ends_the_cover(gate, authority, clock):
    update(gate, clock, make_permit(authority, clock, expires_in_s=1.0))
    assert covers(gate, clock, 4) is None
    clock.advance(0.5)
    assert covers(gate, clock, 4) is None
    clock.advance(0.5)
    assert covers(gate, clock, 4) == EXPIRED
    clock.advance(5.0)
    assert covers(gate, clock, 4) == EXPIRED


def test_evidence_growing_too_old_during_a_move_ends_the_cover(gate, authority, clock):
    update(gate, clock, make_permit(authority, clock, expires_in_s=60.0))
    clock.advance(2.75)
    assert covers(gate, clock, 4) is None
    clock.advance(0.25)
    assert covers(gate, clock, 4) == TOO_OLD
    # A permit that rests on fresh evidence covers the move again.
    update(gate, clock, make_permit(authority, clock, expires_in_s=60.0))
    assert covers(gate, clock, 4) is None


def test_evidence_already_old_on_arrival_runs_out_sooner(gate, authority, clock):
    permit = make_permit(authority, clock, evidence_age_s=2.5, expires_in_s=60.0)
    update(gate, clock, permit)
    clock.advance(0.25)
    assert covers(gate, clock, 4) is None
    clock.advance(0.25)
    assert covers(gate, clock, 4) == TOO_OLD


def test_a_stop_ends_the_cover(gate, authority, clock):
    update(gate, clock, make_permit(authority, clock))
    authority.raise_stop("typed stop", "console")
    assert (
        covers(gate, clock, 0) == "permit: stop number 0 does not match the current 1"
    )
    # Also once the agent has the arm again for a new task.
    authority.set_owner(Owner.AGENT, "new task")
    assert covers(gate, clock, 0) is not None


def test_a_fault_stop_ends_the_cover(gate, authority, clock):
    update(gate, clock, make_permit(authority, clock))
    authority.raise_fault("motor fault")
    assert covers(gate, clock, 0) == "permit: fault-stop latched: motor fault"


def test_a_change_of_control_owner_ends_the_cover(gate, authority, clock):
    update(gate, clock, make_permit(authority, clock))
    authority.set_owner(Owner.NOBODY, "help needed")
    assert covers(gate, clock, 0) == "permit: control owner is nobody, not agent"
    authority.set_owner(Owner.AGENT, "new task")
    assert (
        covers(gate, clock, 0) == "permit: owner number 1 does not match the current 3"
    )


def test_clear_forgets_the_held_permit(gate, authority, clock):
    update(gate, clock, make_permit(authority, clock, through_index=10))
    gate.clear()
    assert covers(gate, clock, 0) == NO_PERMIT
    # A permit that reaches less far than the forgotten one is taken as new.
    assert update(gate, clock, make_permit(authority, clock, through_index=2)) is None
    assert covers(gate, clock, 2) is None
    assert covers(gate, clock, 3) is not None


def test_clear_with_nothing_held_does_nothing(gate, clock):
    gate.clear()
    assert covers(gate, clock, 0) == NO_PERMIT


# The last refusal, for the move that then finds no permit


def refused(authority, clock, **fields) -> Permit:
    """A permit that update() refuses as expired, changed as the test asks."""
    permit = make_permit(authority, clock, expires_in_s=0.0)
    return dataclasses.replace(permit, **fields)


def test_no_refusal_is_remembered_before_one_happens(gate, authority, clock):
    assert gate.last_refusal(MOVE) is None
    assert update(gate, clock, make_permit(authority, clock)) is None
    assert gate.last_refusal(MOVE) is None


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"expires_in_s": 0.0}, EXPIRED),
        ({"evidence_age_s": 3.5}, TOO_OLD),
        ({"evidence_age_s": -0.0625}, "permit evidence is in the future"),
    ],
)
def test_a_refused_permit_is_remembered_for_its_move(
    gate, authority, clock, fields, why
):
    assert update(gate, clock, make_permit(authority, clock, **fields)) == why
    assert gate.last_refusal(MOVE) == why
    # What covers() says is unchanged, and no other move is told of it.
    assert covers(gate, clock, 0) == NO_PERMIT
    assert gate.last_refusal("move-2") is None


def test_a_permit_that_quotes_other_values_is_remembered_too(gate, authority, clock):
    permit = dataclasses.replace(make_permit(authority, clock), owner_number=7)
    why = "permit: owner number 7 does not match the current 1"
    assert update(gate, clock, permit) == why
    assert gate.last_refusal(MOVE) == why


def test_only_the_newest_refusal_is_remembered(gate, authority, clock):
    assert update(gate, clock, refused(authority, clock)) == EXPIRED
    late = make_permit(authority, clock, evidence_age_s=3.5)
    assert update(gate, clock, late) == TOO_OLD
    assert gate.last_refusal(MOVE) == TOO_OLD
    # A refused permit of another move takes its place.
    assert update(gate, clock, refused(authority, clock, move_id="move-2")) == EXPIRED
    assert gate.last_refusal(MOVE) is None
    assert gate.last_refusal("move-2") == EXPIRED


def test_a_permit_that_is_taken_ends_its_moves_refusal(gate, authority, clock):
    assert update(gate, clock, refused(authority, clock)) == EXPIRED
    assert update(gate, clock, make_permit(authority, clock)) is None
    assert gate.last_refusal(MOVE) is None
    assert covers(gate, clock, 0) is None


def test_a_permit_taken_for_another_move_leaves_the_refusal(gate, authority, clock):
    assert update(gate, clock, refused(authority, clock)) == EXPIRED
    other = make_permit(authority, clock, move_id="move-2")
    assert update(gate, clock, other) is None
    assert gate.last_refusal(MOVE) == EXPIRED


def test_a_refusal_after_a_permit_was_taken_stands_beside_it(gate, authority, clock):
    assert update(gate, clock, make_permit(authority, clock, through_index=4)) is None
    assert update(gate, clock, refused(authority, clock)) == EXPIRED
    # The held permit still covers; the refusal says why no newer one came.
    assert covers(gate, clock, 4) is None
    assert covers(gate, clock, 5) == "permit reaches setpoint 4, not 5"
    assert gate.last_refusal(MOVE) == EXPIRED


def test_a_refused_permit_does_not_feed_the_watchdog_or_wake_the_move(
    gate, authority, checker, clock
):
    assert update(gate, clock, refused(authority, clock)) == EXPIRED
    assert checker.fed == []
    assert not gate.arrived.is_set()


def test_clear_forgets_the_refusal(gate, authority, clock):
    assert update(gate, clock, refused(authority, clock)) == EXPIRED
    gate.clear()
    assert gate.last_refusal(MOVE) is None


# What ROBOT_STATE shows of the held permit


def test_robot_state_shows_the_held_permit_and_its_evidence_age(gate, authority, clock):
    assert gate.held_facts(clock.monotonic()) is None
    permit = make_permit(
        authority, clock, through_index=40, evidence_age_s=0.5, expires_in_s=1.0
    )
    assert update(gate, clock, permit) is None
    assert gate.held_facts(clock.monotonic()) == {
        "move_id": MOVE,
        "through_index": 40,
        "evidence_age_s": 0.5,
        "expires_in_s": 1.0,
    }
    # Both run on: the age up, the life down and below 0 once it has expired,
    # the same clock on which covers() refuses the setpoint.
    clock.advance(0.25)
    facts = gate.held_facts(clock.monotonic())
    assert (facts["evidence_age_s"], facts["expires_in_s"]) == (0.75, 0.75)
    clock.advance(1.0)
    facts = gate.held_facts(clock.monotonic())
    assert (facts["evidence_age_s"], facts["expires_in_s"]) == (1.75, -0.25)
    assert covers(gate, clock, 0) == EXPIRED


def test_the_held_facts_follow_a_cancel_and_a_clear(gate, authority, clock):
    first = make_permit(authority, clock, through_index=40, evidence_age_s=0.5)
    assert update(gate, clock, first) is None
    # A cancel lowers the reach and leaves the evidence and the expiry.
    cancel = dataclasses.replace(first, through_index=-1)
    assert update(gate, clock, cancel) is None
    facts = gate.held_facts(clock.monotonic())
    assert (facts["through_index"], facts["evidence_age_s"]) == (-1, 0.5)
    # Evidence a little ahead of the clock is shown as 0 old, never as negative.
    ahead = make_permit(authority, clock, move_id="move-2", evidence_age_s=-0.03125)
    assert update(gate, clock, ahead) is None
    facts = gate.held_facts(clock.monotonic())
    assert (facts["move_id"], facts["evidence_age_s"]) == ("move-2", 0.0)
    gate.clear()
    assert gate.held_facts(clock.monotonic()) is None


def test_a_refused_permit_leaves_the_held_facts(gate, authority, clock):
    assert update(gate, clock, make_permit(authority, clock, through_index=10)) is None
    before = gate.held_facts(clock.monotonic())
    assert update(gate, clock, refused(authority, clock)) == EXPIRED
    assert gate.held_facts(clock.monotonic()) == before
    assert gate.last_refusal(MOVE) == EXPIRED


def test_updates_and_checks_from_two_threads(gate, authority, checker, clock):
    last = 400
    ready = threading.Barrier(2)
    done = threading.Event()
    errors: list[BaseException] = []
    lost_cover: list[int] = []

    def send_permits():
        try:
            ready.wait()
            for through_index in range(last + 1):
                permit = make_permit(authority, clock, through_index=through_index)
                assert update(gate, clock, permit) is None
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    def check_cover():
        try:
            ready.wait()
            reach = -1
            while not done.is_set():
                # The clock stands still, so cover once given is never taken back.
                if reach >= 0 and covers(gate, clock, reach) is not None:
                    lost_cover.append(reach)
                if covers(gate, clock, reach + 1) is None:
                    reach += 1
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=send_permits),
        threading.Thread(target=check_cover),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert lost_cover == []
    assert covers(gate, clock, last) is None
    assert covers(gate, clock, last + 1) is not None
    assert len(checker.fed) == last + 1


# A setpoint without cover waits in a hold only while a later permit can still
# bring it cover (PermitGate.may_renew).


def test_a_late_permit_can_still_come_and_a_cancelled_or_stale_one_cannot(
    gate, authority, clock
):
    assert gate.may_renew(MOVE) is False  # no permit for the move yet
    assert update(gate, clock, make_permit(authority, clock, through_index=10)) is None
    assert covers(gate, clock, 11) == "permit reaches setpoint 10, not 11"
    assert gate.may_renew(MOVE) is True  # the next permit is only late
    assert gate.may_renew("another move") is False
    # An expired permit can be followed by a fresh one.
    clock.advance(2.0)
    assert covers(gate, clock, 5) == EXPIRED
    assert gate.may_renew(MOVE) is True
    # A cancel takes the cover back for good.
    held = make_permit(authority, clock, through_index=10)
    assert update(gate, clock, held) is None
    assert update(gate, clock, make_permit(authority, clock, through_index=-1)) is None
    assert gate.may_renew(MOVE) is False
    # So does a STOP: the held permit quotes the old stop number.
    assert update(gate, clock, make_permit(authority, clock, through_index=10)) is None
    assert gate.may_renew(MOVE) is True
    authority.raise_stop("stop", "test")
    assert gate.may_renew(MOVE) is False
