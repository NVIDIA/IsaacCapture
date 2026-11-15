# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the authority in
isaaccapture_examples.agentic_remote.robot_app.authority."""

from __future__ import annotations

import dataclasses
import logging
import threading
import uuid

import pytest
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    AuthorityState,
    Owner,
    StopLevel,
)

LOGGER = "isaaccapture.examples.agentic_remote.authority"


@pytest.fixture
def authority() -> Authority:
    return Authority()


@pytest.fixture
def owned(authority) -> Authority:
    """An authority whose control owner is the agent: owner number 1, stop number 0."""
    authority.set_owner(Owner.AGENT, "new task")
    return authority


def test_starts_with_nobody_no_stop_and_both_numbers_at_zero(authority):
    assert authority.snapshot() == AuthorityState(
        session_id=authority.session_id,
        owner=Owner.NOBODY,
        owner_number=0,
        stop_number=0,
        stop_level=StopLevel.NONE,
        fault_reason="",
    )


def test_session_id_is_uuid4_hex_and_differs_between_authorities():
    first, second = Authority(), Authority()
    parsed = uuid.UUID(hex=first.session_id)
    assert parsed.version == 4
    assert parsed.hex == first.session_id
    assert first.session_id != second.session_id


def test_session_id_never_changes(owned):
    session_id = owned.session_id
    owned.raise_stop("test", "console")
    owned.raise_fault("motor fault")
    owned.reset(owned.stop_number)
    assert owned.session_id == session_id
    assert owned.snapshot().session_id == session_id


def test_a_clock_may_be_given(clock):
    assert Authority(clock).owner is Owner.NOBODY
    assert Authority(clock=clock).owner_number == 0


def test_properties_read_the_same_values_as_the_snapshot(owned):
    owned.raise_fault("gripper fault")
    state = owned.snapshot()
    assert state.session_id == owned.session_id
    assert state.owner is owned.owner is Owner.NOBODY
    assert state.owner_number == owned.owner_number == 2
    assert state.stop_number == owned.stop_number == 1
    assert state.stop_level is owned.stop_level is StopLevel.FAULT
    assert state.fault_reason == owned.fault_reason == "gripper fault"


def test_snapshot_is_frozen(owned):
    state = owned.snapshot()
    assert isinstance(state, AuthorityState)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.owner = Owner.HUMAN
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.stop_number = 99
    assert owned.snapshot() == state


def test_snapshot_does_not_follow_later_changes(owned):
    before = owned.snapshot()
    owned.raise_fault("motor fault")
    assert before.owner is Owner.AGENT
    assert (before.owner_number, before.stop_number) == (1, 0)
    assert before.stop_level is StopLevel.NONE
    assert before.fault_reason == ""
    assert owned.snapshot() != before


def test_set_owner_returns_the_new_owner_number(authority):
    assert authority.set_owner(Owner.AGENT, "new task") == 1
    assert authority.set_owner(Owner.NOBODY, "help needed") == 2
    assert authority.set_owner(Owner.RECOVERY, "typed rest") == 3
    state = authority.snapshot()
    assert state.owner is Owner.RECOVERY
    assert state.owner_number == 3
    assert state.stop_number == 0


def test_set_owner_to_the_same_owner_still_raises_the_number(owned):
    assert owned.set_owner(Owner.AGENT, "again") == 2
    assert owned.admit(owned.session_id, 1, 0, Owner.AGENT) is not None


def test_set_owner_takes_the_owner_as_text(authority):
    authority.set_owner("human", "takeover")
    assert authority.owner is Owner.HUMAN


def test_set_owner_rejects_an_unknown_owner_and_changes_nothing(owned):
    before = owned.snapshot()
    with pytest.raises(ValueError, match=r"not a valid Owner"):
        owned.set_owner("operator", "typo")
    assert owned.snapshot() == before


def test_set_owner_if_changes_the_owner_while_both_numbers_stand(owned):
    number = owned.set_owner_if(
        Owner.RECOVERY, "return to rest", stop_number=0, owner_number=1
    )
    assert number == 2
    assert (owned.owner, owned.owner_number) == (Owner.RECOVERY, 2)


@pytest.mark.parametrize(
    "change",
    [
        lambda authority: authority.raise_stop("typed stop", "console"),
        lambda authority: authority.raise_fault("motor fault"),
        lambda authority: authority.set_owner(Owner.HUMAN, "takeover"),
    ],
    ids=["stop", "fault", "owner change"],
)
def test_set_owner_if_changes_nothing_once_a_number_has_moved(owned, change):
    seen = owned.snapshot()
    change(owned)
    before = owned.snapshot()
    answer = owned.set_owner_if(
        Owner.RECOVERY,
        "return to rest",
        stop_number=seen.stop_number,
        owner_number=seen.owner_number,
    )
    assert answer is None
    assert owned.snapshot() == before


@pytest.mark.parametrize(("stop_number", "owner_number"), [(1, 1), (0, 0), (0, 2)])
def test_set_owner_if_checks_each_number(owned, stop_number, owner_number):
    answer = owned.set_owner_if(
        Owner.NOBODY, "ended", stop_number=stop_number, owner_number=owner_number
    )
    assert answer is None
    assert (owned.owner, owned.owner_number) == (Owner.AGENT, 1)


def test_set_owner_if_ends_a_stop_level_like_set_owner(owned):
    owned.raise_stop("typed stop", "console")
    owned.set_owner_if("agent", "new task", stop_number=1, owner_number=2)
    assert (owned.owner, owned.stop_level) == (Owner.AGENT, StopLevel.NONE)


def test_set_owner_if_rejects_an_unknown_owner_and_changes_nothing(owned):
    before = owned.snapshot()
    with pytest.raises(ValueError, match=r"not a valid Owner"):
        owned.set_owner_if("operator", "typo", stop_number=0, owner_number=1)
    assert owned.snapshot() == before


def test_a_stop_racing_set_owner_if_never_leaves_an_owner(authority):
    # Whichever comes first, a STOP is never followed by an owner it did not see.
    for _ in range(200):
        seen = authority.snapshot()
        ready = threading.Barrier(2)
        taken: list[int | None] = []

        def take(seen=seen, ready=ready, taken=taken) -> None:
            ready.wait()
            taken.append(
                authority.set_owner_if(
                    Owner.RECOVERY,
                    "return to rest",
                    stop_number=seen.stop_number,
                    owner_number=seen.owner_number,
                )
            )

        def stop(ready=ready) -> None:
            ready.wait()
            authority.raise_stop("typed stop", "console")

        threads = [threading.Thread(target=take), threading.Thread(target=stop)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert authority.owner is Owner.NOBODY
        assert authority.stop_level is StopLevel.STOP
        assert authority.stop_number == seen.stop_number + 1
        # The owner number shows whether recovery had the arm before the STOP.
        took = taken[0] is not None
        assert authority.owner_number == seen.owner_number + (2 if took else 1)


def test_admit_accepts_the_current_values(owned):
    assert owned.admit(owned.session_id, 1, 0, Owner.AGENT) is None
    assert owned.admit(owned.session_id, 1, 0, "agent") is None


def test_admit_reports_the_first_failing_check_in_order(authority):
    authority.set_owner(Owner.AGENT, "new task")
    assert authority.raise_fault("motor 3 fault") == 1
    session_id = authority.session_id

    # Every value is wrong: the session id is reported.
    assert authority.admit("other", 0, 0, Owner.AGENT) == "session id does not match"
    # Then the latched fault-stop, ahead of the old stop number.
    assert (
        authority.admit(session_id, 0, 0, Owner.AGENT)
        == "fault-stop latched: motor 3 fault"
    )
    assert authority.reset(1) is None
    # Then the stop number, ahead of the control owner.
    assert (
        authority.admit(session_id, 0, 0, Owner.AGENT)
        == "stop number 0 does not match the current 1"
    )
    # Then the control owner, ahead of the owner number.
    assert (
        authority.admit(session_id, 0, 1, Owner.AGENT)
        == "control owner is nobody, not agent"
    )
    assert authority.set_owner(Owner.AGENT, "new task") == 3
    # The owner number comes last.
    assert (
        authority.admit(session_id, 0, 1, Owner.AGENT)
        == "owner number 0 does not match the current 3"
    )
    assert authority.admit(session_id, 3, 1, Owner.AGENT) is None


@pytest.mark.parametrize("session_id", ["", "0" * 32, None, 7])
def test_admit_refuses_another_session_id(owned, session_id):
    assert owned.admit(session_id, 1, 0, Owner.AGENT) == "session id does not match"


def test_admit_refuses_the_session_id_of_another_authority(owned):
    other = Authority()
    other.set_owner(Owner.AGENT, "new task")
    assert owned.admit(other.session_id, 1, 0, Owner.AGENT) is not None


def test_admit_refuses_everything_while_a_fault_is_latched(owned):
    owned.raise_fault("joint feedback lost")
    state = owned.snapshot()
    for required in (Owner.NOBODY, Owner.AGENT, None):
        reason = owned.admit(
            state.session_id, state.owner_number, state.stop_number, required
        )
        assert reason == "fault-stop latched: joint feedback lost"


def test_admit_names_a_fault_without_a_reason(owned):
    owned.raise_fault("")
    assert owned.admit(owned.session_id, 2, 1, Owner.NOBODY) == "fault-stop latched"


@pytest.mark.parametrize("stop_number", [-1, 1, 2, None])
def test_admit_refuses_a_stop_number_that_is_not_current(owned, stop_number):
    reason = owned.admit(owned.session_id, 1, stop_number, Owner.AGENT)
    assert reason == f"stop number {stop_number} does not match the current 0"


@pytest.mark.parametrize(
    ("owner", "required"),
    [
        (Owner.AGENT, Owner.HUMAN),
        (Owner.AGENT, Owner.NOBODY),
        (Owner.HUMAN, Owner.AGENT),
        (Owner.RECOVERY, Owner.AGENT),
        (Owner.NOBODY, Owner.RECOVERY),
    ],
)
def test_admit_refuses_a_sender_that_is_not_the_control_owner(
    authority, owner, required
):
    number = authority.set_owner(owner, "test")
    reason = authority.admit(authority.session_id, number, 0, required)
    assert reason == f"control owner is {owner.value}, not {required.value}"


@pytest.mark.parametrize("owner_number", [0, 2, -1])
def test_admit_refuses_an_owner_number_that_is_not_current(owned, owner_number):
    reason = owned.admit(owned.session_id, owner_number, 0, Owner.AGENT)
    assert reason == f"owner number {owner_number} does not match the current 1"


def test_admit_can_require_that_nobody_owns_the_arm(owned):
    assert owned.admit(owned.session_id, 1, 0, Owner.NOBODY) is not None
    number = owned.set_owner(Owner.NOBODY, "task done")
    assert owned.admit(owned.session_id, number, 0, Owner.NOBODY) is None


def test_admit_without_an_owner_number_skips_only_that_check(owned):
    session_id = owned.session_id
    assert owned.admit(session_id, None, 0, Owner.AGENT) is None
    assert "control owner" in owned.admit(session_id, None, 0, Owner.HUMAN)
    assert "stop number" in owned.admit(session_id, None, 5, Owner.AGENT)


def test_admit_without_a_required_owner_skips_only_that_check(owned):
    session_id = owned.session_id
    assert owned.admit(session_id, 1, 0, None) is None
    assert owned.admit(session_id, None, 0, None) is None
    assert "owner number" in owned.admit(session_id, 0, 0, None)
    assert "stop number" in owned.admit(session_id, 1, 5, None)
    assert owned.admit("other", None, 0, None) == "session id does not match"


def test_raise_stop_ends_the_ownership_and_raises_both_numbers(owned):
    assert owned.raise_stop("typed stop", "console") == 1
    state = owned.snapshot()
    assert state.owner is Owner.NOBODY
    assert state.owner_number == 2
    assert state.stop_number == 1
    assert state.stop_level is StopLevel.STOP
    assert state.fault_reason == ""


def test_commands_quoting_the_values_before_a_stop_are_refused(owned):
    before = owned.snapshot()
    owned.raise_stop("right B", "controller")
    reason = owned.admit(
        before.session_id, before.owner_number, before.stop_number, Owner.AGENT
    )
    assert reason == "stop number 0 does not match the current 1"
    # Also after the agent has the arm again for a new task.
    number = owned.set_owner(Owner.AGENT, "new task")
    assert (
        owned.admit(before.session_id, before.owner_number, 1, Owner.AGENT)
        == "owner number 1 does not match the current 3"
    )
    assert owned.admit(before.session_id, number, 1, Owner.AGENT) is None


def test_a_stop_while_nobody_owns_the_arm_still_raises_both_numbers(authority):
    assert authority.raise_stop("typed stop", "thor_terminal") == 1
    assert authority.raise_stop("typed stop", "thor_terminal") == 2
    state = authority.snapshot()
    assert (state.owner_number, state.stop_number) == (2, 2)
    assert state.owner is Owner.NOBODY


def test_a_stop_is_not_latched(owned):
    owned.raise_stop("typed stop", "console")
    state = owned.snapshot()
    assert (
        owned.admit(
            state.session_id, state.owner_number, state.stop_number, Owner.NOBODY
        )
        is None
    )


def test_the_stop_level_ends_when_someone_takes_the_arm(owned):
    owned.raise_stop("typed stop", "console")
    owned.set_owner(Owner.NOBODY, "still nobody")
    assert owned.stop_level is StopLevel.STOP
    owned.set_owner(Owner.RECOVERY, "typed rest")
    assert owned.stop_level is StopLevel.NONE


def test_raise_fault_latches_and_ends_the_ownership(owned):
    assert owned.raise_fault("motor 2 over temperature") == 1
    state = owned.snapshot()
    assert state.owner is Owner.NOBODY
    assert state.owner_number == 2
    assert state.stop_number == 1
    assert state.stop_level is StopLevel.FAULT
    assert state.fault_reason == "motor 2 over temperature"


def test_a_stop_during_a_fault_keeps_the_fault(owned):
    owned.raise_fault("motor fault")
    assert owned.raise_stop("typed stop", "console") == 2
    state = owned.snapshot()
    assert state.stop_level is StopLevel.FAULT
    assert state.fault_reason == "motor fault"
    assert (state.owner_number, state.stop_number) == (3, 2)


def test_a_fault_after_a_stop_latches(owned):
    owned.raise_stop("typed stop", "console")
    assert owned.raise_fault("gripper fault") == 2
    assert owned.stop_level is StopLevel.FAULT
    assert owned.fault_reason == "gripper fault"


def test_a_second_fault_keeps_the_first_reason_and_raises_the_numbers(owned):
    owned.raise_fault("motor fault")
    assert owned.raise_fault("joint feedback lost") == 2
    state = owned.snapshot()
    assert state.fault_reason == "motor fault"
    assert state.stop_level is StopLevel.FAULT
    assert (state.owner_number, state.stop_number) == (3, 2)


def test_taking_the_arm_does_not_clear_a_fault(owned):
    owned.raise_fault("motor fault")
    number = owned.set_owner(Owner.AGENT, "new task")
    assert owned.stop_level is StopLevel.FAULT
    assert "fault-stop" in owned.admit(owned.session_id, number, 1, Owner.AGENT)


@pytest.mark.parametrize("stopped", [False, True])
def test_reset_is_refused_when_no_fault_is_latched(owned, stopped):
    if stopped:
        owned.raise_stop("typed stop", "console")
    before = owned.snapshot()
    assert owned.reset(before.stop_number) == "no fault-stop is latched"
    assert owned.snapshot() == before


@pytest.mark.parametrize("seen", [0, 2, -1, None])
def test_reset_is_refused_with_a_stop_number_that_is_not_current(owned, seen):
    owned.raise_fault("motor fault")
    before = owned.snapshot()
    assert owned.reset(seen) == f"stop number {seen} does not match the current 1"
    assert owned.snapshot() == before


def test_reset_clears_the_latch_and_nothing_else(owned):
    number = owned.raise_fault("motor fault")
    before = owned.snapshot()
    assert owned.reset(number) is None
    assert owned.snapshot() == dataclasses.replace(
        before, stop_level=StopLevel.NONE, fault_reason=""
    )
    assert owned.owner is Owner.NOBODY


def test_reset_works_once(owned):
    number = owned.raise_fault("motor fault")
    assert owned.reset(number) is None
    assert owned.reset(number) == "no fault-stop is latched"


def test_a_stop_after_a_fault_makes_the_reset_number_old(owned):
    seen = owned.raise_fault("motor fault")
    owned.raise_stop("typed stop", "console")
    assert owned.reset(seen) == "stop number 1 does not match the current 2"
    assert owned.reset(2) is None


def test_reset_leaves_nobody_as_owner_even_if_one_was_set_during_the_fault(owned):
    stop_number = owned.raise_fault("motor fault")
    owner_number = owned.set_owner(Owner.AGENT, "raced the fault")
    assert owned.reset(stop_number) is None
    state = owned.snapshot()
    assert state.owner is Owner.NOBODY
    assert state.owner_number == owner_number + 1
    assert (
        owned.admit(state.session_id, owner_number, stop_number, Owner.AGENT)
        is not None
    )


def test_after_a_reset_old_commands_stay_refused_and_new_ones_are_admitted(owned):
    before = owned.snapshot()
    stop_number = owned.raise_fault("motor fault")
    owned.reset(stop_number)
    assert (
        owned.admit(
            before.session_id, before.owner_number, before.stop_number, Owner.AGENT
        )
        is not None
    )
    number = owned.set_owner(Owner.AGENT, "new task")
    assert owned.admit(before.session_id, number, stop_number, Owner.AGENT) is None


def test_changes_are_logged_under_the_example_logger(owned, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        owned.raise_stop("typed stop", "console")
        owned.raise_fault("motor fault")
    assert {record.name for record in caplog.records} == {LOGGER}
    assert "STOP from console: typed stop; stop number 1" in caplog.messages
    assert "fault-stop: motor fault; stop number 2" in caplog.messages


def test_concurrent_stops_are_all_counted_and_never_seen_half_done(authority):
    stoppers, rounds = 6, 300
    ready = threading.Barrier(stoppers + 1)
    done = threading.Event()
    returned: list[int] = []
    half_done: list[AuthorityState] = []

    def stop_many():
        mine = []
        ready.wait()
        for _ in range(rounds):
            mine.append(authority.raise_stop("test", "console"))
        returned.extend(mine)

    def watch():
        ready.wait()
        while not done.is_set():
            state = authority.snapshot()
            # A STOP raises both numbers together.
            if state.owner_number != state.stop_number:
                half_done.append(state)

    watcher = threading.Thread(target=watch)
    threads = [threading.Thread(target=stop_many) for _ in range(stoppers)]
    watcher.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    done.set()
    watcher.join()

    total = stoppers * rounds
    assert half_done == []
    assert sorted(returned) == list(range(1, total + 1))
    assert authority.stop_number == authority.owner_number == total


def test_mixed_changes_from_many_threads_lose_no_count(authority):
    threads_count, rounds = 6, 200
    ready = threading.Barrier(threads_count)
    session_id = authority.session_id
    errors: list[BaseException] = []

    def change_many(index: int):
        try:
            ready.wait()
            for _ in range(rounds):
                authority.set_owner(Owner.AGENT, "new task")
                state = authority.snapshot()
                authority.admit(
                    session_id, state.owner_number, state.stop_number, Owner.AGENT
                )
                if index % 2:
                    authority.raise_stop("test", "agent")
                else:
                    authority.reset(authority.raise_fault("test"))
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=change_many, args=(index,))
        for index in range(threads_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    state = authority.snapshot()
    # Each round changes the owner once and raises one STOP or fault-stop; a reset
    # adds one more owner number only when it finds an owner to remove.
    assert state.stop_number == threads_count * rounds
    assert state.owner_number >= 2 * threads_count * rounds
    assert state.owner_number <= 3 * threads_count * rounds
