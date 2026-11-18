# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""human.jaw_release: how the jaws open when a person eases the trigger.

A trigger that is let go all the way asks for fully open jaws. With
"open_fully" the gripper worker then runs the driver's own opening in one go,
the routine of the typed open, and a trigger eased part of the way gets one
guarded opening to where it asks, at gripper.follow_open_effort; a trigger
that asks for less ends either. With "steps" every opening goes in steps of
gripper.open_step. On the simulated arm, on APAI's mock arm, and on the driver
class of the rig over a stand-in jaw motor with friction: one that moves only
while the pull on it, stiffness times the distance to its target, is above
the friction. The words of test_human_jaws.py apply.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.robot_app.config import (
    JAW_RELEASE_OPEN_FULLY,
    JAW_RELEASE_STEPS,
    JAW_RELEASES,
    RobotAppConfig,
)
from isaaccapture_examples.agentic_remote.robot_app.gripper import (
    CLOSE_UNTIL_CONTACT,
    NO_CLOSING,
    NO_OPENING,
    OPEN_EFFORT,
    OPENING_STALLED,
    PLAIN,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    GripperOutcome,
    Owner,
    StopLevel,
)

from conftest import AppHarness
from test_human_jaws import (
    LIGHT,
    WAIT_S,
    Trigger,
    agent_leaves_the_jaws,
    agents_opening_runs_out_of_time,
    commands,
    holding_the_agents_object,
    jaws,
    rig_person,
    settle,
    view_goes_stale,
)
from test_takeover import (
    feed,
    final,
    hand_back,
    human_config,
    state_now,
    take_over,
)

# The object a person closes the rig's jaws on, as a share of the jaw travel.
OBJECT = 0.10
# The friction of the stand-in jaw motor, in pulls of one step of open_step.
FRICTION_IN_STEPS = 1.2


@pytest.fixture
def jaw_release(request: pytest.FixtureRequest) -> str:
    """human.jaw_release of the robot app under test: "open_fully" unless marked."""
    return getattr(request, "param", JAW_RELEASE_OPEN_FULLY)


IN_STEPS = pytest.mark.parametrize("jaw_release", [JAW_RELEASE_STEPS], indirect=True)
WITH_EITHER = pytest.mark.parametrize("jaw_release", JAW_RELEASES, indirect=True)


@pytest.fixture
def config(jaw_release: str) -> RobotAppConfig:
    """The demo values with human driving switched on and this jaw_release."""
    return human_config(RobotAppConfig.default(), jaw_release=jaw_release)


@pytest.fixture
def human_app(make_robot_app) -> AppHarness:
    """A started robot app at rest: the jaws referenced, fully open and empty."""
    return make_robot_app()


def grip_on_the_simulated_arm(h, width: float) -> Trigger:
    """A person owns the arm and has closed the jaws on an object this wide."""
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    assert jaws(h)["matched"]
    h.arm.jaw_object(width)
    trigger.pull(0.0)
    assert h.arm.gripper_read() == width and h.arm.jaw_stiffness == LIGHT
    return trigger


# On the simulated arm


def test_a_trigger_that_is_let_go_gets_the_one_command_of_the_typed_open(human_app):
    h = human_app
    # The typed open, while nobody owns the arm: the driver's opening.
    assert final(h, h.client.open_jaws()).status == "completed"
    (typed,) = h.arm.jaw_commands
    trigger = grip_on_the_simulated_arm(h, 0.53)
    sent = len(h.arm.jaw_commands)
    trigger.pull(1.0)
    (opened,) = h.arm.jaw_commands[sent:]
    asked = (opened.opening, opened.mode, opened.status)
    assert asked == (typed.opening, typed.mode, typed.status)
    # The simulated arm has no opening of its own: the effort is the file's.
    assert opened.max_effort == h.config.gripper.follow_open_effort
    assert h.arm.gripper_read() == 1.0 and h.arm.gripper_last_target() == 1.0
    facts = jaws(h)
    assert (facts["matched"], facts["opened"], facts["refusal"]) == (True, True, "")
    state = state_now(h).gripper
    assert state.load == "empty" and not state.may_be_holding
    # The trigger stays let go: the jaws are sent nothing more.
    trigger.pull(1.0)
    assert len(h.arm.jaw_commands) == sent + 1
    done = hand_back(h)
    assert done.status == "completed", done.reason


@IN_STEPS
def test_with_steps_a_trigger_that_is_let_go_opens_the_jaws_in_steps(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent = len(h.arm.jaw_commands)
    trigger.pull(1.0)
    step = h.config.gripper.open_step
    opened = h.arm.jaw_commands[sent:]
    assert len(opened) == round(0.5 / step)
    assert [c.opening for c in opened] == pytest.approx(
        [0.5 + step * count for count in range(1, len(opened) + 1)]
    )
    assert {(c.mode, c.status) for c in opened} == {(PLAIN, "reached")}
    assert h.arm.gripper_read() == 1.0 and jaws(h)["opened"]


def test_a_trigger_eased_part_of_the_way_gets_one_opening_to_where_it_asks(
    human_app,
):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent = len(h.arm.jaw_commands)
    trigger.pull(0.8)
    (opened,) = h.arm.jaw_commands[sent:]
    assert (opened.opening, opened.mode, opened.status) == (0.8, PLAIN, "reached")
    assert opened.max_effort == h.config.gripper.follow_open_effort
    assert h.arm.gripper_read() == pytest.approx(0.8)
    # Part open is not fully open: the jaws still count as possibly holding.
    assert state_now(h).gripper.may_be_holding
    # Let go from there, without a pull between: the rest in one go.
    trigger.pull(1.0)
    assert commands(h, sent + 1) == [(1.0, PLAIN, "reached")]
    assert not state_now(h).gripper.may_be_holding


@IN_STEPS
def test_with_steps_a_trigger_eased_part_of_the_way_opens_the_jaws_in_steps(
    human_app,
):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent = len(h.arm.jaw_commands)
    step = h.config.gripper.open_step
    trigger.pull(0.5 + 2 * step)
    assert commands(h, sent) == [
        (pytest.approx(0.5 + step), PLAIN, "reached"),
        (pytest.approx(0.5 + 2 * step), PLAIN, "reached"),
    ]
    assert {c.max_effort for c in h.arm.jaw_commands[sent:]} == {OPEN_EFFORT}


@pytest.mark.parametrize("share", [0.5, 1.0])
def test_the_effort_of_an_opening_is_the_files(make_robot_app, config, share: float):
    effort = share * config.gripper.follow_open_effort
    gripper = dataclasses.replace(config.gripper, follow_open_effort=effort)
    h = make_robot_app(config=dataclasses.replace(config, gripper=gripper))
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent = len(h.arm.jaw_commands)
    trigger.pull(0.8)
    trigger.pull(1.0)
    assert [c.max_effort for c in h.arm.jaw_commands[sent:]] == [effort, effort]
    # A close keeps the effort it had.
    trigger.pull(0.6)
    assert h.arm.jaw_commands[-1].max_effort == OPEN_EFFORT


def test_jaws_short_of_a_grip_level_of_fully_open_open_in_one_go_at_the_match(
    human_app,
):
    h = human_app
    agents_opening_runs_out_of_time(h, 1.0, at=0.7)
    sent = len(h.arm.jaw_commands)
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    facts = jaws(h)
    assert (facts["matched"], facts["grip_level"], facts["opened"]) == (True, 1.0, True)
    assert commands(h, sent) == [(1.0, PLAIN, "reached")]
    assert h.arm.gripper_read() == 1.0


def as_the_jaws_open_past(h, opening: float, act: Callable[[], None]) -> list[float]:
    """Call act() once, as a jaw command brings the jaws to `opening` or above.

    Returns the list that takes every opening the jaws pass from now on.
    """
    passed: list[float] = []
    acted: list[bool] = []

    def on_step(at: float) -> None:
        passed.append(at)
        if not acted and at >= opening - 1e-9:
            acted.append(True)
            act()

    h.arm.on_jaw_step = on_step
    return passed


@pytest.mark.parametrize("view", [True, False])
def test_pulling_the_trigger_ends_the_opening_and_the_jaws_follow_it(
    human_app, view: bool
):
    h = human_app
    trigger = holding_the_agents_object(h)  # 0.5 wide, the jaw target closed
    if not view:
        view_goes_stale(h, trigger)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    # Let go, and pulled back to 0.6 as the jaws pass 0.7.
    passed = as_the_jaws_open_past(h, 0.7, lambda: setattr(trigger, "opening", 0.6))
    trigger.pull(1.0)
    trigger.pull(0.6)
    cut = h.arm.jaw_commands[sent]
    assert (cut.opening, cut.mode, cut.status) == (1.0, PLAIN, "cancelled")
    # The opening went no further than the jaw steps the pull took to arrive.
    assert max(passed) < 0.7 + 3 * h.arm.jaw_speed * h.arm.jaw_step_s
    assert len(h.arm.light_holds) == held + 1
    facts = jaws(h)
    assert facts["matched"] and not facts["opened"]
    if view:
        # Held lightly where it was cut, then closed to where the trigger asks.
        assert commands(h, sent + 1) == [(0.6, CLOSE_UNTIL_CONTACT, "reached")]
        assert h.arm.gripper_read() == pytest.approx(0.6)
        assert facts["refusal"] == ""
    else:
        # Without a fresh live view no close follows: the jaws stay where the
        # opening was cut, held lightly.
        assert len(h.arm.jaw_commands) == sent + 1
        assert h.arm.gripper_read() == pytest.approx(max(passed))
        assert h.arm.jaw_stiffness == LIGHT
        assert facts["refusal"].startswith(NO_CLOSING)


def test_a_trigger_that_stays_let_go_does_not_end_the_opening(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    # Within reach_tol of fully open is no pull: the dead band of the trigger.
    nearly = 1.0 - h.config.gripper.reach_tol / 2
    as_the_jaws_open_past(h, 0.7, lambda: setattr(trigger, "opening", nearly))
    trigger.pull(1.0)
    assert commands(h, sent) == [(1.0, PLAIN, "reached")]
    assert h.arm.gripper_read() == 1.0 and len(h.arm.light_holds) == held


@pytest.mark.parametrize("view", [True, False])
def test_pulling_the_trigger_ends_an_opening_to_a_part_eased_trigger_too(
    human_app, view: bool
):
    h = human_app
    trigger = holding_the_agents_object(h)  # 0.5 wide, the jaw target closed
    if not view:
        view_goes_stale(h, trigger)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    # Eased to 0.9, and pulled back to 0.55 as the jaws pass 0.7.
    passed = as_the_jaws_open_past(h, 0.7, lambda: setattr(trigger, "opening", 0.55))
    trigger.pull(0.9)
    trigger.pull(0.55)
    cut = h.arm.jaw_commands[sent]
    assert (cut.opening, cut.mode, cut.status) == (0.9, PLAIN, "cancelled")
    assert max(passed) < 0.7 + 3 * h.arm.jaw_speed * h.arm.jaw_step_s
    assert len(h.arm.light_holds) == held + 1
    if view:
        assert commands(h, sent + 1) == [(0.55, CLOSE_UNTIL_CONTACT, "reached")]
        assert h.arm.gripper_read() == pytest.approx(0.55)
        assert jaws(h)["refusal"] == ""
    else:
        assert len(h.arm.jaw_commands) == sent + 1
        assert h.arm.gripper_read() == pytest.approx(max(passed))
        assert h.arm.jaw_stiffness == LIGHT
        assert jaws(h)["refusal"].startswith(NO_CLOSING)


def test_a_trigger_eased_further_does_not_end_the_opening(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    # Eased to 0.7, and let go all the way as the jaws pass 0.6.
    as_the_jaws_open_past(h, 0.6, lambda: setattr(trigger, "opening", 1.0))
    trigger.pull(0.7)
    trigger.pull(1.0)
    assert commands(h, sent) == [(0.7, PLAIN, "reached"), (1.0, PLAIN, "reached")]
    assert h.arm.gripper_read() == 1.0 and len(h.arm.light_holds) == held


def test_a_human_target_without_a_jaw_opening_ends_the_opening(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    # The trigger is not usable from the moment the jaws pass 0.7.
    passed = as_the_jaws_open_past(h, 0.7, lambda: setattr(trigger, "opening", None))
    trigger.pull(1.0)
    settle(h)
    assert commands(h, sent) == [(1.0, PLAIN, "cancelled")]
    assert h.arm.gripper_read() == pytest.approx(max(passed))
    assert max(passed) < 0.7 + 3 * h.arm.jaw_speed * h.arm.jaw_step_s
    assert h.arm.jaw_stiffness == LIGHT and len(h.arm.light_holds) == held + 1
    facts = jaws(h)
    assert (facts["matched"], facts["refusal"]) == (False, NO_OPENING)


def test_stale_controller_data_end_the_opening(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent = len(h.arm.jaw_commands)
    reached, cancels = threading.Event(), []
    real = h.arm.gripper_command

    def gripper_command(target, max_effort, mode, cancel):
        cancels.append(cancel)
        return real(target, max_effort, mode, cancel)

    def wait_at_0_7(at: float) -> None:
        # The opening stands at 0.7 until something ends it.
        if not reached.is_set() and at >= 0.7 - 1e-9:
            reached.set()
            cancels[-1].wait(WAIT_S)

    h.arm.gripper_command = gripper_command
    h.arm.on_jaw_step = wait_at_0_7
    trigger.fed = False  # the human program goes silent
    trigger.opening = 1.0
    feed(h, trigger.message())
    assert reached.wait(WAIT_S)
    h.advance(2 * h.config.human.sample_age_limit_s)
    settle(h)
    assert commands(h, sent) == [(1.0, PLAIN, "cancelled")]
    assert h.arm.gripper_read() == pytest.approx(0.7)
    assert h.arm.jaw_stiffness == LIGHT
    facts = jaws(h)
    assert not facts["matched"]
    assert facts["refusal"].startswith("no fresh controller data")


@pytest.mark.parametrize("sender", ["controller", "console"])
def test_a_stop_ends_the_opening_with_the_jaws_held_lightly(human_app, sender: str):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent = len(h.arm.jaw_commands)

    def stop() -> None:
        h.client.stop(reason="right B", sender=sender)
        wait_until(lambda: h.app.snapshot().stop_number > 0)

    passed = as_the_jaws_open_past(h, 0.7, stop)
    trigger.pull(1.0)
    h.client.wait_state(lambda s: s.stop_number == 1 and s.owner == "nobody")
    settle(h)
    assert commands(h, sent) == [(1.0, PLAIN, "cancelled")]
    # Held lightly where the STOP found them: part open, not opened further.
    assert h.arm.gripper_read() == pytest.approx(max(passed))
    assert max(passed) < 0.7 + 3 * h.arm.jaw_speed * h.arm.jaw_step_s
    assert h.arm.jaw_stiffness == LIGHT and not jaws(h)["matched"]
    # Nobody owns the arm: the trigger, still let go, moves nothing.
    trigger.pull(1.0)
    assert len(h.arm.jaw_commands) == sent + 1


def wait_until(done: Callable[[], bool]) -> None:
    """Wait in real time, on the thread that moves the jaws, until done()."""
    waited = time.monotonic() + WAIT_S
    while not done() and time.monotonic() < waited:
        time.sleep(0.001)


def test_a_fault_stop_ends_the_opening(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent = len(h.arm.jaw_commands)

    def fault() -> None:
        h.arm.set_fault("motor 3 over temperature")
        wait_until(lambda: h.app.snapshot().stop_level is StopLevel.FAULT)

    passed = as_the_jaws_open_past(h, 0.7, fault)
    trigger.pull(1.0)
    settle(h)
    assert h.app.snapshot().stop_level is StopLevel.FAULT
    assert commands(h, sent) == [(1.0, PLAIN, "cancelled")]
    assert h.arm.gripper_read() == pytest.approx(max(passed))
    assert max(passed) < 0.7 + 3 * h.arm.jaw_speed * h.arm.jaw_step_s
    assert h.arm.jaw_stiffness == LIGHT


def test_a_lost_robot_connection_ends_the_opening(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)

    def lose_the_connection() -> None:
        h.robot_end.break_link()
        wait_until(lambda: len(h.arm.light_holds) > held)

    passed = as_the_jaws_open_past(h, 0.7, lose_the_connection)
    trigger.fed = False  # nothing can be fed over a lost connection
    trigger.opening = 1.0
    feed(h, trigger.message())
    wait_until(lambda: bool(passed))
    settle(h)
    assert commands(h, sent) == [(1.0, PLAIN, "cancelled")]
    assert h.arm.gripper_read() == pytest.approx(max(passed))
    assert max(passed) < 0.7 + 3 * h.arm.jaw_speed * h.arm.jaw_step_s
    assert h.arm.jaw_stiffness == LIGHT
    assert not h.app._jaw_follower.facts()["matched"]


def test_an_opening_admitted_for_an_earlier_owner_number_sends_the_jaws_nothing(
    human_app,
):
    h = human_app
    grip_on_the_simulated_arm(h, 0.5)
    sent = len(h.arm.jaw_commands)
    now = h.app.snapshot()
    numbers = (now.owner_number, now.stop_number)
    read = h.arm.gripper_read

    def read_as_the_owner_changes() -> float | None:
        # The gripper worker reads the jaws before the step: the person's
        # ownership ends between the admission and the arm command.
        h.arm.gripper_read = read
        changed = h.app._authority.set_owner_if(
            Owner.NOBODY, "a test", stop_number=numbers[1], owner_number=numbers[0]
        )
        assert changed is not None
        return read()

    h.arm.gripper_read = read_as_the_owner_changes
    ended: list[GripperOutcome] = []
    assert h.app._worker.follow(1.0, *numbers, ended.append) is None
    settle(h)
    assert [end.status for end in ended] == ["cancelled"]
    assert "the control owner changed" in ended[0].detail
    assert len(h.arm.jaw_commands) == sent and h.arm.gripper_read() == 0.5
    # The numbers of the person who owned the arm open nothing either.
    refused = h.app._worker.follow(1.0, *numbers, ended.append)
    assert refused is not None and refused.startswith("control owner is nobody")
    assert len(h.arm.jaw_commands) == sent


def test_the_let_go_trigger_of_a_person_it_has_not_matched_opens_nothing(human_app):
    h = human_app
    agent_leaves_the_jaws(h, 0.0, holding=0.5)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    take_over(h)
    trigger = Trigger(h)
    for _ in range(3):
        trigger.pull(1.0)
    assert not jaws(h)["matched"]
    assert len(h.arm.jaw_commands) == sent and len(h.arm.light_holds) == held
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)


def test_an_opening_in_one_go_that_is_blocked_ends_in_the_light_hold(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    h.arm.jaw_block(0.7)  # something keeps the jaws from opening further
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    trigger.pull(1.0)
    assert commands(h, sent) == [(1.0, PLAIN, "stalled")]
    assert h.arm.jaw_stiffness == LIGHT and len(h.arm.light_holds) == held + 1
    told = f"{OPENING_STALLED}0.70"
    facts = jaws(h)
    assert (facts["matched"], facts["refusal"]) == (True, told)
    # The trigger stays let go: the block is not tried again.
    trigger.pull(1.0)
    assert len(h.arm.jaw_commands) == sent + 1 and jaws(h)["refusal"] == told
    # The way is free: pulled a little and let go, the jaws open.
    h.arm.jaw_block(None)
    trigger.pull(0.9)
    trigger.pull(1.0)
    assert h.arm.gripper_read() == 1.0 and jaws(h)["refusal"] == ""


def test_after_a_stalled_opening_letting_go_opens_the_jaws_without_a_pull(human_app):
    h = human_app
    trigger = grip_on_the_simulated_arm(h, 0.5)
    h.arm.jaw_block(0.6)  # something keeps the jaws from opening further
    sent = len(h.arm.jaw_commands)
    trigger.pull(0.8)
    assert commands(h, sent) == [(0.8, PLAIN, "stalled")]
    assert jaws(h)["refusal"] == f"{OPENING_STALLED}0.60"
    assert h.arm.jaw_stiffness == LIGHT
    # The way is free. Let go the rest of the way: no pull comes first.
    h.arm.jaw_block(None)
    trigger.pull(1.0)
    assert commands(h, sent + 1) == [(1.0, PLAIN, "reached")]
    assert h.arm.gripper_read() == 1.0
    facts = jaws(h)
    assert (facts["opened"], facts["refusal"]) == (True, "")


def simulated_closes(h) -> list[tuple[float, str, float, str, str]]:
    """The jaw commands of a person's closes with the trigger, on the simulated arm."""
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    for opening in (0.6, 0.3):
        trigger.pull(opening)
    h.arm.jaw_object(0.2)
    trigger.pull(0.0)
    trigger.pull(0.1)  # eased, tighter than the object is wide
    trigger.pull(0.0)
    return [
        (c.opening, c.mode, c.max_effort, c.status, c.stiffness)
        for c in h.arm.jaw_commands
    ]


@WITH_EITHER
def test_a_close_is_the_same_jaw_commands_with_either_jaw_release(human_app):
    h = human_app
    assert simulated_closes(h) == [
        (0.6, CLOSE_UNTIL_CONTACT, OPEN_EFFORT, "reached", "closing"),
        (0.3, CLOSE_UNTIL_CONTACT, OPEN_EFFORT, "reached", "closing"),
        (0.0, CLOSE_UNTIL_CONTACT, OPEN_EFFORT, "stalled", LIGHT),
    ]
    assert len(h.arm.light_holds) == 1


# On APAI's mock arm


@pytest.mark.apai
def test_on_apais_mock_arm_a_trigger_that_is_let_go_is_one_command_to_open(clock):
    import test_apai_app as apai

    profile = apai.apai_backend.load_apai_profile("mock")
    config = human_config(apai.make_config(profile), jaw_release=JAW_RELEASE_OPEN_FULLY)
    h = apai.ApaiHarness(config, profile, clock)
    h.start()
    try:
        h.hold_at(apai.HOME)
        take_over(h)
        trigger = Trigger(h)
        # The reference at start left the jaws closed: that is the grip level.
        trigger.pull(0.0)
        assert jaws(h)["matched"]
        sent = len(h.spy.jaw_commands)
        trigger.pull(1.0)
        # The mock counts the jaw position from 0 open. Its driver has no
        # opening of its own: the effort is the file's.
        effort = config.gripper.follow_open_effort
        assert h.spy.jaw_commands[sent:] == [(pytest.approx(0.0), effort)]
        assert h.arm.gripper_read() == pytest.approx(1.0) and jaws(h)["opened"]
        # Closed again and eased part of the way: one command to where it asks.
        trigger.pull(0.0)
        sent = len(h.spy.jaw_commands)
        trigger.pull(0.6)
        assert h.spy.jaw_commands[sent:] == [(pytest.approx(0.4), effort)]
    finally:
        h.close()
    apai.check_driver_use(h)


# On the driver class of the rig, with friction in the jaws


@pytest.fixture
def rig(clock, monkeypatch, tmp_path, jaw_release: str) -> Iterator[tuple[Any, Any]]:
    """rig_person() of test_human_jaws.py with this jaw_release."""
    with rig_person(clock, monkeypatch, tmp_path, jaw_release) as started:
        yield started


def travel_of(h) -> float:
    """The jaw travel of the rig's profile, rad."""
    return float(h.arm._jaws.travel_rad)


def step_pull(h) -> float:
    """The most one step of open_step pulls on the jaws: stiffness times rad."""
    stiffness = h.arm.backend._grip_kp * OPEN_EFFORT
    return stiffness * h.config.gripper.open_step * travel_of(h)


def give_friction(jaw: Any, friction: float) -> None:
    """Make the stand-in jaw motor move only while the pull on it is above this.

    It stops where the pull, stiffness times the distance to the target, has
    fallen to the friction.
    """

    def send_mit(q: Any, kp: Any = None, kd: Any = None, tau: Any = None) -> None:
        target, stiffness = float(np.ravel(q)[0]), float(np.ravel(kp)[0])
        jaw.sent.append(([target], [stiffness]))
        motor = jaw._mm["jaw"]
        pull = stiffness * (target - motor.value)
        if abs(pull) <= friction:
            return
        to = target - math.copysign(friction / stiffness, pull)
        if jaw.blocked_at is not None:
            to = min(to, jaw.blocked_at)
        motor.value = max(to, jaw.stop)

    jaw.send_mit = send_mit


def grip_on_the_rig(h, sdk) -> tuple[Trigger, list[Any], list[Any]]:
    """A person owns the rig's arm and has closed its jaws on an object.

    The jaws have friction above the pull of one step of open_step. Returns
    the trigger, the motor commands of the typed open that came first and
    those of the person's close.
    """
    jaw = sdk.gripper
    give_friction(jaw, FRICTION_IN_STEPS * step_pull(h))
    mark = len(jaw.sent)
    assert final(h, h.client.open_jaws()).status == "completed"
    typed = list(jaw.sent[mark:])
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    assert jaws(h)["matched"] and jaws(h)["grip_level"] == 1.0
    jaw.stop = OBJECT * travel_of(h)
    mark = len(jaw.sent)
    trigger.pull(0.0)
    close = list(jaw.sent[mark:])
    assert jaw.at() == pytest.approx(jaw.stop)
    state = state_now(h).gripper
    assert state.may_be_holding and state.load == "loaded"
    return trigger, typed, close


def fully_open(h) -> float:
    """The lowest reading that counts as fully open, as a share of the travel."""
    return 1.0 - h.config.gripper.full_open_tol


@pytest.mark.apai
def test_on_the_rigs_driver_class_letting_go_opens_jaws_that_grip_an_object(rig):
    import test_apai_backend as rs

    h, sdk = rig
    jaw = sdk.gripper
    said: list[str] = []
    h.app._jaw_follower._say = said.append
    trigger, typed, _ = grip_on_the_rig(h, sdk)
    mark = len(jaw.sent)
    trigger.pull(1.0)
    opened = jaw.sent[mark:]
    # The typed open's routine: its stiffness, to its target, and no light
    # hold after it.
    assert {kp[0] for _, kp in opened} == {kp[0] for _, kp in typed}
    assert opened[-1] == typed[-1]
    assert not jaw.light_after(mark)
    # One step of open_step could not have moved these jaws.
    assert max(kp[0] * (q[0] - jaw.stop) for q, kp in opened) > step_pull(h)
    assert jaw.at() / travel_of(h) >= fully_open(h)
    state = state_now(h).gripper
    assert state.opening >= fully_open(h)
    assert state.load == "empty" and not state.may_be_holding
    facts = jaws(h)
    assert (facts["matched"], facts["opened"], facts["refusal"]) == (True, True, "")
    assert not any("stalled" in line for line in said)
    # The trigger stays let go: the jaws are sent nothing more.
    mark = len(jaw.sent)
    for _ in range(20):
        trigger.pull(1.0)
    assert jaw.sent[mark:] == []
    # Pulled again, the close is the motor commands of the first close.
    trigger.pull(0.0)
    kp_close = h.arm.backend._grip_kp * OPEN_EFFORT
    assert jaw.sent[mark:] == [([0.0], [kp_close]), ([0.0], [rs.RS_HOLD_KP])]
    assert h.arm.torque_on() and sdk.disconnects == 0


@pytest.mark.apai
def test_on_the_rigs_driver_class_a_part_eased_trigger_opens_jaws_that_grip_an_object(
    rig,
):
    h, sdk = rig
    jaw = sdk.gripper
    trigger, _, _ = grip_on_the_rig(h, sdk)
    mark = len(jaw.sent)
    trigger.pull(0.5)
    opened = jaw.sent[mark:]
    # One guarded ramp of the driver's to where the trigger asks, at the
    # stiffness of gripper.follow_open_effort, and no light hold after it.
    stiffness = h.arm.backend._grip_kp * h.config.gripper.follow_open_effort
    assert [kp[0] for _, kp in opened] == [pytest.approx(stiffness)] * len(opened)
    targets = [q[0] for q, _ in opened]
    assert targets == sorted(targets)
    assert targets[-1] == pytest.approx(0.5 * travel_of(h))
    assert max(kp[0] * (q[0] - jaw.stop) for q, kp in opened) > step_pull(h)
    # The friction leaves the jaws short of it by less than the driver's
    # guarded opening may end short.
    gripper = h.config.gripper
    short = 0.5 - jaw.at() / travel_of(h)
    assert 0.0 < short <= max(gripper.reach_tol, gripper.full_open_tol)
    facts = jaws(h)
    assert (facts["matched"], facts["refusal"]) == (True, "")
    # Part open is not fully open: the jaws still count as possibly holding.
    assert state_now(h).gripper.may_be_holding
    # The trigger rests there: the jaws are sent nothing more.
    mark = len(jaw.sent)
    for _ in range(20):
        trigger.pull(0.5)
    assert jaw.sent[mark:] == []


@pytest.mark.apai
def test_on_the_rigs_driver_class_letting_go_opens_after_a_stalled_opening_too(rig):
    h, sdk = rig
    jaw = sdk.gripper
    trigger, _, _ = grip_on_the_rig(h, sdk)
    # Eased half way with something against the jaws: the opening stalls.
    jaw.blocked_at = 0.2 * travel_of(h)
    trigger.pull(0.5)
    assert jaw.at() == pytest.approx(jaw.blocked_at)
    assert jaws(h)["refusal"].startswith(OPENING_STALLED)
    # The way is free. Let go from there, without a pull between.
    jaw.blocked_at = None
    trigger.pull(1.0)
    assert jaw.at() / travel_of(h) >= fully_open(h)
    facts = jaws(h)
    assert (facts["opened"], facts["refusal"]) == (True, "")
    state = state_now(h).gripper
    assert state.load == "empty" and not state.may_be_holding


@pytest.mark.apai
@pytest.mark.parametrize("passes", [2, 4])
def test_on_the_rigs_driver_class_a_slip_of_the_trigger_is_taken_back(rig, passes: int):
    import apai.control.rs_gripper as rs_gripper

    h, sdk = rig
    jaw = sdk.gripper
    trigger, _, close = grip_on_the_rig(h, sdk)
    count, widest = [0], [jaw.at()]

    def pull_again() -> None:
        # Inside the guard's sleep of one pass, 0.05 s: after this many of
        # them the trigger is pulled all the way again.
        count[0] += 1
        widest[0] = max(widest[0], jaw.at())
        if count[0] == passes:
            trigger.opening = 0.0
            feed(h, trigger.message())

    sdk.guard_time.on_sleep = pull_again
    mark = len(jaw.sent)
    trigger.pull(1.0)
    sdk.guard_time.on_sleep = None
    trigger.pull(0.0)
    # The jaws opened by less than the guard's ramp ran in that time.
    ramp = rs_gripper.OPEN_SPEED * rs_gripper.STEP_S * passes
    assert widest[0] - jaw.stop < ramp
    assert max(q[0] for q, _ in jaw.sent[mark:]) - jaw.stop <= ramp + 1e-9
    # They are back on the object, by the motor commands of the first close.
    assert jaw.sent[-len(close) :] == close
    assert jaw.at() == pytest.approx(jaw.stop)
    state = state_now(h).gripper
    assert state.may_be_holding and state.load == "loaded"
    facts = jaws(h)
    assert (facts["matched"], facts["refusal"]) == (True, "")


@IN_STEPS
@pytest.mark.apai
def test_on_the_rigs_driver_class_steps_do_not_open_jaws_that_grip_an_object(rig):
    import test_apai_backend as rs

    h, sdk = rig
    jaw = sdk.gripper
    trigger, _, _ = grip_on_the_rig(h, sdk)
    mark = len(jaw.sent)
    trigger.pull(1.0)
    step = jaw.stop + h.config.gripper.open_step * travel_of(h)
    opened = jaw.sent[mark:]
    # One step of open_step from the reading, then the light hold there.
    assert max(q[0] for q, _ in opened) == pytest.approx(step)
    assert opened[-1] == ([pytest.approx(step)], [rs.RS_HOLD_KP])
    assert jaw.at() == pytest.approx(jaw.stop)
    assert jaws(h)["refusal"] == f"{OPENING_STALLED}{OBJECT:.2f}"
    mark = len(jaw.sent)
    for _ in range(20):
        trigger.pull(1.0)
    assert jaw.sent[mark:] == []


def rig_stops() -> dict[str, Callable[[Any], Any]]:
    """What ends a jaw command on the rig's driver class, by name."""
    import test_apai_backend as rs

    return {
        "right B": lambda h: h.client.stop(reason="right B", sender="controller"),
        "a STOP": rs.JAW_STOPS["a STOP"],
        "a fault-stop": rs.JAW_STOPS["a fault-stop"],
        "the robot connection lost": rs.JAW_STOPS["the robot connection lost"],
    }


@pytest.mark.apai
@pytest.mark.parametrize(
    "stopped_by", ["right B", "a STOP", "a fault-stop", "the robot connection lost"]
)
def test_on_the_rigs_driver_class_the_opening_ends_where_it_is_stopped(
    rig, stopped_by: str
):
    import test_apai_backend as rs

    h, sdk = rig
    jaw, marks = sdk.gripper, []
    trigger, typed, _ = grip_on_the_rig(h, sdk)
    part_open = jaw.stop + 0.2 * travel_of(h)

    def stop_part_open() -> None:
        # Inside the guard's sleep of one pass, the jaws part open.
        if marks or jaw.at() < part_open:
            return
        marks.append((len(jaw.sent), jaw.at()))
        rig_stops()[stopped_by](h)
        deadline = time.monotonic() + WAIT_S
        while not jaw.light_after(marks[0][0]) and time.monotonic() < deadline:
            time.sleep(0.001)

    sdk.guard_time.on_sleep = stop_part_open
    trigger.fed = False
    trigger.opening = 1.0
    feed(h, trigger.message())
    deadline = time.monotonic() + WAIT_S
    while not marks and time.monotonic() < deadline:
        time.sleep(0.001)
    assert marks, "the opening never got part open"
    settle(h)
    mark, at = marks[0]
    # Not one more command at the opening stiffness: every command since is
    # at the hold stiffness, the last where the jaws were, and none is aimed
    # past the ramp's last target, which ran ahead of the jaws.
    assert rs.targets_at(jaw, mark, typed[-1][1][0]) == []
    after = jaw.sent[mark:]
    assert after and all(kp == [rs.RS_HOLD_KP] for _, kp in after)
    assert after[-1] == ([pytest.approx(at)], [rs.RS_HOLD_KP])
    assert max(q[0] for q, _ in after) <= jaw.sent[mark - 1][0][0] + 1e-9
    assert jaw.at() == pytest.approx(at) and at / travel_of(h) < fully_open(h)
    assert not h.app._jaw_follower.facts()["matched"]
    assert h.arm.backend._stopped is False  # the driver was not stopped
    assert h.arm.torque_on() and sdk.disconnects == 0


@pytest.mark.apai
def test_on_the_rigs_driver_class_pulling_the_trigger_ends_the_opening(rig):
    import test_apai_backend as rs

    h, sdk = rig
    jaw, marks = sdk.gripper, []
    trigger, typed, close = grip_on_the_rig(h, sdk)
    part_open = jaw.stop + 0.2 * travel_of(h)

    def pull_part_open() -> None:
        # Inside the guard's sleep of one pass: the trigger is pulled all the way.
        if marks or jaw.at() < part_open:
            return
        marks.append((len(jaw.sent), jaw.at()))
        trigger.opening = 0.0
        feed(h, trigger.message())

    sdk.guard_time.on_sleep = pull_part_open
    trigger.pull(1.0)
    trigger.pull(0.0)
    assert marks, "the opening never got part open"
    mark, at = marks[0]
    # The opening ended at the guard's next pass: nothing more at its
    # stiffness, the jaws held lightly where they were.
    assert rs.targets_at(jaw, mark, typed[-1][1][0]) == []
    after = jaw.sent[mark:]
    held = after[: -len(close)]
    assert held and all(sent == ([pytest.approx(at)], [rs.RS_HOLD_KP]) for sent in held)
    # Then the close the trigger asks for: the motor commands of the first.
    assert after[-len(close) :] == close
    assert jaw.at() == pytest.approx(jaw.stop)
    state = state_now(h).gripper
    assert state.may_be_holding and state.load == "loaded"
    facts = jaws(h)
    assert (facts["matched"], facts["opened"], facts["refusal"]) == (True, True, "")
    now = h.app.snapshot()
    assert (now.owner, now.stop_level) == (Owner.HUMAN, StopLevel.NONE)


@pytest.mark.apai
def test_on_the_rigs_driver_class_a_blocked_opening_in_one_go_is_held_lightly(rig):
    import test_apai_backend as rs

    h, sdk = rig
    jaw = sdk.gripper
    trigger, _, _ = grip_on_the_rig(h, sdk)
    # Something keeps the jaws from opening past a quarter of their travel.
    jaw.blocked_at = 0.25 * travel_of(h)
    mark = len(jaw.sent)
    trigger.pull(1.0)
    assert jaw.at() == pytest.approx(jaw.blocked_at)
    # The driver's guard ended the opening; the jaws do not push on.
    assert jaw.sent[-1][1] == [rs.RS_HOLD_KP] and len(jaw.sent) > mark + 1
    facts = jaws(h)
    assert facts["matched"]
    assert facts["refusal"] == f"{OPENING_STALLED}{0.25:.2f}"
    now = h.app.snapshot()
    assert (now.owner, now.stop_level) == (Owner.HUMAN, StopLevel.NONE)
    # The trigger stays let go: the jaws are sent nothing more.
    mark = len(jaw.sent)
    trigger.pull(1.0)
    assert jaw.sent[mark:] == []


def rig_closes(h, sdk) -> list[tuple[list[float], list[float]]]:
    """The motor commands of a person's closes with the trigger, on the rig's driver."""
    jaw = sdk.gripper
    assert final(h, h.client.open_jaws()).status == "completed"
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    mark = len(jaw.sent)
    for opening in (0.6, 0.3):
        trigger.pull(opening)
    jaw.stop = OBJECT * travel_of(h)
    trigger.pull(0.0)
    trigger.pull(0.05)  # eased, tighter than the object is wide
    trigger.pull(0.0)
    return list(jaw.sent[mark:])


@pytest.mark.apai
def test_on_the_rigs_driver_class_a_close_is_the_same_motor_commands_with_either(
    clock, monkeypatch, tmp_path
):
    import test_apai_backend as rs

    sent = {}
    for jaw_release in JAW_RELEASES:
        folder = tmp_path / jaw_release
        folder.mkdir()
        with rig_person(clock, monkeypatch, folder, jaw_release) as (h, sdk):
            sent[jaw_release] = rig_closes(h, sdk)
            kp_close = h.arm.backend._grip_kp * OPEN_EFFORT
            travel = travel_of(h)
    assert sent[JAW_RELEASE_OPEN_FULLY] == sent[JAW_RELEASE_STEPS]
    assert sent[JAW_RELEASE_STEPS] == [
        ([pytest.approx(0.6 * travel)], [kp_close]),
        ([pytest.approx(0.3 * travel)], [kp_close]),
        ([0.0], [kp_close]),
        ([0.0], [rs.RS_HOLD_KP]),
    ]
