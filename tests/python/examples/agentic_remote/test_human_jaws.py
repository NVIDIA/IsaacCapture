# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The jaws under a person's trigger.

On the simulated arm, on APAI's mock arm and, for an opening that something
blocks, on the driver class of the rig over a stand-in motor SDK.

A takeover makes a person the control owner; the hand-back ends it at rest.
Every HUMAN_SETPOINT link message carries the jaw opening the trigger of the
person's controller asks for, 1.0 fully open to 0.0 closed. The grip level is
the jaw opening in force at the takeover. The trigger is ignored until the
opening it asks for has reached the grip level: it has matched. From then on
the jaws follow it. The light hold is the low jaw stiffness a close is left at
once the jaws have stalled on an object.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import HumanSetpoint
from isaaccapture_examples.agentic_remote.robot_app.app import (
    LINK_LOST,
    OPEN_FIRST,
    human_target,
)
from isaaccapture_examples.agentic_remote.robot_app.config import (
    RobotAppConfig,
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.gripper import (
    CLOSE_UNTIL_CONTACT,
    JAWS_BUSY,
    NO_CLOSING,
    NO_GRIP_LEVEL,
    NO_OPENING,
    NO_TIGHTER,
    NOT_MATCHED,
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
from test_takeover import (
    MOTION_WORKER,
    REST,
    RIGHT_A,
    Done,
    ask_for_help,
    drive,
    drives,
    fed_while_the_worker_reads,
    feed,
    final,
    hand_back,
    human_config,
    moved,
    person,
    setpoint,
    state_now,
    take_over,
)

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
# The name of the gripper worker's thread, which runs every jaw command.
GRIPPER_WORKER = "gripper-worker"
# The simulated arm's two jaw stiffness states.
LIGHT, CLOSING = "light", "closing"
JAWS_NOT_EMPTY = "the jaws are not proven empty"


@pytest.fixture
def config() -> RobotAppConfig:
    """The demo values with human driving switched on; the trigger moves the jaws."""
    config = human_config(RobotAppConfig.default())
    assert config.human.trigger_jaws and config.human.jaw_match_tol == 0.05
    return config


@pytest.fixture
def human_app(make_robot_app) -> AppHarness:
    """A started robot app at rest: the jaws referenced, fully open and empty."""
    return make_robot_app()


class Trigger:
    """The trigger of a person's controller: each HUMAN_SETPOINT asks for `opening`.

    A human program sends 50 HUMAN_SETPOINTs per second whatever the jaws do. A
    jaw command moves the fake clock, so the same HUMAN_SETPOINT is fed after
    each wait of the gripper worker: the controller data stay fresh.
    """

    def __init__(self, h, opening: Any = 1.0, **fields: Any) -> None:
        self.h = h
        self.opening = opening
        self.fields = fields  # of every HUMAN_SETPOINT it makes
        self.view = True  # whether a live VIEW_STATUS goes with each one
        self.fed = True  # False: nothing is fed while the jaws move
        sleep = h.clock.sleep

        def sleep_fed(seconds: float) -> None:
            sleep(seconds)
            on_worker = threading.current_thread().name == GRIPPER_WORKER
            if on_worker and self.fed and h.app.running:
                feed(h, self.message(), view=self.view)

        h.clock.sleep = sleep_fed

    def message(self) -> HumanSetpoint:
        return setpoint(self.h, jaw_opening=self.opening, **self.fields)

    def pull(self, opening: Any) -> None:
        """Ask for this opening; what the jaws do about it has ended on return."""
        self.opening = opening
        feed(self.h, self.message(), view=self.view)
        settle(self.h)


def settle(h) -> None:
    """Wait until no jaw command runs or waits."""
    assert h.app._worker.wait_idle(WAIT_S), "the jaws did not come to rest"


def jaws(h) -> dict[str, Any]:
    """health["human"]["jaws"] of the ROBOT_STATE the app would send now."""
    return state_now(h).health["human"]["jaws"]


def commands(h, since: int = 0) -> list[tuple[float, str, str]]:
    """(opening, mode, status) of the jaw commands the arm got from `since` on."""
    sent = h.arm.jaw_commands[since:]
    return [(pytest.approx(c.opening), c.mode, c.status) for c in sent]


def opening_steps(start: float, end: float) -> list[tuple[float, str, str]]:
    """The commands of a guarded opening from start to end, in steps of 0.05."""
    count = round((end - start) / 0.05)
    targets = [start + 0.05 * index for index in range(1, count + 1)]
    return [(pytest.approx(target), PLAIN, "reached") for target in targets]


def agent_leaves_the_jaws(h, opening: float, holding: float | None = None) -> None:
    """The agent sends the jaws to `opening` and then asks for help.

    holding puts an object between the jaws first: the close stalls on it at
    that opening and is left in the light hold. Nobody owns the arm on return.
    """
    h.new_task()
    if holding is not None:
        h.arm.jaw_object(holding)
    result = final(h, h.client.gripper(opening, mode=CLOSE_UNTIL_CONTACT))
    assert result.status == "completed", result.reason
    ask_for_help(h)


def agents_opening_runs_out_of_time(h, target: float, at: float) -> None:
    """The agent's opening from 0.5 toward `target` times out at `at`.

    The jaw target stays at `target`. Then the agent asks for help: nobody
    owns the arm on return.
    """
    h.new_task()
    done = final(h, h.client.gripper(0.5, mode=PLAIN))
    assert done.status == "completed", done.reason
    h.arm.jaw_timeout_s = at - 0.5  # the jaws travel 1.0 of their travel per second
    timed_out = final(h, h.client.gripper(target, mode=PLAIN))
    h.arm.jaw_timeout_s = 3.0
    assert timed_out.detail["outcome"] == "timeout"
    assert h.arm.gripper_read() == pytest.approx(at)
    assert h.arm.gripper_last_target() == target
    ask_for_help(h)


def as_command_begins(h, number: int, act: Callable[[], None]) -> None:
    """Call act() as the arm is given its jaw command `number`, counted from now."""
    real, given = h.arm.gripper_command, [0]

    def gripper_command(opening, max_effort, mode, cancel):
        given[0] += 1
        if given[0] == number:
            act()
        return real(opening, max_effort, mode, cancel)

    h.arm.gripper_command = gripper_command


def as_the_jaws_pass(h, opening: float, act: Callable[[], None]) -> list[float]:
    """Call act() once, as a jaw command brings the jaws to `opening` or below.

    Returns the list that takes every opening the jaws pass from now on.
    """
    passed: list[float] = []
    acted: list[bool] = []

    def on_step(at: float) -> None:
        passed.append(at)
        if not acted and at <= opening + 1e-9:
            acted.append(True)
            act()

    h.arm.on_jaw_step = on_step
    return passed


def view_goes_stale(h, trigger: Trigger) -> None:
    """The live view goes away; the controller data stay fresh."""
    trigger.view = False
    for _ in range(4):
        h.clock.advance(0.2)
        trigger.pull(trigger.opening)
    assert state_now(h).view_age_s > h.config.human.view_age_limit_s


# The takeover, and the trigger before it has matched


def test_a_held_object_stays_held_through_a_takeover_with_the_trigger_released(
    human_app,
):
    h = human_app
    agent_leaves_the_jaws(h, 0.0, holding=0.5)
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    ended = drives(h)
    # The trigger is released: every HUMAN_SETPOINT asks for fully open jaws.
    hand = person(h, jaw_opening=1.0)
    assert jaws(h) == {
        "matched": False,
        "grip_level": 0.0,  # the agent's last jaw target
        "opened": False,
        "asked": 1.0,
        "refusal": NOT_MATCHED,
    }
    # The person moves the arm with the grip squeezed: the jaws stay as they are.
    drive(h, hand, moved(REST), sends=4)
    assert [(end.status, end.sent) for end in ended] == [("released", 4)]
    feed(h, hand())
    assert len(h.arm.jaw_commands) == sent and len(h.arm.light_holds) == held
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)
    assert h.arm.gripper_last_target() == 0.0
    assert not jaws(h)["matched"]


def test_the_trigger_short_of_the_grip_level_moves_nothing(human_app):
    h = human_app
    agent_leaves_the_jaws(h, 0.0, holding=0.5)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    take_over(h)
    trigger = Trigger(h)
    # Pulled further and further, to just outside human.jaw_match_tol.
    for opening in (0.8, 0.4, 0.06):
        trigger.pull(opening)
        assert jaws(h)["refusal"] == NOT_MATCHED
        assert not jaws(h)["matched"] and jaws(h)["asked"] == opening
    assert len(h.arm.jaw_commands) == sent and len(h.arm.light_holds) == held
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)


def test_the_jaws_follow_the_trigger_once_it_reached_the_grip_level(human_app):
    h = human_app
    agent_leaves_the_jaws(h, 0.0, holding=0.5)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(0.5)
    assert not jaws(h)["matched"]
    # Within human.jaw_match_tol of the grip level: the trigger has matched.
    trigger.pull(0.05)
    assert jaws(h) == {
        "matched": True,
        "grip_level": 0.0,
        "opened": False,
        "asked": 0.05,
        "refusal": "",
    }
    # Eased, but still tighter than the object is wide: the light hold stands.
    trigger.pull(0.3)
    assert len(h.arm.jaw_commands) == sent and len(h.arm.light_holds) == held
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)
    # Wider than the object: the jaws open to it in the guarded opening's steps.
    trigger.pull(0.7)
    assert commands(h, sent) == opening_steps(0.5, 0.7)
    assert all(c.max_effort == OPEN_EFFORT for c in h.arm.jaw_commands[sent:])
    assert h.arm.gripper_read() == pytest.approx(0.7)
    # And they close with it again, above the grip level.
    opened = len(h.arm.jaw_commands)
    trigger.pull(0.6)
    assert commands(h, opened) == [(pytest.approx(0.6), CLOSE_UNTIL_CONTACT, "reached")]
    assert h.arm.gripper_read() == pytest.approx(0.6)
    # A trigger that moves less than the dead band asks nothing new of the jaws.
    trigger.pull(0.62)
    assert len(h.arm.jaw_commands) == opened + 1


def test_the_grip_squeezed_or_not_the_jaws_follow_the_trigger(human_app):
    h = human_app
    take_over(h)
    # A squeeze the person has not released since the takeover moves no joint.
    trigger = Trigger(h, engaged=True, grip_released=False, q=list(REST))
    trigger.pull(1.0)
    assert state_now(h).health["human"]["refusal"] != ""
    trigger.pull(0.5)
    assert commands(h) == [(pytest.approx(0.5), CLOSE_UNTIL_CONTACT, "reached")]
    assert h.arm.sent == []


def test_a_trigger_that_crossed_the_grip_level_has_reached_it(human_app):
    h = human_app
    agent_leaves_the_jaws(h, 0.3)  # no object: the jaws are at 0.3
    sent = len(h.arm.jaw_commands)
    take_over(h)
    trigger = Trigger(h)
    # Pulled past the grip level before the takeover, and held there.
    trigger.pull(0.1)
    trigger.pull(0.2)
    assert not jaws(h)["matched"] and jaws(h)["grip_level"] == 0.3
    assert len(h.arm.jaw_commands) == sent
    # Released in one go: from below the grip level to above it.
    trigger.pull(0.6)
    assert jaws(h)["matched"]
    assert commands(h, sent) == opening_steps(0.3, 0.6)
    assert h.arm.gripper_read() == pytest.approx(0.6)


# Jaws that an opening left short of the grip level


def test_jaws_short_of_a_grip_level_of_fully_open_open_once_the_trigger_matched(
    human_app,
):
    h = human_app
    agents_opening_runs_out_of_time(h, 1.0, at=0.7)
    sent = len(h.arm.jaw_commands)
    take_over(h)
    trigger = Trigger(h)
    # The released trigger asks for fully open, which is the grip level.
    trigger.pull(1.0)
    facts = jaws(h)
    assert (facts["matched"], facts["grip_level"], facts["opened"]) == (True, 1.0, True)
    assert commands(h, sent) == opening_steps(0.7, 1.0)
    assert h.arm.gripper_read() == 1.0
    # Measured fully open, the jaws close and open with the trigger.
    trigger.pull(0.5)
    assert h.arm.gripper_read() == pytest.approx(0.5)
    trigger.pull(1.0)
    done = hand_back(h)
    assert done.status == "completed", done.reason


def test_jaws_short_of_the_grip_level_open_to_it_and_no_wider_without_a_live_view(
    human_app,
):
    h = human_app
    # An earlier takeover ended with the jaws measured fully open.
    trigger = following_open_jaws(h)
    trigger.pull(0.9)
    trigger.pull(1.0)
    trigger.fed = False
    assert hand_back(h).status == "completed"
    agents_opening_runs_out_of_time(h, 0.8, at=0.6)
    sent = len(h.arm.jaw_commands)
    take_over(h)
    trigger.fed = True
    trigger.pull(0.3)  # not the grip level
    view_goes_stale(h, trigger)
    assert len(h.arm.jaw_commands) == sent
    trigger.pull(0.8)
    assert jaws(h)["matched"] and jaws(h)["grip_level"] == 0.8
    assert commands(h, sent) == opening_steps(0.6, 0.8)
    assert h.arm.gripper_read() == pytest.approx(0.8)


def test_an_opening_that_is_still_blocked_at_the_match_is_tried_once(human_app):
    h = human_app
    agents_opening_runs_out_of_time(h, 1.0, at=0.7)
    h.arm.jaw_block(0.7)  # something keeps the jaws from opening further
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    take_over(h)
    trigger = Trigger(h)
    for _ in range(3):
        trigger.pull(1.0)
    assert commands(h, sent) == [(pytest.approx(0.75), PLAIN, "stalled")]
    facts = jaws(h)
    assert (facts["matched"], facts["opened"]) == (True, False)
    assert facts["refusal"] == f"{OPENING_STALLED}0.70"
    assert h.arm.jaw_stiffness == LIGHT and len(h.arm.light_holds) == held + 1


# No closing beyond the grip level, and the close after a full open


def held_at_grip_level(h) -> Trigger:
    """A person owns the arm; the trigger has matched the agent's grip.

    The agent closed on an object 0.5 wide with the jaw target 0.3.
    """
    agent_leaves_the_jaws(h, 0.3, holding=0.5)
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(0.3)
    assert jaws(h)["matched"] and jaws(h)["grip_level"] == 0.3
    return trigger


def test_no_closing_beyond_the_grip_level_before_a_measured_full_open(human_app):
    h = human_app
    trigger = held_at_grip_level(h)
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    # Pulled all the way: tighter than the agent's grip.
    trigger.pull(0.0)
    facts = jaws(h)
    assert (facts["matched"], facts["opened"]) == (True, False)
    assert facts["refusal"] == NO_TIGHTER
    assert len(h.arm.jaw_commands) == sent and len(h.arm.light_holds) == held
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)
    # Part open is not fully open: pulled all the way again, the jaws close to
    # the grip level and no further.
    trigger.pull(0.8)
    assert h.arm.gripper_read() == pytest.approx(0.8)
    trigger.pull(0.0)
    last = h.arm.jaw_commands[-1]
    assert (last.opening, last.status, last.stiffness) == (0.3, "stalled", LIGHT)
    assert jaws(h)["refusal"] == NO_TIGHTER and not jaws(h)["opened"]
    assert min(c.opening for c in h.arm.jaw_commands[sent:]) == 0.3


def test_after_a_measured_full_open_a_close_ends_in_the_light_hold(human_app):
    h = human_app
    trigger = held_at_grip_level(h)
    trigger.pull(1.0)
    assert h.arm.gripper_read() == 1.0
    assert jaws(h)["opened"]
    assert not state_now(h).gripper.may_be_holding
    # The object is still between the jaws: the close stalls on it.
    trigger.pull(0.0)
    last = h.arm.jaw_commands[-1]
    assert (last.opening, last.mode, last.status) == (
        0.0,
        CLOSE_UNTIL_CONTACT,
        "stalled",
    )
    assert (last.stiffness, h.arm.jaw_stiffness) == (LIGHT, LIGHT)
    assert h.arm.gripper_read() == 0.5
    assert jaws(h)["refusal"] == ""
    state = state_now(h)
    assert state.gripper.may_be_holding and state.gripper.load == "loaded"
    # Eased a little, the jaws are sent nothing: no second squeeze.
    sent = len(h.arm.jaw_commands)
    trigger.pull(0.2)
    trigger.pull(0.0)
    assert len(h.arm.jaw_commands) == sent and h.arm.jaw_stiffness == LIGHT


def no_tighter_than_0_3_on_the_object(h, trigger: Trigger) -> None:
    """The trigger matches a hold with the jaw target 0.3; pulled on, nothing moves.

    An object 0.5 wide is between the jaws.
    """
    trigger.pull(0.3)
    facts = jaws(h)
    assert (facts["matched"], facts["grip_level"]) == (True, 0.3)
    assert not facts["opened"]
    sent = len(h.arm.jaw_commands)
    trigger.pull(0.0)
    assert jaws(h)["refusal"] == NO_TIGHTER
    assert len(h.arm.jaw_commands) == sent
    assert (h.arm.gripper_read(), h.arm.gripper_last_target()) == (0.5, 0.3)


def test_a_full_open_measured_in_an_earlier_takeover_does_not_count(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    assert jaws(h)["opened"]
    trigger.fed = False
    assert hand_back(h).status == "completed"
    agent_leaves_the_jaws(h, 0.3, holding=0.5)
    take_over(h)
    trigger.fed = True
    no_tighter_than_0_3_on_the_object(h, trigger)


def test_a_full_open_measured_before_a_stop_does_not_count_after_it(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    # The person closes on an object 0.5 wide, with the jaw target 0.3.
    h.arm.jaw_object(0.5)
    trigger.pull(0.3)
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)
    h.client.stop(reason="right B", sender="controller")
    h.client.wait_state(lambda s: s.stop_number == 1 and s.owner == "nobody")
    settle(h)
    take_over(h)
    no_tighter_than_0_3_on_the_object(h, trigger)


# A close under way, and a trigger that is let go


def test_a_close_under_way_ends_when_the_trigger_is_let_go(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    held = len(h.arm.light_holds)
    # Pulled all the way for one jaw step of 0.01 s, and let go.
    passed = as_the_jaws_pass(h, 0.99, lambda: setattr(trigger, "opening", 1.0))
    trigger.pull(0.0)
    trigger.pull(1.0)
    cut, opened = h.arm.jaw_commands
    assert (cut.opening, cut.mode, cut.status) == (
        0.0,
        CLOSE_UNTIL_CONTACT,
        "cancelled",
    )
    # The cut close was left in the light hold, as at a STOP.
    assert cut.stiffness == LIGHT and len(h.arm.light_holds) == held + 1
    assert min(passed) > 0.97
    assert (opened.opening, opened.mode, opened.status) == (1.0, PLAIN, "reached")
    assert h.arm.gripper_read() == 1.0
    assert jaws(h)["matched"] and not state_now(h).gripper.may_be_holding
    done = hand_back(h)
    assert done.status == "completed", done.reason


def test_a_close_under_way_ends_where_an_eased_trigger_asks(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    # Pulled all the way, and eased to half as the jaws pass 0.8.
    as_the_jaws_pass(h, 0.8, lambda: setattr(trigger, "opening", 0.5))
    trigger.pull(0.0)
    trigger.pull(0.5)
    assert commands(h) == [
        (0.0, CLOSE_UNTIL_CONTACT, "cancelled"),
        (0.5, CLOSE_UNTIL_CONTACT, "reached"),
    ]
    assert h.arm.gripper_read() == pytest.approx(0.5)
    assert h.arm.gripper_last_target() == 0.5
    assert jaws(h)["matched"] and jaws(h)["refusal"] == ""


def test_a_cut_close_leaves_the_jaws_aimed_where_the_trigger_asks_not_at_closed(
    human_app,
):
    h = human_app
    trigger = following_open_jaws(h)
    # Eased to where the jaws are as they pass 0.8.
    as_the_jaws_pass(h, 0.8, lambda: setattr(trigger, "opening", 0.8))
    trigger.pull(0.0)
    trigger.pull(0.8)
    assert commands(h) == [
        (0.0, CLOSE_UNTIL_CONTACT, "cancelled"),
        (0.8, PLAIN, "reached"),
    ]
    assert h.arm.gripper_last_target() == 0.8
    assert h.arm.gripper_read() == pytest.approx(0.8)


def test_after_a_cut_a_close_on_an_object_ends_in_the_light_hold_as_before(
    human_app,
):
    h = human_app
    trigger = following_open_jaws(h)
    h.arm.jaw_object(0.5)
    # Eased just before the jaws meet the object, to less than it is wide.
    as_the_jaws_pass(h, 0.52, lambda: setattr(trigger, "opening", 0.3))
    trigger.pull(0.0)
    trigger.pull(0.3)
    assert commands(h) == [
        (0.0, CLOSE_UNTIL_CONTACT, "cancelled"),
        (0.3, CLOSE_UNTIL_CONTACT, "stalled"),
    ]
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)
    assert state_now(h).gripper.may_be_holding
    # Eased again, still tighter than the object is wide: no second squeeze.
    trigger.pull(0.4)
    trigger.pull(0.3)
    assert len(h.arm.jaw_commands) == 2 and h.arm.jaw_stiffness == LIGHT


def test_a_stop_as_a_close_is_cut_ends_the_jaw_motion(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    cancels: list[threading.Event] = []
    real = h.arm.gripper_command

    def gripper_command(opening, max_effort, mode, cancel):
        cancels.append(cancel)
        return real(opening, max_effort, mode, cancel)

    def on_step(at: float) -> None:
        if at <= 0.9 + 1e-9:
            trigger.opening = 1.0  # let go: the next HUMAN_SETPOINT cuts the close
        if cancels[-1].is_set() and h.app.snapshot().stop_number == 0:
            # The close is cut and has not returned: the STOP comes now.
            h.client.stop(reason="right B", sender="controller")
            waited = time.monotonic() + WAIT_S
            while h.app.snapshot().stop_number == 0 and time.monotonic() < waited:
                time.sleep(0.001)

    h.arm.gripper_command = gripper_command
    h.arm.on_jaw_step = on_step
    trigger.pull(0.0)
    h.client.wait_state(lambda s: s.stop_number == 1 and s.owner == "nobody")
    settle(h)
    assert commands(h) == [(0.0, CLOSE_UNTIL_CONTACT, "cancelled")]
    assert h.arm.gripper_read() == pytest.approx(0.89)
    assert h.arm.jaw_stiffness == LIGHT and not jaws(h)["matched"]
    # At the next takeover the light hold of the cut close stands as after any
    # STOP: an eased trigger sends the jaws nothing.
    h.arm.on_jaw_step = None
    take_over(h)
    trigger.pull(0.0)
    assert jaws(h)["matched"] and jaws(h)["grip_level"] == 0.0
    trigger.pull(0.5)
    assert len(h.arm.jaw_commands) == 1 and h.arm.jaw_stiffness == LIGHT


def test_released_all_the_way_the_jaws_end_fully_open_from_any_opening(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    # The person closes on an object 0.53 wide: not a multiple of the step.
    h.arm.jaw_object(0.53)
    trigger.pull(0.0)
    assert h.arm.gripper_read() == 0.53
    trigger.pull(1.0)
    assert h.arm.gripper_read() == 1.0 and h.arm.gripper_last_target() == 1.0
    done = hand_back(h)
    assert done.status == "completed", done.reason


# What ends the jaw motion


def following_open_jaws(h) -> Trigger:
    """A person owns the arm at rest; the released trigger matches the open jaws."""
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    facts = jaws(h)
    assert (facts["matched"], facts["grip_level"], facts["opened"]) == (True, 1.0, True)
    assert h.arm.jaw_commands == []
    return trigger


def test_a_stop_in_the_middle_of_a_human_jaw_motion_holds_the_jaws_lightly(
    human_app,
):
    h = human_app
    trigger = following_open_jaws(h)
    reached = h.hold_jaws_at(0.9)
    held = len(h.arm.light_holds)
    # The trigger is pulled all the way; the close is under way at 0.9.
    trigger.opening = 0.0
    feed(h, trigger.message())
    assert reached.wait(WAIT_S)
    assert h.arm.jaw_stiffness == CLOSING
    h.client.stop(reason="right B", sender="controller")
    h.client.wait_state(lambda s: s.stop_number == 1 and s.owner == "nobody")
    settle(h)
    # Held lightly where they are: not opened, and closed no further.
    assert commands(h) == [(0.0, CLOSE_UNTIL_CONTACT, "cancelled")]
    assert h.arm.gripper_read() == pytest.approx(0.9)
    assert h.arm.jaw_stiffness == LIGHT and len(h.arm.light_holds) > held
    assert not jaws(h)["matched"]
    # The trigger stays pulled: nobody owns the arm, so it moves nothing.
    trigger.pull(0.0)
    trigger.pull(0.3)
    assert len(h.arm.jaw_commands) == 1
    # After the next takeover it must match again: the cut close left the jaw
    # target at closed.
    take_over(h)
    trigger.pull(0.5)
    facts = jaws(h)
    assert (facts["matched"], facts["grip_level"]) == (False, 0.0)
    assert len(h.arm.jaw_commands) == 1 and h.arm.gripper_read() == pytest.approx(0.9)
    trigger.pull(0.0)
    assert jaws(h)["matched"]
    trigger.pull(1.0)
    assert h.arm.gripper_read() == 1.0


def test_stale_controller_data_end_the_jaw_motion(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    held = len(h.arm.light_holds)
    # The human program goes silent for longer than human.sample_age_limit_s.
    h.advance(0.3)
    facts = jaws(h)
    assert not facts["matched"]
    assert facts["refusal"].startswith("no fresh controller data: 0.3")
    assert len(h.arm.light_holds) == held + 1 and h.arm.jaw_stiffness == LIGHT
    # It is back with the trigger pulled half way: that is not the grip level.
    trigger.pull(0.5)
    assert h.arm.jaw_commands == [] and jaws(h)["refusal"] == NOT_MATCHED
    trigger.pull(1.0)
    trigger.pull(0.5)
    assert commands(h) == [(0.5, CLOSE_UNTIL_CONTACT, "reached")]


def test_a_lost_robot_connection_ends_the_jaw_motion(human_app):
    h = human_app
    following_open_jaws(h)
    held = len(h.arm.light_holds)
    h.lose_link()
    facts = h.app._jaw_follower.facts()
    assert (facts["matched"], facts["refusal"]) == (False, LINK_LOST)
    assert len(h.arm.light_holds) > held and h.arm.jaw_stiffness == LIGHT
    # On the next connection the person still owns the arm, and the trigger
    # must match again.
    h.connect()
    assert h.app.snapshot().owner is Owner.HUMAN
    trigger = Trigger(h)
    trigger.pull(0.5)
    assert h.arm.jaw_commands == [] and not jaws(h)["matched"]


def test_a_drive_the_grip_did_not_end_ends_the_jaw_motion(human_app):
    h = human_app
    ended = drives(h)
    hand = person(h, jaw_opening=1.0)
    assert jaws(h)["matched"]
    # A released grip ends a drive and leaves the jaws following the trigger.
    drive(h, hand, moved(REST), sends=4)
    assert jaws(h)["matched"]
    held = len(h.arm.light_holds)
    # The human program ends the next engagement itself.
    hand.squeeze(moved(REST, 0.1))
    hand.fields.update(engaged=False, grip_released=False, note="the hand ran ahead")
    h.idle()
    assert [end.status for end in ended] == ["released", "disengaged"]
    facts = h.app._jaw_follower.facts()
    assert not facts["matched"]
    assert facts["refusal"] == "the human drive ended: the hand ran ahead"
    assert len(h.arm.light_holds) == held + 1 and h.arm.jaw_commands == []


@pytest.mark.parametrize("old", ["the owner number", "the stop number", "both", ""])
def test_a_human_target_that_quotes_an_earlier_takeovers_numbers_moves_no_jaw(
    human_app, old: str
):
    h = human_app
    take_over(h)
    first = h.app.snapshot()
    h.client.stop(reason="right B", sender="controller")
    h.client.wait_state(lambda s: s.stop_number == first.stop_number + 1)
    take_over(h)
    now = h.app.snapshot()
    assert now.owner is Owner.HUMAN
    assert now.owner_number != first.owner_number
    quoted: dict[str, int] = {}
    if old in ("the owner number", "both"):
        quoted["owner_number"] = first.owner_number
    if old in ("the stop number", "both"):
        quoted["stop_number"] = first.stop_number
    # A late HUMAN_SETPOINT of the first takeover: released, then pulled.
    for opening in (1.0, 0.5, 0.5, 0.4):
        feed(h, setpoint(h, jaw_opening=opening, **quoted))
        settle(h)
    if old:
        assert h.arm.jaw_commands == [] and h.arm.gripper_read() == 1.0
        assert not jaws(h)["matched"]
    else:
        # The same HUMAN_SETPOINTs with the numbers of now close the jaws.
        assert commands(h) == [
            (0.5, CLOSE_UNTIL_CONTACT, "reached"),
            (0.4, CLOSE_UNTIL_CONTACT, "reached"),
        ]


@pytest.mark.parametrize(
    ("driver", "source", "moves"),
    [
        ("real", "synthetic", False),
        ("", "synthetic", False),
        ("real", "teleop", True),
        ("sim", "synthetic", True),
    ],
)
def test_a_synthetic_controller_moves_the_jaws_of_a_simulated_arm_only(
    make_robot_app, driver: str, source: str, moves: bool
):
    # `driver` is what the robot app is told its driver is; the arm is simulated.
    h = make_robot_app(driver=driver)
    take_over(h)  # with the headset's controller
    trigger = Trigger(h, source=source)
    for opening in (1.0, 0.5, 0.5):
        trigger.pull(opening)
    if moves:
        assert commands(h) == [(0.5, CLOSE_UNTIL_CONTACT, "reached")]
    else:
        assert h.arm.jaw_commands == [] and h.arm.gripper_read() == 1.0
        facts = jaws(h)
        assert not facts["matched"]
        assert facts["refusal"].startswith("a synthetic controller may drive only")


def hold_the_motion_workers_joint_read(h) -> tuple[threading.Event, threading.Event]:
    """Hold the motion worker before its next read of all joints.

    A hand-back and a torque release read the joints once they have proven the
    jaws empty. Returns (reached, release).
    """
    reached, release = threading.Event(), threading.Event()
    h._releases.append(release)
    real = h.arm.read_all

    def read_all():
        on_worker = threading.current_thread().name == MOTION_WORKER
        if on_worker and not reached.is_set():
            reached.set()
            release.wait(WAIT_S)
        return real()

    h.arm.read_all = read_all
    return reached, release


@pytest.mark.parametrize("work", ["a hand-back", "a torque release"])
def test_the_trigger_moves_no_jaws_once_other_work_has_proven_them_empty(
    human_app, work: str
):
    h = human_app
    trigger = following_open_jaws(h)
    reached, release = hold_the_motion_workers_joint_read(h)
    released = Done()
    if work == "a hand-back":
        command_id = h.client.request_owner("handback", reason=RIGHT_A)
    else:
        assert h.app.request_release(released) is None
    assert reached.wait(WAIT_S), f"{work} did not come to its joint read"
    # The jaws are proven empty, and the person still owns the arm.
    assert h.app.snapshot().owner is Owner.HUMAN
    trigger.pull(0.5)
    assert h.arm.jaw_commands == [] and h.arm.gripper_read() == 1.0
    assert h.app._jaw_follower.facts()["refusal"] == f"busy: {work} is running"
    release.set()
    if work == "a hand-back":
        assert final(h, command_id).status == "completed"
        assert not h.app.takeover_open and h.app.snapshot().owner is Owner.NOBODY
        settle(h)
        assert h.arm.jaw_commands == [] and h.arm.gripper_read() == 1.0
        return
    assert released.wait().status == "completed"
    assert not h.arm.torque_on()
    settle(h)
    # The release itself closed the jaws before the motors went off; the
    # trigger's opening of 0.5 was never sent.
    assert commands(h) == [(0.0, CLOSE_UNTIL_CONTACT, "reached")]
    assert h.arm.gripper_read() == 0.0


def test_no_jaw_step_goes_out_while_a_hand_back_reads_the_jaws_and_the_joints(
    human_app,
):
    h = human_app
    following_open_jaws(h)
    held = len(h.arm.light_holds)
    # Pulled half way from the hand-back's first read on, 50 times per second.
    on = fed_while_the_worker_reads(h, lambda: setpoint(h, jaw_opening=0.5))
    on.append(True)
    done = hand_back(h)
    on.clear()
    settle(h)
    assert h.arm.jaw_commands == [] and h.arm.gripper_read() == 1.0
    assert done.status == "completed", done.reason
    assert h.app.snapshot().owner is Owner.NOBODY
    # The jaw motion ended in a light hold, as at a STOP.
    assert len(h.arm.light_holds) > held


def test_the_trigger_moves_no_jaws_while_a_jaw_reference_runs(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    # No request starts a jaw reference while a person owns the arm, so the
    # gripper worker's answer is made up here.
    h.app._worker.referencing = lambda: True
    trigger.pull(0.5)
    facts = jaws(h)
    assert (facts["matched"], facts["refusal"]) == (False, JAWS_BUSY)
    assert h.arm.jaw_commands == [] and h.arm.gripper_read() == 1.0
    del h.app._worker.referencing
    # With the worker's own answer the trigger has to match again, and does.
    trigger.pull(0.5)
    assert h.arm.jaw_commands == [] and jaws(h)["refusal"] == NOT_MATCHED
    trigger.pull(1.0)
    trigger.pull(0.5)
    assert commands(h) == [(0.5, CLOSE_UNTIL_CONTACT, "reached")]


def test_without_a_fresh_live_view_the_jaws_open_but_do_not_close(human_app):
    h = human_app
    assert h.config.human.require_live_view
    trigger = following_open_jaws(h)
    trigger.pull(0.6)
    assert h.arm.gripper_read() == pytest.approx(0.6)
    # The live view goes away; the controller data stay fresh.
    trigger.view = False
    for _ in range(4):
        h.clock.advance(0.2)
        trigger.pull(0.6)
    assert state_now(h).view_age_s > h.config.human.view_age_limit_s
    sent = len(h.arm.jaw_commands)
    trigger.pull(0.3)
    facts = jaws(h)
    assert facts["matched"]
    assert facts["refusal"].startswith(f"{NO_CLOSING}the live view is not fresh: ")
    assert len(h.arm.jaw_commands) == sent
    trigger.pull(0.9)
    assert commands(h, sent) == opening_steps(0.6, 0.9)
    assert jaws(h)["refusal"] == ""


def holding_the_agents_object(h) -> Trigger:
    """A person owns the arm; the trigger has matched the agent's hold.

    The agent closed on an object 0.5 wide with the jaw target 0.0, and the
    trigger is pulled all the way.
    """
    agent_leaves_the_jaws(h, 0.0, holding=0.5)
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(0.0)
    assert jaws(h)["matched"] and h.arm.gripper_read() == 0.5
    return trigger


@pytest.mark.parametrize("view", [True, False])
def test_an_opening_under_way_stops_where_the_trigger_is_pulled_back(
    human_app, view: bool
):
    h = human_app
    trigger = holding_the_agents_object(h)
    if not view:
        view_goes_stale(h, trigger)
    sent = len(h.arm.jaw_commands)
    # Let go, and pulled back to 0.6 as the second opening step begins.
    as_command_begins(h, 2, lambda: setattr(trigger, "opening", 0.6))
    trigger.pull(1.0)
    trigger.pull(0.6)
    assert commands(h, sent) == opening_steps(0.5, 0.6)
    assert h.arm.gripper_read() == pytest.approx(0.6)
    # The jaws are where the trigger asks: nothing holds them wider.
    facts = jaws(h)
    assert (facts["asked"], facts["refusal"]) == (0.6, "")


def test_without_a_fresh_live_view_no_close_follows_an_opening_step(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    trigger.pull(0.5)
    assert h.arm.gripper_read() == pytest.approx(0.5)
    view_goes_stale(h, trigger)
    sent = len(h.arm.jaw_commands)
    # Eased to 0.6 for one HUMAN_SETPOINT, and pulled back as the step begins.
    as_command_begins(h, 1, lambda: setattr(trigger, "opening", 0.5))
    trigger.pull(0.6)
    trigger.pull(0.5)
    assert commands(h, sent) == [(pytest.approx(0.55), PLAIN, "reached")]
    assert h.arm.gripper_read() == pytest.approx(0.55)
    assert jaws(h)["refusal"].startswith(f"{NO_CLOSING}the live view is not fresh: ")


def test_without_a_fresh_live_view_a_trigger_that_slipped_closes_nothing(human_app):
    h = human_app
    trigger = holding_the_agents_object(h)
    view_goes_stale(h, trigger)
    sent = len(h.arm.jaw_commands)

    def pulled_again() -> None:
        trigger.opening = 0.0
        h.arm.jaw_object(None)  # the object fell as the jaws let go

    # Let go for one HUMAN_SETPOINT, and pulled all the way again.
    as_command_begins(h, 1, pulled_again)
    trigger.pull(1.0)
    trigger.pull(0.0)
    assert commands(h, sent) == [(pytest.approx(0.55), PLAIN, "reached")]
    assert h.arm.gripper_read() == pytest.approx(0.55)
    assert jaws(h)["refusal"].startswith(NO_CLOSING)


def test_a_close_that_loses_the_live_view_ends_on_the_object_and_stays_there(
    human_app,
):
    h = human_app
    trigger = following_open_jaws(h)
    trigger.pull(0.7)
    h.arm.jaw_object(0.5)
    h.arm.jaw_speed = 0.1  # the jaws take 2 s from 0.7 to the object
    sent = len(h.arm.jaw_commands)
    # The live view goes away as the close begins; the trigger stays pulled.
    as_command_begins(h, 1, lambda: setattr(trigger, "view", False))
    trigger.pull(0.2)
    trigger.pull(0.2)
    assert state_now(h).view_age_s > h.config.human.view_age_limit_s
    assert commands(h, sent) == [(pytest.approx(0.2), CLOSE_UNTIL_CONTACT, "stalled")]
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)
    assert jaws(h)["refusal"].startswith(NO_CLOSING)


@pytest.mark.parametrize("may_close", [False, True])
def test_a_jaw_step_without_the_leave_to_close_sends_no_close(
    human_app, may_close: bool
):
    # The gripper worker reads the jaws anew before each step toward a trigger.
    h = human_app
    take_over(h)
    now = h.app.snapshot()
    ended: list[GripperOutcome] = []
    numbers = (now.owner_number, now.stop_number)
    refused = h.app._worker.follow(0.5, *numbers, ended.append, may_close=may_close)
    assert refused is None
    settle(h)
    if may_close:
        assert commands(h) == [(0.5, CLOSE_UNTIL_CONTACT, "reached")]
    else:
        assert h.arm.jaw_commands == []
        assert ended == [GripperOutcome("reached", 1.0)]


# An opening that something blocks


def test_a_blocked_opening_ends_in_the_light_hold_and_the_operator_is_told(
    human_app,
):
    h = human_app
    trigger = following_open_jaws(h)
    trigger.pull(0.5)
    h.arm.jaw_block(0.7)  # something keeps the jaws from opening further
    sent, held = len(h.arm.jaw_commands), len(h.arm.light_holds)
    trigger.pull(1.0)
    assert commands(h, sent) == [
        *opening_steps(0.5, 0.7),
        (pytest.approx(0.75), PLAIN, "stalled"),
    ]
    # Left in the light hold: the jaws do not push on at the opening stiffness.
    assert h.arm.jaw_stiffness == LIGHT and len(h.arm.light_holds) == held + 1
    assert h.arm.gripper_read() == pytest.approx(0.7)
    told = f"{OPENING_STALLED}0.70"
    facts = jaws(h)
    assert (facts["matched"], facts["refusal"]) == (True, told)
    # The trigger stays let go: the block is not tried again.
    trigger.pull(1.0)
    assert len(h.arm.jaw_commands) == sent + 5 and jaws(h)["refusal"] == told
    # The way is free: pulled a little and let go, the jaws open.
    h.arm.jaw_block(None)
    trigger.pull(0.9)
    trigger.pull(1.0)
    assert h.arm.gripper_read() == 1.0 and jaws(h)["refusal"] == ""


# The setting, and a jaw opening that is no opening


@pytest.mark.parametrize("trigger_jaws", [True, False])
def test_with_the_setting_false_the_trigger_does_nothing(
    make_robot_app, config, trigger_jaws: bool
):
    human = dataclasses.replace(config.human, trigger_jaws=trigger_jaws)
    h = make_robot_app(config=dataclasses.replace(config, human=human))
    held = len(h.arm.light_holds)
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    trigger.pull(0.5)
    facts = state_now(h).health["human"]
    if trigger_jaws:
        assert facts["jaws"]["matched"]
        assert commands(h) == [(0.5, CLOSE_UNTIL_CONTACT, "reached")]
        assert h.arm.gripper_read() == 0.5
    else:
        assert "jaws" not in facts
        assert h.arm.jaw_commands == [] and len(h.arm.light_holds) == held
        assert h.arm.gripper_read() == 1.0
        # Nothing ends what never began: a STOP holds the jaws as before.
        h.advance(0.3)
        assert len(h.arm.light_holds) == held


def test_no_jaw_opening_in_the_human_target_leaves_the_jaws_as_they_are(human_app):
    h = human_app
    trigger = following_open_jaws(h)
    held = len(h.arm.light_holds)
    # The trigger is not usable: the HUMAN_SETPOINT carries no jaw opening.
    trigger.pull(None)
    facts = jaws(h)
    assert (facts["matched"], facts["asked"]) == (False, None)
    assert facts["refusal"] == NO_OPENING
    assert h.arm.jaw_commands == [] and len(h.arm.light_holds) == held
    # Usable again, half pulled: it must match the grip level first.
    trigger.pull(0.5)
    assert h.arm.jaw_commands == [] and jaws(h)["refusal"] == NOT_MATCHED


@pytest.mark.parametrize("bad", [1.5, -0.1])
def test_a_jaw_opening_outside_0_to_1_is_refused(human_app, bad: float):
    h = human_app
    trigger = following_open_jaws(h)
    held = len(h.arm.light_holds)
    trigger.pull(bad)
    facts = jaws(h)
    assert not facts["matched"] and facts["asked"] is None
    assert facts["refusal"] == (
        f"refused: jaw_opening must be between 0 and 1, got {bad!r}"
    )
    # The jaw motion ended as at a STOP, and nothing else was sent.
    assert len(h.arm.light_holds) == held + 1 and h.arm.jaw_stiffness == LIGHT
    assert h.arm.jaw_commands == [] and h.arm.gripper_read() == 1.0
    # A good opening after it must match the grip level again.
    trigger.pull(0.5)
    assert h.arm.jaw_commands == [] and jaws(h)["refusal"] == NOT_MATCHED
    trigger.pull(1.0)
    trigger.pull(0.5)
    assert commands(h) == [(0.5, CLOSE_UNTIL_CONTACT, "reached")]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.5"])
def test_a_jaw_opening_that_is_no_finite_number_is_refused(human_app, bad: Any):
    # No link message carries such a value; a human target made on the Thor could.
    h = human_app
    trigger = following_open_jaws(h)
    target = human_target(trigger.message(), h.clock.monotonic())
    h.app.offer_human_target(dataclasses.replace(target, jaw_opening=bad))
    facts = h.app._jaw_follower.facts()
    assert not facts["matched"]
    assert facts["refusal"].startswith("refused: jaw_opening must be ")
    trigger.pull(0.5)
    assert h.arm.jaw_commands == [] and h.arm.gripper_read() == 1.0


# The grip level


def test_without_a_jaw_target_the_grip_level_is_the_measured_opening(human_app):
    h = human_app
    agent_leaves_the_jaws(h, 0.4)
    h.arm._jaw_target = None  # a driver that names no jaw target
    sent = len(h.arm.jaw_commands)
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    assert jaws(h)["grip_level"] == pytest.approx(0.4)
    assert not jaws(h)["matched"] and len(h.arm.jaw_commands) == sent
    trigger.pull(0.4)
    assert jaws(h)["matched"]


def test_without_a_known_grip_level_the_trigger_moves_nothing(human_app):
    h = human_app
    h.arm._jaw_referenced = False  # the driver dropped the jaw reference
    held = len(h.arm.light_holds)
    take_over(h)
    trigger = Trigger(h)
    for opening in (1.0, 0.5, 0.0, 1.0):
        trigger.pull(opening)
    facts = jaws(h)
    assert (facts["matched"], facts["grip_level"]) == (False, None)
    assert facts["refusal"] == f"{NO_GRIP_LEVEL}the jaws are not referenced"
    assert h.arm.jaw_commands == [] and len(h.arm.light_holds) == held


# The hand-back


def test_the_hand_back_still_needs_the_jaws_proven_empty_after_a_person_moved_them(
    human_app,
):
    h = human_app
    trigger = following_open_jaws(h)
    # The person closes the jaws on an object 0.5 wide.
    h.arm.jaw_object(0.5)
    trigger.pull(0.0)
    assert (h.arm.gripper_read(), h.arm.jaw_stiffness) == (0.5, LIGHT)
    refused = hand_back(h)
    assert refused.status == "refused"
    assert refused.reason.startswith(JAWS_NOT_EMPTY)
    assert refused.reason.endswith(OPEN_FIRST)
    assert h.app.takeover_open and h.app.snapshot().owner is Owner.HUMAN
    # Opened part of the way, the jaws are still not proven empty.
    trigger.pull(0.0)
    trigger.pull(0.8)
    assert h.arm.gripper_read() == pytest.approx(0.8)
    part_open = hand_back(h)
    assert part_open.status == "refused"
    assert part_open.reason.startswith(JAWS_NOT_EMPTY)
    # Opened all the way with the trigger, they are: the takeover ends.
    trigger.pull(0.8)
    trigger.pull(1.0)
    assert h.arm.gripper_read() == 1.0
    sent = len(h.arm.jaw_commands)
    done = hand_back(h)
    assert done.status == "completed", done.reason
    assert not h.app.takeover_open and h.app.snapshot().owner is Owner.NOBODY
    # The trigger of a person who gave the arm back moves nothing.
    trigger.pull(0.5)
    assert len(h.arm.jaw_commands) == sent and h.arm.gripper_read() == 1.0


# On APAI's mock arm


@pytest.fixture
def apai_person_app(clock) -> Iterator[AppHarness]:
    """A started robot app on APAI's mock arm with human driving switched on."""
    import test_apai_app as apai

    profile = apai.apai_backend.load_apai_profile("mock")
    harness = apai.ApaiHarness(human_config(apai.make_config(profile)), profile, clock)
    harness.start()
    yield harness
    harness.close()
    apai.check_driver_use(harness)


@pytest.mark.apai
def test_on_apais_mock_arm_the_jaws_follow_the_trigger_into_the_light_hold(
    apai_person_app,
):
    import test_apai_app as apai

    h = apai_person_app
    closed, light = apai.JAWS_CLOSED, pytest.approx(apai.LIGHT)
    h.hold_at(apai.HOME)
    # The reference at start left the jaws closed: that is the grip level.
    assert h.spy.jaw_commands == [(closed, 1.0)]
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(1.0)
    trigger.pull(0.5)
    facts = jaws(h)
    assert (facts["matched"], facts["grip_level"]) == (False, 0.0)
    assert h.spy.jaw_commands == [(closed, 1.0)]
    # Pulled all the way it has matched; released, the jaws open in steps.
    trigger.pull(0.0)
    assert jaws(h)["matched"] and h.spy.jaw_commands == [(closed, 1.0)]
    trigger.pull(1.0)
    steps = h.spy.jaw_commands[1:]
    assert len(steps) == 20
    assert [effort for _, effort in steps] == [OPEN_EFFORT] * 20
    positions = [position for position, _ in steps]
    assert positions == sorted(positions, reverse=True)
    assert positions[-1] == pytest.approx(0.0)  # the mock counts from 0 open
    assert h.arm.gripper_read() == pytest.approx(1.0) and jaws(h)["opened"]
    # An object 0.4 wide: the close stalls on it and is left in the light hold.
    h.spy.backend.object_stop_frac = 0.6
    sent = len(h.spy.jaw_commands)
    trigger.pull(0.0)
    assert h.spy.jaw_commands[sent:] == [(closed, OPEN_EFFORT), (closed, light)]
    state = state_now(h)
    assert state.gripper.opening == pytest.approx(0.4)
    assert state.gripper.may_be_holding and jaws(h)["matched"]
    # Not proven empty: the hand-back is refused, and sends the jaws nothing.
    refused = final(h, h.client.request_owner("handback", reason="right A"))
    assert refused.status == "refused" and refused.reason.startswith(JAWS_NOT_EMPTY)
    assert len(h.spy.jaw_commands) == sent + 2
    assert h.arm.torque_on() and "stop" not in h.spy.calls


@pytest.mark.apai
def test_on_apais_mock_arm_a_cut_close_is_aimed_anew_at_what_the_trigger_asks(
    apai_person_app,
):
    import test_apai_app as apai

    h = apai_person_app
    closed, light = apai.JAWS_CLOSED, pytest.approx(apai.LIGHT)
    h.hold_at(apai.HOME)
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(0.0)
    assert jaws(h)["matched"]
    trigger.pull(1.0)
    assert h.arm.gripper_read() == pytest.approx(1.0)
    # An object 0.4 wide. Pulled all the way, and eased to 0.2 as the close
    # begins: tighter than the object is wide.
    h.spy.backend.object_stop_frac = 0.6
    sent = len(h.spy.jaw_commands)
    as_command_begins(h, 1, lambda: setattr(trigger, "opening", 0.2))
    trigger.pull(0.0)
    trigger.pull(0.2)
    after = h.spy.jaw_commands[sent:]
    # The close to closed, cut and left in the light hold; then the close to
    # 0.2, which stalls on the object and is left in the light hold there.
    assert after[:2] == [(closed, OPEN_EFFORT), (closed, light)]
    at_0_2 = pytest.approx(0.8)  # the mock counts the jaw position from 0 open
    assert after[-2:] == [(at_0_2, OPEN_EFFORT), (at_0_2, light)]
    assert h.arm.gripper_last_target() == pytest.approx(0.2)
    state = state_now(h)
    assert state.gripper.opening == pytest.approx(0.4)
    assert state.gripper.may_be_holding and jaws(h)["matched"]
    assert h.arm.torque_on() and "stop" not in h.spy.calls


# On the driver class of the rig


@pytest.fixture
def rig_person_app(clock, monkeypatch, tmp_path) -> Iterator[tuple[AppHarness, Any]]:
    """The rig file's robot app, started, on the driver class the rig runs.

    APAI's driver of the rebot_rs profile with its gripper guard, over the
    stand-in motor SDK of test_apai_backend.py; the profile is loaded as on
    the Thor (rig.rig_profile), and human driving is switched on in memory.
    Yields the app's harness and that SDK.
    """
    import test_apai_backend as rs
    import test_rig_configs as rig

    config = load_config(rig.ROBOT_RIG)
    on = dataclasses.replace(config.human, enabled=True)
    backend = rs.apai_backend
    profile = rig.rig_profile(tmp_path)
    config = backend.config_for_profile(dataclasses.replace(config, human=on), profile)
    arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
    checkers = backend.build_apai_checkers(config, profile, clock)
    harness = AppHarness(config, arm, checkers, clock)
    # The driver takes the time between two joint reads from the machine's
    # clock: here it reads the app's.
    app_time = types.SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    monkeypatch.setattr(sys.modules[type(arm.backend).__module__], "time", app_time)
    try:
        harness.start()
        yield harness, sdk
    finally:
        harness.close()


@pytest.mark.apai
def test_on_the_rigs_driver_class_a_blocked_opening_ends_in_the_light_hold(
    rig_person_app,
):
    import test_apai_backend as rs

    h, sdk = rig_person_app
    jaw = sdk.gripper

    def motor_commands(mark: int) -> list[tuple[float, float]]:
        """(position in rad, stiffness) of the jaw motor's commands from mark on."""
        return [(round(q[0], 3), kp[0]) for q, kp in jaw.sent[mark:]]

    # The reference at start closed the jaws: that is the grip level.
    take_over(h)
    trigger = Trigger(h)
    trigger.pull(0.0)
    assert jaws(h)["matched"] and jaws(h)["grip_level"] == 0.0
    # Something keeps the jaws from opening past 1.0 rad of their 6.2.
    jaw.blocked_at = 1.0
    mark = len(jaw.sent)
    trigger.pull(1.0)
    sent = motor_commands(mark)
    assert jaw.at() == pytest.approx(1.0)
    # The step to 0.2 of the travel, 1.24 rad, stalled: it is left at the
    # light holding stiffness, not pushing on at the opening stiffness.
    assert sent[-1] == (pytest.approx(1.24), rs.RS_HOLD_KP)
    assert all(kp > rs.RS_HOLD_KP for _, kp in sent[:-1])
    facts = jaws(h)
    assert facts["matched"]
    assert facts["refusal"] == f"{OPENING_STALLED}{1.0 / rs.RS_TRAVEL:.2f}"
    now = h.app.snapshot()
    assert (now.owner, now.stop_level) == (Owner.HUMAN, StopLevel.NONE)
    # The trigger stays let go: the jaws are sent nothing more.
    mark = len(jaw.sent)
    trigger.pull(1.0)
    assert motor_commands(mark) == []
    assert h.arm.torque_on() and sdk.disconnects == 0
