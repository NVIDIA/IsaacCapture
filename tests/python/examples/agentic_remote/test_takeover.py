# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A person takes the arm over, drives it and gives it back, on the simulated arm.

A takeover makes a person the control owner; the hand-back ends it at rest. A
human target is a joint target from a person's hand, sent in a HUMAN_SETPOINT
link message; a human drive is one stretch of human motion. The drive's own
rules are in test_human_drive.py.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import (
    HUMAN_SOURCE_SYNTHETIC,
    HUMAN_SOURCE_TELEOP,
    HumanSetpoint,
    Result,
    RobotState,
    ViewStatus,
)
from isaaccapture_examples.agentic_remote.robot_app import rest as rest_module
from isaaccapture_examples.agentic_remote.robot_app.app import (
    HANDED_BACK,
    HUMAN_OFF,
    NEEDS_RELEASE,
    NO_TAKEOVER,
    NOTE_SQUEEZE_AGAIN,
    TAKEOVER_OPEN,
    human_target,
)
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.gripper import JAWS_BUSY
from isaaccapture_examples.agentic_remote.robot_app.human import (
    SPEED_MARGIN,
    DriveEnd,
    HumanTarget,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    GripperOutcome,
    MotionState,
    Owner,
    StopLevel,
)

from conftest import AppHarness

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
REST = [0.0] * 6
# A pose inside every joint box; the return to rest from it has 209 setpoints.
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
RETURN_SETPOINTS = 209
RACED = "a STOP or an owner change arrived while the request was checked"
OPEN_THE_VIEW = "; open the live view page in a visible browser tab"
BUSY_DRIVING = "busy: human driving is running"
AT_A_CONTACT = "the arm is holding at a contact"
RIGHT_B = "right B held 1.0 s"
RIGHT_A = "right A held 1.0 s"

_SEQ = itertools.count(1)


# What later test files import


def human_config(config: RobotAppConfig, **human_fields: Any) -> RobotAppConfig:
    """The configuration with human driving switched on.

    The human joint box becomes rest.region_min..region_max, which
    human.enabled needs, then human.enabled is True and human_fields replace
    further fields of the human block.
    """
    box = dataclasses.replace(
        config.limits.human,
        joint_min=config.rest.region_min,
        joint_max=config.rest.region_max,
    )
    narrowed = dataclasses.replace(
        config, limits=dataclasses.replace(config.limits, human=box)
    )
    human = dataclasses.replace(config.human, **{"enabled": True, **human_fields})
    return dataclasses.replace(narrowed, human=human)


def keep_human_fed(
    h: AppHarness, setpoint: Callable[[], HumanSetpoint | None], *, view: bool = True
) -> None:
    """Feed the app after every setpoint the arm is sent, in step with the drive.

    It wraps h.arm.send_target: after each send it sends setpoint() over the
    link (None: nothing), with view also a live VIEW_STATUS of age 0, and waits
    one dispatcher pass. On the fake clock a drive then has a fresh human
    target and a fresh view in every period. setpoint is called anew each
    time, so a test can change the target, release the grip or stop feeding.
    """
    real = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        message = setpoint()
        if message is not None:
            h.client.send(message)
        if view:
            h.client.send(ViewStatus(live=True, age_s=0.0))
        h.step()

    h.arm.send_target = send_target


# Helpers of this file


class Done:
    """The on_done of a request made at the Thor terminal."""

    def __init__(self) -> None:
        self.results: list[Result] = []
        self._called = threading.Event()

    def __call__(self, result: Result) -> None:
        self.results.append(result)
        self._called.set()

    def wait(self) -> Result:
        assert self._called.wait(WAIT_S), "the request did not end"
        return self.results[0]


def statuses(h, command_id: str) -> list[str]:
    return [result.status for result in h.client.results(command_id)]


def final(h, command_id: str) -> Result:
    return h.client.wait_result(command_id, timeout_s=WAIT_S)


def setpoint(h, q: Sequence[float] | None = None, **fields: Any) -> HumanSetpoint:
    """A HUMAN_SETPOINT that quotes the app's numbers as they are now.

    With q the grip is squeezed and the clutch engaged toward q; without, the
    grip is released. The controller is ready and its data are 0.02 s old.
    fields replace any field.
    """
    now = h.app.snapshot()
    values: dict[str, Any] = {
        "session_id": now.session_id,
        "owner_number": now.owner_number,
        "stop_number": now.stop_number,
        "seq": next(_SEQ),
        "source": HUMAN_SOURCE_TELEOP,
        "stage": "ready",
        "sample_age_s": 0.02,
        "engaged": q is not None,
        "grip_released": q is None,
        "q": None if q is None else [float(value) for value in q],
        "jaw_opening": None,
        "state_clock_mono_s": None,
        "note": "",
    }
    values.update(fields)
    return HumanSetpoint(**values)


def feed(h, message: HumanSetpoint | None = None, *, view: bool = True) -> None:
    """Send a HUMAN_SETPOINT and a live VIEW_STATUS; the app has taken them on return.

    Without a message: the status of a ready controller with the grip released.
    """
    h.client.send(setpoint(h) if message is None else message)
    if view:
        h.client.send(ViewStatus(live=True, age_s=0.0))
    h.step()
    h.step()


def state_now(h) -> RobotState:
    """The ROBOT_STATE the app would send now; what was sent to it is taken first."""
    h.step()
    h.step()
    return h.app._publisher.build(h.clock.monotonic())


def human_facts(h) -> dict[str, Any]:
    return state_now(h).health["human"]


def ask_takeover(h, reason: str = RIGHT_B) -> Result:
    """OWNER_REQUEST "human" quoting the app's numbers; its final RESULT."""
    return final(h, h.client.request_owner("human", reason=reason))


def take_over(h) -> Result:
    """A person takes the arm over: fresh controller data, a live view, the request."""
    feed(h)
    result = ask_takeover(h)
    assert result.status == "completed", result.reason
    h.client.wait_state(lambda s: s.owner_number == result.detail["owner_number"])
    return result


def hand_back(h) -> Result:
    """OWNER_REQUEST "handback" quoting the app's numbers; its final RESULT."""
    return final(h, h.client.request_owner("handback", reason=RIGHT_A))


def assert_refused(h, result: Result, reason: str, what: str = "human") -> None:
    """The request was refused with this reason, at both terminals."""
    assert (result.status, result.reason) == ("refused", reason)
    assert statuses(h, result.command_id) == ["refused"]
    told = f"owner request {what!r} refused: {reason}"
    assert h.heard("refused")[-1][1] == told


def assert_no_takeover(h, before, owner_events: int = 0) -> None:
    """Nothing changed since `before`: no owner, no takeover, no motion.

    owner_events is how many "owner" events there were before.
    """
    assert h.app.snapshot() == before
    assert not h.app.takeover_open
    assert len(h.heard("owner")) == owner_events
    assert h.arm.sent == [] and h.arm.jaw_commands == []
    assert h.arm.torque_on() and h.arm.released_at == []


class Hand:
    """The person's hand, as keep_human_fed's setpoint: squeezing or released."""

    def __init__(self, h, **fields: Any) -> None:
        self.h = h
        self.fields = fields  # of every HUMAN_SETPOINT it makes
        # The joint target while the grip is squeezed; None: the grip is released.
        self.goal: list[float] | None = None
        # How far the target runs ahead of the last setpoint sent, on joint index
        # 0, while the grip is squeezed; None: the target stays at `goal`.
        self.stride: float | None = None
        # The grip is released once the arm was sent this many setpoints in all.
        self.release_at: int | None = None
        self.silent = False  # the human program sends nothing

    def __call__(self) -> HumanSetpoint | None:
        if self.silent:
            return None
        sent = self.h.arm.sent
        if self.release_at is not None and len(sent) >= self.release_at:
            self.goal = None
        if self.goal is not None and self.stride is not None and sent:
            last = sent[-1][0]
            self.goal = [last[0] + self.stride, *last[1:]]
        return setpoint(self.h, self.goal, **self.fields)

    def squeeze(self, goal: Sequence[float], sends: int | None = None) -> None:
        """Squeeze toward goal; the app has taken the first engaged target on return.

        With sends the grip is released once that many setpoints went out.
        """
        self.goal = [float(value) for value in goal]
        self.release_at = None if sends is None else len(self.h.arm.sent) + sends
        feed(self.h, self())

    def release(self) -> None:
        """Release the grip: the next HUMAN_SETPOINT says so."""
        self.goal = None


def person(h, **fields: Any) -> Hand:
    """A person owns the arm and has released the grip; their hand feeds the app."""
    take_over(h)
    hand = Hand(h, **fields)
    keep_human_fed(h, hand)
    feed(h, hand())
    return hand


def drive(h, hand: Hand, goal: Sequence[float], sends: int = 6) -> None:
    """One human drive toward goal, ended by a released grip after `sends` setpoints."""
    hand.squeeze(goal, sends)
    h.idle()


def drives(h) -> list[DriveEnd]:
    """How each human drive ends, from here on."""
    ended: list[DriveEnd] = []
    real = h.app._human_drive.run

    def run(admitted, should_stop) -> DriveEnd:
        ended.append(real(admitted, should_stop))
        return ended[-1]

    h.app._human_drive.run = run
    return ended


def moved(start: Sequence[float], by: float = 0.05) -> list[float]:
    """The pose with joint index 0 moved on by `by` rad."""
    return [start[0] + by, *start[1:]]


def ask_for_help(h, reason: str = "the agent is stuck") -> Result:
    """OWNER_REQUEST "help_needed", answered, with its ROBOT_STATE received."""
    result = final(h, h.client.request_owner("help_needed", reason=reason))
    assert result.status == "completed", result.reason
    h.client.wait_state(lambda s: s.owner_number == result.detail["owner_number"])
    return result


@pytest.fixture
def config() -> RobotAppConfig:
    """The demo values with human driving switched on.

    In place of conftest's fixture, so that the checkers of this file's apps
    are built with the human joint box that human_config() sets.
    """
    return human_config(RobotAppConfig.default())


@pytest.fixture
def human_app(make_robot_app) -> AppHarness:
    """A started robot app at rest with human driving switched on."""
    return make_robot_app()


@pytest.fixture
def raised_app(human_app) -> AppHarness:
    """As human_app, with the arm holding away from rest, its jaws referenced."""
    human_app.hold_at(RAISED)
    return human_app


# The helpers themselves


def test_human_config_switches_human_driving_on_inside_the_recovery_region():
    config = RobotAppConfig.default()
    assert not config.human.enabled
    on = human_config(config)
    assert on.human == dataclasses.replace(config.human, enabled=True)
    human, rest = on.limits.human, on.rest
    assert (human.joint_min, human.joint_max) == (rest.region_min, rest.region_max)
    assert human.max_joint_speed == config.limits.human.max_joint_speed
    assert (on.limits.agent, on.limits.recovery, on.rest) == (
        config.limits.agent,
        config.limits.recovery,
        config.rest,
    )
    changed = human_config(config, require_live_view=False, view_age_limit_s=2.0)
    assert changed.human.enabled and not changed.human.require_live_view
    assert changed.human.view_age_limit_s == 2.0
    assert not human_config(config, enabled=False).human.enabled


def test_a_human_setpoint_becomes_a_human_target_in_one_place():
    m = HumanSetpoint(
        session_id="s",
        owner_number=3,
        stop_number=1,
        seq=9,
        source=HUMAN_SOURCE_TELEOP,
        stage="ready",
        sample_age_s=0.03,
        engaged=True,
        grip_released=False,
        q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        jaw_opening=0.0,
        state_clock_mono_s=999.75,
        note="following",
    )
    assert human_target(m, 1000.0) == HumanTarget(
        seq=9,
        owner_number=3,
        stop_number=1,
        source=HUMAN_SOURCE_TELEOP,
        stage="ready",
        note="following",
        engaged=True,
        grip_released=False,
        q=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        sample_age_s=0.03,
        arrived_mono=1000.0,
        round_trip_s=0.25,
    )
    status = dataclasses.replace(m, engaged=False, q=None, state_clock_mono_s=None)
    assert human_target(status, 1000.0).q is None
    assert human_target(status, 1000.0).round_trip_s is None
    # An age or a round trip that is not a finite number of 0 or more is unknown.
    for age in (None, -0.001, float("nan"), float("inf")):
        unknown = human_target(dataclasses.replace(m, sample_age_s=age), 1000.0)
        assert unknown.sample_age_s is None, age
    assert human_target(dataclasses.replace(m, sample_age_s=0), 5.0).sample_age_s == 0.0
    for computed_on in (1000.5, float("nan"), float("-inf")):
        ahead = dataclasses.replace(m, state_clock_mono_s=computed_on)
        assert human_target(ahead, 1000.0).round_trip_s is None, computed_on


# A takeover is refused: every reason, with its text; nothing changes


def test_a_takeover_is_refused_while_human_driving_is_switched_off(make_robot_app):
    shipped = RobotAppConfig.default()
    assert not shipped.human.enabled
    h = make_robot_app(config=shipped)
    before = h.app.snapshot()
    feed(h)
    assert HUMAN_OFF == (
        "human driving is switched off in this robot app's file (human.enabled)"
    )
    assert_refused(h, ask_takeover(h), HUMAN_OFF)
    assert_no_takeover(h, before)
    # Also with numbers that are out of date: the switch is named first.
    old = final(h, h.client.request_owner("human", stop_number=7))
    assert old.reason == HUMAN_OFF
    assert human_facts(h)["takeover"] == HUMAN_OFF
    # An engaged human target starts nothing either.
    feed(h, setpoint(h, moved(REST)))
    h.idle()
    assert human_facts(h)["refusal"] == HUMAN_OFF
    assert_no_takeover(h, before)


def test_a_takeover_is_refused_while_the_agent_owns_the_arm(human_app):
    h = human_app
    h.new_task()
    before = h.app.snapshot()
    feed(h)
    assert human_facts(h)["takeover"] == ""  # the owner is not this key's to judge
    assert_refused(h, ask_takeover(h), "control owner is agent, not nobody")
    assert_no_takeover(h, before, owner_events=1)  # the new task's
    # Numbers that are out of date are refused as for every command.
    old = final(h, h.client.request_owner("human", stop_number=5))
    assert old.reason == "stop number 5 does not match the current 0"
    assert h.app.snapshot() == before


def test_a_takeover_is_refused_with_a_fault_stop(human_app):
    h = human_app
    h.arm.set_fault("motor 3 over temperature")
    h.wait_local_event("fault")
    h.arm.clear_faults()
    before = h.app.snapshot()
    assert before.stop_level is StopLevel.FAULT
    feed(h)
    latched = "fault-stop latched: motor 3 over temperature"
    assert human_facts(h)["takeover"] == latched
    assert_refused(h, ask_takeover(h), latched)
    assert_no_takeover(h, before)


def test_a_takeover_is_refused_without_fresh_controller_data(human_app):
    h = human_app
    before = h.app.snapshot()
    h.client.send(ViewStatus(live=True, age_s=0.0))
    none = "no fresh controller data: no HUMAN_SETPOINT has arrived"
    assert human_facts(h)["takeover"] == none
    assert_refused(h, ask_takeover(h), none)
    # The human program waits for the headset.
    feed(h, setpoint(h, stage="waiting_for_headset", sample_age_s=None))
    waiting = "no fresh controller data: stage waiting_for_headset"
    assert human_facts(h)["takeover"] == waiting
    assert_refused(h, ask_takeover(h), waiting)
    # Ready, with an age nobody knows.
    feed(h, setpoint(h, sample_age_s=None))
    unknown = "no fresh controller data: the sample age is unknown"
    assert_refused(h, ask_takeover(h), unknown)
    # Ready, and older than human.sample_age_limit_s once it is asked about.
    feed(h, setpoint(h, sample_age_s=0.2))
    h.advance(0.1, 0.1)
    h.client.send(ViewStatus(live=True, age_s=0.0))
    old = ask_takeover(h)
    assert old.status == "refused"
    assert re.fullmatch(
        r"no fresh controller data: 0\.3\d\d s old, the limit is 0\.25 s", old.reason
    )
    assert human_facts(h)["takeover"].startswith("no fresh controller data: 0.3")
    assert_no_takeover(h, before)
    # Fresh data: the same request is taken.
    assert take_over(h).status == "completed"


def test_a_takeover_is_refused_with_a_stale_view(human_app):
    h = human_app
    before = h.app.snapshot()
    feed(h, view=False)
    none = f"the live view is not fresh: no VIEW_STATUS has arrived{OPEN_THE_VIEW}"
    assert human_facts(h)["takeover"] == none
    assert_refused(h, ask_takeover(h), none)
    # A view that was live, 0.4 s old then, and is older than the limit now.
    h.client.send(ViewStatus(live=True, age_s=0.4))
    h.step()
    h.advance(0.2, 0.2)
    feed(h, view=False)
    stale = ask_takeover(h)
    assert stale.status == "refused"
    assert re.fullmatch(
        r"the live view is not fresh: 0\.[67]\d\d s old, the limit is 0\.5 s"
        + re.escape(OPEN_THE_VIEW),
        stale.reason,
    )
    assert human_facts(h)["takeover"].startswith("the live view is not fresh: 0.")
    assert_no_takeover(h, before)
    assert take_over(h).status == "completed"


def test_a_takeover_is_refused_with_a_view_that_is_not_live(human_app):
    h = human_app
    before = h.app.snapshot()
    feed(h)  # live
    h.client.send(ViewStatus(live=False, age_s=0.0))
    not_live = f"the live view is not fresh: the view is not live{OPEN_THE_VIEW}"
    assert human_facts(h)["takeover"] == not_live
    assert_refused(h, ask_takeover(h), not_live)
    assert_no_takeover(h, before)


def test_no_view_is_asked_for_a_takeover_when_the_setting_says_so(
    make_robot_app, config
):
    unseen = human_config(config, require_live_view=False)
    h = make_robot_app(config=unseen, driver="sim")
    feed(h, view=False)
    assert human_facts(h)["takeover"] == ""
    assert ask_takeover(h).status == "completed"


@pytest.mark.parametrize(
    ("driver", "declared"), [("", "no driver"), ("real", "driver real")]
)
def test_a_takeover_is_refused_with_a_synthetic_controller_on_a_driver_that_is_not_simulated(
    make_robot_app, driver, declared
):
    h = make_robot_app(driver=driver)
    before = h.app.snapshot()
    feed(h, setpoint(h, source=HUMAN_SOURCE_SYNTHETIC))
    synthetic = (
        "a synthetic controller may drive only a simulated arm; this robot app "
        f"declares {declared}"
    )
    assert human_facts(h)["takeover"] == synthetic
    assert_refused(h, ask_takeover(h), synthetic)
    assert_no_takeover(h, before)
    # An engaged target of that source starts no drive either.
    feed(h, setpoint(h, moved(REST), source=HUMAN_SOURCE_SYNTHETIC))
    h.idle()
    assert human_facts(h)["refusal"] == synthetic
    assert h.arm.sent == []
    # The headset's controller may take this arm.
    assert take_over(h).status == "completed"


@pytest.mark.parametrize("driver", ["sim", "apai-mock"])
def test_a_synthetic_controller_may_take_a_simulated_arm(make_robot_app, driver):
    h = make_robot_app(driver=driver)
    feed(h, setpoint(h, source=HUMAN_SOURCE_SYNTHETIC))
    assert human_facts(h)["takeover"] == ""
    assert ask_takeover(h).status == "completed"


def test_a_takeover_is_refused_with_the_arm_outside_the_human_limits(make_robot_app):
    # Inside the recovery joint box of 3.0 rad, outside the human one of 2.5.
    outside = [2.6, 0.0, 0.0, 0.0, 0.0, 0.0]
    h = make_robot_app(pose=outside)
    before = h.app.snapshot()
    feed(h)
    assert human_facts(h)["takeover"] == ""  # no joint is read for this key
    assert_refused(
        h,
        ask_takeover(h),
        "the arm is outside the human limits: pose: joint index 0 at 2.6000 rad "
        "is outside the human joint limits -2.5000..2.5000; type rest",
    )
    assert_no_takeover(h, before)


# Human driving on, with a human joint box of 0.18 rad around the rest pose.
_ON = human_config(RobotAppConfig.default())
_SMALL_BOX = dataclasses.replace(
    _ON,
    limits=dataclasses.replace(
        _ON.limits,
        human=dataclasses.replace(
            _ON.limits.human, joint_min=[-0.18] * 6, joint_max=[0.18] * 6
        ),
    ),
)


@pytest.mark.parametrize("config", [_SMALL_BOX], ids=["a small human box"])
def test_the_typed_rest_a_refused_takeover_names_brings_the_arm_inside(raised_app):
    h = raised_app  # joint index 0 at 0.2 rad: inside rest.region_*
    assert h.config.limits.human.joint_max[0] == 0.18
    feed(h)
    refused = ask_takeover(h)
    assert (refused.status, refused.reason) == (
        "refused",
        "the arm is outside the human limits: pose: joint index 0 at 0.2000 rad "
        "is outside the human joint limits -0.1800..0.1800; type rest",
    )
    # Where the return to rest is taken, the typed rest is the way back.
    assert final(h, h.client.rest()).status == "completed"
    assert take_over(h).status == "completed"


def test_a_takeover_is_refused_when_the_pose_cannot_be_read(human_app):
    h = human_app
    before = h.app.snapshot()
    feed(h)
    h.arm.fail_reads(2, 1)
    refused = ask_takeover(h)
    assert refused.status == "refused"
    assert refused.reason.startswith("the measured pose could not be read: ")
    assert_no_takeover(h, before)


def test_a_takeover_is_refused_while_a_jaw_reference_runs(make_robot_app):
    # Started raised: the jaws have no reference, and one may be typed.
    h = make_robot_app(pose=RAISED, allow_reference=True)
    reached, release = h.hold_call("gripper_reference")
    done = Done()
    assert h.app.request_reference(True, done) is None
    assert reached.wait(WAIT_S)  # the jaws are closing to their stop
    before = h.app.snapshot()
    feed(h)
    assert JAWS_BUSY == "busy: a jaw command is running or waiting"
    assert_refused(h, ask_takeover(h), JAWS_BUSY)
    assert h.app.snapshot() == before and not h.app.takeover_open
    assert h.arm.sent == []
    release.set()
    assert done.wait().status == "completed"
    assert h.app._worker.wait_idle(WAIT_S)
    # The reference is over: the takeover is taken.
    assert take_over(h).status == "completed"


def test_a_takeover_is_refused_while_a_guarded_opening_runs(human_app):
    h = human_app
    # The jaws are part closed, so that the opening has steps to make.
    h.arm.gripper_command(0.2, 0.3, "plain", threading.Event())
    reached, release = threading.Event(), threading.Event()

    def on_jaw_step(at: float) -> None:
        if not reached.is_set():
            reached.set()
            release.wait(WAIT_S)

    h.arm.on_jaw_step = on_jaw_step
    open_id = h.client.open_jaws()
    try:
        assert reached.wait(WAIT_S)  # the jaws are opening, step by step
        assert not h.app._worker.referencing()
        before = h.app.snapshot()
        feed(h)
        assert_refused(h, ask_takeover(h), JAWS_BUSY)
        assert h.app.snapshot() == before and not h.app.takeover_open
        assert h.heard("owner") == [] and h.arm.sent == []
    finally:
        release.set()
    # The opening was not cut short: it ends with the jaws open.
    assert final(h, open_id).status == "completed"
    assert h.app._worker.wait_idle(WAIT_S)
    assert h.arm.gripper_read() == pytest.approx(1.0)
    assert take_over(h).status == "completed"


def test_a_takeover_is_refused_while_a_return_to_rest_runs(raised_app):
    h = raised_app
    reached, release = h.hold_move_after(5)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    feed(h)
    refused = ask_takeover(h)
    assert (refused.status, refused.reason) == (
        "refused",
        "control owner is recovery, not nobody",
    )
    assert not h.app.takeover_open
    release.set()
    assert final(h, rest_id).status == "completed"


# The takeover


def test_a_takeover_makes_the_human_the_owner_and_cancels_the_wait(human_app):
    h = human_app
    h.new_task()
    ask_for_help(h)
    h.client.wait_state(lambda s: s.assistance_wait_s is not None)
    assert not h.app.takeover_open
    result = take_over(h)
    assert statuses(h, result.command_id) == ["completed"]
    assert result.detail == {"owner": "human", "owner_number": 3}
    state = h.client.wait_state(lambda s: s.owner_number == 3)
    assert (state.owner, state.stop_number, state.stop_level) == ("human", 0, "none")
    assert state.limit_set_version == h.config.limits.human.version
    assert state.assistance_wait_s is None
    assert h.app.takeover_open
    _, text, data = h.heard("owner")[-1]
    assert text == (
        "TAKEOVER: a person owns the arm, owner number 3; release the grip, then "
        "squeeze it to move"
    )
    assert data == result.detail
    assert h.client.wait_event("owner", lambda e: e.text == text).data == data
    # The wait is over: no timed return follows.
    h.advance(125.0)
    assert h.heard("expired") == []
    assert h.arm.sent == [] and h.arm.torque_on()
    # Asked again, the person owns the arm already.
    feed(h)
    again = ask_takeover(h)
    assert again.reason == "control owner is human, not nobody"


def test_a_takeover_sends_no_jaw_command(human_app):
    h = human_app
    jaw_calls: list[str] = []
    for name in ("submit", "open_guarded", "reference", "stop_now"):
        real = getattr(h.app._worker, name)

        def noted(*args, _name=name, _real=real, **kwargs):
            jaw_calls.append(_name)
            return _real(*args, **kwargs)

        setattr(h.app._worker, name, noted)
    light_holds = list(h.arm.light_holds)
    take_over(h)
    assert jaw_calls == []
    assert h.arm.jaw_commands == [] and h.arm.light_holds == light_holds
    assert h.arm.sent == []
    assert h.app.motion_state is MotionState.REST_HOLD


def test_a_takeover_from_nobody_at_rest_needs_no_help_request(human_app):
    h = human_app
    assert h.app.motion_state is MotionState.REST_HOLD
    assert h.heard("help_needed") == []
    result = take_over(h)
    assert result.detail == {"owner": "human", "owner_number": 1}
    assert h.app.snapshot().owner is Owner.HUMAN and h.app.takeover_open
    facts = human_facts(h)
    assert (facts["takeover_open"], facts["needs_release"]) == (True, True)


def test_a_stop_between_the_joint_read_and_the_grant_wins(human_app):
    h = human_app
    feed(h)
    reached, release = h.hold_call("read_all")
    command_id = h.client.request_owner("human", reason=RIGHT_B)
    assert reached.wait(WAIT_S)  # admitted; the joints are about to be read
    # The read runs outside the app's lock: the STOP does not wait for it.
    assert h.app.stop("typed stop") == 1
    release.set()
    assert_refused(h, final(h, command_id), RACED)
    state = h.app.snapshot()
    assert (state.owner, state.stop_number) == (Owner.NOBODY, 1)
    assert not h.app.takeover_open and h.heard("owner") == []
    assert h.arm.sent == [] and h.arm.torque_on()
    # Asked on the new stop number, the arm is taken.
    h.client.wait_state(lambda s: s.stop_number == 1)
    assert take_over(h).status == "completed"


# Takeover open: no agent task until the hand-back


def test_an_agent_task_is_refused_while_a_takeover_is_open(human_app):
    h = human_app
    take_over(h)
    assert TAKEOVER_OPEN == (
        "takeover not finished: give the arm back with right A at rest"
    )
    refused = final(h, h.client.request_owner("agent"))
    assert_refused(h, refused, TAKEOVER_OPEN, "agent")
    state = h.app.snapshot()
    assert (state.owner, state.owner_number) == (Owner.HUMAN, 1)
    assert human_facts(h)["takeover_open"] is True


def test_an_agent_task_is_refused_also_after_a_stop_made_nobody_the_owner(
    human_app,
):
    h = human_app
    take_over(h)
    h.client.stop("right B", sender="controller")
    h.client.wait_state(lambda s: s.stop_number == 1)
    assert h.app.snapshot().owner is Owner.NOBODY
    assert h.app.takeover_open
    refused = final(h, h.client.request_owner("agent"))
    assert_refused(h, refused, TAKEOVER_OPEN, "agent")
    assert h.app.snapshot().owner is Owner.NOBODY
    # A lost robot connection leaves it open too.
    h.lose_link()
    assert h.app.takeover_open
    client = h.connect()
    assert client.state.health["human"]["takeover_open"] is True
    assert final(h, client.request_owner("agent")).reason == TAKEOVER_OPEN
    # The hand-back ends it, and the agent's task is taken.
    assert hand_back(h).status == "completed"
    assert h.new_task().status == "completed"


def test_a_fault_stop_and_its_reset_leave_the_takeover_open(human_app):
    h = human_app
    take_over(h)
    h.arm.set_fault("motor 3 over temperature")
    h.wait_local_event("fault")
    h.arm.clear_faults()
    assert h.app.snapshot().owner is Owner.NOBODY and h.app.takeover_open
    h.client.wait_state(lambda s: s.stop_level == "fault")
    assert final(h, h.client.reset()).status == "completed"
    h.client.wait_state(lambda s: s.stop_level == "none")
    assert h.app.takeover_open
    assert final(h, h.client.request_owner("agent")).reason == TAKEOVER_OPEN
    # The person can take the arm again, or give it back.
    assert take_over(h).status == "completed"
    assert hand_back(h).status == "completed"
    assert not h.app.takeover_open and h.arm.torque_on()


# The hand-back


def test_a_handback_at_measured_rest_ends_the_takeover(human_app):
    h = human_app
    take_over(h)
    result = hand_back(h)
    assert statuses(h, result.command_id) == ["accepted", "completed"]
    assert result.detail == {"owner": "nobody", "owner_number": 2}
    assert not h.app.takeover_open
    state = h.client.wait_state(lambda s: s.owner_number == 2)
    assert (state.owner, state.stop_number) == ("nobody", 0)
    assert state.health["human"]["takeover_open"] is False
    _, text, data = h.heard("owner")[-1]
    assert HANDED_BACK == "takeover ended: the arm is given back; nobody owns it"
    assert (text, data) == (HANDED_BACK, result.detail)
    # It moved nothing and cut nothing.
    assert h.arm.sent == [] and h.arm.jaw_commands == []
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.motion_state is MotionState.REST_HOLD
    # The agent's next task is taken.
    assert h.new_task().status == "completed"


def test_a_handback_away_from_rest_is_refused_and_changes_nothing(raised_app):
    h = raised_app
    take_over(h)
    before = h.app.snapshot()
    refused = hand_back(h)
    assert statuses(h, refused.command_id) == ["accepted", "refused"]
    assert refused.reason == (
        "the arm is not at rest: joint index 0 is at 0.2000 rad, rest is 0.0000 "
        "rad (tolerance 0.05 rad); type rest first"
    )
    told = f"owner request 'handback' refused: {refused.reason}"
    assert h.heard("refused")[-1][1] == told
    assert h.app.snapshot() == before and h.app.takeover_open
    assert h.arm.sent == [] and h.arm.torque_on()
    # After the typed rest the same request ends the takeover.
    assert final(h, h.client.rest()).status == "completed"
    assert hand_back(h).status == "completed"
    assert not h.app.takeover_open and h.app.snapshot().owner is Owner.NOBODY


def test_a_handback_with_loaded_jaws_names_stop_and_open(human_app):
    h = human_app
    take_over(h)
    # The jaws closed on something before the person took over.
    h.app._jaw_state.note_command_result(0.0, "plain", GripperOutcome("stalled", 0.4))
    before = h.app.snapshot()
    refused = hand_back(h)
    assert refused.status == "refused"
    assert refused.reason == (
        "the jaws are not proven empty (loaded): the jaws may be holding "
        "something; type stop, then open, then yes, then hold right A again"
    )
    assert h.app.snapshot() == before and h.app.takeover_open
    # What the text says: stop, then the guarded opening, then the hand-back.
    h.client.stop("typed stop")
    h.client.wait_state(lambda s: s.stop_number == 1)
    assert final(h, h.client.open_jaws()).status == "completed"
    assert hand_back(h).status == "completed"
    assert not h.app.takeover_open


def test_a_handback_after_a_stop_is_taken_from_nobody(human_app):
    h = human_app
    take_over(h)
    h.client.stop("right B", sender="controller")
    h.client.wait_state(lambda s: s.stop_number == 1)
    before = h.app.snapshot()
    assert before.owner is Owner.NOBODY and h.app.takeover_open
    result = hand_back(h)
    assert result.status == "completed"
    # Nobody owned the arm: the owner number stands.
    assert result.detail == {"owner": "nobody", "owner_number": before.owner_number}
    assert h.app.snapshot() == before
    assert not h.app.takeover_open
    assert h.heard("owner")[-1][1] == HANDED_BACK
    assert h.new_task().status == "completed"


def test_a_handback_without_a_takeover_is_refused(human_app):
    h = human_app
    before = h.app.snapshot()
    assert NO_TAKEOVER == "no takeover is open"
    assert_refused(h, hand_back(h), NO_TAKEOVER, "handback")
    assert h.app.snapshot() == before
    # The agent's arm is not the person's to give back either.
    h.new_task()
    assert_refused(h, hand_back(h), NO_TAKEOVER, "handback")
    assert h.app.snapshot().owner is Owner.AGENT


def test_a_handback_is_refused_while_a_return_to_rest_runs(raised_app):
    h = raised_app
    take_over(h)
    reached, release = h.hold_move_after(5)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    assert_refused(h, hand_back(h), "control owner is recovery", "handback")
    assert h.app.takeover_open
    release.set()
    assert final(h, rest_id).status == "completed"


def test_a_stop_during_the_handbacks_reads_keeps_the_takeover_open(human_app):
    h = human_app
    take_over(h)
    reached, release = h.hold_call("read_all")
    command_id = h.client.request_owner("handback", reason=RIGHT_A)
    assert reached.wait(WAIT_S)  # accepted; the joints are about to be read
    assert h.app.stop("typed stop") == 1
    release.set()
    refused = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "refused"]
    assert refused.reason == RACED
    assert h.heard("refused")[-1][1] == f"owner request 'handback' refused: {RACED}"
    assert h.app.takeover_open
    assert h.app.snapshot().owner is Owner.NOBODY
    assert len(h.heard("owner")) == 1  # the takeover's
    # Asked again on the new stop number, it ends the takeover.
    h.client.wait_state(lambda s: s.stop_number == 1)
    assert hand_back(h).status == "completed"
    assert not h.app.takeover_open


# A typed rest during a takeover


def test_a_typed_rest_during_a_takeover_gives_the_arm_back_to_the_person(raised_app):
    h = raised_app
    take_over(h)
    result = final(h, h.client.rest())
    assert result.status == "completed", result.reason
    assert len(h.sent()) == RETURN_SETPOINTS
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.motion_state is MotionState.REST_HOLD
    # Recovery had the arm for the return; the person has it again.
    state = h.app.snapshot()
    assert (state.owner, state.owner_number, state.stop_number) == (Owner.HUMAN, 3, 0)
    assert h.app.takeover_open
    assert h.arm.torque_on() and h.arm.released_at == []
    assert final(h, h.client.request_owner("agent")).reason == TAKEOVER_OPEN


def test_a_return_to_rest_sets_the_needs_release_flag(raised_app):
    h = raised_app
    hand = person(h)
    ended = drives(h)
    assert human_facts(h)["needs_release"] is False
    assert final(h, h.client.rest()).status == "completed"
    sent = len(h.arm.sent)
    assert sent == RETURN_SETPOINTS
    assert h.app.snapshot().owner is Owner.HUMAN
    assert human_facts(h)["needs_release"] is True
    # A grip that was squeezed during the return moves nothing afterwards.
    hand.squeeze(moved(REST))
    h.idle()
    assert ended == [] and len(h.arm.sent) == sent
    assert human_facts(h)["refusal"] == NEEDS_RELEASE
    # Released, then squeezed again: the arm follows.
    hand.release()
    feed(h, hand())
    assert human_facts(h)["needs_release"] is False
    drive(h, hand, moved(REST))
    assert [end.status for end in ended] == ["released"]
    assert len(h.arm.sent) == sent + 6


# A human target on the Thor


def test_an_engaged_target_starts_one_drive(human_app):
    h = human_app
    hand = person(h)
    ended = drives(h)
    shown: list[RobotState] = []
    fed = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        fed(q)
        shown.append(h.app._publisher.build(h.clock.monotonic()))

    h.arm.send_target = send_target
    goal = moved(REST)
    drive(h, hand, goal, sends=6)
    # One drive, however many engaged targets arrived while it ran.
    assert [(end.status, end.sent) for end in ended] == [("released", 6)]
    sent = h.sent()
    assert len(sent) == 6 and sent[-1] == goal
    # One speed-limited step per period, from the measured pose.
    human = h.config.limits.human
    most = SPEED_MARGIN * human.max_joint_speed * h.config.executor.period_s
    steps = [abs(b[0] - a[0]) for a, b in zip([REST, *sent], sent)]
    assert steps[0] == pytest.approx(most) and max(steps) <= most + 1e-12
    assert all(q[1:] == REST[1:] for q in sent)
    # While it ran ROBOT_STATE named the human drive and counted its setpoints.
    progress = [(s.motion, s.move_id, s.move_index, s.move_total) for s in shown]
    assert progress == [("moving", "human", index, 0) for index in range(6)]
    assert all(s.health["human"]["driving"] for s in shown)
    assert all(s.teleop_stage == "ready" for s in shown)
    # The first step was cut to the speed clamp, and counted; no other was.
    assert [s.health["clamped"] for s in shown] == [1] * 6
    # Afterwards the arm holds, the person owns it and nothing told them off.
    state = state_now(h)
    assert (state.motion, state.move_id, state.owner) == ("hold", "", "human")
    facts = state.health["human"]
    assert facts["driving"] is False
    assert facts["ended"] == (ended[0].reason or "released")
    assert (facts["needs_release"], facts["refusal"]) == (False, "")
    assert h.heard("human") == []
    assert h.arm.torque_on() and h.arm.released_at == []
    # A new squeeze starts the next drive.
    drive(h, hand, moved(goal), sends=4)
    assert [end.status for end in ended] == ["released", "released"]
    assert len(h.arm.sent) == 10


def test_a_target_with_old_numbers_starts_nothing(human_app):
    h = human_app
    person(h)
    ended = drives(h)
    now = h.app.snapshot()
    goal = moved(REST)
    cases = {
        "owner number 0 does not match the current 1": {"owner_number": 0},
        "stop number 4 does not match the current 0": {"stop_number": 4},
    }
    for reason, old in cases.items():
        feed(h, setpoint(h, goal, **old))
        h.idle()
        assert human_facts(h)["refusal"] == reason
    # A HUMAN_SETPOINT of another session is dropped: not even its stage shows.
    newest = h.app._human_input.newest()
    feed(h, setpoint(h, goal, session_id="an-earlier-run", stage="stale"))
    h.idle()
    assert h.app._human_input.newest() is newest
    # After a STOP nobody owns the arm, whatever a target quotes.
    h.client.stop("typed stop")
    h.client.wait_state(lambda s: s.stop_number == 1)
    feed(h, setpoint(h, goal))
    h.idle()
    assert human_facts(h)["refusal"] == "control owner is nobody, not human"
    feed(h, setpoint(h, goal, owner_number=now.owner_number, stop_number=0))
    h.idle()
    assert human_facts(h)["refusal"] == "stop number 0 does not match the current 1"
    assert ended == [] and h.arm.sent == []
    assert h.arm.torque_on()


def test_an_engaged_target_without_usable_joint_values_starts_nothing(human_app):
    h = human_app
    person(h)
    ended = drives(h)
    feed(h, setpoint(h, [0.05] * 5))
    h.idle()
    assert human_facts(h)["refusal"] == "the target has 5 values, the arm 6 joints"
    feed(h, setpoint(h, None, engaged=True, grip_released=False))
    h.idle()
    assert human_facts(h)["refusal"] == "the engaged target carries no joint values"
    assert ended == [] and h.arm.sent == []
    # A target that is not engaged ends the refusal.
    feed(h)
    assert human_facts(h)["refusal"] == ""


def test_an_engaged_target_without_fresh_controller_data_starts_nothing(human_app):
    h = human_app
    hand = person(h)
    ended = drives(h)
    goal = moved(REST)
    # What only a human program at fault sends with the clutch engaged.
    cases = {
        "stage stale": {"stage": "stale"},
        "stage no_controller": {"stage": "no_controller"},
        "stage unknown": {"stage": ""},
        "the sample age is unknown": {"sample_age_s": None},
        "0.300 s old, the limit is 0.25 s": {"sample_age_s": 0.3},
    }
    for why, fields in cases.items():
        feed(h, hand())  # fresh data that show the grip released
        assert human_facts(h)["needs_release"] is False, why
        feed(h, setpoint(h, goal, **fields))
        h.idle()
        assert human_facts(h)["refusal"] == f"no fresh controller data: {why}"
        # The squeeze that is still down once the data are fresh moves nothing.
        hand.squeeze(goal)
        h.idle()
        assert human_facts(h)["refusal"] == NEEDS_RELEASE, why
        hand.release()
    assert ended == [] and h.arm.sent == []
    assert h.heard("human") == []
    # Released and squeezed on fresh data, the arm follows.
    feed(h, hand())
    drive(h, hand, goal)
    assert [end.status for end in ended] == ["released"]


def test_a_typed_rest_is_refused_while_a_drive_runs(human_app):
    h = human_app
    hand = person(h)
    ended = drives(h)
    reached, release = h.hold_move_after(3)
    hand.squeeze(moved(REST))
    assert reached.wait(WAIT_S)  # the grip is squeezed and the arm follows
    assert h.app.motion_state is MotionState.MOVING
    refused = final(h, h.client.rest())
    assert (refused.status, refused.reason) == ("refused", BUSY_DRIVING)
    assert statuses(h, refused.command_id) == ["refused"]
    # So are a release, a shutdown and the hand-back, and the Thor terminal's rest.
    assert final(h, h.client.release()).reason == BUSY_DRIVING
    assert final(h, h.client.shutdown()).reason == BUSY_DRIVING
    assert final(h, h.client.request_owner("handback")).reason == BUSY_DRIVING
    done = Done()
    assert h.app.request_rest(True, done) == BUSY_DRIVING
    assert done.wait().status == "refused"
    # With the grip released the arm holds and the rest is taken.
    hand.release()
    release.set()
    h.idle()
    assert [end.status for end in ended] == ["released"]
    assert len(h.arm.sent) == 4
    assert final(h, h.client.rest()).status == "completed"
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_return_and_a_release_see_a_human_drive_as_a_running_move(human_app):
    h = human_app
    hand = person(h)
    ended = drives(h)
    reached, release = h.hold_move_after(3)
    hand.squeeze(moved(REST))
    assert reached.wait(WAIT_S)  # the grip is squeezed and the arm follows
    before = h.app.snapshot()
    # Asked of the return to rest itself, past the app's own "busy" refusal.
    assert h.app._rest._move_running() is True
    outcome = h.app._rest.request(rest_module.TYPED, True)
    assert (outcome.status, outcome.reason) == (
        rest_module.REFUSED,
        "a move is running",
    )
    assert h.app._rest.release(before.stop_number) == "a move is running"
    assert h.app.snapshot() == before  # the person still owns the arm
    assert h.arm.torque_on() and h.arm.released_at == []
    hand.release()
    release.set()
    h.idle()
    assert [end.status for end in ended] == ["released"]
    assert h.app._rest._move_running() is False


def test_a_restarted_human_program_with_a_lower_seq_is_taken(human_app):
    h = human_app
    take_over(h)
    ended = drives(h)
    feed(h, setpoint(h, seq=5000))
    assert human_facts(h)["seq"] == 5000
    # The human program starts again and counts from 1.
    seqs = itertools.count(1)
    hand = Hand(h)
    keep_human_fed(h, lambda: dataclasses.replace(hand(), seq=next(seqs)))
    feed(h, setpoint(h, seq=next(seqs)))
    assert human_facts(h)["seq"] == 1
    assert human_facts(h)["needs_release"] is False
    hand.goal = moved(REST)
    hand.release_at = 5
    feed(h, setpoint(h, hand.goal, seq=next(seqs)))
    h.idle()
    assert [(end.status, end.sent) for end in ended] == [("released", 5)]
    assert h.sent()[-1] == moved(REST)
    assert h.app._human_input.newest().seq < 20


# The needs-release flag: the arm follows only a squeeze that began after a
# released grip the Thor saw, with the person the owner


def test_a_grip_squeezed_before_the_takeover_moves_nothing_until_released(
    human_app,
):
    h = human_app
    # The grip is squeezed while the arm is taken over.
    feed(h, setpoint(h, grip_released=False))
    assert ask_takeover(h).status == "completed"
    h.client.wait_state(lambda s: s.owner == "human")
    hand = Hand(h)
    keep_human_fed(h, hand)
    ended = drives(h)
    hand.squeeze(moved(REST))
    h.idle()
    assert ended == [] and h.arm.sent == []
    facts = human_facts(h)
    assert NEEDS_RELEASE == "release the grip, then squeeze it to move"
    assert (facts["needs_release"], facts["refusal"]) == (True, NEEDS_RELEASE)
    # Released, then squeezed: the arm follows.
    hand.release()
    feed(h, hand())
    assert human_facts(h)["needs_release"] is False
    drive(h, hand, moved(REST))
    assert [end.status for end in ended] == ["released"]
    assert h.sent()[-1] == moved(REST)


def test_a_drive_that_did_not_end_by_a_released_grip_needs_one_first(human_app):
    h = human_app
    hand = person(h)
    ended = drives(h)
    goal = moved(REST)
    hand.squeeze(goal)
    # The human program ends the engagement with the grip still squeezed.
    hand.fields = {"engaged": False, "grip_released": False, "note": "no IK solution"}
    h.idle()
    assert [(end.status, end.reason) for end in ended] == [
        ("disengaged", "no IK solution")
    ]
    sent = len(h.arm.sent)
    assert sent >= 1
    _, text, data = h.heard("human")[-1]
    assert text == (
        "human drive ended: no IK solution; the arm holds; release the grip and "
        "squeeze again"
    )
    assert data == {"status": "disengaged", "reason": "no IK solution", "sent": sent}
    assert h.client.wait_event("human").text == text
    facts = human_facts(h)
    assert (facts["ended"], facts["needs_release"]) == ("no IK solution", True)
    # The squeeze that is still down moves nothing.
    hand.fields = {}
    hand.squeeze(goal)
    h.idle()
    assert len(ended) == 1 and len(h.arm.sent) == sent
    assert human_facts(h)["refusal"] == NEEDS_RELEASE
    # The person keeps the arm, which holds with its motors on.
    assert h.app.snapshot().owner is Owner.HUMAN
    assert h.app.motion_state is MotionState.HOLD and h.arm.torque_on()
    # Released and squeezed again, it follows.
    hand.release()
    feed(h, hand())
    drive(h, hand, moved(goal, 0.03))
    assert [end.status for end in ended] == ["disengaged", "released"]
    assert human_facts(h)["ended"] == (ended[1].reason or "released")


def test_the_event_of_an_ended_drive_says_release_and_squeeze_once(human_app):
    h = human_app
    hand = person(h)
    ended = drives(h)
    phrase = "release the grip and squeeze again"

    def ended_with(note: str) -> tuple[str, dict[str, Any]]:
        """One drive that the human program ends with this note; its EVENT."""
        hand.fields = {}
        hand.release()
        feed(h, hand())
        events = len(h.heard("human"))
        hand.squeeze(moved(h.arm.pose(), 0.03))
        hand.fields = {"engaged": False, "grip_released": False, "note": note}
        h.idle()
        assert (ended[-1].status, ended[-1].reason) == ("disengaged", note)
        assert len(h.heard("human")) == events + 1
        _, text, data = h.heard("human")[-1]
        return text, data

    # Its note of a drive it ended begins with the phrase.
    text, data = ended_with(f"{phrase}: no IK solution")
    assert text == f"human drive ended: no IK solution; the arm holds; {phrase}"
    # The EVENT link message has the phrase once, its text and its data together.
    sent = ended[-1].sent
    assert data == {"status": "disengaged", "reason": "no IK solution", "sent": sent}
    event = h.client.wait_event("human")
    assert (event.text, event.data) == (text, data)
    assert f"{event.text} {event.data}".count(phrase) == 1
    assert human_facts(h)["ended"] == "no IK solution"
    # A note that does not begin with the phrase is shown whole.
    whole = "the hand is 7.0 cm ahead of the arm's target"
    text, data = ended_with(whole)
    assert text == f"human drive ended: {whole}; the arm holds; {phrase}"
    assert data["reason"] == whole
    assert human_facts(h)["ended"] == whole
    # The phrase with no reason after it: the drive's status stands in for the
    # text; the data keep the status and the reason apart.
    text, data = ended_with(f"{phrase}: ")
    assert text == f"human drive ended: disengaged; the arm holds; {phrase}"
    assert (data["status"], data["reason"]) == ("disengaged", "")
    assert f"{text} {data}".count(phrase) == 1
    assert human_facts(h)["ended"] == "disengaged"


def test_the_prefix_the_robot_app_removes_is_the_one_the_human_program_writes():
    # The robot app imports nothing of the human program, so each holds the
    # text; the human program's is read here.
    from isaaccapture_examples.agentic_remote.human import logic

    assert NOTE_SQUEEZE_AGAIN == logic.NOTE_SQUEEZE_AGAIN


def test_engaged_false_alone_does_not_clear_the_needs_release_flag(human_app):
    h = human_app
    take_over(h)
    hand = Hand(h)
    keep_human_fed(h, hand)
    ended = drives(h)
    assert human_facts(h)["needs_release"] is True
    # The clutch lets go and the grip stays squeezed.
    feed(h, setpoint(h, engaged=False, grip_released=False))
    assert human_facts(h)["needs_release"] is True
    hand.squeeze(moved(REST))
    h.idle()
    assert ended == [] and h.arm.sent == []
    assert human_facts(h)["refusal"] == NEEDS_RELEASE
    # A target that claims both a released grip and an engaged clutch is no
    # released grip either.
    feed(h, setpoint(h, moved(REST), grip_released=True))
    h.idle()
    assert ended == [] and h.arm.sent == []
    assert human_facts(h)["needs_release"] is True


def test_a_released_grip_counts_only_with_fresh_data_and_the_current_numbers(
    human_app,
):
    h = human_app
    # Before the takeover nobody owns the arm: a released grip is not noted.
    feed(h)
    assert human_facts(h)["needs_release"] is True
    take_over(h)
    now = h.app.snapshot()
    not_counted = [
        {"stage": "stale"},
        {"stage": "no_controller"},
        {"sample_age_s": None},
        {"sample_age_s": 0.3},  # above human.sample_age_limit_s
        {"owner_number": now.owner_number - 1},
        {"stop_number": now.stop_number + 1},
    ]
    for fields in not_counted:
        feed(h, setpoint(h, **fields))
        assert human_facts(h)["needs_release"] is True, fields
    hand = Hand(h)
    keep_human_fed(h, hand)
    ended = drives(h)
    hand.squeeze(moved(REST))
    h.idle()
    assert ended == [] and h.arm.sent == []
    # Fresh data that quote the current numbers count.
    hand.release()
    feed(h, hand())
    assert human_facts(h)["needs_release"] is False
    drive(h, hand, moved(REST))
    assert [end.status for end in ended] == ["released"]


def test_a_squeeze_held_while_the_controller_data_were_away_moves_nothing(human_app):
    h = human_app
    hand = person(h)
    ended = drives(h)
    assert human_facts(h)["needs_release"] is False
    # Two seconds without a HUMAN_SETPOINT; the grip is squeezed meanwhile.
    h.advance(2.0, 0.5)
    assert human_facts(h)["needs_release"] is True
    hand.squeeze(moved(REST))
    h.idle()
    assert ended == [] and h.arm.sent == []
    assert human_facts(h)["refusal"] == NEEDS_RELEASE
    hand.release()
    feed(h, hand())
    assert human_facts(h)["needs_release"] is False
    # The controller is reported gone and comes back with the grip squeezed.
    gone = {"stage": "no_controller", "sample_age_s": None, "grip_released": False}
    feed(h, setpoint(h, **gone))
    assert human_facts(h)["needs_release"] is True
    hand.squeeze(moved(REST))
    h.idle()
    assert ended == [] and h.arm.sent == []
    assert human_facts(h)["refusal"] == NEEDS_RELEASE
    # Data that come back with the grip released cost nothing.
    hand.release()
    h.advance(2.0, 0.5)
    feed(h, hand())
    assert human_facts(h)["needs_release"] is False
    drive(h, hand, moved(REST))
    assert [end.status for end in ended] == ["released"]


def test_lost_joint_feedback_during_a_drive_is_a_fault_stop_with_the_motors_on(
    human_app,
):
    h = human_app
    hand = person(h)
    ended = drives(h)
    fed = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        fed(q)
        if len(h.arm.sent) == 2:
            h.arm.fail_reads(0, 10_000)  # joint index 0 answers no more

    h.arm.send_target = send_target
    hand.squeeze(moved(REST))  # the grip stays squeezed
    h.idle()
    assert [end.status for end in ended] == ["failed"]
    lost = "joint feedback lost: joint index 0 has had no good reading"
    assert ended[0].reason.startswith(f"fault: {lost}")
    # The drive's end is a fault-stop: latched, nobody owns the arm, motors on.
    state = h.app.snapshot()
    assert (state.stop_level, state.owner) == (StopLevel.FAULT, Owner.NOBODY)
    assert state.fault_reason.startswith(lost)
    assert h.heard("fault")[0][1].startswith(f"FAULT-STOP: {lost}")
    assert h.heard("human")[-1][1].startswith(f"human drive ended: fault: {lost}")
    assert h.app.takeover_open
    assert h.app.motion_state is MotionState.HOLD
    assert h.arm.torque_on() and h.arm.released_at == []
    # The squeeze that is still down starts nothing.
    sent = len(h.arm.sent)
    hand.squeeze(moved(REST))
    h.idle()
    assert len(ended) == 1 and len(h.arm.sent) == sent


# Placement: the Thor's side takes a human target, whoever made it


def test_a_drive_runs_on_a_target_offered_without_a_link_message(
    make_robot_app, config
):
    unseen = human_config(config, require_live_view=False)
    h = make_robot_app(config=unseen, driver="sim")
    ended = drives(h)
    seqs = itertools.count(1)

    def offer(q: Sequence[float] | None = None) -> None:
        """A feed inside the robot app's process: the numbers of app.snapshot()."""
        now = h.app.snapshot()
        h.app.offer_human_target(
            HumanTarget(
                seq=next(seqs),
                owner_number=now.owner_number,
                stop_number=now.stop_number,
                source=HUMAN_SOURCE_TELEOP,
                stage="ready",
                note="",
                engaged=q is not None,
                grip_released=q is None,
                q=None if q is None else tuple(q),
                sample_age_s=0.02,
                arrived_mono=h.clock.monotonic(),
            )
        )

    offer()
    assert ask_takeover(h).status == "completed"
    offer()  # the grip is released with the person the owner
    goal = moved(REST)
    real = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        offer(goal if len(h.arm.sent) < 5 else None)

    h.arm.send_target = send_target
    offer(goal)
    h.idle()
    assert [(end.status, end.sent) for end in ended] == [("released", 5)]
    assert h.sent()[-1] == goal
    # ROBOT_STATE shows the offered targets as it shows a HUMAN_SETPOINT's.
    state = state_now(h)
    assert (state.teleop_stage, state.health["human"]["seq"]) == ("ready", 8)
    assert state.owner == "human" and h.arm.torque_on()


def test_a_human_drive_sends_no_jaw_command(human_app):
    h = human_app
    jaw_calls: list[str] = []
    for name in ("submit", "open_guarded", "reference"):
        real = getattr(h.app._worker, name)

        def noted(*args, _name=name, _real=real, **kwargs):
            jaw_calls.append(_name)
            return _real(*args, **kwargs)

        setattr(h.app._worker, name, noted)
    # The trigger is pulled all the way: every HUMAN_SETPOINT asks for closed jaws.
    hand = person(h, jaw_opening=0.0)
    ended = drives(h)
    opening = h.arm.gripper_read()
    drive(h, hand, moved(REST), sends=8)
    assert [(end.status, end.sent) for end in ended] == [("released", 8)]
    assert jaw_calls == [] and h.arm.jaw_commands == []
    assert h.arm.gripper_read() == opening


# The robot connection is lost while a person owns the arm


def test_a_lost_connection_ends_the_drive_and_the_person_keeps_the_arm(human_app):
    h = human_app
    hand = person(h)
    ended = drives(h)
    reached, release = h.hold_move_after(3)
    goal = moved(REST)
    hand.squeeze(goal)
    assert reached.wait(WAIT_S)
    h.lose_link()
    _, text, data = h.heard("connection")[-1]
    assert text == (
        "robot connection lost: the arm holds; the person keeps the arm: squeeze "
        "again when the connection is back; rest can still be typed at the Thor "
        "terminal"
    )
    assert data == {"owner_before": "human", "wait_cancelled": False}
    release.set()
    h.idle()
    assert [(end.status, end.sent) for end in ended] == [("stopped", 3)]
    assert "robot connection lost" in ended[0].reason
    assert len(h.arm.sent) == 3 and h.arm.last_target()[0] == h.sent()[-1]
    assert h.heard("human") == []  # the connection's own event told it
    state = h.app.snapshot()
    assert (state.owner, state.owner_number, state.stop_number) == (Owner.HUMAN, 1, 0)
    assert h.app.takeover_open
    assert h.app.motion_state is MotionState.HOLD
    assert h.arm.torque_on() and h.arm.released_at == []
    # On the next connection the squeeze that is still down moves nothing.
    h.connect()
    hand.squeeze(goal)
    h.idle()
    assert len(ended) == 1 and len(h.arm.sent) == 3
    assert human_facts(h)["refusal"] == NEEDS_RELEASE
    hand.release()
    feed(h, hand())
    drive(h, hand, goal)
    assert [end.status for end in ended] == ["stopped", "released"]


def test_a_lost_connection_forgets_the_human_input(human_app):
    h = human_app
    take_over(h)
    feed(h, setpoint(h, note="squeeze the grip to move"))
    state = state_now(h)
    assert (state.teleop_stage, state.health["human"]["needs_release"]) == (
        "ready",
        False,
    )
    assert state.controller_age_s is not None
    h.lose_link()
    assert h.app._human_input.newest() is None
    assert h.app.snapshot().owner is Owner.HUMAN and h.app.takeover_open
    client = h.connect()
    assert (client.state.teleop_stage, client.state.controller_age_s) == ("", None)
    facts = human_facts(h)
    assert (facts["seq"], facts["note"], facts["source"]) == (None, "", "")
    assert facts["needs_release"] is True
    assert facts["takeover"] == (
        "no fresh controller data: no HUMAN_SETPOINT has arrived"
    )
    # Only a released grip of the new connection counts.
    feed(h)
    assert human_facts(h)["needs_release"] is False


def test_a_drive_pass_while_the_lost_connection_is_handled_ends_stopped(human_app):
    h = human_app
    take_over(h)
    feed(h)  # the grip is released and the view is live
    ended = drives(h)
    reached, forgotten, over = (threading.Event() for _ in range(3))
    run = h.app._human_drive.run

    def noted_run(admitted, should_stop) -> DriveEnd:
        try:
            return run(admitted, should_stop)
        finally:
            over.set()

    h.app._human_drive.run = noted_run
    send = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        send(q)
        if len(h.arm.sent) == 2:
            reached.set()
            forgotten.wait(WAIT_S)

    h.arm.send_target = send_target
    clear = h.app._human_input.clear

    def held_clear() -> None:
        # The human input is forgotten; the drive makes its next pass before
        # the app goes on with the lost connection.
        clear()
        forgotten.set()
        over.wait(WAIT_S)

    h.app._human_input.clear = held_clear
    try:
        h.client.send(setpoint(h, moved(REST)))
        assert reached.wait(WAIT_S)
        h.lose_link()
        h.idle()
    finally:
        forgotten.set()  # no thread stays held when the test fails
        over.set()
    assert [(end.status, end.sent) for end in ended] == [("stopped", 2)]
    assert ended[0].reason == "robot connection lost"
    assert h.heard("human") == []  # the connection's own event told it
    assert h.app.snapshot().owner is Owner.HUMAN and h.arm.torque_on()


def test_rest_typed_at_the_thor_terminal_works_with_the_connection_lost(raised_app):
    h = raised_app
    take_over(h)
    h.lose_link()
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert done.wait().status == "completed"
    h.idle()
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.snapshot().owner is Owner.HUMAN and h.app.takeover_open
    assert h.arm.torque_on() and h.arm.released_at == []


# The assistance wait


def test_status_only_human_setpoints_do_not_cancel_the_assistance_wait(raised_app):
    h = raised_app
    h.new_task()
    ask_for_help(h)
    h.client.wait_state(lambda s: s.assistance_wait_s is not None)
    # The human program reports the headset's status for the whole wait.
    sent = 0
    while not h.heard("expired"):
        assert sent < 2600, "the wait did not run out"
        h.clock.advance(0.05)
        h.client.send(setpoint(h))
        h.step()
        sent += 1
    h.idle()
    assert sent > 1500  # one every 0.05 s of the app's clock for 120 s
    assert h.app._human_input.newest().stage == "ready"
    assert len(h.heard("expired")) == 1
    # The timed return started when the wait ran out.
    assert len(h.sent()) == RETURN_SETPOINTS
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.motion_state is MotionState.REST_HOLD
    assert h.app.snapshot().owner is Owner.NOBODY and not h.app.takeover_open
    assert h.arm.torque_on() and h.arm.released_at == []


# A contact while a takeover is open


def contact_under_a_person(h, hand: Hand, ended: list[DriveEnd]) -> list[float]:
    """The person drives joint index 0 into something; the pose the arm holds at."""
    at = h.arm.pose()[0] + 0.05
    h.arm.block_joint(0, at)
    hand.stride = 0.08  # the hand keeps moving ahead of the arm
    hand.squeeze(moved(h.arm.pose(), 0.08))
    h.idle()
    hand.stride = None
    hand.release()
    assert ended[-1].status == "contact", ended[-1]
    assert "joint index 0 trails its setpoints" in ended[-1].reason
    held = h.arm.holds[-1][0]
    assert held[0] == pytest.approx(at, abs=1e-6)
    return held


def test_a_typed_rest_is_admitted_from_a_contact_hold_while_a_takeover_is_open_and_ends_at_the_next_contact(
    raised_app,
):
    h = raised_app
    hand = person(h)
    ended = drives(h)
    held = contact_under_a_person(h, hand, ended)
    _, text, data = h.heard("human")[-1]
    assert text.startswith("human drive ended: contact")
    assert "joint index 0 trails its setpoints" in text
    assert text.endswith("; the arm holds; release the grip and squeeze again")
    assert data["status"] == "contact" and data["sent"] == ended[-1].sent
    state = state_now(h)
    assert state.health["contact_hold"] is True
    assert (state.owner, state.stop_level, state.motion) == ("human", "none", "hold")
    assert state.health["human"]["needs_release"] is True
    # No contact spot is noted for a person: the spots are the agent's.
    assert all(checker.contact_spots == [] for checker in h.checkers.all())
    # The typed rest is taken from the contact hold. Something lies on its way.
    h.arm.block_joint(0, None)
    h.arm.block_joint(0, 0.12)
    sent = len(h.arm.sent)
    blocked = final(h, h.client.rest())
    assert blocked.status == "failed"
    assert "contact" in blocked.reason
    assert blocked.detail == {"outcome": "failed"}
    assert len(h.arm.sent) > sent  # the return began
    assert h.arm.pose()[0] == pytest.approx(0.12, abs=1e-6)
    assert h.arm.pose()[0] < held[0]
    assert state_now(h).health["contact_hold"] is True
    # The person has the arm again and the motors are on.
    assert h.app.snapshot().owner is Owner.HUMAN and h.app.takeover_open
    assert h.arm.torque_on() and h.arm.released_at == []
    # With the path cleared by hand the next typed rest reaches rest.
    h.arm.block_joint(0, None)
    assert final(h, h.client.rest()).status == "completed"
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert state_now(h).health["contact_hold"] is False
    assert hand_back(h).status == "completed"


def test_a_contact_under_a_person_then_no_controller_still_leaves_typed_rest(
    raised_app,
):
    h = raised_app
    hand = person(h)
    ended = drives(h)
    contact_under_a_person(h, hand, ended)
    h.arm.block_joint(0, None)
    # The headset path is gone: the human program says nothing more.
    hand.silent = True
    h.advance(h.config.human.stage_keep_s + 1.0)
    state = state_now(h)
    assert (state.teleop_stage, state.health["contact_hold"]) == ("", True)
    assert state.health["human"]["takeover"].startswith("no fresh controller data: ")
    # The agent cannot be the one who moves the arm away.
    assert final(h, h.client.request_owner("agent")).reason == TAKEOVER_OPEN
    # Rest typed at the Thor terminal, with the path confirmed clear, is taken.
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert done.wait().status == "completed"
    h.idle()
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.motion_state is MotionState.REST_HOLD
    assert state_now(h).health["contact_hold"] is False
    assert h.app.snapshot().owner is Owner.HUMAN and h.app.takeover_open
    assert h.arm.torque_on() and h.arm.released_at == []
    # Without the typed confirmation nothing is taken, contact hold or not.
    unconfirmed = Done()
    assert h.app.request_rest(False, unconfirmed) == "confirm the path is clear"


def test_without_a_takeover_a_contact_hold_still_refuses_the_rest(human_app):
    h = human_app
    h.new_task()
    h.arm.block_joint(0, 0.05)
    path = [[0.008 * (index + 1)] * 6 for index in range(60)]
    result = h.client.run_move(path, start_q=REST)
    h.arm.block_joint(0, None)
    assert (result.status, result.reason) == ("failed", "contact")
    assert state_now(h).health["contact_hold"] is True
    refused = final(h, h.client.rest())
    assert (refused.status, refused.reason) == ("refused", AT_A_CONTACT)
    # Nobody owning the arm changes nothing while no takeover is open.
    h.client.stop("typed stop")
    h.client.wait_state(lambda s: s.stop_number == 1)
    refused = final(h, h.client.rest())
    assert (refused.status, refused.reason) == ("refused", AT_A_CONTACT)
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert done.wait().reason == AT_A_CONTACT
    assert h.arm.torque_on() and not h.app.takeover_open
    # With a takeover open the same typed rest is taken.
    take_over(h)
    assert final(h, h.client.rest()).status == "completed"
    assert state_now(h).health["contact_hold"] is False
