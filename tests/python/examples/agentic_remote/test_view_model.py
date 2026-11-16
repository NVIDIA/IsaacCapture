# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""view/model.py: when the live view counts as live, and what the page is told.

Every time is a number the test picks: the model reads no clock.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.agent import link_client
from isaaccapture_examples.agentic_remote.agent.camera import stereo_camera
from isaaccapture_examples.agentic_remote.protocol.local_messages import LocalStatus
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Event,
    RobotState,
    ViewStatus,
)
from isaaccapture_examples.agentic_remote.view import model as view_model
from isaaccapture_examples.agentic_remote.view.model import (
    AGE_WHEN_NOT_LIVE_S,
    EVENTS_KEPT,
    FUTURE_TOLERANCE_S,
    NO_PICTURE,
    RECORDED,
    STALE,
    ViewModel,
)
from link_bench import robot_state, welcome

NOT_LIVE = ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)
# How long after a step of the Thor's wall clock, and after the model's first
# ROBOT_STATE, a picture must arrive to count: CLOCK_STEP_QUIET_S.
QUIET = 3.0
# This machine's monotonic clock and the Thor's wall clock at the test's start.
T0 = 500.0
WALL0 = 1.7e9
JPEG = b"\xff\xd8 a picture \xff\xd9"


def near(seconds: float) -> Any:
    # A wall clock time near 1.7e9 s holds steps of about 0.2 microseconds.
    return pytest.approx(seconds, abs=1e-5)


def state_at(wall: float, **fields: Any) -> RobotState:
    """A ROBOT_STATE of that wall time; the Thor's monotonic clock went along."""
    mono = robot_state().clock_mono_s + (wall - WALL0)
    return dataclasses.replace(
        robot_state(), clock_wall_s=wall, clock_mono_s=mono, **fields
    )


def model_with_state(**settings: float) -> ViewModel:
    """A model that took a ROBOT_STATE at T0 whose wall clock said WALL0.

    Its first ROBOT_STATE came a minute before: the pictures of these tests
    arrive long after it.
    """
    model = ViewModel(**settings)
    model.on_message(state_at(WALL0 - 60.0), T0 - 60.0)
    model.on_message(state_at(WALL0), T0)
    return model


def shown(
    model: ViewModel,
    seq: int = 1,
    *,
    exposed_ago: float = 0.1,
    arrived_ago: float = 0.05,
    invented: bool = False,
    visible: bool = True,
    now: float = T0,
) -> None:
    """A picture taken and arrived before `now`, acknowledged by the page at `now`."""
    wall_now = WALL0 + (now - T0)
    model.put_picture(seq, JPEG, wall_now - exposed_ago, now - arrived_ago, invented)
    model.acknowledge(seq, visible, now)


def status(model: ViewModel, now: float) -> ViewStatus:
    return model.view_status(now, model.thor_wall_now(now))


# The four conditions of "live", one by one


def test_not_live_without_an_acknowledged_picture():
    model = model_with_state()
    assert status(model, T0) == NOT_LIVE
    model.put_picture(1, JPEG, WALL0 - 0.1, T0 - 0.05, False)
    assert status(model, T0) == NOT_LIVE
    # An acknowledgement of a picture the model never had counts for nothing.
    model.acknowledge(7, True, T0)
    assert status(model, T0) == NOT_LIVE


def test_live_after_a_visible_acknowledgement():
    model = model_with_state()
    shown(model)
    assert status(model, T0) == ViewStatus(live=True, age_s=near(0.1))


def test_a_hidden_tab_is_not_live():
    model = model_with_state()
    shown(model, visible=False)
    assert status(model, T0) == NOT_LIVE
    # A tab that was visible and is hidden now is not live from that moment.
    shown(model, 2, visible=True)
    assert status(model, T0).live
    model.acknowledge(2, False, T0 + 0.1)
    model.on_message(state_at(WALL0 + 0.1), T0 + 0.1)
    assert status(model, T0 + 0.1) == NOT_LIVE


def test_an_acknowledgement_older_than_the_timeout_is_not_live():
    model = model_with_state(ack_timeout_s=1.0)
    shown(model)
    model.on_message(state_at(WALL0 + 0.9), T0 + 0.9)
    assert status(model, T0 + 0.99).live
    model.on_message(state_at(WALL0 + 1.0), T0 + 1.0)
    assert status(model, T0 + 1.0) == NOT_LIVE
    # The same picture acknowledged again is live again, with its own age.
    model.put_picture(2, JPEG, WALL0 + 1.0, T0 + 1.0, False)
    model.acknowledge(1, True, T0 + 1.1)
    assert status(model, T0 + 1.1) == ViewStatus(live=True, age_s=near(1.2))
    # Once a newer picture was acknowledged, an older one counts for nothing.
    model.acknowledge(2, True, T0 + 1.2)
    model.acknowledge(1, True, T0 + 3.0)
    model.on_message(state_at(WALL0 + 3.0), T0 + 3.0)
    assert status(model, T0 + 3.0) == NOT_LIVE


def test_a_recorded_source_is_never_live():
    model = model_with_state()
    shown(model, invented=True)
    assert status(model, T0) == NOT_LIVE
    assert model.status_json(T0, model.thor_wall_now(T0))["picture"]["overlay"] == (
        RECORDED
    )


def test_not_live_without_a_fresh_robot_state():
    # No ROBOT_STATE at all: the Thor's wall clock is not known.
    model = ViewModel(state_max_age_s=1.0)
    shown(model)
    assert model.thor_wall_now(T0) is None
    assert model.view_status(T0, None) == NOT_LIVE
    assert model.view_status(T0, WALL0) == NOT_LIVE
    # A ROBOT_STATE that is too old.
    model = model_with_state(state_max_age_s=1.0)
    shown(model)
    assert status(model, T0 + 0.5).live
    model.acknowledge(1, True, T0 + 0.9)
    assert status(model, T0 + 0.99).live
    model.acknowledge(1, True, T0 + 1.0)
    assert status(model, T0 + 1.0) == NOT_LIVE
    assert model.thor_wall_now(T0 + 1.0) is None
    # A newer one brings it back.
    model.on_message(state_at(WALL0 + 1.0), T0 + 1.0)
    assert status(model, T0 + 1.0).live


def test_an_exposure_in_the_future_is_not_live():
    model = model_with_state()
    shown(model, exposed_ago=-(FUTURE_TOLERANCE_S + 0.01))
    assert status(model, T0) == NOT_LIVE
    # Inside the tolerance the picture counts, with the arrival age.
    shown(model, 2, exposed_ago=-(FUTURE_TOLERANCE_S - 0.01), arrived_ago=0.02)
    assert status(model, T0) == ViewStatus(live=True, age_s=near(0.02))


# The age


def test_the_age_is_the_larger_of_the_exposure_age_and_the_arrival_age():
    model = model_with_state()
    shown(model, 1, exposed_ago=0.30, arrived_ago=0.10)
    assert status(model, T0).age_s == near(0.30)
    shown(model, 2, exposed_ago=0.10, arrived_ago=0.25)
    assert status(model, T0).age_s == near(0.25)
    # Both ages grow with the time since the acknowledgement: the age is that
    # of the picture on the screen, not of the acknowledgement.
    model.on_message(state_at(WALL0 + 0.4), T0 + 0.4)
    assert status(model, T0 + 0.5) == ViewStatus(live=True, age_s=near(0.75))
    # The Thor's wall clock is taken from the newest ROBOT_STATE: one that says
    # a later time makes the exposure older.
    model.on_message(state_at(WALL0 + 2.0), T0 + 0.5)
    assert status(model, T0 + 0.5).age_s == near(2.1)


def test_the_age_is_of_the_acknowledged_picture_not_of_the_newest_one():
    model = model_with_state()
    shown(model, 1, exposed_ago=0.2, arrived_ago=0.2)
    model.put_picture(2, JPEG, WALL0 + 0.3, T0 + 0.3, False)
    model.on_message(state_at(WALL0 + 0.3), T0 + 0.3)
    assert status(model, T0 + 0.3).age_s == near(0.5)
    model.acknowledge(2, True, T0 + 0.3)
    assert status(model, T0 + 0.3).age_s == near(0.0)


def test_a_not_live_status_carries_the_fixed_age():
    assert AGE_WHEN_NOT_LIVE_S == 3600.0
    model = ViewModel()
    assert model.view_status(T0, None) == ViewStatus(live=False, age_s=3600.0)


# What the page is given


def test_the_newest_picture_is_handed_out_with_its_number():
    model = ViewModel()
    assert model.picture() is None
    model.put_picture(1, b"one", WALL0, T0, False)
    model.put_picture(2, b"two", WALL0, T0, False)
    assert model.picture() == (2, b"two")


def test_the_overlay_says_no_picture_stale_or_recorded():
    model = model_with_state()

    def overlay(now: float) -> str:
        return model.status_json(now, model.thor_wall_now(now))["picture"]["overlay"]

    assert overlay(T0) == NO_PICTURE == "NO PICTURE"
    model.put_picture(1, JPEG, WALL0 - 0.1, T0 - 0.05, False)
    assert overlay(T0) == ""
    assert overlay(T0 + 0.8) == ""
    assert overlay(T0 + 0.96) == STALE == "STALE"
    model.put_picture(2, JPEG, WALL0 + 1.0, T0 + 1.0, True)
    assert overlay(T0 + 1.0) == RECORDED == "RECORDED, NOT LIVE"
    # A recorded picture that stopped coming is stale as well.
    assert overlay(T0 + 3.0) == f"{RECORDED}, {STALE}"


def test_the_status_holds_what_the_page_shows():
    model = ViewModel()
    empty = model.status_json(T0, None)
    assert empty["state"] is None and empty["view_status"] is None
    assert empty["robot"]["connected"] is None
    assert empty["picture"]["seq"] is None and empty["events"] == []

    hello = dataclasses.replace(welcome("s9"), driver="sim")
    model.on_message(hello, T0)
    permit = {
        "move_id": "m-1",
        "through_index": 12,
        "evidence_age_s": 0.4,
        "expires_in_s": 0.6,
    }
    model.on_message(
        state_at(
            WALL0,
            session_id="s9",
            owner="agent",
            owner_number=3,
            stop_number=2,
            stop_level="fault",
            fault_reason="motor 3 is hot",
            motion="moving",
            move_id="m-1",
            move_index=4,
            move_total=20,
            move_stage="sending",
            assistance_wait_s=41.5,
            view_age_s=0.3,
            health={"permit": permit, "permit_refusal": "the grid is too old"},
        ),
        T0,
    )
    model.on_local(
        LocalStatus(True, "", "s9", None, False, {"viewer": 1, "agent": 1}), True
    )
    model.set_camera_note("no camera pictures: the receiver is not there")
    model.put_picture(5, JPEG, WALL0 - 0.2, T0 - 0.1, False)
    model.note_sent(ViewStatus(live=True, age_s=0.2))
    told = model.status_json(T0 + 0.25, model.thor_wall_now(T0 + 0.25))

    assert told["robot"] == {
        "service_alive": True,
        "connected": True,
        "close_reason": "",
        "session_id": "s9",
        "robot_id": "sim-arm",
        "driver": "sim",
        "clients": {"agent": 1, "viewer": 1},
        "stop_unconfirmed": False,
    }
    assert told["state_age_s"] == near(0.25)
    assert told["state"] == {
        "owner": "agent",
        "owner_number": 3,
        "stop_level": "fault",
        "stop_number": 2,
        "fault_reason": "motor 3 is hot",
        "motion": "moving",
        "move_id": "m-1",
        "move_index": 4,
        "move_total": 20,
        "move_stage": "sending",
        "jaw_load": "unknown",
        "torque_on": True,
        "assistance_wait_s": 41.5,
        "permit": permit,
        "permit_refusal": "the grid is too old",
        "view_age_s": 0.3,
    }
    assert told["picture"] == {
        "seq": 5,
        "age_s": near(0.45),
        "overlay": "",
        "note": "no camera pictures: the receiver is not there",
    }
    assert told["view_status"] == {"live": True, "age_s": 0.2}
    json.dumps(told, allow_nan=False)

    # A lost service, and a state without the permit keys.
    model.on_local(None, False)
    model.on_message(state_at(WALL0 + 1.0, health={}), T0 + 1.0)
    told = model.status_json(T0 + 1.0, model.thor_wall_now(T0 + 1.0))
    assert told["robot"]["service_alive"] is False
    assert told["state"]["permit"] is None and told["state"]["permit_refusal"] == ""


def test_the_last_ten_events_are_kept_newest_last():
    assert EVENTS_KEPT == 10
    model = ViewModel()
    for seq in range(1, 14):
        model.on_message(Event(seq, "stop", f"text {seq}", {"n": seq}), T0)
    events = model.status_json(T0, None)["events"]
    assert [event["seq"] for event in events] == list(range(4, 14))
    assert events[-1] == {"seq": 13, "kind": "stop", "text": "text 13"}


def test_a_value_that_is_not_json_does_not_break_the_status():
    model = ViewModel()
    model.on_message(state_at(WALL0, health={"permit": {"evidence_age_s": "x"}}), T0)
    told = model.status_json(T0, None)
    assert told["state"]["permit"] is None
    json.dumps(told, allow_nan=False)


def test_the_model_is_used_from_many_threads():
    model = model_with_state()
    stop = threading.Event()
    problems: list[BaseException] = []

    def work(number: int) -> None:
        try:
            seq = 0
            while not stop.is_set():
                seq += 1
                model.put_picture(seq, JPEG, WALL0, T0, False)
                model.acknowledge(seq, True, T0)
                model.on_message(state_at(WALL0), T0)
                model.status_json(T0, model.thor_wall_now(T0))
                model.view_status(T0, model.thor_wall_now(T0))
                if seq >= 200:
                    break
        except BaseException as problem:
            problems.append(problem)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    stop.set()
    assert problems == []
    assert status(model, T0).live


# A step of the Thor's wall clock


def stepped_state(seconds: float, step: float) -> RobotState:
    """A ROBOT_STATE that long after T0, its wall clock `step` off the old one."""
    old = state_at(WALL0 + seconds)
    return dataclasses.replace(old, clock_wall_s=old.clock_wall_s + step)


def test_a_picture_from_before_a_step_of_the_thors_wall_clock_is_never_live():
    # The same quiet time as the agent's camera keeps after a step.
    assert view_model.CLOCK_STEP_QUIET_S == QUIET == stereo_camera.CLOCK_STEP_QUIET_S
    model = model_with_state(ack_timeout_s=1.0)
    shown(model, now=T0 + 0.1)  # exposed at WALL0, 0.1 s ago
    model.on_message(state_at(WALL0 + 0.15), T0 + 0.15)
    assert status(model, T0 + 0.15) == ViewStatus(live=True, age_s=near(0.15))
    # The Thor's wall clock steps back 0.15 s against its monotonic clock:
    # the picture would count as 0.05 s old when it is 0.2 s old.
    model.on_message(stepped_state(0.2, -0.15), T0 + 0.2)
    assert status(model, T0 + 0.2) == NOT_LIVE
    # It stays out for as long as the page goes on acknowledging it.
    for seconds in (0.9, 1.3, QUIET - 0.1):
        model.on_message(stepped_state(seconds, -0.15), T0 + seconds)
        model.acknowledge(1, True, T0 + seconds)
        assert status(model, T0 + seconds) == NOT_LIVE
    # So does a picture that arrives within the quiet time after the step:
    # its exposure can still be one of the old clock.
    at = 0.2 + QUIET - 0.01
    model.put_picture(2, JPEG, WALL0 + at - 0.15 - 0.1, T0 + at, False)
    for seconds in (at, at + 0.5, 30.0):
        model.on_message(stepped_state(seconds, -0.15), T0 + seconds)
        model.acknowledge(2, True, T0 + seconds)
        assert status(model, T0 + seconds) == NOT_LIVE
    # A picture that arrives after it is live, with its age on the new clock.
    at = 31.0
    model.on_message(stepped_state(at, -0.15), T0 + at)
    model.put_picture(3, JPEG, WALL0 + at - 0.15 - 0.1, T0 + at - 0.02, False)
    model.acknowledge(3, True, T0 + at)
    assert status(model, T0 + at) == ViewStatus(live=True, age_s=near(0.1))
    # A step forward counts the same.
    model.on_message(stepped_state(at + 0.1, 0.1), T0 + at + 0.1)
    model.acknowledge(3, True, T0 + at + 0.1)
    assert status(model, T0 + at + 0.1) == NOT_LIVE


@pytest.mark.parametrize("ack_timeout_s", [0.3, 1.0])
def test_a_shorter_ack_timeout_does_not_let_a_picture_from_before_the_step_in(
    ack_timeout_s: float,
) -> None:
    model = model_with_state(ack_timeout_s=ack_timeout_s)
    # Exposed 0.05 s before the Thor's wall clock steps back 0.4 s; it takes
    # 0.25 s to arrive, after the ROBOT_STATE that shows the step.
    model.on_message(stepped_state(0.06, -0.4), T0 + 0.06)
    model.put_picture(1, JPEG, WALL0, T0 + 0.25, False)
    # The camera stalls, and the page acknowledges that picture every 0.2 s.
    for tenth in range(3, 80, 2):
        now = T0 + tenth / 10.0
        model.on_message(stepped_state(tenth / 10.0, -0.4), now)
        model.acknowledge(1, True, now)
        assert status(model, now) == NOT_LIVE, f"{now - T0:.1f} s after the exposure"


def test_a_step_before_the_models_first_robot_state_is_not_missed():
    model = ViewModel(ack_timeout_s=1.0)
    # The camera connection is up before the robot connection: a picture
    # exposed at WALL0 arrives and is acknowledged.
    model.put_picture(1, JPEG, WALL0, T0 + 0.05, False)
    model.acknowledge(1, True, T0 + 0.1)
    # The first ROBOT_STATE comes 0.2 s after that exposure, from a wall clock
    # that stepped back 0.1 s meanwhile: nothing in the model shows the step.
    model.on_message(stepped_state(0.2, -0.1), T0 + 0.2)
    assert status(model, T0 + 0.2) == NOT_LIVE
    for seconds in (0.25, 0.9, 1.1, QUIET, 20.0):
        model.on_message(stepped_state(seconds, -0.1), T0 + seconds)
        model.acknowledge(1, True, T0 + seconds)
        assert status(model, T0 + seconds) == NOT_LIVE
    # The view is live once a picture arrives the quiet time after that state.
    at = 0.2 + QUIET
    model.on_message(stepped_state(at, -0.1), T0 + at)
    model.put_picture(2, JPEG, WALL0 + at - 0.1 - 0.05, T0 + at, False)
    model.acknowledge(2, True, T0 + at)
    assert status(model, T0 + at) == ViewStatus(live=True, age_s=near(0.05))
    # A robot connection that comes back on the same clocks marks nothing.
    again = dataclasses.replace(stepped_state(at + 0.1, -0.1), session_id="s2")
    model.on_message(again, T0 + at + 0.1)
    assert status(model, T0 + at + 0.1).live


def test_a_change_within_the_links_threshold_is_no_step():
    assert view_model.CLOCK_STEP_S is link_client.CLOCK_STEP_S == 0.02
    model = model_with_state()
    shown(model)
    drifted = dataclasses.replace(state_at(WALL0 + 0.1), clock_wall_s=WALL0 + 0.119)
    model.on_message(drifted, T0 + 0.1)
    assert status(model, T0 + 0.1).live
