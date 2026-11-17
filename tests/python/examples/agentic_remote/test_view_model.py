# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""view/model.py: when the live view counts as live, and what the page is told.

Every time is a number the test picks: the model reads no clock. The view
token is the secret a page gets when it is loaded; only the page loaded last
has the valid one.
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.agent import link_client
from isaaccapture_examples.agentic_remote.agent.camera import stereo_camera
from isaaccapture_examples.agentic_remote.protocol.local_messages import LocalStatus
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Event,
    GripperState,
    JointSample,
    RobotState,
    ViewStatus,
)
from isaaccapture_examples.agentic_remote.view import model as view_model
from isaaccapture_examples.agentic_remote.view.model import (
    AGE_WHEN_NOT_LIVE_S,
    EVENTS_KEPT,
    FUTURE_TOLERANCE_S,
    HUMAN_DRIVING,
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


class Page:
    """A page that was loaded on a model, with the view token it was given."""

    def __init__(self, model: ViewModel, now: float = T0 - 30.0) -> None:
        self.model = model
        self.token = model.new_page(now)

    def fetch(self, seq: int, now: float) -> bool:
        """The page is served picture `seq`."""
        return self.model.note_served(seq, self.token, now)

    def acknowledge(self, seq: int, visible: bool, now: float) -> bool:
        return self.model.acknowledge(seq, visible, now, self.token)

    def draw(self, seq: int, now: float, visible: bool = True) -> bool:
        """What the page does with a picture: it fetches and acknowledges it."""
        return self.fetch(seq, now) and self.acknowledge(seq, visible, now)


def shown(
    page: Page,
    seq: int = 1,
    *,
    exposed_ago: float = 0.1,
    arrived_ago: float = 0.05,
    invented: bool = False,
    visible: bool = True,
    now: float = T0,
) -> None:
    """A picture taken and arrived before `now`, drawn by the page at `now`."""
    wall_now = WALL0 + (now - T0)
    page.model.put_picture(
        seq, JPEG, wall_now - exposed_ago, now - arrived_ago, invented
    )
    assert page.draw(seq, now, visible)


def status(model: ViewModel, now: float) -> ViewStatus:
    return model.view_status(now, model.thor_wall_now(now))


# The four conditions of "live", one by one


def test_not_live_without_an_acknowledged_picture():
    model = model_with_state()
    page = Page(model)
    assert status(model, T0) == NOT_LIVE
    model.put_picture(1, JPEG, WALL0 - 0.1, T0 - 0.05, False)
    assert status(model, T0) == NOT_LIVE
    # A picture the page was served and did not acknowledge counts for nothing.
    assert page.fetch(1, T0)
    assert status(model, T0) == NOT_LIVE
    # So does an acknowledgement of a picture the model never had.
    assert page.fetch(7, T0) and not page.acknowledge(7, True, T0)
    assert status(model, T0) == NOT_LIVE


def test_live_after_a_visible_acknowledgement():
    model = model_with_state()
    shown(Page(model))
    assert status(model, T0) == ViewStatus(live=True, age_s=near(0.1))


def test_a_hidden_tab_is_not_live():
    model = model_with_state()
    page = Page(model)
    shown(page, visible=False)
    assert status(model, T0) == NOT_LIVE
    # A tab that was visible and is hidden now is not live from that moment.
    shown(page, 2, visible=True)
    assert status(model, T0).live
    assert page.acknowledge(2, False, T0 + 0.1)
    model.on_message(state_at(WALL0 + 0.1), T0 + 0.1)
    assert status(model, T0 + 0.1) == NOT_LIVE


def test_an_acknowledgement_older_than_the_timeout_is_not_live():
    model = model_with_state(ack_timeout_s=1.0)
    page = Page(model)
    shown(page)
    model.on_message(state_at(WALL0 + 0.9), T0 + 0.9)
    assert status(model, T0 + 0.99).live
    model.on_message(state_at(WALL0 + 1.0), T0 + 1.0)
    assert status(model, T0 + 1.0) == NOT_LIVE
    # The same picture drawn again is live again, with its own age.
    model.put_picture(2, JPEG, WALL0 + 1.0, T0 + 1.0, False)
    assert page.draw(1, T0 + 1.1)
    assert status(model, T0 + 1.1) == ViewStatus(live=True, age_s=near(1.2))
    # Once a newer picture was acknowledged, an older one counts for nothing.
    assert page.draw(2, T0 + 1.2)
    assert not page.draw(1, T0 + 3.0)
    model.on_message(state_at(WALL0 + 3.0), T0 + 3.0)
    assert status(model, T0 + 3.0) == NOT_LIVE


def test_a_recorded_source_is_never_live():
    model = model_with_state()
    shown(Page(model), invented=True)
    assert status(model, T0) == NOT_LIVE
    assert model.status_json(T0, model.thor_wall_now(T0))["picture"]["overlay"] == (
        RECORDED
    )


def test_not_live_without_a_fresh_robot_state():
    # No ROBOT_STATE at all: the Thor's wall clock is not known.
    model = ViewModel(state_max_age_s=1.0)
    shown(Page(model))
    assert model.thor_wall_now(T0) is None
    assert model.view_status(T0, None) == NOT_LIVE
    assert model.view_status(T0, WALL0) == NOT_LIVE
    # A ROBOT_STATE that is too old.
    model = model_with_state(state_max_age_s=1.0)
    page = Page(model)
    shown(page)
    assert status(model, T0 + 0.5).live
    assert page.draw(1, T0 + 0.9)
    assert status(model, T0 + 0.99).live
    assert page.draw(1, T0 + 1.0)
    assert status(model, T0 + 1.0) == NOT_LIVE
    assert model.thor_wall_now(T0 + 1.0) is None
    # A newer one brings it back.
    model.on_message(state_at(WALL0 + 1.0), T0 + 1.0)
    assert status(model, T0 + 1.0).live


def test_an_exposure_in_the_future_is_not_live():
    model = model_with_state()
    page = Page(model)
    shown(page, exposed_ago=-(FUTURE_TOLERANCE_S + 0.01))
    assert status(model, T0) == NOT_LIVE
    # Inside the tolerance the picture counts, with the arrival age.
    shown(page, 2, exposed_ago=-(FUTURE_TOLERANCE_S - 0.01), arrived_ago=0.02)
    assert status(model, T0) == ViewStatus(live=True, age_s=near(0.02))


# The view token


@pytest.fixture
def model() -> ViewModel:
    """model_with_state() that holds picture 1, exposed 0.1 s before T0."""
    made = model_with_state()
    made.put_picture(1, JPEG, WALL0 - 0.1, T0 - 0.05, False)
    return made


def test_an_acknowledgement_without_the_pages_token_is_refused_and_the_view_is_not_live(
    model,
):
    # Before a page was loaded no token is the valid one, the empty one neither.
    assert not model.is_current_page("")
    assert not model.note_served(1, "", T0)
    assert not model.acknowledge(1, True, T0, "")
    page = Page(model)
    assert page.fetch(1, T0)
    others = (
        "",
        "x" * len(page.token),
        page.token[:-1],
        page.token + "x",
        page.token.swapcase(),
        "é" * len(page.token),
        "\ud800",  # text that is no UTF-8
    )
    for token in others:
        assert not model.is_current_page(token), token
        assert not model.note_served(1, token, T0), token
        assert not model.acknowledge(1, True, T0, token), token
    # Without a token at all, as a program does that only knows the route.
    assert model.acknowledge(1, True, T0) is False
    assert model._ack is None
    assert status(model, T0) == NOT_LIVE


def test_an_acknowledgement_with_the_token_makes_the_view_live(model):
    page = Page(model)
    # 24 random bytes, as text for a URL.
    assert len(page.token) == 32 and model.is_current_page(page.token)
    assert model.note_served(1, page.token, T0) is True
    assert status(model, T0) == NOT_LIVE
    assert model.acknowledge(1, True, T0 + 0.01, page.token) is True
    assert status(model, T0 + 0.01) == ViewStatus(live=True, age_s=near(0.11))


def test_an_acknowledgement_of_a_picture_the_page_never_fetched_is_refused(model):
    page = Page(model)
    model.put_picture(2, JPEG, WALL0 - 0.1, T0 - 0.05, False)
    # The page was served picture 1 and no other.
    assert page.fetch(1, T0)
    assert not page.acknowledge(2, True, T0)
    assert model._ack is None and status(model, T0) == NOT_LIVE
    assert page.acknowledge(1, True, T0)
    # A refused acknowledgement changes nothing, whatever it says.
    assert not page.acknowledge(2, False, T0)
    assert model._ack == (1, True, T0) and status(model, T0).live


@pytest.mark.parametrize("ack_timeout_s", [0.3, 1.0])
def test_an_acknowledgement_counts_only_within_the_timeout_after_the_fetch(
    ack_timeout_s: float,
) -> None:
    model = model_with_state(ack_timeout_s=ack_timeout_s)
    page = Page(model)
    model.put_picture(1, JPEG, WALL0 - 0.1, T0 - 0.05, False)
    assert page.fetch(1, T0)
    assert not page.acknowledge(1, True, T0 + ack_timeout_s)
    assert model._ack is None
    # The time runs from the newest fetch of that picture.
    assert page.fetch(1, T0 + ack_timeout_s)
    late = T0 + 2 * ack_timeout_s - 0.01
    assert page.acknowledge(1, True, late)
    assert model._ack == (1, True, late)


def test_a_tab_hidden_too_long_after_its_fetch_is_not_live(model):
    page = Page(model)
    assert page.fetch(1, T0)
    # Shown 0.9 s after the page fetched the picture: that still counts.
    assert page.acknowledge(1, True, T0 + 0.9)
    assert status(model, T0 + 0.95) == ViewStatus(live=True, age_s=near(1.05))
    # Hidden a timeout after the fetch: the model takes no word about that
    # picture any more, and the view is not live from that moment.
    assert not page.acknowledge(1, False, T0 + 1.0)
    assert model._ack == (1, True, T0 + 0.9)
    model.on_message(state_at(WALL0 + 1.0), T0 + 1.0)
    for seconds in (1.0, 1.2, 1.5, 1.89):
        assert status(model, T0 + seconds) == NOT_LIVE, seconds
    # A page that fetches and acknowledges again is live again.
    assert page.draw(1, T0 + 1.9)
    assert status(model, T0 + 1.9) == ViewStatus(live=True, age_s=near(2.0))


def test_a_new_page_takes_the_token_and_forgets_what_the_page_before_did(model):
    first = Page(model)
    assert first.draw(1, T0)
    assert status(model, T0).live
    second = Page(model, T0)
    assert second.token != first.token
    assert status(model, T0) == NOT_LIVE
    # The page loaded before is served nothing and counts for nothing.
    assert not model.is_current_page(first.token)
    assert not first.fetch(1, T0) and not first.acknowledge(1, True, T0)
    # What it was served is forgotten: the new page did not fetch picture 1.
    assert not second.acknowledge(1, True, T0)
    assert status(model, T0) == NOT_LIVE
    assert second.draw(1, T0)
    assert status(model, T0) == ViewStatus(live=True, age_s=near(0.1))


def test_tokens_are_compared_in_constant_time(model, monkeypatch):
    page = Page(model)
    compared: list[tuple[bytes, bytes]] = []
    compare_digest = hmac.compare_digest

    def noted(given: bytes, valid: bytes) -> bool:
        compared.append((given, valid))
        return compare_digest(given, valid)

    monkeypatch.setattr(view_model.hmac, "compare_digest", noted)
    assert not model.is_current_page("a guess")
    assert not model.note_served(1, "a guess", T0)
    assert not model.acknowledge(1, True, T0, "a guess")
    assert compared == [(b"a guess", page.token.encode())] * 3
    assert page.draw(1, T0)
    assert compared[3:] == [(page.token.encode(), page.token.encode())] * 2


def test_the_served_pictures_that_are_remembered_are_bounded():
    model = model_with_state()
    page = Page(model)
    kept = view_model.PICTURES_REMEMBERED
    for seq in range(1, kept + 11):
        model.put_picture(seq, JPEG, WALL0 - 0.1, T0 - 0.05, False)
        assert page.fetch(seq, T0)
    assert list(model._served) == list(range(11, kept + 11))
    assert not page.acknowledge(10, True, T0)
    assert page.acknowledge(kept + 10, True, T0)


# "hidden": an acknowledgement that says the page's tab does not show;
# "visible": one that says it shows


def test_visible_after_hidden_for_the_same_fetch_does_not_count(model):
    page = Page(model)
    assert page.fetch(1, T0)
    # The page's two words about one picture are two requests: the server can
    # handle "hidden" first and the older "visible" after it.
    assert page.acknowledge(1, False, T0 + 0.02) is True
    assert page.acknowledge(1, True, T0 + 0.021) is False
    assert model._ack == (1, False, T0 + 0.02)
    for seconds in (0.021, 0.19):
        assert status(model, T0 + seconds) == NOT_LIVE, seconds
    # The next picture the page fetches counts again: here the same number,
    # as when the camera gave no newer one.
    assert page.draw(1, T0 + 0.2) is True
    assert status(model, T0 + 0.2) == ViewStatus(live=True, age_s=near(0.3))


def test_a_tab_that_shows_again_is_live_with_the_next_picture_it_fetches(model):
    page = Page(model)
    assert page.draw(1, T0)
    # The tab is hidden and shows again: the page says each about the picture
    # it drew last, which it fetched before it said "hidden".
    assert page.acknowledge(1, False, T0 + 0.1)
    assert not page.acknowledge(1, True, T0 + 0.3)
    assert status(model, T0 + 0.3) == NOT_LIVE
    model.put_picture(2, JPEG, WALL0 + 0.3, T0 + 0.35, False)
    assert page.draw(2, T0 + 0.4)
    assert status(model, T0 + 0.4) == ViewStatus(live=True, age_s=near(0.1))


def test_a_late_hidden_for_an_older_picture_does_not_block_a_newer_one(model):
    page = Page(model)
    model.put_picture(2, JPEG, WALL0 + 0.1, T0 + 0.15, False)
    model.put_picture(3, JPEG, WALL0 + 0.3, T0 + 0.35, False)
    assert page.draw(1, T0) and page.draw(2, T0 + 0.2)
    assert page.fetch(3, T0 + 0.4)
    # "hidden" for picture 1 is handled only now, after picture 2 was
    # acknowledged: it does not count and leaves no mark.
    assert not page.acknowledge(1, False, T0 + 0.41)
    assert model._ack == (2, True, T0 + 0.2)
    # So picture 3, which the page fetched before that, counts.
    assert page.acknowledge(3, True, T0 + 0.42)
    assert status(model, T0 + 0.42) == ViewStatus(live=True, age_s=near(0.12))


@pytest.mark.parametrize(
    "who, seq",
    [
        ("replaced", 2),  # the page loaded before: its token is valid no longer
        ("page", 1),  # fetched ack_timeout_s ago
        ("page", 3),  # never fetched
        ("page", 7),  # fetched, and no picture the model ever had
    ],
)
def test_a_hidden_that_is_refused_leaves_no_mark(model, who: str, seq: int) -> None:
    pages = {"replaced": Page(model)}
    pages["page"] = page = Page(model, T0)
    model.put_picture(2, JPEG, WALL0 + 0.9, T0 + 0.95, False)
    assert page.fetch(1, T0) and page.fetch(2, T0 + 0.95) and page.fetch(7, T0 + 0.95)
    assert not pages[who].acknowledge(seq, False, T0 + 1.0)
    # Picture 2 was fetched before that word: it counts all the same.
    model.on_message(state_at(WALL0 + 1.0), T0 + 1.0)
    assert page.acknowledge(2, True, T0 + 1.01)
    assert status(model, T0 + 1.01) == ViewStatus(live=True, age_s=near(0.11))


def test_hidden_words_handled_out_of_order_keep_the_later_time(model):
    page = Page(model)
    assert page.draw(1, T0)
    # The tab is hidden at +0.1, shows while the page draws the picture again
    # (fetched at +0.2, drawn at +0.25) and is hidden from +0.3. The server
    # reads its clock before the model takes a word: the model can get the
    # later "hidden" first and "visible" last.
    assert page.fetch(1, T0 + 0.2)
    assert page.acknowledge(1, False, T0 + 0.3)
    assert page.acknowledge(1, False, T0 + 0.1)
    assert not page.acknowledge(1, True, T0 + 0.25)
    assert status(model, T0 + 0.3) == NOT_LIVE


def test_a_visible_held_up_past_a_second_fetch_of_its_number_still_counts(model):
    # A known limit: the model cannot tell which fetch of a picture number a
    # word is about, so a picture's time is that of its newest fetch.
    page = Page(model)
    assert page.fetch(1, T0)
    assert page.acknowledge(1, False, T0 + 0.01)
    # The hidden tab fetches again and gets the same number: no newer picture
    # came. Only then the model takes "visible" for the first fetch.
    assert page.fetch(1, T0 + 0.2)
    assert page.acknowledge(1, True, T0 + 0.201)
    assert status(model, T0 + 0.201).live
    # That ends with the hidden tab's word about its second fetch.
    assert page.acknowledge(1, False, T0 + 0.203)
    assert status(model, T0 + 0.203) == NOT_LIVE
    assert not page.acknowledge(1, True, T0 + 0.204)


def test_a_page_loaded_after_a_hidden_is_live_with_its_first_picture(model):
    first = Page(model)
    assert first.draw(1, T0, visible=False)
    second = Page(model, T0 + 0.1)
    assert second.draw(1, T0 + 0.1)
    assert status(model, T0 + 0.1) == ViewStatus(live=True, age_s=near(0.2))
    # A page load forgets "hidden" with everything else the page before
    # said, whatever clock reading that word carried.
    assert second.acknowledge(1, False, T0 + 0.5)
    third = Page(model, T0 + 0.2)
    assert third.draw(1, T0 + 0.3)
    assert status(model, T0 + 0.3) == ViewStatus(live=True, age_s=near(0.4))


# The age


def test_the_age_is_the_larger_of_the_exposure_age_and_the_arrival_age():
    model = model_with_state()
    page = Page(model)
    shown(page, 1, exposed_ago=0.30, arrived_ago=0.10)
    assert status(model, T0).age_s == near(0.30)
    shown(page, 2, exposed_ago=0.10, arrived_ago=0.25)
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
    page = Page(model)
    shown(page, 1, exposed_ago=0.2, arrived_ago=0.2)
    model.put_picture(2, JPEG, WALL0 + 0.3, T0 + 0.3, False)
    model.on_message(state_at(WALL0 + 0.3), T0 + 0.3)
    assert status(model, T0 + 0.3).age_s == near(0.5)
    assert page.draw(2, T0 + 0.3)
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


def test_the_page_shows_human_driving():
    assert HUMAN_DRIVING == "HUMAN DRIVING"
    model = ViewModel()

    def banner() -> str:
        return model.status_json(T0, None)["banner"]

    # No ROBOT_STATE yet.
    assert banner() == ""
    # The human part of the health as the robot app fills it while a person
    # owns the arm and the grip is released.
    human = {"enabled": True, "takeover_open": True, "driving": False, "note": ""}
    for health in (
        {},
        {"human": human},
        {"human": {}},
        {"human": None},
        {"human": "driving"},
        {"human": ["driving"]},
        {"human": {"driving": "yes"}},
        {"human": {"driving": 1}},
        {"driving": True},
    ):
        model.on_message(state_at(WALL0, health=health), T0)
        assert banner() == "", health
    # While the Thor says a human drive runs.
    model.on_message(
        state_at(WALL0, owner="human", health={"human": {**human, "driving": True}}),
        T0,
    )
    assert banner() == "HUMAN DRIVING"
    told = model.status_json(T0, model.thor_wall_now(T0))
    assert told["banner"] == HUMAN_DRIVING and told["picture"]["overlay"] == NO_PICTURE
    json.dumps(told, allow_nan=False)
    # The words are those of the newest ROBOT_STATE, however old it is; the
    # status says how old.
    told = model.status_json(T0 + 30.0, model.thor_wall_now(T0 + 30.0))
    assert told["banner"] == HUMAN_DRIVING and told["state_age_s"] == near(30.0)
    # The next ROBOT_STATE without it takes the words away.
    model.on_message(state_at(WALL0, owner="human", health={"human": human}), T0)
    assert banner() == ""


def test_the_status_holds_what_the_page_shows():
    model = ViewModel()
    empty = model.status_json(T0, None)
    assert empty["state"] is None and empty["view_status"] is None
    assert empty["robot"]["connected"] is None
    assert empty["picture"]["seq"] is None and empty["events"] == []
    assert empty["banner"] == ""

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

    assert set(told) == {
        "robot",
        "state",
        "state_age_s",
        "picture",
        "banner",
        "view_status",
        "events",
    }
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
        # This robot app reports nothing of the hold.
        "torque": "on",
        "torque_bad": False,
        "assistance_wait_s": 41.5,
        "permit": permit,
        "permit_refusal": "the grid is too old",
        "view_age_s": 0.3,
        # This ROBOT_STATE carries no joint, no jaw reading and no count.
        "joints": [],
        "jaw": {"valid": False, "opening": None, "target": None, "diff": None},
        "reading_age_s": None,
        "missed_deadlines": None,
    }
    assert told["picture"] == {
        "seq": 5,
        "age_s": near(0.45),
        "overlay": "",
        "note": "no camera pictures: the receiver is not there",
    }
    assert told["banner"] == ""
    assert told["view_status"] == {"live": True, "age_s": 0.2}
    json.dumps(told, allow_nan=False)

    # A lost service, and a state without the permit keys.
    model.on_local(None, False)
    model.on_message(state_at(WALL0 + 1.0, health={}), T0 + 1.0)
    told = model.status_json(T0 + 1.0, model.thor_wall_now(T0 + 1.0))
    assert told["robot"]["service_alive"] is False
    assert told["state"]["permit"] is None and told["state"]["permit_refusal"] == ""


@pytest.mark.parametrize(
    ("fields", "torque", "bad"),
    [
        # ROBOT_STATE's torque_on says only that the robot app has not
        # released the motors; health["motors"] is what the joint readings say.
        ({"health": {"motors": {"state": "holding"}}}, "on (holding)", False),
        ({"health": {"motors": {"state": "not holding"}}}, "NOT HOLDING", True),
        ({"health": {"motors": {"state": "unknown"}}}, "on, hold unknown", False),
        ({"health": {"motors": "broken"}}, "on", False),
        ({"torque_on": False, "motion": "rest_unpowered"}, "off", False),
        # After a torque release that failed the motors may be on or off.
        ({"motion": "release_failed"}, "unknown", True),
        (
            {"motion": "release_failed", "health": {"motors": {"state": "holding"}}},
            "unknown",
            True,
        ),
    ],
    ids=["holding", "not holding", "unknown", "garbled", "off", "failed", "failed2"],
)
def test_the_page_is_told_what_the_motors_do(fields, torque, bad) -> None:
    model = ViewModel()
    model.on_message(state_at(WALL0, **fields), T0)
    state = model.status_json(T0, model.thor_wall_now(T0))["state"]
    assert (state["torque"], state["torque_bad"]) == (torque, bad)
    # The page's Torque row writes that word, as a warning where it is one,
    # and no longer its own reading of torque_on.
    page = Path(view_model.__file__).with_name("page.html").read_text("utf-8")
    assert 'row("Torque", state.torque, state.torque_bad);' in page
    assert "state.torque_on" not in page


# The measured joints, the last target sent and the setpoint stream's health.
# A missed deadline is a setpoint of a move that the robot app sent late.

MEASURED = [0.5, -0.05, -0.048, 1.0, -1.5707963267948966, 0.0]
TARGET = [0.52, -0.05, -0.05, 0.9, -1.5707963267948966, 0.25]
# What the page is told of the jaws when nothing about them is a number.
NO_JAW = {"valid": False, "opening": None, "target": None, "diff": None}


def with_joints(**fields: Any) -> RobotState:
    """A ROBOT_STATE with six measured joints, a target and a jaw reading."""
    made = {
        "joints": [
            JointSample(joint, value, 990.0, 990.001, True)
            for joint, value in enumerate(MEASURED)
        ],
        "target_q": list(TARGET),
        "target_t": 990.0,
        "gripper": GripperState(0.4, 990.0, True, True, 0.5, False, "empty"),
        "health": {"reading_age_s": 0.125, "missed_deadlines": 3},
    }
    return state_at(WALL0, **{**made, **fields})


def state_told(state: RobotState) -> dict[str, Any]:
    """What the page is told of that ROBOT_STATE; it must be strict JSON."""
    model = ViewModel()
    model.on_message(state, T0)
    told = model.status_json(T0 + 0.5, None)
    return json.loads(json.dumps(told, allow_nan=False))["state"]


def test_the_status_holds_each_joint_beside_its_last_target():
    told = state_told(with_joints())
    assert [row["joint"] for row in told["joints"]] == [0, 1, 2, 3, 4, 5]
    assert all(row["valid"] is True for row in told["joints"])
    assert [row["q"] for row in told["joints"]] == MEASURED
    assert [row["target"] for row in told["joints"]] == TARGET
    # Degrees of the measured value; the difference is measured minus target.
    assert [row["deg"] for row in told["joints"]] == pytest.approx(
        [28.6479, -2.8648, -2.7502, 57.2958, -90.0, 0.0], abs=1e-4
    )
    assert [row["diff"] for row in told["joints"]] == pytest.approx(
        [-0.02, 0.0, 0.002, 0.1, 0.0, -0.25], abs=1e-12
    )
    assert set(told["joints"][0]) == {"joint", "valid", "q", "deg", "target", "diff"}
    # The jaws, as shares of the jaw travel.
    assert told["jaw"] == {
        "valid": True,
        "opening": 0.4,
        "target": 0.5,
        "diff": pytest.approx(-0.1),
    }
    # As the robot app reported them with this ROBOT_STATE.
    assert told["reading_age_s"] == 0.125 and told["missed_deadlines"] == 3


def test_a_joint_without_a_valid_reading_is_marked_and_has_no_value():
    # What the robot app sends for a joint that has no valid reading.
    joints = with_joints().joints
    joints[3] = JointSample(3, 0.0, 0.0, 0.0, False)
    health = {"reading_age_s": None, "missed_deadlines": 0}
    told = state_told(with_joints(joints=joints, health=health))
    assert told["joints"][3] == {
        "joint": 3,
        "valid": False,
        "q": None,
        "deg": None,
        "target": 0.9,
        "diff": None,
    }
    # The other joints keep their values.
    others = [row for row in told["joints"] if row["joint"] != 3]
    assert all(row["valid"] is True for row in others)
    assert [row["q"] for row in others] == MEASURED[:3] + MEASURED[4:]
    assert told["reading_age_s"] is None and told["missed_deadlines"] == 0
    # Jaws without a reading: the last jaw target is still told.
    no_reading = GripperState(None, None, False, False, 0.5, False, "unknown")
    assert state_told(with_joints(gripper=no_reading))["jaw"] == {
        "valid": False,
        "opening": None,
        "target": 0.5,
        "diff": None,
    }


def test_without_a_target_the_joints_have_no_target_and_no_difference():
    never_sent = GripperState(0.4, 990.0, True, True, None, False, "empty")
    told = state_told(with_joints(target_q=None, target_t=None, gripper=never_sent))
    assert [row["q"] for row in told["joints"]] == MEASURED
    assert all(row["target"] is None for row in told["joints"])
    assert all(row["diff"] is None for row in told["joints"])
    assert told["jaw"] == {"valid": True, "opening": 0.4, "target": None, "diff": None}
    # A target with fewer joints than were measured: the rest have none.
    short = state_told(with_joints(target_q=TARGET[:2]))
    assert [row["target"] for row in short["joints"]] == [0.52, -0.05, *[None] * 4]
    assert [row["diff"] for row in short["joints"]][2:] == [None] * 4
    assert short["joints"][0]["diff"] == pytest.approx(-0.02)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.1", True, None, [1]])
def test_joint_numbers_that_are_not_json_are_told_as_unknown(bad: Any):
    joints = with_joints().joints
    joints[0] = dataclasses.replace(joints[0], value=bad)
    told = state_told(
        with_joints(
            joints=joints,
            target_q=[*TARGET[:5], bad],
            gripper=GripperState(bad, 990.0, True, True, bad, False, "empty"),
            health={"reading_age_s": bad, "missed_deadlines": bad},
        )
    )
    assert told["joints"][0]["valid"] is False and told["joints"][0]["q"] is None
    assert told["joints"][5]["target"] is None and told["joints"][5]["diff"] is None
    assert told["joints"][1]["q"] == MEASURED[1]
    assert told["jaw"] == NO_JAW
    assert told["reading_age_s"] is None and told["missed_deadlines"] is None


@pytest.mark.parametrize("count", [-1, 2.0, 2.5])
def test_a_count_of_missed_deadlines_is_a_whole_number_of_0_or_more(count: Any):
    told = state_told(with_joints(health={"missed_deadlines": count}))
    assert told["missed_deadlines"] is None


def test_the_view_token_is_not_in_the_status():
    model = model_with_state()
    page = Page(model)
    shown(page)
    model.note_sent(status(model, T0))
    told = json.dumps(model.status_json(T0, model.thor_wall_now(T0)))
    assert '"live": true' in told
    assert page.token not in told and "token" not in told


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
    page = Page(model)
    stop = threading.Event()
    problems: list[BaseException] = []

    def work(number: int) -> None:
        try:
            seq = 0
            while not stop.is_set():
                seq += 1
                model.put_picture(seq, JPEG, WALL0, T0, False)
                # Another thread may have acknowledged a newer picture already.
                page.draw(seq, T0)
                model.is_current_page(page.token)
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
    # A page load in the middle of it all ends "live" at once.
    Page(model, T0)
    assert status(model, T0) == NOT_LIVE


# A step of the Thor's wall clock


def stepped_state(seconds: float, step: float) -> RobotState:
    """A ROBOT_STATE that long after T0, its wall clock `step` off the old one."""
    old = state_at(WALL0 + seconds)
    return dataclasses.replace(old, clock_wall_s=old.clock_wall_s + step)


def test_a_picture_from_before_a_step_of_the_thors_wall_clock_is_never_live():
    # The same quiet time as the agent's camera keeps after a step.
    assert view_model.CLOCK_STEP_QUIET_S == QUIET == stereo_camera.CLOCK_STEP_QUIET_S
    model = model_with_state(ack_timeout_s=1.0)
    page = Page(model)
    shown(page, now=T0 + 0.1)  # exposed at WALL0, 0.1 s ago
    model.on_message(state_at(WALL0 + 0.15), T0 + 0.15)
    assert status(model, T0 + 0.15) == ViewStatus(live=True, age_s=near(0.15))
    # The Thor's wall clock steps back 0.15 s against its monotonic clock:
    # the picture would count as 0.05 s old when it is 0.2 s old.
    model.on_message(stepped_state(0.2, -0.15), T0 + 0.2)
    assert status(model, T0 + 0.2) == NOT_LIVE
    # It stays out for as long as the page goes on drawing it.
    for seconds in (0.9, 1.3, QUIET - 0.1):
        model.on_message(stepped_state(seconds, -0.15), T0 + seconds)
        assert page.draw(1, T0 + seconds)
        assert status(model, T0 + seconds) == NOT_LIVE
    # So does a picture that arrives within the quiet time after the step:
    # its exposure can still be one of the old clock.
    at = 0.2 + QUIET - 0.01
    model.put_picture(2, JPEG, WALL0 + at - 0.15 - 0.1, T0 + at, False)
    for seconds in (at, at + 0.5, 30.0):
        model.on_message(stepped_state(seconds, -0.15), T0 + seconds)
        assert page.draw(2, T0 + seconds)
        assert status(model, T0 + seconds) == NOT_LIVE
    # A picture that arrives after it is live, with its age on the new clock.
    at = 31.0
    model.on_message(stepped_state(at, -0.15), T0 + at)
    model.put_picture(3, JPEG, WALL0 + at - 0.15 - 0.1, T0 + at - 0.02, False)
    assert page.draw(3, T0 + at)
    assert status(model, T0 + at) == ViewStatus(live=True, age_s=near(0.1))
    # A step forward counts the same.
    model.on_message(stepped_state(at + 0.1, 0.1), T0 + at + 0.1)
    assert page.draw(3, T0 + at + 0.1)
    assert status(model, T0 + at + 0.1) == NOT_LIVE


@pytest.mark.parametrize("ack_timeout_s", [0.3, 1.0])
def test_a_shorter_ack_timeout_does_not_let_a_picture_from_before_the_step_in(
    ack_timeout_s: float,
) -> None:
    model = model_with_state(ack_timeout_s=ack_timeout_s)
    page = Page(model)
    # Exposed 0.05 s before the Thor's wall clock steps back 0.4 s; it takes
    # 0.25 s to arrive, after the ROBOT_STATE that shows the step.
    model.on_message(stepped_state(0.06, -0.4), T0 + 0.06)
    model.put_picture(1, JPEG, WALL0, T0 + 0.25, False)
    # The camera stalls, and the page draws that picture again every 0.2 s.
    for tenth in range(3, 80, 2):
        now = T0 + tenth / 10.0
        model.on_message(stepped_state(tenth / 10.0, -0.4), now)
        assert page.draw(1, now)
        assert status(model, now) == NOT_LIVE, f"{now - T0:.1f} s after the exposure"


def test_a_step_before_the_models_first_robot_state_is_not_missed():
    model = ViewModel(ack_timeout_s=1.0)
    page = Page(model)
    # The camera connection is up before the robot connection: a picture
    # exposed at WALL0 arrives and is acknowledged.
    model.put_picture(1, JPEG, WALL0, T0 + 0.05, False)
    assert page.draw(1, T0 + 0.1)
    # The first ROBOT_STATE comes 0.2 s after that exposure, from a wall clock
    # that stepped back 0.1 s meanwhile: nothing in the model shows the step.
    model.on_message(stepped_state(0.2, -0.1), T0 + 0.2)
    assert status(model, T0 + 0.2) == NOT_LIVE
    for seconds in (0.25, 0.9, 1.1, QUIET, 20.0):
        model.on_message(stepped_state(seconds, -0.1), T0 + seconds)
        assert page.draw(1, T0 + seconds)
        assert status(model, T0 + seconds) == NOT_LIVE
    # The view is live once a picture arrives the quiet time after that state.
    at = 0.2 + QUIET
    model.on_message(stepped_state(at, -0.1), T0 + at)
    model.put_picture(2, JPEG, WALL0 + at - 0.1 - 0.05, T0 + at, False)
    assert page.draw(2, T0 + at)
    assert status(model, T0 + at) == ViewStatus(live=True, age_s=near(0.05))
    # A robot connection that comes back on the same clocks marks nothing.
    again = dataclasses.replace(stepped_state(at + 0.1, -0.1), session_id="s2")
    model.on_message(again, T0 + at + 0.1)
    assert status(model, T0 + at + 0.1).live


def test_a_change_within_the_links_threshold_is_no_step():
    assert view_model.CLOCK_STEP_S is link_client.CLOCK_STEP_S == 0.02
    model = model_with_state()
    shown(Page(model))
    drifted = dataclasses.replace(state_at(WALL0 + 0.1), clock_wall_s=WALL0 + 0.119)
    model.on_message(drifted, T0 + 0.1)
    assert status(model, T0 + 0.1).live
