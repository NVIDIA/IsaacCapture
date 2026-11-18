# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the live view knows: the newest camera picture, the robot's state and
whether a browser is showing the view.

The live view is the viewer's page with its live picture showing. A camera
picture is a camera frame made small and packed as JPEG. The page acknowledges
each picture it has drawn; VIEW_STATUS says "live" only on such an
acknowledgement. The view token is the secret a page gets when it is loaded:
only the page loaded last has the valid one, and only that page's
acknowledgements of pictures it was served count. The model reads no clock and
does no I/O: every time is handed in, on this machine's monotonic clock unless
the name says "wall".
"""

from __future__ import annotations

import hmac
import math
import secrets
import threading
from collections import OrderedDict, deque
from typing import Any

from ..agent.link_client import CLOCK_STEP_S
from ..protocol.local_messages import LocalStatus
from ..protocol.messages import Event, Message, RobotState, ViewStatus, Welcome
from ..robot_app.text import TORQUE_NOT_HOLDING, TORQUE_UNKNOWN, torque_text

# The age a VIEW_STATUS carries when the view is not live.
AGE_WHEN_NOT_LIVE_S = 3600.0
# How far ahead of the Thor's wall clock an exposure may lie and still count.
FUTURE_TOLERANCE_S = 0.05
# A picture that arrives less than this long after a step of the Thor's wall
# clock, or after the model's first ROBOT_STATE, can be dated on another wall
# clock than the one its age is measured on: it is never live. Longer than the
# camera pictures take to arrive; the agent's camera refuses camera frames for
# as long after a step (CLOCK_STEP_QUIET_S of agent/camera/stereo_camera.py).
CLOCK_STEP_QUIET_S = 3.0
EVENTS_KEPT = 10
# Pictures whose times are kept for an acknowledgement that comes late.
PICTURES_REMEMBERED = 64

# What status_json() says of the picture, for a page to write over it.
NO_PICTURE = "NO PICTURE"
STALE = "STALE"
RECORDED = "RECORDED, NOT LIVE"
# While the Thor says a person is driving the arm (health["human"]["driving"]).
HUMAN_DRIVING = "HUMAN DRIVING"

_PERMIT_KEYS = ("move_id", "through_index", "evidence_age_s", "expires_in_s")
# ROBOT_STATE's motion after a torque release whose driver call failed.
_RELEASE_FAILED = "release_failed"


class ViewModel:
    """The live view's state; every method may be called from any thread."""

    def __init__(
        self,
        *,
        ack_timeout_s: float = 1.0,
        state_max_age_s: float = 1.0,
        stale_picture_s: float = 1.0,
    ) -> None:
        self._ack_timeout_s = float(ack_timeout_s)
        self._state_max_age_s = float(state_max_age_s)
        self._stale_picture_s = float(stale_picture_s)
        self._lock = threading.Lock()
        self._welcome: Welcome | None = None
        self._state: RobotState | None = None
        self._state_at = 0.0
        self._clock_quiet_s = float(CLOCK_STEP_QUIET_S)
        # When the first ROBOT_STATE arrived, or the newest one that showed a
        # step of the Thor's wall clock against its monotonic clock.
        self._clock_mark_at: float | None = None
        self._events: deque[Event] = deque(maxlen=EVENTS_KEPT)
        self._local: LocalStatus | None = None
        self._service_alive = True
        self._picture: tuple[int, bytes] | None = None
        # Picture number -> (exposure on the Thor's wall clock, arrival, invented).
        self._times: OrderedDict[int, tuple[float, float, bool]] = OrderedDict()
        # The valid view token and when its page was loaded; None before a load.
        self._token: bytes | None = None
        self._page_at: float | None = None
        # Picture number -> when it last went to the page of the valid token.
        self._served: OrderedDict[int, float] = OrderedDict()
        # The newest acknowledgement: picture number, tab visible, when.
        self._ack: tuple[int, bool, float] | None = None
        # The latest time at which the page said that its tab is hidden.
        self._hidden_at: float | None = None
        self._camera_note = ""
        self._sent: ViewStatus | None = None

    # What comes in

    def on_message(self, message: Message, now_mono: float) -> None:
        """Take a WELCOME, a ROBOT_STATE or an EVENT; anything else is ignored."""
        with self._lock:
            if isinstance(message, RobotState):
                earlier = self._state
                # Before the first one nothing says on which wall clock the
                # pictures taken in so far are dated.
                stepped = earlier is None
                if earlier is not None:
                    # As AgentLink._take_state: the exposures taken in so far
                    # are dated on the wall clock from before the step.
                    was = earlier.clock_wall_s - earlier.clock_mono_s
                    now = message.clock_wall_s - message.clock_mono_s
                    stepped = abs(now - was) > CLOCK_STEP_S
                if stepped:
                    self._clock_mark_at = now_mono
                self._state = message
                self._state_at = now_mono
            elif isinstance(message, Event):
                self._events.append(message)
            elif isinstance(message, Welcome):
                self._welcome = message

    def on_local(self, status: LocalStatus | None, alive: bool) -> None:
        """The newest LOCAL_STATUS and whether the robot link service is there."""
        with self._lock:
            self._local = status
            self._service_alive = bool(alive)

    def set_camera_note(self, text: str) -> None:
        """What status_json() says about the camera pictures' source; "" for nothing."""
        with self._lock:
            self._camera_note = text

    def put_picture(
        self,
        seq: int,
        jpeg: bytes,
        exposure_wall_s: float,
        arrival_mono_s: float,
        invented: bool,
    ) -> None:
        """The newest camera picture, with a number that only grows.

        `invented` marks a picture whose times nobody measured (a recording).
        """
        with self._lock:
            self._picture = (seq, jpeg)
            self._times[seq] = (exposure_wall_s, arrival_mono_s, bool(invented))
            while len(self._times) > PICTURES_REMEMBERED:
                self._times.popitem(last=False)

    def new_page(self, now_mono: float) -> str:
        """The page was loaded: its view token, from now on the only valid one.

        Every acknowledgement and every served picture of the page loaded
        before is forgotten: the view is not live until this page has drawn a
        picture.
        """
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._token = token.encode()
            self._page_at = now_mono
            self._served.clear()
            self._ack = None
            self._hidden_at = None
        return token

    def is_current_page(self, token: str) -> bool:
        """Whether `token` is the view token of the page loaded last."""
        with self._lock:
            return self._is_valid(token)

    def note_served(self, seq: int, token: str, now_mono: float) -> bool:
        """Picture `seq` goes to the page of `token` now.

        False, and nothing is remembered, when the token is not the valid one.
        """
        with self._lock:
            if not self._is_valid(token):
                return False
            self._served[seq] = now_mono
            self._served.move_to_end(seq)
            while len(self._served) > PICTURES_REMEMBERED:
                self._served.popitem(last=False)
            return True

    def acknowledge(
        self, seq: int, visible: bool, now_mono: float, token: str = ""
    ) -> bool:
        """The page has drawn picture `seq`; `visible` is whether its tab shows.

        It counts only with the valid view token and for a picture that went
        to that page less than ack_timeout_s ago. A number the model does not
        remember, and one below the picture acknowledged last, count for
        nothing either. Nor does "the tab shows" for a picture that went to
        the page before it last said "the tab is hidden": the next picture it
        fetches counts again. False when it did not count: nothing has changed
        then.
        """
        with self._lock:
            if not self._is_valid(token):
                return False
            # When that number last went to the page: the model cannot tell
            # which fetch of a number a word is about.
            served_at = self._served.get(seq)
            if served_at is None or not now_mono - served_at < self._ack_timeout_s:
                return False
            if seq not in self._times:
                return False
            if self._ack is not None and seq < self._ack[0]:
                return False
            if not visible:
                # The server reads its clock before this lock is taken: two
                # of these can come in against the order of their times.
                if self._hidden_at is None or now_mono > self._hidden_at:
                    self._hidden_at = now_mono
            elif self._hidden_at is not None and served_at < self._hidden_at:
                # The page's two words about one picture are two requests:
                # "the tab shows" can be handled after a later "is hidden".
                return False
            self._ack = (seq, bool(visible), now_mono)
            return True

    def note_sent(self, status: ViewStatus) -> None:
        """Remember the VIEW_STATUS that was sent last, for status_json()."""
        with self._lock:
            self._sent = status

    # What goes out

    def picture(self) -> tuple[int, bytes] | None:
        """The newest camera picture as (number, JPEG), None without one."""
        with self._lock:
            return self._picture

    def thor_wall_now(self, now_mono: float) -> float | None:
        """The Thor's wall clock now, from the newest ROBOT_STATE's clock pair.

        None without a ROBOT_STATE younger than state_max_age_s.
        """
        with self._lock:
            return self._thor_wall_now(now_mono)

    def view_status(self, now_mono: float, thor_wall_now: float | None) -> ViewStatus:
        """The VIEW_STATUS to send now.

        Live only when the page loaded last acknowledged, as visible and less
        than ack_timeout_s ago, a picture it fetched less than ack_timeout_s
        ago: one of the camera connection whose exposure does not lie in the
        Thor's future. A fresh ROBOT_STATE must give the Thor's wall clock. The
        picture must have arrived CLOCK_STEP_QUIET_S or more after the model's
        first ROBOT_STATE and after the newest step of that clock against the
        Thor's monotonic clock: an earlier one is never live. The age is then
        the larger of that picture's age since its exposure and since its
        arrival on this machine.
        """
        with self._lock:
            age = self._live_age(now_mono, thor_wall_now)
        if age is None:
            return ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)
        return ViewStatus(live=True, age_s=age)

    def status_to_send(self, now_mono: float) -> ViewStatus:
        """The VIEW_STATUS to send now; a live one says when it is sent.

        As view_status() with thor_wall_now() of this moment, both of the
        Thor's clocks read off one ROBOT_STATE. sent_mono_s is its
        clock_mono_s plus the time since it arrived: never later than the
        Thor's monotonic clock reads now. The Thor adds the time from then to
        the age.
        """
        with self._lock:
            age = self._live_age(now_mono, self._thor_wall_now(now_mono))
            if age is None:
                return ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)
            # A live view has a fresh ROBOT_STATE: _live_age asks for one.
            passed = max(0.0, now_mono - self._state_at)
            sent = self._state.clock_mono_s + passed
        return ViewStatus(live=True, age_s=age, sent_mono_s=sent)

    def status_json(
        self, now_mono: float, thor_wall_now: float | None
    ) -> dict[str, Any]:
        """The robot's status and the picture's, as JSON values.

        The view token is not among them.
        """
        with self._lock:
            local, state, sent = self._local, self._state, self._sent
            welcome = self._welcome
            clients = {} if local is None else dict(sorted(local.roles.items()))
            return {
                "robot": {
                    "service_alive": self._service_alive,
                    "connected": None if local is None else local.robot_connected,
                    "close_reason": "" if local is None else local.close_reason,
                    "session_id": "" if local is None else local.session_id,
                    "robot_id": "" if welcome is None else welcome.robot_id,
                    "driver": "" if welcome is None else welcome.driver,
                    "clients": clients,
                    "stop_unconfirmed": local is not None and local.stop_unconfirmed,
                },
                "state": None if state is None else _state_json(state),
                "state_age_s": (
                    None if state is None else max(0.0, now_mono - self._state_at)
                ),
                "picture": self._picture_json(now_mono, thor_wall_now),
                # In large letters over the picture; "" for nothing.
                "banner": HUMAN_DRIVING if _human_driving(state) else "",
                "view_status": (
                    None if sent is None else {"live": sent.live, "age_s": sent.age_s}
                ),
                "events": [
                    {"seq": event.seq, "kind": event.kind, "text": event.text}
                    for event in self._events
                ],
            }

    # Under the lock

    def _is_valid(self, token: str) -> bool:
        valid = self._token
        if valid is None:
            return False
        return hmac.compare_digest(token.encode("utf-8", "replace"), valid)

    def _thor_wall_now(self, now_mono: float) -> float | None:
        state = self._state
        if state is None:
            return None
        passed = now_mono - self._state_at
        if not passed < self._state_max_age_s:
            return None
        return state.clock_wall_s + max(0.0, passed)

    def _live_age(self, now_mono: float, thor_wall_now: float | None) -> float | None:
        """The age of the acknowledged picture, None when the view is not live."""
        ack = self._ack
        if ack is None:
            return None
        seq, visible, at = ack
        if not visible or not now_mono - at < self._ack_timeout_s:
            return None
        # acknowledge() takes no word about a picture that went to the page
        # ack_timeout_s ago or longer, "the tab is hidden" neither: from then
        # on that picture is not live.
        served_at = self._served.get(seq)
        if served_at is None or not now_mono - served_at < self._ack_timeout_s:
            return None
        times = self._times.get(seq)
        if times is None:
            return None
        exposure_wall_s, arrival_mono_s, invented = times
        if invented:
            return None
        mark = self._clock_mark_at
        if mark is None or arrival_mono_s - mark < self._clock_quiet_s:
            return None
        # The model's own ROBOT_STATE must be fresh, whatever clock was handed in.
        if thor_wall_now is None or self._thor_wall_now(now_mono) is None:
            return None
        if exposure_wall_s - thor_wall_now > FUTURE_TOLERANCE_S:
            return None
        age = max(thor_wall_now - exposure_wall_s, now_mono - arrival_mono_s, 0.0)
        return age if math.isfinite(age) else None

    def _picture_json(
        self, now_mono: float, thor_wall_now: float | None
    ) -> dict[str, Any]:
        picture = self._picture
        if picture is None:
            return {
                "seq": None,
                "age_s": None,
                "overlay": NO_PICTURE,
                "note": self._camera_note,
            }
        exposure_wall_s, arrival_mono_s, invented = self._times[picture[0]]
        age = max(0.0, now_mono - arrival_mono_s)
        if not invented and thor_wall_now is not None:
            age = max(age, thor_wall_now - exposure_wall_s)
        words = []
        if invented:
            words.append(RECORDED)
        if not age <= self._stale_picture_s:
            words.append(STALE)
        return {
            "seq": picture[0],
            "age_s": age if math.isfinite(age) else None,
            "overlay": ", ".join(words),
            "note": self._camera_note,
        }


def _state_json(state: RobotState) -> dict[str, Any]:
    health = state.health if isinstance(state.health, dict) else {}
    refusal = health.get("permit_refusal")
    motors = health.get("motors")
    hold = motors.get("state") if isinstance(motors, dict) else None
    # After a torque release that failed the motors may be on or off.
    torque = torque_text(state.torque_on, hold, state.motion == _RELEASE_FAILED)
    return {
        "owner": state.owner,
        "owner_number": state.owner_number,
        "stop_level": state.stop_level,
        "stop_number": state.stop_number,
        "fault_reason": state.fault_reason,
        "motion": state.motion,
        "move_id": state.move_id,
        "move_index": state.move_index,
        "move_total": state.move_total,
        "move_stage": state.move_stage,
        "jaw_load": state.gripper.load,
        "torque_on": state.torque_on,
        # What the robot app says of the motors, and whether it is a warning.
        # torque_on alone says only that the robot app has not released the
        # motors.
        "torque": torque.removeprefix("torque "),
        "torque_bad": torque in (TORQUE_NOT_HOLDING, TORQUE_UNKNOWN),
        "assistance_wait_s": state.assistance_wait_s,
        "permit": _permit_json(health.get("permit")),
        "permit_refusal": refusal if isinstance(refusal, str) else "",
        "view_age_s": state.view_age_s,
        "joints": _joints_json(state),
        "jaw": _jaw_json(state),
        # How old the oldest of the joints' readings was when the robot app
        # built this ROBOT_STATE; None while a joint has no valid reading.
        "reading_age_s": _number_or_none(health.get("reading_age_s")),
        # A missed deadline is a setpoint of a move that the robot app sent
        # late; it counts them from its start. None when it reports no count.
        "missed_deadlines": _count_or_none(health.get("missed_deadlines")),
    }


def _joints_json(state: RobotState) -> list[dict[str, Any]]:
    """Per joint: its measured value, the last target sent and their difference.

    "q", "target" and "diff" (measured minus target) are in rad; "deg" is "q"
    in degrees. A joint without a valid reading has "valid" false and no
    value; one without a target has no target. Either way it has no difference.
    """
    targets = state.target_q if isinstance(state.target_q, list) else []
    rows = []
    for sample in state.joints:
        valid = sample.valid is True and _is_number(sample.value)
        q = sample.value if valid else None
        target = None
        if 0 <= sample.joint < len(targets):
            target = _number_or_none(targets[sample.joint])
        rows.append(
            {
                "joint": sample.joint,
                "valid": valid,
                "q": q,
                "deg": None if q is None else math.degrees(q),
                "target": target,
                "diff": None if q is None or target is None else q - target,
            }
        )
    return rows


def _jaw_json(state: RobotState) -> dict[str, Any]:
    """The jaw opening, the last jaw target and their difference.

    All three are shares of the jaw travel, 1.0 fully open. Without a valid
    jaw reading "valid" is false and there is no opening.
    """
    jaws = state.gripper
    valid = jaws.valid is True and _is_number(jaws.opening)
    opening = jaws.opening if valid else None
    target = _number_or_none(jaws.last_target)
    return {
        "valid": valid,
        "opening": opening,
        "target": target,
        "diff": None if opening is None or target is None else opening - target,
    }


def _human_driving(state: RobotState | None) -> bool:
    """Whether the newest ROBOT_STATE says that a person is driving the arm."""
    if state is None or not isinstance(state.health, dict):
        return False
    human = state.health.get("human")
    return isinstance(human, dict) and human.get("driving") is True


def _permit_json(permit: Any) -> dict[str, Any] | None:
    """The held permit's facts when they are of the kinds status_json() gives."""
    if not isinstance(permit, dict):
        return None
    move_id, through = permit.get("move_id"), permit.get("through_index")
    ages = [permit.get("evidence_age_s"), permit.get("expires_in_s")]
    if not isinstance(move_id, str) or not _is_number(through):
        return None
    if not all(_is_number(age) for age in ages):
        return None
    return {key: permit[key] for key in _PERMIT_KEYS}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _number_or_none(value: Any) -> float | None:
    return value if _is_number(value) else None


def _count_or_none(value: Any) -> int | None:
    """A count as status_json() gives it: a whole number of 0 or more, else None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
