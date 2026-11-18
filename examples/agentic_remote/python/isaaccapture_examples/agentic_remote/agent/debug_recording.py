# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the debug recording stores of the obstacle map, and what it does at its size.

The debug recording is the experiment's (thor_recording.EvidenceRecorder): its
hooks hand every event to the recorder's emit(), and a thread of the recorder
writes them, a row of events.jsonl each and the event's arrays as one file
under arrays/. That thread stops for good when the next event would pass the
size limit. APAI's watcher (WorldWatcher) takes one camera frame per tick: it
refreshes the obstacle map with it and runs the detector on it. A tick is the
unit here: one map picture.

MapRate stands in front of emit() and decides only what is written. Of the
watcher's ticks it passes a given number a second whole; the map events of the
others are held in memory for a short while and dropped. Three things store a
held map refresh after all: a refusal or a route segment that names the map
revision it made, a call of the grid bridge inside it that the map does not
make on every refresh, and no newer map for a period of the rate or until the
recording ends. What is recorded outside a tick is always stored.

While the program waits for a task at its prompt (TaskWait) and nothing
happens at the arm, the rate passes one tick per --debug-idle-map-s seconds
instead. Nothing happens at the arm while ROBOT_STATE shows nobody as its
control owner and no motion; the thread that measures every ROBOT_STATE tells
the rate (MapRate.arm_state). An owner, motion, or no ROBOT_STATE for
ARM_STATE_AGE_S brings the rate back, the next tick at once, until the arm has
been without both for ARM_QUIET_S: a person who takes the arm over during the
wait is recorded as a task is. The first tick after a task is taken at once.
The three things above store a held map refresh as at any other time.

Keeper keeps the recording under its size: before the recorder's thread
checks an event against the limit, the array files of the oldest ticks are
removed, so the recording holds the newest stretch and that thread does not
stop for the size. No row is removed, and no array of what the arm did.

Both put their numbers into recording-status.json, which says "complete" no
longer once a map picture was left out or dropped. The map, the checks and the
arm see none of this.

FullNotice says once what happened at the size, and once more with the next
task prompt.

numpy and APAI are imported inside the functions, so the package imports
without them.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from numbers import Integral
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

_log = logging.getLogger("isaaccapture.examples.agentic_remote.debug_recording")

# Map pictures stored a second without --debug-map-hz: what the viewer's page,
# which asks for the state every 500 ms, can show.
DEBUG_MAP_HZ = 2.0
# Seconds between two map pictures stored while the program waits for a task
# and nothing happens at the arm, without --debug-idle-map-s: the viewer still
# shows a current map.
DEBUG_IDLE_MAP_S = 10.0
IDLE_OPTION = "--debug-idle-map-s"
# ROBOT_STATE.owner of an arm nobody owns; any other word is an owner.
ARM_NOBODY = "nobody"
# ROBOT_STATE.motion of an arm that stands; any other word is motion.
ARM_STILL = frozenset({"idle", "hold", "rest_hold", "rest_unpowered", "release_failed"})
# How far a joint may be from the pose the arm last stood at before that is
# motion, in rad. On the rig's run 20261118T022210Z-3147525 the readings of the
# standing arm spread over 0.00065 rad in 346 s at rest with the motors on;
# with the motors off no joint is read and the last reading stands. In six
# of its seven holds behind a move no joint went further than 0.0040 rad from
# where the hold began; in the seventh 0.028 rad, in its first 0.6 s.
ARM_TRAVEL_RAD = 0.005
# The same for the jaw opening, 0 closed to 1 open; that run's spread: 0.00007.
ARM_JAW_TRAVEL = 0.02
# How long the arm has to be without an owner and without motion before a
# wait is thinned, in s.
ARM_QUIET_S = 5.0
# A newest ROBOT_STATE older than this says nothing of the arm, in s: the
# robot connection is down, and the wait is not thinned. They come 0.05 s
# apart; the longest gap of that run was 0.18 s.
ARM_STATE_AGE_S = 2.0
# Why a wait is not thinned, as RATE_BLOCK counts it.
ARM_OWNER, ARM_MOTION, ARM_NO_STATE = "owner", "motion", "no_state"
# What one map picture takes on disk, in bytes, and how many the watcher takes
# a second on the workstation. Measured on the rig's run of 2026-11-17 (camera
# 1280 x 992): 17.17 GB for 3671 ticks in 554 s.
MAP_PICTURE_BYTES = 4.7e6
WATCHER_HZ = 6.6

# The events the limit may leave out, and only inside a tick of the watcher.
MAP_KINDS = frozenset(
    {
        "map.frame",
        "map.depth_input",
        "map.commit",
        "map.refresh_end",
        "mask.pose",
        "mask.mesh",
        "mask.state_source",
        "bridge.request",
        "bridge.response",
        "object.mask",
    }
)
# Of them, the ones whose arrays only the viewer's picture reads: the camera
# frame, the depth and the masks. The arrays of the others are what its state
# reads: the bridge's answer with the map's points.
PICTURE_KINDS = frozenset(
    {"map.frame", "map.depth_input", "mask.mesh", "bridge.request", "object.mask"}
)
# The events that name a map revision: (map id, refreshes).
REFUSAL = "safety.refusal"
SEGMENTS = ("route.segment.begin", "route.segment.end")
# What the map asks of the grid bridge on every refresh. A tick with any other
# call (retire_target: a grasped object taken out of the map) is stored.
ROUTINE_ACTIONS = frozenset(
    {"probe", "integrate", "integrate_depth", "integrate_masked_depth", "query"}
)
# What recorded_late names when no event asked for the refresh: no newer map
# came for a period of the rate, and the newest map when the rate closes.
LATE_STALL = "map.stall"
LATE_END = "map_rate.end"
# What recorded_late names for a refresh that shows what the arm did.
_EVIDENCE_FOR = frozenset({REFUSAL, *SEGMENTS, "bridge.request"})
# The event this module adds: the rate at the start, the counts at the end.
RATE_EVENT = "recording.map_rate"
RATE_NOTE = (
    "Of the ticks of APAI's watcher, map_hz a second are stored whole: map.frame "
    "to map.refresh_end with their bridge calls and masks, and the detector's "
    "object.mask of the same camera frame. The map refresh of another tick is "
    "stored only when a safety.refusal or a route.segment event names its map "
    "revision, when it holds a bridge call that is not made on every refresh, "
    "or as the newest map when none newer came for a period of the rate or "
    "before the recording ended: then late, and each event stored so says in "
    "recorded_late when it happened and what it was stored for. Events from "
    "outside a tick are all stored."
)
# What the rate's first event says of the wait for a task, with idle_map_s.
IDLE_NOTE = (
    "While the agent program waits for a task at its prompt and nothing "
    "happens at the arm, one tick per idle_map_s seconds is stored (0: none). "
    "Nothing happens at the arm while ROBOT_STATE shows nobody as its control "
    "owner and no motion. Motion is a motion word other than those of an arm "
    "that stands, a joint further than arm_travel_rad from the pose the arm "
    "last stood at, or a jaw opening further than arm_jaw_travel from it. From "
    "the first ROBOT_STATE that shows an owner or motion, and when none came "
    "for arm_state_age_s seconds, map_hz ticks a second are stored again, the "
    "next one at once, until the arm was without an owner and without motion "
    "for arm_quiet_s seconds. The first tick after a task is taken is stored "
    "at once. What an event names, a bridge call that is not made on every "
    "refresh and every event from outside a tick are stored as at any other "
    "time."
)
# The map refreshes of skipped ticks held for a refusal or a segment to name.
HELD_TICKS = 3
# The most one held tick may take in memory, in bytes.
HELD_TICK_BYTES = 256 * 1024**2
# The stored map revisions remembered, to tell a named one that is missing.
_STORED_KEPT = 4096
# The data fields of the recorder's hooks that carry another event's id.
_ID_FIELDS = ("request_id", "query_request_id", "call_id")
_PASS = object()

# --debug-at-limit: what the recording does at its size.
AT_LIMIT_DROP = "drop-oldest"
AT_LIMIT_STOP = "stop"
AT_LIMITS = (AT_LIMIT_DROP, AT_LIMIT_STOP)
AT_LIMIT_OPTION = "--debug-at-limit"
# The newest ticks whose arrays the keeper leaves, whatever the size.
KEPT_NEWEST = 4
# What the recorder's writer adds to an event's size when it checks the limit.
_CHECK_BYTES = 4096
# The most the keeper frees beyond what the next event needs.
_RESERVE_BYTES = 2**20
# The part of the size the map points of dropped pictures may take: one in
# this many. On the rig's run of 2026-11-17 a map's points took 0.16 MB.
POINTS_SHARE = 4
# The map revisions the keeper remembers, to find the one an event names.
_REVISIONS_KEPT = 65536
_LATE_FRAMES_KEPT = 256
# The blocks this module adds to recording-status.json.
RATE_BLOCK = "map_rate"
DROPPED_BLOCK = "dropped_at_limit"
STATUS_FILE = "recording-status.json"
_STATUS_PARTIAL = "recording-status.partial"

# What the recorder's writer says when it stops for its size.
_SIZE_REASON = "size limit"
FULL_OPTION = "--debug-max-gib"
WATCH_THREAD = "debug-recording-watch"
_WATCH_PERIOD_S = 0.25


def map_hz_of(value: float | None) -> float:
    """--debug-map-hz as a number; ValueError when it cannot be used."""
    if value is None:
        return DEBUG_MAP_HZ
    rate = float(value)
    if not (0.0 <= rate < float("inf")):
        raise ValueError(
            f"--debug-map-hz must be a finite number, 0 or above, not {value!r}"
        )
    return rate


def idle_map_s_of(value: float | None) -> float:
    """--debug-idle-map-s as a number; ValueError when it cannot be used."""
    if value is None:
        return DEBUG_IDLE_MAP_S
    seconds = float(value)
    if not (0.0 <= seconds < float("inf")):
        raise ValueError(
            f"{IDLE_OPTION} must be a finite number, 0 or above, not {value!r}"
        )
    return seconds


def idle_text(map_hz: float, idle_map_s: float | None) -> str:
    """A sentence for the operator on the wait for a task; "" without the rule."""
    if not map_hz > 0 or idle_map_s is None:
        return ""
    stored = "no map picture is stored"
    if idle_map_s > 0:
        stored = f"one map picture is stored every {idle_map_s:g} s"
    many = "picture" if map_hz == 1 else "pictures"
    return (
        ". While the program waits for a task and nobody owns or moves the "
        f"arm, {stored}, which {IDLE_OPTION} sets; a person who takes the arm "
        f"over in that wait is recorded at {map_hz:g} map {many} a second, and "
        "what the arm did or was refused is stored as always"
    )


def at_limit_of(value: str | None) -> str:
    """--debug-at-limit as one of AT_LIMITS; ValueError when it is none."""
    if value is None:
        return AT_LIMIT_DROP
    if value not in AT_LIMITS:
        raise ValueError(
            f"{AT_LIMIT_OPTION} must be one of {', '.join(AT_LIMITS)}, not {value!r}"
        )
    return value


def expected_minutes(max_gib: float, map_hz: float) -> float:
    """How long a recording of max_gib lasts on the workstation, in minutes."""
    rate = min(map_hz, WATCHER_HZ) if map_hz > 0 else WATCHER_HZ
    return max_gib * 1024**3 / (MAP_PICTURE_BYTES * rate) / 60.0


def expected_text(max_gib: float, map_hz: float) -> str:
    """The same for the operator: "about 30 min at 2 map pictures a second"."""
    minutes = expected_minutes(max_gib, map_hz)
    span = "under 1 min" if minutes < 1 else f"about {round(minutes)} min"
    if 0 < map_hz < WATCHER_HZ:
        many = "picture" if map_hz == 1 else "pictures"
        return f"{span} at {map_hz:g} map {many} a second"
    return f"{span} with every map picture stored, about {WATCHER_HZ:.0f} a second"


def _minutes_text(seconds: float) -> str:
    return "less than 1 min" if seconds < 60 else f"{round(seconds / 60)} min"


def _copied(value: Any, np: Any, size: list[int]) -> Any:
    """value with every array copied, as the recorder would freeze it now."""
    if isinstance(value, np.ndarray):
        size[0] += value.nbytes
        return np.array(value, copy=True, order="C")
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        return _copied(value.detach().cpu().numpy(), np, size)
    if isinstance(value, dict):
        return {key: _copied(item, np, size) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copied(item, np, size) for item in value]
    return value


def _number(value: Any) -> int | None:
    """A map revision above 0, else None: before the first refresh none is named."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        return None
    return int(value)


def _stand_in(value: Any) -> bool:
    """Whether an event id is one MapRate gave out; the recorder's are above 0."""
    return isinstance(value, int) and not isinstance(value, bool) and value < 0


def _odd_call(kind: str, data: dict[str, Any]) -> bool:
    """Whether this is a call of the grid bridge the map does not make each refresh."""
    if kind != "bridge.request":
        return False
    payload = data.get("payload")
    action = payload.get("action") if isinstance(payload, dict) else None
    return action not in ROUTINE_ACTIONS


def _evidence(data: dict[str, Any]) -> bool:
    """Whether an event was stored late for what the arm did."""
    late = data.get("recorded_late")
    return isinstance(late, dict) and late.get("for") in _EVIDENCE_FOR


def named_revisions(kind: str, data: dict[str, Any]) -> set[tuple[Any, int]]:
    """The map revisions an event names, as (map id, refreshes)."""
    if kind == REFUSAL:
        report = data.get("report")
        report = report if isinstance(report, dict) else {}
        record = report.get("map")
        record = record if isinstance(record, dict) else {}
        map_id = data.get("map_id")
        numbers = [
            data.get("gate_refreshes"),
            report.get("gate_refreshes"),
            record.get("refreshes"),
        ]
    elif kind in SEGMENTS:
        state = data.get("map")
        state = state if isinstance(state, dict) else {}
        map_id, numbers = state.get("map_id"), [state.get("refreshes")]
    else:
        return set()
    return {(map_id, n) for n in map(_number, numbers) if n is not None}


class _Tick:
    """What the recorder was given during one tick of the watcher."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stored: bool | None = None  # None until its first map event
        # (kind, data, refresh id, stand-in id, monotonic ns, thread name)
        self.held: list[tuple[str, dict[str, Any], Any, int, int, str]] = []
        self.frames: set[int] = set()  # the stand-in ids of its map.frame events
        self.ids: dict[int, Any] = {}  # stand-in id -> the recorder's id
        self.revisions: list[tuple[Any, int]] = []
        self.commit = 0  # the number of its map.commit among all; 0: none
        self.bytes = 0
        self.too_large = False


class MapRate:
    """Passes map_hz ticks of the watcher a second to the recorder's emit().

    emit() takes the place of the recorder's on the recorder itself, so the
    experiment's hooks call it unchanged. It runs on the thread of the event:
    the watcher's for a tick, any other for the rest. A held event is copied
    there as the recorder would copy it; nothing is written on that thread.
    An error in here passes the event on as it came.

    With idle_map_s, between idle_begin() and idle_end() (the program waits
    for a task) and while arm_state() has shown an arm without an owner and
    without motion for ARM_QUIET_S, one tick per idle_map_s seconds of the
    clock is passed instead, none with 0: the wait is thinned. The first tick
    after idle_end(), and after an owner, motion or ARM_STATE_AGE_S without
    arm_state() ended the thinning, is passed at once. Without arm_state()
    no wait is thinned.
    """

    def __init__(
        self,
        recording: Any,
        map_hz: float,
        clock: Callable[[], float] = time.monotonic,
        idle_map_s: float | None = None,
    ) -> None:
        import numpy

        self._np = numpy
        self._recording = recording
        self._emit = recording.emit  # what stands behind this gate
        self._period = 1.0 / map_hz
        self._clock = clock
        self._due = float("-inf")
        self._lock = threading.Lock()
        self._local = threading.local()
        self._held: OrderedDict[tuple[Any, int], _Tick] = OrderedDict()
        self._stored: OrderedDict[tuple[Any, Any], None] = OrderedDict()
        self._last_id = 0
        self._commits = 0  # the map.commit events so far, stored or held
        self._stored_commit = 0  # the newest of them that is stored
        self._map_at = float("-inf")  # when a commit was last stored
        self._picture_at = float("-inf")  # when a tick was last stored by its turn
        self._told = False
        self.map_hz = float(map_hz)
        # The wait for a task. None: the rate knows no such wait.
        self.idle_map_s = None if idle_map_s is None else float(idle_map_s)
        self._idle_since: float | None = None  # the clock at idle_begin()
        self._idle_total = 0.0  # seconds of the waits that have ended
        self.idle_skipped = 0  # of the skipped ticks, those of a thinned wait
        # The arm, as arm_state() was told. _arm_at: the newest ROBOT_STATE's
        # arrival. _arm_quiet: since when each one showed nobody and no motion;
        # None while the newest shows one of them or is too old.
        self._arm_at: float | None = None
        self._arm_quiet: float | None = None
        self._arm_pose: list[float] | None = None  # where the arm last stood
        self._arm_jaw: float | None = None
        self._arm_failed = False
        # The thinned part of the waits, and the part an owner, motion or a
        # missing ROBOT_STATE took from them after the arm had been quiet.
        self._thin_since: float | None = None
        self._thin_total = 0.0
        self._broken_since: float | None = None
        self._broken_total = 0.0
        self.idle_interrupted = dict.fromkeys((ARM_OWNER, ARM_MOTION, ARM_NO_STATE), 0)
        # ticks: those with a map event. stored: by the rate. rescued: stored
        # after all: for an event that named them, for a bridge call of their
        # own, or as the newest map. missing: named, and neither stored nor
        # held any more.
        self.counts = dict.fromkeys(
            ("ticks", "stored", "skipped", "rescued", "missing"), 0
        )

    # The watcher's thread: one tick.

    def begin(self) -> Any:
        """A tick begins on this thread; returns what end() takes."""
        before = getattr(self._local, "tick", None)
        self._local.tick = _Tick()
        return before

    def end(self, before: Any) -> None:
        """The tick ends. One without a commit stores the newest held map."""
        tick = getattr(self._local, "tick", None)
        self._local.tick = before
        if tick is not None and not tick.commit:
            self._store_newest(LATE_STALL, when_due=True)

    def close(self) -> None:
        """The rate ends: the newest held map is stored when none newer is."""
        self._store_newest(LATE_END, when_due=False)

    # The program's thread: the wait for a task.

    def idle_begin(self) -> None:
        """The program waits for a task from now on; never raises."""
        if self.idle_map_s is None:
            return
        try:
            now = self._clock()
            with self._lock:
                if self._idle_since is None:
                    self._idle_since = now
        except Exception:
            self._trouble()

    def idle_end(self) -> None:
        """A task was taken, or the wait ended otherwise; never raises.

        The next tick is stored at once.
        """
        if self.idle_map_s is None:
            return
        try:
            now = self._clock()
            with self._lock:
                if self._idle_since is not None:
                    self._thinned(now)
                    self._idle_total += max(0.0, now - self._idle_since)
                    self._thin_ends(now)
                    if self._broken_since is not None:
                        self._broken_total += max(0.0, now - self._broken_since)
                        self._broken_since = None
                    self._idle_since = None
                    self._due = float("-inf")
        except Exception:
            self._trouble()

    # The thread that measures every ROBOT_STATE: the arm.

    def arm_state(
        self,
        owner: Any,
        motion: Any,
        joints: Any = None,
        jaw: Any = None,
        at: float | None = None,
    ) -> None:
        """What a ROBOT_STATE shows of the arm; never raises.

        owner and motion are its words; joints its measured joints in rad,
        None when they are not whole; jaw its jaw opening, None when it is
        not valid; at its arrival on the clock, None for now.
        """
        if self.idle_map_s is None:
            return
        try:
            now = self._clock() if at is None else float(at)
            with self._lock:
                self._thinned(now)  # a gap before this one is counted first
                why = self._arm_shows(owner, motion, joints, jaw)
                self._arm_at = now if self._arm_at is None else max(self._arm_at, now)
                if why:
                    self._arm_busy(now, why)
                elif self._arm_quiet is None:
                    self._arm_quiet = now
        except Exception:
            if not self._arm_failed:
                self._arm_failed = True
                _log.warning(
                    "what ROBOT_STATE shows of the arm did not reach the debug "
                    "recording: a wait for a task is recorded as a task is",
                    exc_info=True,
                )

    def _arm_shows(self, owner: Any, motion: Any, joints: Any, jaw: Any) -> str:
        """Why this ROBOT_STATE is no arm at which nothing happens; "" for one.

        The caller holds the lock. Travel is measured from the pose the arm
        last stood at, so that a slow drift adds up; that pose follows an arm
        that is owned or moves.
        """
        why = ""
        if owner != ARM_NOBODY:
            why = ARM_OWNER
        elif motion not in ARM_STILL:
            why = ARM_MOTION
        pose = [] if joints is None else [float(value) for value in joints]
        if pose and all(math.isfinite(value) for value in pose):
            stood = self._arm_pose
            moved = stood is not None and (
                len(stood) != len(pose)
                or max(abs(a - b) for a, b in zip(pose, stood, strict=True))
                > ARM_TRAVEL_RAD
            )
            if moved or why or stood is None:
                self._arm_pose = pose
            why = why or (ARM_MOTION if moved else "")
        opening = None if jaw is None else float(jaw)
        if opening is not None and math.isfinite(opening):
            stood_at = self._arm_jaw
            moved = stood_at is not None and abs(opening - stood_at) > ARM_JAW_TRAVEL
            if moved or why or stood_at is None:
                self._arm_jaw = opening
            why = why or (ARM_MOTION if moved else "")
        return why

    def _arm_busy(self, at: float, why: str) -> None:
        """Something happens at the arm from `at` on; the caller holds the lock."""
        was_quiet, self._arm_quiet = self._arm_quiet is not None, None
        if self._thin_since is not None:
            self._thin_ends(at)
            self._due = float("-inf")  # the next tick is stored at once
        waiting = self._idle_since is not None
        if was_quiet and waiting and self._broken_since is None:
            self._broken_since = max(at, self._idle_since)
            self.idle_interrupted[why] += 1

    def _thin_ends(self, at: float) -> None:
        if self._thin_since is not None:
            self._thin_total += max(0.0, at - self._thin_since)
            self._thin_since = None

    def _thinned(self, now: float) -> bool:
        """Whether the wait for a task is thinned at `now`.

        The caller holds the lock. Thinned: the program waits, and every
        ROBOT_STATE of the last ARM_QUIET_S showed an arm nobody owns that
        stands, the newest no older than ARM_STATE_AGE_S.
        """
        if self._arm_quiet is not None and now - self._arm_at > ARM_STATE_AGE_S:
            self._arm_busy(self._arm_at + ARM_STATE_AGE_S, ARM_NO_STATE)
        if self._idle_since is None:
            return False
        if self._thin_since is None:
            quiet = self._arm_quiet
            if quiet is None or now - quiet < ARM_QUIET_S:
                return False
            self._thin_since = max(quiet + ARM_QUIET_S, self._idle_since)
            if self._broken_since is not None:
                broken = self._thin_since - self._broken_since
                self._broken_total += max(0.0, broken)
                self._broken_since = None
        return True

    def waits(self) -> tuple[float, float, float]:
        """The waits for a task so far, in seconds of the clock.

        (all of them, their thinned part, the part after an owner, motion
        or a missing ROBOT_STATE had ended a quiet stretch of the arm).
        """
        now = self._clock()
        with self._lock:
            self._thinned(now)
            parts = []
            for since, total in (
                (self._idle_since, self._idle_total),
                (self._thin_since, self._thin_total),
                (self._broken_since, self._broken_total),
            ):
                parts.append(total if since is None else total + max(0.0, now - since))
        return parts[0], parts[1], parts[2]

    def idle_seconds(self) -> float:
        """How long the waits for a task have been thinned so far, on the clock."""
        return self.waits()[1]

    def rule(self) -> dict[str, Any]:
        """What the rate's first event says: the rate, and the wait's rule."""
        told: dict[str, Any] = {"map_hz": self.map_hz, "note": RATE_NOTE}
        if self.idle_map_s is not None:
            told.update(
                idle_map_s=self.idle_map_s,
                idle_note=IDLE_NOTE,
                arm_quiet_s=ARM_QUIET_S,
                arm_state_age_s=ARM_STATE_AGE_S,
                arm_travel_rad=ARM_TRAVEL_RAD,
                arm_jaw_travel=ARM_JAW_TRAVEL,
            )
        return told

    def status(self) -> dict[str, Any]:
        """The block RATE_BLOCK of recording-status.json: the numbers so far."""
        block: dict[str, Any] = {"map_hz": self.map_hz, **self.counts}
        if self.idle_map_s is not None:
            waited, thinned, broken = self.waits()
            with self._lock:
                by = dict(self.idle_interrupted)
            block.update(
                idle_map_s=self.idle_map_s,
                # The waits for a task, and their thinned part.
                wait_seconds=round(waited, 3),
                idle_seconds=round(thinned, 3),
                idle_skipped=self.idle_skipped,
                # The times the arm ended a quiet stretch of a wait, and how
                # long the wait then went on at map_hz.
                idle_interrupted=sum(by.values()),
                idle_interrupted_seconds=round(broken, 3),
                idle_interrupted_by=by,
            )
        return block

    # Any thread: the recorder's intake.

    def emit(self, kind: str, **data: Any) -> Any:
        """In place of the recorder's emit(kind, **data); returns its event id."""
        if getattr(self._recording, "disabled", False):
            return self._emit(kind, **data)  # stopped: nothing is held or copied
        try:
            answer = self._sorted(kind, data)
        except Exception:
            self._trouble()
            answer = self._stored_anyway(kind, data)
        return self._emit(kind, **data) if answer is _PASS else answer

    def _stored_anyway(self, kind: str, data: dict[str, Any]) -> Any:
        """The answer for an event the limit failed on: its tick is stored."""
        tick = getattr(self._local, "tick", None)
        if tick is None or tick.stored is not False:
            return _PASS
        try:
            with tick.lock:
                self._give(tick, kind)
                return self._passed(tick, kind, data)
        except Exception:
            tick.too_large, tick.held = True, []
            return None

    def _sorted(self, kind: str, data: dict[str, Any]) -> Any:
        """_PASS for an event the recorder takes as it is, else the id to return."""
        if kind == REFUSAL or kind in SEGMENTS:
            self._rescue(kind, data)
        tick = getattr(self._local, "tick", None)
        commit = self._next_commit() if kind == "map.commit" else 0
        if tick is None:
            if commit:
                self._note_stored(data, commit)
            return _PASS
        with tick.lock:
            if commit:
                tick.commit = commit
            if kind in MAP_KINDS and tick.stored is None:
                self._decide(tick)
            if kind in MAP_KINDS and not tick.stored and _odd_call(kind, data):
                self._give(tick, kind)  # this tick's refresh is no routine one
            if kind not in MAP_KINDS or tick.stored:
                if commit:
                    self._note_stored(data, commit)
                return self._passed(tick, kind, data)
            return self._hold(tick, kind, data)

    def _next_commit(self) -> int:
        with self._lock:
            self._commits += 1
            return self._commits

    def _decide(self, tick: _Tick) -> None:
        now = self._clock()
        with self._lock:
            idle = self.idle_map_s is not None and self._thinned(now)
            if idle:
                # One per idle_map_s since a map was last stored; none with 0.
                last = max(self._picture_at, self._map_at)
                tick.stored = 0 < self.idle_map_s <= now - last
            else:
                tick.stored = now >= self._due
            self.counts["ticks"] += 1
            self.counts["stored" if tick.stored else "skipped"] += 1
            if tick.stored:
                due = self._due + self._period
                self._due = due if due > now else now + self._period
                self._picture_at = now
            elif idle:
                self.idle_skipped += 1

    def _hold(self, tick: _Tick, kind: str, data: dict[str, Any]) -> Any:
        """Keep a map event of a skipped tick; returns its stand-in id, or None."""
        refresh = getattr(self._recording.local, "refresh_id", None)
        if tick.too_large or not (kind == "map.frame" or refresh in tick.frames):
            return None  # the detector's object.mask: no part of the map refresh
        size = [0]
        copy = _copied(data, self._np, size)
        tick.bytes += size[0]
        if tick.bytes > HELD_TICK_BYTES:
            tick.too_large, tick.held = True, []
            return None
        with self._lock:
            self._last_id -= 1
            stand_in = self._last_id
        thread = threading.current_thread().name
        tick.held.append((kind, copy, refresh, stand_in, time.monotonic_ns(), thread))
        if kind == "map.frame":
            tick.frames.add(stand_in)
        key = (data.get("map_id"), _number(data.get("refreshes")))
        if kind == "map.commit" and key[1] is not None:
            tick.revisions.append(key)
            with self._lock:
                self._held[key] = tick
                while len(self._held) > HELD_TICKS:
                    self._held.popitem(last=False)
        return stand_in

    def _passed(self, tick: _Tick, kind: str, data: dict[str, Any]) -> Any:
        """_PASS, or the id of an event that carried a stand-in id of this tick."""
        local = self._recording.local
        current = getattr(local, "refresh_id", None)
        fields = [name for name in _ID_FIELDS if _stand_in(data.get(name))]
        if not fields and not _stand_in(current):
            return _PASS
        changed = dict(data)
        for name in fields:
            changed[name] = tick.ids.get(changed[name])
        local.refresh_id = tick.ids.get(current) if _stand_in(current) else current
        try:
            return self._emit(kind, **changed)
        finally:
            local.refresh_id = current

    def _note_stored(self, data: dict[str, Any], commit: int) -> None:
        key = (data.get("map_id"), _number(data.get("refreshes")))
        now = self._clock()
        with self._lock:
            self._remember(key)
            self._stored_commit = max(self._stored_commit, commit)
            self._map_at = now

    def _remember(self, key: tuple[Any, Any]) -> None:
        self._stored[key] = None
        while len(self._stored) > _STORED_KEPT:
            self._stored.popitem(last=False)

    def _rescue(self, kind: str, data: dict[str, Any]) -> None:
        """Store the held map refresh of each revision this event names."""
        for key in named_revisions(kind, data):
            with self._lock:
                tick = self._held.pop(key, None)
                known = key in self._stored
            if tick is not None:
                with tick.lock:
                    self._give(tick, kind)
            elif not known:
                with self._lock:
                    self.counts["missing"] += 1

    def _store_newest(self, why: str, *, when_due: bool) -> None:
        """Store the newest held map refresh when no newer commit is stored.

        when_due: only once a period of the rate has passed since a commit
        was stored, so that a stall adds no more than the rate does.
        """
        try:
            if getattr(self._recording, "disabled", False):
                return
            now = self._clock()
            with self._lock:
                if not self._held:
                    return
                # In a thinned wait: no more often than its turn stores.
                thinned = self.idle_map_s is not None and self._thinned(now)
                period = self.idle_map_s if thinned else self._period
                if when_due and (period <= 0 or now - self._map_at < period):
                    return
                key = next(reversed(self._held))
                tick = self._held[key]
                if tick.commit <= self._stored_commit:
                    return
                del self._held[key]
            with tick.lock:
                self._give(tick, why)
        except Exception:
            self._trouble()

    def _give(self, tick: _Tick, why: str) -> None:
        """Give the held events of a skipped tick to the recorder, in their order.

        The caller holds the tick's lock. From here the tick is a stored one:
        what it records next goes to the recorder as it comes.
        """
        if tick.stored:
            return
        local = self._recording.local
        before = getattr(local, "refresh_id", None)
        try:
            for kind, data, refresh, stand_in, at_ns, thread in tick.held:
                row = dict(data)
                for name in _ID_FIELDS:
                    if _stand_in(row.get(name)):
                        row[name] = tick.ids.get(row[name])
                row["recorded_late"] = {
                    "event_mono_ns": at_ns,
                    "thread": thread,
                    "for": why,
                }
                inside = tick.ids.get(refresh) if _stand_in(refresh) else refresh
                local.refresh_id = inside
                tick.ids[stand_in] = self._emit(kind, **row)
        finally:
            local.refresh_id = before
            tick.held, tick.stored = [], True
        now = self._clock()
        with self._lock:
            self.counts["rescued"] += 1
            for key in tick.revisions:
                self._held.pop(key, None)
                self._remember(key)
            if tick.commit:
                self._stored_commit = max(self._stored_commit, tick.commit)
                self._map_at = now

    def _trouble(self) -> None:
        if not self._told:
            self._told = True
            _log.warning(
                "the limit on the debug recording's map pictures failed; the "
                "events are stored as they come",
                exc_info=True,
            )


class Dropped(NamedTuple):
    """What the keeper dropped so far. The recorder's thread replaces it whole."""

    refreshes: int = 0  # ticks whose pictures went
    bytes: int = 0  # what that freed on disk
    # The recorder's monotonic ns from which every picture is there; None
    # while nothing was dropped.
    kept_from_ns: int | None = None
    points: int = 0  # of the ticks, those whose map points went too
    named_late: int = 0  # refusals and segments that named a dropped tick
    reached_s: float | None = None  # seconds into the recording at the first drop
    # As kept_from_ns, for the map's points; None while none of them went.
    points_from_ns: int | None = None


class _Refresh:
    """The array files of one tick that may be dropped, by their event ids."""

    __slots__ = (
        "commit_ns",
        "end_ns",
        "kept",
        "keys",
        "pictures",
        "points",
        "points_bytes",
        "serial",
    )

    def __init__(self, serial: int) -> None:
        self.serial = serial
        self.pictures: list[int] = []  # of PICTURE_KINDS
        self.points: list[int] = []  # of the other MAP_KINDS
        self.points_bytes = 0  # what they take, once the pictures went
        self.kept = False  # evidence, or the session's first map: never dropped
        self.commit_ns: int | None = None
        self.end_ns = 0
        self.keys: list[tuple[Any, Any]] = []  # the map revisions it committed


class _Span:
    """One tick on one thread, by the recorder's event ids: after < id <= until."""

    __slots__ = ("after", "serial", "until")

    def __init__(self, serial: int, after: int) -> None:
        self.serial = serial
        self.after = after
        self.until: int | None = None  # None while the tick runs


class Keeper:
    """Drops the oldest map pictures so that the recorder never stops for its size.

    get() takes the place of the recorder's events.get, so it runs on the
    recorder's writer thread, once per event and before that thread checks the
    event against the size. When the check would fail it removes array files
    of the oldest ticks and lowers the recorder's written_bytes by their size.
    Only that thread reads and writes written_bytes, so nothing is locked for
    it. On the watcher's thread there are only begin() and end(), which mark a
    tick by the recorder's event counter; they take no lock, so that thread
    never waits for the writer. An event belongs to a tick when the thread of
    that name recorded it between the two marks.

    What may go: the arrays of MAP_KINDS events recorded inside a tick, oldest
    tick first, a tick's pictures together. The arrays its map points are in
    stay behind, for the viewer's state of that time; the oldest of those go
    once they take more than one part in POINTS_SHARE of the size, or when no
    picture is left to drop. What stays: every row, every array from outside
    a tick, the ticks a refusal or a route segment names or that were stored
    late for one, the first map of the session and the KEPT_NEWEST newest
    ticks. With nothing left to drop the recorder stops for its size as it
    does alone.
    """

    def __init__(self, recording: Any) -> None:
        self._recording = recording
        self._get = recording.events.get  # its queue's own
        self._arrays = Path(recording.path) / "arrays"
        self._reserve = min(_RESERVE_BYTES, int(recording.max_bytes) // 64)
        self._points_room = int(recording.max_bytes) // POINTS_SHARE
        self._local = threading.local()
        self._serials = itertools.count(1)
        # Thread name -> its ticks, oldest first. A tick's thread appends,
        # the writer's thread takes from the left.
        self._spans: dict[str, deque[_Span]] = {}
        # From here on: the recorder's writer thread alone.
        self._refreshes: OrderedDict[int, _Refresh] = OrderedDict()
        self._thinned: OrderedDict[int, _Refresh] = OrderedDict()  # points left
        self._thinned_bytes = 0
        # Map revision -> its tick; None once the tick's pictures went.
        self._revisions: OrderedDict[tuple[Any, Any], _Refresh | None] = OrderedDict()
        self._late_frames: OrderedDict[Any, None] = OrderedDict()
        self._newest: _Refresh | None = None
        self._shown: _Refresh | None = None  # the newest tick with a commit
        self._first_kept = False
        self._last_serial = 0  # the newest tick whose pictures went
        self._last_end_ns = 0
        self._zero_ns: int | None = None
        self._told = False
        self.dropped = Dropped()

    # The watcher's thread: one tick.

    def begin(self) -> Any:
        """A tick begins on this thread; returns what end() takes."""
        name = threading.current_thread().name
        span = _Span(next(self._serials), self._recording.seq)
        self._spans.setdefault(name, deque()).append(span)
        before = getattr(self._local, "span", None)
        self._local.span = span
        return before

    def end(self, before: Any) -> None:
        span = getattr(self._local, "span", None)
        if span is not None:
            span.until = self._recording.seq
        self._local.span = before

    # The recorder's writer thread.

    def get(self, *args: Any, **kwargs: Any) -> Any:
        """In place of the recorder's events.get(): the next event to write."""
        item = self._get(*args, **kwargs)
        try:
            row, arrays, size = item
            self._note(row, bool(arrays))
            self._make_room(int(size))
        except Exception:
            self._trouble()
        return item

    def _tick_of(self, row: dict[str, Any]) -> int:
        """The tick a map event was recorded in, or 0 for none.

        The events come in the order of their ids, so a tick that ended
        before this one is done with.
        """
        event, found = row["id"], 0
        for name, spans in list(self._spans.items()):
            while spans:
                span = spans[0]
                until = span.until
                if until is not None and until < event:
                    spans.popleft()
                    continue
                if name == row.get("thread") and span.after < event:
                    found = span.serial
                break
        return found

    def _note(self, row: dict[str, Any], has_arrays: bool) -> None:
        """Sort one event that is about to be written."""
        kind, data = row["kind"], row["data"]
        serial = self._tick_of(row)
        if kind not in MAP_KINDS or _evidence(data):
            serial = 0
        newest = self._newest
        if not serial or (newest is not None and serial < newest.serial):
            # No part of a tick that may go.
            if kind == REFUSAL or kind in SEGMENTS:
                self._keep_named(kind, data)
            elif kind == "map.frame" and _evidence(data):
                self._late_frames[row["id"]] = None
                while len(self._late_frames) > _LATE_FRAMES_KEPT:
                    self._late_frames.popitem(last=False)
            return
        if newest is None or serial != newest.serial:
            newest = self._newest = self._refreshes[serial] = _Refresh(serial)
        newest.end_ns = row["mono_ns"]
        if row.get("refresh_id") in self._late_frames:
            newest.kept = True  # the rest of a refresh that was stored late
        if has_arrays:
            files = newest.pictures if kind in PICTURE_KINDS else newest.points
            files.append(row["id"])
        if kind == "map.commit":
            key = (data.get("map_id"), _number(data.get("refreshes")))
            newest.commit_ns = row["mono_ns"]
            newest.keys.append(key)
            self._revisions[key] = newest
            while len(self._revisions) > _REVISIONS_KEPT:
                self._revisions.popitem(last=False)
            self._shown = newest
            if not self._first_kept:
                self._first_kept = newest.kept = True

    def _keep_named(self, kind: str, data: dict[str, Any]) -> None:
        """The ticks whose map revision this refusal or segment names stay."""
        late = 0
        for key in named_revisions(kind, data):
            if key not in self._revisions:
                continue
            refresh = self._revisions[key]
            if refresh is None:
                late += 1
            else:
                refresh.kept = True
        if late:
            self.dropped = self.dropped._replace(
                named_late=self.dropped.named_late + late
            )

    def _make_room(self, size: int) -> None:
        """Drop the oldest ticks' arrays when the next event would not fit."""
        recording = self._recording
        if recording.disabled:
            return
        over = recording.written_bytes + size + _CHECK_BYTES - recording.max_bytes
        if over <= 0:
            return
        over += self._reserve
        freed = refreshes = points = 0
        while freed < over:
            # Pictures first, unless the points left behind take their share.
            crowded = self._thinned_bytes > self._points_room
            refresh = None if crowded else self._oldest()
            if refresh is not None:
                del self._refreshes[refresh.serial]
                gone, files = self._remove(refresh.pictures)
                refresh.pictures = []
                if files:
                    for key in refresh.keys:
                        if key in self._revisions:
                            self._revisions[key] = None
                    self._last_serial = refresh.serial
                    self._last_end_ns = refresh.end_ns
                    refreshes += 1
                if refresh.points:
                    refresh.points_bytes = self._bytes(refresh.points)
                    self._thinned[refresh.serial] = refresh
                    self._thinned_bytes += refresh.points_bytes
            elif self._thinned:
                _, refresh = self._thinned.popitem(last=False)
                self._thinned_bytes -= refresh.points_bytes
                gone, files = self._remove(refresh.points)
                points += bool(files)
            else:
                break
            freed += gone
        recording.written_bytes -= freed
        if refreshes or points:
            self._publish(freed, refreshes, points)

    def _oldest(self) -> _Refresh | None:
        """The oldest tick whose pictures may go, or None."""
        candidates = len(self._refreshes) - KEPT_NEWEST
        found, stay = None, []
        for index, refresh in enumerate(self._refreshes.values()):
            if index >= candidates:
                break
            if refresh.kept:
                stay.append(refresh.serial)
            elif refresh is not self._shown:
                found = refresh
                break
        for serial in stay:
            del self._refreshes[serial]
        return found

    def _remove(self, ids: list[int]) -> tuple[int, int]:
        """Remove the array files of these events; (bytes freed, files removed)."""
        freed = files = 0
        for event_id in ids:
            path = self._arrays / f"{event_id:09d}.npz"
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue  # an event the recorder did not write
            except OSError:
                self._trouble()
                continue
            freed += size
            files += 1
        return freed, files

    def _bytes(self, ids: list[int]) -> int:
        """What the array files of these events take on disk."""
        total = 0
        for event_id in ids:
            try:
                total += (self._arrays / f"{event_id:09d}.npz").stat().st_size
            except OSError:
                continue
        return total

    def _publish(self, freed: int, refreshes: int, points: int) -> None:
        kept_from = None
        for refresh in self._refreshes.values():
            if refresh.serial > self._last_serial and refresh.commit_ns is not None:
                kept_from = refresh.commit_ns
                break
        told = self.dropped
        kept_from = kept_from or self._last_end_ns or told.kept_from_ns
        points_from = told.points_from_ns
        if points or points_from is not None:
            # The oldest tick whose points are left, else the oldest picture.
            oldest = next(iter(self._thinned.values()), None)
            points_from = (oldest and oldest.commit_ns) or kept_from
        reached = told.reached_s
        if reached is None:
            start = getattr(self._recording, "start_mono_ns", None)
            reached = (time.monotonic_ns() - start) / 1e9 if start else 0.0
        self.dropped = Dropped(
            told.refreshes + refreshes,
            told.bytes + freed,
            kept_from,
            told.points + points,
            told.named_late,
            reached,
            points_from,
        )

    # Any thread: what was dropped.

    def _zero(self) -> int:
        """The monotonic ns of the recording's first row: the viewer's time 0."""
        if self._zero_ns is None:
            try:
                events = Path(self._recording.path) / "events.jsonl"
                with events.open(encoding="utf-8") as stream:
                    self._zero_ns = int(json.loads(stream.readline())["mono_ns"])
            except (OSError, ValueError, KeyError, TypeError):
                return int(getattr(self._recording, "start_mono_ns", 0))
        return self._zero_ns

    def kept_from(self, points: bool = False) -> tuple[float, float] | None:
        """From when every map picture is there, or None with nothing dropped.

        (wall clock seconds, seconds on the viewer's time line). With points:
        from when every map's points are there.
        """
        told = self.dropped
        at_ns = told.points_from_ns if points else told.kept_from_ns
        if at_ns is None:
            return None
        recording = self._recording
        wall_ns = recording.start_wall_ns + at_ns - recording.start_mono_ns
        return wall_ns / 1e9, max(0, at_ns - self._zero()) / 1e9

    def status(self) -> dict[str, Any]:
        """The block DROPPED_BLOCK of recording-status.json."""

        def on_the_line(kept: tuple[float, float] | None) -> float | None:
            # Rounded up: from there on, every map has what is named.
            return None if kept is None else math.ceil(kept[1] * 1e3) / 1e3

        told = self.dropped
        return {
            "refreshes": told.refreshes,
            "bytes": told.bytes,
            # Seconds on the viewer's time line. From the first on every map
            # has its picture; from the second on, its points.
            "first_time_kept": on_the_line(self.kept_from()),
            "points_dropped": told.points,
            "first_time_points_kept": on_the_line(self.kept_from(points=True)),
            "named_after_drop": told.named_late,
        }

    def _trouble(self) -> None:
        if not self._told:
            self._told = True
            _log.warning(
                "the debug recording's oldest map pictures could not be dropped; "
                "the recording may stop at its size",
                exc_info=True,
            )


def _status_with(recording: Any, gate: MapRate | None, keeper: Keeper | None) -> Any:
    """In place of the recorder's _status(finalized): the same file with the blocks.

    The recorder writes its status first, not as finalized when a map picture
    was left out or dropped, so that it never says "complete" for such a run;
    the file is then written once more with the blocks and what "finalized"
    is. On the recorder's writer thread, as its own.
    """
    own = recording._status
    told: list[bool] = []

    def status(finalized: bool) -> None:
        blocks: dict[str, Any] = {}
        if gate is not None:
            blocks[RATE_BLOCK] = gate.status()
        if keeper is not None:
            blocks[DROPPED_BLOCK] = keeper.status()
        thinned = bool(
            blocks.get(RATE_BLOCK, {}).get("skipped")
            or blocks.get(DROPPED_BLOCK, {}).get("refreshes")
        )
        own(bool(finalized) and not thinned)
        try:
            path = Path(recording.path) / STATUS_FILE
            written = json.loads(path.read_text(encoding="utf-8"))
            written["finalized"] = bool(finalized)
            if thinned:
                written["complete"] = False
            written.update(blocks)
            partial = path.with_name(_STATUS_PARTIAL)
            partial.write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")
            partial.replace(path)
        except Exception:
            if not told:
                told.append(True)
                _log.warning(
                    "the debug recording's status lacks what was left out or dropped",
                    exc_info=True,
                )

    return status


def _takes_events(recording: Any) -> bool:
    """Whether the recorder takes its events through emit(), as the experiment's."""
    return callable(getattr(recording, "emit", None)) and hasattr(recording, "local")


def _has_writer(recording: Any) -> bool:
    """Whether the recorder writes from a queue against a size, as the experiment's."""
    names = ("written_bytes", "max_bytes", "path", "disabled", "_status", "seq")
    names += ("start_mono_ns", "start_wall_ns")
    events = getattr(recording, "events", None)
    return (
        _takes_events(recording)
        and callable(getattr(events, "get", None))
        and all(hasattr(recording, name) for name in names)
    )


class TaskWait:
    """Tells the rate when the agent program waits for a task at its prompt,
    and what each ROBOT_STATE shows of the arm.

    The program opens waiting() around its read of the operator's task line.
    The thread that measures every ROBOT_STATE calls arm_state(). limited()
    binds its rate for as long as it is open; without one, both do nothing.
    A question in the middle of a task is no such wait.
    """

    def __init__(self) -> None:
        self._gate: MapRate | None = None

    def bind(self, gate: MapRate | None) -> None:
        self._gate = gate

    def arm_state(
        self,
        owner: Any,
        motion: Any,
        joints: Any = None,
        jaw: Any = None,
        at: float | None = None,
    ) -> None:
        """MapRate.arm_state() of the bound rate; never raises."""
        gate = self._gate
        if gate is not None:
            gate.arm_state(owner, motion, joints, jaw, at)

    @contextmanager
    def waiting(self) -> Iterator[None]:
        """Open while the program waits for a task; never raises of its own."""
        gate = self._gate
        if gate is not None:
            gate.idle_begin()
        try:
            yield
        finally:
            if gate is not None:
                gate.idle_end()


class Limits(NamedTuple):
    """What limited() put on the recorder; None for what it did not."""

    gate: MapRate | None
    keeper: Keeper | None


@contextmanager
def limited(
    recording: Any,
    map_hz: float,
    at_limit: str = AT_LIMIT_DROP,
    clock: Callable[[], float] = time.monotonic,
    *,
    idle_map_s: float | None = None,
    task_wait: TaskWait | None = None,
) -> Iterator[Limits]:
    """While open, the recorder stores map_hz map pictures a second of clock,
    and with at_limit AT_LIMIT_DROP it drops its oldest ones at its size.

    With idle_map_s the rate stores one map picture per that many seconds
    (0: none) while task_wait.waiting() is open and task_wait.arm_state() has
    shown an arm nobody owns or moves for ARM_QUIET_S.

    Yields the MapRate and the Keeper. The first is None with map_hz 0, the
    second with AT_LIMIT_STOP; both without a recorder and with a recorder
    that is not the experiment's. RuntimeError when APAI's watcher has no
    tick to mark. The keeper and the status blocks stay on the recorder when
    this closes: its thread writes what is left, and the last status, after.
    """
    rate = recording is not None and map_hz > 0
    keep = recording is not None and at_limit == AT_LIMIT_DROP
    if rate and not _takes_events(recording):
        _log.warning(
            "the debug recorder takes no events through emit(): --debug-map-hz "
            "does not limit it, and every map picture is stored"
        )
        rate = False
    if keep and not _has_writer(recording):
        _log.warning(
            "the debug recorder is not the experiment's: %s %s cannot drop its "
            "oldest map pictures, and it stops at its size",
            AT_LIMIT_OPTION,
            AT_LIMIT_DROP,
        )
        keep = False
    if not rate and not keep:
        yield Limits(None, None)
        return
    from apai.perception.world import WorldWatcher

    original = getattr(WorldWatcher, "_tick", None)
    if original is None:
        raise RuntimeError(
            "APAI's WorldWatcher has no _tick, so the debug recording can "
            "neither be limited to map pictures a second nor drop its oldest "
            f"ones; --debug-map-hz 0 stores every one, with {AT_LIMIT_OPTION} "
            f"{AT_LIMIT_STOP}"
        )
    own_emit = vars(recording).get("emit", _PASS)
    keeper = Keeper(recording) if keep else None
    gate = MapRate(recording, map_hz, clock, idle_map_s) if rate else None
    # The rate first: what it stores at the end of a tick is inside the
    # keeper's marks of that tick.
    parts: list[Any] = [part for part in (gate, keeper) if part is not None]

    @wraps(original)
    def tick(*args: Any, **kwargs: Any) -> Any:
        marks = [part.begin() for part in parts]
        try:
            return original(*args, **kwargs)
        finally:
            for part, before in zip(parts, marks, strict=True):
                part.end(before)

    with patch.object(WorldWatcher, "_tick", tick):
        if keeper is not None:
            recording.events.get = keeper.get
        if hasattr(recording, "_status"):
            recording._status = _status_with(recording, gate, keeper)
        if gate is not None:
            recording.emit = gate.emit
        hooks = getattr(recording, "hooks", None)
        if isinstance(hooks, list):
            if gate is not None:
                hooks.append(f"{__name__}.MapRate.emit")
            if keeper is not None:
                hooks.append(f"{__name__}.Keeper.get")
            hooks.append(f"{WorldWatcher.__module__}.WorldWatcher._tick")
        if gate is not None:
            gate.emit(RATE_EVENT, **gate.rule())
            if task_wait is not None:
                task_wait.bind(gate)
        try:
            yield Limits(gate, keeper)
        finally:
            if gate is not None:
                if task_wait is not None:
                    task_wait.bind(None)
                gate.idle_end()
                gate.close()
                gate.emit(RATE_EVENT, **gate.status())
            if own_emit is _PASS:
                vars(recording).pop("emit", None)
            else:
                recording.emit = own_emit


@contextmanager
def map_rate(
    recording: Any, map_hz: float, clock: Callable[[], float] = time.monotonic
) -> Iterator[MapRate | None]:
    """limited() with the rate alone: at its size the recorder stops."""
    with limited(recording, map_hz, AT_LIMIT_STOP, clock) as limits:
        yield limits.gate


class FullNotice:
    """Tells the operator what the debug recording did at its size.

    While watching() is open a thread of this class looks at the recorder and
    its keeper. It says one line when the size is first reached and the oldest
    map pictures go, and one when the recorder's writer has stopped.
    prompt_lines() gives each line once more, for the next task prompt.
    Nothing is said for a recording that ends with its run.
    """

    def __init__(self, say: Callable[[str], None], max_gib: float) -> None:
        self._say = say
        self._max_gib = float(max_gib)
        self._lock = threading.Lock()
        self._line: str | None = None  # the recorder stopped
        self._again = False
        self._keeper: Keeper | None = None  # the size was reached; the oldest go
        self._reached_s = 0.0
        self._reached_again = False
        self.told = 0  # how often a line was said or given

    def line(self, reason: str, after_s: float) -> str:
        """The line for a recording that stopped after_s seconds for reason."""
        after = _minutes_text(after_s)
        effect = "the viewer no longer updates; the arm and the task are not affected"
        if _SIZE_REASON in reason:
            return (
                f"debug recording full after {after} ({FULL_OPTION} "
                f"{self._max_gib:g}): {effect}. End the session and start the "
                f"agent again for a new recording, or raise {FULL_OPTION}"
            )
        return (
            f"debug recording stopped after {after} ({reason}): {effect}. With "
            "a full disk, delete old runs by hand first. End the session and "
            "start the agent again for a new recording"
        )

    def reached_line(self, after_s: float, kept: tuple[float, float] | None) -> str:
        """The line for a recording that reached its size after after_s seconds.

        kept is Keeper.kept_from(): from when every map picture is there.
        """
        line = (
            f"debug recording reached {FULL_OPTION} {self._max_gib:g} after "
            f"{_minutes_text(after_s)}: from now on the oldest map pictures are "
            "dropped; the viewer keeps following"
        )
        if kept is None:
            return line
        clock = time.strftime("%H:%M:%S", time.localtime(kept[0]))
        return (
            f"{line}, and times before {clock} ({kept[1]:.0f} s on its time "
            "line) show no picture. That time moves on with the run"
        )

    def look(self, recording: Any, began: float, keeper: Keeper | None = None) -> bool:
        """Say what is new of the recorder; True once it has stopped."""
        lines = []
        with self._lock:
            if self._line is not None:
                return True
            if keeper is not None and self._keeper is None:
                dropped = keeper.dropped
                if dropped.refreshes:
                    self._keeper, self._reached_again = keeper, True
                    self._reached_s = dropped.reached_s or time.monotonic() - began
                    lines.append(self.reached_line(self._reached_s, keeper.kept_from()))
            if getattr(recording, "disabled", False):
                reason = getattr(recording, "writer_error", None)
                line = self.line(
                    str(reason or "its writer stopped"), time.monotonic() - began
                )
                self._line, self._again = line, True
                lines.append(line)
            self.told += len(lines)
        for line in lines:
            try:
                self._say(line)
            except Exception:
                _log.warning("not said at the terminal: %s", line)
        return self._line is not None

    def prompt_lines(self) -> list[str]:
        """What is said before the next task prompt: each line, once."""
        lines = []
        with self._lock:
            if self._reached_again and self._keeper is not None:
                self._reached_again = False
                kept = self._keeper.kept_from()  # where it is by now
                lines.append(self.reached_line(self._reached_s, kept))
            if self._again and self._line is not None:
                self._again = False
                lines.append(self._line)
            self.told += len(lines)
        return lines

    @contextmanager
    def watching(
        self, recording: Any, keeper: Keeper | None = None
    ) -> Iterator[FullNotice]:
        """While open, a thread looks at this recorder and its keeper."""
        if recording is None or not hasattr(recording, "disabled"):
            yield self
            return
        started = getattr(recording, "start_mono_ns", None)
        began = started / 1e9 if isinstance(started, int) else time.monotonic()
        done = threading.Event()

        def watch() -> None:
            while not self.look(recording, began, keeper):
                if done.wait(_WATCH_PERIOD_S):
                    break

        thread = threading.Thread(target=watch, name=WATCH_THREAD, daemon=True)
        thread.start()
        try:
            yield self
        finally:
            done.set()
            thread.join(2.0)
