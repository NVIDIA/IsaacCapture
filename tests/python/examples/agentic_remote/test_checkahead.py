# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The check ahead: which setpoints a PERMIT may cover, and on which grid.

The tests marked "apai" run APAI's SafetyHarness on a required obstacle grid
that an in-process client fills; the others use a scripted checker.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.agent import checkahead
from isaaccapture_examples.agentic_remote.agent.checkahead import (
    CheckAhead,
    Evidence,
    FrameTimes,
    Grant,
    GridBusy,
    MoveCheck,
)

PERIOD_S = 0.02
REACH = 25
HOME = (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)
GOAL = (0.4, 1.3, 1.0, 0.1, 0.2, 0.3)
# The Thor's wall clock minus the workstation's monotonic clock, in these tests.
WALL_OFFSET_S = 1.7e9


def min_jerk_path(
    start: Sequence[float], goal: Sequence[float], steps: int
) -> list[np.ndarray]:
    a, b = np.array(start, dtype=float), np.array(goal, dtype=float)
    fractions = (i / steps for i in range(1, steps + 1))
    return [a + (b - a) * (10 * t**3 - 15 * t**4 + 6 * t**5) for t in fractions]


def in_thread(call: Callable[[], Any]) -> Any:
    """Run call on another thread; its result, or its exception raised here."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = call()
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(10.0)
    assert not thread.is_alive()
    if "error" in box:
        raise box["error"]
    return box.get("result")


class held:
    """Holds a lock on another thread for the length of a with block."""

    def __init__(self, lock: Any) -> None:
        self._lock = lock
        self._taken = threading.Event()
        self._free = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def _run(self) -> None:
        with self._lock:
            self._taken.set()
            self._free.wait(10.0)

    def __enter__(self) -> held:
        self._thread.start()
        assert self._taken.wait(5.0)
        return self

    def __exit__(self, *exc: object) -> None:
        self._free.set()
        self._thread.join(5.0)


# A scripted checker: the rules of a pass without APAI


class ScriptedGrid:
    def __init__(self) -> None:
        self._refresh_lock = threading.RLock()
        self._last_refresh: float | None = None
        self.refreshes = 0
        self.cache_clears = 0


class ScriptedHarness:
    """Records every check as (kind, index of q_next, grid stamp)."""

    def __init__(self, path: Sequence[np.ndarray]) -> None:
        self.occupancy = ScriptedGrid()
        self.violations: list[str] = []
        self.calls: list[tuple[str, int, tuple[int, int]]] = []
        self.refuse: dict[int, str] = {}
        self._path = path

    def _index(self, q_next: np.ndarray) -> int:
        return next(i for i, q in enumerate(self._path) if np.array_equal(q, q_next))

    def _stamp(self) -> tuple[int, int]:
        return (self.occupancy.refreshes, self.occupancy.cache_clears)

    def vet_step(self, q_prev: Any, q_next: Any, dt: float) -> str | None:
        index = self._index(q_next)
        expected = HOME if index == 0 else self._path[index - 1]
        assert np.array_equal(q_prev, expected) and dt == PERIOD_S
        self.calls.append(("vet", index, self._stamp()))
        return self.refuse.get(index)

    def approve(self, q_prev: Any, q_next: Any, dt: float) -> None:
        index = self._index(q_next)
        expected = HOME if index == 0 else self._path[index - 1]
        assert np.array_equal(q_prev, expected) and dt == PERIOD_S
        self.calls.append(("approve", index, self._stamp()))
        if index in self.refuse:
            raise RuntimeError(self.refuse[index])

    def indices(self, kind: str) -> list[int]:
        return [index for what, index, _ in self.calls if what == kind]


class Scripted:
    def __init__(self, steps: int = 100, **settings: Any) -> None:
        self.path = min_jerk_path(HOME, GOAL, steps)
        self.harness = ScriptedHarness(self.path)
        self.times = FrameTimes()
        self.check = CheckAhead(self.harness, self.times, period_s=PERIOD_S, **settings)
        self.new_grid()

    def new_grid(self, known: bool = True) -> float:
        """Install a grid from a new camera frame; its capture time."""
        grid = self.harness.occupancy
        frame_t = time.monotonic()
        with grid._refresh_lock:
            grid._last_refresh = frame_t
            grid.refreshes += 1
        if known:
            self.times.note(frame_t, frame_t + WALL_OFFSET_S)
        return frame_t + WALL_OFFSET_S

    def begin(self) -> MoveCheck:
        return self.check.begin(self.harness.approve, HOME, self.path)


def test_the_module_imports_without_apai() -> None:
    package = Path(checkahead.__file__).resolve().parents[3]
    code = (
        "import sys\n"
        "from isaaccapture_examples.agentic_remote.agent import checkahead\n"
        "checkahead.FrameTimes().note(1.0, 2.0)\n"
        "assert 'apai' not in sys.modules\n"
        "assert not any(n == 'isaaccapture' or n.startswith('isaaccapture.') "
        "for n in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(package), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_frame_times_return_what_was_noted_and_none_otherwise() -> None:
    times = FrameTimes()
    times.note(10.5, 1000.25)
    times.note(10.75, 1000.5)
    assert times.lookup(10.5) == 1000.25
    assert times.lookup(10.75) == 1000.5
    assert times.lookup(10.6) is None
    times.note(10.5, 2000.0)  # the same camera frame noted again: the newer value
    assert times.lookup(10.5) == 2000.0
    times.clear()
    assert times.lookup(10.5) is None and len(times) == 0


def test_frame_times_keep_the_last_256() -> None:
    times = FrameTimes()
    for i in range(300):
        times.note(float(i), 1000.0 + i)
    assert len(times) == 256
    assert times.lookup(43.0) is None
    assert times.lookup(44.0) == 1044.0
    assert times.lookup(299.0) == 1299.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_frame_times_refuse_a_time_that_is_not_finite(bad: float) -> None:
    times = FrameTimes()
    with pytest.raises(ValueError):
        times.note(bad, 1.0)
    with pytest.raises(ValueError):
        times.note(1.0, bad)
    assert len(times) == 0


def test_frame_times_take_notes_from_many_threads() -> None:
    times = FrameTimes()

    def writer(base: int) -> None:
        for i in range(500):
            times.note(base + i, float(i))
            times.lookup(base + i - 1)

    threads = [threading.Thread(target=writer, args=(k * 1000,)) for k in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    assert len(times) == 256


def test_a_checker_without_a_grid_is_refused() -> None:
    harness = ScriptedHarness([])
    harness.occupancy = None
    with pytest.raises(ValueError, match="obstacle grid"):
        CheckAhead(harness, FrameTimes(), period_s=PERIOD_S)


def test_grid_busy_is_not_a_safety_refusal() -> None:
    assert issubclass(GridBusy, Exception)
    assert not issubclass(GridBusy, RuntimeError)  # so not APAI's SafetyViolation


def test_first_pass_clears_through_the_reach_once_and_in_order() -> None:
    rig = Scripted()
    move = rig.begin()
    assert move.approved_through == -1 and move.refusal is None
    wall = rig.new_grid()
    grant = move.step(0, REACH)
    # Nothing is sent yet: the reach is setpoints 0..REACH-1.
    assert grant == Grant(through_index=REACH - 1, evidence_wall_s=wall)
    assert rig.harness.indices("approve") == list(range(REACH))
    assert rig.harness.indices("vet") == []
    assert move.approved_through == REACH - 1


def test_each_index_is_cleared_once_and_never_past_the_reach() -> None:
    rig = Scripted(steps=100)
    move = rig.begin()
    for sent in (0, 0, 3, 7, 7, 30, 60, 74, 90, 99, 100, 100):
        before = len(rig.harness.indices("approve"))
        grant = move.step(sent, REACH)
        assert grant is not None
        # The last setpoint sent is sent - 1: REACH setpoints beyond it.
        beyond = grant.through_index - (sent - 1)
        assert beyond == min(REACH, 100 - sent)
        assert grant.through_index == move.approved_through <= 99
        cleared = rig.harness.indices("approve")[before:]
        assert all(index <= sent + REACH - 1 for index in cleared)
    assert rig.harness.indices("approve") == list(range(100))


def test_a_reach_of_one_setpoint_still_moves_on() -> None:
    rig = Scripted(steps=6)
    move = rig.begin()
    for sent in range(6):
        grant = move.step(sent, 1)
        assert grant is not None and grant.through_index == sent  # the next one
    assert move.step(6, 1).through_index == 5  # never past the move's last index
    assert rig.harness.indices("approve") == list(range(6))


def test_a_pass_with_nothing_new_vets_the_next_setpoint_to_send() -> None:
    rig = Scripted()
    move = rig.begin()
    move.step(0, REACH)
    rig.harness.calls.clear()
    assert move.step(0, REACH) is not None
    assert rig.harness.calls == [("vet", 0, (1, 0))]
    rig.harness.calls.clear()
    assert move.step(100, REACH) is not None  # every setpoint is sent
    assert rig.harness.indices("vet") == []


def test_a_new_grid_vets_again_what_is_cleared_and_not_yet_sent() -> None:
    rig = Scripted()
    move = rig.begin()
    move.step(0, REACH)
    wall = rig.new_grid()
    rig.harness.calls.clear()
    grant = move.step(10, REACH)
    assert grant == Grant(through_index=34, evidence_wall_s=wall)
    assert rig.harness.calls == [
        *(("vet", index, (2, 0)) for index in range(10, 25)),
        *(("approve", index, (2, 0)) for index in range(25, 35)),
    ]
    rig.harness.calls.clear()
    move.step(10, REACH)  # the same grid: nothing is vetted twice
    assert rig.harness.calls == [("vet", 10, (2, 0))]


def test_a_dropped_grid_counts_as_a_new_one() -> None:
    rig = Scripted()
    move = rig.begin()
    move.step(0, REACH)
    rig.harness.occupancy.cache_clears += 1
    rig.harness.calls.clear()
    move.step(20, REACH)
    assert rig.harness.indices("vet") == list(range(20, 25))


def test_hold_max_caps_what_one_pass_clears() -> None:
    rig = Scripted(hold_max=8)
    move = rig.begin()
    assert move.step(0, REACH).through_index == 7
    assert move.step(0, REACH).through_index == 15
    assert move.step(0, REACH).through_index == 23
    assert move.step(0, REACH).through_index == REACH - 1
    assert rig.harness.indices("approve") == list(range(REACH))


def test_a_busy_grid_lock_gives_no_permit_and_checks_nothing() -> None:
    rig = Scripted(lock_wait_s=0.02)
    move = rig.begin()
    with held(rig.harness.occupancy._refresh_lock):
        began = time.monotonic()
        assert move.step(0, REACH) is None
        assert time.monotonic() - began < 0.5
        with pytest.raises(GridBusy):
            rig.check.vet(HOME, rig.path, 0, 5)
        assert rig.check.grid_stamp() == (1, 0)  # never blocks
    assert rig.harness.calls == [] and move.refusal is None
    assert move.step(0, REACH) is not None


def test_nothing_cleared_yet_gives_no_permit() -> None:
    rig = Scripted()
    move = rig.begin()
    assert move.step(0, -1) is None
    assert move.refusal is None and move.approved_through == -1


def test_a_callback_that_raises_is_kept_and_nothing_is_cleared_after_it() -> None:
    rig = Scripted()
    rig.harness.refuse[12] = "boom"
    move = rig.begin()
    assert move.step(0, REACH) is None
    assert isinstance(move.refusal, RuntimeError) and str(move.refusal) == "boom"
    assert move.approved_through == 11
    assert rig.harness.violations == []  # the callback files its own refusals
    kept = move.refusal
    rig.harness.calls.clear()
    rig.harness.refuse.clear()
    rig.new_grid()
    assert move.step(5, REACH) is None
    assert rig.harness.calls == [] and move.refusal is kept


def test_a_signal_raised_in_the_callback_is_not_swallowed() -> None:
    rig = Scripted()

    def approve(q_prev: Any, q_next: Any, dt: float) -> None:
        raise KeyboardInterrupt

    move = rig.check.begin(approve, HOME, rig.path)
    with pytest.raises(KeyboardInterrupt):
        move.step(0, REACH)
    assert rig.harness.occupancy._refresh_lock.acquire(timeout=0.5)  # released
    rig.harness.occupancy._refresh_lock.release()


def test_a_move_check_is_stepped_only_on_the_thread_that_made_it() -> None:
    rig = Scripted()
    move = rig.begin()
    with pytest.raises(RuntimeError, match="skill thread"):
        in_thread(lambda: move.step(0, REACH))
    assert rig.harness.calls == []


def test_vet_checks_the_named_setpoints_and_returns_the_grid_it_used() -> None:
    rig = Scripted()
    wall = rig.new_grid()
    evidence = rig.check.vet(HOME, rig.path, 3, 9)
    assert evidence == Evidence(wall_s=wall, grid_stamp=(2, 0))
    assert rig.harness.calls == [("vet", index, (2, 0)) for index in range(3, 10)]
    for first, last in ((-1, 3), (5, 4), (0, 100)):
        with pytest.raises(ValueError):
            rig.check.vet(HOME, rig.path, first, last)


def test_vet_whole_holds_the_lock_for_at_most_hold_max_setpoints() -> None:
    rig = Scripted(steps=500)
    lock = rig.harness.occupancy._refresh_lock
    holds: list[int] = []

    class Counting:
        def acquire(self, timeout: float = -1) -> bool:
            holds.append(len(rig.harness.calls))
            return lock.acquire(timeout=timeout)

        def release(self) -> None:
            lock.release()

    rig.check._lock = Counting()
    evidence = rig.check.vet_whole(HOME, rig.path)
    assert holds == list(range(0, 500, 64))
    assert rig.harness.indices("vet") == list(range(500))
    assert evidence.grid_stamp == (1, 0)
    with pytest.raises(ValueError):
        rig.check.vet_whole(HOME, [])


class GridAtHolds:
    """The grid lock, with a new grid installed before the holds it names."""

    def __init__(self, rig: Scripted, new_grid_at: Callable[[int], bool]) -> None:
        self._rig, self._new_grid_at = rig, new_grid_at
        self._lock = rig.harness.occupancy._refresh_lock
        self.holds = 0
        self.walls: dict[tuple[int, int], float] = {}

    def acquire(self, timeout: float = -1) -> bool:
        self.holds += 1
        if self._new_grid_at(self.holds):
            wall = self._rig.new_grid()
            self.walls[self._rig.check.grid_stamp()] = wall
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()


def test_vet_whole_starts_over_when_a_grid_arrives_between_two_holds() -> None:
    rig = Scripted(steps=100)  # two holds
    lock = rig.check._lock = GridAtHolds(rig, lambda hold: hold == 2)
    evidence = rig.check.vet_whole(HOME, rig.path)
    assert lock.holds == 4  # the first two count for nothing
    assert evidence == Evidence(wall_s=lock.walls[(2, 0)], grid_stamp=(2, 0))
    # Every setpoint was vetted on the grid the evidence names.
    on_it = [index for kind, index, stamp in rig.harness.calls if stamp == (2, 0)]
    assert on_it[-100:] == list(range(100))


def test_vet_whole_gives_up_after_three_tries_and_quotes_the_oldest_grid() -> None:
    rig = Scripted(steps=130)  # three holds
    lock = rig.check._lock = GridAtHolds(rig, lambda hold: True)
    evidence = rig.check.vet_whole(HOME, rig.path)
    # Two tries end at their second hold; the third runs to the end.
    assert lock.holds == 2 + 2 + 3
    stamps = [stamp for _, _, stamp in rig.harness.calls[-130:]]
    assert rig.harness.indices("vet")[-130:] == list(range(130))
    assert sorted(set(stamps)) == [(6, 0), (7, 0), (8, 0)]
    oldest = min(lock.walls[stamp] for stamp in set(stamps))
    assert evidence == Evidence(wall_s=oldest, grid_stamp=(6, 0))
    assert lock.walls[(8, 0)] > oldest


# On APAI's checker with a required obstacle grid


class GridClient:
    """Stands in for APAI's grid bridge: a distance grid around the arm."""

    def __init__(self) -> None:
        self.distance_m = 1.0  # from every cell to the nearest obstacle

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload["action"]
        if action in ("integrate", "integrate_depth"):
            return {}
        if action == "query":
            return {
                "points": np.empty((0, 3), dtype=np.float32),
                "grid": np.full((5, 5, 5), self.distance_m, dtype=np.float32),
                "origin": np.full(3, -2.0),
                "voxel": 1.0,
            }
        if action == "probe":
            return {"ok": True, "backend": "fake", "device": "cpu", "esdf": True}
        raise AssertionError(f"unexpected grid request {action!r}")

    def probe(self, timeout_ms: int = 300) -> dict[str, Any]:
        return self.request({"action": "probe"})


class Rig:
    """APAI's checker on a required grid, with the camera-frame times noted."""

    def __init__(self, kinematics: Any, safety: Any, **settings: Any) -> None:
        from apai.perception.occupancy import OccupancyMap
        from apai.safety.harness import SafetyHarness, SafetyLimits

        self.client = GridClient()
        self.grid = OccupancyMap(
            client=self.client,
            region_min=np.full(3, -2.0),
            region_max=np.full(3, 2.0),
            required=True,
            max_age_s=3.0,
        )
        self.harness = SafetyHarness(
            SafetyLimits.from_config(safety), kinematics=kinematics, occupancy=self.grid
        )
        self.times = FrameTimes()
        self.check = CheckAhead(self.harness, self.times, period_s=PERIOD_S, **settings)
        self.log: list[tuple[str, tuple[int, int], float | None]] = []
        self.new_grid()
        self.harness.heartbeat()
        self.harness.begin_motion()

    def new_grid(self, obstacle: bool = False, known: bool = True) -> float:
        """A camera frame goes into the grid; its capture time on the Thor."""
        from apai.types import Frame

        self.client.distance_m = 0.0 if obstacle else 1.0
        frame = Frame(
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_m=np.full((8, 8), 0.5, dtype=np.float32),
            K=np.array([[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]]),
        )
        if known:
            self.times.note(frame.t, frame.t + WALL_OFFSET_S)
        self.grid.refresh(frame, np.eye(4))
        assert self.grid.last_error is None and self.grid._last_refresh == frame.t
        return frame.t + WALL_OFFSET_S

    def recorded(self, path: Sequence[np.ndarray]) -> Callable[..., None]:
        """The checker callback, logging (index, stamp, evidence) per check."""
        vet_step = self.harness.vet_step

        def note(kind: str, q_next: Any) -> None:
            index = next(i for i, q in enumerate(path) if np.array_equal(q, q_next))
            wall = self.times.lookup(self.grid._last_refresh)
            self.log.append((f"{kind} {index}", self.check.grid_stamp(), wall))

        def logged_vet(q_prev: Any, q_next: Any, dt: float) -> str | None:
            note("vet", q_next)
            return vet_step(q_prev, q_next, dt)

        def approve(q_prev: Any, q_next: Any, dt: float) -> None:
            note("approve", q_next)
            self.harness.approve(q_prev, q_next, dt)

        self.harness.vet_step = logged_vet
        return approve


@pytest.fixture(scope="module")
def profile() -> Any:
    from isaaccapture_examples.agentic_remote.robot_app import apai_backend

    return apai_backend.load_apai_profile("mock")


@pytest.fixture
def rig(profile: Any) -> Rig:
    return Rig(profile.kinematics, profile.apai_cfg.safety)


@pytest.fixture
def path() -> list[np.ndarray]:
    return min_jerk_path(HOME, GOAL, 100)


@pytest.mark.apai
def test_check_refused_is_apais_safety_violation() -> None:
    from apai.types import MotionHalted, SafetyViolation

    refused = checkahead.CheckRefused
    assert refused is checkahead.CheckRefused and issubclass(refused, SafetyViolation)
    assert not issubclass(refused, MotionHalted)
    error = refused(7, "too close", 120)
    assert (error.index, error.reason, error.count) == (7, "too close", 120)
    assert str(error) == "check ahead refused setpoint 7 of 120: too close"
    assert str(refused(7, "too close")) == "check ahead refused setpoint 7: too close"


@pytest.mark.apai
def test_apais_callback_clears_each_setpoint_once_and_counts_it(
    rig: Rig, path: list[np.ndarray]
) -> None:
    move = rig.check.begin(rig.harness.approve, HOME, path)
    wall = rig.new_grid()
    assert move.step(0, REACH) == Grant(through_index=REACH - 1, evidence_wall_s=wall)
    assert rig.harness.approved_commands == REACH
    assert rig.harness.motions_begun == 1
    for sent in range(0, 101, 5):
        move.step(sent, REACH)
    assert move.approved_through == 99 and move.refusal is None
    assert rig.harness.approved_commands == 100  # vet_step counts nothing
    assert rig.harness.violations == []


@pytest.mark.apai
def test_every_permitted_setpoint_was_checked_on_the_quoted_grid(
    rig: Rig, path: list[np.ndarray]
) -> None:
    move = rig.check.begin(rig.recorded(path), HOME, path)
    walls = {rig.check.grid_stamp(): rig.times.lookup(rig.grid._last_refresh)}
    sent = 0
    for number in range(40):
        if number % 3 == 1:  # perception installs a grid between two passes
            wall = rig.new_grid()
            walls[rig.check.grid_stamp()] = wall
        grant = move.step(sent, REACH)
        assert grant is not None
        stamp = rig.check.grid_stamp()
        assert grant.evidence_wall_s == walls[stamp]
        assert grant.through_index <= min(99, sent + REACH - 1)
        on_this_grid = {
            int(what.split()[1])
            for what, at, wall in rig.log
            if at == stamp and wall == grant.evidence_wall_s
        }
        unsent = set(range(sent, grant.through_index + 1))
        assert unsent <= on_this_grid, sorted(unsent - on_this_grid)
        sent = min(100, sent + 4)
    assert sent == 100 and move.approved_through == 99
    cleared = [int(w.split()[1]) for w, _, _ in rig.log if w.startswith("approve")]
    assert cleared == list(range(100))


@pytest.mark.apai
def test_an_obstacle_in_a_new_grid_refuses_a_cleared_setpoint(
    rig: Rig, path: list[np.ndarray]
) -> None:
    move = rig.check.begin(rig.harness.approve, HOME, path)
    assert move.step(0, REACH) is not None
    rig.new_grid(obstacle=True)
    assert move.step(6, REACH) is None
    refusal = move.refusal
    assert isinstance(refusal, checkahead.CheckRefused)
    assert (refusal.index, refusal.count) == (6, 100)
    assert "clearance" in refusal.reason and "occupancy map" in refusal.reason
    assert str(refusal).startswith("check ahead refused setpoint 6 of 100: point")
    assert rig.harness.violations == [refusal.reason]
    assert move.approved_through == REACH - 1  # nothing more was cleared
    rig.new_grid()  # a free grid does not bring the move back
    assert move.step(6, REACH) is None and move.refusal is refusal
    assert rig.harness.approved_commands == REACH


@pytest.mark.apai
def test_an_obstacle_ahead_is_the_callbacks_own_refusal(
    rig: Rig, path: list[np.ndarray]
) -> None:
    from apai.types import SafetyViolation

    rig.new_grid(obstacle=True)
    move = rig.check.begin(rig.harness.approve, HOME, path)
    assert move.step(0, REACH) is None
    assert type(move.refusal) is SafetyViolation
    assert "clearance" in str(move.refusal)
    assert move.approved_through == -1
    assert rig.harness.violations == [str(move.refusal)]  # filed once, by APAI


@pytest.mark.apai
def test_a_grid_grown_old_refuses_also_when_nothing_is_left_to_clear(
    rig: Rig, path: list[np.ndarray]
) -> None:
    move = rig.check.begin(rig.harness.approve, HOME, path)
    assert move.step(0, REACH) is not None
    assert move.step(0, REACH) is not None
    rig.grid._last_refresh -= 3.5  # no camera frame for longer than max_age_s
    rig.times.note(rig.grid._last_refresh, 1.0)
    assert move.step(0, REACH) is None
    refusal = move.refusal
    assert isinstance(refusal, checkahead.CheckRefused) and refusal.index == 0
    assert refusal.reason.startswith(
        "required occupancy unavailable: distance grid is stale"
    )


@pytest.mark.apai
def test_a_grid_grown_old_stops_the_callback(rig: Rig, path: list[np.ndarray]) -> None:
    from apai.types import SafetyViolation

    move = rig.check.begin(rig.harness.approve, HOME, path)
    rig.grid._last_refresh -= 3.5
    assert move.step(0, REACH) is None
    assert type(move.refusal) is SafetyViolation
    assert "distance grid is stale" in str(move.refusal)
    assert move.approved_through == -1 and rig.harness.approved_commands == 0


@pytest.mark.apai
def test_a_latched_self_mask_refuses(rig: Rig, path: list[np.ndarray]) -> None:
    move = rig.check.begin(rig.harness.approve, HOME, path)
    assert move.step(0, REACH) is not None
    rig.grid._body_error = "robot body pose unavailable"
    assert move.step(0, REACH) is None
    refusal = move.refusal
    assert isinstance(refusal, checkahead.CheckRefused)
    assert refusal.reason.startswith("occupancy unsafe: robot body pose unavailable")
    with pytest.raises(checkahead.CheckRefused, match="occupancy unsafe") as caught:
        rig.check.vet_whole(HOME, path)
    assert caught.value.index == 0


@pytest.mark.apai
def test_a_grid_without_a_known_camera_frame_time_is_never_quoted(
    rig: Rig, path: list[np.ndarray]
) -> None:
    move = rig.check.begin(rig.harness.approve, HOME, path)
    assert move.step(0, REACH) is not None
    rig.new_grid(known=False)
    assert move.step(4, REACH) is None
    refusal = move.refusal
    assert isinstance(refusal, checkahead.CheckRefused)
    assert (refusal.index, refusal.reason) == (4, checkahead.NO_FRAME_TIME)
    assert rig.harness.violations == [checkahead.NO_FRAME_TIME]
    with pytest.raises(checkahead.CheckRefused, match="camera-frame time"):
        rig.check.vet(HOME, path, 0, 3)
    rig.times.clear()  # as after a step of the Thor's wall clock
    rig.new_grid(known=False)
    with pytest.raises(checkahead.CheckRefused, match="camera-frame time"):
        rig.check.vet_whole(HOME, path)


@pytest.mark.apai
def test_the_checkers_reason_comes_before_the_missing_frame_time(
    rig: Rig, path: list[np.ndarray]
) -> None:
    move = rig.check.begin(rig.harness.approve, HOME, path)
    assert move.step(0, REACH) is not None
    rig.new_grid(obstacle=True, known=False)
    assert move.step(4, REACH) is None
    assert "clearance" in move.refusal.reason


@pytest.mark.apai
def test_a_latch_refuses_through_the_checker(rig: Rig, path: list[np.ndarray]) -> None:
    move = rig.check.begin(rig.harness.approve, HOME, path)
    assert move.step(0, REACH) is not None
    rig.harness.estop("skill stop")
    assert move.step(0, REACH) is None
    assert move.refusal.reason == "e-stop latched"


@pytest.mark.apai
def test_vet_whole_passes_a_free_path_and_names_the_refused_setpoint(
    rig: Rig, path: list[np.ndarray]
) -> None:
    wall = rig.new_grid()
    evidence = rig.check.vet_whole(HOME, path)
    assert evidence == Evidence(wall_s=wall, grid_stamp=rig.check.grid_stamp())
    assert rig.harness.approved_commands == 0 and rig.harness.violations == []
    too_fast = [*path[:40], np.array(GOAL), *path[41:]]
    with pytest.raises(checkahead.CheckRefused) as caught:
        rig.check.vet_whole(HOME, too_fast)
    assert (caught.value.index, caught.value.count) == (40, 100)
    assert rig.harness.violations == [caught.value.reason]
    with pytest.raises(checkahead.CheckRefused) as caught:
        rig.check.vet(HOME, too_fast, 30, 50)
    assert caught.value.index == 40


@pytest.mark.apai
def test_vet_whole_refuses_when_the_grid_stays_busy(
    profile: Any, path: list[np.ndarray]
) -> None:
    rig = Rig(profile.kinematics, profile.apai_cfg.safety, whole_lock_wait_s=0.05)
    with held(rig.grid._refresh_lock):
        with pytest.raises(
            checkahead.CheckRefused, match=r"busy for 0\.05 s"
        ) as caught:
            rig.check.vet_whole(HOME, path)
    assert caught.value.index == 0


@pytest.mark.apai
def test_wait_fresh_grid_returns_at_once_on_a_young_grid(rig: Rig) -> None:
    rig.new_grid()
    with held(rig.grid._refresh_lock):  # it takes no lock
        began = time.monotonic()
        rig.check.wait_fresh_grid(1.0, 3.0)
        assert time.monotonic() - began < 0.2


@pytest.mark.apai
def test_wait_fresh_grid_waits_for_the_next_camera_frame(rig: Rig) -> None:
    rig.grid._last_refresh -= 2.0
    timer = threading.Timer(0.15, rig.new_grid)
    timer.start()
    try:
        began = time.monotonic()
        rig.check.wait_fresh_grid(1.0, 3.0)
        assert 0.1 < time.monotonic() - began < 1.0
    finally:
        timer.join()


@pytest.mark.apai
def test_wait_fresh_grid_gives_up_with_a_safety_violation(rig: Rig) -> None:
    from apai.types import SafetyViolation

    rig.grid._last_refresh -= 2.0
    began = time.monotonic()
    with pytest.raises(SafetyViolation) as caught:
        rig.check.wait_fresh_grid(1.0, 0.2)
    assert 0.2 <= time.monotonic() - began < 0.6
    assert str(caught.value) == (
        "no camera frame newer than 1 s in the obstacle grid; no motion sent"
    )
    assert rig.harness.violations == []
    with rig.grid._refresh_lock:
        rig.grid._drop_cache()
    with pytest.raises(SafetyViolation, match="no camera frame newer"):
        rig.check.wait_fresh_grid(1.0, 0.05)


@pytest.mark.apai
def test_times_of_the_checks_on_this_machine(
    rig: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prints what 5.2 and 5.3 ask for; the limits are only sanity bounds."""
    long_path = min_jerk_path(HOME, GOAL, 500)
    whole = []
    for _ in range(5):
        rig.new_grid()
        began = time.perf_counter()
        rig.check.vet_whole(HOME, long_path)
        whole.append(time.perf_counter() - began)
    first, later = [], []
    for _ in range(5):
        move = rig.check.begin(rig.harness.approve, HOME, long_path)
        began = time.perf_counter()
        assert move.step(0, REACH) is not None
        first.append(time.perf_counter() - began)
        began = time.perf_counter()
        assert move.step(1, REACH) is not None
        later.append(time.perf_counter() - began)
    rig.new_grid()
    began = time.perf_counter()
    assert move.step(1, REACH) is not None
    regrid = time.perf_counter() - began
    with capsys.disabled():
        print(
            f"\ncheck ahead on this machine (APAI's checker, a 5x5x5 grid, no load): "
            f"vet_whole of 500 setpoints {min(whole) * 1e3:.1f} ms "
            f"({min(whole) / 500 * 1e6:.0f} us per vet_step, 8 holds); "
            f"first pass with {REACH} checker callbacks {min(first) * 1e3:.2f} ms; "
            f"a pass that clears one setpoint {min(later) * 1e3:.2f} ms; "
            f"a pass after a new grid ({REACH} vet_step, no callback) "
            f"{regrid * 1e3:.2f} ms"
        )
    assert min(whole) < 2.0 and min(first) < 0.25
