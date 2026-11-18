# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the debug recording stores of the obstacle map, and what it does at its size.

The debug recording is the experiment's: the experiment is the folder of the
rig's earlier, single machine program, and its recorder (thor_recording.py)
stores what its hooks on APAI hand it. The agent program limits how many
pictures of the obstacle map that recorder stores a second (--debug-map-hz,
agent/debug_recording.py). At the recording's size (--debug-max-gib) it drops
the oldest of them, so that the recording and the experiment's viewer go on
(--debug-at-limit drop-oldest), or lets the recorder stop as it does alone
(stop). It says which happened.

A map picture is one tick of APAI's watcher: one camera frame through the
obstacle map's refresh and the detector. The tests run APAI's own watcher and
obstacle map on a stand-in camera with small pictures, a stand-in for the grid
bridge (the program that keeps the obstacle grid), and stand-ins for SAM, the
route check and the safety harness's report. The recorder, its hooks, its
viewer and its report are the experiment's own.
"""

from __future__ import annotations

import errno
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import test_launcher
import test_viewer
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent import debug_recording, launcher
from isaaccapture_examples.agentic_remote.agent.config import RunOptions
from test_launcher import EXAMPLE, QUICK, Stage, settings_file
from test_viewer import CELLS, Bridge, served

from conftest import import_experiment

apai = pytest.mark.apai
experiment = pytest.mark.experiment
# The fixtures of test_launcher.py and test_viewer.py, for the tests of this file.
make_stage = test_launcher.make_stage
stage = test_launcher.stage
viewer = test_viewer.viewer

README = EXAMPLE / "README.md"
# The stand-in camera's picture: rows, columns.
PICTURE = (48, 64)
# The watcher's ticks a second in the test that runs its thread.
TICK_HZ = 20.0
# The events of one map refresh of the stand-in scene, in the recorder's
# order: the frame, the map's input, the bridge's calls (a request and its
# response each), the commit and the end.
REFRESH_BEGINS = ["map.frame", "map.depth_input"]
BRIDGE_CALL = ["bridge.request", "bridge.response"]
REFRESH_ENDS = ["map.commit", "map.refresh_end"]
RATE_EVENT = debug_recording.RATE_EVENT
FULL = "debug recording full after "
REACHED = "debug recording reached "
DROP = debug_recording.AT_LIMIT_DROP
STOP = debug_recording.AT_LIMIT_STOP
RATE_BLOCK = debug_recording.RATE_BLOCK
DROPPED_BLOCK = debug_recording.DROPPED_BLOCK
# The stand-in camera's picture in the tests of the size: it does not
# compress, so a map picture (the camera frame for the map and once more for
# the detector) takes about 0.46 MB on disk. What stays of a dropped one, its
# rows and the map's points, is about 0.02 MB.
LARGE = (240, 320)
MB = 10**6
# A size the mock scene's pick and place passes: --debug-max-gib, in GiB.
# The run writes about 2.1 MB, 0.9 MB of them rows, and the recorder's check
# takes the 2 MB of a bridge answer before it is compressed.
SMALL_GIB = 0.0036
# The recorder's thread that writes the events.
WRITER = "evidence-writer"
# What the experiment's recorder says when it stops for its size.
SIZE_REASON = "recording size limit reached (conservative uncompressed reservation)"


class Clock:
    """A monotonic clock the test moves."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Harness:
    """In place of APAI's SafetyHarness, where the recorder's hooks read it."""

    def __init__(self, occupancy: Any) -> None:
        self.occupancy = occupancy
        self._clearance_reports: list[tuple[Any, dict[str, Any]]] = []
        self._clearance_lost = 0

    def _clearance_report(
        self,
        ctx: Any,
        gate: str,
        reason: Any,
        hit: Any,
        exempt: Any,
        pose: Any,
        raised: Any,
    ) -> None:
        report = {
            "gate": gate,
            "gate_refreshes": ctx.refreshes,
            "sample": [0.25, 0.125, 0.1],
            "clearance_m": 0.01,
            "margin_m": 0.03,
            "voxels": [],
        }
        self._clearance_reports.append((None, report))


def sam_masks(sam: Any, frame: Any, boxes: Any) -> list[Any]:
    """In place of SAM (BoxSegmenter.masks): one mask of the whole picture."""
    import numpy as np

    return [np.ones(frame.rgb.shape[:2], dtype=bool)]


def route_check(
    harness: Any, start: Any, goal: Any, duration_s: float, **options: Any
) -> None:
    """In place of APAI's vet_segment: the segment is accepted."""


class Scene:
    """APAI's watcher and obstacle map on a stand-in camera and grid bridge.

    tick() is one tick of the watcher on the calling thread: a new camera
    frame, the map's refresh, the detector. refuse(), segment() and
    grasp_mask() are what a skill's thread does: a refusal of the harness and
    a route segment, each on the map as it is, and SAM on a frame of its own.
    """

    def __init__(self, picture: tuple[int, int] = PICTURE, noisy: bool = False) -> None:
        import numpy as np
        from apai.perception import occupancy, segmenter, world
        from apai.safety import harness as safety_harness
        from apai.safety import trajectory
        from apai.types import Frame

        self._np, self._frame = np, Frame
        self._segmenter, self._trajectory = segmenter, trajectory
        self._safety, self._bridge_error = safety_harness, occupancy.OccupancyError
        self.bridge = Bridge()
        # A required map, as the rig's: a commit names its frame's capture time.
        self.map = occupancy.OccupancyMap(
            client=self.bridge,
            region_min=np.full(3, -1.0),
            region_max=np.full(3, 1.0),
            required=True,
        )
        self.harness = Harness(self.map)
        self.frames = 0
        self.picture = picture
        # Pictures of noise do not compress: their files have a size.
        self._noise = np.random.default_rng(0) if noisy else None
        # True: the camera has no new frame, so a tick ends at once.
        self.no_frame = False
        self.keeper: Any = None  # the Keeper of this scene's recording, if any
        # Called on the watcher's thread between the refresh and the detector.
        self.before_detector: Callable[[], None] | None = None
        # Called there inside the map's refresh, before the depth is sent.
        self.inside_refresh: Callable[[], None] | None = None
        scene = self
        body_mask = self.map._robot_depth_mask

        def masked(*args: Any, **kwargs: Any) -> Any:
            if scene.inside_refresh is not None:
                scene.inside_refresh()
            return body_mask(*args, **kwargs)

        self.map._robot_depth_mask = masked

        class Stream:
            name = "stand-in"

            def latest(self) -> Any:
                return None if scene.no_frame else scene.frame()

            def set_overlay(self, **shown: Any) -> None:
                pass

        class Detector:
            def detect(self, frame: Any, classes: Any = None) -> list[Any]:
                if scene.before_detector is not None:
                    scene.before_detector()
                scene.sam(frame)  # as the brain detector asks SAM for its boxes
                return []

        self.cam = world.WatchedCamera(
            stream=Stream(),
            depth=SimpleNamespace(ensure_depth=lambda frame: frame),
            extrinsics=SimpleNamespace(cam_to_base=lambda: np.eye(4)),
            fuse=False,  # no beliefs: the map alone takes the depth
            map_depth=True,
        )
        self.watcher = world.WorldWatcher(
            [self.cam], Detector(), beliefs=None, rate_hz=TICK_HZ, occupancy=self.map
        )

    def frame(self) -> Any:
        """The camera's next frame: a new picture, a metre deep everywhere."""
        np = self._np
        self.frames += 1
        K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]])
        shape = (*self.picture, 3)
        if self._noise is None:
            rgb = np.full(shape, self.frames % 251, dtype=np.uint8)
        else:
            rgb = self._noise.integers(0, 255, shape, dtype=np.uint8)
        return self._frame(
            rgb=rgb,
            depth_m=np.ones(self.picture, dtype=np.float32),
            K=K,
            frame_id=self.frames,
            depth_source="sensor",
        )

    def sam(self, frame: Any) -> None:
        boxes = self._np.array([[0.0, 0.0, 8.0, 8.0]], dtype=self._np.float32)
        self._segmenter.BoxSegmenter.masks(object(), frame, boxes)

    def tick(self) -> None:
        self.watcher._tick(self.cam)

    def refuse(self) -> int:
        """The harness refuses a pose on the map as it is; returns its revision."""
        np = self._np
        named = self.map.refreshes
        ctx = self._safety.GateCtx(
            "approve", np.zeros(6), None, np.eye(4), named, self.map.cache_clears
        )
        self._safety.SafetyHarness._file_clearance_report(
            self.harness,
            ctx,
            "tcp",
            "clearance 0.010 m under 0.030 m",
            None,
            None,
            np.eye(4),
        )
        return named

    def segment(self) -> int:
        """A route segment is checked on the map as it is; returns its revision."""
        named = self.map.refreshes
        self._trajectory.vet_segment(self.harness, [0.0] * 6, [0.1] * 6, 1.0)
        return named

    def grasp_mask(self) -> None:
        """SAM on a camera frame a skill took: an object mask of a grasp."""
        self.sam(self.frame())

    def retire(self) -> None:
        """The experiment's retire_target: a grasped object is taken out of the map.

        A call the bridge gets once, inside a refresh of the watcher. The
        stand-in bridge does not know it and answers as an old bridge does.
        """
        try:
            self.bridge.request({"action": "retire_target", "target_id": "cup"})
        except self._bridge_error:
            pass


@contextmanager
def recorded(
    run: Path,
    map_hz: float | None,
    clock: Callable[[], float] = time.monotonic,
    *,
    by_launcher: bool = False,
    max_bytes: int = 2**30,
    at_limit: str = DROP,
    picture: tuple[int, int] = PICTURE,
    noisy: bool = False,
) -> Iterator[tuple[Scene, Any, Any]]:
    """A Scene recorded into `run`; yields (scene, recorder, the limit or None).

    map_hz None is the experiment's recorder alone. by_launcher starts the
    recording as the agent program does (launcher.experiment_sessions).
    at_limit is what the recording does at max_bytes; its Keeper, if it has
    one, is scene.keeper.
    """
    from apai.perception import occupancy, segmenter
    from apai.safety import trajectory

    recording = import_experiment("thor_recording")
    with ExitStack() as stack:
        # Before the recorder's hooks: each wraps what it finds.
        for owner, name, stand_in in (
            (occupancy, "OccupancyClient", Bridge),
            (segmenter.BoxSegmenter, "masks", sam_masks),
            (trajectory, "vet_segment", route_check),
        ):
            stack.enter_context(mock.patch.object(owner, name, stand_in))
        gate = keeper = None
        if by_launcher:
            options = RunOptions(record_debug=True, debug_max_gib=max_bytes / 2**30)
            sessions = launcher.experiment_sessions(
                options, run, lambda prompt: "yes", map_hz=map_hz, at_limit=at_limit
            )
            _, _, recorder = stack.enter_context(sessions)
            # The keeper stands in place of the recorder's queue's get().
            keeper = getattr(recorder.events.get, "__self__", None)
            if not isinstance(keeper, debug_recording.Keeper):
                keeper = None
        else:
            session = recording.recording_session(run, max_bytes=max_bytes)
            recorder = stack.enter_context(session)
            if map_hz is not None:
                limits = debug_recording.limited(recorder, map_hz, at_limit, clock)
                gate, keeper = stack.enter_context(limits)
        scene = Scene(picture, noisy)
        scene.keeper = keeper
        yield scene, recorder, gate


def listed(run: Path, folder: Path) -> Path:
    """Make `run` a run folder the experiment's viewer lists; returns it."""
    settings = agent_config.load_settings(settings_file(folder))
    cfg = agent_config.build_apai_config(settings, run, camera="mock")
    agent_config.write_resolved_config(run, cfg)
    return run


def events(run: Path) -> list[dict[str, Any]]:
    text = (run / "debug" / "events.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines()]


def status(run: Path) -> dict[str, Any]:
    return json.loads((run / "debug" / "recording-status.json").read_text())


def refreshes(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The stored map refreshes: each its map.frame, then the events inside it.

    No event of a refresh is in the recording without that refresh's frame.
    """
    found = {row["id"]: [row] for row in rows if row["kind"] == "map.frame"}
    for row in rows:
        if row["refresh_id"] is not None:
            assert row["refresh_id"] in found, f"no map.frame for {row['kind']}"
            found[row["refresh_id"]].append(row)
    return list(found.values())


def check_whole(refresh: list[dict[str, Any]]) -> int:
    """One stored refresh holds all its events, joined by their ids; its revision."""
    calls = refresh[2:-2]
    assert [row["kind"] for row in refresh] == (
        REFRESH_BEGINS + BRIDGE_CALL * (len(calls) // 2) + REFRESH_ENDS
    )
    assert [row["id"] for row in refresh] == sorted(row["id"] for row in refresh)
    frame, commit, end = refresh[0], refresh[-2], refresh[-1]
    # The map's first refresh asks the bridge what it is, before the two
    # calls of every refresh: the depth, and the query for the grid.
    actions = [row["data"]["payload"]["action"] for row in calls[::2]]
    assert actions in (
        ["integrate_depth", "query"],
        ["probe", "integrate_depth", "query"],
    )
    for asked, answer in zip(calls[::2], calls[1::2], strict=True):
        assert answer["data"]["request_id"] == asked["id"]
    assert commit["data"]["query_request_id"] == calls[-2]["id"]
    assert end["data"]["refreshes"] == commit["data"]["refreshes"]
    assert end["data"]["exception"] is None and "arrays" in frame
    return commit["data"]["refreshes"]


def revisions(rows: list[dict[str, Any]]) -> list[int]:
    """The map revisions in the recording, each checked whole, in file order."""
    return [check_whole(refresh) for refresh in refreshes(rows)]


def kinds(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row["kind"] for row in rows)


# The rate


@apai
@experiment
def test_of_twenty_refreshes_a_second_about_two_are_stored_whole_and_the_viewer_follows(
    apai_scratch: Path, viewer: Any
) -> None:
    import cv2
    import numpy as np

    recording = import_experiment("thor_recording")
    runs = apai_scratch / "runs"
    run = listed(runs / "20261117T120000Z-1", apai_scratch)
    seconds = 3.0
    with recorded(run, 2.0, by_launcher=True) as (scene, recorder, _):
        with served(viewer, runs) as page:
            began = time.monotonic()
            scene.watcher.start()
            try:
                # While the watcher runs the page's state goes on: more
                # events, and a later revision of the map.
                first = page.state_when(lambda s: s["map"], "a map")
                later = page.state_when(
                    lambda s: (
                        s["map"]
                        and s["map"]["revision"] > first["map"]["revision"]
                        and s["events"] > first["events"]
                    ),
                    "a later map",
                )
                time.sleep(max(0.0, began + seconds - time.monotonic()))
            finally:
                scene.watcher.stop()
            took = time.monotonic() - began
        assert first["live"] is True and later["live"] is True
        assert first["map"]["points"] == CELLS and first["map"]["frame_id"]
        # The map itself took every camera frame: one refresh and one query
        # of the bridge per tick, 20 a second.
        ticks = scene.watcher.ticks
        assert ticks >= 0.5 * TICK_HZ * seconds  # a loaded machine ticks slower
        assert scene.map.refreshes == ticks == scene.frames
        assert scene.bridge.requests.count("query") == ticks
        assert scene.map.last_error is None
        hooks = list(recorder.hooks)
    rows = events(run)
    # About two a second were stored, each whole, and the detector's object
    # mask of the same camera frame with each. When the rate ended the newest
    # map was stored too, unless the rate had stored it: late, and it says so.
    stored = revisions(rows)
    frames = [row for row in rows if row["kind"] == "map.frame"]
    late = [row for row in frames if "recorded_late" in row["data"]]
    assert len(late) <= 1 and late == frames[len(frames) - len(late) :]
    for row in late:
        assert row["data"]["recorded_late"]["for"] == debug_recording.LATE_END
    by_rate = len(stored) - len(late)
    assert 1.2 * took <= by_rate <= 2 * took + 2, (by_rate, took)
    assert stored == sorted(set(stored)) and stored[0] == 1
    assert stored[-1] == ticks  # the map as it was at the end
    counts = kinds(rows)
    assert counts["object.mask"] == by_rate and counts["map.frame"] == len(stored)
    gaps = np.diff([row["mono_ns"] for row in frames[:by_rate]]) / 1e9
    assert gaps.min() > 0.3, gaps  # never two inside half a second
    # What the recording says of itself: the rate, and what it left out.
    opened, closed = (row["data"] for row in rows if row["kind"] == RATE_EVENT)
    assert opened == {"map_hz": 2.0, "note": debug_recording.RATE_NOTE}
    numbers = {
        "map_hz": 2.0,
        "ticks": ticks,
        "stored": by_rate,
        "skipped": ticks - by_rate,
        "rescued": len(late),
        "missing": 0,
    }
    assert closed == numbers
    assert f"{debug_recording.__name__}.MapRate.emit" in hooks
    assert f"{debug_recording.__name__}.Keeper.get" in hooks
    assert "apai.perception.world.WorldWatcher._tick" in hooks
    # No event was lost: what was left out never reached the recorder. The
    # status names the rate and its numbers, and does not say "complete":
    # map pictures were left out.
    left = status(run)
    assert left["finalized"] is True and left["dropped_events"] == 0
    assert left["written_events"] == left["accepted_events"] == len(rows)
    assert left["complete"] is False and left[RATE_BLOCK] == numbers
    assert left[DROPPED_BLOCK] == {
        "refreshes": 0,
        "bytes": 0,
        "first_time_kept": None,
        "points_dropped": 0,
        "first_time_points_kept": None,
        "named_after_drop": 0,
    }
    assert left["counts"] == dict(counts) and left["hooks"] == hooks
    # The experiment's viewer reads it: every map with its camera frame, its
    # input and the bridge's answer, and the picture of the last one.
    index = viewer.RunIndex(run)
    state = index.snapshot()
    assert state["events"] == len(rows) and state["map"]["revision"] == stored[-1]
    assert state["map"]["points"] == CELLS and state["map"]["query_present"] is True
    assert [entry["commit"]["data"]["refreshes"] for _, entry in index.maps] == stored
    for _, entry in index.maps:
        assert entry["frame"]["id"] == entry["commit"]["refresh_id"]
        assert entry["input"]["refresh_id"] == entry["frame"]["id"]
        assert (
            entry["query"]["data"]["request_id"]
            == (entry["commit"]["data"]["query_request_id"])
        )
    picture = cv2.imdecode(np.frombuffer(index.image(), np.uint8), cv2.IMREAD_COLOR)
    assert picture.shape == (*PICTURE, 3)
    # And the experiment's report.
    report = recording.generate_report(run).read_text()
    assert "Finalized: True; complete recorded coverage: False" in report
    for kind, count in counts.items():
        assert f"\n{kind}: {count}\n" in report, kind
    assert "Malformed/truncated event lines: 0" in report


@apai
@experiment
def test_the_rate_is_kept_over_a_watcher_that_is_no_multiple_of_it(
    apai_scratch: Path,
) -> None:
    run = apai_scratch / "run"
    clock = Clock()
    with recorded(run, 2.0, clock) as (scene, _, gate):
        # 6.6 ticks a second, as on the workstation, for 10 s.
        for _ in range(66):
            scene.tick()
            clock.now += 1 / 6.6
        # A watcher that stood for 5 s: one tick is stored when it goes on,
        # and no burst of them.
        clock.now += 5.0
        for _ in range(4):
            scene.tick()
            clock.now += 0.05
        assert gate.counts == {
            "ticks": 70,
            "stored": 21,
            "skipped": 49,
            "rescued": 0,
            "missing": 0,
        }
    rows = events(run)
    stored = revisions(rows)
    # The 21 of the rate, the last of them the 67th tick, and the map as it
    # was when the rate ended: the 70th, stored late.
    assert len(stored) == 22 and stored[-2:] == [67, 70]
    # 20 in the 10 s: every half second the first tick that is due.
    assert stored[:5] == [1, 5, 8, 11, 15]
    last = refreshes(rows)[-1]
    assert all(
        row["data"]["recorded_late"]["for"] == debug_recording.LATE_END for row in last
    )
    closed = [row["data"] for row in rows if row["kind"] == RATE_EVENT][-1]
    assert closed["stored"] == 21 and closed["rescued"] == 1


@apai
@experiment
def test_a_slower_watcher_than_the_rate_is_stored_whole(apai_scratch: Path) -> None:
    run = apai_scratch / "run"
    clock = Clock()
    with recorded(run, 2.0, clock) as (scene, _, gate):
        for _ in range(5):
            scene.tick()
            clock.now += 0.75
        assert gate.counts["stored"] == 5 and gate.counts["skipped"] == 0
    assert revisions(events(run)) == [1, 2, 3, 4, 5]


# The rate 0, and what the limit leaves as it is


def script(scene: Scene, clock: Clock) -> dict[str, int]:
    """Twelve ticks, 60 ms apart, with what a skill does between them.

    Returns the map revision each of the skill's events names.
    """
    named = {}
    for number in range(1, 13):
        scene.tick()
        clock.now += 0.06
        if number == 4:
            named["refusal"] = scene.refuse()
        if number == 7:
            scene.grasp_mask()
        if number == 9:
            named["segment"] = scene.segment()
    return named


@apai
@experiment
def test_with_the_rate_0_the_recording_is_the_experiments_own_and_the_map_never_differs(
    apai_scratch: Path,
) -> None:
    left = {}
    for name, map_hz, by_launcher, at_limit in (
        ("alone", None, False, STOP),  # the experiment's recorder, nothing of ours
        ("zero", 0.0, True, STOP),  # --debug-map-hz 0 --debug-at-limit stop
        ("kept", 0.0, True, DROP),  # --debug-map-hz 0
        ("two", 2.0, True, DROP),  # the defaults
    ):
        run = apai_scratch / name
        how = {"by_launcher": by_launcher, "at_limit": at_limit}
        with recorded(run, map_hz, **how) as (scene, recorder, _):
            named = script(scene, Clock())
            hooks = list(recorder.hooks)
            # What the agent plans with: the map took all twelve frames.
            seen = (scene.map.refreshes, scene.watcher.ticks, scene.bridge.requests)
        left[name] = (events(run), status(run), hooks, named, seen)
    alone, zero, kept, two = (left[name] for name in ("alone", "zero", "kept", "two"))
    # With 0 and "stop" every event is stored as the experiment's recorder
    # stores it: the same kinds in the same order with the same ids, nothing
    # added, and the same status, key by key.
    assert [(row["id"], row["kind"]) for row in zero[0]] == [
        (row["id"], row["kind"]) for row in alone[0]
    ]
    assert zero[1]["counts"] == alone[1]["counts"] and zero[2] == alone[2]
    assert zero[1]["counts"]["map.frame"] == 12 == len(revisions(zero[0]))
    assert zero[1]["counts"]["object.mask"] == 13  # twelve ticks and the grasp
    assert RATE_EVENT not in zero[1]["counts"]
    assert set(zero[1]) == set(alone[1]) and zero[1]["complete"] is True
    # With 0 alone the recording is the same until its size is reached, which
    # is not here: the keeper adds no event, and the status says that nothing
    # was dropped and stays "complete".
    assert [(row["id"], row["kind"]) for row in kept[0]] == [
        (row["id"], row["kind"]) for row in alone[0]
    ]
    assert kept[2] == [
        *alone[2],
        f"{debug_recording.__name__}.Keeper.get",
        "apai.perception.world.WorldWatcher._tick",
    ]
    assert kept[1]["complete"] is True and kept[1][DROPPED_BLOCK]["refreshes"] == 0
    assert set(kept[1]) - set(alone[1]) == {DROPPED_BLOCK}
    assert alone[3] == kept[3] and alone[4] == kept[4]
    # The map, its bridge and what the skill's events name are the same in
    # all three: the limit decides only what is written.
    assert alone[3] == zero[3] == two[3] == {"refusal": 4, "segment": 9}
    assert alone[4] == zero[4] == two[4]
    assert two[4][0] == 12 and two[4][2].count("query") == 12
    # The default stored fewer map pictures, and every other event.
    thinned = two[1]["counts"]
    assert thinned["map.frame"] < 12 and thinned[RATE_EVENT] == 2
    for kind in ("session.start", "safety.refusal", "route.segment.begin"):
        assert thinned[kind] == alone[1]["counts"][kind] == 1, kind
    assert thinned["route.segment.end"] == 1
    # It left map pictures out, so its status does not say "complete".
    assert two[1][RATE_BLOCK]["skipped"] > 0 and two[1]["complete"] is False
    assert two[1]["finalized"] is True and two[1]["dropped_events"] == 0


# What is always stored


@apai
@experiment
def test_a_refusal_and_a_route_segment_are_stored_with_the_map_they_name(
    apai_scratch: Path, viewer: Any
) -> None:
    run = listed(apai_scratch / "runs" / "20261117T120000Z-1", apai_scratch)
    clock = Clock()
    with recorded(run, 2.0, clock) as (scene, _, gate):
        named = script(scene, clock)
        map_id = id(scene.map)
        assert gate.counts == {
            "ticks": 12,
            "stored": 2,  # the first, and the tenth half a second later
            "skipped": 10,
            "rescued": 2,
            "missing": 0,
        }
    rows = events(run)
    by_kind = {row["kind"]: row for row in rows}  # the last of each kind
    refusal, begin = by_kind["safety.refusal"], by_kind["route.segment.begin"]
    end = by_kind["route.segment.end"]
    assert named == {"refusal": 4, "segment": 9}
    assert (
        refusal["data"]["gate_refreshes"]
        == 4
        == (refusal["data"]["report"]["gate_refreshes"])
    )
    assert begin["data"]["map"]["refreshes"] == 9 == end["data"]["map"]["refreshes"]
    assert end["data"]["call_id"] == begin["id"] and end["data"]["reason"] is None
    # Of twelve refreshes five are stored, each whole: the two of the rate,
    # the two that were named, and the last one when the rate ended.
    assert revisions(rows) == [1, 4, 9, 10, 12]
    stored = {check_whole(refresh): refresh for refresh in refreshes(rows)}
    assert all(
        row["data"]["recorded_late"]["for"] == debug_recording.LATE_END
        for row in stored[12]
    )
    for number, event in ((4, refusal), (9, begin)):
        refresh = stored[number]
        commit = refresh[-2]
        assert (
            commit["data"]["map_id"]
            == map_id
            == event["data"].get("map_id", event["data"].get("map", {}).get("map_id"))
        )
        # Stored just before the event that names it, and it says so: each
        # of its events carries when it happened and on which thread.
        assert refresh[-1]["id"] == event["id"] - 1
        for row in refresh:
            late = row["data"]["recorded_late"]
            assert late["for"] == event["kind"] and late["thread"] == "MainThread"
            assert late["event_mono_ns"] <= row["mono_ns"] <= event["mono_ns"]
        happened = [row["data"]["recorded_late"]["event_mono_ns"] for row in refresh]
        assert happened == sorted(happened)
    for number in (1, 10):
        assert all("recorded_late" not in row["data"] for row in stored[number])
    # The detector's object mask is stored with the two ticks of the rate;
    # the mask of the grasp, from the skill's thread, always.
    masks = [row for row in rows if row["kind"] == "object.mask"]
    assert len(masks) == 3 and all(row["refresh_id"] is None for row in masks)
    # The experiment's viewer at the refusal: the map it was checked against.
    index = viewer.RunIndex(run)
    at = (refusal["mono_ns"] - rows[0]["mono_ns"]) / 1e9 + 1e-6
    state = index.snapshot(at)
    assert state["failure"]["id"] == refusal["id"]
    assert state["failure"]["map_matches"] is True and state["map"]["revision"] == 4
    assert state["failure"]["obstacles"]["nearest"] in CELLS
    assert index.image(at)[:2] == b"\xff\xd8"  # a JPEG: the frame of that map


@apai
@experiment
def test_a_map_named_while_its_tick_still_runs_is_stored_in_one_piece(
    apai_scratch: Path,
) -> None:
    run = apai_scratch / "run"
    clock = Clock()
    with recorded(run, 2.0, clock) as (scene, _, gate):
        scene.tick()  # stored by the rate
        clock.now += 0.05

        def skill(act: Callable[[], Any]) -> None:
            """What another thread does while the watcher's tick goes on."""
            thread = threading.Thread(target=act, name="skill")
            thread.start()
            thread.join(10.0)
            assert not thread.is_alive()

        # Between the map's commit and the end of its refresh.
        report = scene.map._report_refresh_state
        scene.map._report_refresh_state = lambda: (skill(scene.refuse), report())[1]
        scene.tick()
        scene.map._report_refresh_state = report
        clock.now += 0.05
        # Between the refresh and the detector.
        scene.before_detector = lambda: skill(scene.segment)
        scene.tick()
        scene.before_detector = None
        assert gate.counts["rescued"] == 2 and gate.counts["missing"] == 0
    rows = events(run)
    assert revisions(rows) == [1, 2, 3]
    second, third = refreshes(rows)[1:]
    # The refusal stands between the commit it names and that refresh's end,
    # which the watcher's thread recorded itself, under the frame's id.
    refusal = next(row for row in rows if row["kind"] == "safety.refusal")
    assert second[-2]["id"] < refusal["id"] < second[-1]["id"]
    assert "recorded_late" in second[-2]["data"]
    assert "recorded_late" not in second[-1]["data"]
    assert refusal["thread"] == "skill" and refusal["refresh_id"] is None
    # A tick that was stored for its map goes on as a stored one: the
    # detector's object mask, which came after, is in the recording.
    masks = [row["id"] for row in rows if row["kind"] == "object.mask"]
    assert len(masks) == 3 and masks[1] > second[-1]["id"]
    assert masks[2] > third[-1]["id"]


@apai
@experiment
def test_a_refresh_with_a_bridge_call_of_its_own_is_stored_whole(
    apai_scratch: Path,
) -> None:
    run = apai_scratch / "run"
    clock = Clock()
    with recorded(run, 2.0, clock) as (scene, _, gate):
        for retiring in (False, True, False):
            scene.inside_refresh = scene.retire if retiring else None
            scene.tick()
            clock.now += 0.05
        assert gate.counts == {
            "ticks": 3,
            "stored": 1,
            "skipped": 2,
            "rescued": 1,  # the second, for its retire_target
            "missing": 0,
        }
    rows = events(run)
    # The third tick's refresh is there too: the newest map when the rate ended.
    first, second, third = refreshes(rows)
    assert check_whole(first) == 1 and check_whole(third) == 3
    # The refresh of the second tick, from its frame on: the frame was held
    # and says so; the bridge's call and what came after went straight on.
    frame, retire, answer, *routine = second
    assert frame["kind"] == "map.frame"
    assert frame["data"]["recorded_late"]["for"] == "bridge.request"
    assert retire["data"]["payload"] == {"action": "retire_target", "target_id": "cup"}
    assert answer["data"]["request_id"] == retire["id"]
    assert "unknown action 'retire_target'" in answer["data"]["exception"]
    assert [row["kind"] for row in routine] == [
        "map.depth_input",
        *BRIDGE_CALL * 2,
        *REFRESH_ENDS,
    ]
    assert routine[-2]["data"]["refreshes"] == 2
    assert all("recorded_late" not in row["data"] for row in second[1:])
    assert [row["id"] for row in second] == sorted(row["id"] for row in second)


@apai
@experiment
def test_what_the_arm_did_is_stored_whatever_the_rate_also_inside_a_tick(
    apai_scratch: Path,
) -> None:
    # A refusal and a route segment: the test before the last.
    always = (
        "verification.measurement",
        "robot.feedback",
        launcher.STATE_EVENT,
        "robot.call",
        "robot.return",
        "session.end",
    )
    run = apai_scratch / "run"
    clock = Clock()
    with recorded(run, 2.0, clock) as (scene, recorder, gate):
        scene.tick()
        clock.now += 0.05
        # A tick the rate skips: the watcher's thread records each kind in it.
        scene.before_detector = lambda: [
            recorder.record(kind, lambda: {"note": "inside a tick"}) for kind in always
        ]
        scene.tick()
        scene.before_detector = None
        # Outside a tick: the skill's thread and the measuring thread.
        for kind in always:
            recorder.emit(kind, note="outside")
        for _ in range(3):
            scene.grasp_mask()
        assert gate.counts["skipped"] == 1 and gate.counts["rescued"] == 0
    rows = events(run)
    counts = kinds(rows)
    # The skipped tick's map was not stored by the rate: it is there as the
    # newest map when the rate ended, behind everything the tick recorded.
    assert revisions(rows) == [1, 2]
    skipped = refreshes(rows)[1]
    assert skipped[0]["data"]["recorded_late"]["for"] == debug_recording.LATE_END
    assert skipped[0]["id"] > max(row["id"] for row in rows if "note" in row["data"])
    for kind in always:
        assert counts[kind] == (3 if kind == "session.end" else 2), kind
    assert counts["object.mask"] == 4  # the first tick's and the three grasps'
    assert counts["session.start"] == 1
    inside = [row for row in rows if row["data"].get("note") == "inside a tick"]
    assert len(inside) == len(always)
    assert all(row["refresh_id"] is None for row in inside)


@apai
@experiment
def test_a_named_map_that_is_no_longer_held_is_counted_and_the_event_is_stored(
    apai_scratch: Path,
) -> None:
    run = apai_scratch / "run"
    clock = Clock()
    held = debug_recording.HELD_TICKS
    with recorded(run, 2.0, clock) as (scene, _, gate):
        for _ in range(held + 3):  # revisions 1 to held + 3 inside half a second
            scene.tick()
            clock.now += 0.01
        newest = scene.map.refreshes
        scene.map.refreshes = 2  # a revision the limit let go
        assert scene.refuse() == 2
        scene.map.refreshes = 1  # one the rate stored
        scene.refuse()
        scene.map.refreshes = newest
        for back in range(held):  # the newest ones are held
            scene.map.refreshes = newest - back
            scene.refuse()
        scene.map.refreshes = newest
        assert gate.counts["missing"] == 1 and gate.counts["rescued"] == held
    rows = events(run)
    assert kinds(rows)["safety.refusal"] == held + 2
    assert sorted(revisions(rows)) == [1, *range(newest - held + 1, newest + 1)]


@apai
@experiment
@pytest.mark.parametrize(
    ("owner", "name"),
    [
        (debug_recording.MapRate, "_decide"),  # whether a tick is stored
        (debug_recording, "_copied"),  # the copy of a skipped tick's event
    ],
)
def test_an_error_in_the_limit_stores_the_event_and_reaches_nothing_else(
    apai_scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    owner: Any,
    name: str,
) -> None:
    def broken(*args: Any) -> None:
        raise RuntimeError("the limit is broken")

    monkeypatch.setattr(owner, name, broken)
    run = apai_scratch / "run"
    clock = Clock()
    with caplog.at_level(logging.WARNING), recorded(run, 2.0, clock) as (scene, _, _):
        for _ in range(5):
            scene.tick()
            clock.now += 0.05
        assert scene.map.refreshes == 5 and scene.map.last_error is None
    # Every refresh is in the recording, as without the limit; said once.
    assert revisions(events(run)) == [1, 2, 3, 4, 5]
    told = [
        r for r in caplog.records if "the events are stored as they come" in r.message
    ]
    assert len(told) == 1


def test_a_recorder_that_is_not_the_experiments_is_left_as_it_is(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Evidence:
        """A recorder with record() alone, as the stand-in of test_agent_options.py."""

        def __init__(self) -> None:
            self.hooks: list[str] = []

        def record(self, kind: str, builder: Callable[[], dict[str, Any]]) -> int:
            return 1

    evidence = Evidence()
    with caplog.at_level(logging.WARNING):
        with debug_recording.map_rate(evidence, 2.0) as gate:
            assert gate is None and evidence.hooks == []
    assert "--debug-map-hz does not limit it" in caplog.text
    # No recorder, or the rate 0: nothing is limited and nothing is said.
    caplog.clear()
    for recording, map_hz in ((None, 2.0), (evidence, 0.0)):
        with debug_recording.map_rate(recording, map_hz) as gate:
            assert gate is None
    assert caplog.text == ""


@apai
def test_a_watcher_without_a_tick_is_refused_before_anything_is_put_on_the_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apai.perception import world

    def emit(kind: str, **data: Any) -> int:
        raise AssertionError(f"{kind} was recorded")

    recorder = SimpleNamespace(emit=emit, local=threading.local(), hooks=[])
    monkeypatch.delattr(world.WorldWatcher, "_tick")
    with pytest.raises(RuntimeError, match="--debug-map-hz 0 stores every one"):
        with debug_recording.map_rate(recorder, 2.0):
            raise AssertionError("the limit began")
    assert recorder.emit is emit and recorder.hooks == []


@apai
@experiment
def test_the_limit_leaves_the_recorder_and_the_watcher_as_they_were(
    apai_scratch: Path,
) -> None:
    from apai.perception import world

    recording = import_experiment("thor_recording")
    tick = world.WorldWatcher._tick
    with recorded(apai_scratch / "run", 2.0) as (scene, recorder, gate):
        assert recorder.emit == gate.emit and world.WorldWatcher._tick is not tick
        assert world.WorldWatcher._tick.__wrapped__ is tick
        scene.tick()
    # The recorder's own emit() again, and the watcher's own tick.
    assert "emit" not in vars(recorder)
    assert recorder.emit.__func__ is recording.EvidenceRecorder.emit
    assert world.WorldWatcher._tick is tick


def test_the_module_imports_without_apai_numpy_or_the_experiment() -> None:
    import os
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from isaaccapture_examples.agentic_remote.agent import debug_recording\n"
        "bad = [n for n in sys.modules if n.split('.')[0] in "
        "('apai', 'torch', 'numpy', 'isaaccapture', 'thor_recording')]\n"
        "assert not bad, bad\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AGENTIC_REMOTE_") and key != "PYTHONPATH"
    }
    env["PYTHONPATH"] = str(EXAMPLE / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


# The newest map when no newer one comes


@apai
@experiment
def test_a_stall_of_the_map_stores_the_newest_held_map_after_a_period_of_the_rate(
    apai_scratch: Path, viewer: Any
) -> None:
    run = listed(apai_scratch / "runs" / "20261117T120000Z-1", apai_scratch)
    clock = Clock()

    def shown() -> int:
        """The map revision the experiment's viewer shows as the newest."""
        recorder.events.join()
        return viewer.RunIndex(run).snapshot()["map"]["revision"]

    def uncommitted() -> None:
        """A tick whose refresh ends without a commit, as without the arm's pose."""

        def failing() -> None:
            raise RuntimeError("robot body pose unavailable")

        scene.inside_refresh = failing
        with pytest.raises(RuntimeError, match="pose unavailable"):
            scene.tick()  # APAI's watcher loop catches it
        scene.inside_refresh = None

    with recorded(run, 2.0, clock) as (scene, recorder, gate):
        began = clock.now
        scene.tick()  # revision 1: stored by the rate
        clock.now = began + 0.05
        scene.tick()  # revision 2: skipped and held
        assert shown() == 1
        # The camera gives no new frame: the tick ends at once. A map was
        # stored 0.1 s ago, so nothing is stored for it.
        scene.no_frame = True
        clock.now = began + 0.10
        scene.tick()
        scene.no_frame = False
        clock.now = began + 0.45
        uncommitted()
        assert shown() == 1 and gate.counts["rescued"] == 0
        # Half a second after the last stored map: the held one is stored.
        scene.no_frame = True
        clock.now = began + 0.55
        scene.tick()
        assert shown() == 2 and gate.counts["rescued"] == 1
        for late in (0.60, 0.90, 1.15):  # and nothing more while the stall lasts
            clock.now = began + late
            scene.tick()
        scene.no_frame = False
        assert gate.counts["rescued"] == 1
        # The map goes on: 3 is stored, 4 is skipped and held, 5 is stored.
        for at in (1.20, 1.25, 1.80):
            clock.now = began + at
            scene.tick()
        # A stall behind a stored map: the held one is older and stays out,
        # also when the rate ends.
        scene.no_frame = True
        clock.now = began + 2.50
        scene.tick()
        assert shown() == 5
        assert gate.counts == {
            "ticks": 6,  # those with a map event: five commits and the failed one
            "stored": 3,
            "skipped": 3,
            "rescued": 1,
            "missing": 0,
        }
    rows = events(run)
    stored = [refresh for refresh in refreshes(rows) if len(refresh) > 2]
    assert [check_whole(refresh) for refresh in stored] == [1, 2, 3, 5]
    for row in stored[1]:
        late = row["data"]["recorded_late"]
        assert late["for"] == debug_recording.LATE_STALL
        assert late["event_mono_ns"] < row["mono_ns"]
    assert all("recorded_late" not in row["data"] for row in stored[3])
    assert status(run)[RATE_BLOCK]["rescued"] == 1


# At the recording's size


def folder_bytes(folder: Path) -> int:
    """What a folder takes on disk: the sizes of its files."""
    return sum(file.stat().st_size for file in folder.rglob("*") if file.is_file())


def by_revision(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """The stored map refreshes that committed, by the revision they made."""
    found = {}
    for refresh in refreshes(rows):
        for row in refresh:
            if row["kind"] == "map.commit":
                found[row["data"]["refreshes"]] = refresh
    return found


def there(run: Path, row: dict[str, Any]) -> bool:
    """Whether the arrays of an event are in the recording; True without any."""
    return "arrays" not in row or (run / "debug" / row["arrays"]).is_file()


def seconds(rows: list[dict[str, Any]], row: dict[str, Any]) -> float:
    """Just behind an event, in seconds on the viewer's time line."""
    return (row["mono_ns"] - rows[0]["mono_ns"]) / 1e9 + 1e-6


def is_picture(jpeg: bytes, shape: tuple[int, int] = LARGE) -> bool:
    import cv2
    import numpy as np

    picture = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return picture is not None and picture.shape == (*shape, 3)


@apai
@experiment
def test_at_its_size_the_recording_drops_its_oldest_map_pictures_and_goes_on(
    apai_scratch: Path, viewer: Any
) -> None:
    import numpy as np

    recording = import_experiment("thor_recording")
    runs = apai_scratch / "runs"
    run = listed(runs / "20261117T120000Z-1", apai_scratch)
    log = run / "debug" / "events.jsonl"
    # 160 stored map pictures, 74 MB, into 12 MB: six times its size.
    limit = 12 * MB
    ticks = 320
    clock = Clock()
    noise = np.random.default_rng(1).integers(0, 255, (300, 300), dtype=np.uint8)
    # What a skill does between the ticks, by the tick it comes after.
    refusals, segments = (20, 120), (41, 141)
    # The newest stored map behind each tick that is looked at: the rate
    # stores the odd ticks; these even ones are stored for what names them.
    named = {20: 20, 80: 80, 120: 120}
    grasp = measured = 0
    with recorded(run, 2.0, clock, max_bytes=limit, picture=LARGE, noisy=True) as (
        scene,
        recorder,
        gate,
    ):
        keeper = scene.keeper
        index = viewer.Viewer(runs).run("latest")
        written, largest = b"", 0
        for number in range(1, ticks + 1):
            if number == 80:
                scene.inside_refresh = scene.retire
            scene.tick()
            scene.inside_refresh = None
            clock.now += 0.25  # every second tick is stored by the rate
            if number in refusals:
                assert scene.refuse() == number  # a skipped tick: stored late
            if number in segments:
                assert scene.segment() == number  # a tick the rate stored
            if number == 60:
                scene.grasp_mask()
                grasp = recorder.seq
            if number == 100:
                measured = recorder.emit("verification.measurement", mask=noise)
            if number % 20:
                continue
            recorder.events.join()  # everything so far is written
            # The recorder goes on, and its folder stays under the size.
            assert recorder.writer_error is None and not recorder.disabled, number
            largest = max(largest, folder_bytes(run / "debug"))
            # events.jsonl is only appended to.
            now = log.read_bytes()
            assert now.startswith(written) and len(now) > len(written)
            written = now
            # The experiment's viewer follows: the newest stored map, its
            # points and its picture.
            state = index.snapshot()
            assert state["map"]["revision"] == named.get(number, number - 1), number
            assert state["map"]["points"] == CELLS and state["map"]["query_present"]
            assert is_picture(index.image()), number
        assert scene.map.refreshes == ticks and scene.map.last_error is None
        counts = dict(gate.counts)
    dropped = keeper.dropped
    # The size was passed several times over, and never on disk.
    assert largest <= limit and folder_bytes(run / "debug") <= limit
    assert dropped.bytes >= 3 * limit and dropped.refreshes > 100
    rows = events(run)
    left = status(run)
    assert log.read_bytes().startswith(written)
    assert left["writer_error"] is None and left["dropped_events"] == 0
    assert left["written_events"] == left["accepted_events"] == len(rows)
    assert left["written_bytes"] <= limit == left["max_bytes"]
    # No row went: the ticks the rate stored are all there, from the first.
    stored = by_revision(rows)
    assert counts["stored"] == ticks // 2 and counts["skipped"] == ticks // 2
    assert set(range(1, ticks, 2)) <= set(stored)
    # What the status says: the rate's numbers, what was dropped, and not
    # "complete". The last tick was stored when the rate ended.
    assert left["finalized"] is True and left["complete"] is False
    assert left[RATE_BLOCK] == {
        "map_hz": 2.0,
        **counts,
        "rescued": counts["rescued"] + 1,
    }
    assert counts["rescued"] == 3 and counts["missing"] == 0  # 20, 80 and 120
    kept_from = left[DROPPED_BLOCK]["first_time_kept"]
    assert left[DROPPED_BLOCK] == {
        "refreshes": dropped.refreshes,
        "bytes": dropped.bytes,
        "first_time_kept": kept_from,
        "points_dropped": 0,
        "first_time_points_kept": None,
        "named_after_drop": 0,
    }
    report = recording.generate_report(run).read_text()
    assert "Finalized: True; complete recorded coverage: False" in report
    # What stays whatever was dropped. The first map of the session, the
    # maps a refusal or a segment named, stored late for it or by the rate,
    # the refresh with a bridge call of its own, and the newest ones: whole.
    by_id = {row["id"]: row for row in rows}
    for number in (1, *refusals, *segments, 80, ticks - 1, ticks):
        assert all(there(run, row) for row in stored[number]), number
    for number in segments:  # with the detector's mask of the same tick
        mask = by_id[stored[number][-1]["id"] + 1]
        assert mask["kind"] == "object.mask" and "arrays" in mask and there(run, mask)
    # The mask of the grasp and the measurement, from the skill's thread.
    assert by_id[grasp]["kind"] == "object.mask" and "arrays" in by_id[grasp]
    assert by_id[measured]["kind"] == "verification.measurement"
    assert there(run, by_id[grasp]) and there(run, by_id[measured])
    evidence = {
        debug_recording.REFUSAL,
        *debug_recording.SEGMENTS,
        "verification.measurement",
    }
    read = kinds(list(recording.read_events(run, kinds=evidence)))
    assert read == {
        debug_recording.REFUSAL: 2,
        "route.segment.begin": 2,
        "route.segment.end": 2,
        "verification.measurement": 1,
    }
    # What went: the pictures of an old tick, the camera frame, the depth
    # and the detector's mask. Its rows are there, and the bridge's answer
    # with the map's points.
    gone = stored[101]
    frame, commit = gone[0], gone[-2]
    assert check_whole(gone) == 101 and not there(run, frame)
    assert not there(run, by_id[gone[-1]["id"] + 1])  # the detector's mask
    answer = by_id[commit["data"]["query_request_id"] + 1]
    assert answer["kind"] == "bridge.response" and "arrays" in answer
    assert there(run, answer)
    # The experiment's viewer on the finished run. Live: the last map.
    index = viewer.Viewer(runs).run("latest")
    state = index.snapshot()
    assert state["map"]["revision"] == ticks and state["recording"] == left
    assert is_picture(index.image())
    # A time in the dropped stretch: the map and its points, and no picture.
    at = seconds(rows, commit)
    state = index.snapshot(at)
    assert state["map"]["revision"] == 101 and state["map"]["points"] == CELLS
    assert state["map"]["frame_id"] == frame["id"]
    with pytest.raises(FileNotFoundError, match="No such file"):
        index.image(at)
    # Following live works after it.
    assert index.snapshot()["map"]["revision"] == ticks and is_picture(index.image())
    # A refusal: the map it was checked against, with its picture.
    for number in refusals:
        refusal = next(
            row
            for row in rows
            if row["kind"] == debug_recording.REFUSAL
            and row["data"]["gate_refreshes"] == number
        )
        state = index.snapshot(seconds(rows, refusal))
        assert state["failure"]["id"] == refusal["id"]
        assert state["failure"]["map_matches"] is True
        assert state["map"]["revision"] == number
        assert is_picture(index.image(seconds(rows, refusal)))
    # The first map of the session.
    first = seconds(rows, stored[1][-2])
    assert index.snapshot(first)["map"]["revision"] == 1 and is_picture(
        index.image(first)
    )
    # From first_time_kept on every map has its picture; the one before has not.
    # The block names that time rounded up to a millisecond.
    commits = [row for row in rows if row["kind"] == "map.commit"]
    times = [(row["mono_ns"] - rows[0]["mono_ns"]) / 1e9 for row in commits]
    edge = next(i for i, moment in enumerate(times) if moment > kept_from - 1e-3)
    assert times[edge] <= kept_from
    earlier, later = commits[:edge], commits[edge:]
    assert len(later) >= debug_recording.KEPT_NEWEST and len(earlier) > 100
    assert all(there(run, by_id[row["refresh_id"]]) for row in later)
    assert not there(run, by_id[earlier[-1]["refresh_id"]])
    assert is_picture(index.image(kept_from))
    assert (
        index.snapshot(kept_from)["map"]["revision"] == (later[0]["data"]["refreshes"])
    )


# The methods of the keeper that run on the recorder's writer thread, and
# the two that run on the thread of a tick: its marks.
WRITER_SIDE = (
    "get",
    "_note",
    "_tick_of",
    "_make_room",
    "_oldest",
    "_remove",
    "_bytes",
    "_publish",
    "status",
)
EVENT_SIDE = ("begin", "end")


@pytest.fixture
def keeper_threads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """(method, thread name) of every call of the keeper's methods."""
    calls: list[tuple[str, str]] = []

    def noted(name: str) -> Callable[..., Any]:
        real = getattr(debug_recording.Keeper, name)

        def method(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append((name, threading.current_thread().name))
            return real(self, *args, **kwargs)

        return method

    for name in (*WRITER_SIDE, *EVENT_SIDE):
        monkeypatch.setattr(debug_recording.Keeper, name, noted(name))
    return calls


@apai
@experiment
def test_the_viewers_page_follows_live_past_the_size_and_a_dropped_time_has_no_picture(
    apai_scratch: Path, viewer: Any, keeper_threads: list[tuple[str, str]]
) -> None:
    runs = apai_scratch / "runs"
    run = listed(runs / "20261117T120000Z-1", apai_scratch)
    limit = 12 * MB

    def live_picture() -> bool:
        code, kind, body = page.get("/api/image?run=latest")
        return (code, kind) == (200, "image/jpeg") and is_picture(body)

    # Started as the agent program starts it, with every map picture stored:
    # 20 a second, about 9 MB a second.
    with recorded(
        run, 0.0, by_launcher=True, max_bytes=limit, picture=LARGE, noisy=True
    ) as (scene, recorder, _):
        keeper = scene.keeper
        with served(viewer, runs) as page:
            scene.watcher.start()
            try:
                # What the page asks every half second while it follows live.
                shown: list[int] = []
                beyond: list[int] = []  # of them, those behind the first drop
                end = time.monotonic() + 60.0
                while keeper.dropped.bytes < 3 * limit:
                    assert time.monotonic() < end, keeper.dropped
                    reached = keeper.dropped.refreshes > 0
                    state = page.state()
                    assert recorder.writer_error is None and not recorder.disabled
                    if state["map"] is not None:
                        assert state["live"] is True
                        assert state["map"]["points"] == CELLS
                        assert live_picture()
                        shown.append(state["map"]["revision"])
                        beyond += shown[-1:] if reached else []
                    time.sleep(0.05)
            finally:
                scene.watcher.stop()
            assert shown == sorted(shown) and len(set(beyond)) > 5
            assert keeper.dropped.points == 0  # pictures alone went
            assert scene.map.last_error is None
            assert scene.map.refreshes == scene.watcher.ticks
            recorder.events.join()
            rows = events(run)
            stored = by_revision(rows)
            assert len(stored) == scene.map.refreshes  # every tick, by its rows
            # The operator moves the time back into the dropped stretch: the
            # state is there, the map with its points; the picture is not,
            # and the viewer says so for that request alone.
            assert not there(run, stored[2][0])
            at = seconds(rows, stored[2][-2])
            state = page.state(at=at)
            assert state["map"]["revision"] == 2 and state["map"]["points"] == CELLS
            code, kind, body = page.get(f"/api/image?run=latest&at={at}")
            assert code == 400 and kind.startswith("text/html")
            assert b"No such file or directory" in body
            # "Follow live" after it, and the page loaded anew.
            state = page.state()
            assert state["map"]["revision"] == scene.map.refreshes and live_picture()
            code, kind, body = page.get("/")
            assert (code, kind) == (200, "text/html; charset=utf-8")
            assert b'"dropped_at_limit"' in body
    left = status(run)
    assert left["writer_error"] is None and left["dropped_events"] == 0
    assert left["complete"] is False and RATE_BLOCK not in left
    assert left[DROPPED_BLOCK]["bytes"] >= 3 * limit
    assert folder_bytes(run / "debug") <= limit
    # Where the keeper ran. All its work: on the recorder's writer thread.
    # On the watcher's thread: the two marks of a tick, and nothing per
    # event, for the keeper stands in no event's way.
    threads = {name: set() for name in (*WRITER_SIDE, *EVENT_SIDE)}
    for name, thread in keeper_threads:
        threads[name].add(thread)
    for name in WRITER_SIDE:
        assert threads[name] == {WRITER}, name
    assert threads["begin"] == threads["end"] == {"world-watcher"}
    # The keeper has no other method but these: what it is made with, what
    # _note calls for a refusal or a segment, and what tells what was dropped.
    others = {"__init__", "_keep_named", "kept_from", "_zero", "_trouble"}
    methods = {
        name for name, value in vars(debug_recording.Keeper).items() if callable(value)
    }
    assert methods - others == {*WRITER_SIDE, *EVENT_SIDE}
    begun = sum(1 for name, _ in keeper_threads if name == "begin")
    ended = sum(1 for name, _ in keeper_threads if name == "end")
    assert begun == ended >= scene.watcher.ticks


@apai
@experiment
def test_when_no_picture_is_left_to_drop_the_oldest_map_points_go_too(
    apai_scratch: Path, viewer: Any
) -> None:
    import numpy as np

    runs = apai_scratch / "runs"
    run = listed(runs / "20261117T120000Z-1", apai_scratch)
    limit = 16 * MB
    clock = Clock()
    # Small pictures that compress to nothing, and a grid that does not: 2 MB
    # for the bridge's answer of every refresh.
    with recorded(run, 0.0, clock, max_bytes=limit) as (scene, recorder, _):
        noise = np.random.default_rng(2)
        scene.bridge.grid = noise.random(scene.bridge.grid.shape, dtype=np.float32)
        keeper = scene.keeper
        index = viewer.RunIndex(run)
        with served(viewer, runs) as page:
            for number in range(1, 41):
                scene.tick()
                clock.now += 0.5
                recorder.events.join()
                assert recorder.writer_error is None and not recorder.disabled
                assert index.snapshot()["map"]["revision"] == number
                assert is_picture(index.image(), PICTURE)
            assert folder_bytes(run / "debug") <= limit
            assert keeper.dropped.points > 10 and keeper.dropped.refreshes > 10
            rows = events(run)
            stored = by_revision(rows)
            # A time whose map points went: the viewer has no state for that
            # request, an error it names, and follows live after it.
            at = seconds(rows, stored[2][-2])
            code, kind, body = page.get(f"/api/state?run=latest&at={at}")
            assert code == 400 and kind.startswith("text/html")
            assert b"No such file or directory" in body
            assert page.state()["map"]["revision"] == 40
            # The first map of the session is there, points and picture.
            first = seconds(rows, stored[1][-2])
            assert page.state(at=first)["map"]["points"] == CELLS
            assert is_picture(index.image(first), PICTURE)
    left = status(run)
    assert left["writer_error"] is None and left["complete"] is False
    block = left[DROPPED_BLOCK]
    assert block["points_dropped"] == keeper.dropped.points
    # The status names from when every map has its points: there the viewer
    # has a state, and just before it has none. The pictures begin later.
    index = viewer.RunIndex(run)
    points_from = block["first_time_points_kept"]
    assert 0 < points_from <= block["first_time_kept"]
    assert index.snapshot(points_from)["map"]["points"] == CELLS
    with pytest.raises(FileNotFoundError, match="No such file"):
        index.snapshot(points_from - 0.002)


def one_tick(recorder: Any, keeper: Any, number: int, picture: Any) -> None:
    """What the recorder's hooks record of one tick, for a recorder without APAI.

    The map's refresh, which commits revision `number`, and the detector's
    mask of the same camera frame.
    """
    before = keeper.begin() if keeper is not None else None
    try:
        frame = recorder.emit("map.frame", map_id=7, frame={"rgb": picture})
        recorder.local.refresh_id = frame
        asked = recorder.emit("bridge.request", payload={"action": "query"})
        cells = {"points": picture[:4, :64].astype("float32")}
        recorder.emit("bridge.response", request_id=asked, result=cells)
        recorder.emit("map.commit", map_id=7, refreshes=number, query_request_id=asked)
        recorder.emit("map.refresh_end", map_id=7, refreshes=number)
        recorder.local.refresh_id = None
        recorder.emit("object.mask", method="box", frame={"rgb": picture})
    finally:
        if keeper is not None:
            keeper.end(before)


def noise_picture() -> Any:
    """100 kB that do not compress."""
    import numpy as np

    return np.random.default_rng(0).integers(0, 255, (100, 1000), dtype=np.uint8)


@apai
@experiment
def test_a_burst_of_map_pictures_does_not_stop_the_recorder_for_its_size(
    tmp_path: Path,
) -> None:
    recording = import_experiment("thor_recording")
    limit = 2 * MB
    picture = noise_picture()
    recorder = recording.EvidenceRecorder(tmp_path, max_bytes=limit)
    waiting = 0
    try:
        with debug_recording.limited(recorder, 0.0, DROP) as limits:
            assert limits.gate is None
            # 300 ticks, 60 MB, as fast as this thread can hand them over:
            # the writer is far behind.
            for number in range(1, 301):
                one_tick(recorder, limits.keeper, number, picture)
                waiting = max(waiting, recorder.events.qsize())
            recorder.events.join()
            assert recorder.writer_error is None and not recorder.disabled
            assert folder_bytes(tmp_path / "debug") <= limit
    finally:
        recorder.close()
    assert waiting > 100
    left = json.loads((tmp_path / "debug" / "recording-status.json").read_text())
    assert left["writer_error"] is None and left["dropped_events"] == 0
    assert left["written_events"] == left["accepted_events"] == 300 * 6
    assert left["finalized"] is True and left["complete"] is False
    assert left[DROPPED_BLOCK]["bytes"] >= 25 * limit
    # Every row is there, and the arrays of the first and the newest ticks.
    rows = [
        json.loads(line)
        for line in (tmp_path / "debug" / "events.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in rows] == list(range(1, 1801))
    kept = {row["id"] for row in rows if there(tmp_path, row)}
    newest = 6 * (300 - debug_recording.KEPT_NEWEST)
    assert set(range(1, 7)) <= kept and set(range(newest + 1, 1801)) <= kept
    # Of the ticks 2 to 250 the pictures went and the map's points stayed.
    for row in rows[6 : 6 * 250]:
        points = row["kind"] == "bridge.response"
        assert there(tmp_path, row) == ("arrays" not in row or points), row["id"]


@apai
@experiment
def test_a_real_writer_error_still_stops_the_recorder_and_is_said(
    tmp_path: Path,
) -> None:
    recording = import_experiment("thor_recording")
    picture = noise_picture()
    recorder = recording.EvidenceRecorder(tmp_path, max_bytes=2 * MB)
    write = recorder._write_arrays
    files: list[Any] = []

    def full_disk(path: Any, arrays: Any) -> None:
        files.append(path)
        if len(files) > 120:
            raise OSError(errno.ENOSPC, "No space left on device")
        write(path, arrays)

    recorder._write_arrays = full_disk
    said: list[str] = []
    notice = debug_recording.FullNotice(said.append, 2 * MB / 2**30)
    try:
        with debug_recording.limited(recorder, 0.0, DROP) as limits:
            with notice.watching(recorder, limits.keeper):
                end = time.monotonic() + 30.0
                number = 0
                while len(said) < 2 and time.monotonic() < end:
                    number += 1
                    one_tick(recorder, limits.keeper, number, picture)
                    time.sleep(0.005)
                dropped = limits.keeper.dropped
    finally:
        recorder.close()
    # The size was reached first, and the oldest pictures went: said once.
    # Then the disk was full: the recorder stopped, with its reason.
    reason = "OSError: [Errno 28] No space left on device"
    assert recorder.disabled and recorder.writer_error == reason
    assert dropped.refreshes > 0 and len(said) == 2
    assert said[0].startswith(
        f"{REACHED}--debug-max-gib {2 * MB / 2**30:g} after less than 1 min: "
        "from now on the oldest map pictures are dropped; the viewer keeps following"
    )
    assert said[1] == (
        f"debug recording stopped after less than 1 min ({reason}): the viewer "
        "no longer updates; the arm and the task are not affected. With a full "
        "disk, delete old runs by hand first. End the session and start the "
        "agent again for a new recording"
    )
    # Each once more at the next task prompt, and then no more.
    again = notice.prompt_lines()
    assert len(again) == 2 and again[0].startswith(REACHED) and again[1] == said[1]
    assert notice.prompt_lines() == []
    left = json.loads((tmp_path / "debug" / "recording-status.json").read_text())
    assert left["writer_error"] == reason and left["complete"] is False
    assert left[DROPPED_BLOCK]["refreshes"] > 0


@apai
@experiment
def test_pictures_that_cannot_be_removed_let_the_recorder_stop_for_its_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recording = import_experiment("thor_recording")
    picture = noise_picture()
    recorder = recording.EvidenceRecorder(tmp_path, max_bytes=2 * MB)

    def denied(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", str(self))

    said: list[str] = []
    notice = debug_recording.FullNotice(said.append, 0.002)
    try:
        with caplog.at_level(logging.WARNING), monkeypatch.context() as patched:
            patched.setattr(Path, "unlink", denied)
            with debug_recording.limited(recorder, 0.0, DROP) as limits:
                with notice.watching(recorder, limits.keeper):
                    end = time.monotonic() + 30.0
                    number = 0
                    while not said and time.monotonic() < end:
                        number += 1
                        one_tick(recorder, limits.keeper, number, picture)
                        time.sleep(0.005)
                    dropped = limits.keeper.dropped
    finally:
        recorder.close()
    # Nothing could be dropped, so the recorder stopped as it does alone,
    # and the line is the one of a full recording.
    assert recorder.writer_error == SIZE_REASON and dropped.refreshes == 0
    assert len(said) == 1 and said[0].startswith(
        f"{FULL}less than 1 min (--debug-max-gib 0.002): the viewer no longer updates"
    )
    told = [r for r in caplog.records if "could not be dropped" in r.message]
    assert len(told) == 1
    left = json.loads((tmp_path / "debug" / "recording-status.json").read_text())
    assert left["writer_error"] == SIZE_REASON
    assert left[DROPPED_BLOCK]["refreshes"] == 0 == left[DROPPED_BLOCK]["bytes"]


@apai
@experiment
def test_with_stop_the_recorder_ends_at_its_size_as_alone_and_the_rate_copies_no_more(
    apai_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = import_experiment("thor_recording")
    copies: list[int] = []
    copied = debug_recording._copied

    def counted(value: Any, np: Any, size: list[int]) -> Any:
        copies.append(1)
        return copied(value, np, size)

    monkeypatch.setattr(debug_recording, "_copied", counted)
    run = apai_scratch / "run"
    limit = 3 * MB
    clock = Clock()
    said: list[str] = []
    notice = debug_recording.FullNotice(said.append, limit / 2**30)
    with recorded(
        run, 2.0, clock, max_bytes=limit, at_limit=STOP, picture=LARGE, noisy=True
    ) as (scene, recorder, gate):
        # Nothing of the keeper is on the recorder: its queue is its own.
        assert scene.keeper is None
        assert "get" not in vars(recorder.events)
        assert not [hook for hook in recorder.hooks if "Keeper" in hook]
        with notice.watching(recorder):
            for _ in range(200):
                scene.tick()
                clock.now += 0.25
                recorder.events.join()
                if recorder.disabled:
                    break
            # The experiment's own stop, with its reason.
            assert recorder.disabled and recorder.writer_error == SIZE_REASON
            end = time.monotonic() + 10.0
            while not said and time.monotonic() < end:
                time.sleep(0.02)
        at_stop = (scene.map.refreshes, len(copies), dict(gate.counts))
        assert at_stop[1] > 0
        # The map goes on; the rate holds and copies nothing for a recorder
        # that has stopped.
        for _ in range(10):
            scene.tick()
            clock.now += 0.25
        assert scene.map.refreshes == at_stop[0] + 10 and scene.map.last_error is None
        assert len(copies) == at_stop[1] and gate.counts == at_stop[2]
    assert said == [
        f"debug recording full after less than 1 min (--debug-max-gib "
        f"{limit / 2**30:g}): the viewer no longer updates; the arm and the task "
        "are not affected. End the session and start the agent again for a new "
        "recording, or raise --debug-max-gib"
    ]
    left = status(run)
    assert left["writer_error"] == SIZE_REASON and left["complete"] is False
    assert left["dropped_events"] > 0 and DROPPED_BLOCK not in left
    assert left[RATE_BLOCK] == {"map_hz": 2.0, **at_stop[2]}
    # Every array the recorder wrote is still there.
    rows = events(run)
    assert all(there(run, row) for row in rows) and folder_bytes(run) <= limit
    report = recording.generate_report(run).read_text()
    assert "complete recorded coverage: False" in report


@apai
@experiment
def test_what_the_keeper_adds_on_the_thread_of_a_tick_is_its_two_marks(
    tmp_path: Path,
) -> None:
    recording = import_experiment("thor_recording")
    recorder = recording.EvidenceRecorder(tmp_path, max_bytes=2**30)
    own = recorder.emit
    try:
        with debug_recording.limited(recorder, 0.0, DROP) as limits:
            keeper = limits.keeper
            # No event passes through the keeper on its way to the recorder,
            # and the keeper holds no lock: a tick never waits for the writer.
            assert recorder.emit == own and "emit" not in vars(recorder)
            assert not [
                value
                for value in vars(keeper).values()
                if isinstance(value, type(threading.Lock()))
            ]
            # From its next event on the writer takes every event through
            # the keeper: it was waiting in its queue when the keeper came.
            recorder.emit("robot.state", seq=0)
            recorder.events.join()
            # The two marks of a tick, the best of five rounds of 2000 ticks.
            best = float("inf")
            for _ in range(5):
                began = time.perf_counter()
                for _ in range(2000):
                    keeper.end(keeper.begin())
                best = min(best, (time.perf_counter() - began) / 2000)
                # A tick without an event leaves nothing behind once a later
                # event is written.
                recorder.emit("robot.state", seq=1)
                recorder.events.join()
                assert [len(spans) for spans in keeper._spans.values()] == [0]
    finally:
        recorder.close()
    # Measured on the workstation: about 1 microsecond a tick. The bound is
    # wide for a loaded machine.
    assert best < 100e-6, best


REPLAY_TOOLS = [
    ("thor_replay_target_footprint", (), "Complete recording required"),
    ("thor_replay_pickup_success", ("cpu",), "A complete recording is required"),
    (
        "thor_replay_target_map",
        (1, 1),
        "A complete recording is required for full-history replay",
    ),
    ("thor_replay_mask_margin", (6,), "A complete recording is required"),
]


def own_check(folder: Path, tool: str) -> tuple[Callable[[dict[str, Any]], bool], str]:
    """A replay tool's own check of recording-status.json, from its source.

    Returns whether its replay() refuses a run with a given status, and the
    words it refuses with. The check is the first thing replay() does with
    the status it reads.
    """
    import ast

    tree = ast.parse((folder / f"{tool}.py").read_text())
    replay = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "replay"
    )
    read = next(
        (number, node)
        for number, node in enumerate(replay.body)
        if isinstance(node, ast.Assign)
        and "debug/recording-status.json" in ast.unparse(node)
    )
    name = read[1].targets[0].id
    check = replay.body[read[0] + 1]
    assert isinstance(check, ast.If) and isinstance(check.body[0], ast.Raise)
    assert f"{name}" in ast.unparse(check.test) and "complete" in ast.unparse(
        check.test
    )
    test = compile(ast.Expression(check.test), tool, "eval")
    words = ast.literal_eval(check.body[0].exc.args[0])
    return (lambda status: bool(eval(test, {name: status}))), words


@apai
@experiment
@pytest.mark.parametrize(("tool", "arguments", "refusal"), REPLAY_TOOLS)
def test_the_experiments_replay_tools_refuse_a_run_with_map_pictures_left_out_or_dropped(
    apai_scratch: Path,
    experiment_dir: Path,
    tool: str,
    arguments: tuple[Any, ...],
    refusal: str,
) -> None:
    import importlib

    refuses, words = own_check(experiment_dir, tool)
    assert words == refusal
    # A status that says "complete" passes the tool's check; one of the
    # experiment's own, of a run that ended at its size, does not.
    assert not refuses({"complete": True}) and refuses({"complete": False})
    # The tool itself, where this interpreter has what it imports.
    import_experiment("thor_recording")  # the experiment folder on the path
    try:
        module = importlib.import_module(tool)
    except ModuleNotFoundError as missing:
        assert missing.name != tool
        module = None
    for name, map_hz, limit, ticks in (
        ("left-out", 2.0, 2**30, 4),  # by the rate
        ("dropped", 0.0, 6 * MB, 30),  # at the size
    ):
        run = listed(apai_scratch / "runs" / name, apai_scratch)
        with recorded(
            run, map_hz, Clock(), max_bytes=limit, picture=LARGE, noisy=True
        ) as (scene, _, _):
            for _ in range(ticks):
                scene.tick()
        left = status(run)
        thinned = left.get(RATE_BLOCK, {}).get("skipped", 0)
        assert thinned + left[DROPPED_BLOCK]["refreshes"] > 0, name
        # Nothing else is wrong with the run: it is finalized, and no event
        # was lost. Only the map pictures make it incomplete.
        assert left["finalized"] is True and left["writer_error"] is None
        assert left["dropped_events"] == 0 == left["snapshot_errors"]
        assert left["complete"] is False and refuses(left)
        if module is not None:
            with pytest.raises(ValueError, match=refusal):
                module.replay(run, *arguments)


# The word that the recording is full


def stopped(reason: str | None = SIZE_REASON) -> SimpleNamespace:
    """What FullNotice reads of the experiment's recorder."""
    return SimpleNamespace(
        disabled=reason is not None,
        writer_error=reason,
        start_mono_ns=time.monotonic_ns() - int(10.4 * 60e9),
    )


def test_the_full_line_is_said_once_when_the_recording_stops_and_once_at_the_next_prompt() -> (
    None
):
    said: list[str] = []
    notice = debug_recording.FullNotice(said.append, 16.0)
    recorder = stopped(None)
    line = (
        "debug recording full after 10 min (--debug-max-gib 16): the viewer no "
        "longer updates; the arm and the task are not affected. End the session "
        "and start the agent again for a new recording, or raise --debug-max-gib"
    )
    with notice.watching(recorder):
        time.sleep(0.6)
        # A recording that goes on: nothing is said, also not at a prompt.
        assert said == [] and notice.prompt_lines() == []
        recorder.writer_error, recorder.disabled = SIZE_REASON, True
        end = time.monotonic() + 10.0
        while not said and time.monotonic() < end:
            time.sleep(0.02)
        assert said == [line]
        time.sleep(0.6)
        assert said == [line]  # once, however long the run goes on
        # The next task prompt gets it once more; the prompts after it do not.
        assert notice.prompt_lines() == [line]
        assert notice.prompt_lines() == [] and notice.told == 2
    assert said == [line]
    assert not [
        t for t in threading.enumerate() if t.name == debug_recording.WATCH_THREAD
    ]


def test_a_recording_that_stopped_for_another_reason_says_that_reason() -> None:
    said: list[str] = []
    notice = debug_recording.FullNotice(said.append, 16.0)
    reason = "OSError: less than 2 GiB free; stopping evidence recording"
    recorder = stopped(reason)
    recorder.start_mono_ns = time.monotonic_ns() - int(20e9)
    with notice.watching(recorder):
        end = time.monotonic() + 10.0
        while not said and time.monotonic() < end:
            time.sleep(0.02)
    assert said == [
        f"debug recording stopped after less than 1 min ({reason}): the viewer no "
        "longer updates; the arm and the task are not affected. With a full "
        "disk, delete old runs by hand first. End the session and start the "
        "agent again for a new recording"
    ]
    assert notice.prompt_lines() == said

    # A line that cannot be said stops nothing, and no recorder is no watch.
    def mute(line: str) -> None:
        raise OSError("the terminal is gone")

    quiet = debug_recording.FullNotice(mute, 16.0)
    assert quiet.look(stopped(), time.monotonic()) is True
    with quiet.watching(None), quiet.watching(SimpleNamespace(hooks=[])):
        pass


def test_the_line_of_a_reached_size_is_said_once_and_once_more_at_the_next_prompt() -> (
    None
):
    said: list[str] = []
    notice = debug_recording.FullNotice(said.append, 16.0)
    recorder = stopped(None)  # it goes on
    # What the keeper tells: nothing dropped yet. Then: from when every map
    # picture is there, as (wall clock seconds, seconds on the viewer's line).
    kept = [(1794950921.0, 62.4)]
    keeper = SimpleNamespace(
        dropped=debug_recording.Dropped(), kept_from=lambda: kept[0]
    )

    def line(wall_s: float, on_the_line: int) -> str:
        clock = time.strftime("%H:%M:%S", time.localtime(wall_s))
        return (
            "debug recording reached --debug-max-gib 16 after 31 min: from now on "
            "the oldest map pictures are dropped; the viewer keeps following, and "
            f"times before {clock} ({on_the_line} s on its time line) show no "
            "picture. That time moves on with the run"
        )

    with notice.watching(recorder, keeper):
        time.sleep(0.6)
        assert said == [] and notice.prompt_lines() == []
        keeper.dropped = debug_recording.Dropped(
            refreshes=3, bytes=14 * MB, kept_from_ns=1, reached_s=31.4 * 60
        )
        end = time.monotonic() + 10.0
        while not said and time.monotonic() < end:
            time.sleep(0.02)
        assert said == [line(1794950921.0, 62)]
        keeper.dropped = keeper.dropped._replace(refreshes=900)
        time.sleep(0.6)
        assert len(said) == 1  # once, however much is dropped after it
        # The next task prompt gets it once more, with the time as it is by
        # then; the prompts after it do not.
        kept[0] = (1794950921.0 + 300, 362.4)
        assert notice.prompt_lines() == [line(1794951221.0, 362)]
        assert notice.prompt_lines() == [] and notice.told == 2
        # A recorder that stops after all, for a full disk: its own line.
        reason = "OSError: less than 2 GiB free; stopping evidence recording"
        recorder.writer_error, recorder.disabled = reason, True
        end = time.monotonic() + 10.0
        while len(said) < 2 and time.monotonic() < end:
            time.sleep(0.02)
    assert len(said) == 2 and said[1].startswith(
        f"debug recording stopped after 10 min ({reason}): the viewer no longer updates"
    )
    assert notice.prompt_lines() == said[1:] and notice.told == 4
    # Without the time, which a keeper always has once it dropped something.
    assert notice.reached_line(20.0, None) == (
        "debug recording reached --debug-max-gib 16 after less than 1 min: from "
        "now on the oldest map pictures are dropped; the viewer keeps following"
    )


@experiment
def test_the_experiments_recorder_at_its_limit_is_what_the_line_is_said_for(
    tmp_path: Path,
) -> None:
    import numpy as np

    recording = import_experiment("thor_recording")
    said: list[str] = []
    notice = debug_recording.FullNotice(said.append, 0.25)
    recorder = recording.EvidenceRecorder(tmp_path, max_bytes=300_000)
    try:
        with notice.watching(recorder):
            # 100 kB that do not compress: the third does not fit.
            noise = np.random.default_rng(0)
            picture = noise.integers(0, 255, (100, 1000), dtype=np.uint8)
            end = time.monotonic() + 20.0
            while not said and time.monotonic() < end:
                recorder.emit("map.frame", picture=picture)
                time.sleep(0.01)
    finally:
        recorder.close()
    assert recorder.writer_error == SIZE_REASON
    assert len(said) == 1 and said[0].startswith(
        f"{FULL}less than 1 min (--debug-max-gib 0.25): the viewer no longer updates"
    )
    left = json.loads((tmp_path / "debug" / "recording-status.json").read_text())
    assert left["writer_error"] == SIZE_REASON and left["written_events"] >= 1


@apai
@experiment
def test_a_run_whose_recording_is_full_says_so_twice_and_goes_on(stage: Stage) -> None:
    run = stage.folder / "runs" / "20261117T120000Z-1"
    # A limit the recording's first event reaches.
    options = RunOptions(record_debug=True, debug_max_gib=1e-6)
    seen: list[int] = []

    def at_the_prompt() -> None:
        """The program waits for a task line: the line was said by then."""
        end = time.monotonic() + 30.0
        while not stage.said(FULL) and time.monotonic() < end:
            time.sleep(0.05)
        seen.append(len(stage.said(FULL)))

    code = test_viewer.record(stage, run, options, QUICK, QUICK, between=at_the_prompt)
    assert code == launcher.EXIT_OK, stage.lines
    # Both tasks ran after the recording had stopped: the arm was taken and
    # given back twice.
    assert stage.order.count("agent:completed") == 2
    assert status(run)["writer_error"] == SIZE_REASON
    # Once when it stopped and once more with the next task prompt: the same
    # line, and no third time at the prompts after it.
    told = stage.said(FULL)
    assert len(told) == 2 and told[0] == told[1]
    assert told[0] == (
        "debug recording full after less than 1 min (--debug-max-gib 1e-06): the "
        "viewer no longer updates; the arm and the task are not affected. End "
        "the session and start the agent again for a new recording, or raise "
        "--debug-max-gib"
    )
    assert len(seen) == 3 and seen[0] >= 1 and seen[1:] == [2, 2]
    # The second saying came with a prompt: the program's last line before
    # it read the task.
    (start,) = stage.said("debug recording: ")
    assert stage.lines.index(start) < stage.lines.index(told[0])


# A run of the agent program


@apai
@experiment
def test_a_run_of_the_agent_program_stores_two_map_pictures_a_second_and_every_segment(
    stage: Stage, viewer: Any
) -> None:
    """The agent program on APAI's mock scene, with the rate it has by default."""
    run = stage.folder / "runs" / "20261117T120000Z-1"
    code = test_viewer.record(stage, run, test_viewer.RECORDED, test_launcher.PICK)
    # As in test_viewer.py: the pick and place moved the arm, whatever the
    # mock scene's task report says of it.
    assert code in (launcher.EXIT_OK, launcher.EXIT_FAILED), stage.lines
    (result,) = stage.results("pick_and_place")
    assert result["picked"] and len(stage.h.spy.targets) > 100
    (start,) = stage.said("debug recording: ")
    assert "at most 1 GiB: when that is reached the oldest map pictures are" in start
    assert "That size holds about 2 min at 2 map pictures a second" in start
    assert stage.said(REACHED) == [] and stage.said(FULL) == []
    rows = events(run)
    counts = kinds(rows)
    opened, closed = (row["data"] for row in rows if row["kind"] == RATE_EVENT)
    assert opened["map_hz"] == 2.0
    # The watcher's ticks: more than were stored, and none unaccounted for.
    assert closed["ticks"] == closed["stored"] + closed["skipped"]
    assert closed["skipped"] > 0 and closed["missing"] == 0
    seconds = (rows[-1]["mono_ns"] - rows[0]["mono_ns"]) / 1e9
    assert closed["stored"] <= 2 * seconds + 2
    # In the recording: those of the rate and those a segment named.
    assert counts["map.frame"] == closed["stored"] + closed["rescued"]
    assert counts["map.frame"] < closed["ticks"]
    # Every stored refresh has its frame, and every commit its query.
    stored = refreshes(rows)
    committed = {}
    for refresh in stored:
        by_kind = {row["kind"]: row for row in refresh}
        if "map.commit" in by_kind:
            asked = by_kind["map.commit"]["data"]["query_request_id"]
            assert asked in {row["id"] for row in refresh}
            committed[by_kind["map.commit"]["data"]["refreshes"]] = by_kind
        assert "map.refresh_end" in by_kind
    # Every route segment of the pick and place is there, begin and end, and
    # the map revision each of them names is in the recording before it.
    segments = [row for row in rows if row["kind"].startswith("route.segment.")]
    assert counts["route.segment.begin"] == counts["route.segment.end"] >= 2
    for row in segments:
        named = row["data"]["map"]["refreshes"]
        assert named in committed, (row["kind"], named)
        assert committed[named]["map.commit"]["id"] < row["id"]
    # The status: the rate's numbers, nothing dropped at the size, which was
    # not reached, and not "complete", for the map pictures left out.
    left = status(run)
    assert left["finalized"] is True and left["dropped_events"] == 0
    assert left[RATE_BLOCK] == closed and left["complete"] is False
    assert left[DROPPED_BLOCK]["refreshes"] == 0
    # The experiment's viewer on it.
    state = viewer.RunIndex(run).snapshot()
    assert state["map"]["points"] == CELLS and state["map"]["frame_id"]
    assert state["robot"]["source"] == launcher.FEEDBACK_SOURCE


@apai
@experiment
def test_the_agent_program_sends_the_arm_the_same_commands_with_the_keeper_as_without(
    make_stage: Callable[..., Stage], viewer: Any
) -> None:
    """The agent program on APAI's mock scene: a pick and place, then a jaw command.

    Once as the experiment records it, every map picture and a stop at the
    size, which is not reached; once with the defaults and a size the pick
    and place passes several times.
    """
    left = {}
    for name, map_hz, at_limit, gib in (
        ("alone", 0.0, STOP, 1.0),
        ("kept", debug_recording.DEBUG_MAP_HZ, DROP, SMALL_GIB),
    ):
        stage = make_stage()
        run = stage.folder / "runs" / f"20261117T120000Z-{name}"
        options = RunOptions(
            timing=True, model_trace=True, record_debug=True, debug_max_gib=gib
        )
        with (
            mock.patch.object(launcher, "_debug_map_hz", lambda *_, hz=map_hz: hz),
            mock.patch.object(launcher, "_debug_at_limit", lambda *_, at=at_limit: at),
        ):
            code = test_viewer.record(stage, run, options, test_launcher.PICK, QUICK)
        assert code in (launcher.EXIT_OK, launcher.EXIT_FAILED), stage.lines
        (result,) = stage.results("pick_and_place")
        assert result["picked"]
        rows = events(run)
        segments = [
            (row["kind"], row["data"].get("reason"))
            for row in rows
            if row["kind"].startswith("route.segment.")
        ]
        left[name] = {
            "code": code,
            "targets": [
                [float(value) for value in target] for target in stage.h.spy.targets
            ],
            "jaws": stage.jaw_commands(),
            "motion": list(stage.motion),
            "order": list(stage.order),
            "segments": segments,
            "said": (stage.said(REACHED), stage.said(FULL)),
            "status": status(run),
            "run": run,
        }
    alone, kept = left["alone"], left["kept"]
    # What the arm was sent: the same setpoints, jaw commands, moves and
    # owner requests, and the same route checks with the same answers.
    assert len(alone["targets"]) > 100 and len(alone["segments"]) >= 4
    for what in ("code", "targets", "jaws", "motion", "order", "segments"):
        assert kept[what] == alone[what], what
    # Without the keeper the size was not reached, and nothing was said.
    assert alone["said"] == ([], []) and alone["status"]["complete"] is True
    assert DROPPED_BLOCK not in alone["status"] and RATE_BLOCK not in alone["status"]
    # With it the size was reached during the pick and place: said once
    # then, and once more with the next task prompt. The recorder went on.
    reached, full = kept["said"]
    assert full == [] and len(reached) == 2
    for line in reached:
        assert line.startswith(
            f"{REACHED}--debug-max-gib {SMALL_GIB:g} after less than 1 min: from "
            "now on the oldest map pictures are dropped; the viewer keeps "
            "following, and times before "
        ), line
        assert line.endswith(
            " s on its time line) show no picture. That time moves on with the run"
        )
    told = kept["status"]
    assert told["writer_error"] is None and told["dropped_events"] == 0
    assert told["finalized"] is True and told["complete"] is False
    assert told[DROPPED_BLOCK]["refreshes"] > 0 and told[RATE_BLOCK]["skipped"] > 0
    assert folder_bytes(kept["run"] / "debug") <= SMALL_GIB * 2**30
    # The experiment's viewer on it: the last map, with its picture.
    index = viewer.RunIndex(kept["run"])
    state = index.snapshot()
    assert state["map"]["points"] == CELLS and index.image()[:2] == b"\xff\xd8"
    assert state["recording"][DROPPED_BLOCK] == told[DROPPED_BLOCK]


# What the program says at the start, and the README


def roomy_disk(monkeypatch: pytest.MonkeyPatch, free_gib: float = 1000.0) -> None:
    """The disk the launcher asks about has `free_gib` GiB free."""
    free = SimpleNamespace(free=int(free_gib * 2**30))
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda path: free)


def test_the_start_line_warns_when_the_disk_cannot_hold_the_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = RunOptions(record_debug=True)
    warning = (
        "ends there instead. WARNING: only 17 GiB are free on this disk: the "
        "recording stops for the disk before it reaches that size, and the "
        "viewer then no longer updates. Delete old run folders first; "
        "./run_viewer.sh shows it"
    )
    # Under the size plus the 2 GiB the experiment's recorder stops at.
    roomy_disk(monkeypatch, 17.4)
    assert warning in launcher._option_lines(options, Path("run"))[-1]
    roomy_disk(monkeypatch, 18.0)
    assert "WARNING" not in launcher._option_lines(options, Path("run"))[-1]
    # A disk that cannot be asked gives no sentence.
    monkeypatch.setattr(launcher.shutil, "disk_usage", mock.Mock(side_effect=OSError))
    assert "WARNING" not in launcher._option_lines(options, Path("run"))[-1]


def test_the_start_line_names_how_long_the_recording_is_expected_to_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roomy_disk(monkeypatch)
    options = RunOptions(timing=True, record_debug=True)
    assert options.debug_max_gib == 16.0
    for map_hz, lasts in (
        (None, "about 30 min at 2 map pictures a second"),
        (2.0, "about 30 min at 2 map pictures a second"),
        (1.0, "about 61 min at 1 map picture a second"),
        (0.5, "about 122 min at 0.5 map pictures a second"),
        (0.0, "about 9 min with every map picture stored, about 7 a second"),
        (30.0, "about 9 min with every map picture stored, about 7 a second"),
    ):
        given = () if map_hz is None else (map_hz,)
        # By default the oldest map pictures go at the size.
        lines = launcher._option_lines(options, Path("run"), *given)
        (line,) = (line for line in lines if line.startswith("debug recording: "))
        assert line.startswith(
            f"debug recording: {Path('run') / 'debug'}, at most 16 GiB: when "
            "that is reached the oldest map pictures are dropped and the viewer "
            f"keeps following. That size holds {lasts}, which --debug-map-hz "
            "sets; with --debug-at-limit stop the recording ends there instead; "
            "./run_viewer.sh shows it"
        ), line
        # With "stop" the recording ends there, as the experiment's does.
        rate = debug_recording.map_hz_of(map_hz)
        line = launcher._option_lines(options, Path("run"), rate, STOP)[-1]
        assert line.startswith(
            f"debug recording: {Path('run') / 'debug'}, at most 16 GiB, "
            f"expected to last {lasts}, which --debug-map-hz sets; then the "
            "recording stops and the viewer no longer updates, as "
            "--debug-at-limit stop asks; ./run_viewer.sh shows it"
        ), line
    larger = RunOptions(timing=True, record_debug=True, debug_max_gib=64.0)
    (line,) = launcher._option_lines(larger, Path("run"))[-1:]
    assert "at most 64 GiB: when that is reached" in line
    assert "That size holds about 122 min at 2 map pictures a second" in line
    assert debug_recording.expected_text(0.001, 2.0).startswith("under 1 min at 2 ")
    # The numbers are the rig's run of 2026-11-17: 17.17 GB for 3671 map
    # pictures in 554 s, and 16 GiB that lasted nine and a half minutes.
    assert debug_recording.MAP_PICTURE_BYTES == pytest.approx(17.17e9 / 3671, rel=0.01)
    assert debug_recording.WATCHER_HZ == pytest.approx(3671 / 554, rel=0.01)
    assert debug_recording.expected_minutes(16.0, 0.0) == pytest.approx(9.5, abs=0.5)


def one_line(text: str) -> str:
    return " ".join(text.split())


def test_the_readme_tells_the_rate_the_duration_and_the_line_of_a_full_recording() -> (
    None
):
    readme = README.read_text(encoding="utf-8")
    whole = one_line(readme)
    # The option, in the usage and where the recording's size is told.
    program = readme[readme.index("\n### The agent program\n") :]
    usage = program[program.index("```text") : program.index("\n```\n")]
    assert "[--debug-map-hz HZ]" in usage
    default = debug_recording.DEBUG_MAP_HZ
    assert f"`--debug-map-hz HZ` (default {default:g})" in whole
    assert "`--debug-map-hz 0`" in whole
    # What the recording does at its size: the option and its two values.
    assert debug_recording.AT_LIMITS == (DROP, STOP)
    assert f"[--debug-at-limit {DROP}|{STOP}]" in usage
    assert f"`--debug-at-limit {DROP}|{STOP}` (default `{DROP}`)" in whole
    assert whole.count(f"`--debug-at-limit {STOP}`") >= 4
    # How much the default size holds, as the program computes and says it.
    for map_hz in (default, 0.0):
        lasts = debug_recording.expected_text(agent_config.DEBUG_MAX_GIB, map_hz)
        assert whole.count(lasts.split(" with ")[0].split(" at ")[0]) >= 2, lasts
    assert f"{debug_recording.MAP_PICTURE_BYTES / 1e6:g} MB" in whole
    # What is stored whatever the rate, by the names of the events.
    for kind in (
        debug_recording.REFUSAL,
        *debug_recording.SEGMENTS,
        "verification.measurement",
        "robot.feedback",
        launcher.STATE_EVENT,
        RATE_EVENT,
        "recorded_late",
    ):
        assert f"`{kind}`" in whole, kind
    # The line of a full recording, as FullNotice says it.
    line = debug_recording.FullNotice(print, agent_config.DEBUG_MAX_GIB).line(
        SIZE_REASON, 600.0
    )
    assert line.startswith("debug recording full after 10 min (--debug-max-gib 16): ")
    quoted = line.replace("10 min", "N min")
    assert whole.count(f"`{quoted}`") >= 1
    troubleshooting = one_line(
        readme[readme.index("\n### Troubleshooting: what a program printed") :]
    )
    assert "`debug recording full after N min" in troubleshooting
    # The line of a reached size, as FullNotice says it for a time and a place
    # on the viewer's time line.
    notice = debug_recording.FullNotice(print, agent_config.DEBUG_MAX_GIB)
    wall = 1794950921.0
    line = notice.reached_line(600.0, (wall, 77.0))
    clock = time.strftime("%H:%M:%S", time.localtime(wall))
    assert line.count(clock) == 1 and line.count("(77 s on") == 1
    quoted = line.replace("10 min", "N min").replace(clock, "HH:MM:SS")
    quoted = quoted.replace("(77 s on", "(S s on")
    assert quoted.startswith("debug recording reached --debug-max-gib 16 after N min: ")
    assert whole.count(f"`{quoted}`") == 1
    head = quoted[: quoted.index("; the viewer keeps following") + 28]
    assert whole.count(f"`{head}`") == 1  # the operator's steps, after a run
    assert f"`{quoted[: quoted.index(' pictures are dropped;') + 22]} ...`" in (
        troubleshooting
    )
    # What stays, what the status says, and what "complete" means for the
    # experiment's replay tools, each by its name.
    for said in (
        "**Never dropped**",
        f"`{RATE_BLOCK}`",
        f"`{DROPPED_BLOCK}`",
        "`first_time_kept`",
        "`points_dropped`",
        "`first_time_points_kept`",
        "`named_after_drop`",
        f"`{debug_recording.LATE_STALL}`",
        f"`{debug_recording.LATE_END}`",
        "`debug_at_limit`",
        "**What `complete` means.** `complete` in that file is `false`",
        f"`--debug-map-hz 0 --debug-at-limit {STOP}`",
        "by hand",
    ):
        assert said in whole, said
    for tool, _, _ in REPLAY_TOOLS:
        assert f"`{tool}.py`" in whole, tool
    # The share of the size the map's points of dropped pictures may take.
    assert debug_recording.POINTS_SHARE == 4 and "a quarter of the limit" in whole
    assert debug_recording.KEPT_NEWEST == 4 and "the four newest map pictures" in whole
