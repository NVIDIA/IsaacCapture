# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The viewer's live view on a real robot link service, and its source rules.

The live view is the viewer's page with its live picture showing. Here the
viewer's own parts (live_view_bench.py) run on threads of the test, attached
to a robot link service of the test. Most tests script the service's robot
side; the one marked "link" puts the robot app on the simulated arm behind
it. A stand-in source gives the camera frames, and an HTTP client on 127.0.0.1
plays the browser: a handful of requests per test on one connection.

The last tests read the viewer's files: it can send VIEW_STATUS and nothing
else, and it needs no GPU library.
"""

from __future__ import annotations

import ast
import dataclasses
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import StereoPair
from isaaccapture_examples.agentic_remote.protocol.messages import ViewStatus
from link_bench import Bench, decode, robot_state, until, welcome
from live_view_bench import BLUE, RED, Browser, LiveView, StandInSource

# The live picture is packed with OpenCV; without it this file is skipped.
cv2 = pytest.importorskip("cv2")

from isaaccapture_examples.agentic_remote.viewer_live import (  # noqa: E402
    live as live_module,
)
from isaaccapture_examples.agentic_remote.viewer_live.view_model import (  # noqa: E402
    AGE_WHEN_NOT_LIVE_S,
    ViewModel,
)

link = pytest.mark.link

WAIT_S = 10.0  # real seconds a test gives another thread before it fails
VIEWER = Path(live_module.__file__).resolve().parent
NOT_LIVE = ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)
# The model's quiet time: how long after the viewer's first ROBOT_STATE, and
# after a step of the Thor's wall clock, a picture must arrive to count.
CLOCK_STEP_QUIET_S = 3.0


@pytest.fixture
def folder(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("vl")  # a short path, mode 0700


@pytest.fixture
def bench(folder) -> Iterator[Bench]:
    made = Bench(folder)
    yield made
    made.close()


@pytest.fixture
def views() -> Iterator[list[LiveView]]:
    """The live views a test started; each is ended when the test is over."""
    made: list[LiveView] = []
    yield made
    for view in made:
        view.close()


# The whole path


@link
def test_view_status_reaches_the_robot_app_and_shows_in_robot_state(
    make_realtime_robot_app, folder, views
):
    from link_stack import Stack

    stack = Stack(make_realtime_robot_app(connected=False), folder)
    try:
        stack.start_service()
        stack.dial()
        watcher = stack.client("tool")
        stack.connected(watcher)
        assert watcher.ws.state.view_age_s is None

        source = StandInSource()
        # An acknowledgement counts for 0.6 s in this test.
        view = LiveView(stack.path, source, period_s=0.05, ack_timeout_s=0.6)
        views.append(view)
        browser = Browser(view.port)
        seq, jpeg = browser.first_frame()
        # The picture is the left eye; a camera frame narrower than the
        # viewer's greatest width keeps its size.
        picture = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert picture.shape == (96, 128, 3)
        assert np.abs(picture[48, 32].astype(int) - RED).max() < 60
        assert np.abs(picture[48, 96].astype(int) - BLUE).max() < 60

        # A picture nobody acknowledged: VIEW_STATUS arrives, and is not live.
        until(lambda: browser.live()["view"]["live"] is not None or time.sleep(0.1))
        assert browser.live()["view"] == {
            "attached": True,
            "why": "",
            "live": False,
            "age_s": None,
        }
        assert watcher.ws.state.view_age_s is None

        # The page acknowledges what it drew. No picture counts that arrived
        # in the quiet time after the viewer's first ROBOT_STATE; after it the
        # Thor reports the view's age.
        marked = view.model._clock_mark_at
        for _ in range(15):
            seq, _ = browser.frame()
            browser.shown(seq)
            if watcher.ws.state.view_age_s is not None:
                break
            time.sleep(0.4)
        state = watcher.ws.wait_state(lambda s: s.view_age_s is not None, WAIT_S)
        assert time.monotonic() - marked >= CLOCK_STEP_QUIET_S
        assert 0.0 <= state.view_age_s < 1.0
        told = browser.live()
        assert told["available"] is True and told["view"]["live"] is True
        assert 0.0 <= told["view"]["age_s"] < 1.0
        # The viewer has the robot's state too: what the robot app measures on
        # the simulated arm, which holds at rest and was sent no target.
        robot = browser.robot()
        assert [one["joint"] for one in robot["joints"]] == list(
            range(len(state.joints))
        )
        assert len(robot["joints"]) == 6
        assert all(one["valid"] for one in robot["joints"])
        assert [one["q"] for one in robot["joints"]] == pytest.approx(
            [sample.value for sample in state.joints], abs=1e-6
        )
        assert robot["target_q"] == state.target_q
        assert robot["gripper"]["opening"] == state.gripper.opening
        assert 0.0 <= robot["reading_age_s"] < 1.0 and robot["owner"] == "nobody"

        # The page stops acknowledging: past the time an acknowledgement
        # counts for, the age is unknown.
        watcher.ws.wait_state(lambda s: s.view_age_s is None, WAIT_S)
        assert browser.live()["view"]["live"] is False

        # Live again, then the tab is hidden: it says so with its next picture.
        seq, _ = browser.frame()
        browser.shown(seq)
        watcher.ws.wait_state(lambda s: s.view_age_s is not None, WAIT_S)
        seq, _ = browser.frame()
        browser.shown(seq, visible=False)
        model = view.model
        assert model._ack[:2] == (seq, False)
        now = time.monotonic()
        assert model.view_status(now, model.thor_wall_now(now)) == NOT_LIVE
        watcher.ws.wait_state(lambda s: s.view_age_s is None, WAIT_S)

        # Live once more, then the viewer ends: its last word is "not live".
        seq, _ = browser.frame()
        browser.shown(seq)
        watcher.ws.wait_state(lambda s: s.view_age_s is not None, WAIT_S)
        browser.close()
        views.remove(view)
        view.close()
        assert source.closes >= 1
        watcher.ws.wait_state(lambda s: s.view_age_s is None, 0.4)
        # Nothing but VIEW_STATUS came from the viewer: the app saw no command.
        assert stack.h.app.session_id == watcher.ws.state.session_id
        assert watcher.ws.state.stop_level == "none"
        assert watcher.ws.state.stop_number == state.stop_number
    finally:
        stack.close()


class ThorClock:
    """The Thor's wall clock, as the stand-ins of a test read it.

    This machine's wall clock, and from step() on that many seconds off it.
    """

    def __init__(self) -> None:
        self._stepped_at: float | None = None
        self._step_s = 0.0

    def step(self, by_s: float) -> None:
        self._stepped_at, self._step_s = time.time(), by_s

    def wall(self, at: float | None = None) -> float:
        """What the Thor's wall clock said at this machine's wall time `at`."""
        at = time.time() if at is None else at
        stepped = self._stepped_at is not None and at >= self._stepped_at
        return at + (self._step_s if stepped else 0.0)


class ThorCamera(StandInSource):
    """StandInSource whose pairs were exposed transit_s before they arrive.

    The exposures are stamped on the ThorClock. While `stalled` no pair comes.
    exposed holds, for each pair, its exposure on the Thor's wall clock and on
    this machine's monotonic clock.
    """

    def __init__(self, clock: ThorClock, transit_s: float = 0.2) -> None:
        super().__init__()
        self.clock, self.transit_s = clock, transit_s
        self.stalled = False
        self.exposed: list[tuple[float, float]] = []

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        if self.stalled:
            time.sleep(min(timeout_s, 0.05))
            return None
        pair = super().next_pair(timeout_s)
        exposed_ns = int(self.clock.wall(time.time() - self.transit_s) * 1e9)
        self.exposed.append((exposed_ns / 1e9, time.monotonic() - self.transit_s))
        return dataclasses.replace(
            pair,
            sof_wall_ns=(exposed_ns, exposed_ns + 1000),
            receiver_rx_mono_s=time.monotonic(),
        )

    def exposed_mono(self, exposure_wall_s: float) -> float:
        """When the pair of that exposure was exposed, on the monotonic clock."""
        return min(self.exposed, key=lambda one: abs(one[0] - exposure_wall_s))[1]


class SteppedThor:
    """A robot side on the ThorClock, and a page that acknowledges every picture.

    The robot side sends a ROBOT_STATE every 50 ms and keeps each VIEW_STATUS
    the service forwards: (when, the status, the true age then of the picture
    the page acknowledged last). The page does its work on the viewer's model,
    and asks for pictures as a page does, so that the camera is read at the
    asked rate.
    """

    def __init__(self, bench: Bench, clock: ThorClock, camera: ThorCamera) -> None:
        self.clock, self.camera = clock, camera
        self.thor = bench.robot(None)
        self.thor.send(welcome())
        self.statuses: list[tuple[float, ViewStatus, float | None]] = []
        self.view: LiveView | None = None
        self._acknowledged: float | None = None  # that picture's exposure
        self._lock = threading.Lock()
        self._off = threading.Event()
        self._threads = [
            threading.Thread(target=self._robot, daemon=True),
            threading.Thread(target=self._page, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _robot(self) -> None:
        seq = 0
        while not self._off.wait(0.05):
            seq += 1
            state = dataclasses.replace(
                robot_state(seq=seq),
                clock_wall_s=self.clock.wall(),
                clock_mono_s=time.monotonic(),
            )
            self.thor.send(state)
            while (got := self.thor.carrier.receive(0)) is not None:
                message = decode(*got)
                if isinstance(message, ViewStatus):
                    self._note(message)

    def _note(self, status: ViewStatus) -> None:
        now = time.monotonic()
        with self._lock:
            exposure = self._acknowledged
            age = None
            if exposure is not None:
                age = now - self.camera.exposed_mono(exposure)
            self.statuses.append((now, status, age))

    def _page(self) -> None:
        token = None  # the view token of the one page this stand-in loads
        while not self._off.wait(0.25):
            view = self.view
            if view is None:
                continue
            view.picture.note_fetch()
            model = view.model
            picture = model.picture()
            if picture is None:
                continue
            now = time.monotonic()
            token = token or model.new_page(now)
            # As the page: it is served the picture and acknowledges it.
            model.note_served(picture[0], token, now)
            if model.acknowledge(picture[0], True, now, token):
                with self._lock:
                    self._acknowledged = model._times[picture[0]][0]

    def since(self, when: float) -> list[tuple[float, ViewStatus, float | None]]:
        with self._lock:
            return [row for row in self.statuses if row[0] >= when]

    def wait_live(self, since: float) -> tuple[float, ViewStatus, float | None]:
        """The first live VIEW_STATUS that arrived after `since`."""
        found: list[Any] = []
        until(
            lambda: found.extend(r for r in self.since(since) if r[1].live) or found,
            "a live VIEW_STATUS",
        )
        return found[0]

    def close(self) -> None:
        self._off.set()
        for thread in self._threads:
            thread.join(WAIT_S)


def test_on_real_clocks_no_picture_from_before_a_clock_step_is_reported_live(
    bench, views
):
    clock = ThorClock()
    camera = ThorCamera(clock, transit_s=0.4)
    thor = SteppedThor(bench, clock, camera)
    try:
        began = time.monotonic()
        # The viewer's own values: a VIEW_STATUS every 0.2 s, and 1 s for an
        # acknowledgement and for a ROBOT_STATE. Five pictures a second.
        view = LiveView(bench.path, camera, fps=5.0, period_s=0.2)
        views.append(view)
        thor.view = view
        # Live only once a picture came the quiet time after the first state.
        at, first, age = thor.wait_live(began)
        assert at - began >= CLOCK_STEP_QUIET_S
        # The age of the picture the page acknowledged, within the page's and
        # the status's periods.
        assert age - 0.3 < first.age_s < age + 0.5
        # The Thor's wall clock steps back 0.5 s, and the camera stalls: the
        # page goes on acknowledging a picture that is dated on the old clock.
        clock.step(-0.5)
        camera.stalled = True
        stepped = time.monotonic()
        time.sleep(1.8)  # past the time an acknowledgement counts for
        after = thor.since(stepped + 0.3)
        assert len(after) >= 5 and not [row for row in after if row[1].live]
        # With the camera running again the view is live once a picture came
        # the quiet time after the step, with that picture's own age.
        camera.stalled = False
        at, again, age = thor.wait_live(stepped + 0.3)
        assert at - stepped >= CLOCK_STEP_QUIET_S
        assert age - 0.3 < again.age_s < age + 0.5
        # No VIEW_STATUS of the whole run called a picture younger than it was.
        for _, status, age in thor.since(began):
            assert not status.live or status.age_s > age - 0.3
    finally:
        thor.close()


def test_the_viewer_sends_view_status_and_nothing_else(bench, views):
    thor = bench.robot()
    view = LiveView(bench.path, period_s=0.05)
    views.append(view)
    began = time.monotonic()
    received = [thor.next_message() for _ in range(4)]
    # One every period (0.05 s here), also while not live.
    assert time.monotonic() - began > 0.1
    assert received == [NOT_LIVE] * 4
    views.remove(view)
    view.close()
    assert {msg_type for msg_type, _ in thor.drain()} <= {ViewStatus.TYPE}
    # No camera: one line that there is no live picture; one that it attached.
    said = "\n".join(view.lines)
    assert "no live picture: this test has no camera" in said
    assert "live view: attached to the robot link service" in said


# Source rules of the viewer's files

VIEWER_FILES = (
    "__init__.py",
    "__main__.py",
    "live.py",
    "picture.py",
    "robot.py",
    "settings.py",
    "view_link.py",
    "view_model.py",
)
# The files that reach the robot link service: the launcher hands the link
# its way to attach, and the link sends.
ATTACHING = ("__main__.py", "view_link.py")
COMMANDS = {
    "Stop",
    "Move",
    "Gripper",
    "Permit",
    "CheckerState",
    "OwnerRequest",
    "Reset",
    "Release",
    "Open",
    "Shutdown",
    "WorkstationClient",
    "send_reliable",
    "send_latest",
}


def names_in(tree: ast.AST) -> set[str]:
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    return names


def test_the_viewer_can_send_only_view_status():
    assert sorted(path.name for path in VIEWER.glob("*.py")) == sorted(VIEWER_FILES)
    sends: list[tuple[str, str]] = []
    for file in VIEWER_FILES:
        tree = ast.parse((VIEWER / file).read_text(encoding="utf-8"))
        assert not names_in(tree) & COMMANDS, file
        sends += [
            (file, ast.unparse(node.args[1]))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "send"
        ]
    # Two sends in the whole viewer: the periodic VIEW_STATUS and the last one.
    assert sorted(sends) == [("view_link.py", "NOT_LIVE"), ("view_link.py", "status")]
    link_source = (VIEWER / "view_link.py").read_text(encoding="utf-8")
    assert "status = self._model.status_to_send(self._monotonic())" in link_source
    assert "NOT_LIVE = ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)" in link_source
    # The model, the live picture, the live arm and the rest cannot send at all.
    for file in VIEWER_FILES:
        if file in ATTACHING:
            continue
        tree = ast.parse((VIEWER / file).read_text(encoding="utf-8"))
        modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not [m for m in modules if "carrier" in m or "link_service" in m], file
    # What the link sends is the model's word.
    assert callable(ViewModel.status_to_send)


def test_the_viewer_needs_no_gpu_library():
    for file in VIEWER_FILES:
        tree = ast.parse((VIEWER / file).read_text(encoding="utf-8"))
        imported = {
            (alias.name if isinstance(node, ast.Import) else node.module or "")
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not [m for m in imported if m.split(".")[0] in ("torch", "cupy", "warp")]
