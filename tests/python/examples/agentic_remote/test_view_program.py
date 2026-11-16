# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""view/__main__.py: the live view as a program.

The view's main() runs on a thread of the test, on a real robot link service.
Most tests script the service's robot side; the one marked "link" puts the
robot app on the simulated arm behind it. A stand-in source gives the camera
frames, and an HTTP client on 127.0.0.1 plays the browser: a handful of
requests per test on one connection.
"""

from __future__ import annotations

import ast
import http.client
import json
import os
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from isaaccapture_examples.agentic_remote.agent.camera import frame_source
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import (
    BridgeSource,
    CameraError,
    StereoPair,
    bgr_to_nv12,
)
from isaaccapture_examples.agentic_remote.protocol.messages import ViewStatus
from isaaccapture_examples.agentic_remote.view import __main__ as view_main
from isaaccapture_examples.agentic_remote.view.__main__ import (
    PictureLoop,
    ViewProgram,
    ViewSettings,
    encode_picture,
    load_view_settings,
)
from isaaccapture_examples.agentic_remote.view.model import (
    AGE_WHEN_NOT_LIVE_S,
    NO_PICTURE,
    RECORDED,
    ViewModel,
)
from link_bench import Bench, until

# The view packs its pictures with OpenCV; without it this file is skipped.
cv2 = pytest.importorskip("cv2")
link = pytest.mark.link

WAIT_S = 10.0  # real seconds a test gives another thread before it fails
EXAMPLE = Path(view_main.__file__).resolve().parents[4]
VIEW = Path(view_main.__file__).resolve().parent
NOT_LIVE = ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)
SIZE = (128, 96)  # width and height of the stand-in camera frames
RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)  # as BGR


def eye(left_half: tuple[int, int, int], right_half: tuple[int, int, int]) -> Any:
    """A camera frame of two plain halves, as BGR."""
    bgr = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    bgr[:, : SIZE[0] // 2] = left_half
    bgr[:, SIZE[0] // 2 :] = right_half
    return bgr


class StandInSource:
    """A camera source of small stereo pairs exposed just now on this machine.

    The left eye is red and blue, the right eye green. With `fail` open()
    raises a CameraError of that text; `ends_after` pairs the source fails.
    """

    def __init__(self, fail: str = "", ends_after: int | None = None) -> None:
        self.fail = fail
        self.ends_after = ends_after
        self.opens = 0
        self.closes = 0
        self._open = False
        self._seq = 0
        self._left = bgr_to_nv12(eye(RED, BLUE))
        self._right = bgr_to_nv12(eye(GREEN, GREEN))

    def open(self) -> None:
        self.opens += 1
        if self.fail:
            raise CameraError(self.fail)
        self._open = True

    def close(self) -> None:
        self.closes += 1
        self._open = False

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        if not self._open:
            raise CameraError("the stand-in source is closed")
        if self.ends_after is not None and self._seq >= self.ends_after:
            self.fail = "the camera receiver went away"
            raise CameraError(self.fail)
        time.sleep(0.02)
        self._seq += 1
        exposed_ns = int((time.time() - 0.03) * 1e9)
        return StereoPair(
            session_id=1,
            seq=self._seq,
            left=self._left,
            right=self._right,
            sof_wall_ns=(exposed_ns, exposed_ns + 1000),
            device_index=(5, 7),
            rotation=0,
            receiver_rx_mono_s=time.monotonic() - 0.01,
            check={"source": "test"},
        )


def write_config(folder: Path, socket_path: str, **blocks: dict[str, Any]) -> Path:
    """A view.yaml of the test: a port the system picks, quick status."""
    settings: dict[str, Any] = {
        "socket_path": socket_path,
        "state_dir": str(folder / "state"),
        "http": {"address": "127.0.0.1", "port": 0},
        "picture": {"width_px": 64, "fps": 15.0, "jpeg_quality": 90},
        "status": {"period_s": 0.05, "ack_timeout_s": 0.6, "state_max_age_s": 10.0},
    }
    for name, block in blocks.items():
        settings.setdefault(name, {}).update(block)
    file = folder / f"view_{len(list(folder.iterdir()))}.yaml"
    file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return file


class Running:
    """The view's main() on a thread, until the test ends it."""

    def __init__(self, config: Path, *args: str, source: Any = None) -> None:
        self.said: list[str] = []
        self.program: ViewProgram | None = None
        self.code: int | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        argv = ["--config", str(config), *args]
        self._thread = threading.Thread(target=self._run, args=(argv, source))
        self._thread.start()

    def _run(self, argv: list[str], source: Any) -> None:
        try:
            self.code = view_main.main(
                argv, source=source, wait=self._wait, say=self.said.append
            )
        finally:
            self._ready.set()

    def _wait(self, program: ViewProgram) -> str:
        self.program = program
        self._ready.set()
        while program.carrier.alive and not self._stop.wait(0.02):
            pass
        return "the test ends it"

    def ready(self) -> ViewProgram:
        assert self._ready.wait(WAIT_S), "the view did not start"
        assert self.program is not None, (self.code, self.said)
        return self.program

    def end(self) -> int | None:
        """End the view; its exit code."""
        self._stop.set()
        self._thread.join(WAIT_S + 10.0)
        assert not self._thread.is_alive(), "the view did not end"
        return self.code


class Browser:
    """What the page does, over one HTTP connection."""

    def __init__(self, port: int) -> None:
        self.http = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT_S)

    def frame(self) -> tuple[int | None, bytes]:
        self.http.request("GET", "/frame.jpg")
        reply = self.http.getresponse()
        seq = reply.getheader("X-Picture-Seq")
        return (None if seq is None else int(seq)), reply.read()

    def first_frame(self) -> tuple[int, bytes]:
        """The first camera picture; asks at most 20 times, 0.1 s apart."""
        for _ in range(20):
            seq, jpeg = self.frame()
            if seq is not None:
                return seq, jpeg
            time.sleep(0.1)
        raise AssertionError("no camera picture was served")

    def shown(self, seq: int, visible: bool = True) -> None:
        body = json.dumps({"seq": seq, "visible": visible})
        self.http.request("POST", "/shown", body=body)
        reply = self.http.getresponse()
        reply.read()
        assert reply.status == 204

    def status(self) -> dict[str, Any]:
        self.http.request("GET", "/status.json")
        return json.loads(self.http.getresponse().read())

    def close(self) -> None:
        self.http.close()


@pytest.fixture
def folder(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("vw")  # a short path, mode 0700


@pytest.fixture
def bench(folder) -> Iterator[Bench]:
    made = Bench(folder)
    yield made
    made.close()


@pytest.fixture(autouse=True)
def log_levels(monkeypatch) -> list[str]:
    """The console log levels main() asked for; no handler is attached in a test."""
    levels: list[str] = []
    monkeypatch.setattr(view_main, "set_console_level", levels.append)
    return levels


@pytest.fixture
def running() -> Iterator[list[Running]]:
    """The views a test started; each is ended when the test is over."""
    made: list[Running] = []
    yield made
    for view in made:
        view.end()


def start(running: list[Running], config: Path, *args: str, source: Any = None) -> Any:
    view = Running(config, *args, source=source)
    running.append(view)
    view.ready()
    return view


def never(program: ViewProgram) -> str:
    raise AssertionError("the view started")


def no_attach(*args: Any) -> Any:
    raise AssertionError("the view attached to the robot link service")


# The whole path


@link
def test_view_status_reaches_the_robot_app_and_shows_in_robot_state(
    make_realtime_robot_app, folder, running
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
        view = start(running, write_config(folder, stack.path), source=source)
        browser = Browser(view.program.port)
        seq, jpeg = browser.first_frame()
        # The picture is the left eye, made small.
        picture = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert picture.shape == (48, 64, 3)
        assert np.abs(picture[24, 16].astype(int) - RED).max() < 60
        assert np.abs(picture[24, 48].astype(int) - BLUE).max() < 60

        # A picture nobody acknowledged: VIEW_STATUS arrives, and is not live.
        until(lambda: browser.status()["view_status"] is not None or time.sleep(0.1))
        assert browser.status()["view_status"] == {"live": False, "age_s": 3600.0}
        assert watcher.ws.state.view_age_s is None

        # The page acknowledges what it drew: the Thor reports the view's age.
        for _ in range(3):
            seq, _ = browser.frame()
            browser.shown(seq)
            time.sleep(0.15)
        state = watcher.ws.wait_state(lambda s: s.view_age_s is not None, WAIT_S)
        assert 0.0 <= state.view_age_s < 1.0
        told = browser.status()
        assert told["view_status"]["live"] is True
        assert 0.0 <= told["view_status"]["age_s"] < 1.0
        assert told["picture"]["overlay"] == "" and told["robot"]["connected"] is True
        assert told["robot"]["driver"] == "sim"
        assert told["robot"]["clients"] == {"tool": 1, "viewer": 1}

        # The page stops acknowledging: after ack_timeout_s the age is unknown.
        watcher.ws.wait_state(lambda s: s.view_age_s is None, WAIT_S)
        assert browser.status()["view_status"] == {"live": False, "age_s": 3600.0}

        # Live again, then the tab is hidden.
        seq, _ = browser.frame()
        browser.shown(seq)
        watcher.ws.wait_state(lambda s: s.view_age_s is not None, WAIT_S)
        browser.shown(seq, visible=False)
        watcher.ws.wait_state(lambda s: s.view_age_s is None, WAIT_S)

        # Live once more, then the view ends: its last word is "not live".
        seq, _ = browser.frame()
        browser.shown(seq)
        watcher.ws.wait_state(lambda s: s.view_age_s is not None, WAIT_S)
        browser.close()
        assert view.end() == 0
        assert source.closes >= 1
        watcher.ws.wait_state(lambda s: s.view_age_s is None, 0.4)
        # Nothing but VIEW_STATUS came from the view: the app saw no command.
        assert stack.h.app.session_id == watcher.ws.state.session_id
        assert watcher.ws.state.stop_level == "none"
        assert watcher.ws.state.stop_number == state.stop_number
    finally:
        stack.close()


def test_the_view_sends_view_status_and_nothing_else(bench, folder, running):
    thor = bench.robot()
    view = start(running, write_config(folder, bench.path), "--source", "none")
    began = time.monotonic()
    received = [thor.next_message() for _ in range(4)]
    # One every status.period_s (0.05 s), also while not live.
    assert time.monotonic() - began > 0.1
    assert received == [NOT_LIVE] * 4
    assert view.end() == 0
    assert {msg_type for msg_type, _ in thor.drain()} <= {ViewStatus.TYPE}
    assert view.program.carrier.notices == []
    assert "The robot is not touched" in view.said[-1]


def test_a_recorded_source_shows_pictures_and_is_never_live(bench, folder, running):
    recorded = folder / "recorded"
    recorded.mkdir()
    cv2.imwrite(str(recorded / "left_raw.png"), eye(RED, BLUE))
    cv2.imwrite(str(recorded / "right_raw.png"), eye(GREEN, GREEN))
    thor = bench.robot()
    config = write_config(folder, bench.path)
    view = start(running, config, "--source", f"recorded:{recorded}")
    assert "never live" in view.said[0]
    browser = Browser(view.program.port)
    for _ in range(3):
        seq, jpeg = browser.first_frame()
        browser.shown(seq)
        time.sleep(0.1)
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).shape == (48, 64, 3)
    told = browser.status()
    assert told["picture"]["overlay"] == RECORDED
    assert told["state"] is not None and told["robot"]["connected"] is True
    assert told["view_status"] == {"live": False, "age_s": 3600.0}
    thor.drain()
    assert [thor.next_message() for _ in range(2)] == [NOT_LIVE] * 2
    browser.close()


# One view at a time, and the ways the program ends


def test_a_second_view_ends_with_exit_code_1(bench, folder, running, capsys):
    config = write_config(folder, bench.path)
    first = start(running, config, "--source", "none")
    said: list[str] = []
    code = view_main.main(
        ["--config", str(config), "--source", "none"], wait=never, say=said.append
    )
    assert code == 1 and said == []
    assert "live view: another live view is attached" in capsys.readouterr().err
    until(lambda: bench.service.roles() == {"viewer": 1}, "the second view's detach")
    # The first one is not disturbed.
    assert first.program.carrier.alive
    assert Browser(first.program.port).status()["robot"]["clients"] == {"viewer": 1}
    assert first.end() == 0


def test_without_a_robot_link_service_the_view_ends_with_exit_code_1(folder, capsys):
    config = write_config(folder, str(folder / "nobody.sock"))
    code = view_main.main(["--config", str(config), "--source", "none"], wait=never)
    assert code == 1
    assert "cannot attach to the robot link service" in capsys.readouterr().err


def test_when_the_robot_link_service_ends_the_view_ends_with_exit_code_1(
    bench, folder, running, capsys
):
    view = start(running, write_config(folder, bench.path), "--source", "none")
    bench.service.close()
    assert view.end() == 1
    assert "the robot link service is gone" in capsys.readouterr().err


def test_a_taken_port_ends_the_view_with_exit_code_1(bench, folder, capsys):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        code = view_main.main(
            [
                "--config",
                str(write_config(folder, bench.path)),
                "--source",
                "none",
                "--port",
                str(port),
            ],
            wait=never,
        )
    assert code == 1
    assert f"cannot serve on 127.0.0.1:{port}" in capsys.readouterr().err
    until(lambda: bench.service.roles() == {}, "the view's detach")


@link
@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT])
def test_the_view_process_ends_with_exit_code_0_on_a_signal(folder, number):
    from link_stack import Program, ServiceProcess

    path = str(folder / "s.sock")
    service = ServiceProcess(folder, path)
    view = None
    try:
        service.wait_ready()
        config = write_config(folder, path)
        view = Program(
            "isaaccapture_examples.agentic_remote.view",
            "--config",
            str(config),
            "--source",
            "none",
        )
        line = view.wait_line("live view: open http://127.0.0.1:")
        port = int(line.split("http://127.0.0.1:")[1].split("/")[0])
        browser = Browser(port)
        assert browser.status()["robot"]["connected"] is False
        browser.close()
        view.signal(number)
        assert view.wait_exit() == 0
        closing = f"live view: {signal.Signals(number).name}; closing."
        assert any(line.startswith(closing) for line in view.lines)
    finally:
        if view is not None:
            view.close()
        service.close()


# Without a camera


def test_the_view_runs_without_a_camera_and_says_so(
    bench, folder, running, monkeypatch
):
    monkeypatch.setattr(view_main, "OPEN_RETRY_S", 0.01)
    thor = bench.robot()
    source = StandInSource(fail="no camera receiver at /tmp/r.sock")
    view = start(running, write_config(folder, bench.path), source=source)
    model = view.program.model

    def note() -> str:
        return model.status_json(0.0, None)["picture"]["note"]

    until(lambda: note().startswith("no camera pictures"), "the last attempt")
    assert source.opens == 12
    assert note() == (
        "no camera pictures: the camera source did not open: no camera receiver "
        "at /tmp/r.sock (12 attempts, 0.01 s apart); the view shows the status only"
    )
    assert f"live view: {note()}" in view.said
    # The status is served and VIEW_STATUS is sent, not live.
    browser = Browser(view.program.port)
    assert browser.frame() == (None, b"")
    told = browser.status()
    assert told["picture"]["overlay"] == NO_PICTURE
    assert told["robot"]["connected"] is True and told["state"]["motion"] == "rest_hold"
    thor.drain()
    assert thor.next_message() == NOT_LIVE
    browser.close()
    assert view.end() == 0


def test_source_none_shows_the_status_only(bench, folder, running, log_levels):
    bench.robot()
    config = write_config(folder, bench.path)
    view = start(running, config, "--source", "none", "--log-level", "DEBUG")
    assert log_levels == ["debug"]
    assert view.said[0].endswith(
        "camera pictures: none; the page shows the status only"
    )
    assert f"open http://127.0.0.1:{view.program.port}/ in a browser" in view.said[0]
    browser = Browser(view.program.port)
    assert browser.frame() == (None, b"")
    until(lambda: browser.status()["state"] is not None or time.sleep(0.1))
    assert browser.status()["picture"] == {
        "seq": None,
        "age_s": None,
        "overlay": NO_PICTURE,
        "note": "no camera (--source none)",
    }
    browser.close()


class Pauses:
    """In place of waiting: keeps how long each wait was asked to be."""

    def __init__(self, end_at: int | None = None) -> None:
        self.asked: list[float] = []
        self.end_at = end_at

    def __call__(self, seconds: float) -> bool:
        if seconds > 0.0:
            self.asked.append(seconds)
        return self.end_at is not None and len(self.retries) >= self.end_at

    @property
    def retries(self) -> list[float]:
        return [s for s in self.asked if s == view_main.OPEN_RETRY_S]


def picture_loop(source: StandInSource, pauses: Pauses) -> tuple[Any, list[str]]:
    said: list[str] = []
    model = ViewModel()
    loop = PictureLoop(
        model,
        source,
        invented=False,
        settings=ViewSettings(fps=15.0, width_px=64),
        clock=time,
        pause=pauses,
        say=said.append,
    )
    loop.run()
    return model, said


def test_source_open_attempts_are_paced_and_counted():
    assert (view_main.OPEN_ATTEMPTS, view_main.OPEN_RETRY_S) == (12, 5.0)
    source, pauses = StandInSource(fail="nobody listens"), Pauses()
    model, said = picture_loop(source, pauses)
    # 12 attempts with 5 s between two of them, then no more.
    assert source.opens == 12 and source.closes == 0
    assert pauses.asked == [5.0] * 11
    again = "live view: the camera source did not open: nobody listens; trying again"
    assert [line.startswith(again) for line in said] == [True] * 11 + [False]
    assert said[-1] == (
        "live view: no camera pictures: the camera source did not open: nobody "
        "listens (12 attempts, 5 s apart); the view shows the status only"
    )
    assert model.picture() is None


def test_a_source_that_ends_is_opened_again_at_the_same_pace():
    # Three pictures, then the source fails and does not open again.
    source, pauses = StandInSource(ends_after=3), Pauses()
    model, said = picture_loop(source, pauses)
    assert model.picture()[0] == 3
    assert source.opens == 12 and source.closes == 1
    assert pauses.retries == [5.0] * 11
    assert "the camera source ended: the camera receiver went away" in said[0]
    # Every other wait is the rest of a picture period (1/15 s).
    assert all(0.0 < s <= 1 / 15 for s in pauses.asked if s != 5.0)


def test_the_picture_loop_ends_in_a_wait_when_told_to():
    source, pauses = StandInSource(fail="nobody listens"), Pauses(end_at=2)
    picture_loop(source, pauses)
    assert source.opens == 2


def test_a_camera_frame_becomes_a_small_jpeg():
    nv12 = bgr_to_nv12(eye(RED, BLUE))
    small = cv2.imdecode(np.frombuffer(encode_picture(nv12, 64, 90), np.uint8), 1)
    assert small.shape == (48, 64, 3)
    assert np.abs(small[24, 16].astype(int) - RED).max() < 60
    assert np.abs(small[24, 48].astype(int) - BLUE).max() < 60
    # A camera frame narrower than width_px keeps its size.
    same = cv2.imdecode(np.frombuffer(encode_picture(nv12, 960, 80), np.uint8), 1)
    assert same.shape == (96, 128, 3)
    with pytest.raises(ValueError, match="not an NV12 image"):
        encode_picture(np.zeros((100, 128), np.uint8), 64, 80)


# The view's own camera frame bridge


def test_the_bridge_child_of_the_view_names_gpu_0(folder, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere")
    monkeypatch.setenv("PYTHONPATH", "/apai/src")
    camera = {
        "bridge_python": "/venv/bin/python",
        "camera_pr": "/the/pr",
        "receiver_socket": "/tmp/r.sock",
        "streams": ["a.left", "a.right"],
        "max_skew_ms": 7.0,
    }
    settings = load_view_settings(write_config(folder, "", camera=camera))
    source, invented = view_main.make_source(settings, "link", ViewModel(), time)
    assert type(source) is BridgeSource and invented is False
    env = source._environment()
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "LD_LIBRARY_PATH" not in env
    assert env["PYTHONPATH"] == str(frame_source.PACKAGE_ROOT)
    assert env["AGENTIC_REMOTE_CAMERA_PR"] == "/the/pr"
    # The same environment as the bridge child of the agent program.
    agents = BridgeSource(
        python="/venv/bin/python", receiver_socket="r", run_dir="d", camera_pr="/the/pr"
    )
    assert env == agents._environment()
    # The same bridge, as a second reader of the same camera receiver.
    assert source._command == ["/venv/bin/python", "-m", frame_source.BRIDGE_MODULE]
    assert source._receiver_socket == "/tmp/r.sock"
    assert source._streams == ("a.left", "a.right")
    assert source._max_skew_ms == 7.0
    assert (source._size, source._devices) == (agents._size, agents._devices)
    assert source._run_dir == folder / "state" / "view"
    assert source._process is None  # nothing was started


def test_the_script_sets_the_environment_and_names_what_is_missing(tmp_path):
    script = (EXAMPLE / "run_view.sh").read_text()
    assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in script
    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert "export PYTHONDONTWRITEBYTECODE=1" in script
    assert "[ " not in script.replace("[[ ", "")  # [[ ]] only
    env = {"PATH": os.environ["PATH"]}
    done = subprocess.run(
        [str(EXAMPLE / "run_view.sh")],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "AGENTIC_REMOTE_VIEW_PYTHON (or AGENTIC_REMOTE_AGENT_PYTHON)" in done.stderr
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$CUDA_DEVICE_ORDER $CUDA_VISIBLE_DEVICES $PYTHONDONTWRITEBYTECODE '
        '${LD_LIBRARY_PATH:-unset} $PYTHONPATH $*"\n'
    )
    fake.chmod(0o755)
    expected = [
        "PCI_BUS_ID",
        "0",
        "1",
        "unset",
        f"{EXAMPLE}/python",
        "-m",
        "isaaccapture_examples.agentic_remote.view",
        "--config",
        "view.yaml",
    ]
    # The view's own interpreter, else the agent program's.
    for name in ("AGENTIC_REMOTE_AGENT_PYTHON", "AGENTIC_REMOTE_VIEW_PYTHON"):
        env.update({name: str(fake), "CUDA_VISIBLE_DEVICES": "1"})
        env["LD_LIBRARY_PATH"] = "/somewhere"
        done = subprocess.run(
            [str(EXAMPLE / "run_view.sh"), "--config", "view.yaml"],
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == expected
        env["AGENTIC_REMOTE_AGENT_PYTHON"] = "/nowhere/python"


# The settings and the command line


def test_the_shipped_file_holds_the_first_values():
    settings = load_view_settings(EXAMPLE / "configs" / "view.yaml")
    assert settings == ViewSettings(state_dir=EXAMPLE / ".state")
    assert (settings.http_address, settings.http_port) == ("127.0.0.1", 47620)
    assert (settings.width_px, settings.fps, settings.jpeg_quality) == (960, 5.0, 80)
    assert (settings.period_s, settings.ack_timeout_s) == (0.2, 1.0)
    assert settings.state_max_age_s == 1.0 and settings.socket_path is None
    assert settings.bridge_python == "" and settings.camera_pr == ""
    assert settings.receiver_socket == "/tmp/camera_receiver_avs1_cuda_ipc.sock"
    assert settings.streams == ("head.left", "head.right")
    assert settings.max_skew_ms == 5.0


def test_paths_are_taken_from_the_files_folder_and_may_name_a_variable(
    folder, monkeypatch
):
    monkeypatch.setenv("AGENTIC_REMOTE_CAMERA_PR", "/the/pr")
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    camera = {
        "bridge_python": "${AGENTIC_REMOTE_CAMERA_PR}/consumer/python",
        "camera_pr": "pr",
    }
    file = write_config(folder, "", camera=camera)
    settings = load_view_settings(file)
    assert settings.bridge_python == "/the/pr/consumer/python"
    assert settings.camera_pr == str(folder / "pr")
    camera["camera_pr"] = "${NOT_SET_ANYWHERE}/pr"
    with pytest.raises(view_main.ViewConfigError, match=r"camera\.camera_pr"):
        load_view_settings(write_config(folder, "", camera=camera))


@pytest.mark.parametrize(
    ("block", "key", "value", "says"),
    [
        ("http", "address", "0.0.0.0", "http.address must be 127.0.0.1"),
        ("http", "address", "localhost", "http.address must be 127.0.0.1"),
        ("http", "port", 70000, "http.port must be from 0 to 65535"),
        ("http", "port", "80", "http.port must be a whole number"),
        ("http", "host", "x", "unknown setting(s): http.host"),
        ("picture", "fps", 15.5, "picture.fps must be above 0 and at most 15"),
        ("picture", "fps", 0, "picture.fps must be above 0"),
        ("picture", "width_px", 32, "picture.width_px must be from 64 to 4096"),
        ("picture", "jpeg_quality", 101, "picture.jpeg_quality must be from 1 to 100"),
        ("status", "period_s", 2.0, "status.period_s must be above 0 and at most 1"),
        ("status", "ack_timeout_s", True, "status.ack_timeout_s must be a number"),
        ("status", "live", True, "unknown setting(s): status.live"),
        ("camera", "streams", ["a", "a"], "camera.streams must be two different"),
        ("camera", "max_skew_ms", 51, "camera.max_skew_ms must be"),
        ("camera", "calibration_dir", "x", "unknown setting(s): camera.calibration"),
        ("", "log_level", "loud", "log_level must be one of"),
        ("", "buttons", True, "unknown setting(s): buttons"),
        ("", "http", [1], "http must be a mapping"),
    ],
)
def test_a_setting_that_cannot_be_used_ends_with_exit_code_2(
    folder, capsys, block, key, value, says
):
    blocks = {block: {key: value}} if block else {}
    file = write_config(folder, "", **blocks)
    if not block:
        data = yaml.safe_load(file.read_text())
        data[key] = value
        file.write_text(yaml.safe_dump(data), encoding="utf-8")
    code = view_main.main(["--config", str(file)], attach=no_attach, wait=never)
    assert code == 2
    assert says in capsys.readouterr().err


def test_arguments_that_cannot_be_used_end_with_exit_code_2(folder, capsys):
    file = str(write_config(folder, ""))
    for argv, says in (
        ([], "--config"),
        (["--config", file, "--source", "usb"], "--source must be link, recorded:"),
        (["--config", file, "--source", f"recorded:{folder}/none"], "is not a folder"),
        (["--config", file, "--port", "70000"], "--port must be from 0 to 65535"),
        (["--config", file, "--log-level", "loud"], "--log-level"),
        (["--config", str(folder / "none.yaml")], "cannot read"),
        # The shipped file names no interpreter for the camera frame bridge.
        (["--config", file], "camera.bridge_python"),
    ):
        code = view_main.main(argv, attach=no_attach, wait=never)
        assert code == 2, argv
        assert says in capsys.readouterr().err, argv
    assert view_main.main(["--help"], attach=no_attach, wait=never) == 0
    assert "--source" in capsys.readouterr().out


# Source rules of the view's files

VIEW_FILES = ("__init__.py", "__main__.py", "model.py", "server.py")
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


def test_the_view_can_send_only_view_status():
    assert sorted(path.name for path in VIEW.glob("*.py")) == sorted(VIEW_FILES)
    sends: list[tuple[str, str]] = []
    for file in VIEW_FILES:
        tree = ast.parse((VIEW / file).read_text(encoding="utf-8"))
        assert not names_in(tree) & COMMANDS, file
        sends += [
            (file, ast.unparse(node.args[1]))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "send"
        ]
    # Two sends in the whole view: the periodic VIEW_STATUS and the last one.
    assert sorted(sends) == [
        ("__main__.py", "ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)"),
        ("__main__.py", "status"),
    ]
    main_source = (VIEW / "__main__.py").read_text(encoding="utf-8")
    assert "status = self.model.view_status(now, self.model.thor_wall_now(now))" in (
        main_source
    )
    # The model and the HTTP server cannot send at all.
    for file in ("model.py", "server.py"):
        tree = ast.parse((VIEW / file).read_text(encoding="utf-8"))
        modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not [m for m in modules if "carrier" in m or "link_service" in m], file


def test_the_view_needs_no_gpu_library():
    for file in VIEW_FILES:
        tree = ast.parse((VIEW / file).read_text(encoding="utf-8"))
        imported = {
            (alias.name if isinstance(node, ast.Import) else node.module or "")
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not [m for m in imported if m.split(".")[0] in ("torch", "cupy", "warp")]
