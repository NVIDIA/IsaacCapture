# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""viewer_live: the experiment's viewer with the camera's live picture.

The launcher runs the experiment's viewer (thor_viewer.py) as it is and adds
/api/live.mjpg, /api/live.json, /api/live.jpg, /api/live/shown and a block in
the page. A stand-in camera makes the pictures of most tests: each one shows
its own number. The tests of the whole launcher start it with run_viewer.sh,
on a fake camera receiver with the real camera frame bridge program on file
slots: no GPU, no CuPy. Everything is asked over HTTP on 127.0.0.1, on a port
the system picks; the page's script is run by node on a stand-in for the page,
where node is installed.

The live arm is the arm of each ROBOT_STATE as it arrives, drawn by the
experiment's own page: /api/live/robot.json and the block's second part. A
stand-in for the robot app gives the viewer 20 ROBOT_STATEs a second; with
readings that stand still, as with the motors off, the arm is not called live.

The viewer's page with its live picture showing is the live view: the viewer
tells the robot app, with VIEW_STATUS link messages, how fresh the picture is
that the page loaded last has drawn and acknowledged. A takeover makes a person
the control owner; a human drive is one stretch of a person's motion. The robot
link service of these tests is one of their own, on a socket in a temporary
folder: none attaches to a service of this machine.

The viewer instance is one start of the viewer, to its end. A page that was
open across a restart of the viewer, and a page that a newer load replaced,
run under node and in headless Chrome on the launcher's real page: there the
viewer is ended with every connection the browser has open and started again
on the same port, one the system picked.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
import test_launcher
import test_viewer
import yaml
from camera_bench import BRIDGE_CHILD, FakeReceiver
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent import launcher
from isaaccapture_examples.agentic_remote.agent.camera import frame_source
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import (
    BridgeSource,
    CameraError,
    StereoPair,
)
from test_frame_source import Publishing
from test_launcher import EXAMPLE, QUICK, SHIPPED, settings_file
from test_viewer import RECORDED, ViewerProcess, record, script_run

from conftest import APAI_SRC_ENV, CAMERA_PR_ENV

# The live picture is packed with OpenCV; without it this file is skipped.
cv2 = pytest.importorskip("cv2")

from isaaccapture_examples.agentic_remote.viewer_live import (  # noqa: E402
    __main__ as live_main,
)
from isaaccapture_examples.agentic_remote.link_service import local  # noqa: E402
from isaaccapture_examples.agentic_remote.link_service.local import (  # noqa: E402
    connect_local,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (  # noqa: E402
    GripperState,
    JointSample,
    ViewStatus,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app import (  # noqa: E402
    app as app_module,
)
from isaaccapture_examples.agentic_remote.robot_app.config import (  # noqa: E402
    load_config,
)
from isaaccapture_examples.agentic_remote.viewer_live import live as live_module  # noqa: E402
from isaaccapture_examples.agentic_remote.viewer_live import (  # noqa: E402
    settings as live_settings,
)
from isaaccapture_examples.agentic_remote.viewer_live import (  # noqa: E402
    view_link as link_module,
)
from isaaccapture_examples.agentic_remote.viewer_live import (  # noqa: E402
    robot as robot_module,
)
from isaaccapture_examples.agentic_remote.viewer_live.live import (  # noqa: E402
    BOUNDARY,
    INSTANCE_HEADER,
    INSTANCE_PLACE,
    PICTURE_PATH,
    ROBOT_PATH,
    SHOWN_PATH,
    STATE_PATH,
    STREAM_PATH,
    TOKEN_HEADER,
    TOKEN_PLACE,
    LivePicture,
    live_handler,
    with_page_block,
)
from isaaccapture_examples.agentic_remote.viewer_live.picture import (  # noqa: E402
    encode_picture,
)
from isaaccapture_examples.agentic_remote.viewer_live.robot import (  # noqa: E402
    ArmModel,
    ArmModelError,
    LiveRobot,
)
from isaaccapture_examples.agentic_remote.viewer_live.view_link import (  # noqa: E402
    ViewLink,
)
from isaaccapture_examples.agentic_remote.viewer_live.view_model import (  # noqa: E402
    AGE_WHEN_NOT_LIVE_S,
    CLOCK_STEP_QUIET_S,
    ViewModel,
)
from link_bench import robot_state  # noqa: E402

apai = pytest.mark.apai
experiment = pytest.mark.experiment
link = pytest.mark.link
# The fixtures of test_launcher.py and test_viewer.py, for the tests of this file.
make_stage = test_launcher.make_stage
stage = test_launcher.stage
viewer = test_viewer.viewer
stand_in = test_viewer.stand_in

WAIT_S = 30.0  # real seconds a test waits for another thread or process
CONFIGS = EXAMPLE / "configs"
PACKAGE = Path(live_module.__file__).resolve().parent
BLOCK = live_module.PAGE_BLOCK_FILE.read_bytes()
# How a served page names its view token: the secret of that load of the page.
VIEW_TOKEN = re.compile(rb"const TOKEN = '([A-Za-z0-9_-]{32})';")
# How a served page names the viewer instance that served it: one start of
# the viewer, to its end.
VIEWER_INSTANCE = re.compile(rb"const INSTANCE = '([0-9a-f]{16})';")
# What only the block holds: it is in a served page once.
BLOCK_MARK = f"const PICTURE = '{PICTURE_PATH}'".encode()
NOT_LIVE = ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)
MODULE = "isaaccapture_examples.agentic_remote.viewer_live"
# The squares along a stand-in picture's top edge: the bits of its number.
BITS = 16


def until(holds: Callable[[], Any], what: str) -> Any:
    """holds() once it is true; at most WAIT_S."""
    end = time.monotonic() + WAIT_S
    while True:
        value = holds()
        if value:
            return value
        assert time.monotonic() < end, f"{what} did not come"
        time.sleep(0.02)


class Camera:
    """A stand-in camera: picture N exists from N / hz seconds after open().

    next_pair() gives the newest picture when it is newer than the last one it
    gave, else it waits for the next. The left picture shows N as BITS black or
    white squares along its top edge; with `noisy` the rest is noise, which
    makes a large JPEG. With `fail` open() raises a CameraError of that text;
    after `ends_after` pairs the source fails. Each picture was exposed 0.03 s
    before it is given, on this machine's wall clock.
    """

    def __init__(
        self,
        hz: float = 60.0,
        size: tuple[int, int] = (128, 96),
        *,
        noisy: bool = False,
        fail: str = "",
        ends_after: int | None = None,
    ) -> None:
        self.hz = hz
        self.fail = fail
        self.ends_after = ends_after
        self.opens = 0
        self.opened_at: list[float] = []  # time.monotonic() of each open()
        self.closes = 0
        self.asked = 0  # calls of next_pair()
        self.last_none = ""
        self.stopped = False  # True: the camera sends nothing, and stays open
        self._width, self._height = size
        self._base = np.full((self._height * 3 // 2, self._width), 128, np.uint8)
        if noisy:
            noise = np.random.default_rng(1).integers(0, 256, size[::-1])
            self._base[: self._height] = noise
        self._open = False
        self._began = 0.0
        self._given = 0  # pairs handed out since open()
        self._last = 0

    def open(self) -> None:
        self.opened_at.append(time.monotonic())
        self.opens += 1
        if self.fail:
            raise CameraError(self.fail)
        self._open = True
        self._began = time.monotonic()
        self._given = self._last = 0

    def close(self) -> None:
        self.closes += 1
        self._open = False

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        self.asked += 1
        if not self._open:
            raise CameraError("the stand-in camera is closed")
        if self.ends_after is not None and self._given >= self.ends_after:
            raise CameraError("the camera receiver went away")
        if self.stopped:
            time.sleep(timeout_s)
            self.last_none = "the camera sends nothing"
            return None
        now = time.monotonic()
        number = int((now - self._began) * self.hz) + 1
        if number <= self._last:
            number = self._last + 1
            wait = self._began + (number - 1) / self.hz - now
            if wait > timeout_s:
                time.sleep(timeout_s)
                self.last_none = "the next picture is not there yet"
                return None
            time.sleep(max(0.0, wait))
        self._last = number
        self._given += 1
        left = self._base.copy()
        wide, high = self._width // BITS, self._height // 6
        for bit in range(BITS):
            left[:high, bit * wide : (bit + 1) * wide] = 235 * (number >> bit & 1) + 16
        exposed_ns = int((time.time() - 0.03) * 1e9)
        return StereoPair(
            session_id=1,
            seq=number,
            left=left,
            right=self._base,
            sof_wall_ns=(exposed_ns, exposed_ns + 1000),
            device_index=(5, 7),
            rotation=0,
            receiver_rx_mono_s=time.monotonic() - 0.01,
            check={"source": "test"},
        )


def number_of(jpeg: bytes) -> int:
    """The number a stand-in picture shows."""
    gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    height, width = gray.shape
    row = gray[height // 12]
    bits = [row[int((bit + 0.5) * width / BITS)] > 128 for bit in range(BITS)]
    return sum(1 << bit for bit, white in enumerate(bits) if white)


PAGE = b'<!doctype html><html><body><img id="photo">the page</body></html>'


class PlainViewer(BaseHTTPRequestHandler):
    """A stand-in for the viewer's request handler: a page and one more path."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            body, kind = PAGE, "text/html; charset=utf-8"
        elif path == "/api/runs":
            body, kind = b"[]", "application/json"
        else:
            self.send_error(400, "Unknown endpoint")
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def bearer(headers: Any, token: str) -> bool:
    """A stand-in for the viewer's access rule: the token as a Bearer header."""
    return headers.get("Authorization", "") == f"Bearer {token}"


@contextmanager
def serving(handler: type) -> Iterator[int]:
    """A server of this handler class on a port the system picks; the port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True  # as the viewer has it: a stream ends with it
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        thread.join(WAIT_S)
        server.server_close()


@contextmanager
def showing(
    camera: Any,
    *,
    base: type = PlainViewer,
    token: str | None = None,
    authorized: Callable[[Any, str], bool] = bearer,
    lines: list[str] | None = None,
    view: Callable[[], dict[str, Any]] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    this_machine: Callable[[Any], bool] | None = None,
    robot: LiveRobot | None = None,
    **settings: Any,
) -> Iterator[tuple[LivePicture, int]]:
    """A started LivePicture on this camera, served with `base`; it and the port."""
    said = lines.append if lines is not None else lambda line: None
    live = LivePicture(lambda: camera, say=said, **settings)
    live.start()
    try:
        handler = live_handler(
            base, token, live, authorized, view, monotonic, this_machine, robot
        )
        with serving(handler) as port:
            yield live, port
    finally:
        live.close()


def block_in(body: bytes, view_token: bytes) -> bytes:
    """The block as a served page holds it, with this view token.

    The block holds the page's viewer instance where the file holds
    INSTANCE_PLACE, and the view token where it holds TOKEN_PLACE.
    """
    assert BLOCK.count(TOKEN_PLACE) == 1 and BLOCK.count(BLOCK_MARK) == 1
    assert BLOCK.count(INSTANCE_PLACE) == 1
    found = VIEWER_INSTANCE.search(body)
    assert found is not None, "the served page names no viewer instance"
    return BLOCK.replace(TOKEN_PLACE, view_token).replace(
        INSTANCE_PLACE, found.group(1)
    )


def served_block(body: bytes, page: bytes) -> str:
    """The view token of a served page.

    The body is `page` with the block before its </body>, and nothing else;
    the block holds that load's view token where the file holds TOKEN_PLACE.
    """
    found = VIEW_TOKEN.search(body)
    assert found is not None, "the served page holds no view token"
    at = page.rindex(b"</body>")
    assert body == page[:at] + block_in(body, found.group(1)) + page[at:]
    return found.group(1).decode()


def thor_clock(model: ViewModel, since_s: float = 10.0) -> None:
    """A ROBOT_STATE of a Thor whose clocks are this machine's, handed to the model.

    A model without a fresh one is first handed one that arrived since_s ago:
    longer ago than the quiet time its pictures wait out after the first.
    """
    now, wall = time.monotonic(), time.time()
    if model.thor_wall_now(now) is None:
        first = dataclasses.replace(
            robot_state(1), clock_mono_s=now - since_s, clock_wall_s=wall - since_s
        )
        model.on_message(first, now - since_s)
    state = dataclasses.replace(robot_state(2), clock_mono_s=now, clock_wall_s=wall)
    model.on_message(state, now)


def status_now(model: ViewModel) -> ViewStatus:
    """The VIEW_STATUS the model would send now, with a fresh ROBOT_STATE."""
    thor_clock(model)
    return model.status_to_send(time.monotonic())


class Tab:
    """What the block's script does in a browser tab, over HTTP.

    load() fetches the page: a new view token, from then on the only valid
    one. While it shows the live picture it fetches the picture after the one
    it drew last and acknowledges it. leave() is what the script does when the
    live picture leaves the screen ("Map picture", a replay, a hidden tab): no
    more fetches, and one word that the picture drawn last shows no longer.
    freeze() stops without a word, as a browser that is frozen or closed.
    """

    def __init__(self, port: int, **headers: str) -> None:
        self.port, self.headers = port, headers
        self.token = ""
        self.seq = 0  # the picture drawn last
        self.drawn: list[int] = []
        self.taken = 0  # acknowledgements the viewer counted
        self.refused: list[int] = []  # the status codes of those it did not
        self.replaced = False  # a fetch said that this page counts no longer
        self.failed: list[BaseException] = []
        self._showing = threading.Event()
        self._thread: threading.Thread | None = None

    def _http(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=WAIT_S)

    def page(self) -> bytes:
        """One GET of the page; its body."""
        connection = self._http()
        try:
            connection.request("GET", "/", headers=self.headers)
            reply = connection.getresponse()
            status, body = reply.status, reply.read()
        finally:
            connection.close()
        assert status == 200, status
        return body

    def load(self) -> str:
        body = self.page()
        found = VIEW_TOKEN.search(body)
        assert found is not None, "the served page holds no view token"
        self.token = found.group(1).decode()
        return self.token

    def fetch(
        self, connection: http.client.HTTPConnection, after: int | None = None
    ) -> tuple[int, int | None, bytes]:
        """One GET of the next picture: (status code, its number, the JPEG)."""
        after = self.seq if after is None else after
        headers = {TOKEN_HEADER: self.token, **self.headers}
        connection.request("GET", f"{PICTURE_PATH}?after={after}", headers=headers)
        reply = connection.getresponse()
        seq = reply.getheader("X-Picture-Seq")
        return reply.status, (None if seq is None else int(seq)), reply.read()

    def shown(
        self,
        connection: http.client.HTTPConnection,
        seq: int,
        visible: bool = True,
        **body: Any,
    ) -> int:
        """One acknowledgement; the status code."""
        told = {"seq": seq, "visible": visible, "token": self.token, **body}
        headers = {"Content-Type": "application/json", **self.headers}
        connection.request("POST", SHOWN_PATH, json.dumps(told), headers)
        reply = connection.getresponse()
        reply.read()
        return reply.status

    def show(self) -> None:
        """The live picture shows from now on."""
        assert self._thread is None, "the tab shows the live picture already"
        if not self.token:
            self.load()
        self._showing.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        connection = self._http()
        try:
            while self._showing.is_set():
                status, seq, _ = self.fetch(connection)
                if status == 403:
                    self.replaced = True
                    return
                if status != 200 or not self._showing.is_set():
                    continue
                self.seq = seq
                self.drawn.append(seq)
                code = self.shown(connection, seq)
                if code == 204:
                    self.taken += 1
                else:
                    self.refused.append(code)
        except (OSError, http.client.HTTPException) as error:
            self.failed.append(error)
        finally:
            connection.close()

    def freeze(self) -> None:
        """Stop fetching and acknowledging, without a word."""
        self._showing.clear()
        if self._thread is not None:
            self._thread.join(WAIT_S)
            assert not self._thread.is_alive(), "the tab did not stop"
            self._thread = None

    def leave(self) -> int:
        """The live picture leaves the screen; the status code of that word."""
        self.freeze()
        connection = self._http()
        try:
            return self.shown(connection, self.seq, visible=False)
        finally:
            connection.close()

    def close(self) -> None:
        self.freeze()


# The address the tests' "other machine" asks from: a second loopback address.
OTHER_MACHINE = "127.0.0.2"
# How a page served to another machine names its view token: it has none.
NO_VIEW_TOKEN = b"const TOKEN = '';"


def only_127_0_0_1(connection: Any) -> bool:
    """A stand-in for on_this_machine(): OTHER_MACHINE is not this machine.

    As the rule itself, it reads the client's address off the connection.
    """
    return connection.getpeername()[0] == "127.0.0.1"


class Elsewhere(Tab):
    """A tab of a browser on another machine: every request comes from OTHER_MACHINE.

    load() fetches the page, which carries no view token; watch() is what the
    block's script does on such a page: a picture to watch, with no token.
    """

    def _http(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=WAIT_S, source_address=(OTHER_MACHINE, 0)
        )

    def load(self) -> str:
        body = self.page()
        assert body.count(NO_VIEW_TOKEN) == 1 and VIEW_TOKEN.search(body) is None
        assert body.count(BLOCK_MARK) == 1
        return ""

    def watch(
        self, connection: http.client.HTTPConnection, after: int = 0
    ) -> tuple[int, int | None, bytes]:
        """One GET of a picture to watch: (status code, its number, the JPEG)."""
        connection.request(
            "GET", f"{PICTURE_PATH}?after={after}&watch=1", headers=self.headers
        )
        reply = connection.getresponse()
        seq = reply.getheader("X-Picture-Seq")
        return reply.status, (None if seq is None else int(seq)), reply.read()


def get(port: int, path: str, **headers: str) -> tuple[int, str, bytes]:
    """(status code, content type, body) of one GET with these headers."""
    connection = HTTPConnection("127.0.0.1", port, timeout=WAIT_S)
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type", ""), response.read()
    finally:
        connection.close()


def live_state(port: int, query: str = "", **headers: str) -> dict[str, Any]:
    status, kind, body = get(port, STATE_PATH + query, **headers)
    assert (status, kind) == (200, "application/json"), (status, body[:300])
    return json.loads(body)


class Stream:
    """One browser's GET of the live stream, read picture by picture."""

    def __init__(
        self,
        port: int,
        path: str = STREAM_PATH,
        *,
        receive_buffer: int | None = None,
        source: str | None = None,
        **headers: str,
    ) -> None:
        self._sock = socket.socket()
        if receive_buffer is not None:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer)
        self._sock.settimeout(WAIT_S)
        if source is not None:  # the address the browser asks from
            self._sock.bind((source, 0))
        self._sock.connect(("127.0.0.1", port))
        lines = [f"GET {path} HTTP/1.1", "Host: 127.0.0.1"]
        lines += [f"{name}: {value}" for name, value in headers.items()]
        self._sock.sendall("\r\n".join([*lines, "", ""]).encode())
        self._file = self._sock.makefile("rb")
        self.status = int(self._file.readline().split()[1])
        self.headers = self._headers()

    def _headers(self) -> dict[str, str]:
        found = {}
        while True:
            line = self._file.readline().decode().strip()
            if not line:
                return found
            name, _, value = line.partition(":")
            found[name.lower()] = value.strip()

    def picture(self) -> bytes:
        """The next picture's JPEG."""
        assert self._file.readline() == f"--{BOUNDARY}\r\n".encode()
        headers = self._headers()
        assert headers["content-type"] == "image/jpeg"
        jpeg = self._file.read(int(headers["content-length"]))
        assert self._file.readline() == b"\r\n"
        return jpeg

    def pictures_for(self, seconds: float) -> list[bytes]:
        """The pictures that arrive within `seconds` from now."""
        end = time.monotonic() + seconds
        got = []
        while True:
            jpeg = self.picture()
            if time.monotonic() > end:
                return got
            got.append(jpeg)

    def close(self) -> None:
        self._file.close()
        self._sock.close()


# The stream


@pytest.mark.parametrize(("fps", "at_least"), [(30.0, 20), (5.0, 4)])
def test_the_stream_gives_the_cameras_pictures_at_no_more_than_the_asked_rate(
    fps: float, at_least: int
) -> None:
    camera = Camera(hz=60.0)
    with showing(camera, fps=fps) as (_, port):
        until(lambda: live_state(port)["available"], "the live picture")
        stream = Stream(port, f"{STREAM_PATH}?page=abc&n=1")
        try:
            assert stream.status == 200
            kind = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            assert stream.headers["content-type"] == kind
            assert stream.headers["cache-control"] == "no-store"
            stream.pictures_for(0.5)  # the first ones: the rate settles
            pictures = stream.pictures_for(1.0)
            numbers = [number_of(jpeg) for jpeg in pictures]
            # Every picture is another one, newer than the one before, and
            # there are no more of them than were asked for.
            assert numbers == sorted(set(numbers)), numbers
            assert at_least <= len(numbers) <= fps + 1, numbers
            shape = cv2.imdecode(np.frombuffer(pictures[-1], np.uint8), 1).shape
            assert shape == (96, 128, 3)  # a narrow camera frame keeps its size
            state = live_state(port, "?page=abc")
            assert state["available"] is True and state["reason"] == ""
            assert state["fps_asked"] == fps
            assert at_least <= state["fps_measured"] <= fps + 1
            assert (state["width"], state["width_asked"]) == (128, 960)
            assert 0.0 <= state["age_s"] < 0.5
            # This page's stream is open; another page's is not.
            assert (state["clients"], state["streaming"]) == (1, True)
            assert live_state(port, "?page=other")["streaming"] is False
        finally:
            stream.close()
        # Without a browser on the stream the camera is asked once a second.
        until(lambda: live_state(port)["clients"] == 0, "the stream's end")
        time.sleep(1.2)
        asked = camera.asked
        time.sleep(2.0)
        assert 1 <= camera.asked - asked <= 3
        idle = live_state(port)
        assert idle["available"] is True and idle["fps_measured"] <= 2.0
        assert idle["streaming"] is False
    assert camera.opens == 1 and camera.closes >= 1 and not camera._open


def test_a_browser_that_reads_slowly_skips_pictures_and_slows_no_other() -> None:
    camera = Camera(hz=60.0, size=(1280, 960), noisy=True)
    with showing(camera, fps=30.0) as (live, port):
        until(lambda: live_state(port)["available"], "the live picture")
        slow = Stream(port, receive_buffer=32 * 1024)
        fast = Stream(port)
        fast_numbers: list[int] = []

        def read_fast() -> None:
            fast.pictures_for(0.5)
            fast_numbers.extend(number_of(jpeg) for jpeg in fast.pictures_for(2.0))

        reader = threading.Thread(target=read_fast, daemon=True)
        reader.start()
        try:
            slow_numbers = []
            for _ in range(8):
                jpeg = slow.picture()
                slow_numbers.append(number_of(jpeg))
                time.sleep(0.3)
            reader.join(WAIT_S)
            assert not reader.is_alive()
        finally:
            slow.close()
            fast.close()
        # The pictures are far larger than the stream's send buffer.
        assert len(jpeg) > 2 * live_module._SEND_BUFFER_BYTES
        picture = cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1)
        assert picture.shape == (720, 960, 3)  # made 960 px wide
        # The slow browser got newer and newer pictures and missed most: eight
        # of the 70 the camera's 2.4 s gave at 30 a second.
        assert slow_numbers == sorted(set(slow_numbers)), slow_numbers
        assert slow_numbers[-1] - slow_numbers[0] >= 3 * len(slow_numbers)
        # The fast one had its 20 and more a second all the while.
        assert fast_numbers == sorted(set(fast_numbers)), fast_numbers
        assert 40 <= len(fast_numbers) <= 62, fast_numbers
        assert live.state()["width"] == 960


def test_the_live_paths_need_the_token_as_the_viewers_own_paths_do() -> None:
    token = "t" * 43
    with showing(Camera(), token=token) as (live, port):
        until(lambda: live.state()["available"], "the live picture")
        for path in (STATE_PATH, STREAM_PATH, f"{STATE_PATH}?token={token}"):
            status, _, body = get(port, path)
            assert status == 401 and b"private access link" in body, path
            assert get(port, path, Authorization="Bearer wrong")[0] == 401, path
        assert live_state(port, Authorization=f"Bearer {token}")["available"] is True
        stream = Stream(port, Authorization=f"Bearer {token}")
        try:
            assert stream.status == 200 and number_of(stream.picture()) > 0
        finally:
            stream.close()
        refused = Stream(port)
        try:
            assert refused.status == 401
        finally:
            refused.close()
    # Without a token, as with --no-auth, both are open.
    with showing(Camera(), token=None) as (live, port):
        until(lambda: live_state(port)["available"], "the live picture")
        stream = Stream(port)
        try:
            assert stream.status == 200 and number_of(stream.picture()) > 0
        finally:
            stream.close()


def test_without_a_camera_the_viewer_serves_the_rest_and_says_one_line() -> None:
    lines: list[str] = []
    camera = Camera(fail="AGENTIC_REMOTE_CAMERA_PR is not set")
    with showing(camera, lines=lines, retry_pauses_s=(0.05,)) as (_, port):
        until(lambda: camera.opens >= 4, "four tries to open the camera")
        state = live_state(port)
        assert state["available"] is False and state["age_s"] is None
        assert state["width"] is None and state["fps_measured"] == 0.0
        why = "AGENTIC_REMOTE_CAMERA_PR is not set; trying again in 0.05 s"
        assert state["reason"] == why
        status, _, body = get(port, STREAM_PATH)
        assert status == 503 and b"No live picture" in body
        # The viewer's own paths answer as ever, the page with the block.
        assert get(port, "/api/runs") == (200, "application/json", b"[]")
        assert get(port, "/api/nothing")[0] == 400
        served_block(get(port, "/")[2], PAGE)
    # One line for the many failed tries with the same reason and pause.
    assert lines == [
        f"viewer: no live picture: {why}. The recorded runs are served as before"
    ]


def test_a_camera_that_ends_is_opened_again_and_each_change_is_said_once() -> None:
    lines: list[str] = []
    camera = Camera(ends_after=5)
    with showing(camera, lines=lines, retry_pauses_s=(0.05, 60.0)) as (_, port):
        until(lambda: camera.opens >= 3, "the camera opened three times")
        assert live_state(port)["available"] is True  # the pictures go on
    assert camera.closes >= camera.opens and not camera._open
    arrive = "viewer: live picture: the camera's pictures arrive"
    # The first pause each time: pictures arrived before the camera ended.
    gone = (
        "viewer: no live picture: the camera source ended: the camera receiver "
        "went away; trying again in 0.05 s. The recorded runs are served as before"
    )
    assert lines[:3] == [arrive, gone, arrive]
    assert set(lines) == {arrive, gone}


def no_live_picture(why: str, pause_s: float) -> str:
    """The line of a failed try with this reason and this pause."""
    return (
        f"viewer: no live picture: {why}; trying again in {pause_s:g} s. The "
        "recorded runs are served as before"
    )


class Unpaused(LivePicture):
    """A LivePicture that waits out no pause of 5 s or more; those it was asked for.

    The pause between two pictures, 1 s at most, is waited out as ever.
    """

    def __init__(self, *args: Any, **settings: Any) -> None:
        super().__init__(*args, **settings)
        self.pauses: list[int] = []

    def _pause(self, until: float, watched: bool) -> None:
        left = until - time.monotonic()
        if left < 4.5:
            super()._pause(until, watched)
        else:
            self.pauses.append(round(left))


def test_the_pause_after_a_failed_try_grows_to_60_s_and_is_5_s_after_pictures() -> None:
    assert live_module.RETRY_PAUSES_S == (5.0, 10.0, 20.0, 40.0, 60.0)
    lines: list[str] = []
    camera = Camera(fail="nobody listens")
    live = Unpaused(lambda: camera, say=lines.append)  # the launcher's own pauses
    live.start()
    try:
        until(lambda: camera.opens >= 8, "eight tries to open the camera")
        assert live.state()["reason"] == "nobody listens; trying again in 60 s"
        # From now on the camera opens, gives one picture and ends.
        camera.ends_after = 1
        camera.fail = ""
        until(lambda: len(lines) >= 9, "the lines of two cameras that ended")
    finally:
        live.close()
    pauses = live.pauses
    assert pauses[:7] == [5, 10, 20, 40, 60, 60, 60], pauses[:9]
    # 60 s until pictures arrived, 5 s after every camera that gave pictures.
    after_pictures = pauses.index(5, 1)
    assert set(pauses[7:after_pictures]) <= {60}
    assert len(pauses[after_pictures:]) >= 2 and set(pauses[after_pictures:]) == {5}
    # Each pause is said once, also the one that stays.
    ended = "the camera source ended: the camera receiver went away"
    arrive = "viewer: live picture: the camera's pictures arrive"
    assert lines[:9] == [
        *(no_live_picture("nobody listens", pause) for pause in (5, 10, 20, 40, 60)),
        arrive,
        no_live_picture(ended, 5),
        arrive,
        no_live_picture(ended, 5),
    ]


def test_the_pause_of_a_failed_try_is_waited_out() -> None:
    lines: list[str] = []
    camera = Camera(fail="nobody listens")
    with showing(camera, lines=lines, retry_pauses_s=(0.05, 0.5)):
        until(lambda: camera.opens >= 4, "four tries to open the camera")
    waited = np.diff(camera.opened_at[:4])
    assert 0.05 <= waited[0] < 0.5 <= min(waited[1:]), waited
    assert lines == [
        no_live_picture("nobody listens", 0.05),
        no_live_picture("nobody listens", 0.5),
    ]


def test_a_stream_ends_with_the_live_picture_and_close_ends_both_threads() -> None:
    camera = Camera()
    with showing(camera) as (live, port):
        until(lambda: live.state()["available"], "the live picture")
        stream = Stream(port)
        assert number_of(stream.picture()) > 0
        live.close()
        live.close()  # a second close does nothing
        try:
            with pytest.raises((AssertionError, OSError)):
                for _ in range(1000):
                    stream.picture()
        finally:
            stream.close()
        state = live.state()
        assert state["available"] is False
        assert state["reason"] == "the viewer is ending"
        assert get(port, STREAM_PATH)[0] == 503
    assert not [thread for thread in live._threads if thread.is_alive()]
    assert camera.closes >= 1 and not camera._open


def test_only_a_page_gets_the_block() -> None:
    def response(status: str, kind: str, body: bytes) -> bytes:
        head = f"HTTP/1.0 {status}\r\nServer: x\r\nContent-Type: {kind}\r\n"
        return f"{head}Content-Length: {len(body)}\r\n\r\n".encode() + body

    html = "text/html; charset=utf-8"
    page = b"<html><body>one</body> two </body></html>"
    grown = with_page_block(response("200 OK", html, page), b"<b>")
    # Before the last </body>, with the length of the new body.
    assert grown == response(
        "200 OK", html, b"<html><body>one</body> two <b></body></html>"
    )
    for same in (
        response("303 See Other", html, page),
        response("401 Unauthorized", html, page),
        response("200 OK", "application/json", page),
        response("200 OK", html, b"<html>no end"),
        b"not an answer",
    ):
        assert with_page_block(same, b"<b>") == same


# The numbered pictures and their acknowledgements


def test_a_page_that_draws_and_acknowledges_pictures_makes_the_view_live() -> None:
    with showing(Camera(hz=60.0), fps=30.0) as (live, port):
        model = live.model
        until(lambda: live_state(port)["available"], "the live picture")
        tab = Tab(port)
        connection = tab._http()
        try:
            # Without a page, and for a page that only fetched: not live.
            assert status_now(model) == NOT_LIVE
            tab.load()
            status, seq, jpeg = tab.fetch(connection, after=0)
            assert status == 200 and seq == number_of(jpeg) > 0
            assert status_now(model) == NOT_LIVE
            # The picture after that one comes once the camera has it: a new
            # number, and the picture that shows it.
            status, newer, jpeg = tab.fetch(connection, after=seq)
            assert status == 200 and newer == number_of(jpeg) > seq
            # Drawn and acknowledged in a visible tab: live, with that
            # picture's age since its exposure 0.03 s before it was given.
            assert tab.shown(connection, newer) == 204
            status = status_now(model)
            assert status.live and 0.03 <= status.age_s < 0.5
            assert status.sent_mono_s == pytest.approx(time.monotonic(), abs=0.5)
            # The same three requests went over one connection.
            assert connection.sock is not None
            # The picture left the screen: not live, and "it shows" for that
            # picture counts no longer.
            assert tab.shown(connection, newer, visible=False) == 204
            assert status_now(model) == NOT_LIVE
            assert tab.shown(connection, newer) == 403
            assert status_now(model) == NOT_LIVE
            # The next picture the page fetches and draws counts again.
            status, third, _ = tab.fetch(connection, after=newer)
            assert status == 200 and tab.shown(connection, third) == 204
            assert status_now(model).live
            # Without a word for longer than the acknowledgement's time limit
            # the view is not live.
            time.sleep(1.1)
            assert status_now(model) == NOT_LIVE
        finally:
            connection.close()


def test_only_the_page_loaded_last_gets_pictures_and_counts() -> None:
    with showing(Camera(hz=60.0), fps=30.0) as (live, port):
        model = live.model
        until(lambda: live_state(port)["available"], "the live picture")
        first, second = Tab(port), Tab(port)
        one, two = first._http(), second._http()
        try:
            # No page was loaded: no token is valid, and none is no token.
            assert first.fetch(one)[0] == 403
            first.token = "A" * 32
            assert first.fetch(one)[0] == 403 and first.shown(one, 1) == 403
            first.load()
            _, seq, _ = first.fetch(one, after=0)
            assert first.shown(one, seq) == 204 and status_now(model).live
            # The page is loaded again, here by another tab: the view is not
            # live until that page has drawn a picture, and the first page's
            # requests and words count no longer.
            assert second.load() != first.token
            assert status_now(model) == NOT_LIVE
            assert first.fetch(one)[0] == 403
            assert first.shown(one, seq) == 403
            assert status_now(model) == NOT_LIVE
            # Nor does the new page's word for a picture it never fetched.
            assert second.shown(two, seq) == 403
            _, newer, _ = second.fetch(two, after=0)
            assert second.shown(two, newer) == 204 and status_now(model).live
            # A number the page was not served, and a body that is not the
            # three values, count for nothing.
            assert second.shown(two, newer + 1000) == 403
            word = {"seq": newer, "visible": True, "token": second.token}
            for wrong in ({"seq": "1"}, {"visible": 1}, {"token": 5}):
                body = json.dumps({**word, **wrong}).encode()
                assert post(port, SHOWN_PATH, body)[0] == 400, wrong
            assert status_now(model).live
        finally:
            one.close()
            two.close()


def counting_page(tab: Tab) -> dict[str, bool]:
    """What the viewer answers the page of this tab about the page that counts."""
    connection = tab._http()
    try:
        headers = {TOKEN_HEADER: tab.token} if tab.token else {}
        connection.request("GET", STATE_PATH, headers={**headers, **tab.headers})
        reply = connection.getresponse()
        body = reply.read()
    finally:
        connection.close()
    assert reply.status == 200, reply.status
    return json.loads(body)["counting_page"]


def test_the_viewer_names_its_instance_and_tells_a_page_whether_it_counts() -> None:
    # Of the page that counts: whether there is one, whether it is the page
    # that asks, and whether it asks for pictures.
    none = {"loaded": False, "is_this_page": False, "asks": False}
    another, asking = {**none, "loaded": True}, {"asks": True}
    counts = {**another, "is_this_page": True}
    with showing(Camera(hz=60.0), fps=30.0, this_machine=only_127_0_0_1) as (
        live,
        port,
    ):
        model = live.model
        until(lambda: live_state(port)["available"], "the live picture")
        # The instance is one start of the viewer: a name in the answer about
        # the live picture and in every page. No page was loaded: none counts.
        state = live_state(port)
        instance = state["instance"]
        assert re.fullmatch(r"[0-9a-f]{16}", instance)
        assert state["counting_page"] == none
        first, second, watcher = Tab(port), Tab(port), Elsewhere(port)
        for body in (first.page(), watcher.page()):
            assert VIEWER_INSTANCE.findall(body) == [instance.encode()]
        # A page loaded on the workstation counts, and has asked for no picture.
        first.load()
        assert counting_page(first) == counts
        # Not for a request without its view token, or from another machine.
        assert live_state(port)["counting_page"] == another
        watcher.token = first.token
        assert counting_page(watcher) == another
        first.show()
        until(lambda: status_now(model).live, "the first page's live view")
        assert counting_page(first) == {**counts, **asking}
        # A newer load: the first page is refused, by this viewer instance,
        # and is told so when it asks about the live picture.
        second.load()
        until(lambda: first.replaced, "the first page's 403")
        first.freeze()
        one = first._http()
        try:
            one.request(
                "GET", f"{PICTURE_PATH}?after=0", headers={TOKEN_HEADER: first.token}
            )
            reply = one.getresponse()
            assert (reply.status, reply.read()) == (403, b"")
            assert reply.getheader(INSTANCE_HEADER) == instance
            assert counting_page(first) == another
            assert counting_page(second) == counts
            second.show()
            until(lambda: status_now(model).live, "the second page's live view")
            assert counting_page(first) == {**another, **asking}
            # The replaced page goes on to watch, as the block's script does:
            # pictures that count for nothing, and questions with its old
            # view token. Nothing of it reaches the model.
            seq = 0
            for _ in range(10):
                one.request(
                    "GET",
                    f"{PICTURE_PATH}?after={seq}&watch=1",
                    headers={TOKEN_HEADER: first.token},
                )
                reply = one.getresponse()
                seq = int(reply.getheader("X-Picture-Seq"))
                assert reply.status == 200 and number_of(reply.read()) > 0
                assert first.shown(one, seq) == 403
                assert first.shown(one, seq, visible=False) == 403
                assert counting_page(first)["is_this_page"] is False
                assert model.is_current_page(second.token)
            counts_on(second)
            assert status_now(model).live and second.refused == []
            # The page that counts is closed: the viewer says that it asks no
            # more, and the replaced page still counts for nothing.
            second.freeze()
            began = time.monotonic()
            until(lambda: not counting_page(first)["asks"], "the viewer's word")
            assert time.monotonic() - began < live_module.FETCH_WATCH_S + 1.5
            assert counting_page(first) == another
            assert first.fetch(one)[0] == 403 and first.shown(one, seq) == 403
            time.sleep(live_settings.ACK_TIMEOUT_S + 0.1)
            assert status_now(model) == NOT_LIVE
            assert model.is_current_page(second.token)
            # Only a load of the page makes it the one that counts.
            assert first.load() != second.token
            status, seq, _ = first.fetch(one, after=0)
            assert status == 200 and first.shown(one, seq) == 204
            assert status_now(model).live and counting_page(first)["is_this_page"]
            assert counting_page(second)["is_this_page"] is False
        finally:
            one.close()
            second.close()
        old_token = first.token
    # Another start of the viewer is another instance: it refuses the view
    # token of the one before and names itself.
    with showing(Camera(hz=60.0)) as (live, port):
        until(lambda: live_state(port)["available"], "the live picture")
        later = live_state(port)["instance"]
        assert re.fullmatch(r"[0-9a-f]{16}", later) and later != instance
        status, _, body = get(port, PICTURE_PATH, **{TOKEN_HEADER: old_token})
        assert (status, body) == (403, b"")
        tab = Tab(port)
        tab.token = old_token
        connection = tab._http()
        try:
            connection.request(
                "GET", f"{PICTURE_PATH}?after=0", headers={TOKEN_HEADER: old_token}
            )
            reply = connection.getresponse()
            reply.read()
            assert reply.status == 403 and reply.getheader(INSTANCE_HEADER) == later
        finally:
            connection.close()
        # No page of this instance has a view token: a page that was open
        # across the restart may load itself for one.
        assert counting_page(tab) == none
        Tab(port).load()
        assert counting_page(tab) == another
        assert status_now(live.model) == NOT_LIVE


class Connection:
    """A stand-in for a request's connection: the two names a socket gives."""

    def __init__(self, client: Any, arrived_on: Any) -> None:
        self._names = client, arrived_on

    def getpeername(self) -> Any:
        if isinstance(self._names[0], OSError):
            raise self._names[0]
        return self._names[0]

    def getsockname(self) -> Any:
        return self._names[1]


@pytest.mark.parametrize(
    ("client", "arrived_on", "here"),
    [
        # A loopback address, whatever address the viewer listens on.
        (("127.0.0.1", 50000), ("127.0.0.1", 8000), True),
        (("127.0.0.1", 50000), ("0.0.0.0", 8000), True),
        (("127.8.9.1", 50000), ("127.0.0.1", 8000), True),
        (("::1", 50000, 0, 0), ("::1", 8000, 0, 0), True),
        (("::ffff:127.0.0.1", 50000, 0, 0), ("::ffff:127.0.0.1", 8000, 0, 0), True),
        # This machine's own address: the one the request arrived on.
        (("192.168.1.20", 50000), ("192.168.1.20", 8000), True),
        (
            ("::ffff:192.168.1.20", 50000, 0, 0),
            ("::ffff:192.168.1.20", 8000, 0, 0),
            True,
        ),
        (("fe80::1%eth0", 50000, 0, 2), ("fe80::1%eth0", 8000, 0, 2), True),
        # Any other machine.
        (("192.168.1.21", 50000), ("192.168.1.20", 8000), False),
        (("10.0.0.7", 50000), ("192.168.1.20", 8000), False),
        (
            ("::ffff:192.168.1.21", 50000, 0, 0),
            ("::ffff:192.168.1.20", 8000, 0, 0),
            False,
        ),
        (("fe80::2%eth0", 50000, 0, 2), ("fe80::1%eth0", 8000, 0, 2), False),
        (("2001:db8::7", 50000, 0, 0), ("2001:db8::1", 8000, 0, 0), False),
        # No IP address to read: not this machine.
        ("", "/tmp/viewer.sock", False),
        (("not an address", 1), ("127.0.0.1", 8000), False),
        (OSError("the client is gone"), ("127.0.0.1", 8000), False),
    ],
)
def test_a_browser_is_on_this_machine_by_the_addresses_of_its_connection(
    client: Any, arrived_on: Any, here: bool
) -> None:
    assert live_module.on_this_machine(Connection(client, arrived_on)) is here


def test_the_rule_is_asked_of_the_connection_and_no_header_changes_it() -> None:
    asked: list[Any] = []

    def nobody_is_here(connection: Any) -> bool:
        asked.append(connection)
        return False

    claims = {
        "X-Forwarded-For": "127.0.0.1",
        "Forwarded": "for=127.0.0.1",
        "X-Real-IP": "127.0.0.1",
        "Host": "127.0.0.1",
    }
    with showing(Camera(), this_machine=nobody_is_here) as (live, port):
        until(lambda: live.state()["available"], "the live picture")
        for headers in ({}, claims):
            status, _, body = get(port, "/", **headers)
            assert status == 200 and body.count(NO_VIEW_TOKEN) == 1, headers
            assert VIEW_TOKEN.search(body) is None
    # The rule was handed the request's own socket.
    assert asked and all(isinstance(one, socket.socket) for one in asked)
    # Without a rule of the test's: on_this_machine(), for which a request
    # from 127.0.0.1 is one of this machine.
    with showing(Camera()) as (live, port):
        assert VIEW_TOKEN.search(get(port, "/")[2]) is not None


def counts_on(tab: Tab) -> None:
    """The viewer counts one more picture this tab draws and acknowledges."""
    taken = tab.taken
    until(lambda: tab.taken > taken, "the tab's next counted picture")


def test_a_page_opened_from_another_machine_only_watches() -> None:
    with showing(Camera(hz=60.0), fps=30.0, this_machine=only_127_0_0_1) as (
        live,
        port,
    ):
        model = live.model
        until(lambda: live_state(port)["available"], "the live picture")
        # The operator's page, opened on the workstation, draws and acknowledges.
        operator = Tab(port)
        operator.show()
        until(lambda: status_now(model).live, "the operator's live view")
        watcher = Elsewhere(port)
        other = watcher._http()
        try:
            # The page for another machine is the page with the block, without
            # a view token: the operator's page is still the page loaded last.
            body = watcher.page()
            at = PAGE.rindex(b"</body>")
            assert body == PAGE[:at] + block_in(body, b"") + PAGE[at:]
            watcher.load()
            assert model.is_current_page(operator.token)
            assert status_now(model).live and not operator.replaced
            # It sees pictures: the camera's newest, then a newer one.
            status, seq, jpeg = watcher.watch(other)
            assert status == 200 and seq > 0
            first = number_of(jpeg)
            status, newer, jpeg = watcher.watch(other, after=seq)
            assert status == 200 and newer > seq and number_of(jpeg) > first > 0
            # Its word about a picture counts for nothing, and neither does a
            # picture asked for as the operator's page asks.
            assert watcher.fetch(other)[0] == 403
            assert watcher.shown(other, newer) == 403
            assert watcher.shown(other, newer, visible=False) == 403
            # Nor with the operator's view token, should it get hold of it: it
            # cannot end the operator's live view, and gets no counted picture.
            watcher.token = operator.token
            counts_on(operator)
            assert watcher.shown(other, operator.seq, visible=False) == 403
            assert watcher.shown(other, operator.seq) == 403
            assert watcher.fetch(other, after=0)[0] == 403
            assert status_now(model).live and model.is_current_page(operator.token)
            # The stream is there to watch as well.
            stream = Stream(port, source=OTHER_MACHINE)
            try:
                assert stream.status == 200 and number_of(stream.picture()) > 0
            finally:
                stream.close()
        finally:
            other.close()
        # Watchers come and go: the operator's page keeps counting.
        for _ in range(5):
            passing = Elsewhere(port)
            passing.load()
            connection = passing._http()
            try:
                assert passing.watch(connection)[0] == 200
            finally:
                connection.close()
            counts_on(operator)
            status = status_now(model)
            assert status.live and status.age_s < 0.5
        assert operator.refused == [] and operator.failed == []
        assert not operator.replaced
        # A second page opened on the workstation still replaces the first.
        second = Tab(port)
        second.load()
        until(lambda: operator.replaced, "the first page's 403")
        assert status_now(model) == NOT_LIVE
        second.show()
        try:
            until(lambda: status_now(model).live, "the second page's live view")
        finally:
            second.close()
            operator.close()


def test_watched_pictures_are_never_counted() -> None:
    with showing(Camera(hz=60.0), fps=30.0, this_machine=only_127_0_0_1) as (
        live,
        port,
    ):
        model = live.model
        until(lambda: live_state(port)["available"], "the live picture")
        # No page on the workstation: watchers alone never make the view live.
        watcher = Elsewhere(port)
        watcher.load()
        connection = watcher._http()
        try:
            seq = 0
            for _ in range(10):
                status, seq, _ = watcher.watch(connection, after=seq)
                assert status == 200
                assert watcher.shown(connection, seq) == 403
                assert status_now(model) == NOT_LIVE
            # The camera is read at the asked rate for them.
            assert live_state(port)["fetching"] is True
        finally:
            connection.close()
        # On the workstation too a watched picture is not one the page was
        # served: its acknowledgement counts for nothing.
        local = Tab(port)
        local.load()
        connection = local._http()
        try:
            connection.request("GET", f"{PICTURE_PATH}?after=0&watch=1")
            reply = connection.getresponse()
            watched = int(reply.getheader("X-Picture-Seq"))
            assert reply.status == 200 and number_of(reply.read()) > 0
            assert local.shown(connection, watched) == 403
            assert status_now(model) == NOT_LIVE
            # The same page's counted picture does count.
            _, seq, _ = local.fetch(connection, after=watched)
            assert local.shown(connection, seq) == 204 and status_now(model).live
        finally:
            connection.close()


def test_the_picture_and_the_acknowledgement_need_the_viewers_token_too() -> None:
    token = "t" * 43
    auth = {"Authorization": f"Bearer {token}"}
    with showing(Camera(), token=token) as (live, port):
        until(lambda: live.state()["available"], "the live picture")
        allowed = Tab(port, **auth)
        allowed.load()
        open_tab = Tab(port)
        open_tab.token = allowed.token  # the view token alone is not enough
        connection = open_tab._http()
        try:
            # As the viewer's own paths answer without its token.
            refused = get(port, STATE_PATH)
            assert refused[0] == 401 and b"private access link" in refused[2]
            assert get(port, PICTURE_PATH, **{TOKEN_HEADER: allowed.token}) == refused
            status, seq, body = open_tab.fetch(connection)
            assert (status, seq) == (401, None) and body == refused[2]
        finally:
            connection.close()
        connection = open_tab._http()
        try:
            assert open_tab.shown(connection, 1) == 401
        finally:
            connection.close()
        assert status_now(live.model) == NOT_LIVE
        connection = allowed._http()
        try:
            status, seq, _ = allowed.fetch(connection, after=0)
            assert status == 200 and allowed.shown(connection, seq) == 204
        finally:
            connection.close()
        assert status_now(live.model).live


def post(port: int, path: str, body: bytes, **headers: str) -> tuple[int, bytes]:
    """(status code, body) of one POST."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT_S)
    try:
        connection.request("POST", path, body, headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_an_acknowledgement_that_is_not_one_is_refused() -> None:
    with showing(Camera()) as (live, port):
        until(lambda: live.state()["available"], "the live picture")
        tab = Tab(port)
        tab.load()
        connection = tab._http()
        try:
            _, seq, _ = tab.fetch(connection, after=0)
        finally:
            connection.close()
        word = json.dumps({"seq": seq, "visible": True, "token": tab.token}).encode()
        # From a page of another site, in a browser that sends the viewer's
        # cookie along: refused. From the viewer's own page: taken.
        host = f"127.0.0.1:{port}"
        assert post(port, SHOWN_PATH, word, Origin="http://elsewhere.example")[0] == 403
        assert status_now(live.model) == NOT_LIVE
        for body in (b"", b"not JSON", b"[1]", b"\xff\xfe", word[:-1]):
            code, _ = post(port, SHOWN_PATH, body)
            assert code in (400, 403), body
        assert post(port, SHOWN_PATH, b"x" * 5000)[0] == 413
        assert status_now(live.model) == NOT_LIVE
        assert post(port, SHOWN_PATH, word, Origin=f"http://{host}")[0] == 204
        assert status_now(live.model).live
        # The viewer alone takes no POST: every other path answers as it does.
        with serving(PlainViewer) as alone:
            for path in ("/", "/api/runs", "/api/live/other", STATE_PATH):
                mine, theirs = post(port, path, word), post(alone, path, word)
                assert mine == theirs and mine[0] == 501, path


def test_a_live_paths_connection_stays_open_and_the_viewers_own_answer_ends_it() -> (
    None
):
    with showing(Camera(hz=60.0), fps=30.0) as (_, port):
        until(lambda: live_state(port)["available"], "the live picture")
        tab = Tab(port)
        tab.load()
        connection = tab._http()
        try:
            _, seq, _ = tab.fetch(connection, after=0)
            sock = connection.sock
            for _ in range(5):
                status, seq, _ = tab.fetch(connection, after=seq)
                assert status == 200 and tab.shown(connection, seq) == 204
            connection.request("GET", STATE_PATH)
            assert json.loads(connection.getresponse().read())["available"] is True
            assert connection.sock is sock  # one connection for all of them
            # The viewer's own path on that connection: its own answer, as on
            # a connection of its own, and the connection ends with it.
            connection.request("GET", "/api/runs")
            reply = connection.getresponse()
            assert (reply.version, reply.status, reply.read()) == (10, 200, b"[]")
            assert reply.will_close
        finally:
            connection.close()
        # A page that asks for one closed connection gets it.
        connection = tab._http()
        try:
            connection.request(
                "GET", f"{PICTURE_PATH}?after=0", headers={TOKEN_HEADER: tab.token}
            )
            connection.getresponse().read()
            assert connection.sock is not None
            connection.request("GET", STATE_PATH, headers={"Connection": "close"})
            reply = connection.getresponse()
            reply.read()
            assert reply.getheader("Connection") == "close" and reply.will_close
        finally:
            connection.close()


def test_the_camera_is_read_at_the_asked_rate_while_a_page_asks_for_pictures() -> None:
    camera = Camera(hz=60.0)
    with showing(camera, fps=30.0) as (_, port):
        until(lambda: live_state(port)["available"], "the live picture")
        time.sleep(1.2)
        idle = live_state(port)
        assert idle["fps_measured"] <= 2.0 and idle["fetching"] is False
        tab = Tab(port)
        tab.show()
        try:
            time.sleep(0.5)  # the first ones: the rate settles
            before = len(tab.drawn)
            time.sleep(1.0)
            drawn = tab.drawn[before:]
            state = live_state(port)
        finally:
            tab.close()
        # Every picture is another one, newer than the one before; no more of
        # them than were asked for, and no stream was open for them. This tab
        # shares the test's process with the viewer and takes fewer.
        assert drawn == sorted(set(drawn)), drawn
        assert 8 <= len(drawn) <= 31, drawn
        assert state["fetching"] is True and state["clients"] == 0
        assert 20 <= state["fps_measured"] <= 31
        assert tab.failed == [] and tab.refused == [] and not tab.replaced
        # The page asks no more: back to one picture a second.
        time.sleep(live_module.FETCH_WATCH_S + 1.2)
        asked = camera.asked
        time.sleep(2.0)
        assert 1 <= camera.asked - asked <= 3
        assert live_state(port)["fetching"] is False


@pytest.mark.parametrize("idle_s", [2.2, 2.5])
def test_the_first_pictures_for_a_page_come_at_the_asked_rate_at_once(
    idle_s: float,
) -> None:
    with showing(Camera(hz=60.0), fps=30.0) as (live, port):
        until(lambda: live_state(port)["available"], "the live picture")
        tab = Tab(port)
        tab.load()
        # No page asks: one picture a second, and the next one is due up to a
        # second from now.
        time.sleep(idle_s)
        connection = tab._http()
        try:
            began = time.monotonic()
            at, seq = [], 0
            while len(at) < 8:
                status, seq, _ = tab.fetch(connection, after=seq)
                assert status == 200 and tab.shown(connection, seq) == 204
                at.append(time.monotonic() - began)
            age = status_now(live.model).age_s
        finally:
            connection.close()
        # The page's first request starts the period: no picture waits out the
        # rest of the idle second, and the eighth one is fresh.
        gaps = np.diff([0.0, *at])
        assert max(gaps) < 0.25, at
        assert at[-1] < 0.6 and age < 0.25, (at, age)


def test_a_request_for_a_picture_waits_for_a_newer_one_and_gives_up_after_a_second() -> (
    None
):
    camera = Camera(hz=60.0)
    with showing(camera, fps=30.0) as (_, port):
        until(lambda: live_state(port)["available"], "the live picture")
        tab = Tab(port)
        tab.load()
        connection = tab._http()
        try:
            _, seq, _ = tab.fetch(connection, after=0)
            camera.stopped = True
            # The pictures under way arrive; then there is no newer one.
            for _ in range(5):
                began = time.monotonic()
                status, newer, body = tab.fetch(connection, after=seq)
                if status == 204:
                    break
                seq = newer
            waited = time.monotonic() - began
            assert (status, body) == (204, b"")
            assert live_module.PICTURE_WAIT_S <= waited < live_module.PICTURE_WAIT_S + 1
        finally:
            connection.close()


def test_what_the_robot_is_told_is_in_the_viewers_answer_about_the_live_picture() -> (
    None
):
    told = {"attached": True, "why": "", "live": True, "age_s": 0.08}
    with showing(Camera(), view=lambda: told) as (_, port):
        assert live_state(port)["view"] == told
    # A viewer that reports to nobody says so.
    with showing(Camera()) as (_, port):
        assert live_state(port)["view"] == {
            "attached": False,
            "why": "this viewer does not report the live view",
            "live": None,
            "age_s": None,
        }


# The page's block


NODE = shutil.which("node")
# Runs the block's script on a stand-in for the page: its picture in its
# column, its "Follow live" button, its run selector and its two switches. The
# viewer is the harness: it answers the script's requests when a step says so.
# The browser is the harness too: it says how much of the live picture is
# inside the window, through an IntersectionObserver or, in the scenario
# without one, through the picture's box. Prints one JSON line.
PAGE_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const scenario = process.argv[3];
const block = fs.readFileSync(process.argv[2], 'utf8');
// A page for another machine holds no view token. The viewer instance that
// served the page is 'aaaa'.
const script = block.slice(block.indexOf('<script>') + 8, block.lastIndexOf('</script>'))
  .replace('__VIEW_TOKEN__', scenario.startsWith('watch') ? '' : 'token-of-this-load')
  .replace('__VIEWER_INSTANCE__', 'aaaa');
const listeners = {}, windowListeners = {}, intervals = [], timeouts = [];
// The tab's history entry, and how often the page loaded itself again. In the
// scenario 'loaded again' the page has done so once, for the instance 'bbbb'.
const history = {
  state: scenario === 'loaded again' ? { liveInstance: 'bbbb' } : null,
  replaceState(state) { this.state = state; },
};
let reloads = 0;
// Every request, the picture requests that wait for their answer, the tokens
// they carried, the acknowledgements, and the pictures drawn.
const asked = [], waiting = [], tokens = [], posts = [], drawn = [];
// The view token each question about the live picture carried.
const stated = [];
let clock = 1000000;
// The share of the live picture's area inside the 1000 x 800 window while it
// is on the page, and what the observer was last told.
let share = 1, reported = null, observing = null;
function element(id) {
  const made = {
    id, style: { display: '' }, children: [], heard: {}, disabled: false, textContent: '',
    classes: new Set(), width: 0, height: 0,
    addEventListener(kind, work) { (this.heard[kind] ||= []).push(work); },
    append(...nodes) { this.children.push(...nodes); },
    insertBefore(node, before) {
      const at = this.children.indexOf(before);
      this.children.splice(at < 0 ? this.children.length : at, 0, node);
    },
    get firstChild() { return this.children[0] || null; },
    getContext() { return { drawImage(bitmap) { drawn.push(bitmap.seq); } }; },
    // A picture 400 x 300 whose lower part is below the window's edge.
    getBoundingClientRect() {
      if (this.style.display !== 'block') return { left: 0, right: 0, top: 0, bottom: 0, width: 0, height: 0 };
      const top = 800 - 300 * share;
      return { left: 600, right: 1000, top, bottom: top + 300, width: 400, height: 300 };
    },
    click() {
      if (this.onclick) this.onclick();
      for (const work of this.heard.click || []) work();
      for (const work of listeners.click || []) work();
    },
  };
  made.classList = { toggle(name, on) { on ? made.classes.add(name) : made.classes.delete(name); } };
  return made;
}
// The page's right column: its refusal panel, then the panel with its picture.
const column = element('column'), panel = element('panel');
column.children.push(element('refusals'), panel);
const ids = { photo: element('photo'), live: element('live'), run: element('run') };
ids.photo.parentNode = panel;
ids.photo.closest = name => (name === 'aside' ? column : null);
panel.children.push(ids.photo);
ids.run.value = 'latest';
if (scenario === 'no photo') delete ids.photo;
const made = [];
let answer = { available: false, reason: 'the camera source is being opened' };
const globals = {
  document: {
    hidden: false,
    getElementById: id => ids[id] || null,
    createElement(tag) { const node = element(''); node.tag = tag; made.push(node); return node; },
    addEventListener(kind, work) { (listeners[kind] ||= []).push(work); },
  },
  window: {
    innerWidth: 1000, innerHeight: 800,
    addEventListener(kind, work) { (windowListeners[kind] ||= []).push(work); },
  },
  history,
  location: { reload() { reloads += 1; } },
  setInterval(work) { intervals.push(work); },
  setTimeout(work) { timeouts.push(work); },
  fetch(url, options) {
    asked.push(url);
    if (url.startsWith('/api/live.jpg')) {
      tokens.push((options.headers || {})['X-View-Token'] ?? null);
      return new Promise((resolve, reject) => waiting.push({ url, resolve, reject }));
    }
    if (url === '/api/live.json') stated.push((options.headers || {})['X-View-Token'] ?? null);
    if (url === '/api/live/shown') {
      posts.push([options.method, JSON.parse(options.body)]);
      return Promise.resolve({ ok: true, status: 204 });
    }
    if (answer === null) return Promise.reject(Error('refused'));
    return Promise.resolve({ ok: true, status: 200, json: async () => answer });
  },
  createImageBitmap: async blob => ({ width: 960, height: 744, seq: blob.seq, close() {} }),
  Date: { now: () => clock },
};
if (scenario !== 'page without observer') {
  globals.IntersectionObserver = function (callback, options) {
    observing = { callback, options, nodes: [] };
    this.observe = node => observing.nodes.push(node.id);
  };
}
const context = vm.createContext(globals);
// The page's own script: its switches, and what "Follow live" does.
const own = scenario === 'no switches' ? '' : 'let follow = true, playing = false;';
vm.runInContext(own + `
  function pageFollows() { follow = true; playing = false; ids.run.value = 'latest'; }
  function pageSwitches(a, b) { follow = a; playing = b; }
`, Object.assign(context, { ids }));
if (ids.live) ids.live.onclick = () => vm.runInContext('pageFollows()', context);
vm.runInContext(script, context);

// The viewer answers every waiting request: with picture `seq`, or a status code.
function give(seq) {
  for (const request of waiting.splice(0)) {
    request.resolve({ status: 200, headers: { get: () => String(seq) }, blob: async () => ({ seq }) });
  }
}
function reply(status) {
  for (const request of waiting.splice(0)) request.resolve({ status, headers: { get: () => null } });
}
// The viewer instance `instance` refuses every waiting request: 403.
function refuse(instance) {
  const get = name => (name === 'X-Viewer-Instance' ? instance : null);
  for (const request of waiting.splice(0)) request.resolve({ status: 403, headers: { get } });
}
// Every waiting request fails: the viewer cannot be reached.
function fail() {
  for (const request of waiting.splice(0)) request.reject(Error('refused'));
}
const settle = () => new Promise(done => setImmediate(done));
// The browser has drawn the page: an observer hears of a change of the share
// of the live picture that is inside the window. Off the page it has none.
function render() {
  const now = made[3].style.display === 'block' ? share : 0;
  if (!observing || now === reported) return;
  reported = now;
  observing.callback([{ isIntersecting: now > 0, intersectionRatio: now }]);
}
async function turn() {
  // A quarter of a second: the 0 ms timers, then the script's two intervals.
  clock += 250;
  for (const work of timeouts.splice(0)) work();
  for (const work of intervals) work();
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 8; j++) await settle();
    render();
  }
  // The quarter-second timer once more: what the page shows at the end.
  intervals[0]();
}
// The page is scrolled: this share of the live picture is inside the window.
function scroll(to) {
  share = to;
  for (const work of windowListeners.scroll || []) work();
}
function seen() {
  const [bar, button, note, picture, why, box] = made;
  return {
    button: button.textContent, disabled: button.disabled, note: note.textContent,
    why: why.textContent, active: button.classes.has('active'), tag: picture.tag,
    live_display: picture.style.display, photo_display: ids.photo.style.display,
    column: column.children.map(node => node.id), box: box.children.map(node => node.id),
    bar: bar.children.map(node => node.id), panel: panel.children.map(node => node.id),
    style: { ...picture.style }, size: [picture.width, picture.height],
    drawn: [...drawn], posts: [...posts], waiting: waiting.map(request => request.url),
    // The note's colour, how often the page loaded itself again, and what
    // the tab's history entry keeps.
    color: note.style.color || '', reloads, kept: history.state && { ...history.state },
  };
}
(async () => {
  const out = [];
  const step = async (name, work) => { if (work) work(); await turn(); out.push([name, seen()]); };
  const visibility = hidden => {
    context.document.hidden = hidden;
    for (const work of listeners.visibilitychange) work();
  };
  const live = { attached: true, why: '', live: true, age_s: 0.0834 };
  const done = () => console.log(JSON.stringify({
    steps: out, asked, tokens, stated, observed: observing && [observing.nodes, observing.options],
    window_listeners: Object.keys(windowListeners).sort(),
  }));
  if (scenario === 'no photo' || scenario === 'no switches') {
    await settle();
    console.log(JSON.stringify({ made: made.length, asked: asked.length, timers: intervals.length }));
    return;
  }
  out.push(['at first', seen()]);
  if (scenario === 'watch') {
    // A page opened from another machine, while the operator's page is live.
    answer = { available: true, reason: '', view: live };
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    await step('the picture is scrolled out of the window', () => scroll(0.2));
    await step('picture 8 arrives', () => give(8));
    await step('Map picture is pressed', () => made[1].click());
    await step('Live picture is pressed', () => made[1].click());
    await step('the tab is hidden', () => visibility(true));
    await step('the tab is shown', () => visibility(false));
    await step('a 403 ends nothing', () => reply(403));
    await step('and the page asks again');
    answer = { available: false, reason: 'the camera source ended: gone' };
    await step('the camera is gone');
    done();
    return;
  }
  const pass = async turns => { for (let i = 0; i < turns; i++) await turn(); };
  if (scenario === 'watch restart') {
    // A page opened from another machine, across a restart of the viewer.
    const elsewhere = { loaded: true, is_this_page: false, asks: true };
    answer = { available: true, reason: '', view: live, instance: 'aaaa', counting_page: elsewhere };
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    answer = { ...answer, instance: 'bbbb' };
    await step('the new viewer answers');
    await pass(8);
    out.push(['two seconds after that', seen()]);
    done();
    return;
  }
  // What the viewer instance that served the page says while the page counts;
  // what a later instance says while none of its pages has a view token, and
  // once one has.
  const counts = { loaded: true, is_this_page: true, asks: true };
  const vacant = { loaded: false, is_this_page: false, asks: false };
  const taken = { loaded: true, is_this_page: false, asks: false };
  const showing = { available: true, reason: '', view: live, instance: 'aaaa', counting_page: counts };
  if (scenario === 'outage') {
    // The viewer gives no answer for a while, and is the same one after it.
    answer = showing;
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    answer = null;
    await step('the viewer is gone', fail);
    await pass(8);
    out.push(['two seconds later', seen()]);
    await step('the tab is hidden', () => visibility(true));
    await step('the tab is shown', () => visibility(false));
    answer = showing;
    await step('the viewer answers again');
    await step('picture 8 arrives', () => give(8));
    await pass(9);
    out.push(['no picture came for two seconds', seen()]);
    await step('picture 9 arrives', () => give(9));
    done();
    return;
  }
  if (scenario === 'restart' || scenario === 'restart refused') {
    // The viewer ends and another instance, 'bbbb', answers five seconds later.
    answer = showing;
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    answer = null;
    await step('the viewer is gone', fail);
    await pass(20);
    out.push(['down for five seconds', seen()]);
    answer = { available: false, reason: 'the camera source is being opened', instance: 'bbbb', counting_page: vacant };
    if (scenario === 'restart refused') {
      // It refuses the old view token before the page's next question about
      // the live picture is due: the page asks at once.
      refuse('bbbb');
      for (let i = 0; i < 8; i++) await settle();
      out.push(['the new viewer refuses the view token', seen()]);
    }
    await step('the new viewer answers');
    await pass(8);
    out.push(['two seconds after that', seen()]);
    done();
    return;
  }
  if (scenario === 'loaded again') {
    // The page that came out of a load for 'bbbb' is still not of that instance.
    answer = { ...showing, instance: 'bbbb' };
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    await pass(4);
    out.push(['a second later', seen()]);
    answer = { ...answer, instance: 'cccc', counting_page: vacant };
    await step('the viewer is restarted once more');
    done();
    return;
  }
  if (scenario === 'restart taken') {
    // A page of the new viewer has its view token before this page hears of it.
    answer = showing;
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    answer = { ...showing, instance: 'bbbb', counting_page: taken };
    await step('the new viewer answers, and a page of it counts');
    await step('picture 2 arrives to watch', () => give(2));
    answer = { ...answer, counting_page: { ...taken, asks: true } };
    await step('that page asks for pictures');
    answer = { ...answer, instance: 'cccc', counting_page: vacant };
    await pass(4);
    out.push(['the viewer is restarted once more', seen()]);
    done();
    return;
  }
  if (scenario === 'hidden restart') {
    answer = showing;
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    await step('the tab is hidden', () => visibility(true));
    answer = { ...showing, instance: 'bbbb', counting_page: vacant };
    await pass(8);
    out.push(['the viewer was restarted meanwhile', seen()]);
    // A minute in which the hidden tab's timers do not run, and at the end
    // of which the new viewer is gone as well.
    const restarted = answer;
    answer = null;
    clock += 60000;
    await step('the tab is shown and the viewer is gone', () => visibility(false));
    answer = restarted;
    await step('the viewer answers');
    done();
    return;
  }
  if (scenario === 'map restart' || scenario === 'replay restart') {
    // The operator left the live picture: for the map picture, or for a
    // replay. The viewer is restarted meanwhile.
    answer = showing;
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    if (scenario === 'map restart') {
      await step('the operator leaves the live picture', () => made[1].click());
    } else {
      await step('the operator leaves the live picture', () => {
        vm.runInContext('pageSwitches(false, false)', context);
        for (const work of listeners.input) work();
      });
    }
    answer = { ...showing, instance: 'bbbb', counting_page: vacant };
    await pass(8);
    out.push(['the viewer was restarted meanwhile', seen()]);
    if (scenario === 'map restart') {
      await step('the operator goes back to the live picture', () => made[1].click());
    } else {
      await step('the operator goes back to the live picture', () => ids.live.click());
    }
    done();
    return;
  }
  if (scenario === 'replaced') {
    // The viewer says that another page counts, while this one asks for no picture.
    answer = showing;
    await step('the live picture shows');
    await step('picture 7 arrives', () => give(7));
    await step('Map picture is pressed', () => made[1].click());
    answer = { ...showing, counting_page: { ...taken, asks: true } };
    await step('another page was loaded on the workstation');
    await step('Live picture is pressed', () => made[1].click());
    await step('picture 8 arrives', () => give(8));
    await step('the picture is scrolled out of the window', () => scroll(0.2));
    await step('the tab is hidden', () => visibility(true));
    await step('the tab is shown', () => visibility(false));
    answer = { ...answer, counting_page: taken };
    await step('the page that counts asks no more');
    // Another viewer instance answers, with its own pictures and with no
    // page that has its view token: no load.
    answer = { ...answer, instance: 'bbbb', counting_page: vacant };
    await pass(8);
    await step('the viewer was restarted', () => give(3));
    done();
    return;
  }
  await step('the viewer has no picture yet');
  answer = { available: true, reason: '', view: { attached: true, why: '', live: false, age_s: null } };
  await step('the live picture shows');
  await step('picture 7 arrives', () => give(7));
  answer.view = { attached: true, why: '', live: true, age_s: 0.0834 };
  await step('picture 8 arrives', () => give(8));
  await step('no newer picture in the wait', () => reply(204));
  await step('the timeline is moved', () => {
    vm.runInContext('pageSwitches(false, false)', context);
    for (const work of listeners.input) work();
  });
  await step('picture 9 arrives late', () => give(9));
  await step('Live picture is pressed', () => made[1].click());
  await step('picture 10 arrives', () => give(10));
  await step('Map picture is pressed', () => made[1].click());
  await step('Follow live is pressed', () => ids.live.click());
  await step('a recording plays', () => vm.runInContext('pageSwitches(false, true)', context));
  await step('Follow live again', () => ids.live.click());
  await step('an older run is picked', () => { ids.run.value = '20261116T120000Z-1'; });
  await step('the newest run again', () => { ids.run.value = 'latest'; });
  await step('the tab is hidden', () => visibility(true));
  await step('the tab is shown', () => visibility(false));
  await step('picture 11 arrives', () => give(11));
  await step('the picture is scrolled half out of the window', () => scroll(0.5));
  await step('picture 12 arrives out of the window', () => give(12));
  await step('three quarters of it are scrolled back', () => scroll(0.75));
  await step('picture 13 arrives', () => give(13));
  answer = { available: false, reason: 'the camera source ended: gone' };
  await step('the camera is gone');
  answer = null;
  await step('the viewer is gone');
  answer = { available: true, reason: '', view: { attached: false, why: 'nobody listens', live: null, age_s: null } };
  await step('the viewer is back');
  await step('the page was loaded again elsewhere', () => reply(403));
  await step('picture 14 arrives to watch', () => give(14));
  answer = { ...answer, counting_page: { is_this_page: false, asks: true } };
  await step('the page that counts asks for pictures');
  await step('Map picture is pressed again', () => made[1].click());
  await step('Live picture is pressed again', () => made[1].click());
  await step('the tab is hidden again', () => visibility(true));
  await step('the tab is shown again', () => visibility(false));
  await step('picture 15 arrives to watch', () => give(15));
  done();
})();
"""


def run_page(tmp_path: Path, scenario: str) -> dict[str, Any]:
    harness = tmp_path / "page_harness.js"
    harness.write_text(PAGE_HARNESS, encoding="utf-8")
    done = subprocess.run(
        [NODE, str(harness), str(live_module.PAGE_BLOCK_FILE), scenario],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


STORED = {"live_display": "none", "photo_display": ""}
LIVE = {"live_display": "block", "photo_display": "none"}
# The colour of the page's note while its picture is not live or does not
# count, and what the note then says.
WARN = "#ffbd69"
UNREACHED = "the viewer cannot be reached; trying again every second"
RESTARTED = "the viewer was restarted: this page loads itself again"
LOADS_LATER = (
    "the viewer was restarted: this page loads itself again once Live picture is "
    "pressed"
)
NOT_LOADED = (
    "watch only: the viewer was restarted; reload this page to make it the live view"
)
ANOTHER_PAGE = (
    "watch only: another page on the workstation is the live view; reload this "
    "page to make it the live view"
)
NO_PAGE = (
    "watch only: no page is the live view now; reload this page to make it the "
    "live view"
)
WATCH_ONLY = "watch only: the live view is the page opened on the workstation"


def page_shows(
    steps: dict[str, Any], step: str, expected: dict[str, str], button: str, why: str
) -> dict[str, Any]:
    """What the page showed after a step: which picture, the button, and why not live."""
    seen = steps[step]
    assert {key: seen[key] for key in expected} == expected, step
    assert (seen["button"], seen["why"]) == (button, why), step
    assert seen["active"] is (expected is LIVE), step
    assert seen["disabled"] is (button == "no live picture"), step
    return seen


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("scenario", ["page", "page without observer"])
def test_the_page_draws_and_acknowledges_pictures_only_while_the_live_picture_shows(
    tmp_path: Path, scenario: str
) -> None:
    told = run_page(tmp_path, scenario)
    steps = dict(told["steps"])
    assert [name for name, _ in told["steps"]] == list(steps)
    # The button with its few words, the note and the live picture, a canvas,
    # are a panel of their own at the top of the column that holds the page's
    # own picture; the page's own panel is as it was.
    first = steps["at first"]
    assert first["column"] == ["livePanel", "refusals", "panel"]
    assert first["box"] == ["", "livePictureNote", "livePhoto"]
    assert first["bar"] == ["livePicture", "livePictureWhy"]
    assert first["panel"] == ["photo"] and first["tag"] == "canvas"
    # The picture keeps its own shape: none of the page's style for its 3D
    # scene, a canvas 580 px high that is dragged.
    assert first["style"] == {
        "display": "none",
        "width": "100%",
        "height": "auto",
        "cursor": "default",
        "background": "none",
        "touchAction": "auto",
    }
    # How much of the picture is inside the window: the browser's observer
    # says so at three quarters of its area, else its box on scroll and resize.
    if scenario == "page":
        assert told["observed"] == [["livePhoto"], {"threshold": [0, 0.75, 1]}]
        assert told["window_listeners"] == []
    else:
        assert told["observed"] is None
        assert told["window_listeners"] == ["resize", "scroll"]
    stored, live = STORED, LIVE
    token = "token-of-this-load"
    see_below = "no live picture: see below"

    def shows(
        step: str, expected: dict[str, str], button: str, why: str = ""
    ) -> dict[str, Any]:
        if expected is stored and button == "Live picture":
            why = why or "Map picture showing"
        return page_shows(steps, step, expected, button, why)

    def word(seq: int, visible: bool) -> list[Any]:
        return ["POST", {"seq": seq, "visible": visible, "token": token}]

    def after(seq: int) -> str:
        return f"{PICTURE_PATH}?after={seq}"

    # Before the viewer's first answer the page is the experiment's own, and
    # asks for no picture.
    first = shows("at first", stored, "no live picture", see_below)
    assert first["note"] == "asking the viewer" and first["waiting"] == []
    waiting = shows(
        "the viewer has no picture yet", stored, "no live picture", see_below
    )
    assert waiting["note"] == "the camera source is being opened"
    assert waiting["waiting"] == [] and waiting["posts"] == []
    # Following live with a live picture: the page asks for the newest picture
    # and draws none yet; the button offers the map's picture. Beside it: what
    # the viewer says the robot is told.
    not_yet = "robot told: no live view"
    up = shows("the live picture shows", live, "Map picture", not_yet)
    assert (up["waiting"], up["drawn"], up["posts"]) == ([after(0)], [], [])
    # A picture is drawn, then acknowledged as visible, and the next is asked
    # for by the number of the one drawn.
    seven = shows("picture 7 arrives", live, "Map picture", not_yet)
    assert seven["drawn"] == [7] and seven["posts"] == [word(7, True)]
    assert seven["waiting"] == [after(7)] and seven["size"] == [960, 744]
    assert seven["note"] == (
        "Live camera picture, 1 a second on this screen. Shown, not stored: no "
        "mask and no markers. The robot is told: no live view."
    )
    eight = shows("picture 8 arrives", live, "Map picture")
    assert eight["drawn"] == [7, 8] and eight["posts"][1:] == [word(8, True)]
    assert eight["note"].startswith("Live camera picture, 2 a second on this screen.")
    assert eight["note"].endswith(" The robot is told: live view, 0.08 s old.")
    # Without a newer picture nothing is drawn or said; the page asks again.
    same = shows("no newer picture in the wait", live, "Map picture")
    assert same["drawn"] == [7, 8] and len(same["posts"]) == 2
    assert same["waiting"] == [after(8)]
    # A replay shows the stored picture: the page says at once that picture 8
    # shows no longer, and a picture that arrives late is neither drawn nor
    # acknowledged.
    moved = shows("the timeline is moved", stored, "Live picture")
    assert moved["posts"][2:] == [word(8, False)]
    assert moved["note"].startswith("The stored picture of the obstacle map.")
    late = shows("picture 9 arrives late", stored, "Live picture")
    assert late["drawn"] == [7, 8] and len(late["posts"]) == 3
    assert late["waiting"] == []
    # "Live picture" goes back to following live: pictures are asked for again.
    again = shows("Live picture is pressed", live, "Map picture")
    assert again["waiting"] == [after(8)] and len(again["posts"]) == 3
    ten = shows("picture 10 arrives", live, "Map picture")
    assert ten["drawn"] == [7, 8, 10] and ten["posts"][3:] == [word(10, True)]
    # "Map picture" keeps the stored picture while the page follows live, and
    # says that the live one shows no longer.
    mapped = shows("Map picture is pressed", stored, "Live picture")
    assert mapped["posts"][4:] == [word(10, False)]
    shows("Follow live is pressed", live, "Map picture")
    played = shows("a recording plays", stored, "Live picture")
    assert played["posts"][5:] == [word(10, False)]
    shows("Follow live again", live, "Map picture")
    older = shows("an older run is picked", stored, "Live picture")
    assert older["posts"][6:] == [word(10, False)]
    shows("the newest run again", live, "Map picture")
    # A hidden tab: the same word, and no picture is drawn while it is hidden.
    hidden = shows("the tab is hidden", stored, "Live picture", "tab hidden")
    assert hidden["posts"][7:] == [word(10, False)]
    back = shows("the tab is shown", live, "Map picture")
    assert len(back["posts"]) == 8 and back["drawn"] == [7, 8, 10]
    eleven = shows("picture 11 arrives", live, "Map picture")
    assert eleven["drawn"] == [7, 8, 10, 11] and eleven["posts"][8:] == [word(11, True)]
    # Only the newest of the page's loops draws: one request waits.
    assert eleven["waiting"] == [after(11)]
    # Half of the live picture is scrolled out of the window: the page says at
    # once that picture 11 shows no longer. The picture stays on the page and
    # is drawn on, and no picture drawn there is acknowledged.
    scroll = "scroll the live picture into view"
    out = shows(
        "the picture is scrolled half out of the window", live, "Map picture", scroll
    )
    assert out["posts"][9:] == [word(11, False)] and out["waiting"] == [after(11)]
    assert out["note"].endswith(" The robot is told: no live view.")
    unseen = shows("picture 12 arrives out of the window", live, "Map picture", scroll)
    assert unseen["drawn"] == [7, 8, 10, 11, 12] and len(unseen["posts"]) == 10
    assert unseen["waiting"] == [after(12)]
    # With three quarters of it inside again the next picture drawn counts.
    inside = shows("three quarters of it are scrolled back", live, "Map picture")
    assert len(inside["posts"]) == 10 and inside["drawn"] == unseen["drawn"]
    thirteen = shows("picture 13 arrives", live, "Map picture")
    assert thirteen["posts"][10:] == [word(13, True)]
    assert thirteen["note"].endswith(" The robot is told: live view, 0.08 s old.")
    # Without a live picture the page is the experiment's own again.
    gone = shows("the camera is gone", stored, "no live picture", see_below)
    assert gone["note"] == "the camera source ended: gone"
    assert gone["posts"][11:] == [word(13, False)]
    # Up to here the note was in the warning colour only without a live picture.
    names = [name for name, _ in told["steps"]]
    for name, seen in told["steps"][: names.index("the viewer is gone")]:
        warned = WARN if seen["button"] == "no live picture" else ""
        assert seen["color"] == warned, name
    # The viewer gives no answer: the page says so, and that it asks again.
    not_reached = "viewer not reached: see below"
    lost = shows("the viewer is gone", stored, "no live picture", not_reached)
    assert (lost["note"], lost["color"]) == (UNREACHED, WARN)
    returned = shows(
        "the viewer is back", live, "Map picture", "robot not told: see below"
    )
    assert returned["note"].endswith(
        " The robot is not told of this live view: nobody listens."
    )
    assert returned["color"] == "" and returned["waiting"][-1] == after(13)
    # A newer load of the page has the valid view token: this page goes on
    # showing the pictures, to watch, says so and tells the viewer nothing
    # more, not even that its picture counts no longer.
    watch_only = "watch only: see below"
    sent = len(returned["posts"])

    def watched(seq: int) -> str:
        return f"{after(seq)}&watch=1"

    def watches(step: str, expected: dict[str, str], button: str, line: str) -> Any:
        seen = page_shows(steps, step, expected, button, watch_only)
        assert seen["note"].startswith(f"{line}. ") and seen["color"] == WARN, step
        assert len(seen["posts"]) == sent and seen["reloads"] == 0, step
        return seen

    replaced = watches(
        "the page was loaded again elsewhere", live, "Map picture", NO_PAGE
    )
    assert replaced["waiting"] == [watched(13)]
    assert replaced["note"] == (
        f"{NO_PAGE}. Live camera picture, 0 a second on this screen. Shown, not "
        "stored: no mask and no markers. The robot is not told of a live view: "
        "nobody listens."
    )
    fourteen = watches("picture 14 arrives to watch", live, "Map picture", NO_PAGE)
    assert fourteen["drawn"][-1] == 14 and fourteen["waiting"] == [watched(14)]
    # The viewer says whether the page that counts asks for pictures.
    watches("the page that counts asks for pictures", live, "Map picture", ANOTHER_PAGE)
    # The button, a hidden tab and a tab that shows again leave it watching.
    watches("Map picture is pressed again", stored, "Live picture", ANOTHER_PAGE)
    watches("Live picture is pressed again", live, "Map picture", ANOTHER_PAGE)
    watches("the tab is hidden again", stored, "Live picture", ANOTHER_PAGE)
    watches("the tab is shown again", live, "Map picture", ANOTHER_PAGE)
    final = watches("picture 15 arrives to watch", live, "Map picture", ANOTHER_PAGE)
    assert final["drawn"] == [7, 8, 10, 11, 12, 13, 14, 15]
    assert final["waiting"] == [watched(15)]
    # Every request for a picture that counts carried this load's view token,
    # and none after the page was replaced did: those are pictures to watch.
    pictures = [url for url in told["asked"] if url.startswith(PICTURE_PATH)]
    assert set(told["tokens"]) == {token, None}
    for url, with_token in zip(pictures, told["tokens"], strict=True):
        assert (with_token is None) is url.endswith("&watch=1"), url
    first_watched = pictures.index(watched(13))
    assert all(url.endswith("&watch=1") for url in pictures[first_watched:])
    assert len(pictures[:first_watched]) >= 8
    # Each question about the live picture carried it too: the viewer answers
    # whether this page counts. Every other request is one of the block's three.
    assert set(told["stated"]) == {token}
    paths = {url.split("?")[0] for url in told["asked"]}
    assert paths == {STATE_PATH, PICTURE_PATH, SHOWN_PATH}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_page_without_a_view_token_shows_the_pictures_and_tells_the_viewer_nothing(
    tmp_path: Path,
) -> None:
    told = run_page(tmp_path, "watch")
    steps = dict(told["steps"])
    watch_only = "watch only: the live view is the page opened on the workstation. "
    live_note = (
        f"{watch_only}Live camera picture, 1 a second on this screen. Shown, not "
        "stored: no mask and no markers. The robot is told: live view, 0.08 s old."
    )

    def shows(step: str, expected: dict[str, str], button: str) -> dict[str, Any]:
        # Beside the button, whatever shows: this page only watches.
        return page_shows(steps, step, expected, button, "watch only: see below")

    def after(seq: int) -> str:
        return f"{PICTURE_PATH}?after={seq}&watch=1"

    first = shows("at first", STORED, "no live picture")
    assert first["note"] == f"{watch_only}asking the viewer"
    up = shows("the live picture shows", LIVE, "Map picture")
    assert up["waiting"] == [after(0)]
    # A picture is drawn, and the page's note says what the operator's page
    # has the robot told.
    seven = shows("picture 7 arrives", LIVE, "Map picture")
    assert seven["drawn"] == [7] and seven["waiting"] == [after(7)]
    assert seven["note"] == live_note
    # Out of the window or in it: the pictures go on.
    shows("the picture is scrolled out of the window", LIVE, "Map picture")
    eight = shows("picture 8 arrives", LIVE, "Map picture")
    assert eight["drawn"] == [7, 8] and eight["waiting"] == [after(8)]
    # The button switches between the two pictures here as well.
    mapped = shows("Map picture is pressed", STORED, "Live picture")
    assert mapped["note"] == (
        f"{watch_only}The stored picture of the obstacle map. Live picture shows "
        "the camera as it is now. The robot is told: live view, 0.08 s old."
    )
    shows("Live picture is pressed", LIVE, "Map picture")
    shows("the tab is hidden", STORED, "Live picture")
    shows("the tab is shown", LIVE, "Map picture")
    # No answer makes this page "replaced": it asks again after a pause.
    refused = shows("a 403 ends nothing", LIVE, "Map picture")
    assert refused["waiting"] == []
    asking = shows("and the page asks again", LIVE, "Map picture")
    assert asking["waiting"] == [after(8)]
    gone = shows("the camera is gone", STORED, "no live picture")
    assert gone["note"] == f"{watch_only}the camera source ended: gone"
    # It told the viewer nothing, and no request carried a view token.
    assert gone["posts"] == [] and gone["drawn"] == [7, 8]
    assert set(told["tokens"]) == {None} and len(told["tokens"]) >= 5
    paths = {url.split("?")[0] for url in told["asked"]}
    assert paths == {STATE_PATH, PICTURE_PATH}
    assert all(url.endswith("&watch=1") for url in told["asked"] if "jpg" in url)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_page_without_a_view_token_says_in_the_warning_colour_that_it_only_watches(
    tmp_path: Path,
) -> None:
    told = run_page(tmp_path, "watch")
    for name, seen in told["steps"]:
        assert seen["note"].startswith(f"{WATCH_ONLY}. "), name
        assert (seen["color"], seen["reloads"]) == (WARN, 0), name
    # It asks the viewer about the live picture without a view token.
    assert told["stated"] and set(told["stated"]) == {None}
    # The viewer's word that another page counts does not make it a replaced
    # page, and across a restart of the viewer it loads itself again, once:
    # that takes no view token from anyone.
    told = run_page(tmp_path, "watch restart")
    steps = dict(told["steps"])
    seven = steps["picture 7 arrives"]
    assert seven["note"].startswith(f"{WATCH_ONLY}. Live camera picture, 1 a second")
    assert (seven["reloads"], seven["kept"]) == (0, None)
    for name in ("the new viewer answers", "two seconds after that"):
        seen = page_shows(steps, name, LIVE, "Map picture", "watch only: see below")
        assert (seen["reloads"], seen["kept"]) == (1, {"liveInstance": "bbbb"}), name
        assert seen["note"].startswith(f"{WATCH_ONLY}. ") and seen["posts"] == []
    assert set(told["tokens"]) == {None} and set(told["stated"]) == {None}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_while_the_viewer_gives_no_answer_the_page_keeps_its_picture_and_says_how_old(
    tmp_path: Path,
) -> None:
    told = run_page(tmp_path, "outage")
    steps = dict(told["steps"])
    not_reached = "viewer not reached: see below"

    def shows(step: str, expected: dict[str, str], button: str, why: str = "") -> Any:
        return page_shows(steps, step, expected, button, why)

    seven = shows("picture 7 arrives", LIVE, "Map picture")
    assert seven["color"] == "" and "opacity" not in seven["style"]
    # No answer: the last picture stays on the page, dimmed, and the note says
    # in the warning colour that the viewer is asked again and how old it is.
    gone = shows("the viewer is gone", LIVE, "Map picture", not_reached)
    assert gone["note"] == f"{UNREACHED}; this picture is 0 s old"
    assert (gone["color"], gone["style"]["opacity"]) == (WARN, "0.45")
    later = shows("two seconds later", LIVE, "Map picture", not_reached)
    assert later["note"] == f"{UNREACHED}; this picture is 2 s old"
    assert (later["color"], later["style"]["opacity"]) == (WARN, "0.45")
    # It asked for the next picture again, with its view token.
    assert later["waiting"] == [f"{PICTURE_PATH}?after=7"]
    assert later["drawn"] == [7] and later["posts"] == seven["posts"]
    # A hidden tab shows the page's own picture, as ever; shown again, it has
    # the old picture and its age.
    hidden = shows("the tab is hidden", STORED, "Live picture", not_reached)
    assert (hidden["note"], hidden["color"]) == (UNREACHED, WARN)
    shown = shows("the tab is shown", LIVE, "Map picture", not_reached)
    assert shown["note"] == f"{UNREACHED}; this picture is 3 s old"
    # The same viewer instance answers again: the page counts as before.
    back = shows("the viewer answers again", LIVE, "Map picture")
    assert (back["color"], back["style"]["opacity"]) == ("", "")
    assert back["note"].endswith(" The robot is told: live view, 0.08 s old.")
    eight = shows("picture 8 arrives", LIVE, "Map picture")
    word = {"seq": 8, "visible": True, "token": "token-of-this-load"}
    assert eight["drawn"] == [7, 8] and eight["posts"][-1] == ["POST", word]
    # A picture that no newer one follows for two seconds is called old too,
    # whatever the viewer answers.
    old = shows(
        "no picture came for two seconds",
        LIVE,
        "Map picture",
        "no new picture: see below",
    )
    assert old["note"] == "no new picture from the viewer; this picture is 2 s old"
    assert (old["color"], old["style"]["opacity"]) == (WARN, "0.45")
    nine = shows("picture 9 arrives", LIVE, "Map picture")
    assert (nine["color"], nine["style"]["opacity"]) == ("", "")
    assert nine["note"].startswith("Live camera picture, 1 a second on this screen.")
    # The page never loaded itself again: the viewer was the same one.
    for name, seen in told["steps"]:
        assert (seen["reloads"], seen["kept"]) == (0, None), name
    assert set(told["tokens"]) == {"token-of-this-load"}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("scenario", ["restart", "restart refused"])
def test_a_page_that_hears_another_viewer_instance_loads_itself_again_once(
    tmp_path: Path, scenario: str
) -> None:
    told = run_page(tmp_path, scenario)
    steps = dict(told["steps"])
    acknowledged = steps["picture 7 arrives"]["posts"]
    # While the viewer cannot be reached the page loads nothing.
    down = page_shows(
        steps,
        "down for five seconds",
        LIVE,
        "Map picture",
        "viewer not reached: see below",
    )
    assert down["note"] == f"{UNREACHED}; this picture is 5 s old"
    assert (down["reloads"], down["kept"]) == (0, None)
    names = [name for name, _ in told["steps"]]
    for name, seen in told["steps"][: names.index("down for five seconds")]:
        assert (seen["reloads"], seen["kept"]) == (0, None), name
    # The first answer of another instance about itself: it has given no page
    # a view token. One load, and the instance it was made for is kept in the
    # tab's history entry. The page asks for that answer every second or,
    # sooner, at once on a 403 of another instance for its view token.
    restarted = "viewer restarted: see below"
    heard = "the new viewer answers"
    if scenario == "restart refused":
        heard = "the new viewer refuses the view token"
    first = page_shows(steps, heard, STORED, "no live picture", restarted)
    assert (first["reloads"], first["kept"]) == (1, {"liveInstance": "bbbb"})
    assert first["note"].startswith(f"{RESTARTED}. ") and first["color"] == WARN
    # While it loads, the same instance's answers start no second load, and
    # the page tells the new viewer nothing.
    for name in ("the new viewer answers", "two seconds after that"):
        seen = page_shows(steps, name, STORED, "no live picture", restarted)
        assert seen["note"] == f"{RESTARTED}. the camera source is being opened"
        assert (seen["reloads"], seen["kept"]) == (1, {"liveInstance": "bbbb"}), name
        assert seen["posts"] == acknowledged, name


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_page_loads_itself_again_once_for_each_viewer_instance_and_no_more(
    tmp_path: Path,
) -> None:
    # The page that came out of its load for the instance 'bbbb' was served
    # by another one still: it does not load itself a second time.
    told = run_page(tmp_path, "loaded again")
    steps = dict(told["steps"])
    restarted = "viewer restarted: see below"
    for name in ("the live picture shows", "picture 7 arrives", "a second later"):
        seen = page_shows(steps, name, LIVE, "Map picture", restarted)
        assert (seen["reloads"], seen["kept"]) == (0, {"liveInstance": "bbbb"}), name
        # It shows the pictures to watch and says what to do.
        assert seen["note"].startswith(f"{NOT_LOADED}. Live camera picture, "), name
        assert seen["color"] == WARN and seen["posts"] == [], name
    assert steps["picture 7 arrives"]["drawn"] == [7]
    assert steps["a second later"]["waiting"] == [f"{PICTURE_PATH}?after=7&watch=1"]
    assert set(told["tokens"]) == {None}
    # A further instance: one load for that one.
    again = steps["the viewer is restarted once more"]
    assert (again["reloads"], again["kept"]) == (1, {"liveInstance": "cccc"})
    assert again["note"].startswith(f"{RESTARTED}. ")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_page_loads_itself_only_while_no_page_of_the_new_viewer_has_a_view_token(
    tmp_path: Path,
) -> None:
    # A load that a page makes by itself takes the live view from no page:
    # another page was loaded on the restarted viewer first.
    told = run_page(tmp_path, "restart taken")
    steps = dict(told["steps"])
    acknowledged = steps["picture 7 arrives"]["posts"]

    def watches(step: str, line: str) -> Any:
        seen = page_shows(steps, step, LIVE, "Map picture", "watch only: see below")
        assert seen["note"].startswith(f"{line}. ") and seen["color"] == WARN, step
        assert (seen["reloads"], seen["kept"]) == (0, None), step
        assert seen["posts"] == acknowledged, step
        return seen

    # This page watches from then on, as a page that was replaced.
    watches("the new viewer answers, and a page of it counts", NO_PAGE)
    two = watches("picture 2 arrives to watch", NO_PAGE)
    assert two["drawn"] == [7, 2]
    assert two["waiting"] == [f"{PICTURE_PATH}?after=2&watch=1"]
    watches("that page asks for pictures", ANOTHER_PAGE)
    # And it stays one: a further restart does not make it load itself,
    # though no page of that viewer has a view token.
    watches("the viewer is restarted once more", NO_PAGE)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_hidden_tab_loads_itself_again_only_once_it_shows(tmp_path: Path) -> None:
    told = run_page(tmp_path, "hidden restart")
    steps = dict(told["steps"])
    restarted = "viewer restarted: see below"
    # Hidden, it would take the view token from the page a person looks at.
    hidden = page_shows(
        steps, "the viewer was restarted meanwhile", STORED, "Live picture", restarted
    )
    assert (hidden["reloads"], hidden["kept"]) == (0, None)
    assert hidden["note"].startswith(f"{RESTARTED}. ") and hidden["color"] == WARN
    # It shows a minute later, when that viewer is gone too: what it heard
    # while hidden is old, and a load now would leave no page. It asks first.
    gone = page_shows(
        steps,
        "the tab is shown and the viewer is gone",
        LIVE,
        "Map picture",
        "viewer not reached: see below",
    )
    assert (gone["reloads"], gone["kept"]) == (0, None)
    assert gone["note"].startswith(f"{RESTARTED}. {UNREACHED}; this picture is ")
    # The viewer's answer: one load.
    shown = steps["the viewer answers"]
    assert (shown["reloads"], shown["kept"]) == (1, {"liveInstance": "bbbb"})
    # Hidden or shown, it told the new viewer nothing.
    assert shown["posts"] == steps["the tab is hidden"]["posts"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("scenario", ["map restart", "replay restart"])
def test_a_page_off_the_live_picture_loads_itself_only_once_it_is_back_on_it(
    tmp_path: Path, scenario: str
) -> None:
    told = run_page(tmp_path, scenario)
    steps = dict(told["steps"])
    left = steps["the operator leaves the live picture"]["posts"]
    # A load would end the map picture or the replay and make the view live
    # with nobody at the page: it waits, and the note says for what.
    waits = page_shows(
        steps,
        "the viewer was restarted meanwhile",
        STORED,
        "Live picture",
        "viewer restarted: see below",
    )
    assert (waits["reloads"], waits["kept"]) == (0, None)
    assert waits["note"].startswith(f"{LOADS_LATER}. ") and waits["color"] == WARN
    assert waits["posts"] == left
    # Back on the live picture, by the block's button or by "Follow live":
    # one load, and nothing was told to the new viewer.
    back = steps["the operator goes back to the live picture"]
    assert (back["reloads"], back["kept"]) == (1, {"liveInstance": "bbbb"})
    assert back["posts"] == left


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_page_that_the_viewer_calls_replaced_watches_from_then_on(
    tmp_path: Path,
) -> None:
    told = run_page(tmp_path, "replaced")
    steps = dict(told["steps"])
    mapped = page_shows(
        steps, "Map picture is pressed", STORED, "Live picture", "Map picture showing"
    )
    sent = mapped["posts"]
    assert [word["visible"] for _, word in sent] == [True, False]

    def watches(step: str, expected: dict[str, str], button: str, line: str) -> Any:
        seen = page_shows(steps, step, expected, button, "watch only: see below")
        assert seen["note"].startswith(f"{line}. ") and seen["color"] == WARN, step
        # Not a word to the viewer, and no load of the page.
        assert seen["posts"] == sent and seen["reloads"] == 0, step
        assert seen["kept"] is None, step
        return seen

    def watched(seq: int) -> str:
        return f"{PICTURE_PATH}?after={seq}&watch=1"

    # The page asks for no picture: the viewer's answer about the live
    # picture says that another page counts.
    watches(
        "another page was loaded on the workstation",
        STORED,
        "Live picture",
        ANOTHER_PAGE,
    )
    pressed = watches("Live picture is pressed", LIVE, "Map picture", ANOTHER_PAGE)
    assert pressed["waiting"][-1] == watched(7)
    eight = watches("picture 8 arrives", LIVE, "Map picture", ANOTHER_PAGE)
    assert eight["drawn"] == [7, 8] and eight["waiting"] == [watched(8)]
    assert eight["note"].endswith(" The robot is told: live view, 0.08 s old.")
    # Out of the window, hidden and shown again: it watches, and counts never.
    watches(
        "the picture is scrolled out of the window", LIVE, "Map picture", ANOTHER_PAGE
    )
    watches("the tab is hidden", STORED, "Live picture", ANOTHER_PAGE)
    watches("the tab is shown", LIVE, "Map picture", ANOTHER_PAGE)
    # The page that counts asks for no more pictures: it was closed or hidden.
    watches("the page that counts asks no more", LIVE, "Map picture", NO_PAGE)
    # A restart of the viewer does not make a replaced page load itself: it
    # would take the view token. It watches the new instance's pictures.
    after_restart = watches("the viewer was restarted", LIVE, "Map picture", NO_PAGE)
    assert after_restart["drawn"] == [7, 8, 3]
    assert after_restart["waiting"] == [watched(3)]
    # Only the requests from before it was replaced carried the view token.
    pictures = [url for url in told["asked"] if url.startswith(PICTURE_PATH)]
    for url, with_token in zip(pictures, told["tokens"], strict=True):
        assert (with_token is None) is url.endswith("&watch=1"), url


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("scenario", ["no photo", "no switches"])
def test_on_a_page_without_what_it_needs_the_block_does_nothing(
    tmp_path: Path, scenario: str
) -> None:
    assert run_page(tmp_path, scenario) == {"made": 0, "asked": 0, "timers": 0}


def test_the_block_is_one_script_that_names_the_pages_own_parts() -> None:
    text = BLOCK.decode()
    assert text.count("<script>") == text.count("</script>") == 1
    assert "SPDX-License-Identifier: Apache-2.0" in text
    for piece in (
        "getElementById('photo')",
        "getElementById('live')",
        "getElementById('run')",
        "typeof follow === 'boolean' && typeof playing === 'boolean'",
        f"PICTURE = '{PICTURE_PATH}', SHOWN = '{SHOWN_PATH}', STATE = '{STATE_PATH}'",
        f"const TOKEN = '{TOKEN_PLACE.decode()}';",
        f"headers: {{'{TOKEN_HEADER}': TOKEN}}",
        "createElement('canvas')",
        "'no live picture'",
        "'Map picture'",
        "'Live picture'",
        # The picture's own shape, not the page's style for its 3D scene.
        "picture.style.height = 'auto';",
        "picture.style.cursor = 'default';",
        "picture.style.background = 'none';",
        # Most of the picture inside the window, and where it is put.
        "const SHARE = 0.75;",
        "new IntersectionObserver(",
        "photo.closest('aside')",
        "column.insertBefore(box, column.firstChild)",
        # Why the robot is not told "live", in a few words beside the button.
        "'scroll the live picture into view'",
        "'tab hidden'",
        "'Map picture showing'",
        "'watch only: see below'",
        "'watch only: the live view is the page opened on the workstation'",
    ):
        assert piece in text, piece
    # The one thing it sends is the acknowledgement of a picture it drew:
    # no storage, no upload, no other address, and no MJPEG image element.
    assert text.count("method: 'POST'") == 1 and text.count("fetch(SHOWN,") == 1
    for piece in ("localStorage", "http://", "https://", "XMLHttpRequest", ".mjpg"):
        assert piece not in text, piece


@experiment
def test_the_experiments_page_has_what_the_block_takes_hold_of(
    experiment_dir: Path,
) -> None:
    page = (experiment_dir / "thor_viewer.html").read_text(encoding="utf-8")
    assert page.count("</body>") == 1
    for piece in (
        '<img id="photo"',
        '<button id="live" class="active">Follow live</button>',
        '<select id="run"><option value="latest">',
        "let data=null, follow=true, playing=false,",
        "setInterval(tick,500)",  # the page's own pace: the stored picture's
    ):
        assert piece in page, piece


# Runs the experiment's own page script and then the block's, as a browser
# does, on stand-ins for the page's elements; then drives the page's own
# handlers. The viewer's answer to a picture request never comes. Prints one
# JSON line.
OWN_PAGE_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const inner = text => text.slice(text.lastIndexOf('<script>') + 8, text.lastIndexOf('</script>'));
const block = inner(fs.readFileSync(process.argv[2], 'utf8'));
const page = inner(fs.readFileSync(process.argv[3], 'utf8'));
const listeners = {}, intervals = [], timeouts = [], made = [], elements = new Map();
const RUNS = ['20261116T120000Z-2', '20261115T100000Z-1'];
let fetches = 0, words = 0;
// What the page asks of its canvas and of anything else: it gives itself, and 0.
const anything = new Proxy(function () {}, {
  get: (target, key) => (key === Symbol.toPrimitive ? () => 0 : anything),
  apply: () => anything,
});
function element(id) {
  const own = {
    id, style: { display: '' }, value: '', src: '', checked: true, hidden: false,
    textContent: id === 'initial' ? 'null' : '', heard: {}, classes: new Set(), children: [],
  };
  own.classList = { toggle(name, on) { on ? own.classes.add(name) : own.classes.delete(name); } };
  own.addEventListener = (kind, work) => { (own.heard[kind] ||= []).push(work); };
  own.append = (...nodes) => { own.children.push(...nodes); };
  own.insertBefore = node => { own.children.push(node); };
  own.replaceChildren = (...nodes) => { own.children = nodes; };
  own.click = () => {
    if (own.onclick) own.onclick();
    for (const work of own.heard.click || []) work();
    for (const work of listeners.click || []) work();
  };
  return new Proxy(own, { get: (target, key) => (key in target ? target[key] : anything) });
}
const byId = id => elements.get(id) || elements.set(id, element(id)).get(id);
byId('photo').parentNode = element('panel');
let answer = { available: false, reason: 'the camera source is being opened' };
const state = {
  run: RUNS[0], seconds: 3, duration: 10, events: 5, live: true, map: null, robot: null,
  failure: null, failures: [], recording: {}, recording_start_utc: null,
};
const context = vm.createContext({
  document: {
    hidden: false, getElementById: byId,
    createElement() { const node = element(''); made.push(node); return node; },
    addEventListener(kind, work) { (listeners[kind] ||= []).push(work); },
  },
  window: { addEventListener() {} },
  location: { search: '', pathname: '/', origin: 'http://viewer' },
  history: { replaceState() {} },
  devicePixelRatio: 1,
  URLSearchParams,
  Option: function (text, value) { this.text = text; this.value = value; },
  setInterval(work) { intervals.push(work); },
  setTimeout(work) { timeouts.push(work); },
  fetch(url) {
    if (url.startsWith('/api/live.jpg')) { fetches += 1; return new Promise(() => {}); }
    if (url.startsWith('/api/live/shown')) { words += 1; return Promise.resolve({ ok: true, status: 204 }); }
    const value = url.startsWith('/api/live.json') ? answer
      : url.startsWith('/api/runs') ? RUNS
      : url.startsWith('/api/bookmarks') ? {} : state;
    return Promise.resolve({ ok: true, status: 200, json: async () => value, text: async () => '' });
  },
});
vm.runInContext(page, context);
const mine = made.length;
vm.runInContext(block, context);

const settle = () => new Promise(done => setImmediate(done));
async function turn() {
  for (const work of timeouts.splice(0)) work();
  for (const work of intervals) work();
  for (let i = 0; i < 4; i++) await settle();
}
function seen() {
  const [bar, button, note, picture] = made.slice(mine, mine + 4);
  const [follow, playing] = vm.runInContext('[follow, playing]', context);
  return {
    button: button.textContent, fetches, live_display: picture.style.display,
    photo_display: byId('photo').style.display, follow, playing,
    mode: byId('mode').textContent, run: byId('run').value,
  };
}
const fire = kind => { for (const work of listeners[kind] || []) work(); };
(async () => {
  const out = [];
  const step = async (name, work) => { if (work) work(); await turn(); out.push([name, seen()]); };
  await step('the page is loaded');
  answer = { available: true, reason: '', view: null };
  await step('the live picture shows');
  await step('the timeline is moved', () => {
    byId('timeline').value = '1.5'; byId('timeline').oninput(); fire('input');
  });
  await step('Live picture is pressed', () => made[mine + 1].click());
  await step('Play recording is pressed', () => byId('play').click());
  await step('Follow live is pressed', () => byId('live').click());
  await step('a refusal is picked', () => {
    byId('failure').value = '2.5'; byId('failure').onchange(); fire('change');
  });
  await step('Follow live again', () => byId('live').click());
  await step('an older run is picked', () => {
    byId('run').value = RUNS[1]; byId('run').onchange(); fire('change');
  });
  await step('Live picture again', () => made[mine + 1].click());
  await step('Map picture is pressed', () => made[mine + 1].click());
  console.log(JSON.stringify({ steps: out, words }));
})();
"""


@experiment
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_block_follows_the_experiments_own_page_script(
    experiment_dir: Path, tmp_path: Path
) -> None:
    harness = tmp_path / "own_page_harness.js"
    harness.write_text(OWN_PAGE_HARNESS, encoding="utf-8")
    page = experiment_dir / "thor_viewer.html"
    done = subprocess.run(
        [NODE, str(harness), str(live_module.PAGE_BLOCK_FILE), str(page)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    told = json.loads(done.stdout)
    steps = dict(told["steps"])
    stored = {"live_display": "none", "photo_display": ""}
    live = {"live_display": "block", "photo_display": "none"}
    fetches = 0  # the picture requests so far: one more each time it shows

    def shows(
        step: str,
        expected: dict[str, str],
        button: str,
        follow: bool,
        playing: bool = False,
    ) -> dict[str, Any]:
        nonlocal fetches
        seen = steps[step]
        assert {key: seen[key] for key in expected} == expected, step
        assert seen["button"] == button, step
        # The page's own two switches, as its own handlers left them.
        assert (seen["follow"], seen["playing"]) == (follow, playing), step
        # A picture is asked for exactly when the live picture comes to show.
        fetches += expected is live
        assert seen["fetches"] == fetches, step
        return seen

    # The page's own script ran: it shows the run it follows.
    loaded = shows("the page is loaded", stored, "no live picture", True)
    assert loaded["mode"] == "LIVE · 20261116T120000Z-2" and loaded["run"] == "latest"
    shows("the live picture shows", live, "Map picture", True)
    # The page's own handlers for the slider, a replay and a refusal leave
    # "Follow live": the stored picture shows.
    moved = shows("the timeline is moved", stored, "Live picture", False)
    assert moved["mode"] == "REPLAY · 20261116T120000Z-2"
    # "Live picture" presses the page's "Follow live".
    shows("Live picture is pressed", live, "Map picture", True)
    shows("Play recording is pressed", stored, "Live picture", False, True)
    shows("Follow live is pressed", live, "Map picture", True)
    shows("a refusal is picked", stored, "Live picture", False)
    shows("Follow live again", live, "Map picture", True)
    # An older run is a recording, although the page's "follow" is on.
    older = shows("an older run is picked", stored, "Live picture", True)
    assert older["run"] == "20261115T100000Z-1"
    newest = shows("Live picture again", live, "Map picture", True)
    assert newest["run"] == "latest"
    # "Map picture" leaves the page following live.
    shows("Map picture is pressed", stored, "Live picture", True)
    # No picture was drawn, so none was acknowledged.
    assert told["words"] == 0


# Runs the block's script, as a viewer serves it, against that viewer over
# HTTP: a tab that draws and acknowledges for some seconds, and presses "Map
# picture" on the way when told to. Prints one JSON line.
HTTP_TAB_HARNESS = r"""
const vm = require('vm');
const base = process.argv[2], seconds = Number(process.argv[3]), mapAt = Number(process.argv[4]);
const drawn = [], posts = [], statuses = {}, listeners = {}, made = [];
function element(tag) {
  const own = {
    tag, style: { display: '' }, children: [], heard: {}, disabled: false, textContent: '',
    classes: new Set(), width: 0, height: 0,
    addEventListener(kind, work) { (this.heard[kind] ||= []).push(work); },
    append(...nodes) { this.children.push(...nodes); },
    insertBefore(node) { this.children.push(node); },
    getContext() { return { drawImage(bitmap) { drawn.push([Date.now(), bitmap.bytes]); } }; },
    click() {
      for (const work of this.heard.click || []) work();
      for (const work of listeners.click || []) work();
    },
  };
  own.classList = { toggle(name, on) { on ? own.classes.add(name) : own.classes.delete(name); } };
  return own;
}
const ids = { photo: element('img'), live: element('button'), run: element('select') };
ids.photo.parentNode = element('div');
ids.run.value = 'latest';
(async () => {
  const page = await (await fetch(base + '/')).text();
  const script = page.slice(page.lastIndexOf('<script>') + 8, page.lastIndexOf('</script>'));
  const context = vm.createContext({
    document: {
      hidden: false,
      getElementById: id => ids[id] || null,
      createElement(tag) { const node = element(tag); made.push(node); return node; },
      addEventListener(kind, work) { (listeners[kind] ||= []).push(work); },
    },
    setInterval, setTimeout,
    // The browser: the whole live picture is inside the window.
    IntersectionObserver: function (callback) {
      const entry = { isIntersecting: true, intersectionRatio: 1 };
      this.observe = () => setTimeout(() => callback([entry]), 0);
    },
    async fetch(url, options) {
      const answer = await fetch(base + url, options);
      const path = url.split('?')[0];
      const codes = statuses[path] ||= {};
      codes[answer.status] = (codes[answer.status] || 0) + 1;
      if (options && options.method === 'POST') posts.push([JSON.parse(options.body).visible, answer.status]);
      return answer;
    },
    // No browser: a picture counts as drawn once its bytes are all here.
    async createImageBitmap(blob) {
      return { width: 960, height: 720, bytes: (await blob.arrayBuffer()).byteLength, close() {} };
    },
  });
  vm.runInContext('let follow = true, playing = false;' + script, context);
  const began = Date.now();
  if (mapAt > 0) setTimeout(() => made[1].click(), mapAt * 1000);
  setTimeout(() => {
    const perSecond = [];
    for (let second = 0; second < seconds; second++) {
      const from = began + 1000 * second;
      perSecond.push(drawn.filter(([at]) => at >= from && at < from + 1000).length);
    }
    console.log(JSON.stringify({
      perSecond, drawn: drawn.length, smallest: Math.min(...drawn.map(one => one[1])),
      taken: posts.filter(([visible, status]) => visible && status === 204).length,
      refused: posts.filter(([visible, status]) => visible && status !== 204).length,
      left: posts.filter(([visible]) => !visible).map(([, status]) => status),
      statuses, note: made[2].textContent, button: made[1].textContent,
    }));
    process.exit(0);
  }, seconds * 1000);
})();
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_blocks_script_keeps_the_view_live_over_http_at_15_pictures_a_second(
    tmp_path: Path,
) -> None:
    harness = tmp_path / "http_tab_harness.js"
    harness.write_text(HTTP_TAB_HARNESS, encoding="utf-8")
    live_module.use_one_encoder_thread()
    camera = Camera(hz=60.0, size=(1280, 960))
    with showing(camera, fps=30.0) as (live, port):
        until(lambda: live_state(port)["available"], "the live picture")
        assert status_now(live.model) == NOT_LIVE  # no page has drawn a picture
        tab = subprocess.Popen(
            [NODE, str(harness), f"http://127.0.0.1:{port}", "6", "3.5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # While the script draws and acknowledges, the view is live, with
            # the age of a picture that is a few camera periods old.
            time.sleep(1.5)
            ages = []
            for _ in range(15):
                status = status_now(live.model)
                assert status.live, ages
                ages.append(status.age_s)
                time.sleep(0.1)
            assert 0.03 <= min(ages) and max(ages) < 0.5, ages
            # The camera is read at the asked rate for the script's requests.
            assert live_state(port)["fps_measured"] >= 15
            # "Map picture" at 3.5 s: not live at once, and not again.
            time.sleep(1.5)
            for _ in range(5):
                assert status_now(live.model) == NOT_LIVE
                time.sleep(0.1)
            out, err = tab.communicate(timeout=WAIT_S)
        finally:
            if tab.poll() is None:
                tab.kill()
                tab.wait(WAIT_S)
        assert tab.returncode == 0, err
    told = json.loads(out)
    # On screen: at least 15 pictures in each whole second before "Map picture",
    # none in the last one. Every one is a whole JPEG of the camera's.
    assert min(told["perSecond"][1:3]) >= 15, told
    assert max(told["perSecond"][1:3]) <= 31 and told["perSecond"][5] == 0, told
    assert told["smallest"] > 1000
    # Each picture drawn was acknowledged, and nearly every word counted: one
    # that is handled after the next picture's counts for nothing.
    assert told["taken"] + told["refused"] == told["drawn"]
    assert told["refused"] <= told["drawn"] // 10, told
    assert told["left"] == [204]  # the one word that the picture shows no longer
    assert told["button"] == "Live picture"
    assert set(told["statuses"][PICTURE_PATH]) <= {"200", "204"}
    print(
        f"measured: the block's script drew {told['perSecond'][1:3]} pictures in "
        f"two whole seconds; the acknowledged picture was {min(ages):.3f} to "
        f"{max(ages):.3f} s old"
    )


# VIEW_STATUS: what the viewer tells the robot


class Wire:
    """A stand-in for the local API's carrier: what was sent on it, in order."""

    def __init__(self, roles: dict[str, int] | None = None) -> None:
        self.sent: list[tuple[int, bytes]] = []
        self.alive = True
        self.close_reason = ""
        self.closed = False
        self.status = None if roles is None else SimpleNamespace(roles=roles)
        self._inbox: queue.SimpleQueue[tuple[int, bytes]] = queue.SimpleQueue()

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        self.sent.append((msg_type, payload))

    send_latest = send_reliable

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        try:
            return self._inbox.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def close(self) -> None:
        self.alive, self.closed = False, True

    def give(self, message: Any) -> None:
        """A link message of the Thor arrives."""
        self._inbox.put((message.TYPE, message.to_payload()))

    def die(self, reason: str) -> None:
        """The robot link service ends."""
        self.close_reason, self.alive = reason, False

    def statuses(self) -> list[ViewStatus]:
        sent = list(self.sent)
        assert {msg_type for msg_type, _ in sent} <= {ViewStatus.TYPE}
        return [decode(msg_type, payload) for msg_type, payload in sent]


class Attaching:
    """attach() of a ViewLink: the next of `wires` each time, or `problem`."""

    def __init__(self, *wires: Wire, problem: BaseException | None = None) -> None:
        self.wires = list(wires)
        self.problem = problem
        self.calls: list[tuple[Any, ...]] = []
        self.at: list[float] = []  # time.monotonic() of each call

    def __call__(self, *args: Any) -> Wire:
        self.calls.append(args)
        self.at.append(time.monotonic())
        if self.problem is not None:
            raise self.problem
        return self.wires.pop(0)


def test_the_viewers_values_for_view_status_are_its_fixed_ones() -> None:
    # One VIEW_STATUS every 0.2 s; an acknowledgement counts for 1 s, and so
    # does a ROBOT_STATE as the source of the Thor's wall clock.
    assert link_module.PERIOD_S == live_settings.STATUS_PERIOD_S == 0.2
    assert (live_settings.ACK_TIMEOUT_S, live_settings.STATE_MAX_AGE_S) == (1.0, 1.0)
    model = LivePicture(lambda: Camera()).model
    assert type(model) is ViewModel
    assert model._ack_timeout_s == live_settings.ACK_TIMEOUT_S
    assert model._state_max_age_s == live_settings.STATE_MAX_AGE_S
    # The attach is tried again with the camera's pauses, in the role
    # "viewer" and with a time limit of 5 s.
    assert link_module.RETRY_PAUSES_S == live_module.RETRY_PAUSES_S
    assert link_module.ATTACH_TIMEOUT_S == live_settings.ATTACH_TIMEOUT_S == 5.0
    wire = Wire()
    attach = Attaching(wire)
    link = ViewLink(
        ViewModel(), "/tmp/somewhere.sock", attach=attach, say=lambda line: None
    )
    link.start()
    try:
        until(lambda: wire.sent, "the first VIEW_STATUS")
    finally:
        link.close()
    assert attach.calls == [("/tmp/somewhere.sock", "viewer", "viewer", 5.0)]
    assert link.socket_path == "/tmp/somewhere.sock"
    # Without a path: the one the console attaches to.
    assert ViewLink(ViewModel()).socket_path == local.default_socket_path()


def test_without_a_service_the_link_says_one_line_and_tries_at_growing_pauses() -> None:
    lines: list[str] = []
    attach = Attaching(problem=ConnectionRefusedError("nobody listens"))
    model = ViewModel()
    link = ViewLink(
        model,
        "/tmp/s.sock",
        attach=attach,
        say=lines.append,
        retry_pauses_s=(0.05, 0.2, 0.4),
    )
    link.start()
    try:
        until(lambda: len(attach.at) >= 5, "five tries to attach")
        why = "cannot attach to the robot link service: nobody listens"
        assert link.state() == {
            "attached": False,
            "why": why,
            "live": None,
            "age_s": None,
        }
        waited = np.diff(attach.at[:5])
        assert 0.05 <= waited[0] < 0.2 <= waited[1] < 0.4 <= min(waited[2:]), waited
        # One line for all the tries.
        unreported = (
            f"viewer: live view: not reported to the robot: {why}; trying again "
            "every 0.05 to 0.4 s. The page is served as before"
        )
        assert lines == [unreported]

        # A service is there: attached, and VIEW_STATUS goes out, not live.
        wire = Wire()
        attach.wires, attach.problem = [wire], None
        until(lambda: len(wire.sent) >= 3, "VIEW_STATUS on the new attachment")
        attached = (
            "viewer: live view: attached to the robot link service at /tmp/s.sock; "
            "the robot is told how fresh the page's live picture is"
        )
        assert lines == [unreported, attached]
        assert set(wire.statuses()) == {NOT_LIVE}
        assert link.state() == {
            "attached": True,
            "why": "",
            "live": False,
            "age_s": None,
        }

        # The service ends: one line, and the first pause again.
        tries = len(attach.at)
        attach.problem = FileNotFoundError("no socket")
        wire.die("the service closed the connection")
        until(lambda: len(attach.at) >= tries + 2, "two more tries")
        gone = (
            "viewer: live view: not reported to the robot: the robot link service "
            "is gone (the service closed the connection); trying again every 0.05 "
            "to 0.4 s. The page is served as before"
        )
        assert lines == [unreported, attached, gone]
        assert wire.closed and link.state()["attached"] is False
        # The first pause came first again: the second try waited the second.
        assert 0.2 <= attach.at[tries + 1] - attach.at[tries] < 0.4
        again = Wire()
        attach.wires, attach.problem = [again], None
        until(lambda: again.sent, "VIEW_STATUS on the second attachment")
        assert lines == [unreported, attached, gone, attached]
    finally:
        link.close()
    # The last word on the attachment that was open: the view is over.
    assert again.statuses()[-1] == NOT_LIVE and again.closed
    assert len(lines) == 4


def test_a_second_live_view_on_the_service_keeps_the_viewer_silent() -> None:
    lines: list[str] = []
    taken = Wire(roles={"viewer": 2, "console": 1})
    attach = Attaching(taken, problem=None)
    link = ViewLink(
        ViewModel(), attach=attach, say=lines.append, retry_pauses_s=(30.0,)
    )
    link.start()
    try:
        until(lambda: lines, "the line")
        assert taken.closed and taken.sent == []
        assert link.state()["why"] == live_settings.SECOND_VIEW
        assert lines == [
            "viewer: live view: not reported to the robot: another live view is "
            "attached; trying again every 30 s. The page is served as before"
        ]
    finally:
        began = time.monotonic()
        link.close()
    assert time.monotonic() - began < 5.0  # close() waits out no pause


class SetClock:
    """A monotonic clock that stands until the test sets it."""

    def __init__(self, now: float) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


class Pairs:
    """A camera source that gives the pairs a test puts in, one by one."""

    def __init__(self) -> None:
        self._pairs: queue.SimpleQueue[StereoPair] = queue.SimpleQueue()
        self.last_none = ""

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        try:
            return self._pairs.get(timeout=min(timeout_s, 0.05))
        except queue.Empty:
            return None

    def put(self, pair: StereoPair) -> None:
        self._pairs.put(pair)


THOR_MONO_S, THOR_WALL_S = 500.0, 1.7e9  # the Thor's two clocks at SetClock 1000


def thor_state(seq: int, at: float) -> Any:
    """The Thor's ROBOT_STATE at the time `at` of the test's clock."""
    passed = at - 1000.0
    return dataclasses.replace(
        robot_state(seq),
        clock_mono_s=THOR_MONO_S + passed,
        clock_wall_s=THOR_WALL_S + passed,
    )


def pair_at(seq: int, at: float, **check: Any) -> StereoPair:
    """A stereo pair exposed 0.05 s and arrived 0.02 s before `at`."""
    exposed_ns = int((THOR_WALL_S + at - 1000.0 - 0.05) * 1e9)
    left = np.full((96 * 3 // 2, 128), 16 + seq, np.uint8)
    return StereoPair(
        session_id=1,
        seq=seq,
        left=left,
        right=left,
        sof_wall_ns=(exposed_ns, exposed_ns + 1000),
        device_index=(5, 7),
        rotation=0,
        receiver_rx_mono_s=at - 0.02,
        check={"source": "test", **check},
    )


class Timed:
    """The viewer's live view on a clock that the test sets.

    Its live picture, its request handler and its link get the ROBOT_STATEs
    and the stereo pairs a test hands them, at the times of that clock. The
    link sends its VIEW_STATUS on a Wire every 0.02 s.
    """

    def __init__(self) -> None:
        self.clock = SetClock(1000.0)
        self.wire, self.pairs = Wire(), Pairs()
        self.live = LivePicture(lambda: self.pairs, fps=15.0, say=lambda line: None)
        self.link = ViewLink(
            self.live.model,
            attach=Attaching(self.wire),
            say=lambda line: None,
            period_s=0.02,
            monotonic=self.clock.monotonic,
        )
        handler = live_handler(
            PlainViewer, None, self.live, bearer, self.link.state, self.clock.monotonic
        )
        self._serving = serving(handler)
        self.live.start()
        self.port = self._serving.__enter__()
        self.link.start()
        self.pages: list[tuple[Tab, http.client.HTTPConnection]] = []
        self.sent: list[tuple[str, ViewStatus]] = []

    def close(self) -> None:
        for _, connection in self.pages:
            connection.close()
        self.link.close()
        self._serving.__exit__(None, None, None)
        self.live.close()

    def state(self, seq: int) -> None:
        """A ROBOT_STATE of this moment arrives."""
        message = thor_state(seq, self.clock.now)
        model = self.live.model
        self.wire.give(message)
        until(
            lambda: model.thor_wall_now(self.clock.now) == message.clock_wall_s,
            "the ROBOT_STATE",
        )

    def picture(self, seq: int, **check: Any) -> None:
        """A stereo pair of this moment arrives; it is picture `seq`."""
        pair = pair_at(seq, self.clock.now, **check)
        self.pairs.put(pair)
        model = self.live.model
        until(lambda: (model.picture() or (0,))[0] == seq, f"picture {seq}")
        # The picture is the left eye, packed with the viewer's width and quality.
        assert model.picture()[1] == encode_picture(
            pair.left, live_settings.DEFAULT_WIDTH_PX, live_settings.JPEG_QUALITY
        )

    def load(self) -> int:
        """A page is loaded; its index."""
        tab = Tab(self.port)
        tab.load()
        self.pages.append((tab, tab._http()))
        return len(self.pages) - 1

    def fetch(self, page: int) -> int | None:
        """That page fetches the newest picture; its number, None when refused."""
        tab, connection = self.pages[page]
        _, seq, _ = tab.fetch(connection, after=0)
        return seq

    def shown(self, page: int, seq: int, visible: bool = True) -> int:
        """That page acknowledges picture `seq`; the status code."""
        tab, connection = self.pages[page]
        return tab.shown(connection, seq, visible)

    def sends(self, situation: str) -> ViewStatus:
        """The VIEW_STATUS the viewer sends in this situation."""
        count = len(self.wire.sent)
        until(
            lambda: len(self.wire.sent) >= count + 2,
            f"VIEW_STATUS in: {situation}",
        )
        kind, payload = self.wire.sent[-1]
        assert kind == ViewStatus.TYPE
        status = decode(kind, payload)
        self.sent.append((situation, status))
        return status


def test_what_the_viewer_sends_situation_by_situation() -> None:
    timed = Timed()
    try:
        clock = timed.clock
        assert timed.sends("nothing has arrived") == NOT_LIVE
        timed.state(1)
        assert timed.sends("a ROBOT_STATE, no picture") == NOT_LIVE
        # Past the quiet time after the first ROBOT_STATE.
        clock.now = 1004.0
        timed.state(2)
        timed.picture(1)
        assert timed.sends("a picture and no page") == NOT_LIVE
        page = timed.load()
        assert timed.sends("a page that fetched nothing") == NOT_LIVE
        assert timed.fetch(page) == 1
        assert timed.sends("a picture fetched and not acknowledged") == NOT_LIVE
        assert timed.shown(page, 1) == 204
        # Live: the picture's 0.05 s since its exposure, sent at the Thor's
        # monotonic clock of this moment.
        live = timed.sends("a picture drawn in a visible tab")
        assert live == ViewStatus(live=True, age_s=live.age_s, sent_mono_s=504.0)
        assert live.age_s == pytest.approx(0.05, abs=1e-6)
        # The same picture a little later, with a ROBOT_STATE of that moment.
        clock.now = 1004.4
        timed.state(3)
        older = timed.sends("the same picture 0.4 s later")
        assert older.live and older.age_s == pytest.approx(0.45, abs=1e-6)
        assert older.sent_mono_s == pytest.approx(504.4)
        # The live picture leaves the screen, or the tab is hidden.
        assert timed.shown(page, 1, visible=False) == 204
        assert timed.sends("the picture shows no longer") == NOT_LIVE
        assert timed.shown(page, 1) == 403
        assert timed.sends("'it shows' for the picture from before") == NOT_LIVE
        timed.picture(2)
        assert timed.fetch(page) == 2
        assert timed.shown(page, 2) == 204
        assert timed.sends("the next picture, drawn and acknowledged").live
        # The page is loaded again: the page before counts no longer.
        other = timed.load()
        assert timed.sends("a newer load of the page") == NOT_LIVE
        assert timed.shown(page, 2) == 403
        assert timed.fetch(page) is None
        assert timed.sends("a word from the page that was not loaded last") == NOT_LIVE
        assert timed.fetch(other) == 2
        assert timed.shown(other, 2) == 204
        assert timed.sends("the page loaded last draws and acknowledges").live
        # A picture whose times nobody measured is never live.
        timed.picture(3, invented_times=True)
        assert timed.fetch(other) == 3
        assert timed.shown(other, 3) == 204
        assert timed.sends("a picture with invented times") == NOT_LIVE
        timed.picture(4)
        assert timed.fetch(other) == 4
        assert timed.shown(other, 4) == 204
        assert timed.sends("a measured picture again").live
        # The camera gives nothing more and the page says nothing more: past
        # the acknowledgement's time limit the view is not live.
        clock.now = 1005.5
        timed.state(4)
        assert timed.sends("no word for 1.1 s") == NOT_LIVE
        # No ROBOT_STATE for longer than its limit: the Thor's clock is unknown.
        timed.picture(5)
        assert timed.fetch(other) == 5
        assert timed.shown(other, 5) == 204
        assert timed.sends("a picture acknowledged again").live
        clock.now = 1006.6
        assert timed.sends("no ROBOT_STATE for 1.1 s") == NOT_LIVE
    finally:
        timed.close()
    # The viewer ends with the word that the view is over.
    assert decode(*timed.wire.sent[-1]) == NOT_LIVE
    # Nothing but VIEW_STATUS went out.
    assert {kind for kind, _ in timed.wire.sent} == {ViewStatus.TYPE}
    assert sum(status.live for _, status in timed.sent) == 6


# The whole path: the viewer's page, the robot link service, the robot app on
# the rig's file


VIEW_NOT_FRESH, OPEN_THE_VIEW = app_module.VIEW_NOT_FRESH, app_module.OPEN_THE_VIEW
# How a human drive's end for the live view begins, in the robot app's words.
DRIVE_VIEW = "the live view is not fresh ("
# Off the two joint stops the rig's arm rests on: where a squeeze takes it.
OFF_THE_STOPS = [0.0, 0.03, 0.03, 0.0, 0.0, 0.0]
# Real seconds a test gives the robot app's side before it fails.
RIG_WAIT_S = 15.0


class Launched:
    """The viewer's launcher, viewer_live's main(), on a thread of the test.

    It runs the experiment's viewer on `port` (0: one the system picks), on a
    stand-in camera, with --link-socket at the test's robot link service;
    `attach` stands in for the attachment to one. end() ends it as Ctrl+C
    does; die() ends it as the end of its process does.
    """

    def __init__(
        self,
        viewer: Any,
        experiment_dir: Path,
        runs: Path,
        link_socket: str,
        camera: Any,
        monkeypatch: pytest.MonkeyPatch,
        attach: Any = connect_local,
        port: int = 0,
    ) -> None:
        self.lines: list[str] = []
        self.code: int | None = None
        self._attach = attach
        servers = self._servers = []
        connections = self._connections = []

        class Server(viewer.ThreadingHTTPServer):  # type: ignore[misc, name-defined]
            def __init__(inner, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                servers.append(inner)

            def get_request(inner) -> Any:
                request, address = super().get_request()
                connections.append(request)
                return request, address

        # The viewer's own server class, kept so that the test can end it.
        monkeypatch.setattr(viewer, "ThreadingHTTPServer", Server)
        argv = ["--experiment-dir", str(experiment_dir), "--link-socket", link_socket]
        argv += ["--runs", str(runs), "--port", str(port), "--no-auth"]
        self._thread = threading.Thread(
            target=self._run, args=(argv, camera), daemon=True
        )
        self._thread.start()
        until(lambda: servers or self.code is not None, "the viewer's server")
        assert servers, (self.code, self.lines)
        self.port = servers[0].server_address[1]

    def _run(self, argv: list[str], camera: Any) -> None:
        self.code = live_main.main(
            argv,
            make_source=lambda: camera,
            attach=self._attach,
            say=self.lines.append,
        )

    def end(self) -> int | None:
        if self._thread.is_alive():
            self._servers[0].shutdown()
            self._thread.join(WAIT_S)
        assert not self._thread.is_alive(), "the viewer did not end"
        return self.code

    def die(self) -> int | None:
        """End the viewer with every connection a browser has open to it.

        A process that ends closes them; on a thread of the test the viewer's
        request threads would go on answering on them.
        """
        self._cut()
        code = self.end()
        self._cut()
        return code

    def _cut(self) -> None:
        for connection in list(self._connections):
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)


def human(state: Any) -> dict[str, Any]:
    """health["human"] of a ROBOT_STATE: the robot app's facts of the human path."""
    return state.health["human"]


class Rig:
    """The programs between a person's browser tab and the robot app.

    The robot app runs in this process in real time, on the simulated arm and
    on the rig's file with human driving switched on in memory. It dials a
    robot link service that runs as a process of its own, on a socket in a
    temporary folder. Attached to that service: the viewer's launcher, a
    stand-in for the human program that sends 50 HUMAN_SETPOINTs a second, and
    a local client that reads ROBOT_STATE.
    """

    def __init__(
        self, stack: Any, watcher: Any, launched: Launched, camera: Camera, hand: Any
    ) -> None:
        self.stack, self.watcher, self.launched = stack, watcher, launched
        self.camera, self.hand = camera, hand
        self.tab = Tab(launched.port)

    def shows(self, shown: Callable[[Any], Any], wait_s: float = RIG_WAIT_S) -> Any:
        """Wait for a ROBOT_STATE that `shown` accepts; returns it."""
        return self.watcher.ws.wait_state(shown, wait_s)


@pytest.fixture
def rig(
    make_realtime_robot_app: Callable[..., Any],
    tmp_path_factory: pytest.TempPathFactory,
    viewer: Any,
    experiment_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Rig]:
    from link_stack import ServiceProcess, Stack
    from test_real_driver import Controller

    shipped = load_config(CONFIGS / "robot_app_rebot.yaml")
    # As the rig's file ships: human driving off, and a takeover needs a fresh
    # live view, at most 0.5 s old.
    assert shipped.human.enabled is False
    assert shipped.human.require_live_view is True
    assert shipped.human.view_age_limit_s == 0.5
    on = dataclasses.replace(shipped.human, enabled=True)
    config = dataclasses.replace(shipped, human=on)
    folder = tmp_path_factory.mktemp("vl")  # a short path: a socket path's limit
    runs = tmp_path / "runs"
    runs.mkdir()
    stack = Stack(make_realtime_robot_app(config=config, connected=False), folder)
    service = ServiceProcess(folder, stack.path)
    launched = hand = made = None
    try:
        stack.port = service.wait_ready()
        stack.dial()
        watcher = stack.client("tool")
        stack.connected(watcher)
        camera = Camera(hz=60.0)
        launched = Launched(
            viewer, experiment_dir, runs, stack.path, camera, monkeypatch
        )
        hand = Controller(stack.path)
        made = Rig(stack, watcher, launched, camera, hand)
        yield made
    finally:
        if made is not None:
            made.tab.close()
        if hand is not None:
            hand.close()
        if launched is not None:
            launched.end()
        stack.close()
        service.close()


def the_page_says_the_live_picture_shows_no_longer(rig: Rig) -> None:
    # What the block's script does for "Map picture", a replay, an older run
    # and a hidden tab.
    assert rig.tab.leave() == 204


def the_browser_says_nothing_more(rig: Rig) -> None:
    rig.tab.freeze()  # a frozen tab, or a browser that was closed


def the_camera_stops(rig: Rig) -> None:
    rig.camera.stopped = True


def the_viewer_ends(rig: Rig) -> None:
    assert rig.launched.end() == 0


def the_page_is_loaded_again_elsewhere(rig: Rig) -> None:
    Tab(rig.launched.port).load()  # another tab, which shows no live picture


# What takes the live view away, by what a person does or what breaks.
BREAKS: dict[str, Callable[[Rig], None]] = {
    "Map picture or a hidden tab": the_page_says_the_live_picture_shows_no_longer,
    "a frozen or closed browser": the_browser_says_nothing_more,
    "the camera stops": the_camera_stops,
    "the viewer ends": the_viewer_ends,
    "a word from a page that was not loaded last": the_page_is_loaded_again_elsewhere,
}


@apai
@experiment
@link
@pytest.mark.parametrize("how", list(BREAKS))
def test_on_the_rig_file_a_person_drives_only_while_the_viewers_page_shows_the_live_picture(
    rig: Rig, how: str
) -> None:
    hand, tab, h = rig.hand, rig.tab, rig.stack.h
    # The viewer is attached and no page shows the live picture: the robot app
    # says why no takeover is possible, in words that name the viewer's page,
    # and refuses one.
    waiting = rig.shows(
        lambda s: (
            s.teleop_stage == "ready"
            and human(s)["takeover"]
            == f"{VIEW_NOT_FRESH}the view is not live{OPEN_THE_VIEW}"
        )
    )
    assert human(waiting)["takeover"] == (
        "the live view is not fresh: the view is not live; show the viewer's live "
        "picture in a visible browser tab"
    )
    assert human(waiting)["live_view_required"] is True and waiting.view_age_s is None
    refused = hand.right_b()
    assert (refused.status, refused.reason) == ("refused", human(waiting)["takeover"])
    attached = [line for line in rig.launched.lines if "live view: attached" in line]
    assert attached == [
        f"viewer: live view: attached to the robot link service at {rig.stack.path}; "
        "the robot is told how fresh the page's live picture is"
    ]

    # The page draws and acknowledges the live pictures: the robot app reports
    # the view's age, inside the rig's limit, and takes the takeover.
    tab.show()
    ready = rig.shows(lambda s: s.view_age_s is not None and human(s)["takeover"] == "")
    assert 0.0 <= ready.view_age_s < 0.5
    taken = hand.right_b()
    assert taken.status == "completed", taken.reason
    owned = rig.shows(lambda s: s.owner == "human" and not human(s)["needs_release"])
    assert human(owned)["takeover_open"] and h.sent() == []
    # A squeeze: the arm follows the hand, setpoint after setpoint.
    hand.goal = list(OFF_THE_STOPS)
    driven = rig.shows(lambda s: human(s)["driving"] and s.move_index >= 10)
    assert driven.view_age_s is not None and driven.view_age_s < 0.5

    BREAKS[how](rig)

    # The drive ends in a hold: the motors stay on and the person keeps the arm.
    held = rig.shows(lambda s: not human(s)["driving"] and human(s)["ended"] != "")
    assert human(held)["ended"].startswith(DRIVE_VIEW), human(held)["ended"]
    assert (held.owner, held.owner_number) == ("human", owned.owner_number)
    assert (held.motion, held.torque_on, held.stop_level) == ("hold", True, "none")
    h.idle()
    sent = len(h.sent())
    assert sent >= 10 and h.sent()[-1] == pytest.approx(OFF_THE_STOPS, abs=1e-9)
    # The grip is still squeezed and moves nothing; a new squeeze neither.
    time.sleep(0.3)
    hand.goal = None
    rig.shows(lambda s: human(s)["takeover"].startswith(VIEW_NOT_FRESH))
    hand.goal = [0.0] * 6
    time.sleep(0.3)
    assert len(h.sent()) == sent
    hand.goal = None
    # Right B is a STOP, after which nobody owns the arm; the takeover it asks
    # for is refused, in words that name the viewer's page.
    hand.ws.stop(reason="right B", sender="controller")
    stopped = rig.shows(lambda s: s.stop_number == held.stop_number + 1)
    assert (stopped.owner, stopped.torque_on) == ("nobody", True)
    # The request quotes the stop number the human program's stand-in has.
    hand.ws.wait_state(lambda s: s.stop_number == stopped.stop_number, RIG_WAIT_S)
    again = hand.right_b()
    assert again.status == "refused"
    assert again.reason.startswith(VIEW_NOT_FRESH), again.reason
    assert again.reason.endswith(
        "; show the viewer's live picture in a visible browser tab"
    )
    now = rig.shows(lambda s: s.seq > stopped.seq)
    assert (now.owner, now.motion, now.torque_on) == ("nobody", "hold", True)
    assert len(h.sent()) == sent
    assert h.arm.torque_on() and h.arm.released_at == []
    print(
        f"measured ({how}): the view was {ready.view_age_s:.3f} s old at the "
        f"takeover; the drive ended after {sent} setpoints with: "
        f"{human(held)['ended']}; then the takeover was refused with: {again.reason}"
    )


# The viewer's page in a real browser

CHROME = shutil.which("google-chrome") or shutil.which("chromium")
# Goes behind the block in the pages of the browser tests. Five times a second,
# and at each step, it tells the test's own path /measured what is on the page;
# the steps (after how many ms, the name of what holds from then, what to do)
# take the place of STEPS. edge(share) is how far to scroll from the page's top
# for that share of the live picture to lie above the window's top edge.
MEASURE = rb"""
<script>
(function () {
  const id = name => document.getElementById(name);
  function sample(phase) {
    const canvas = id('livePhoto'), column = document.querySelector('aside');
    const box = canvas.getBoundingClientRect(), style = getComputedStyle(canvas);
    const stored = id('photo').parentNode.getBoundingClientRect();
    const told = {
      phase, button: id('livePicture').textContent, why: id('livePictureWhy').textContent,
      note: id('livePictureNote').textContent, display: canvas.style.display,
      box: [box.left, box.top, box.width, box.height], pixels: [canvas.width, canvas.height],
      cursor: style.cursor, background: style.backgroundImage,
      first: column.firstElementChild.id, stored: [stored.left, stored.top, stored.width],
      window: [innerWidth, innerHeight], hidden: document.hidden,
    };
    fetch('/measured?' + encodeURIComponent(JSON.stringify(told)), {cache: 'no-store'}).catch(() => {});
  }
  const edge = share => {
    const box = id('livePhoto').getBoundingClientRect();
    return box.top + scrollY + box.height * share;
  };
  let phase = 'at the top';
  setInterval(() => sample(phase), 200);
  for (const [after, name, work] of STEPS) {
    setTimeout(() => { phase = name; work(); sample(name); }, after);
  }
})();
</script>
"""
SCROLLS = rb"""[
  [4000, 'scrolled out', () => id('timing').scrollIntoView()],
  [7000, 'at the top again', () => scrollTo(0, 0)],
  [10000, 'half out', () => scrollTo(0, edge(0.5))],
  [13000, 'most of it in', () => scrollTo(0, edge(0.15))],
  [16000, 'done', () => {}],
]"""
ONLY_WATCHED = rb"[[5000, 'done', () => {}]]"


class Stamped(Wire):
    """A Wire that keeps when each VIEW_STATUS was sent on it."""

    def __init__(self) -> None:
        super().__init__()
        self.stamped: list[tuple[float, ViewStatus]] = []

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        self.stamped.append((time.monotonic(), decode(msg_type, payload)))

    send_latest = send_reliable


class Browsed:
    """What a real browser did with the viewer's page, by the test's clock."""

    def __init__(self) -> None:
        self.reports: list[tuple[float, dict[str, Any]]] = []  # of MEASURE
        self.asked: list[tuple[str, str]] = []  # (method, path and query)
        self.statuses: list[tuple[float, ViewStatus]] = []
        self.lines: list[str] = []

    def phase(self, name: str) -> tuple[float, float]:
        """When the page said that `name` began, and when the next step did."""
        starts = sorted(
            (min(at for at, told in self.reports if told["phase"] == phase), phase)
            for phase in {told["phase"] for _, told in self.reports}
        )
        names = [phase for _, phase in starts]
        at = names.index(name)
        return starts[at][0], starts[at + 1][0]

    def told(self, since: float, until_: float) -> list[dict[str, Any]]:
        return [told for at, told in self.reports if since <= at < until_]

    def sent(self, since: float, until_: float) -> list[ViewStatus]:
        return [status for at, status in self.statuses if since <= at < until_]


def browse(
    viewer: Any,
    experiment_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
    steps: bytes,
) -> Browsed:
    """Headless Chrome on the viewer's own page until its last step, "done".

    The viewer is the launcher's, on a port the system picks, on a stand-in
    camera of the rig's shape and a stand-in for the robot link service that
    hands it a Thor's ROBOT_STATE ten times a second. The browser's profile
    lies under tmp_path.
    """
    browsed = Browsed()
    measure = MEASURE.replace(b"STEPS", steps)
    monkeypatch.setattr(live_module, "page_block", lambda: BLOCK + measure)
    real_handler = live_main.live_handler

    def measured(*args: Any, **kwargs: Any) -> type:
        class Measured(real_handler(*args, **kwargs)):  # type: ignore[misc]
            def do_GET(self) -> None:
                url = urlsplit(self.path)
                if url.path == "/measured":
                    told = json.loads(unquote(url.query))
                    browsed.reports.append((time.monotonic(), told))
                    self.send_response(204)
                    self.end_headers()
                    return
                browsed.asked.append(("GET", self.path))
                super().do_GET()

            def do_POST(self) -> None:
                browsed.asked.append(("POST", self.path))
                super().do_POST()

        return Measured

    monkeypatch.setattr(live_main, "live_handler", measured)
    runs = tmp_path / "runs"
    runs.mkdir()
    wire = Stamped()
    browsed.statuses = wire.stamped
    launched = Launched(
        viewer,
        experiment_dir,
        runs,
        str(tmp_path / "no-service.sock"),
        Camera(hz=60.0, size=(1280, 992)),
        monkeypatch,
        attach=Attaching(wire),
    )
    browsed.lines = launched.lines
    feeding = threading.Event()

    def feed() -> None:
        seq = 0
        while not feeding.wait(0.1):
            seq += 1
            now, wall = time.monotonic(), time.time()
            wire.give(
                dataclasses.replace(
                    robot_state(seq), clock_mono_s=now, clock_wall_s=wall
                )
            )

    threading.Thread(target=feed, daemon=True).start()
    chrome = None
    try:
        # Past the quiet time after the viewer's first ROBOT_STATE.
        time.sleep(CLOCK_STEP_QUIET_S + 0.5)
        chrome = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--disable-extensions",
                "--disable-default-apps",
                "--password-store=basic",
                "--remote-debugging-port=0",
                f"--user-data-dir={tmp_path / 'chrome'}",
                f"--window-size={window}",
                f"http://127.0.0.1:{launched.port}/",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        end = time.monotonic() + 60.0
        while not any(told["phase"] == "done" for _, told in list(browsed.reports)):
            assert time.monotonic() < end, "the browser did not reach its last step"
            assert chrome.poll() is None, "the browser ended"
            time.sleep(0.1)
    finally:
        if chrome is not None:
            chrome.terminate()
            try:
                chrome.wait(15)
            except subprocess.TimeoutExpired:
                chrome.kill()
                chrome.wait(15)
        feeding.set()
        launched.end()
    return browsed


@experiment
@pytest.mark.skipif(CHROME is None, reason="no Chrome is installed")
@pytest.mark.parametrize("window", ["1920,1080", "1280,800"])
def test_in_a_real_browser_the_view_is_live_only_with_the_picture_in_the_window(
    viewer: Any,
    experiment_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    browsed = browse(viewer, experiment_dir, tmp_path, monkeypatch, window, SCROLLS)
    scroll = "scroll the live picture into view"

    # As loaded, without a scroll: the live picture is drawn in its own shape,
    # wholly inside the window, at the top of the column of the page's own
    # picture, and the robot is told "live".
    since, until_ = browsed.phase("at the top")
    top = browsed.told(since + 2.0, until_)
    assert len(top) >= 5
    for told in top:
        left, up, wide, high = told["box"]
        assert told["display"] == "block" and told["pixels"] == [960, 744], told
        assert high / wide == pytest.approx(744 / 960, rel=0.01), told
        assert (told["cursor"], told["background"]) == ("default", "none"), told
        assert up >= 0 and up + high <= told["window"][1], told
        assert left >= 0 and left + wide <= told["window"][0], told
        assert told["first"] == "livePanel", told
        # Above the panel of the page's own picture, as wide as it.
        assert told["stored"][0] == pytest.approx(left, abs=2), told
        assert told["stored"][2] == pytest.approx(wide, abs=4), told
        assert told["stored"][1] >= up + high, told
        assert (told["button"], told["why"]) == ("Map picture", ""), told
        assert re.search(
            r" The robot is told: live view, 0\.\d\d s old\.$", told["note"]
        )
    steady = browsed.sent(since + 2.0, until_)
    assert len(steady) >= 5 and all(status.live for status in steady)
    assert max(status.age_s for status in steady) < 0.5

    def not_live(name: str) -> list[dict[str, Any]]:
        """The page's reports in a step during which the view is not live."""
        since, until_ = browsed.phase(name)
        # Not live at once, and not once in the rest of the step.
        first = [
            at for at, status in browsed.statuses if at >= since and not status.live
        ]
        assert first and first[0] - since < 0.5, name
        rest = browsed.sent(since + 0.5, until_)
        assert len(rest) >= 5 and not any(status.live for status in rest), name
        seen = browsed.told(since + 0.5, until_)
        assert len(seen) >= 5, name
        for told in seen:
            # The live picture stays on the page: only the robot is not told.
            assert (told["display"], told["button"]) == ("block", "Map picture"), told
            assert told["why"] == scroll, told
            assert told["note"].endswith(" The robot is told: no live view."), told
        return seen

    def live_again(name: str) -> list[dict[str, Any]]:
        since, until_ = browsed.phase(name)
        rest = browsed.sent(since + 1.0, until_)
        assert len(rest) >= 5 and all(status.live for status in rest), name
        seen = browsed.told(since + 1.5, until_)
        assert seen and all(told["why"] == "" for told in seen), name
        return seen

    # Scrolled down to the timing graph: the picture is above the window.
    for told in not_live("scrolled out"):
        assert told["box"][1] + told["box"][3] <= 0, told
    live_again("at the top again")
    # Half of it above the window's edge: less than three quarters inside.
    for told in not_live("half out"):
        assert told["box"][1] == pytest.approx(-0.5 * told["box"][3], abs=2), told
    # 85 % of it inside: it counts.
    for told in live_again("most of it in"):
        assert told["box"][1] == pytest.approx(-0.15 * told["box"][3], abs=2), told
    # The page asked for counted pictures only, with its acknowledgements.
    paths = {urlsplit(path).path for _, path in browsed.asked}
    assert {PICTURE_PATH, SHOWN_PATH, STATE_PATH} <= paths
    assert not any("watch=1" in path for _, path in browsed.asked)
    print(
        f"measured ({window}): the picture's box {top[-1]['box']} in a window of "
        f"{top[-1]['window']}; view {min(s.age_s for s in steady):.3f} to "
        f"{max(s.age_s for s in steady):.3f} s old when sent"
    )


@experiment
@pytest.mark.skipif(CHROME is None, reason="no Chrome is installed")
def test_in_a_real_browser_a_page_from_another_machine_watches_and_says_so(
    viewer: Any, experiment_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # For this viewer no browser runs on its machine.
    monkeypatch.setattr(live_module, "on_this_machine", lambda connection: False)
    browsed = browse(
        viewer, experiment_dir, tmp_path, monkeypatch, "1920,1080", ONLY_WATCHED
    )
    since, until_ = browsed.phase("at the top")
    seen = browsed.told(since + 2.0, until_)
    assert len(seen) >= 5
    start = (
        "watch only: the live view is the page opened on the workstation. Live "
        "camera picture, "
    )
    for told in seen:
        # The pictures are drawn, in the same place and shape.
        assert told["display"] == "block" and told["pixels"] == [960, 744], told
        assert told["box"][3] / told["box"][2] == pytest.approx(744 / 960, rel=0.01)
        assert (told["button"], told["why"]) == ("Map picture", "watch only: see below")
        assert told["note"].startswith(start), told
        assert int(told["note"][len(start) :].split()[0]) >= 10, told
        assert told["note"].endswith(" The robot is told: no live view."), told
    # The robot was never told "live", and the page sent no acknowledgement:
    # every picture was one to watch.
    assert len(browsed.statuses) >= 10
    assert not any(status.live for _, status in browsed.statuses)
    assert not any(method == "POST" for method, _ in browsed.asked)
    pictures = [path for _, path in browsed.asked if path.startswith(PICTURE_PATH)]
    assert len(pictures) >= 30 and all(path.endswith("&watch=1") for path in pictures)


# Goes behind the block in the pages of the browser tests that start and end
# the viewer themselves. Five times a second it tells the test's own server,
# which is not the viewer, what is on the page: the page reports while the
# viewer is gone. `load` names this load of the page, `draws` counts the live
# pictures drawn since it.
REPORT = rb"""
<script>
(function () {
  const id = name => document.getElementById(name);
  const load = Math.random().toString(36).slice(2);
  let draws = 0;
  const draw = CanvasRenderingContext2D.prototype.drawImage;
  CanvasRenderingContext2D.prototype.drawImage = function (...what) {
    if (this.canvas === id('livePhoto')) draws += 1;
    return draw.apply(this, what);
  };
  function report() {
    const canvas = id('livePhoto'), note = id('livePictureNote');
    if (!canvas || !note) return;
    const told = {
      load, draws, button: id('livePicture').textContent, why: id('livePictureWhy').textContent,
      note: note.textContent, color: note.style.color, display: canvas.style.display,
      opacity: canvas.style.opacity, hidden: document.hidden,
      // The viewer instance the tab's history entry keeps.
      kept: (history.state && history.state.liveInstance) || '',
    };
    fetch('http://127.0.0.1:REPORT_PORT/?' + encodeURIComponent(JSON.stringify(told)),
      {mode: 'no-cors', cache: 'no-store'}).catch(() => {});
  }
  setInterval(report, 200);
  report();
})();
</script>
"""
# The warning colour as a browser names it.
WARN_RGB = "rgb(255, 189, 105)"
PICTURES_A_SECOND = re.compile(r"Live camera picture, (\d+) a second on this screen\.")
ROBOT_TOLD_LIVE = re.compile(r" The robot is told: live view, 0\.\d\d s old\.$")


def free_port() -> int:
    """A port of 127.0.0.1 that the system picked, and that is free again."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Reporting:
    """Headless Chrome on the page of a viewer that the test starts and ends.

    The viewer is the launcher's, on one port of 127.0.0.1 that the system
    picked, on a stand-in camera of the rig's shape. Each start of it has a
    stand-in for the robot link service of its own, which hands it a Thor's
    ROBOT_STATE ten times a second and keeps each VIEW_STATUS with its time.
    The page reports to a server of the test's. The browser's profile lies
    under tmp_path.
    """

    def __init__(
        self,
        viewer: Any,
        experiment_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._viewer, self._experiment_dir = viewer, experiment_dir
        self._tmp_path, self._monkeypatch = tmp_path, monkeypatch
        self.reports: list[tuple[float, dict[str, Any]]] = []  # of REPORT
        # (when, method, path and query, whether the browser asked).
        self.asked: list[tuple[float, str, str, bool]] = []
        self.port = free_port()
        self.launched: Launched | None = None
        self.wire: Stamped | None = None  # of the viewer that runs now
        self.first_state_at = 0.0  # when that viewer got its first ROBOT_STATE
        self._chrome: subprocess.Popen[bytes] | None = None
        self._ended = threading.Event()
        (tmp_path / "runs").mkdir()
        reports, asked = self.reports, self.asked

        class Reports(BaseHTTPRequestHandler):
            def log_message(inner, *args: Any) -> None:
                pass

            def do_GET(inner) -> None:
                told = json.loads(unquote(urlsplit(inner.path).query))
                reports.append((time.monotonic(), told))
                inner.send_response(204)
                inner.end_headers()

        self._reports = serving(Reports)
        report_port = str(self._reports.__enter__()).encode()
        # What goes into the viewer's page: the block, then REPORT.
        self.block = block = BLOCK + REPORT.replace(b"REPORT_PORT", report_port)
        monkeypatch.setattr(live_module, "page_block", lambda: block)
        real_handler = live_main.live_handler

        def recorded(*args: Any, **kwargs: Any) -> type:
            class Recorded(real_handler(*args, **kwargs)):  # type: ignore[misc]
                def _note(inner, method: str) -> None:
                    browser = "Chrome" in inner.headers.get("User-Agent", "")
                    asked.append((time.monotonic(), method, inner.path, browser))

                def do_GET(inner) -> None:
                    inner._note("GET")
                    super().do_GET()

                def do_POST(inner) -> None:
                    inner._note("POST")
                    super().do_POST()

            return Recorded

        monkeypatch.setattr(live_main, "live_handler", recorded)
        threading.Thread(target=self._feed, daemon=True).start()

    def _feed(self) -> None:
        seq, fed = 0, None
        while not self._ended.wait(0.1):
            wire = self.wire
            if wire is None or not wire.alive:
                continue
            seq += 1
            now, wall = time.monotonic(), time.time()
            if wire is not fed:
                fed, self.first_state_at = wire, now
            wire.give(
                dataclasses.replace(
                    robot_state(seq), clock_mono_s=now, clock_wall_s=wall
                )
            )

    def start_viewer(self) -> float:
        """Start the viewer on the port; when its server was there."""
        self.wire = Stamped()
        self.launched = Launched(
            self._viewer,
            self._experiment_dir,
            self._tmp_path / "runs",
            str(self._tmp_path / "no-service.sock"),
            Camera(hz=60.0, size=(1280, 992)),
            self._monkeypatch,
            attach=Attaching(self.wire),
            port=self.port,
        )
        assert self.launched.port == self.port
        return time.monotonic()

    def stop_viewer(self) -> float:
        """End the viewer as the end of its process does; when it was gone."""
        assert self.launched is not None
        assert self.launched.die() == 0
        return time.monotonic()

    def open(self) -> None:
        """Start the browser on the viewer's page."""
        self._chrome = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--disable-extensions",
                "--disable-default-apps",
                "--password-store=basic",
                "--remote-debugging-port=0",
                f"--user-data-dir={self._tmp_path / 'chrome'}",
                "--window-size=1920,1080",
                f"http://127.0.0.1:{self.port}/",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def close(self) -> None:
        if self._chrome is not None:
            self._chrome.terminate()
            try:
                self._chrome.wait(15)
            except subprocess.TimeoutExpired:
                self._chrome.kill()
                self._chrome.wait(15)
        self._ended.set()
        if self.launched is not None:
            self.launched.end()
        self._reports.__exit__(None, None, None)

    def live(self) -> bool:
        """Whether the viewer that runs now told its robot link service "live" last."""
        sent = self.wire.stamped if self.wire is not None else []
        return bool(sent) and sent[-1][1].live

    def sent(self, since: float) -> list[ViewStatus]:
        """Each VIEW_STATUS the viewer that runs now sent from `since` on."""
        assert self.wire is not None
        return [status for at, status in list(self.wire.stamped) if at >= since]

    def told(self, since: float, until_: float | None = None) -> list[dict[str, Any]]:
        """The page's reports from `since` on, before until_."""
        end = time.monotonic() if until_ is None else until_
        return [told for at, told in list(self.reports) if since <= at < end]

    def wait(
        self,
        holds: Callable[[dict[str, Any]], Any],
        what: str,
        since: float = 0.0,
        within_s: float = WAIT_S,
    ) -> tuple[float, dict[str, Any]]:
        """The page's first report from `since` on that holds, with its time."""
        end = time.monotonic() + within_s
        while True:
            assert self._chrome is not None and self._chrome.poll() is None
            for at, told in list(self.reports):
                if at >= since and holds(told):
                    return at, told
            assert time.monotonic() < end, f"{what} did not come"
            time.sleep(0.05)

    def loads(self) -> list[float]:
        """When the browser asked for the page, each time."""
        return [
            at
            for at, method, path, browser in list(self.asked)
            if browser and method == "GET" and urlsplit(path).path == "/"
        ]

    def browser_asked(self, since: float) -> list[tuple[str, str]]:
        """(method, path and query) of the browser's requests from `since` on."""
        return [
            (method, path)
            for at, method, path, browser in list(self.asked)
            if browser and at >= since
        ]


@pytest.fixture
def reporting(
    viewer: Any, experiment_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Reporting]:
    page = Reporting(viewer, experiment_dir, tmp_path, monkeypatch)
    try:
        yield page
    finally:
        page.close()


def shows_live(told: dict[str, Any]) -> bool:
    """Whether a report of the page says: live picture, and the robot is told so."""
    return (
        told["display"] == "block"
        and told["why"] == ""
        and ROBOT_TOLD_LIVE.search(told["note"]) is not None
    )


@experiment
@pytest.mark.skipif(CHROME is None, reason="no Chrome is installed")
def test_in_a_real_browser_a_page_open_across_a_restart_of_the_viewer_is_the_live_view_again(
    reporting: Reporting,
) -> None:
    page = reporting
    page.start_viewer()
    # Past the quiet time after the viewer's first ROBOT_STATE.
    time.sleep(CLOCK_STEP_QUIET_S + 0.5)
    page.open()
    until(page.live, "the page's live view")
    began, first = page.wait(shows_live, "the page's word that the robot is told")
    time.sleep(2.0)
    assert all(status.live for status in page.sent(began + 0.5))
    assert len(page.loads()) == 1 and first["kept"] == ""
    measured = []

    def restart(down_s: float) -> str:
        """End the viewer, start it again down_s later; the new load's name."""
        before = page.reports[-1][1]["load"]
        ended = page.stop_viewer()
        time.sleep(down_s)
        # While the viewer is gone the page keeps its last picture, dimmed,
        # and says in the warning colour how old it is and that it asks again.
        answered = page.start_viewer()
        down = page.told(ended + 1.5, answered)
        assert len(down) >= (down_s - 1.5) * 3, len(down)
        ages = []
        for told in down:
            # The same load of the page: it did not load itself again.
            assert told["load"] == before, told
            assert (told["display"], told["opacity"]) == ("block", "0.45"), told
            assert told["why"] == "viewer not reached: see below", told
            assert told["color"] == WARN_RGB, told
            found = re.fullmatch(
                rf"{re.escape(UNREACHED)}; this picture is (\d+) s old", told["note"]
            )
            assert found is not None, told
            ages.append(int(found.group(1)))
        assert ages == sorted(ages), ages
        if down:
            assert 1 <= ages[0] <= 3 and down_s - 1 <= ages[-1] <= down_s + 2, ages
            assert len({told["draws"] for told in down}) == 1
        # The new viewer answers: the page loads itself again, once, and
        # shows the live picture within about 3 s.
        shown_at, shown = page.wait(
            lambda told: (
                told["load"] != before
                and told["display"] == "block"
                and told["draws"] >= 3
                and told["note"].startswith("Live camera picture, ")
            ),
            "the live picture of the new viewer",
            since=answered,
        )
        assert shown_at - answered < 3.0, shown_at - answered
        # The tab's history entry keeps, across the load, the viewer instance
        # the page loaded itself for: the one that answers now.
        assert shown["kept"] == live_state(page.port)["instance"]
        # Without any input it is the live view again: the new viewer tells
        # its robot link service so once its quiet time is over.
        until(page.live, "the live view of the new viewer")
        live_at = next(at for at, status in list(page.wire.stamped) if status.live)
        assert live_at - page.first_state_at < CLOCK_STEP_QUIET_S + 2.0
        told_at, _ = page.wait(
            lambda told: told["load"] == shown["load"] and shows_live(told),
            "the new load's word that the robot is told",
            since=answered,
        )
        measured.append(
            (down_s, shown_at - answered, live_at - answered, told_at - answered)
        )
        return shown["load"]

    # Down for five seconds, then down as briefly as a restart is.
    second = restart(5.0)
    time.sleep(2.0)
    third = restart(0.0)
    time.sleep(2.0)
    # One load of the page for each restart, and none else.
    assert len({first["load"], second, third}) == 3
    assert {told["load"] for _, told in page.reports} == {first["load"], second, third}
    loads = page.loads()
    assert len(loads) == 3, loads
    # After it the page is as a page that was loaded by hand: live, its
    # note in no warning colour, and every picture it asks for counts.
    steady = page.told(time.monotonic() - 1.5)
    sent = page.sent(time.monotonic() - 1.5)
    assert len(steady) >= 5
    for told in steady:
        assert shows_live(told) and told["load"] == third, (told, sent)
        assert (told["color"], told["opacity"]) == ("", ""), told
    assert len(sent) >= 5 and all(status.live for status in sent)
    assert max(status.age_s for status in sent) < 0.5
    asked = page.browser_asked(loads[2])
    pictures = [path for _, path in asked if path.startswith(PICTURE_PATH)]
    assert len(pictures) >= 30 and not any("watch=1" in path for path in pictures)
    assert any(method == "POST" for method, _ in asked)
    for down_s, shown_s, live_s, told_s in measured:
        print(
            f"measured (the viewer down for {down_s:g} s): the live picture showed "
            f"again {shown_s:.2f} s after the new viewer's server was there; "
            f"VIEW_STATUS was live again after {live_s:.2f} s (the new viewer's "
            f"quiet time is {CLOCK_STEP_QUIET_S:g} s) and the page said so after "
            f"{told_s:.2f} s"
        )


@experiment
@pytest.mark.skipif(CHROME is None, reason="no Chrome is installed")
def test_in_a_real_browser_a_page_loads_itself_once_for_a_viewer_instance_and_no_more(
    reporting: Reporting, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = reporting
    # A viewer whose every page names another instance than the one that
    # answers: what a load that brought an old page back would be. For this
    # viewer no browser runs on its machine: a watch-only page loads itself
    # whether or not a page has a view token.
    stale = page.block.replace(INSTANCE_PLACE, b"0" * 16)
    monkeypatch.setattr(live_module, "page_block", lambda: stale)
    monkeypatch.setattr(live_module, "on_this_machine", lambda connection: False)
    page.start_viewer()
    time.sleep(CLOCK_STEP_QUIET_S + 0.5)
    page.open()
    instance = live_state(page.port)["instance"]
    # The page loads itself again, once: the load after it finds the instance
    # it was made for in the tab's history entry, and makes no further one.
    again_at, again = page.wait(
        lambda told: told["kept"] == instance and told["draws"] >= 3,
        "the page after its one load",
    )
    time.sleep(6.0)
    loads = page.loads()
    assert len(loads) == 2, loads
    assert len({told["load"] for _, told in page.reports}) == 2
    seen = page.told(again_at + 1.0)
    assert len(seen) >= 15
    for told in seen:
        assert (told["load"], told["kept"]) == (again["load"], instance), told
        # It shows the pictures to watch.
        assert told["display"] == "block", told
        assert told["why"] == "watch only: see below", told
        assert told["note"].startswith(f"{WATCH_ONLY}. Live camera picture, "), told
        assert told["color"] == WARN_RGB, told
    assert seen[-1]["draws"] - seen[0]["draws"] >= 50
    # It never counted: no picture it asked for did, it sent no
    # acknowledgement, and the robot was never told "live".
    asked = page.browser_asked(0.0)
    pictures = [path for _, path in asked if path.startswith(PICTURE_PATH)]
    assert len(pictures) >= 50 and all(path.endswith("&watch=1") for path in pictures)
    assert not any(method == "POST" for method, _ in asked)
    sent = page.sent(0.0)
    assert len(sent) >= 20 and not any(status.live for status in sent)


@experiment
@pytest.mark.skipif(CHROME is None, reason="no Chrome is installed")
def test_in_a_real_browser_a_page_takes_no_view_token_by_a_load_of_its_own(
    reporting: Reporting, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = reporting
    # A page on the workstation that names another instance than the viewer
    # that answers, one of whose pages has its view token: a load of its own
    # would take it.
    stale = page.block.replace(INSTANCE_PLACE, b"0" * 16)
    monkeypatch.setattr(live_module, "page_block", lambda: stale)
    page.start_viewer()
    time.sleep(CLOCK_STEP_QUIET_S + 0.5)
    page.open()
    began, first = page.wait(lambda told: told["draws"] >= 3, "the pictures")
    time.sleep(6.0)
    # It did not load itself: it watches, and says what makes it the live view.
    assert len(page.loads()) == 1
    assert {told["load"] for _, told in page.reports} == {first["load"]}
    seen = page.told(began + 1.0)
    assert len(seen) >= 15
    for told in seen:
        assert (told["display"], told["kept"]) == ("block", ""), told
        assert told["why"] == "watch only: see below", told
        assert told["note"].startswith(f"{NO_PAGE}. Live camera picture, "), told
        assert told["color"] == WARN_RGB, told
    asked = page.browser_asked(0.0)
    pictures = [path for _, path in asked if path.startswith(PICTURE_PATH)]
    assert len(pictures) >= 50 and all(path.endswith("&watch=1") for path in pictures)
    assert not any(method == "POST" for method, _ in asked)
    sent = page.sent(0.0)
    assert len(sent) >= 20 and not any(status.live for status in sent)


@experiment
@pytest.mark.skipif(CHROME is None, reason="no Chrome is installed")
def test_in_a_real_browser_a_replaced_page_keeps_its_pictures_and_never_counts(
    reporting: Reporting,
) -> None:
    page = reporting
    page.start_viewer()
    time.sleep(CLOCK_STEP_QUIET_S + 0.5)
    page.open()
    until(page.live, "the page's live view")
    began, first = page.wait(shows_live, "the page's word that the robot is told")
    time.sleep(2.0)
    counted = page.told(began + 1.0)
    rate = min(int(PICTURES_A_SECOND.search(told["note"]).group(1)) for told in counted)
    assert rate >= 10, rate

    def watches(line: str, since: float, seconds: float) -> list[dict[str, Any]]:
        """The page's reports of a stretch in which it watches and says `line`."""
        time.sleep(max(0.0, since + seconds - time.monotonic()))
        seen = page.told(since, since + seconds)
        assert len(seen) >= seconds * 3, len(seen)
        for told in seen:
            # The same load of the page, with the live picture on it.
            assert (told["load"], told["display"]) == (first["load"], "block"), told
            assert told["button"] == "Map picture", told
            assert told["why"] == "watch only: see below", told
            assert told["note"].startswith(f"{line}. Live camera picture, "), told
            assert (told["color"], told["opacity"]) == (WARN_RGB, ""), told
        # It draws pictures at the rate it had while it counted.
        rates = [int(PICTURES_A_SECOND.search(told["note"]).group(1)) for told in seen]
        assert min(rates[2:]) >= 0.7 * rate, (rates, rate)
        assert seen[-1]["draws"] - seen[0]["draws"] >= 0.7 * rate * (seconds - 0.5)
        # Every picture is one to watch, and the viewer is told nothing.
        asked = [
            (method, path)
            for at, method, path, browser in list(page.asked)
            if browser and since <= at < since + seconds
        ]
        pictures = [path for _, path in asked if path.startswith(PICTURE_PATH)]
        assert len(pictures) >= 0.7 * rate * (seconds - 0.5), len(pictures)
        assert all(path.endswith("&watch=1") for path in pictures)
        assert not any(method == "POST" for method, _ in asked)
        assert not any(urlsplit(path).path == "/" for _, path in asked)
        return seen

    # Another page is loaded on the workstation, and shows the live picture:
    # a stand-in for a tab that does over HTTP what the block's script does.
    other = Tab(page.port)
    other.show()
    try:
        replaced_at, _ = page.wait(
            lambda told: told["note"].startswith(f"{ANOTHER_PAGE}. "),
            "the page's word that another page is the live view",
            since=began,
            within_s=5.0,
        )
        watches(ANOTHER_PAGE, replaced_at, 3.0)
        # The robot is told what the newest page shows: live, by that page.
        sent = page.sent(replaced_at + 1.0)
        assert len(sent) >= 8 and all(status.live for status in sent)
        assert counting_page(other) == {
            "loaded": True,
            "is_this_page": True,
            "asks": True,
        }
        assert other.refused == [] and not other.replaced
        # The page that counts is closed: the page says that no page is the
        # live view now, and it does not become it by itself.
        other.freeze()
        closed_at = time.monotonic()
        alone_at, _ = page.wait(
            lambda told: told["note"].startswith(f"{NO_PAGE}. "),
            "the page's word that no page is the live view",
            since=closed_at,
            within_s=live_module.FETCH_WATCH_S + 3.0,
        )
        watches(NO_PAGE, alone_at, 3.0)
        after = page.sent(closed_at + live_settings.ACK_TIMEOUT_S + 0.3)
        assert len(after) >= 8 and not any(status.live for status in after)
        assert counting_page(other)["is_this_page"] is True
    finally:
        other.close()
    assert len(page.loads()) == 1
    assert {told["load"] for _, told in page.reports} == {first["load"]}
    last = PICTURES_A_SECOND.search(page.reports[-1][1]["note"]).group(1)
    print(
        f"measured: the replaced page drew {rate} and more pictures a second "
        f"while it counted and {last} a second at the end, watching"
    )


# The launcher's settings and its camera frame bridge


def test_the_first_values_are_the_viewers_and_the_camera_receivers() -> None:
    assert live_settings.DEFAULT_FPS == 30.0
    assert live_settings.FPS_RANGE == (1.0, 60.0)
    assert (live_settings.DEFAULT_WIDTH_PX, live_settings.JPEG_QUALITY) == (960, 80)
    assert live_settings.WIDTH_RANGE_PX == (64, 4096)
    assert live_settings.RECEIVER_SOCKET == "/tmp/camera_receiver_avs1_cuda_ipc.sock"
    assert live_settings.STREAMS == ("head.left", "head.right")
    assert live_settings.MAX_SKEW_MS == 5.0
    # The camera receiver's file names the same socket and the same two eyes.
    receiver = yaml.safe_load((CONFIGS / "camera_receiver_rebot.yaml").read_text())
    assert live_settings.RECEIVER_SOCKET == receiver["ipc"]["socket_path"]
    eyes = receiver["streams"][0]["ipc"]
    assert live_settings.STREAMS == (eyes["left"], eyes["right"])
    # The live picture's own values are the same ones.
    picture = LivePicture(lambda: Camera())
    assert (picture.fps, picture.width_px, picture._quality) == (30.0, 960, 80)
    parser = live_main._parser()
    args = parser.parse_args([])
    assert (args.live_fps, args.live_width) == (30.0, 960)
    assert args.receiver_socket == live_settings.RECEIVER_SOCKET


def test_the_viewers_fixed_values_lie_inside_their_bounds() -> None:
    values = live_settings
    assert 1 <= values.JPEG_QUALITY <= 100
    low, high = values.WIDTH_RANGE_PX
    assert 64 <= low <= values.DEFAULT_WIDTH_PX <= high <= 4096
    low, high = values.FPS_RANGE
    assert 0.0 < low <= values.DEFAULT_FPS <= high
    # One VIEW_STATUS a second at least; an acknowledgement and a ROBOT_STATE
    # count for 10 s at most; the two eyes lie 50 ms apart at most.
    assert 0.0 < values.STATUS_PERIOD_S <= 1.0
    assert 0.0 < values.ACK_TIMEOUT_S <= 10.0
    assert 0.0 < values.STATE_MAX_AGE_S <= 10.0
    assert 0.0 <= values.MAX_SKEW_MS <= 50.0
    left, right = values.STREAMS
    assert left.strip() and right.strip() and left != right
    # The status goes out several times within the time an acknowledgement
    # counts for.
    assert values.STATUS_PERIOD_S * 3 <= values.ACK_TIMEOUT_S


def test_the_view_token_is_in_no_answer_about_the_live_view() -> None:
    with arm_showing() as arm:
        port = arm.port
        arm.feed.start()
        until(lambda: robot_json(port)["available"], "the live arm")
        tab = Tab(port)
        token = tab.load().encode()
        connection = tab._http()
        try:
            _, seq, _ = tab.fetch(connection, after=0)
            assert tab.shown(connection, seq) == 204
        finally:
            connection.close()
        # Neither path needs the view token, and neither holds it.
        for path in (STATE_PATH, ROBOT_PATH):
            status, _, body = get(port, path)
            assert status == 200 and token not in body and b"token" not in body, path


def test_a_camera_frame_becomes_a_small_jpeg() -> None:
    red, blue = (0, 0, 255), (255, 0, 0)  # as BGR
    # A camera frame of two plain halves, 128 wide and 96 high.
    bgr = np.zeros((96, 128, 3), np.uint8)
    bgr[:, :64], bgr[:, 64:] = red, blue
    nv12 = frame_source.bgr_to_nv12(bgr)
    small = cv2.imdecode(np.frombuffer(encode_picture(nv12, 64, 90), np.uint8), 1)
    assert small.shape == (48, 64, 3)
    assert np.abs(small[24, 16].astype(int) - red).max() < 60
    assert np.abs(small[24, 48].astype(int) - blue).max() < 60
    # A camera frame narrower than width_px keeps its size.
    same = cv2.imdecode(np.frombuffer(encode_picture(nv12, 960, 80), np.uint8), 1)
    assert same.shape == (96, 128, 3)
    with pytest.raises(ValueError, match="not an NV12 image"):
        encode_picture(np.zeros((100, 128), np.uint8), 64, 80)


def test_the_launchers_bridge_is_a_second_reader_of_the_camera_receiver_on_gpu_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere")
    monkeypatch.setenv("PYTHONPATH", "/apai/src")
    camera_pr = tmp_path / "pr"
    python = camera_pr / "cuda_ipc_consumer" / ".venv" / "bin" / "python"
    environ = {CAMERA_PR_ENV: str(camera_pr)}
    make = live_module.bridge_source_factory("/tmp/r.sock", tmp_path / "run", environ)
    # Without the consumer's Python, and without the variable, no bridge.
    with pytest.raises(CameraError, match="the camera consumer's Python is not there"):
        make()
    nowhere = live_module.bridge_source_factory("/tmp/r.sock", tmp_path / "run", {})
    with pytest.raises(CameraError, match=f"{CAMERA_PR_ENV} is not set"):
        nowhere()
    python.parent.mkdir(parents=True)
    python.write_text("")
    source = make()
    assert type(source) is BridgeSource
    # The same bridge as the agent program's, as a second reader of the same
    # camera receiver: the same environment, size, devices and rotation.
    agents = BridgeSource(
        python=str(python),
        receiver_socket="r",
        run_dir="d",
        camera_pr=str(camera_pr),
    )
    assert source._command == [str(python), "-m", frame_source.BRIDGE_MODULE]
    assert source._receiver_socket == "/tmp/r.sock"
    assert source._streams == ("head.left", "head.right")
    assert source._max_skew_ms == 5.0
    assert (source._size, source._devices) == (agents._size, agents._devices)
    assert source._rotation == agents._rotation
    env = source._environment()
    assert env == agents._environment()
    assert (env["CUDA_DEVICE_ORDER"], env["CUDA_VISIBLE_DEVICES"]) == (
        "PCI_BUS_ID",
        "0",
    )
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONPATH"] == str(frame_source.PACKAGE_ROOT)
    assert env[CAMERA_PR_ENV] == str(camera_pr) and "LD_LIBRARY_PATH" not in env
    assert source._run_dir == tmp_path / "run"
    assert source._process is None  # nothing was started


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--live-fps", "0.5"], "--live-fps must be from 1 to 60, got 0.5"),
        (["--live-fps", "61"], "--live-fps must be from 1 to 60, got 61"),
        (["--live-width", "32"], "--live-width must be from 64 to 4096, got 32"),
        (["--live-width", "5000"], "--live-width must be from 64 to 4096, got 5000"),
        (
            ["--experiment-dir", "{empty}"],
            "must name the folder that holds thor_viewer.py (it is '{empty}')",
        ),
        ([], "must name the folder that holds thor_viewer.py (it is '')"),
    ],
)
def test_the_launcher_refuses_what_it_cannot_use_with_one_line_and_exit_code_2(
    arguments: list[str],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(live_main.EXPERIMENT_DIR_ENV, raising=False)
    arguments = [piece.format(empty=tmp_path) for piece in arguments]
    made: list[int] = []
    code = live_main.main(arguments, make_source=lambda: made.append(1))
    said = capsys.readouterr()
    assert code == live_main.EXIT_NOT_RUN and said.out == "" and made == []
    assert said.err.startswith("viewer: ") and said.err.count("\n") == 1
    assert said.err.endswith(f"{message.format(empty=tmp_path)}\n")


# The experiment's viewer under the launcher's handler


def answers(port: int, paths: list[str]) -> dict[str, tuple[int, str, bytes]]:
    return {path: get(port, path) for path in paths}


@apai
@experiment
def test_every_path_of_the_viewer_answers_as_the_viewer_alone_does(
    stage: Any, viewer: Any
) -> None:
    runs = stage.folder / "runs"
    run = runs / "20261116T120000Z-1"
    assert record(stage, run, RECORDED, QUICK) == launcher.EXIT_OK, stage.lines
    # An old recording: the viewer's "live" flag cannot change between two asks.
    events = run / "debug" / "events.jsonl"
    before = time.time() - 3600
    os.utime(events, (before, before))
    name = run.name
    paths = [
        "/",
        f"/?run={name}",
        "/api/runs",
        "/api/bookmarks",
        "/api/state",
        "/api/image",
        f"/api/state?run={name}&at=0.5",
        f"/api/image?run={name}&at=0.5",
        f"/api/timing?run={name}",
        f"/timing?run={name}",
        "/api/state?run=unknown",
        "/api/state?at=nan",
        "/api/nothing",
        "/thor_viewer.html",
    ]
    word = b'{"seq": 1, "visible": true, "token": ""}'
    before = files(runs), files(PACKAGE)
    with test_viewer.served(viewer, runs) as page:
        alone = answers(page.port, paths)
        alone_posts = [post(page.port, path, word) for path in paths]
    own = viewer.handler(viewer.Viewer(runs), None)
    with showing(Camera(), base=own, authorized=viewer.authorized) as (_, port):
        until(lambda: live_state(port)["available"], "the live picture")
        stream = Stream(port)
        tab = Tab(port)
        tab.show()  # a page draws and acknowledges all the while
        try:
            assert number_of(stream.picture()) > 0
            until(lambda: tab.taken >= 3, "three acknowledged pictures")
            with_live = answers(port, paths)  # while the stream runs
            posts = [post(port, path, word) for path in paths]
        finally:
            tab.close()
            stream.close()
    # The viewer alone takes no POST, and with the live picture its paths
    # answer one as it does.
    assert posts == alone_posts and {code for code, _ in posts} == {501}
    # Nothing was written: not into the runs, not into the launcher's folder.
    assert (files(runs), files(PACKAGE)) == before
    statuses = {path: answer[0] for path, answer in alone.items()}
    for path in paths[:6] + paths[8:10]:
        assert statuses[path] == 200, path
    for path in paths[10:]:
        assert statuses[path] == 400, path
    assert alone["/api/runs"][2] == json.dumps([name]).encode()
    assert (
        alone["/api/image"][1] == "image/jpeg"
        and alone["/api/image"][2][:2] == b"\xff\xd8"
    )
    for path in paths:
        if urlsplit(path).path == "/":
            continue
        assert with_live[path] == alone[path], path
    # The page is the experiment's page with the block before its </body>,
    # and with nothing else; each load has a view token of its own.
    tokens = set()
    for path in paths[:2]:
        status, kind, body = alone[path]
        assert BLOCK_MARK not in body
        assert with_live[path][:2] == (status, kind), path
        tokens.add(served_block(with_live[path][2], body))
    assert len(tokens) == 2


@apai
@experiment
def test_the_viewers_own_token_rule_holds_for_the_live_paths(
    viewer: Any, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    token = viewer.access_token(tmp_path / "private" / "viewer.token")
    cookie = f"thor_viewer_session={token}"
    own = viewer.handler(viewer.Viewer(runs), token)
    with showing(Camera(), base=own, token=token, authorized=viewer.authorized) as (
        live,
        port,
    ):
        until(lambda: live.state()["available"], "the live picture")
        refused = get(port, "/api/runs")
        assert refused[0] == 401
        own_paths = (STATE_PATH, STREAM_PATH, PICTURE_PATH)
        for path in (*own_paths, f"{STATE_PATH}?token={token}"):
            # The same answer as the viewer's own paths give without the token.
            assert get(port, path) == refused, path
            assert get(port, path, Cookie="thor_viewer_session=wrong") == refused
        assert post(port, SHOWN_PATH, b"{}") == (401, refused[2])
        for headers in ({"Authorization": f"Bearer {token}"}, {"Cookie": cookie}):
            assert get(port, "/api/runs", **headers)[0] == 200
            assert live_state(port, **headers)["available"] is True
            stream = Stream(port, **headers)
            try:
                assert stream.status == 200 and number_of(stream.picture()) > 0
            finally:
                stream.close()
            # A page loaded with the viewer's token draws and acknowledges.
            tab = Tab(port, **headers)
            tab.load()
            connection = tab._http()
            try:
                status, seq, _ = tab.fetch(connection, after=0)
                assert status == 200 and tab.shown(connection, seq) == 204
            finally:
                connection.close()
            assert status_now(live.model).live
        # The link with the token opens the page: the viewer's own redirect,
        # which the launcher leaves as it is.
        # It makes no new view token: the page loaded last still counts.
        status, _, body = get(port, f"/?token={token}")
        assert status == 303 and BLOCK_MARK not in body
        assert live.model.is_current_page(tab.token)
        assert get(port, "/")[0] == 401 and live.model.is_current_page(tab.token)
        status, _, body = get(port, "/", Cookie=cookie)
        assert status == 200 and body.count(BLOCK_MARK) == 1
        assert not live.model.is_current_page(tab.token)
    # With --no-auth the viewer has no token, and the live paths are open.
    own = viewer.handler(viewer.Viewer(runs), None)
    with showing(Camera(), base=own, authorized=viewer.authorized) as (live, port):
        until(lambda: live_state(port)["available"], "the live picture")
        stream = Stream(port)
        try:
            assert stream.status == 200 and number_of(stream.picture()) > 0
        finally:
            stream.close()


# The start script


def test_the_start_script_starts_the_launcher_when_the_camera_streamer_is_named(
    stand_in: dict[str, Any], tmp_path: Path
) -> None:
    script, example = stand_in["script"], stand_in["example"]
    env = {**stand_in["env"], CAMERA_PR_ENV: str(tmp_path)}
    folder = env["AGENTIC_REMOTE_EXPERIMENT_DIR"]
    own = ["--no-auth", "--runs", str(tmp_path), "--port", "0"]
    done = script_run(
        script,
        *own[:3],
        "--live-fps",
        "20",
        *own[3:],
        "--live-width=640",
        "--receiver-socket",
        "/tmp/rx.sock",
        "--link-socket",
        "/tmp/link.sock",
        **env,
    )
    assert (done.returncode, done.stderr) == (0, ""), done.stderr
    lines = done.stdout.splitlines()
    assert lines[1].startswith("run_viewer.sh: live picture: on; ")
    assert "--no-live turns it off" in lines[1]
    # The example's package comes before APAI's on the import path.
    assert lines[2] == f"PYTHONPATH={example / 'python'}:{env[APAI_SRC_ENV]}"
    assert lines[3:6] == ["BYTECODE=1", "LD=unset", "CUDA=PCI_BUS_ID:0"]
    # The launcher's own arguments first, then the script's and the caller's
    # for the viewer, as the viewer alone gets them.
    assert lines[6:] == [
        "-m",
        MODULE,
        "--experiment-dir",
        folder,
        "--live-fps",
        "20",
        "--live-width=640",
        "--receiver-socket",
        "/tmp/rx.sock",
        "--link-socket",
        "/tmp/link.sock",
        "--host",
        "127.0.0.1",
        *own,
    ]
    # --link-socket=PATH goes to the launcher too.
    done = script_run(script, "--link-socket=/tmp/link.sock", *own, **env)
    assert done.stdout.splitlines()[6:] == [
        "-m",
        MODULE,
        "--experiment-dir",
        folder,
        "--link-socket=/tmp/link.sock",
        "--host",
        "127.0.0.1",
        *own,
    ]
    # --no-live: the experiment's viewer alone, started as without the variable.
    viewer_file = str(Path(folder) / "thor_viewer.py")
    for arguments in (
        ["--no-live", *own],
        [*own, "--live-fps", "20", "--no-live", "--receiver-socket=/tmp/rx.sock"],
        ["--link-socket", "/tmp/link.sock", "--no-live", *own],
    ):
        done = script_run(script, *arguments, **env)
        assert (done.returncode, done.stderr) == (0, ""), arguments
        lines = done.stdout.splitlines()
        assert lines[1].startswith("run_viewer.sh: live picture: off (--no-live); ")
        # Without the launcher nothing tells the robot of a live view.
        assert lines[1].endswith(", and the robot is told of no live view")
        assert lines[2] == f"PYTHONPATH={env[APAI_SRC_ENV]}"
        assert lines[6:] == [viewer_file, "--host", "127.0.0.1", *own], arguments
    # An argument of the live picture without its value is refused.
    for name in ("--live-fps", "--live-width", "--receiver-socket", "--link-socket"):
        done = script_run(script, *own, name, **env)
        assert (done.returncode, done.stdout) == (2, "")
        assert done.stderr == f"run_viewer.sh: {name} needs a value\n"
    # The help names them, the variable, and what the live view is.
    told = script_run(script, "--help", **env).stdout
    for piece in (
        "--no-live",
        "--live-fps N",
        "--live-width PIXELS",
        "--link-socket PATH",
        CAMERA_PR_ENV,
        "The viewer's page with its live picture showing is the live view",
        "human.require_live_view: true",
    ):
        assert piece in told, piece
    assert told.splitlines()[0].startswith("Start the experiment's viewer")
    assert told.rstrip().endswith("folder that cannot be made.")


def test_the_script_names_the_launchers_first_values() -> None:
    script = (EXAMPLE / "run_viewer.sh").read_text(encoding="utf-8")
    low, high = live_settings.FPS_RANGE
    fps = f"{low:g} to {high:g}. Default: {live_settings.DEFAULT_FPS:g}"
    assert f"#   --live-fps N       the most live pictures a second, {fps}" in script
    low, high = live_settings.WIDTH_RANGE_PX
    width = f"{low} to {high}. Default: {live_settings.DEFAULT_WIDTH_PX}"
    assert f"the live picture's greatest width, {width}" in script
    assert f"-m {MODULE} \\\n" in script
    receiver = yaml.safe_load((CONFIGS / "camera_receiver_rebot.yaml").read_text())
    assert receiver["ipc"]["socket_path"] == live_settings.RECEIVER_SOCKET


# The launcher as a process, on a fake camera receiver


@pytest.fixture
def short_tmp() -> Iterator[Path]:
    """A folder with a short path: the launcher's bridge puts its socket in it."""
    folder = Path(tempfile.mkdtemp(prefix="vl"))
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture
def camera_pr(tmp_path: Path) -> dict[str, Any]:
    """A stand-in for the camera streamer's folder.

    Its consumer's Python runs the real camera frame bridge program on a fake
    receiver's file slots (camera_bench.BRIDGE_CHILD). "child" is the bridge's
    script: a process with it on its command line is a bridge of this test.
    """
    child = tmp_path / "bridge_child.py"
    child.write_text(BRIDGE_CHILD, encoding="utf-8")
    folder = tmp_path / "pr"
    python = folder / "cuda_ipc_consumer" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    # Called as: python -m <the bridge's module> <the bridge's arguments>.
    python.write_text(
        "#!/usr/bin/env bash\nshift 2\n"
        f'exec {shlex.quote(sys.executable)} {shlex.quote(str(child))} "$@"\n'
    )
    python.chmod(0o755)
    return {"folder": folder, "child": str(child)}


class NumberedReceiver(FakeReceiver):
    """FakeReceiver whose pictures of 90 pairs in a row give 90 different JPEGs.

    Most of FakeReceiver's plain colours end as the same JPEG: the colour
    conversion clips them.
    """

    @staticmethod
    def pixel(seq: int, eye: int) -> int:
        return 82 + (seq + eye) % 90


def bridges(child: str) -> list[int]:
    """The process ids of the bridges that run this script."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            words = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if child.encode() in words:
            found.append(int(entry.name))
    return found


def files(folder: Path) -> list[tuple[str, int, int]]:
    """Every file under a folder with its size and the time of its last change."""
    return sorted(
        (str(path.relative_to(folder)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in folder.rglob("*")
        if path.is_file()
    )


def launcher_env(experiment_dir: Path, camera_pr: Path, tmp: Path) -> dict[str, str]:
    """The start script's variables.

    XDG_RUNTIME_DIR puts the default socket of the robot link service into a
    folder of the test: a viewer started without --link-socket finds nothing
    there, and never a service of this machine.
    """
    return {
        "AGENTIC_REMOTE_AGENT_PYTHON": sys.executable,
        "AGENTIC_REMOTE_EXPERIMENT_DIR": str(experiment_dir),
        "AGENTIC_REMOTE_APAI_SRC": os.environ[APAI_SRC_ENV],
        CAMERA_PR_ENV: str(camera_pr),
        "TMPDIR": str(tmp),
        "XDG_RUNTIME_DIR": str(tmp / "runtime"),
    }


def no_service(path: Path | str) -> str:
    """Why the launcher is not attached when nothing is at `path`."""
    return (
        "cannot attach to the robot link service: [Errno 2] No such file or "
        f"directory: '{path}'"
    )


def unreported(path: Path | str) -> str:
    """The launcher's one line without a robot link service at `path`."""
    return (
        f"viewer: live view: not reported to the robot: {no_service(path)}; trying "
        "again every 5 to 60 s. The page is served as before"
    )


def port_of(program: ViewerProcess) -> int:
    address = program.line("Thor read-only map viewer: http://127.0.0.1:")
    return int(address.rstrip("/").rsplit(":", 1)[1])


@apai
@experiment
def test_the_start_script_shows_the_cameras_pictures_and_stores_none(
    stage: Any,
    viewer: Any,
    experiment_dir: Path,
    camera_pr: dict[str, Any],
    short_tmp: Path,
    tmp_path: Path,
) -> None:
    runs = stage.folder / "runs"
    run = runs / "20261116T120000Z-1"
    assert record(stage, run, RECORDED, QUICK) == launcher.EXIT_OK, stage.lines
    rx = tmp_path / "rx"
    rx.mkdir()
    receiver = NumberedReceiver(rx, width=2560, height=1984)
    token_file = tmp_path / "private" / "viewer.token"
    arguments = ["--runs", str(runs), "--port", "0", "--token-file", str(token_file)]
    arguments += ["--receiver-socket", receiver.socket_path]
    # No robot link service of this test: the viewer serves all the same.
    link_socket = tmp_path / "nobody" / "robot_link.sock"
    arguments += ["--link-socket", str(link_socket)]
    env = launcher_env(experiment_dir, camera_pr["folder"], short_tmp)
    try:
        with Publishing(receiver, period_s=1 / 60):
            program = ViewerProcess(*arguments, env=env)
            try:
                program.line("run_viewer.sh: live picture: on; ")
                said = program.line("viewer: live picture: on the page and at ")
                assert said == (
                    f"viewer: live picture: on the page and at {STREAM_PATH}, at "
                    "most 30 a second, at most 960 px wide, JPEG quality 80; shown, "
                    f"not stored. Camera receiver: {receiver.socket_path}"
                )
                said = program.line("viewer: live view: this page")
                assert said == (
                    "viewer: live view: this page with its live picture showing, in "
                    "a visible browser tab; the robot is told its age through the "
                    f"robot link service at {link_socket}"
                )
                assert program.line("not reported") == unreported(link_socket)
                port = port_of(program)
                program.line("viewer: live picture: the camera's pictures arrive")
                token = token_file.read_text().strip()
                auth = {"Authorization": f"Bearer {token}"}
                (bridge,) = bridges(camera_pr["child"])
                before = files(runs), files(token_file.parent), files(PACKAGE)
                # The viewer's own answers, and the page with the block.
                assert get(port, "/api/runs")[0] == 401
                assert get(port, STATE_PATH)[0] == 401
                assert (
                    get(port, "/api/runs", **auth)[2] == json.dumps([run.name]).encode()
                )
                status, _, body = get(port, "/", **auth)
                assert status == 200 and body.count(BLOCK_MARK) == 1
                assert get(port, f"/?token={token}")[0] == 303
                # The stream: pictures of the full size made 960 px wide, a
                # different one each time.
                stream = Stream(port, **auth)
                try:
                    stream.pictures_for(0.5)
                    pictures = stream.pictures_for(1.0)
                    state = live_state(port, **auth)
                finally:
                    stream.close()
                assert 10 <= len(set(pictures)) == len(pictures) <= 31
                picture = cv2.imdecode(np.frombuffer(pictures[-1], np.uint8), 1)
                assert picture.shape == (744, 960, 3)
                assert state["available"] is True and state["width"] == 960
                assert state["fps_asked"] == 30.0 and state["fps_measured"] >= 10
                assert 0.0 <= state["age_s"] < 1.0
                assert state["view"] == {
                    "attached": False,
                    "why": no_service(link_socket),
                    "live": None,
                    "age_s": None,
                }
                # The numbered pictures of a page, through the real bridge:
                # fetched, acknowledged, each another picture of that size.
                tab = Tab(port, **auth)
                tab.show()
                try:
                    until(lambda: tab.taken >= 10, "ten acknowledged pictures")
                finally:
                    tab.close()
                assert tab.drawn == sorted(set(tab.drawn)) and tab.failed == []
                connection = tab._http()
                try:
                    _, _, jpeg = tab.fetch(connection, after=0)
                finally:
                    connection.close()
                shape = cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).shape
                assert shape == (744, 960, 3)
                # Nothing was written: not into the runs, not beside the token,
                # not into the launcher's own folder.
                assert (files(runs), files(token_file.parent), files(PACKAGE)) == before
                assert program.end() == 0, program.text
            finally:
                program.close()
    finally:
        receiver.stop()
    # The bridge ended with the viewer, and the launcher's own folder is gone.
    until(lambda: not os.path.exists(f"/proc/{bridge}"), "the bridge's end")
    assert bridges(camera_pr["child"]) == []
    assert list(short_tmp.iterdir()) == []
    assert "Traceback" not in program.text and "no live picture" not in program.text
    # One line for all the tries to attach, and nothing made where it looked.
    assert program.text.count("not reported to the robot") == 1
    assert not link_socket.parent.exists()


@apai
@experiment
@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGHUP])
def test_the_bridge_ends_with_the_viewer_on_a_signal(
    viewer: Any,
    experiment_dir: Path,
    camera_pr: dict[str, Any],
    short_tmp: Path,
    tmp_path: Path,
    number: signal.Signals,
) -> None:
    runs = tmp_path / "runs"
    rx = tmp_path / "rx"
    runs.mkdir()
    rx.mkdir()
    receiver = FakeReceiver(rx, width=2560, height=1984)
    arguments = ["--runs", str(runs), "--port", "0", "--no-auth"]
    arguments += ["--receiver-socket", receiver.socket_path, "--live-fps", "10"]
    arguments += ["--link-socket", str(tmp_path / "nobody.sock")]
    env = launcher_env(experiment_dir, camera_pr["folder"], short_tmp)
    try:
        with Publishing(receiver, period_s=1 / 30):
            program = ViewerProcess(*arguments, env=env)
            try:
                port = port_of(program)
                program.line("viewer: live picture: the camera's pictures arrive")
                (bridge,) = bridges(camera_pr["child"])
                stream = Stream(port)  # a browser has the stream open
                try:
                    assert len(stream.picture()) > 0
                    program.process.send_signal(number)
                    assert program.process.wait(WAIT_S) == 0, program.text
                finally:
                    stream.close()
            finally:
                program.close()
    finally:
        receiver.stop()
    until(lambda: not os.path.exists(f"/proc/{bridge}"), "the bridge's end")
    assert bridges(camera_pr["child"]) == []
    assert list(short_tmp.iterdir()) == [] and list(runs.iterdir()) == []


@apai
@experiment
@pytest.mark.parametrize("missing", ["consumer", "receiver"])
def test_without_a_camera_the_launcher_serves_the_runs_and_says_why(
    viewer: Any,
    experiment_dir: Path,
    camera_pr: dict[str, Any],
    short_tmp: Path,
    tmp_path: Path,
    missing: str,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    folder = tmp_path / "empty" if missing == "consumer" else camera_pr["folder"]
    folder.mkdir(exist_ok=True)
    arguments = ["--runs", str(runs), "--port", "0", "--no-auth"]
    arguments += ["--receiver-socket", str(tmp_path / "nobody.sock")]
    arguments += ["--link-socket", str(tmp_path / "no_service.sock")]
    env = launcher_env(experiment_dir, folder, short_tmp)
    program = ViewerProcess(*arguments, env=env)
    try:
        port = port_of(program)
        line = program.line("viewer: no live picture: ")
        why = (
            "the camera consumer's Python is not there: "
            if missing == "consumer"
            else "the camera frame bridge ended with code 1: camera frame bridge: "
            f"no camera receiver at {tmp_path / 'nobody.sock'}: "
        )
        assert line.startswith(f"viewer: no live picture: {why}"), line
        assert line.endswith(
            "; trying again in 5 s. The recorded runs are served as before"
        )
        state = live_state(port)
        assert state["available"] is False and why in state["reason"]
        assert get(port, STREAM_PATH)[0] == 503
        assert get(port, "/api/runs") == (200, "application/json", b"[]")
        status, _, body = get(port, "/")
        assert status == 200 and body.count(BLOCK_MARK) == 1
        # A page gets no picture, and the robot is told of no live view.
        tab = Tab(port)
        tab.token = served_block(body, viewers_page(experiment_dir))
        connection = tab._http()
        try:
            assert tab.fetch(connection, after=0)[0] == 204
        finally:
            connection.close()
        assert state["view"]["attached"] is False
        assert program.end() == 0, program.text
    finally:
        program.close()
    # One line for each pause; the second try comes 5 s after the first.
    said = [text for text in program.text.splitlines() if "no live picture" in text]
    assert said[0] == line and len(set(said)) == len(said), said
    assert bridges(camera_pr["child"]) == []
    assert list(short_tmp.iterdir()) == []


def viewers_page(experiment_dir: Path) -> bytes:
    """The experiment's page as its viewer serves it without a run."""
    page = (experiment_dir / "thor_viewer.html").read_text(encoding="utf-8")
    initial = json.dumps({"runs": [], "bookmarks": {}, "state": None})
    return page.replace(
        '<script id="initial" type="application/json">null</script>',
        f'<script id="initial" type="application/json">{initial}</script>',
    ).encode()


@apai
@experiment
@link
def test_without_a_robot_link_service_the_viewer_serves_and_attaches_once_one_runs(
    viewer: Any,
    experiment_dir: Path,
    camera_pr: dict[str, Any],
    short_tmp: Path,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from link_stack import ServiceProcess

    runs = tmp_path / "runs"
    rx = tmp_path / "rx"
    runs.mkdir()
    rx.mkdir()
    folder = tmp_path_factory.mktemp("vs")  # a short path: a socket path's limit
    path = str(folder / "s.sock")
    receiver = NumberedReceiver(rx, width=2560, height=1984)
    arguments = ["--runs", str(runs), "--port", "0", "--no-auth"]
    arguments += ["--receiver-socket", receiver.socket_path, "--link-socket", path]
    env = launcher_env(experiment_dir, camera_pr["folder"], short_tmp)
    service = tool = None
    try:
        with Publishing(receiver, period_s=1 / 60):
            program = ViewerProcess(*arguments, env=env)
            try:
                # No service listens: one line, and everything is served.
                assert program.line("not reported") == unreported(path)
                port = port_of(program)
                program.line("viewer: live picture: the camera's pictures arrive")
                assert get(port, "/api/runs") == (200, "application/json", b"[]")
                status, _, body = get(port, "/")
                assert status == 200
                served_block(body, viewers_page(experiment_dir))
                stream = Stream(port)
                try:
                    assert len(stream.picture()) > 0
                finally:
                    stream.close()
                tab = Tab(port)
                tab.show()
                try:
                    until(lambda: tab.taken >= 5, "five acknowledged pictures")
                    view = live_state(port)["view"]
                    assert view["attached"] is False and view["live"] is None
                    assert view["why"].startswith(
                        "cannot attach to the robot link service: "
                    )

                    # The service starts: the viewer attaches by itself, within
                    # its first pauses, in the role "viewer".
                    service = ServiceProcess(folder, path)
                    service.wait_ready()
                    tool = connect_local(path, "tool", "test", WAIT_S)
                    began = time.monotonic()
                    said = program.line("viewer: live view: attached")
                    assert said == (
                        "viewer: live view: attached to the robot link service at "
                        f"{path}; the robot is told how fresh the page's live "
                        "picture is"
                    )
                    assert time.monotonic() - began < 25.0
                    assert tool.wait_status(
                        lambda s: s.roles.get("viewer") == 1, WAIT_S
                    )
                    until(
                        lambda: live_state(port)["view"]["live"] is False,
                        "the first VIEW_STATUS",
                    )
                    # No robot is connected: nothing gives the Thor's clock,
                    # so the view is not live, and the page is served on.
                    assert live_state(port)["view"] == {
                        "attached": True,
                        "why": "",
                        "live": False,
                        "age_s": None,
                    }
                    taken = tab.taken
                    until(lambda: tab.taken >= taken + 5, "five more pictures")
                finally:
                    tab.close()
                assert tab.failed == [] and tab.refused == []
                assert program.end() == 0, program.text
                # The viewer's end detaches it from the service.
                assert tool.wait_status(lambda s: "viewer" not in s.roles, WAIT_S)
            finally:
                program.close()
    finally:
        receiver.stop()
        if tool is not None:
            tool.close()
        if service is not None:
            service.close()
    assert program.text.count("not reported to the robot") == 1
    assert program.text.count("viewer: live view: attached") == 1
    assert "Traceback" not in program.text
    assert bridges(camera_pr["child"]) == []
    assert list(short_tmp.iterdir()) == [] and list(runs.iterdir()) == []


@apai
@experiment
def test_no_live_starts_the_experiments_viewer_as_before(
    viewer: Any, experiment_dir: Path, camera_pr: dict[str, Any], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    page = (experiment_dir / "thor_viewer.html").read_text(encoding="utf-8")
    env = launcher_env(experiment_dir, camera_pr["folder"], tmp_path)
    program = ViewerProcess(
        "--runs", str(runs), "--port", "0", "--no-auth", "--no-live", env=env
    )
    try:
        program.line("run_viewer.sh: live picture: off (--no-live); ")
        port = port_of(program)
        status, _, body = get(port, "/")
        # The experiment's page with its run list, and nothing of the block.
        initial = json.dumps({"runs": [], "bookmarks": {}, "state": None})
        assert status == 200 and body.decode() == page.replace(
            '<script id="initial" type="application/json">null</script>',
            f'<script id="initial" type="application/json">{initial}</script>',
        )
        for path in (STATE_PATH, STREAM_PATH, PICTURE_PATH):
            status, _, body = get(port, path)
            assert status == 400 and b"Unknown endpoint" in body
        assert post(port, SHOWN_PATH, b"{}")[0] == 501
        assert program.end() == 0, program.text
    finally:
        program.close()
    assert not [
        line for line in program.text.splitlines() if line.startswith("viewer: ")
    ]
    assert bridges(camera_pr["child"]) == []


# The live arm: each ROBOT_STATE as it arrives


# ROBOT_STATEs a second, as the rig's robot app publishes them.
ARM_HZ = 20.0
# The first joint of the stand-in robot app's states starts here, far from
# the recorded poses of these tests, and turns on with every state.
ARM_FROM, ARM_STEP = 0.5, 0.002
LIVE_SOURCE = "ROBOT_STATE as it arrives; shown, not stored"


def arm_pose(k: int) -> list[float]:
    """The joints ROBOT_STATE k measured, in rad."""
    return [ARM_FROM + ARM_STEP * (k % 200), 0.1, 0.1, 0.0, 0.0, 0.0]


def arm_state(k: int) -> Any:
    """ROBOT_STATE k of a robot app whose agent moves the arm, read off now."""
    now = time.monotonic()
    # One joint is read every 4 ms: the six readings span 20 ms.
    joints = [
        JointSample(j, value, now - 0.004 * (6 - j), now - 0.004 * (5 - j), True)
        for j, value in enumerate(arm_pose(k))
    ]
    return dataclasses.replace(
        robot_state(k, owner="agent"),
        clock_mono_s=now,
        clock_wall_s=time.time(),
        joints=joints,
        target_q=arm_pose(k + 1),
        gripper=GripperState(0.5, now, True, True, 0.6, False, "empty"),
        motion="moving",
        # As the robot app says it: the age of the oldest of the six readings.
        health={"reading_age_s": 0.02, "motors": {"state": "holding"}},
    )


def frozen_state(k: int, read_at: float, why: str) -> Any:
    """ROBOT_STATE k of a robot app that read its joints last at `read_at`.

    Every joint's newest reading is carried on, valid, however old it is.
    why "torque off": the motors are off at rest, and no joint is read. why
    "feedback lost": the joint reads fail in a hold until the robot app faults.
    """
    now = time.monotonic()
    joints = [
        JointSample(
            j, value, read_at - 0.004 * (6 - j), read_at - 0.004 * (5 - j), True
        )
        for j, value in enumerate(arm_pose(0))
    ]
    off = why == "torque off"
    lost = "joint feedback lost for 0.36 s in hold"
    return dataclasses.replace(
        robot_state(k, owner="nobody"),
        clock_mono_s=now,
        clock_wall_s=time.time(),
        joints=joints,
        target_q=None if off else arm_pose(0),
        # The jaws were read last when the joints were.
        gripper=GripperState(0.5, read_at, True, True, 0.6, False, "empty"),
        motion="rest_unpowered" if off else "hold",
        torque_on=not off,
        stop_level="none" if off else "fault",
        fault_reason="" if off else lost,
        health={
            "reading_age_s": now - (read_at - 0.02),
            "motors": {"state": "unknown" if off else "holding"},
        },
    )


def stick(q: Any) -> dict[str, Any]:
    """A stand-in for the experiment's geometry: a link that the joints place."""
    q = [float(value) for value in q]
    return {"joints": [[0.0, 0.0, 0.0], [0.3, q[0], 0.2]], "q": q, "boxes": []}


class Feed:
    """A stand-in robot app: ROBOT_STATE after ROBOT_STATE on a Wire, ARM_HZ a second.

    .make(k) gives state k: a moving arm's, read off at that moment, until a
    test sets another.
    """

    def __init__(self, wire: Wire, hz: float = ARM_HZ) -> None:
        self.wire, self.hz = wire, hz
        self.make: Callable[[int], Any] = arm_state
        self.sent = 0  # the number of the newest state given
        self._end = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        due = time.monotonic()
        while not self._end.wait(max(0.0, due - time.monotonic())):
            self.wire.give(self.make(self.sent + 1))
            self.sent += 1
            due += 1.0 / self.hz

    def stop(self) -> None:
        self._end.set()
        if self._thread is not None:
            self._thread.join(WAIT_S)


@contextmanager
def arm_showing(
    *,
    geometry: Any = stick,
    robot: LiveRobot | None = None,
    attach: Attaching | None = None,
    retry_pauses_s: tuple[float, ...] = (0.05,),
    **serving_with: Any,
) -> Iterator[SimpleNamespace]:
    """The viewer's live paths with a live arm, on a stand-in robot link service.

    A LiveRobot gets the ROBOT_STATEs of a ViewLink that is attached to a Wire;
    .feed gives that Wire the states of a stand-in robot app once it is started.
    The rest is showing()'s.
    """
    wire = Wire()
    robot = LiveRobot(geometry) if robot is None else robot
    lines: list[str] = []
    with showing(Camera(), robot=robot, **serving_with) as (live, port):
        link = ViewLink(
            live.model,
            "/tmp/no-service-of-this-test.sock",
            attach=Attaching(wire) if attach is None else attach,
            say=lines.append,
            retry_pauses_s=retry_pauses_s,
            robot=robot,
        )
        feed = Feed(wire)
        link.start()
        try:
            yield SimpleNamespace(
                port=port, robot=robot, link=link, wire=wire, feed=feed, lines=lines
            )
        finally:
            feed.stop()
            link.close()
            robot.close()


def robot_json(
    port: int,
    after: int | None = None,
    connection: HTTPConnection | None = None,
    **headers: str,
) -> dict[str, Any]:
    """What /api/live/robot.json answers, once it is not number `after`."""
    path = ROBOT_PATH if after is None else f"{ROBOT_PATH}?after={after}"
    if connection is None:
        status, kind, body = get(port, path, **headers)
    else:
        connection.request("GET", path, headers=headers)
        reply = connection.getresponse()
        status, kind, body = reply.status, reply.getheader("Content-Type"), reply.read()
    assert (status, kind) == (200, "application/json"), (status, body[:300])
    return json.loads(body)


def test_a_page_that_asks_at_once_gets_every_robot_state_and_a_slow_one_the_newest() -> (
    None
):
    with arm_showing() as arm:
        feed, port = arm.feed, arm.port
        feed.start()
        until(lambda: robot_json(port)["per_second"] >= 19, "a second of states")
        fast: list[tuple[float, dict[str, Any]]] = []
        seconds = 2.4

        def ask_at_once() -> None:
            connection = HTTPConnection("127.0.0.1", port, timeout=WAIT_S)
            try:
                told = robot_json(port, connection=connection)
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    told = robot_json(port, told["n"], connection)
                    fast.append((time.monotonic(), told))
                assert connection.sock is not None  # one connection for all
            finally:
                connection.close()

        reader = threading.Thread(target=ask_at_once, daemon=True)
        reader.start()
        # Meanwhile a page that takes 0.3 s over each state.
        slow, behind = [], []
        connection = HTTPConnection("127.0.0.1", port, timeout=WAIT_S)
        try:
            told = robot_json(port, connection=connection)
            for _ in range(7):
                time.sleep(0.3)
                began = time.monotonic()
                told = robot_json(port, told["n"], connection)
                assert time.monotonic() - began < 0.1  # it waited for none
                behind.append(feed.sent - told["seq"])
                slow.append(told["n"])
        finally:
            connection.close()
        reader.join(WAIT_S)
        assert not reader.is_alive()
        newest = robot_json(port)
        assert 0 <= feed.sent - newest["seq"] <= 1
        sent_only = arm.wire.statuses()
    # Every state reached the page that asked again at once, in its order:
    # 18 a second and more of the stand-in robot app's 20.
    numbers = [told["n"] for _, told in fast]
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers))), numbers
    assert len(numbers) >= 18 * seconds - 1, numbers
    assert [told["seq"] for _, told in fast] == numbers
    for _, told in fast:
        k = told["seq"]
        assert told["available"] is True and told["why"] == ""
        assert 0.0 <= told["age_s"] < 0.2 and 17 <= told["per_second"] <= 22, told
        assert abs(told["wall_s"] - time.time()) < 60.0
        # The measured joints, the last target, the jaws and who has the arm.
        assert [one["joint"] for one in told["joints"]] == list(range(6))
        assert [one["q"] for one in told["joints"]] == pytest.approx(arm_pose(k))
        assert all(one["valid"] for one in told["joints"])
        assert told["joints"][0]["deg"] == pytest.approx(np.degrees(arm_pose(k)[0]))
        assert told["joints"][0]["target"] == pytest.approx(arm_pose(k + 1)[0])
        assert told["target_q"] == pytest.approx(arm_pose(k + 1))
        # The jaw reading is of the state's own moment: as old as the state.
        jaw_age = told["gripper"].pop("age_s")
        assert jaw_age == pytest.approx(told["age_s"], abs=0.002)
        assert told["gripper"] == {
            "valid": True,
            "opening": 0.5,
            "target": 0.6,
            "load": "empty",
            "fresh": True,
        }
        assert (told["owner"], told["stop_level"], told["motion"]) == (
            "agent",
            "none",
            "moving",
        )
        assert (told["torque_on"], told["torque"]) == (True, "on (holding)")
        assert told["reading_age_s"] == 0.02
        # Live: the state's age and the age of its readings are within the limit.
        assert told["stale_s"] == robot_module.STALE_S == 1.0
        assert told["pose_age_s"] == pytest.approx(told["age_s"] + 0.02, abs=0.002)
        assert told["last_read"] is None and told["model_run"] is None
        # The arm as the page draws it: the geometry of those joints.
        arm_drawn = told["robot"]
        assert arm_drawn["q"] == pytest.approx(arm_pose(k))
        assert arm_drawn["joints"] == stick(arm_pose(k))["joints"]
        assert (arm_drawn["boxes"], arm_drawn["source"]) == ([], LIVE_SOURCE)
        assert arm_drawn["sample_span_s"] == pytest.approx(0.02)
        assert arm_drawn["age_s"] == pytest.approx(told["age_s"] + 0.02, abs=0.002)
    # The slow page got the newest state each time, at most one behind the
    # robot app's newest, and missed the ones between.
    assert max(behind) <= 1 and min(behind) >= 0, behind
    assert slow == sorted(set(slow)) and min(np.diff(slow)) >= 4, slow
    # The viewer still sent the robot nothing but VIEW_STATUS, as without a page.
    assert len(sent_only) >= 5 and set(sent_only) == {NOT_LIVE}
    assert robot_module.LIVE_SOURCE == LIVE_SOURCE


def test_a_state_is_with_the_page_a_few_milliseconds_after_the_viewer_has_it() -> None:
    writes: list[int] = []

    class Counted(PlainViewer):
        """PlainViewer that notes the size of each write to its connection."""

        def setup(self) -> None:
            super().setup()
            raw = self.wfile

            class Noted:
                def write(self, data: bytes) -> int:
                    writes.append(len(data))
                    return raw.write(data)

                def __getattr__(self, name: str) -> Any:
                    return getattr(raw, name)

            self.wfile = Noted()

    with arm_showing(base=Counted) as arm:
        port = arm.port
        arm.feed.start()
        until(lambda: robot_json(port)["per_second"] >= 19, "a second of states")
        # One connection that stays open, as the page's: each answer follows
        # the request for it at once.
        connection = HTTPConnection("127.0.0.1", port, timeout=WAIT_S)
        delays = []
        try:
            told = robot_json(port, connection=connection)
            writes.clear()
            for _ in range(60):
                told = robot_json(port, told["n"], connection)
                # From the viewer's answer, by its wall clock, to its receipt.
                delays.append(time.time() - told["wall_s"])
            assert connection.sock is not None
            # Each answer went out in one write, headers and body together:
            # the body of a second small write would wait about 40 ms for the
            # browser's acknowledgement of the first.
            assert len(writes) == 60 and min(writes) > 500, writes
            # So do the live picture's answers on that connection.
            writes.clear()
            connection.request("GET", STATE_PATH)
            reply = connection.getresponse()
            assert reply.status == 200 and reply.read().startswith(b"{")
            connection.request("GET", f"{PICTURE_PATH}?after=0&watch=1")
            reply = connection.getresponse()
            assert reply.status == 200 and len(reply.read()) > 100
            assert len(writes) == 2, writes
        finally:
            connection.close()
    delays = sorted(delays[5:])
    median = delays[len(delays) // 2]
    assert median < 0.010, delays
    print(
        "measured: from the viewer's answer to its receipt by a client that asks "
        f"again at once, {delays[0] * 1e3:.1f} / {median * 1e3:.1f} / "
        f"{delays[-1] * 1e3:.1f} ms (least, median, most) over {len(delays)} states"
    )


def test_the_robot_state_path_follows_the_viewers_access_rule_and_needs_no_view_token() -> (
    None
):
    token = "t" * 43
    auth = {"Authorization": f"Bearer {token}"}
    with arm_showing(token=token, this_machine=only_127_0_0_1) as arm:
        port = arm.port
        arm.feed.start()
        until(lambda: robot_json(port, **auth)["available"], "the live arm")
        # Without the viewer's token: the answer of every other path of it.
        refused = get(port, STATE_PATH)
        assert refused[0] == 401 and b"private access link" in refused[2]
        for path in (
            ROBOT_PATH,
            f"{ROBOT_PATH}?after=0",
            f"{ROBOT_PATH}?token={token}",
        ):
            assert get(port, path) == refused, path
            assert get(port, path, Authorization="Bearer wrong") == refused, path
        # With it a page of another machine reads the states too: a page that
        # only watches has no view token, and this path tells the robot nothing.
        elsewhere = HTTPConnection(
            "127.0.0.1", port, timeout=WAIT_S, source_address=(OTHER_MACHINE, 0)
        )
        try:
            told = robot_json(port, connection=elsewhere, **auth)
            newer = robot_json(port, told["n"], elsewhere, **auth)
        finally:
            elsewhere.close()
        assert told["available"] and newer["n"] == told["n"] + 1
        # The path takes no POST, as no path of the viewer alone does.
        assert post(port, ROBOT_PATH, b"{}", **auth)[0] == 501
        assert set(arm.wire.statuses()) == {NOT_LIVE}
    # Without a token, as with --no-auth, it is open.
    with arm_showing() as arm:
        arm.feed.start()
        until(lambda: robot_json(arm.port)["available"], "the live arm")


def test_without_a_live_state_the_path_says_why() -> None:
    attach = Attaching(problem=ConnectionRefusedError("nobody listens"))
    with arm_showing(attach=attach) as arm:
        port = arm.port

        def says(why: str) -> dict[str, Any]:
            return until(
                lambda: (told := robot_json(port))["why"].startswith(why) and told, why
            )

        # No robot link service: no state, and why.
        told = says("cannot attach to the robot link service: nobody listens")
        assert abs(told.pop("wall_s") - time.time()) < 60.0
        assert told == {
            "n": 0,
            "available": False,
            "why": "cannot attach to the robot link service: nobody listens",
            "age_s": None,
            "per_second": 0,
            "stale_s": 1.0,
            "pose_age_s": None,
            "last_read": None,
            "model_run": None,
            "robot": None,
        }
        # A page that has that answer waits for another one, a second at most.
        began = time.monotonic()
        assert robot_json(port, after=0)["n"] == 0
        waited = time.monotonic() - began
        assert robot_module.ROBOT_WAIT_S <= waited < robot_module.ROBOT_WAIT_S + 1
        # Attached, and the service has no robot connection.
        wire = Wire(roles={"viewer": 1})
        wire.status.robot_connected = False
        wire.status.close_reason = "the Thor closed the connection"
        attach.wires, attach.problem = [wire], None
        says(
            "the robot is not connected to the robot link service (the Thor closed "
            "the connection)"
        )
        wire.status.robot_connected, wire.status.close_reason = True, ""
        assert says("no ROBOT_STATE has arrived")["n"] == 0
        # A state without a valid reading of one joint: its readings, no arm.
        state = arm_state(1)
        bad = dataclasses.replace(state.joints[2], valid=False)
        joints = [*state.joints[:2], bad, *state.joints[3:]]
        wire.give(dataclasses.replace(state, joints=joints))
        told = says("a joint has no valid reading")
        assert (told["n"], told["available"], told["robot"]) == (1, False, None)
        assert [one["valid"] for one in told["joints"]] == [1, 1, 0, 1, 1, 1]
        assert told["joints"][2]["q"] is None and told["joints"][2]["deg"] is None
        assert told["joints"][1]["q"] == pytest.approx(0.1) and told["owner"] == "agent"
        # The next whole one is drawn; a waiting page gets it as it arrives.
        asked: list[dict[str, Any]] = []
        waiting = threading.Thread(
            target=lambda: asked.append(robot_json(port, after=1)), daemon=True
        )
        waiting.start()
        time.sleep(0.2)
        assert asked == []
        wire.give(arm_state(2))
        waiting.join(WAIT_S)
        assert asked[0]["n"] == 2 and asked[0]["available"] is True
        assert asked[0]["robot"]["q"] == pytest.approx(arm_pose(2))
        # No further state for more than a second: no live state, with the
        # readings of the last one and its age.
        time.sleep(robot_module.STALE_S + 0.2)
        old = robot_json(port)
        assert (old["n"], old["available"], old["robot"]) == (2, False, None)
        assert old["why"].startswith("no ROBOT_STATE for 1.") and old["age_s"] > 1.0
        assert old["joints"][0]["q"] == pytest.approx(arm_pose(2)[0])
        # The service ends: that is why, then that it cannot be attached to.
        attach.problem = FileNotFoundError("no socket")
        wire.die("the service closed the connection")
        gone = says("the robot link service is gone (the service closed the")
        assert gone["n"] == 2 and gone["available"] is False
        says("cannot attach to the robot link service: no socket")
    # A viewer without a live arm says so on that path.
    with showing(Camera()) as (_, port):
        told = robot_json(port)
        assert (told["n"], told["available"], told["robot"]) == (0, False, None)
        assert told["why"] == "this viewer has no live robot state"


def test_a_state_whose_arm_cannot_be_placed_keeps_its_readings_and_says_why() -> None:
    def no_model(q: Any) -> dict[str, Any]:
        raise ArmModelError("no run folder names the arm's model")

    def broken(q: Any) -> dict[str, Any]:
        raise ValueError("six joints expected")

    def not_finite(q: Any) -> dict[str, Any]:
        return {"joints": [[float("nan"), 0.0, 0.0]], "q": list(q), "boxes": []}

    for geometry, why in (
        (None, "this viewer draws no live arm"),
        (no_model, "no run folder names the arm's model"),
        (broken, "the arm cannot be placed: ValueError: six joints expected"),
        (not_finite, "the arm cannot be placed: ValueError: Out of range float"),
    ):
        robot = LiveRobot(geometry)
        robot.on_link("")
        robot.on_state(arm_state(7), time.monotonic())
        told = robot.answer()
        assert (told["n"], told["available"], told["robot"]) == (1, False, None)
        assert told["why"].startswith(why), told["why"]
        assert [one["q"] for one in told["joints"]] == pytest.approx(arm_pose(7))
        json.dumps(told, allow_nan=False)
    # Each state is placed once, however many pages ask for it.
    placed: list[list[float]] = []
    robot = LiveRobot(lambda q: placed.append(list(q)) or stick(q))
    robot.on_state(arm_state(1), time.monotonic())
    assert [robot.answer()["robot"]["q"] for _ in range(5)] == [arm_pose(1)] * 5
    robot.on_state(arm_state(2), time.monotonic())
    assert robot.answer(after=1)["n"] == 2 and placed == [arm_pose(1), arm_pose(2)]
    # close() ends the wait of an answer.
    threading.Timer(0.2, robot.close).start()
    began = time.monotonic()
    assert robot.answer(after=2, timeout_s=WAIT_S)["n"] == 2
    assert time.monotonic() - began < 5.0


# Readings that are old: the arm is not live, however many states arrive

LAST_READ_SOURCE = "ROBOT_STATE: the joints' last reading; shown, not stored"
# Why the robot app reads no joint, as the viewer says it.
NOT_READ = {
    "torque off": "torque off: the joints are not read",
    "feedback lost": "joint feedback lost",
}
LAST_READ_WHY = re.compile(r"the joints were last read (\d+\.\d) s ago \((.+)\)")


def asked_at_once(port: int, seconds: float) -> list[tuple[float, dict[str, Any]]]:
    """(when, the answer) of a page that asks again at once, for `seconds`."""
    got = []
    connection = HTTPConnection("127.0.0.1", port, timeout=WAIT_S)
    try:
        told = robot_json(port, connection=connection)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            told = robot_json(port, told["n"], connection)
            got.append((time.monotonic(), told))
    finally:
        connection.close()
    return got


@pytest.mark.parametrize("why", list(NOT_READ))
def test_twenty_states_a_second_with_frozen_readings_are_never_called_live(
    why: str,
) -> None:
    with arm_showing() as arm:
        port, feed = arm.port, arm.feed
        feed.start()
        until(lambda: robot_json(port)["available"], "the live arm")
        until(lambda: robot_json(port)["per_second"] >= 19, "a second of states")
        # The robot app reads no joint any longer and goes on publishing, 20
        # states a second: each carries the readings of 45 s ago, all valid.
        read_at = time.monotonic() - 45.0
        feed.make = lambda k: frozen_state(k, read_at, why)
        until(lambda: robot_json(port)["last_read"], "a state with old readings")
        got = asked_at_once(port, 2.0)
        # The joints are read again: the arm is live again.
        feed.make = arm_state
        back = until(
            lambda: (told := robot_json(port))["available"] and told, "fresh readings"
        )
        sent_only = arm.wire.statuses()
    numbers = [told["n"] for _, told in got]
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers))), numbers
    assert len(numbers) >= 34, numbers
    for at, told in got:
        # The states arrive, each fresh on this machine, and none is a live arm.
        assert told["available"] is False, told
        assert 0.0 <= told["age_s"] < 0.2 and 17 <= told["per_second"] <= 22, told
        # How old the oldest reading is, and why the robot app has no newer one.
        age = told["pose_age_s"]
        assert age == pytest.approx(at - read_at + 0.02, abs=0.2) and age > 45.0
        assert told["reading_age_s"] == pytest.approx(age - told["age_s"], abs=0.002)
        assert told["last_read"] == {"age_s": age, "why": NOT_READ[why]}
        said = LAST_READ_WHY.fullmatch(told["why"])
        assert said is not None and said.group(2) == NOT_READ[why], told["why"]
        assert float(said.group(1)) == pytest.approx(age, abs=0.06)
        # The pose the joints were last read at, for the page to draw as that.
        drawn = told["robot"]
        assert drawn["q"] == pytest.approx(arm_pose(0))
        assert drawn["joints"] == stick(arm_pose(0))["joints"]
        assert drawn["source"] == LAST_READ_SOURCE
        assert drawn["age_s"] == pytest.approx(age, abs=0.002)
        # The jaws were read last when the joints were.
        jaws = told["gripper"]
        assert (jaws["valid"], jaws["opening"], jaws["fresh"]) == (True, 0.5, False)
        assert jaws["age_s"] == pytest.approx(age - 0.02, abs=0.01)
        assert told["torque_on"] is (why != "torque off")
    assert back["last_read"] is None and back["robot"]["source"] == LIVE_SOURCE
    assert back["gripper"]["fresh"] is True
    assert set(sent_only) == {NOT_LIVE}
    assert robot_module.LAST_READ_SOURCE == LAST_READ_SOURCE
    assert robot_module.TORQUE_OFF_WHY == NOT_READ["torque off"]
    assert robot_module.FEEDBACK_LOST == NOT_READ["feedback lost"]


def test_the_arm_is_live_only_while_its_readings_are_within_the_limit() -> None:
    with arm_showing() as arm:
        port, feed = arm.port, arm.feed
        # From its first state on this robot app reads no joint: the motors
        # are off. The readings are fresh at first and grow older.
        read_at = time.monotonic()
        feed.make = lambda k: frozen_state(k, read_at, "torque off")
        feed.start()
        until(lambda: robot_json(port)["n"], "the first state")
        got = [told for _, told in asked_at_once(port, 2.2)]
    live = [told for told in got if told["available"]]
    old = [told for told in got if not told["available"]]
    assert len(live) >= 10 and len(old) >= 15, (len(live), len(old))
    # Live up to the limit and not after it: the state's own age on this
    # machine and the age of its readings together.
    limit = robot_module.STALE_S
    for told in live:
        assert told["pose_age_s"] <= limit and told["last_read"] is None
        assert told["robot"]["source"] == LIVE_SOURCE
    for told in old:
        assert told["pose_age_s"] > limit and told["age_s"] < 0.2
        assert told["last_read"]["why"] == NOT_READ["torque off"], told
        assert told["robot"]["source"] == LAST_READ_SOURCE
    flags = [told["available"] for told in got]
    assert flags == [True] * len(live) + [False] * len(old)


def test_a_page_that_waits_on_a_live_state_is_told_when_it_is_live_no_longer() -> None:
    robot = LiveRobot(stick)
    robot.on_link("")
    robot.on_state(arm_state(1), time.monotonic())
    began = time.monotonic()
    assert robot.answer()["available"] is True
    # No other state arrives: the wait ends when the readings pass the limit,
    # not a second later.
    told = robot.answer(after=1, timeout_s=WAIT_S)
    waited = time.monotonic() - began
    assert 0.9 < waited < robot_module.STALE_S + 0.1, waited
    assert (told["n"], told["available"]) == (1, False)
    # The last pose, as that: the readings were fresh when the state came.
    assert told["last_read"]["why"] == "no newer ROBOT_STATE for 1.0 s"
    assert told["pose_age_s"] == pytest.approx(1.0, abs=0.05)
    assert told["robot"]["source"] == LAST_READ_SOURCE
    # A little later the state itself is too old to be drawn at all.
    time.sleep(0.15)
    gone = robot.answer()
    assert (gone["available"], gone["last_read"], gone["robot"]) == (False, None, None)
    assert gone["why"].startswith("no ROBOT_STATE for 1.")


def test_a_state_that_does_not_say_how_old_its_readings_are_is_not_live() -> None:
    whole = arm_state(1)
    for health in ({}, {"reading_age_s": None}, {"reading_age_s": "0.01"}):
        robot = LiveRobot(stick)
        robot.on_state(dataclasses.replace(whole, health=health), time.monotonic())
        told = robot.answer()
        assert (told["available"], told["robot"], told["last_read"]) == (
            False,
            None,
            None,
        ), health
        assert told["why"] == (
            "the ROBOT_STATE does not say how old its joint readings are"
        )
        assert [one["q"] for one in told["joints"]] == pytest.approx(arm_pose(1))
    # The readings' own times count as well: where they are older than the
    # state says, the arm is as old as they are.
    late = [
        dataclasses.replace(read, t_start=read.t_start - 5.0, t_end=read.t_end - 5.0)
        for read in whole.joints
    ]
    robot = LiveRobot(stick)
    robot.on_state(dataclasses.replace(whole, joints=late), time.monotonic())
    told = robot.answer()
    assert told["available"] is False and told["pose_age_s"] == pytest.approx(
        5.02, abs=0.05
    )
    assert told["last_read"]["why"] == (
        "the robot app sends no newer reading; motion moving"
    )


def test_the_robot_apps_own_state_is_live_with_fresh_readings_and_not_with_old_ones() -> (
    None
):
    from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
    from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
    from test_state import World

    def answered(state: Any) -> dict[str, Any]:
        robot = LiveRobot(stick)
        # As it arrives: through the link message's bytes.
        robot.on_state(decode(state.TYPE, state.to_payload()), time.monotonic())
        return robot.answer()

    clock = FakeClock()
    world = World(clock, RobotAppConfig.default())
    world.read_every_joint()
    fresh = world.build()
    assert fresh.health["reading_age_s"] < 0.5
    told = answered(fresh)
    assert told["available"] is True and told["last_read"] is None
    # 45 s on, with no joint read meanwhile: the robot app's state carries the
    # same readings, each valid, and says how old they are.
    clock.advance(45.0)
    old = world.build()
    assert all(read.valid for read in old.joints)
    assert old.health["reading_age_s"] == pytest.approx(45.0, abs=0.5)
    told = answered(old)
    assert told["available"] is False
    assert told["pose_age_s"] == pytest.approx(old.health["reading_age_s"], abs=0.1)
    assert told["robot"]["source"] == LAST_READ_SOURCE
    assert LAST_READ_WHY.fullmatch(told["why"]) is not None, told["why"]


@apai
@experiment
def test_the_live_arm_is_the_experiments_own_geometry_of_the_newest_run_folder(
    apai_scratch: Path, experiment_dir: Path, viewer: Any
) -> None:
    runs = apai_scratch / "runs"
    runs.mkdir()
    model = ArmModel(viewer, viewer.Viewer(runs), check_s=0.0)
    with pytest.raises(ArmModelError, match=r"no run folder in .* holds resolved-"):
        model.geometry(arm_pose(1))
    assert robot_module.RESOLVED_CONFIG == agent_config.RESOLVED_CONFIG
    # A run without --record-debug: APAI's configuration and no recording.
    # The viewer lists no run; the arm is that run's all the same.
    first = runs / "20261116T120000Z-1"
    settings = agent_config.load_settings(settings_file(apai_scratch))
    cfg = agent_config.build_apai_config(settings, first, camera="mock")
    agent_config.write_resolved_config(first, cfg)
    before = files(runs)
    assert viewer.Viewer(runs).names() == []
    drawn = model.geometry(arm_pose(3))
    assert drawn == viewer.RunIndex(first).geometry(arm_pose(3))
    assert len(drawn["joints"]) == 8 and drawn["q"] == arm_pose(3)
    assert drawn["joints"] != model.geometry(arm_pose(4))["joints"]
    assert files(runs) == before  # nothing was written into the run folder
    # A newer run folder is the arm from then on: here the rig's, with the
    # gripper's collision volumes.
    rig = runs / "20261117T120000Z-2"
    cfg = agent_config.build_apai_config(
        agent_config.load_settings(SHIPPED), rig, camera="mock"
    )
    agent_config.write_resolved_config(rig, cfg)
    drawn = model.geometry(arm_pose(3))
    assert drawn == viewer.RunIndex(rig).geometry(arm_pose(3))
    assert [box["name"] for box in drawn["boxes"]] == ["housing", "fingers", "jaws"]
    # The model names the run it is of, and a LiveRobot hands that on.
    assert model.run() == rig.name
    robot = LiveRobot(model.geometry, model_run=model.run)
    robot.on_state(arm_state(5), time.monotonic())
    told = robot.answer()
    assert told["available"] is True and told["model_run"] == rig.name
    # A pose with another number of joints than the model's is not drawn.
    five = f"the ROBOT_STATE has 5 joints and the arm model of run {rig.name} has 6"
    with pytest.raises(ArmModelError, match=five):
        model.geometry(arm_pose(3)[:5])
    state = arm_state(6)
    robot.on_state(
        dataclasses.replace(state, joints=state.joints[:5]), time.monotonic()
    )
    told = robot.answer()
    assert (told["available"], told["robot"], told["why"]) == (False, None, five)
    assert told["model_run"] == rig.name and len(told["joints"]) == 5
    # A run folder whose configuration the viewer cannot build an arm from.
    broken = runs / "20261118T120000Z-3"
    broken.mkdir()
    (broken / robot_module.RESOLVED_CONFIG).write_text("arm: {}\n")
    with pytest.raises(ArmModelError, match=f"cannot build the arm of {broken.name}"):
        model.geometry(arm_pose(3))
    # As a LiveRobot answers with it.
    robot = LiveRobot(
        ArmModel(viewer, viewer.Viewer(first.parent), check_s=0.0).geometry
    )
    robot.on_state(arm_state(5), time.monotonic())
    told = robot.answer()
    assert told["available"] is False and "cannot build the arm" in told["why"]


@apai
@experiment
def test_with_the_live_arm_every_path_of_the_viewer_answers_as_the_viewer_alone_does(
    stage: Any, viewer: Any
) -> None:
    runs = stage.folder / "runs"
    run = runs / "20261116T120000Z-1"
    assert record(stage, run, RECORDED, QUICK) == launcher.EXIT_OK, stage.lines
    # An old recording: the viewer's "live" flag cannot change between two asks.
    events = run / "debug" / "events.jsonl"
    long_ago = time.time() - 3600
    os.utime(events, (long_ago, long_ago))
    name = run.name
    paths = [
        "/",
        f"/?run={name}",
        "/api/runs",
        "/api/bookmarks",
        "/api/state",
        "/api/image",
        f"/api/state?run={name}&at=0.5",
        f"/api/timing?run={name}",
        f"/timing?run={name}",
        "/api/state?run=unknown",
        "/api/nothing",
    ]
    before = files(runs), files(PACKAGE)
    with test_viewer.served(viewer, runs) as page:
        alone = answers(page.port, paths)
    own_viewer = viewer.Viewer(runs)
    own = viewer.handler(own_viewer, None)
    # As the launcher builds it: the arm of the newest run folder.
    robot = LiveRobot(ArmModel(viewer, own_viewer).geometry)
    asking = threading.Event()
    got: list[dict[str, Any]] = []
    with arm_showing(base=own, authorized=viewer.authorized, robot=robot) as arm:
        port = arm.port
        arm.feed.start()
        until(lambda: robot_json(port)["available"], "the live arm")

        def ask() -> None:  # a page draws state after state all the while
            told = robot_json(port)
            while not asking.is_set():
                told = robot_json(port, told["n"])
                got.append(told)

        reader = threading.Thread(target=ask, daemon=True)
        reader.start()
        try:
            until(lambda: len(got) >= 10, "ten states")
            with_live = answers(port, paths)
        finally:
            asking.set()
            reader.join(WAIT_S)
        sent_only = arm.wire.statuses()
    # The arm of each state is the one the viewer itself draws of those joints.
    index = viewer.RunIndex(run)
    for told in got[:10]:
        drawn = index.geometry(arm_pose(told["seq"]))
        assert told["robot"]["joints"] == drawn["joints"], told["seq"]
        assert told["robot"]["boxes"] == drawn["boxes"]
        assert len(drawn["joints"]) == 8
    # The viewer's own paths: the same bytes as the viewer alone gives.
    for path in paths:
        if urlsplit(path).path != "/":
            assert with_live[path] == alone[path], path
    for path in paths[:2]:
        assert with_live[path][:2] == alone[path][:2] == (200, alone[path][1])
        served_block(with_live[path][2], alone[path][2])
    # Nothing was written, and nothing but VIEW_STATUS went to the robot.
    assert (files(runs), files(PACKAGE)) == before
    assert len(sent_only) >= 1 and set(sent_only) == {NOT_LIVE}


@apai
@experiment
def test_the_viewers_own_token_rule_holds_for_the_robot_state_path(
    viewer: Any, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    token = viewer.access_token(tmp_path / "private" / "viewer.token")
    own = viewer.handler(viewer.Viewer(runs), token)
    with arm_showing(base=own, token=token, authorized=viewer.authorized) as arm:
        port = arm.port
        arm.feed.start()
        refused = get(port, "/api/runs")
        assert refused[0] == 401
        # The same answer as the viewer's own paths give without the token.
        for path in (
            ROBOT_PATH,
            f"{ROBOT_PATH}?after=3",
            f"{ROBOT_PATH}?token={token}",
        ):
            assert get(port, path) == refused, path
            assert get(port, path, Cookie="thor_viewer_session=wrong") == refused
        # With it, as the page's cookie or as a header: each state as it arrives.
        cookie = {"Cookie": f"thor_viewer_session={token}"}
        for headers in (cookie, {"Authorization": f"Bearer {token}"}):
            told = until(
                lambda headers=headers: (
                    (told := robot_json(port, **headers))["available"] and told
                ),
                "the live arm",
            )
            assert robot_json(port, told["n"], **headers)["n"] == told["n"] + 1


# The page's script with the live arm

# The first joint of the recorded poses of the stand-in viewer below.
RECORDED_Q0 = 9.0
# Runs the experiment's own page script and then the block's, as a browser
# does, on stand-ins for the page's elements, and tells what the page's own
# draw() was given each time it drew: the first joint of the arm in the page's
# state, null without an arm. The recorded state is the harness's: a finished
# recording of 10 s whose arm stands at RECORDED_Q0. Arguments: the block, the
# page, the scenario and, for real time, the viewer's address. Without one the
# clock, the timers and the answers to the live arm's requests are the
# harness's; with one those requests go to that viewer. Prints one JSON line.
ARM_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const [blockFile, pageFile, scenario, base] = process.argv.slice(2);
const inner = text => text.slice(text.lastIndexOf('<script>') + 8, text.lastIndexOf('</script>'));
const block = inner(fs.readFileSync(blockFile, 'utf8'));
// A page that lacks one of the names the live arm takes hold of.
const BARE = 'let data = null, follow = true, playing = false; function draw() {}';
const LACKING = {
  'no draw': 'let data = null, follow = true, playing = false;',
  'no state': 'let follow = true, playing = false; function draw() {}',
  'no scene': BARE, 'no text': BARE, 'no observer': BARE,
};
const page = scenario in LACKING ? LACKING[scenario] : inner(fs.readFileSync(pageFile, 'utf8'));
const gone = { 'no scene': 'scene', 'no text': 'status' }[scenario];
const RUNS = scenario === 'no recording' ? [] : ['20261116T120000Z-2', '20261115T100000Z-1'];
const START = '2026-11-17T10:00:00.000+00:00', LENGTH = 10;
// The recording's last event, on the wall clock.
const END = Date.parse(START) + 1000 * LENGTH;
// The harness's clock, in ms: 0.3 s after the recording's last event.
let clock = END + 300;
const real = Boolean(base);
const now = () => (real ? Date.now() : clock);
const timers = [];
const later = every => (work, ms) => { timers.push({ work, at: clock + (ms || 0), every: every ? ms : 0 }); };
const settle = async () => { for (let i = 0; i < 6; i++) await new Promise(done => setImmediate(done)); };

const listeners = {}, windowListeners = {}, made = [], elements = new Map();
const anything = new Proxy(function () {}, {
  get: (target, key) => (key === Symbol.toPrimitive ? () => 0 : anything),
  apply: () => anything,
});
const fire = kind => { for (const work of listeners[kind] || []) work(); };
function element(id) {
  let text = id === 'initial' ? 'null' : '';
  const own = {
    id, style: { display: '' }, value: '', src: '', checked: true, hidden: false,
    heard: {}, classes: new Set(), children: [], watchers: [], parentNode: null, nextSibling: null,
  };
  // What a browser tells a MutationObserver of an element's new text.
  Object.defineProperty(own, 'textContent', {
    get: () => text,
    set(value) { text = String(value); for (const watch of own.watchers) queueMicrotask(() => watch([])); },
  });
  own.classList = { toggle(name, on) { on ? own.classes.add(name) : own.classes.delete(name); } };
  own.addEventListener = (kind, work) => { (own.heard[kind] ||= []).push(work); };
  own.append = (...nodes) => { for (const node of nodes) { own.children.push(node); node.parentNode = own; } };
  own.insertBefore = node => { own.children.push(node); node.parentNode = own; };
  own.replaceChildren = (...nodes) => { own.children = nodes; };
  own.click = () => {
    if (own.onclick) own.onclick();
    for (const work of own.heard.click || []) work();
    fire('click');
  };
  return new Proxy(own, { get: (target, key) => (key in target ? target[key] : anything) });
}
const byId = id => elements.get(id) || elements.set(id, element(id)).get(id);
byId('photo').parentNode = element('panel');
byId('run').value = 'latest';
// The page's 3D scene in its box, in the page's first panel.
const scenePanel = element('scenePanel'), sceneBox = element('sceneBox');
scenePanel.append(sceneBox);
sceneBox.append(byId('scene'));
// Every draw() of the page begins by clearing its canvas.
const draws = [];
let context = null;
const ctx = new Proxy({}, {
  get: (target, key) => (key !== 'clearRect' ? () => {} : () => {
    const q = vm.runInContext('data && data.robot ? data.robot.q[0] : null', context);
    draws.push([now(), q]);
    if (SLOW && q !== null && q !== 9.0) clock += DRAW_MS;
  }),
  set: () => true,
});
byId('scene').getContext = () => ctx;

// The viewer's recorded state: a new one for each request, as the page parses it.
const recorded = (run, at) => ({
  run, seconds: at ?? LENGTH, duration: LENGTH, events: 50, live: false,
  map: { revision: 3, age_s: 0.2, voxel: 0.05, points: [[0.3, 0.1, 0.1]], frame_id: 7, query_present: true, utc: START },
  robot: { joints: [[0, 0, 0], [0.3, 0.9, 0.2]], q: [9.0, 0, 0, 0, 0, 0], boxes: [], source: 'recorded', sample_span_s: 0.02, age_s: 0.3, utc: START },
  failure: null, failures: [], recording: {}, recording_start_utc: START,
});
// The run whose arm model the viewer draws with: the newest run folder.
const MODEL_RUN = '20261116T120000Z-2';
// The viewer's answer with live state n: the first joint is n hundredths of a rad.
const liveState = n => ({
  n, available: true, why: '', age_s: 0.01, per_second: 20, wall_s: clock / 1000, seq: 1000 + n,
  stale_s: 1, pose_age_s: 0.03, last_read: null, model_run: MODEL_RUN,
  joints: [n / 100, 0.1, 0.1, 0, 0, 0].map((q, joint) => ({ joint, valid: true, q, deg: q * 180 / Math.PI, target: q })),
  target_q: [n / 100, 0.1, 0.1, 0, 0, 0],
  gripper: { valid: true, opening: 0.524, target: 0.6, load: 'empty', age_s: 0.01, fresh: true },
  owner: 'agent', stop_level: 'none', motion: 'moving', torque_on: true, torque: 'on (holding)',
  robot: { joints: [[0, 0, 0], [0.3, n / 100, 0.2]], q: [n / 100, 0.1, 0.1, 0, 0, 0], boxes: [], source: 'live', sample_span_s: 0.02, age_s: 0.02, utc: START },
});
// The viewer's answer with state n of a robot app that reads no joint any
// longer: the readings of `age` seconds ago, the first joint at FROZEN rad.
const FROZEN = 0.07;
const oldState = (n, age, why) => ({
  ...liveState(n), available: false, pose_age_s: age, last_read: { age_s: age, why },
  why: `the joints were last read ${age.toFixed(1)} s ago (${why})`,
  joints: [FROZEN, 0.1, 0.1, 0, 0, 0].map((q, joint) => ({ joint, valid: true, q, deg: q * 180 / Math.PI, target: null })),
  target_q: null,
  gripper: { valid: true, opening: 0.524, target: 0.6, load: 'empty', age_s: age, fresh: false },
  owner: 'nobody', motion: 'rest_unpowered', torque_on: false, torque: 'off',
  robot: { joints: [[0, 0, 0], [0.3, FROZEN, 0.2]], q: [FROZEN, 0.1, 0.1, 0, 0, 0], boxes: [], source: 'last reading', sample_span_s: 0.02, age_s: age, utc: START },
});
const noState = (n, why) => ({ n, available: false, why, age_s: 1.2, per_second: 0, wall_s: clock / 1000, robot: null });
// The live arm's requests, and the ones that wait for the harness's answer.
const asked = [], waiting = [];
function give(told) {
  for (const request of waiting.splice(0)) request.resolve({ ok: true, status: 200, json: async () => told });
}
function refuse() {
  for (const request of waiting.splice(0)) request.reject(Error('refused'));
}
// 'slow draws': a draw of the live arm takes DRAW_MS of the harness's clock,
// as on a large map, and the harness answers as the viewer does: with its
// newest state, at once when that is not the one the page names.
const SLOW = scenario === 'slow draws', DRAW_MS = 60;
let newest = 0;
function answer() {
  for (const request of waiting.splice(0)) {
    if (newest && request.after !== newest) request.resolve({ ok: true, status: 200, json: async () => liveState(newest) });
    else waiting.push(request);
  }
}
const globals = {
  document: {
    hidden: false,
    getElementById: id => (id === gone ? null : byId(id)),
    createElement() { const node = element(''); made.push(node); return node; },
    addEventListener(kind, work) { (listeners[kind] ||= []).push(work); },
  },
  window: { addEventListener(kind, work) { (windowListeners[kind] ||= []).push(work); } },
  location: { search: '', pathname: '/', origin: 'http://viewer' },
  history: { replaceState() {} },
  devicePixelRatio: 1,
  URLSearchParams,
  Option: function (text, value) { this.text = text; this.value = value; },
  setInterval: real ? setInterval : later(true),
  setTimeout: real ? setTimeout : later(false),
  Date: real ? Date : { now: () => clock, parse: Date.parse },
  MutationObserver: function (work) { this.observe = node => node.watchers.push(work); },
  fetch(url, options) {
    if (url.startsWith('/api/live/robot.json')) {
      asked.push(url);
      if (real) return fetch(base + url, options);
      const after = url.includes('after=') ? Number(url.split('after=')[1]) : null;
      const answered = new Promise((resolve, reject) => waiting.push({ resolve, reject, after }));
      if (SLOW) answer();
      return answered;
    }
    if (url.startsWith('/api/live.jpg')) return new Promise(() => {});
    if (url.startsWith('/api/live.json')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ available: false, reason: 'no camera' }) });
    }
    let value = {};
    if (url.startsWith('/api/runs')) value = RUNS;
    if (url.startsWith('/api/state')) {
      const query = new URLSearchParams(url.split('?')[1]);
      const run = query.get('run') === 'latest' ? RUNS[0] : query.get('run');
      if (!RUNS.includes(run)) {
        return Promise.resolve({ ok: false, status: 400, text: async () => 'Unknown recorded run' });
      }
      value = recorded(run, query.has('at') ? Number(query.get('at')) : null);
    }
    return Promise.resolve({ ok: true, status: 200, json: async () => value, text: async () => '' });
  },
};
if (scenario === 'no observer') delete globals.MutationObserver;
context = vm.createContext(globals);
vm.runInContext(page, context);
const before = made.length;
vm.runInContext(block, context);

const mine = id => made.find(node => node.id === id) || null;
const text = id => (mine(id) ? mine(id).textContent : null);
// Let `ms` of the harness's clock pass, 10 ms at a time. After each: what
// the page's canvas shows, the arm of the draw() that came last, and what the
// line under the scene says.
const shown = [], said = [];
async function pass(ms) {
  for (const end = clock + ms; clock < end;) {
    clock += 10;
    for (const timer of timers.slice()) {
      if (timer.at > clock) continue;
      if (timer.every) timer.at += timer.every;
      else timers.splice(timers.indexOf(timer), 1);
      timer.work();
    }
    await settle();
    shown.push(draws.length ? draws[draws.length - 1][1] : null);
    said.push(text('liveArmLine'));
  }
}
const switches = () => vm.runInContext('[follow, playing, data === null]', context);
const moved = () => { byId('timeline').value = '1.5'; byId('timeline').oninput(); fire('input'); };
const older = () => { byId('run').value = RUNS[1]; byId('run').onchange(); fire('change'); };

async function stepped() {
  const out = [];
  let n = 0;
  // States at the rig's pace: one every 50 ms.
  const states = async count => { for (let i = 0; i < count; i++) { give(liveState(++n)); await pass(50); } };
  const step = async (name, work, ms) => {
    // The first step holds what the two scripts did when they were loaded.
    const from = out.length ? [draws.length, shown.length, asked.length, said.length] : [0, 0, 0, 0];
    if (work) await work();
    if (ms) await pass(ms);
    const [follow, playing, none] = switches();
    out.push([name, {
      line: text('liveArmLine'), readings: text('liveArmReadings'), color: mine('liveArmLine').style.color || '',
      drawn: draws.slice(from[0]).map(one => one[1]), shown: shown.slice(from[1]),
      asked: asked.slice(from[2]), waiting: waiting.length, follow, playing, none,
      mode: byId('mode').textContent, error: String(byId('error').textContent).slice(0, 60),
      said: said.slice(from[3]),
    }]);
  };
  await step('the page is loaded', null, 700);
  if (SLOW) {
    // The viewer has a new state every 50 ms; the page asks as it can.
    timers.push({ work() { newest += 1; answer(); }, at: clock + 50, every: 50 });
    await step('the first second', null, 1000);
    await step('three seconds of live states', null, 3000);
    return out;
  }
  await step('a second of live states', () => states(20));
  if (scenario === 'no recording') {
    await step('the page draws by itself', () => byId('top').click());
    await step('the scene is dragged, zoomed and resized', () => {
      const scene = byId('scene');
      scene.onpointerdown({ clientX: 0, clientY: 0, pointerId: 1 });
      for (const work of [scene.onpointermove, ...scene.heard.pointermove]) work({ clientX: 9, clientY: 5, buttons: 1 });
      for (const work of [scene.onwheel, ...scene.heard.wheel]) work({ deltaY: 100, preventDefault() {} });
      for (const work of [context.window.onresize, ...windowListeners.resize]) work();
      // The pointer only passes over the scene: the page does not draw.
      scene.onpointerup();
      for (const work of [scene.onpointermove, ...scene.heard.pointermove]) work({ clientX: 9, clientY: 5, buttons: 0 });
    });
    await step('the service is gone', () => give(noState(n, 'cannot attach to the robot link service: gone')), 700);
    await step('the page draws by itself again', () => byId('top').click());
    return out;
  }
  // The stored recording does not go on: its map grows older.
  await step('two more seconds', () => states(40));
  await step('the timeline is moved', async () => { moved(); await pass(100); give(liveState(++n)); }, 1000);
  await step('Follow live is pressed', async () => { byId('live').click(); await pass(50); await states(10); });
  await step('a recording plays', async () => { byId('play').click(); await pass(100); give(liveState(++n)); }, 1000);
  await step('Follow live again', async () => { byId('live').click(); await pass(50); await states(10); });
  await step('an older run is picked', async () => { older(); await pass(100); give(liveState(++n)); }, 1000);
  await step('the newest run again', async () => { byId('live').click(); await pass(50); await states(10); });
  // The timing graph's page asks for a moment of the run: no event of the page's own.
  await step('a moment of the timing graph is picked', async () => {
    for (const work of windowListeners.message || []) {
      work({ origin: 'http://viewer', source: byId('timing-frame').contentWindow,
        data: { type: 'thor-timing-seek', run: RUNS[0], utc: '2026-11-17T10:00:02.000+00:00' } });
    }
    await pass(20);
    give(liveState(++n));
  }, 1000);
  await step('Follow live a third time', async () => { byId('live').click(); await pass(50); await states(10); });
  await step('the service is gone', () => give(noState(n, 'cannot attach to the robot link service: gone')), 1000);
  await step('the viewer is gone', () => refuse(), 1500);
  await step('the states are back', () => states(10));
  await step('the tab is hidden', () => { context.document.hidden = true; fire('visibilitychange'); give(liveState(++n)); }, 500);
  await step('the tab is shown', async () => { context.document.hidden = false; fire('visibilitychange'); await pass(50); await states(5); });
  // The viewer draws with the arm model of another run than the one on the page.
  await step('the arm model is of another run', async () => { give({ ...liveState(++n), model_run: '20261117T090000Z-9' }); await pass(50); });
  // The robot app goes on publishing, 20 states a second, and reads no joint:
  // each state carries the readings of 45 s ago and more.
  const frozen = async why => { for (let i = 0; i < 20; i++) { give(oldState(++n, 45 + i * 0.05, why)); await pass(50); } };
  await step('the motors are off', () => frozen('torque off: the joints are not read'));
  await step('the joints are read again', () => states(5));
  await step('the joint feedback is lost', () => frozen('joint feedback lost'));
  // A state arrived and no other follows: the page counts its age on.
  await step('fresh readings, then no answer', async () => { await states(3); await pass(1500); });
  return out;
}

// In real time, on a viewer's own states: what was drawn in each stretch.
function timed() {
  const began = Date.now(), marks = [[0, 'following']], lines = [];
  const at = (ms, name, work) => setTimeout(() => {
    lines.push([marks[marks.length - 1][1], text('liveArmLine'), text('liveArmReadings')]);
    if (work) work();
    marks.push([Date.now() - began, name]);
  }, ms);
  at(2500, 'the timeline is moved', moved);
  at(4000, 'following again', () => byId('live').click());
  at(6500, 'a recording plays', () => byId('play').click());
  at(8000, 'following a third time', () => byId('live').click());
  at(10500, 'an older run', older);
  at(12000, 'done', () => {
    console.log(JSON.stringify({ marks, lines, draws: draws.map(([at, q]) => [at - began, q]), asked: asked.length }));
    process.exit(0);
  });
}

if (scenario in LACKING) {
  settle().then(() => console.log(JSON.stringify({
    made: made.length - before, ids: made.map(node => node.id).filter(Boolean), asked, timers: timers.length,
  })));
} else if (real) {
  timed();
} else {
  stepped().then(out => console.log(JSON.stringify({ steps: out, ids: made.map(node => node.id).filter(Boolean) })));
}
"""


def run_arm_page(
    tmp_path: Path, experiment_dir: Path, scenario: str, base: str = ""
) -> dict[str, Any]:
    """What ARM_HARNESS tells of a scenario, on the experiment's own page script."""
    harness = tmp_path / "arm_harness.js"
    harness.write_text(ARM_HARNESS, encoding="utf-8")
    page = experiment_dir / "thor_viewer.html"
    command = [NODE, str(harness), str(live_module.PAGE_BLOCK_FILE), str(page)]
    done = subprocess.run(
        [*command, scenario, *([base] if base else [])],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def live_q0(n: int) -> float:
    """The first joint of the harness's live state n."""
    return n / 100


def live_arms(drawn: list[Any]) -> list[float]:
    """The live arms among what draw() was given, each once: a draw can repeat one."""
    live = [q for q in drawn if q is not None and q != RECORDED_Q0]
    return [q for at, q in enumerate(live) if at == 0 or q != live[at - 1]]


@experiment
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_page_draws_each_live_state_with_its_own_draw_only_while_it_follows(
    experiment_dir: Path, tmp_path: Path
) -> None:
    told = run_arm_page(tmp_path, experiment_dir, "run")
    steps = dict(told["steps"])
    assert [name for name, _ in told["steps"]] == list(steps)
    # Under the 3D scene: the line and the readings, behind the live picture's.
    assert told["ids"][-3:] == ["liveArm", "liveArmLine", "liveArmReadings"]
    recorded = RECORDED_Q0

    def live(first: int, last: int) -> list[float]:
        return [live_q0(n) for n in range(first, last + 1)]

    def only_recorded(step: str) -> dict[str, Any]:
        """A step in which the page drew its recorded arm and no live one."""
        seen = steps[step]
        assert seen["drawn"] and set(seen["drawn"]) == {recorded}, step
        assert set(seen["shown"]) == {recorded}, step
        return seen

    # Loaded: the page's own state, its recorded arm, and one request for the
    # newest live state, which is under way.
    loaded = steps["the page is loaded"]
    assert set(loaded["drawn"]) == {recorded} and loaded["mode"].startswith("RECORDED")
    assert (loaded["asked"], loaded["waiting"]) == ([ROBOT_PATH], 1)
    assert loaded["line"] == "arm: no live state: asking the viewer"
    assert (loaded["readings"], loaded["color"]) == ("", "#ffbd69")
    # A second of states, 20 as on the rig: each one was handed to the page's
    # own draw() as it came. The page's own tick drew its recorded arm twice in
    # that second, and each time the live arm was drawn over it at once: at no
    # moment did the canvas show the recorded arm.
    second = steps["a second of live states"]
    assert live_arms(second["drawn"]) == live(1, 20)
    assert second["drawn"].count(recorded) == 2
    assert set(second["shown"]) == set(live(1, 20))
    # Each request named the state the page had: the viewer answers with another.
    assert second["asked"] == [f"{ROBOT_PATH}?after={n}" for n in range(1, 21)]

    def map_age(line: str, drawn: int) -> float:
        """The map's age in the line of a live arm with `drawn` states a second."""
        found = re.fullmatch(
            rf"arm: live, {drawn} a second; map: (\d+\.\d) s old", line
        )
        assert found is not None, line
        return float(found.group(1))

    # The line under the scene: the arm is live, and the map beside it is the
    # recording's last: 0.2 s old at the recording's last event, 2 s ago.
    assert map_age(second["line"], 20) == pytest.approx(2.15, abs=0.06)
    assert second["color"] == ""
    assert second["readings"] == (
        "joints 11.5 5.7 5.7 0.0 0.0 0.0 deg · jaws 52 % open · owner agent · "
        "moving · torque on (holding)"
    )
    # Everything else is the page's own, from the recording.
    assert second["mode"] == loaded["mode"] and (second["follow"], second["none"]) == (
        True,
        False,
    )
    # The recording stays as it is: two seconds on, its map is that much older.
    later = steps["two more seconds"]
    assert live_arms(later["drawn"]) == live(21, 60)
    assert map_age(later["line"], 20) == pytest.approx(4.15, abs=0.06)

    # The timeline is moved: the recorded arm at once, as the page alone draws
    # it. The state that was under way is not drawn, and none is asked for.
    moved = only_recorded("the timeline is moved")
    assert (moved["line"], moved["readings"]) == ("arm: recorded", "")
    assert (moved["asked"], moved["waiting"], moved["follow"]) == ([], 0, False)
    assert moved["mode"].startswith("REPLAY")
    # "Follow live": the newest state at once, then each one as it comes.
    back = steps["Follow live is pressed"]
    assert back["asked"][0] == ROBOT_PATH and back["follow"] is True
    assert live_arms(back["drawn"]) == live(62, 71)
    assert back["shown"][-1] == live_q0(71)
    assert back["line"].startswith("arm: live, 10 a second; map: ")
    # A recording that plays, an older run and a moment picked in the timing
    # graph are the page's own as well.
    played = only_recorded("a recording plays")
    assert (played["line"], played["playing"], played["asked"]) == (
        "arm: recorded",
        True,
        [],
    )
    assert steps["Follow live again"]["shown"][-1] == live_q0(82)
    old = only_recorded("an older run is picked")
    assert old["line"] == "arm: recorded" and old["follow"] is True
    assert old["mode"].endswith("20261115T100000Z-1") and old["asked"] == []
    assert steps["the newest run again"]["shown"][-1] == live_q0(93)
    picked = only_recorded("a moment of the timing graph is picked")
    assert (picked["line"], picked["follow"], picked["asked"]) == (
        "arm: recorded",
        False,
        [],
    )
    assert steps["Follow live a third time"]["shown"][-1] == live_q0(104)

    # No state arrives: the page says so, with the viewer's reason, and is the
    # experiment's own: its recorded arm, at its own pace.
    gone = only_recorded("the service is gone")
    assert gone["line"] == (
        "arm: no live state: cannot attach to the robot link service: gone"
    )
    assert (gone["readings"], gone["color"]) == ("", "#ffbd69")
    assert gone["drawn"].count(recorded) >= 2 and gone["follow"] is True
    # It goes on asking, by the number it has.
    assert set(gone["asked"]) == {f"{ROBOT_PATH}?after=104"} and gone["waiting"] == 1
    lost = only_recorded("the viewer is gone")
    assert lost["line"] == "arm: no live state: no answer from the viewer: refused"
    assert len(lost["asked"]) == 1  # again after a second
    returned = steps["the states are back"]
    assert live_arms(returned["drawn"]) == live(105, 114)
    assert returned["line"].startswith("arm: live, 10 a second; map: ")
    # A hidden tab asks for nothing and draws no live state.
    hidden = steps["the tab is hidden"]
    assert hidden["asked"] == [] and hidden["waiting"] == 0
    assert set(hidden["drawn"]) <= {recorded} and hidden["line"] == "arm: recorded"
    shown = steps["the tab is shown"]
    assert shown["asked"][0] == ROBOT_PATH and shown["shown"][-1] == live_q0(120)
    # The line names the run whose arm model draws the arm when that is not
    # the run on the page. Up to here it was: nothing was named.
    names = [name for name, _ in told["steps"]]
    before = names[: names.index("the arm model is of another run")]
    assert len(before) == 16
    assert all(
        "arm model" not in line for name in before for line in steps[name]["said"]
    )
    other = steps["the arm model is of another run"]
    assert other["line"].startswith("arm: live, ")
    assert other["line"].endswith(" s old; arm model of run 20261117T090000Z-9")


# What the page shows when the joints' readings are old, per ARM_HARNESS's steps.
FROZEN_Q0 = 0.07
LAST_READ_LINE = (
    r"arm: last read (\d+(?:\.\d)?) s ago \({why}\); map: \d+(?:\.\d)? s old"
)


@experiment
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_page_never_calls_an_arm_live_whose_readings_are_old(
    experiment_dir: Path, tmp_path: Path
) -> None:
    steps = dict(run_arm_page(tmp_path, experiment_dir, "run")["steps"])
    recorded = RECORDED_Q0
    for step, why in (
        ("the motors are off", "torque off: the joints are not read"),
        ("the joint feedback is lost", "joint feedback lost"),
    ):
        old = steps[step]
        # 20 states in that second, each with the readings of 45 s ago: at no
        # moment did the line say "live". It says how old the readings are
        # and why, in the warning colour.
        assert len(old["asked"]) == 20, step
        pattern = re.compile(LAST_READ_LINE.format(why=re.escape(why)))
        ages = []
        for line in old["said"]:
            found = pattern.fullmatch(line)
            assert found is not None, (step, line)
            ages.append(float(found.group(1)))
        assert len(ages) == 100 and ages == sorted(ages), step
        assert 45.0 <= ages[0] <= 45.1 and ages[-1] == 46.0, (step, ages)
        assert old["color"] == "#ffbd69"
        # The pose the joints were last read at is on the canvas the whole
        # time: drawn once, and again over each draw of the page's own.
        assert set(old["shown"]) == {FROZEN_Q0}, step
        assert old["drawn"].count(recorded) == 2, step
        assert old["drawn"].count(FROZEN_Q0) == 3 and len(old["drawn"]) == 5, step
        # The readings are the last ones, the jaw opening with its age.
        assert old["readings"] == (
            "joints 4.0 5.7 5.7 0.0 0.0 0.0 deg · jaws: last read 46 s ago (52 % open) "
            "· owner nobody · rest_unpowered · torque off"
        )
    # Read again: live again, each state drawn.
    again = steps["the joints are read again"]
    assert live_arms(again["drawn"])[-5:] == [live_q0(n) for n in range(142, 147)]
    assert again["line"].startswith("arm: live, 5 a second; map: ")
    assert again["color"] == "" and "jaws 52 % open" in again["readings"]
    # A live state that no other follows: the page counts its age on by
    # itself and stops calling it live at the viewer's limit of 1 s.
    silent = steps["fresh readings, then no answer"]
    lines = silent["said"]
    assert lines[14].startswith("arm: live, ") and lines[80].startswith("arm: live, ")
    late = re.compile(
        LAST_READ_LINE.format(why=re.escape("the viewer sends no newer state"))
    )
    assert all(late.fullmatch(line) for line in lines[135:]), lines[135:]
    assert 1.2 <= float(late.fullmatch(silent["line"]).group(1)) <= 1.6
    assert silent["color"] == "#ffbd69"
    # The arm on the canvas is still that state's: the last pose there is.
    assert silent["shown"][-1] == live_q0(169)


@experiment
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_live_draws_take_at_most_about_half_of_the_pages_time(
    experiment_dir: Path, tmp_path: Path
) -> None:
    # Each draw of the live arm takes 60 ms here, as on a large map, and the
    # viewer has a new state every 50 ms.
    steps = dict(run_arm_page(tmp_path, experiment_dir, "slow draws")["steps"])
    long = steps["three seconds of live states"]
    live = live_arms(long["drawn"])
    # A draw, then as long again without one: about 7 states a second of the
    # 20, beside the two draws a second over the page's own.
    assert 18 <= len(live) <= 26, live
    drawing_ms = 60 * len([q for q in long["drawn"] if q != RECORDED_Q0])
    assert drawing_ms <= 0.6 * 3000, drawing_ms
    # Each state drawn was the viewer's newest when it was asked for: the
    # ones between were skipped, two of three.
    numbers = [round(q * 100) for q in live]
    gaps = np.diff(numbers)
    assert all(gap >= 1 for gap in gaps) and 2.0 <= float(np.mean(gaps)) <= 3.5, numbers
    # Each request named the state the page had.
    assert all(re.fullmatch(rf"{ROBOT_PATH}\?after=\d+", url) for url in long["asked"])
    # The line says so: how many of the arriving states are drawn, and why.
    found = re.fullmatch(
        r"arm: live, (\d+) of (\d+) a second \(a draw takes 60 ms\); map: .+",
        long["line"],
    )
    assert found is not None, long["line"]
    drawn, arriving = int(found.group(1)), int(found.group(2))
    assert 6 <= drawn <= 9 and 17 <= arriving <= 23, long["line"]
    # The recorded arm of the page's own tick was drawn over each time.
    assert RECORDED_Q0 not in long["shown"]


@experiment
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_without_a_recorded_run_the_page_still_draws_the_live_arm(
    experiment_dir: Path, tmp_path: Path
) -> None:
    told = run_arm_page(tmp_path, experiment_dir, "no recording")
    steps = dict(told["steps"])
    # The page alone has nothing to show: no run, no state, and its own words
    # for that.
    loaded = steps["the page is loaded"]
    assert set(loaded["drawn"]) <= {None} and loaded["none"] is True
    assert "Unknown recorded run" in loaded["error"]
    assert loaded["line"] == "arm: no live state: asking the viewer"
    # Each live state is drawn by the page's own draw(), and the page's state
    # is left as the page has it: none.
    second = steps["a second of live states"]
    assert second["drawn"] == [live_q0(n) for n in range(1, 21)]
    assert second["none"] is True and second["follow"] is True
    # No run is on the page: the line names the run whose arm model is used.
    assert second["line"] == (
        "arm: live, 20 a second; map: no recording; arm model of run 20261116T120000Z-2"
    )
    assert second["readings"].startswith("joints 11.5 5.7 5.7 0.0 0.0 0.0 deg · jaws")
    assert "Unknown recorded run" in second["error"]  # the page's own words
    # A draw of the page's own, as for "Top", has no arm: the live one is
    # drawn again at once.
    own = steps["the page draws by itself"]
    assert own["drawn"] == [None, live_q0(20)] and own["none"] is True
    # So it is while the scene is dragged, zoomed or resized.
    moved = steps["the scene is dragged, zoomed and resized"]
    assert moved["drawn"] == [None, live_q0(20)] * 3 and moved["none"] is True
    # Without a live state the page is the experiment's own again: no arm.
    gone = steps["the service is gone"]
    assert gone["line"] == (
        "arm: no live state: cannot attach to the robot link service: gone"
    )
    assert gone["drawn"] == [None] and gone["none"] is True
    assert steps["the page draws by itself again"]["drawn"] == [None]


@experiment
@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize(
    "lacking", ["no draw", "no state", "no scene", "no text", "no observer"]
)
def test_on_a_page_without_what_the_live_arm_needs_that_part_does_nothing(
    experiment_dir: Path, tmp_path: Path, lacking: str
) -> None:
    told = run_arm_page(tmp_path, experiment_dir, lacking)
    # The live picture's part made its own elements and its two timers; the
    # live arm's part made nothing and asked for nothing.
    assert told["made"] == 6 and "liveArm" not in told["ids"]
    assert (told["asked"], told["timers"]) == ([], 2)


@experiment
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_over_http_the_pages_script_draws_18_and_more_states_a_second_while_it_follows(
    experiment_dir: Path, tmp_path: Path
) -> None:
    import resource

    with arm_showing() as arm:
        arm.feed.start()
        until(lambda: robot_json(arm.port)["available"], "the live arm")
        spent = time.process_time(), time.monotonic()
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        told = run_arm_page(
            tmp_path, experiment_dir, "run", f"http://127.0.0.1:{arm.port}"
        )
        share = (time.process_time() - spent[0]) / (time.monotonic() - spent[1])
        node = resource.getrusage(resource.RUSAGE_CHILDREN)
        sent_only = arm.wire.statuses()
    node_s = node.ru_utime + node.ru_stime - children.ru_utime - children.ru_stime
    marks = [(at / 1000, name) for at, name in told["marks"]]
    draws = [(at / 1000, q) for at, q in told["draws"]]
    lines = {name: (line, readings) for name, line, readings in told["lines"]}
    rates = []
    for (since, name), (until_, _) in zip(marks, marks[1:], strict=False):
        drawn = [q for at, q in draws if since <= at < until_]
        live = [q for q in drawn if q is not None and q != RECORDED_Q0]
        if not name.startswith("following"):
            # A replay and an older run: the recorded arm, and no live state
            # from the moment of the switch.
            assert live == [] and drawn and set(drawn) == {RECORDED_Q0}, name
            assert lines[name] == ("arm: recorded", ""), name
            continue
        # Following: in a whole second, 18 and more different states of the
        # stand-in robot app's 20, each newer than the one before.
        second = live_arms([q for at, q in draws if since + 0.4 <= at < since + 1.4])
        assert second == sorted(set(second)) and len(second) >= 18, (name, second)
        assert all(ARM_FROM <= q < ARM_FROM + 200 * ARM_STEP for q in second)
        rates.append(len(second))
        found = re.fullmatch(
            r"arm: live, (\d+) a second; map: (\d+(?:\.\d)?) s old", lines[name][0]
        )
        assert found is not None and int(found.group(1)) >= 18, lines[name]
        assert lines[name][1].endswith(
            " deg · jaws 50 % open · owner agent · moving · torque on (holding)"
        )
    assert len(rates) == 3
    assert set(sent_only) == {NOT_LIVE}
    print(
        f"measured: the page's own draw() was given {rates} different live states "
        "in a whole second of each stretch of following, of 20 a second sent; "
        f"the viewer's side took {share:.2f} of a processor core (this process: "
        f"the live paths, the stand-in robot app and camera); node, with the "
        f"page's script, took {node_s:.1f} s of processor time in 12 s"
    )


def test_the_blocks_second_part_names_what_it_takes_hold_of_and_only_reads() -> None:
    text = BLOCK.decode()
    arm = text[text.index("// The live arm.\n") :]
    for piece in (
        "getElementById('scene')",
        "getElementById('status')",
        "typeof draw === 'function' && typeof data === 'object'",
        "typeof MutationObserver !== 'function'",
        f"const ROBOT = '{ROBOT_PATH}';",
        "`${ROBOT}?after=${after}`",
        # The page's own drawing: its state's arm, and its draw().
        "data.robot = arm.robot;",
        "draw();",
        # The four things the line under the scene says.
        "'arm: recorded'",
        "`arm: live, ${rate()}`",
        "`arm: last read ${seconds(age)} ago (${old ? old.why : "
        "'the viewer sends no newer state'})`",
        "`arm: no live state: ${why}`",
        "'no recording'",
        "`; arm model of run ${run}`",
        # Old jaw readings are shown as such.
        "`jaws: last read ${seconds(jaws.age_s)} ago (${open})`",
        # Each draw is timed, and as long must pass before the next state.
        "setTimeout(done, cost);",
        "await one;",
    ):
        assert piece in arm, piece
    # "live" is said in one place only, behind the check of the readings' age.
    assert arm.count("arm: live") == 1
    assert "if (!old && !(age > limit)) {\n        text = `arm: live, " in arm
    # It asks and never tells: no POST, no view token, and no geometry of its own.
    for piece in ("POST", "TOKEN", "Math.cos", "Math.sin", "project(", "lineTo"):
        assert piece not in arm, piece
    assert arm.count("fetch(") == 1


@experiment
def test_the_experiments_page_has_what_the_live_arm_takes_hold_of(
    experiment_dir: Path, viewer: Any
) -> None:
    page = (experiment_dir / "thor_viewer.html").read_text(encoding="utf-8")
    for piece in (
        '<div class="scene"><canvas id="scene"></canvas>',
        '<div id="status"></div>',
        "function draw(){",
        "arm(data.robot,'#58d5f7');",
        # render() writes its text, then draws: the block draws over it.
        "$('status').textContent=`Map age at selected time: ",
        "draw();}\nasync function tick(){",
        "data.recording_start_utc",
    ):
        assert piece in page, piece
    # The arm of a pose is computed by the viewer, in one routine.
    source = Path(viewer.__file__).read_text(encoding="utf-8")
    assert "    def geometry(self, q):" in source
    assert '**self.geometry(state["q"]),' in source
    assert '"age_s": stamp / 1e9 - commit["capture_t"],' in source


# The live arm in a real browser

# Goes behind the block in the pages of the live arm's browser tests. Each
# draw() of the page begins by clearing its canvas: the first joint of the
# arm in the page's state at that moment is kept. Five times a second, and
# before each step, it tells the test's own path /measured what is on the
# page and what was drawn since. The steps take the place of STEPS.
ARM_MEASURE = rb"""
<script>
(function () {
  const id = name => document.getElementById(name);
  const text = name => (id(name) ? id(name).textContent : null);
  const draws = [];
  const clear = CanvasRenderingContext2D.prototype.clearRect;
  CanvasRenderingContext2D.prototype.clearRect = function (...box) {
    if (this.canvas === id('scene')) {
      draws.push([Math.round(performance.now()), data && data.robot ? data.robot.q[0] : null]);
    }
    return clear.apply(this, box);
  };
  let phase = 'following';
  function report() {
    const told = {
      phase, at: Math.round(performance.now()), line: text('liveArmLine'),
      readings: text('liveArmReadings'), mode: text('mode'), none: data === null,
      draws: draws.splice(0),
    };
    fetch('/measured?' + encodeURIComponent(JSON.stringify(told)), {cache: 'no-store'}).catch(() => {});
  }
  setInterval(report, 200);
  for (const [after, name, work] of STEPS) {
    setTimeout(() => { report(); phase = name; work(); }, after);
  }
})();
</script>
"""
# The step during which the test ends the stand-in robot link service.
SERVICE_GOES = "the service goes"
# The steps during which the stand-in robot app goes on publishing and reads
# no joint, its motors off, and after which it reads them again.
MOTORS_OFF = "the motors are off"
READ_AGAIN = "the joints are read again"
# How old the joints' readings are when the motors go off in a browser test.
READ_AGO_S = 45.0
# (ms after the page's load, the name of what holds from then, what to do).
ARM_STEPS = (
    (
        5000,
        "the timeline is moved",
        "() => { const slider = id('timeline'); slider.value = slider.max / 2; "
        "slider.dispatchEvent(new Event('input', {bubbles: true})); }",
    ),
    (7000, "following again", "() => id('live').click()"),
    (10000, "a recording plays", "() => id('play').click()"),
    (12000, "following a third time", "() => id('live').click()"),
    (
        15000,
        "an older run",
        "() => { const runs = id('run'); runs.value = OLDER; "
        "runs.dispatchEvent(new Event('change', {bubbles: true})); }",
    ),
    (17000, "the newest run again", "() => id('live').click()"),
    (20000, MOTORS_OFF, "() => {}"),
    (23000, READ_AGAIN, "() => {}"),
    (26000, SERVICE_GOES, "() => {}"),
    (30500, "done", "() => {}"),
)
NO_RECORDING_STEPS = (
    (5000, "Top is pressed", "() => id('top').click()"),
    (8000, SERVICE_GOES, "() => {}"),
    (12500, "done", "() => {}"),
)


def cpu_seconds(root: int) -> dict[str, float]:
    """The processor time so far of a browser's processes, by Chrome's kind of process."""
    ticks = os.sysconf("SC_CLK_TCK")
    parents: dict[int, int] = {}
    spent: dict[int, float] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
        except OSError:
            continue
        parents[int(entry.name)] = int(fields[1])
        spent[int(entry.name)] = (int(fields[11]) + int(fields[12])) / ticks
    total: dict[str, float] = {}
    for pid, seconds in spent.items():
        ancestor = pid
        while ancestor not in (root, 0, 1) and ancestor in parents:
            ancestor = parents[ancestor]
        if ancestor != root:
            continue
        try:
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        except OSError:
            continue
        # Chrome writes a child's command line as one text.
        found = re.search(rb"--type=([a-z-]+)", command)
        kind = found.group(1).decode() if found else "browser"
        total[kind] = total.get(kind, 0.0) + seconds
    return total


class ArmBrowsed:
    """What a real browser drew of the arm on the viewer's page, step by step."""

    def __init__(self, steps: tuple[tuple[int, str, str], ...]) -> None:
        self.reports: list[tuple[float, dict[str, Any]]] = []  # of ARM_MEASURE
        self.lines: list[str] = []  # the launcher's
        self.cpu: dict[str, float] = {}  # shares of a processor core
        self._began = {"following": 0.0, **{name: ms / 1000 for ms, name, _ in steps}}
        self._ended = dict(
            zip(self._began, [*list(self._began.values())[1:], 0.0], strict=True)
        )

    def began(self, step: str) -> float:
        """When a step began, in seconds of the page's clock."""
        return self._began[step]

    def told(self, step: str, after_s: float = 0.0) -> list[dict[str, Any]]:
        """The page's reports of a step, from after_s into it."""
        since = self._began[step] + after_s
        return [
            told
            for _, told in self.reports
            if told["phase"] == step and told["at"] / 1000 >= since
        ]

    def draws(self, step: str, since_s: float, seconds: float) -> list[tuple[int, Any]]:
        """(ms of the page's clock, what draw() was given) in a stretch of a step."""
        since = self._began[step] + since_s
        end = min(since + seconds, self._ended[step])
        draws = [
            (at, q)
            for _, told in self.reports
            if told["phase"] == step
            for at, q in told["draws"]
        ]
        draws.sort(key=lambda draw: draw[0])
        return [(at, q) for at, q in draws if since <= at / 1000 < end]

    def drawn(self, step: str, since_s: float, seconds: float) -> list[Any]:
        """What draw() was given in a stretch of a step, in its order."""
        return [q for _, q in self.draws(step, since_s, seconds)]


def browse_arm(
    viewer: Any,
    experiment_dir: Path,
    runs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: tuple[tuple[int, str, str], ...],
    older: str = "",
) -> ArmBrowsed:
    """Headless Chrome on the viewer's own page until its last step, "done".

    The viewer is the launcher's, on a port the system picks, on the run
    folders of `runs`, with a stand-in for the robot link service that hands
    it a moving arm's ROBOT_STATE 20 times a second. That service ends when
    the page says that the step SERVICE_GOES began. The browser's profile
    lies under tmp_path.
    """
    browsed = ArmBrowsed(steps)
    listed = ", ".join(
        f"[{ms}, {json.dumps(name)}, {work}]" for ms, name, work in steps
    )
    listed = listed.replace("OLDER", json.dumps(older))
    measure = ARM_MEASURE.replace(b"STEPS", f"[{listed}]".encode())
    monkeypatch.setattr(live_module, "page_block", lambda: BLOCK + measure)
    real_handler = live_main.live_handler

    def measured(*args: Any, **kwargs: Any) -> type:
        class Measured(real_handler(*args, **kwargs)):  # type: ignore[misc]
            def do_GET(self) -> None:
                url = urlsplit(self.path)
                if url.path != "/measured":
                    super().do_GET()
                    return
                told = json.loads(unquote(url.query))
                browsed.reports.append((time.monotonic(), told))
                self.send_response(204)
                self.end_headers()

        return Measured

    monkeypatch.setattr(live_main, "live_handler", measured)
    wire = Wire()
    attach = Attaching(wire)
    launched = Launched(
        viewer,
        experiment_dir,
        runs,
        str(tmp_path / "no-service.sock"),
        Camera(hz=60.0),
        monkeypatch,
        attach=attach,
    )
    browsed.lines = launched.lines
    feed = Feed(wire)
    feed.start()
    chrome = None
    try:
        until(lambda: robot_json(launched.port)["available"], "the live arm")
        chrome = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--disable-extensions",
                "--disable-default-apps",
                "--password-store=basic",
                "--remote-debugging-port=0",
                f"--user-data-dir={tmp_path / 'chrome'}",
                "--window-size=1920,1080",
                f"http://127.0.0.1:{launched.port}/",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        end = time.monotonic() + 90.0
        mark: tuple[float, float, dict[str, float]] | None = None
        while True:
            assert time.monotonic() < end, "the browser did not reach its last step"
            assert chrome.poll() is None, "the browser ended"
            seen = [told for _, told in list(browsed.reports)]
            phases = {told["phase"] for told in seen}
            if "done" in phases:
                break
            if SERVICE_GOES in phases and wire.alive:
                attach.problem = ConnectionRefusedError("nobody listens")
                wire.die("the service ended")
            if READ_AGAIN in phases:
                feed.make = arm_state
            elif MOTORS_OFF in phases and feed.make is arm_state:
                read_at = time.monotonic() - READ_AGO_S
                feed.make = lambda k, at=read_at: frozen_state(k, at, "torque off")
            # Two seconds of following, from 1.5 s after the page's load: the
            # processor time of this process, with the viewer in it, and of
            # the browser's.
            newest = max((told["at"] for told in seen), default=0)
            if mark is None and newest >= 1500:
                mark = (time.monotonic(), time.process_time(), cpu_seconds(chrome.pid))
            elif mark is not None and not browsed.cpu and newest >= 3500:
                seconds = time.monotonic() - mark[0]
                now = cpu_seconds(chrome.pid)
                browsed.cpu = {
                    kind: (now[kind] - mark[2].get(kind, 0.0)) / seconds for kind in now
                }
                browsed.cpu["viewer"] = (time.process_time() - mark[1]) / seconds
            time.sleep(0.1)
    finally:
        if chrome is not None:
            chrome.terminate()
            try:
                chrome.wait(15)
            except subprocess.TimeoutExpired:
                chrome.kill()
                chrome.wait(15)
        feed.stop()
        launched.end()
    return browsed


ARM_LINE = re.compile(r"arm: live, (\d+) a second; map: (.+)")
ARM_READINGS = re.compile(
    r"joints (\d+\.\d) 5\.7 5\.7 0\.0 0\.0 0\.0 deg · jaws 50 % open · owner agent · "
    r"moving · torque on \(holding\)"
)
SERVICE_GONE = "arm: no live state: the robot link service is gone (the service ended)"
OLD_ARM_LINE = re.compile(
    r"arm: last read (\d+) s ago \(torque off: the joints are not read\); map: .+"
)
OLD_ARM_READINGS = re.compile(
    r"joints 28\.6 5\.7 5\.7 0\.0 0\.0 0\.0 deg · jaws: last read \d+ s ago "
    r"\(50 % open\) · owner nobody · rest_unpowered · torque off"
)


def follows_live(browsed: ArmBrowsed, step: str) -> tuple[int, int, list[str]]:
    """A step in which the page follows the newest run: the live arm is drawn.

    Returns how many different live states the page's draw() was given in a
    whole second, how often it drew in that second, and the map's ages its
    line named.
    """

    def is_live(q: Any) -> bool:
        return q is not None and q >= ARM_FROM - 1e-9

    draws = browsed.draws(step, 1.5, 1.0)
    drawn = [q for _, q in draws]
    live = [q for q in drawn if is_live(q)]
    states = [q for at, q in enumerate(live) if at == 0 or q != live[at - 1]]
    # 18 and more of the 20 states a second, each newer than the one before.
    assert len(states) >= 18, (step, states)
    turned = np.diff(states)
    assert all(step_ > 0 or step_ < -0.3 for step_ in turned), (step, states)
    # A draw of the page's own, with its recorded arm or with none, is drawn
    # over with the live arm in the same turn of the browser: before it paints.
    for (at, q), (then, after) in zip(draws, draws[1:], strict=False):
        assert is_live(q) or (is_live(after) and then - at <= 5), (step, draws)
    ages = []
    told = browsed.told(step, 1.5)
    assert len(told) >= 2, step
    for one in told:
        found = ARM_LINE.fullmatch(one["line"])
        assert found is not None and int(found.group(1)) >= 18, (step, one["line"])
        ages.append(found.group(2))
        found = ARM_READINGS.fullmatch(one["readings"])
        assert found is not None, (step, one["readings"])
        degrees = float(found.group(1))
        assert np.degrees(ARM_FROM) - 0.1 <= degrees <= np.degrees(ARM_FROM + 0.4) + 0.1
    return len(states), len(drawn), ages


def no_live_arm(browsed: ArmBrowsed, step: str, after_s: float = 0.0) -> list[Any]:
    """A step in which no live state is drawn; what draw() was given in it."""
    drawn = browsed.drawn(step, after_s, 60.0)
    assert not [q for q in drawn if q is not None and q > 0.25], (step, drawn)
    return drawn


@apai
@experiment
@pytest.mark.skipif(CHROME is None, reason="no Chrome is installed")
def test_in_a_real_browser_the_arm_follows_each_live_state_only_while_the_page_follows(
    stage: Any,
    viewer: Any,
    experiment_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = stage.folder / "runs"
    run = runs / "20261116T120000Z-2"
    assert record(stage, run, RECORDED, QUICK) == launcher.EXIT_OK, stage.lines
    # An older run for the run selector: the same recording under an older name.
    older = runs / "20261115T100000Z-1"
    shutil.copytree(run, older)
    ended_wall = (run / "debug" / "events.jsonl").stat().st_mtime
    before = files(runs), files(PACKAGE)
    opened_wall = time.time()
    browsed = browse_arm(
        viewer, experiment_dir, runs, tmp_path, monkeypatch, ARM_STEPS, older.name
    )
    assert (files(runs), files(PACKAGE)) == before  # nothing was written
    assert (
        f"viewer: live arm: on the page and at {ROBOT_PATH}: each ROBOT_STATE of "
        "that robot link service as it arrives; shown, not stored"
    ) in browsed.lines

    # Following the newest run, whose recording has ended: the arm is the live
    # one, redrawn with each state, and the line says how old the map is. That
    # is the time since the recording ended, and it grows.
    states, redraws, ages = follows_live(browsed, "following")
    assert all(age.endswith(" s old") for age in ages), ages
    seconds = [float(age.removesuffix(" s old")) for age in ages]
    # At least the time since the recording's file was last written; the line
    # names whole seconds from 10 s on.
    assert seconds[0] >= opened_wall - ended_wall - 0.6, (seconds, ended_wall)
    assert seconds[-1] <= time.time() - ended_wall + 60.0
    assert 1.0 <= seconds[-1] - seconds[0] <= 5.0, seconds
    for told in browsed.told("following", 1.5):
        assert told["mode"].endswith(f" · {run.name}"), told
    for step in ("the timeline is moved", "a recording plays", "an older run"):
        # The page's own: its recorded arm, which stands at its start pose,
        # at once and for the whole step.
        drawn = no_live_arm(browsed, step)
        stored = [q for q in drawn if q is not None]
        assert drawn and all(abs(q) < 0.05 for q in stored), (step, drawn)
        told = browsed.told(step)
        assert len(told) >= 5, step
        # A replay from its start has no arm before the first recorded pose.
        said = {(one["line"], one["readings"]) for one in told}
        assert said <= {("arm: recorded", ""), ("arm: none recorded", "")}, step
        if step != "a recording plays":
            assert stored and said == {("arm: recorded", "")}, step
    assert all(
        one["mode"].endswith(older.name) for one in browsed.told("an older run", 0.7)
    )
    for step in ("following again", "following a third time", "the newest run again"):
        follows_live(browsed, step)
    # The motors go off and the robot app goes on publishing, 20 states a
    # second, without reading a joint: the arm is not called live. The page
    # says how old the readings are and why, and draws the pose they give.
    old = browsed.told(MOTORS_OFF, 1.0)
    assert len(old) >= 6
    read_ages = []
    for one in old:
        found = OLD_ARM_LINE.fullmatch(one["line"])
        assert found is not None, one["line"]
        read_ages.append(int(found.group(1)))
        assert OLD_ARM_READINGS.fullmatch(one["readings"]) is not None, one["readings"]
    assert read_ages == sorted(read_ages)
    assert READ_AGO_S <= read_ages[0] <= read_ages[-1] <= READ_AGO_S + 4
    frozen = browsed.draws(MOTORS_OFF, 1.0, 60.0)
    assert len(frozen) >= 2

    def last_read(q: Any) -> bool:
        return q is not None and abs(q - ARM_FROM) < 1e-9

    for (at, q), (then, after) in zip(frozen, frozen[1:], strict=False):
        # The last read pose, or the page's own arm with that pose drawn over
        # it in the same turn; no state of the moving arm.
        assert last_read(q) or (last_read(after) and then - at <= 5), frozen
    # Readings that stand still are one pose: it is not drawn 20 times a second.
    assert len(frozen) <= 12, frozen
    follows_live(browsed, READ_AGAIN)
    # The service ends: the page says so and is the experiment's own, with its
    # recorded arm at its own pace.
    gone = browsed.told(SERVICE_GOES, 2.5)
    assert len(gone) >= 5
    assert {(one["line"], one["readings"]) for one in gone} == {(SERVICE_GONE, "")}
    left = no_live_arm(browsed, SERVICE_GOES, 2.5)
    assert len(left) >= 2 and None not in left
    cpu = browsed.cpu
    browser = sum(share for kind, share in cpu.items() if kind != "viewer")
    print(
        f"measured: of 20 ROBOT_STATEs a second the page's own draw() was given "
        f"{states} different ones in a whole second and drew {redraws} times; "
        f"the map was {ages[0]} to {ages[-1]}; processor cores: "
        f"{cpu.get('viewer', 0):.2f} this process (the viewer with its stand-in "
        f"camera and robot app), {cpu.get('renderer', 0):.2f} the browser's tab, "
        f"{browser:.2f} the whole browser"
    )


@apai
@experiment
@pytest.mark.skipif(CHROME is None, reason="no Chrome is installed")
def test_in_a_real_browser_the_arm_moves_without_any_recording(
    apai_scratch: Path,
    viewer: Any,
    experiment_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A run without --record-debug: APAI's configuration and nothing else.
    runs = apai_scratch / "runs"
    run = runs / "20261116T120000Z-1"
    settings = agent_config.load_settings(settings_file(apai_scratch))
    cfg = agent_config.build_apai_config(settings, run, camera="mock")
    agent_config.write_resolved_config(run, cfg)
    assert viewer.Viewer(runs).names() == []
    before = files(runs), files(PACKAGE)
    browsed = browse_arm(
        viewer, experiment_dir, runs, tmp_path, monkeypatch, NO_RECORDING_STEPS
    )
    assert (files(runs), files(PACKAGE)) == before
    # The page has no run and no state of its own, and the arm moves: each
    # live state is drawn, also after a draw of the page's own.
    for step in ("following", "Top is pressed"):
        states, redraws, ages = follows_live(browsed, step)
        # No run is on the page: the line names the run of the arm's model.
        assert set(ages) == {f"no recording; arm model of run {run.name}"}
        told = browsed.told(step, 1.5)
        assert all(one["none"] and one["mode"] == "Loading recording…" for one in told)
    # Without the service the page says so and draws no arm, as it does alone.
    gone = browsed.told(SERVICE_GOES, 2.5)
    assert len(gone) >= 5
    assert {(one["line"], one["readings"]) for one in gone} == {(SERVICE_GONE, "")}
    assert set(no_live_arm(browsed, SERVICE_GOES, 2.5)) <= {None}
    print(
        f"measured, without a recording: the page's own draw() was given {states} "
        f"different live states in a whole second and drew {redraws} times"
    )


# The live arm behind the real robot link service, and the start script


@apai
@experiment
@link
def test_on_the_rig_file_each_state_the_robot_app_publishes_reaches_the_viewers_path(
    rig: Rig,
) -> None:
    port = rig.launched.port
    # The rig's file publishes 20 ROBOT_STATEs a second.
    assert load_config(CONFIGS / "robot_app_rebot.yaml").state.publish_period_s == 0.05
    until(lambda: robot_json(port)["per_second"] >= 19, "a second of states")
    connection = HTTPConnection("127.0.0.1", port, timeout=WAIT_S)
    got = []
    try:
        told = robot_json(port, connection=connection)
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            told = robot_json(port, told["n"], connection)
            got.append(told)
    finally:
        connection.close()
    # Through the robot connection, the service and the viewer's attachment:
    # every state, in the robot app's order.
    numbers = [told["n"] for told in got]
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers))), numbers
    seqs = [told["seq"] for told in got]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs))) and len(seqs) >= 36
    state = rig.shows(lambda s: s.seq >= seqs[-1])
    for told in got:
        # The simulated arm rests, with the motors on, and nobody has it.
        assert [one["joint"] for one in told["joints"]] == list(range(6))
        assert all(one["valid"] for one in told["joints"])
        measured = [read.value for read in state.joints]
        assert [one["q"] for one in told["joints"]] == pytest.approx(measured, abs=1e-6)
        assert (told["owner"], told["stop_level"]) == (state.owner, state.stop_level)
        assert (told["motion"], told["torque_on"]) == (state.motion, True)
        assert 0.0 <= told["age_s"] < 0.2
        # This viewer's runs folder is empty: nothing names the arm's model,
        # so the readings are there and no arm is drawn.
        assert told["available"] is False and told["robot"] is None
        assert told["why"].startswith("no run folder in ")
        assert told["why"].endswith(
            " holds resolved-config.yaml, which names the arm's model"
        )


def test_the_start_script_says_that_the_arm_is_live_and_what_turns_it_off(
    stand_in: dict[str, Any], tmp_path: Path
) -> None:
    script = stand_in["script"]
    env = {**stand_in["env"], CAMERA_PR_ENV: str(tmp_path)}
    own = ["--no-auth", "--runs", str(tmp_path), "--port", "0"]
    assert script_run(script, *own, **env).stdout.splitlines()[1] == (
        "run_viewer.sh: live picture: on; the page shows the camera's pictures and "
        "the arm of each ROBOT_STATE as they arrive, and stores neither (--no-live "
        "turns it off, and the live arm with it)"
    )
    # Without the launcher the arm is the recorded one.
    off = (
        "); no live arm either: the page shows the stored pictures and the "
        "recorded arm of the runs only, and the robot is told of no live view"
    )
    assert script_run(script, "--no-live", *own, **env).stdout.splitlines()[1] == (
        f"run_viewer.sh: live picture: off (--no-live{off}"
    )
    assert script_run(script, *own, **stand_in["env"]).stdout.splitlines()[1] == (
        f"run_viewer.sh: live picture: off ({CAMERA_PR_ENV} is not set{off}"
    )
    # The live arm takes no argument of its own.
    told = script_run(script, "--help", **env).stdout
    for piece in (
        "The live arm is the arm\nof each ROBOT_STATE as it arrives, in the page's "
        "3D view. Neither is stored.",
        "--no-live          no live picture, no live arm and no live view",
        "./run_viewer.sh --no-live          # the experiment's viewer alone: nothing live",
    ):
        assert piece in told, piece
    assert "robot.json" not in told and "--live-arm" not in told


@apai
@experiment
def test_no_live_starts_the_viewer_without_the_live_arm(
    viewer: Any, experiment_dir: Path, camera_pr: dict[str, Any], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    env = launcher_env(experiment_dir, camera_pr["folder"], tmp_path)
    program = ViewerProcess(
        "--runs", str(runs), "--port", "0", "--no-auth", "--no-live", env=env
    )
    try:
        port = port_of(program)
        status, _, body = get(port, ROBOT_PATH)
        assert status == 400 and b"Unknown endpoint" in body
        assert b"liveArm" not in get(port, "/")[2]
        assert program.end() == 0, program.text
    finally:
        program.close()
    assert "live arm" not in program.text.replace("no live arm either", "")
