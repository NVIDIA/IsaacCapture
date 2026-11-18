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

The viewer's page with its live picture showing is the live view: the viewer
tells the robot app, with VIEW_STATUS link messages, how fresh the picture is
that the page loaded last has drawn and acknowledged. A takeover makes a person
the control owner; a human drive is one stretch of a person's motion. The robot
link service of these tests is one of their own, on a socket in a temporary
folder: none attaches to a service of this machine.
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
from contextlib import contextmanager
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
from isaaccapture_examples.agentic_remote.agent import launcher
from isaaccapture_examples.agentic_remote.agent.camera import frame_source
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import (
    BridgeSource,
    CameraError,
    StereoPair,
)
from test_frame_source import Publishing
from test_launcher import EXAMPLE, QUICK
from test_viewer import RECORDED, ViewerProcess, record, script_run

from conftest import APAI_SRC_ENV, CAMERA_PR_ENV

# The live picture is packed with OpenCV; without it this file is skipped.
cv2 = pytest.importorskip("cv2")

from isaaccapture_examples.agentic_remote.view import (  # noqa: E402
    __main__ as view_main,
)
from isaaccapture_examples.agentic_remote.viewer_live import (  # noqa: E402
    __main__ as live_main,
)
from isaaccapture_examples.agentic_remote.link_service import local  # noqa: E402
from isaaccapture_examples.agentic_remote.link_service.local import (  # noqa: E402
    connect_local,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (  # noqa: E402
    ViewStatus,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app import (  # noqa: E402
    app as app_module,
)
from isaaccapture_examples.agentic_remote.robot_app.config import (  # noqa: E402
    load_config,
)
from isaaccapture_examples.agentic_remote.view.model import (  # noqa: E402
    AGE_WHEN_NOT_LIVE_S,
    CLOCK_STEP_QUIET_S,
    ViewModel,
)
from isaaccapture_examples.agentic_remote.view.server import (  # noqa: E402
    TOKEN_HEADER,
    TOKEN_PLACE,
)
from isaaccapture_examples.agentic_remote.viewer_live import live as live_module  # noqa: E402
from isaaccapture_examples.agentic_remote.viewer_live import (  # noqa: E402
    view_link as link_module,
)
from isaaccapture_examples.agentic_remote.viewer_live.live import (  # noqa: E402
    BOUNDARY,
    PICTURE_PATH,
    SHOWN_PATH,
    STATE_PATH,
    STREAM_PATH,
    LivePicture,
    live_handler,
    with_page_block,
)
from isaaccapture_examples.agentic_remote.viewer_live.view_link import (  # noqa: E402
    ViewLink,
)
from link_bench import robot_state  # noqa: E402
from test_view_program import Browser  # noqa: E402

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
    **settings: Any,
) -> Iterator[tuple[LivePicture, int]]:
    """A started LivePicture on this camera, served with `base`; it and the port."""
    said = lines.append if lines is not None else lambda line: None
    live = LivePicture(lambda: camera, say=said, **settings)
    live.start()
    try:
        handler = live_handler(
            base, token, live, authorized, view, monotonic, this_machine
        )
        with serving(handler) as port:
            yield live, port
    finally:
        live.close()


def served_block(body: bytes, page: bytes) -> str:
    """The view token of a served page.

    The body is `page` with the block before its </body>, and nothing else;
    the block holds that load's view token where the file holds TOKEN_PLACE.
    """
    assert BLOCK.count(TOKEN_PLACE) == 1 and BLOCK.count(BLOCK_MARK) == 1
    found = VIEW_TOKEN.search(body)
    assert found is not None, "the served page holds no view token"
    at = page.rindex(b"</body>")
    block = BLOCK.replace(TOKEN_PLACE, found.group(1))
    assert body == page[:at] + block + page[at:]
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
            assert body == PAGE[:at] + BLOCK.replace(TOKEN_PLACE, b"") + PAGE[at:]
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
// A page for another machine holds no view token.
const script = block.slice(block.indexOf('<script>') + 8, block.lastIndexOf('</script>'))
  .replace('__VIEW_TOKEN__', scenario === 'watch' ? '' : 'token-of-this-load');
const listeners = {}, windowListeners = {}, intervals = [], timeouts = [];
// Every request, the picture requests that wait for their answer, the tokens
// they carried, the acknowledgements, and the pictures drawn.
const asked = [], waiting = [], tokens = [], posts = [], drawn = [];
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
  setInterval(work) { intervals.push(work); },
  setTimeout(work) { timeouts.push(work); },
  fetch(url, options) {
    asked.push(url);
    if (url.startsWith('/api/live.jpg')) {
      tokens.push((options.headers || {})['X-View-Token'] ?? null);
      return new Promise(resolve => waiting.push({ url, resolve }));
    }
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
    style: picture.style, size: [picture.width, picture.height],
    drawn: [...drawn], posts: [...posts], waiting: waiting.map(request => request.url),
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
    steps: out, asked, tokens, observed: observing && [observing.nodes, observing.options],
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
  await step('and stays replaced');
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
    lost = shows("the viewer is gone", stored, "no live picture", see_below)
    assert lost["note"] == "no answer from the viewer: refused"
    returned = shows(
        "the viewer is back", live, "Map picture", "robot not told: see below"
    )
    assert returned["note"].endswith(
        " The robot is not told of this live view: nobody listens."
    )
    # A newer load of the page has the valid view token: this one says so,
    # shows the stored picture and asks for no more pictures.
    reload = "page loaded again: see below"
    replaced = shows(
        "the page was loaded again elsewhere", stored, "no live picture", reload
    )
    assert replaced["note"] == (
        "This page was loaded again, here or in another tab: reload it for the "
        "live picture."
    )
    final = shows("and stays replaced", stored, "no live picture", reload)
    assert final["waiting"] == [] and final["drawn"] == [7, 8, 10, 11, 12, 13]
    assert final["note"] == replaced["note"]
    # Every request for a picture carried this load's view token, and every
    # other request is one of the block's three.
    assert set(told["tokens"]) == {token} and len(told["tokens"]) >= 8
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


def test_the_viewers_first_values_are_the_live_view_programs() -> None:
    # What the live view program of view/ runs with by its shipped file.
    shipped = yaml.safe_load((CONFIGS / "view.yaml").read_text(encoding="utf-8"))
    settings = view_main.ViewSettings()
    status = shipped["status"]
    assert link_module.PERIOD_S == settings.period_s == status["period_s"]
    assert settings.ack_timeout_s == status["ack_timeout_s"]
    assert settings.state_max_age_s == status["state_max_age_s"]
    model = LivePicture(lambda: Camera()).model
    assert type(model) is ViewModel
    assert model._ack_timeout_s == status["ack_timeout_s"]
    assert model._state_max_age_s == status["state_max_age_s"]
    # The attach is tried again with the camera's pauses, in the role and
    # with the time limit of the live view program.
    assert link_module.RETRY_PAUSES_S == live_module.RETRY_PAUSES_S
    assert link_module.ATTACH_TIMEOUT_S == view_main.ATTACH_TIMEOUT_S
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
    # Without a path: the one the console and the live view program attach to.
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
        assert link.state()["why"] == view_main.SECOND_VIEW
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


class Twins:
    """The live view program of view/ and the viewer, side by side.

    Both get the same ROBOT_STATEs and the same stereo pairs at the same times
    of one clock that the test sets, and a page each that does the same. Each
    sends its VIEW_STATUS on a Wire of its own, every 0.02 s.
    """

    def __init__(self) -> None:
        self.clock = SetClock(1000.0)
        settings = view_main.ViewSettings(http_port=0, period_s=0.02, fps=15.0)
        # The live view program.
        self.old_wire, self.old_pairs = Wire(), Pairs()
        self.old_model = ViewModel(
            ack_timeout_s=settings.ack_timeout_s,
            state_max_age_s=settings.state_max_age_s,
        )
        self.old = view_main.ViewProgram(
            settings,
            self.old_wire,
            self.old_model,
            self.old_pairs,
            False,
            self.clock,
            lambda line: None,
        )
        # The viewer: its live picture, its handler and its link.
        self.new_wire, self.new_pairs = Wire(), Pairs()
        self.live = LivePicture(lambda: self.new_pairs, fps=15.0, say=lambda line: None)
        self.link = ViewLink(
            self.live.model,
            attach=Attaching(self.new_wire),
            say=lambda line: None,
            period_s=0.02,
            monotonic=self.clock.monotonic,
        )
        handler = live_handler(
            PlainViewer, None, self.live, bearer, self.link.state, self.clock.monotonic
        )
        self._serving = serving(handler)
        self.old.start()
        self.live.start()
        self.port = self._serving.__enter__()
        self.link.start()
        self.old_pages: list[Browser] = []
        self.new_pages: list[tuple[Tab, http.client.HTTPConnection]] = []
        self.compared: list[tuple[str, ViewStatus]] = []

    def close(self) -> None:
        for browser in self.old_pages:
            browser.close()
        for _, connection in self.new_pages:
            connection.close()
        self.link.close()
        self._serving.__exit__(None, None, None)
        self.live.close()
        self.old.close()

    # What happens to both

    def state(self, seq: int) -> None:
        """A ROBOT_STATE of this moment arrives at both."""
        message = thor_state(seq, self.clock.now)
        for wire, model in (
            (self.old_wire, self.old_model),
            (self.new_wire, self.live.model),
        ):
            wire.give(message)
            until(
                lambda model=model: (
                    model.thor_wall_now(self.clock.now) == message.clock_wall_s
                ),
                "the ROBOT_STATE",
            )

    def picture(self, seq: int, **check: Any) -> None:
        """A stereo pair of this moment arrives at both; it is picture `seq`."""
        for pairs, model in (
            (self.old_pairs, self.old_model),
            (self.new_pairs, self.live.model),
        ):
            pairs.put(pair_at(seq, self.clock.now, **check))
            until(
                lambda model=model: (model.picture() or (0,))[0] == seq,
                f"picture {seq}",
            )
        assert self.old_model.picture() == self.live.model.picture()  # the same JPEG

    def load(self) -> int:
        """A page is loaded on both; its index."""
        browser = Browser(self.old.port)
        browser.reload()
        tab = Tab(self.port)
        tab.load()
        self.old_pages.append(browser)
        self.new_pages.append((tab, tab._http()))
        return len(self.old_pages) - 1

    def fetch(self, page: int) -> tuple[int | None, int | None]:
        """That page fetches the newest picture on both; the two numbers."""
        tab, connection = self.new_pages[page]
        old_seq, _ = self.old_pages[page].frame()
        _, new_seq, _ = tab.fetch(connection, after=0)
        return old_seq, new_seq

    def shown(self, page: int, seq: int, visible: bool = True) -> tuple[int, int]:
        """That page acknowledges picture `seq` on both; the two status codes."""
        tab, connection = self.new_pages[page]
        word = {"seq": seq, "visible": visible, "token": self.old_pages[page].token}
        return (
            self.old_pages[page].shown_raw(word),
            tab.shown(connection, seq, visible),
        )

    def same(self, situation: str) -> ViewStatus:
        """Both send the same bytes in this situation; the VIEW_STATUS they are."""
        counts = len(self.old_wire.sent), len(self.new_wire.sent)
        until(
            lambda: (
                len(self.old_wire.sent) >= counts[0] + 2
                and len(self.new_wire.sent) >= counts[1] + 2
            ),
            f"VIEW_STATUS from both in: {situation}",
        )
        old, new = self.old_wire.sent[-1], self.new_wire.sent[-1]
        assert old == new, (situation, decode(*old), decode(*new))
        assert old[0] == ViewStatus.TYPE
        status = decode(*old)
        self.compared.append((situation, status))
        return status


def test_the_viewer_sends_the_bytes_the_live_view_program_sends() -> None:
    twins = Twins()
    try:
        clock = twins.clock
        assert twins.same("nothing has arrived") == NOT_LIVE
        twins.state(1)
        assert twins.same("a ROBOT_STATE, no picture") == NOT_LIVE
        # Past the quiet time after the first ROBOT_STATE.
        clock.now = 1004.0
        twins.state(2)
        twins.picture(1)
        assert twins.same("a picture and no page") == NOT_LIVE
        page = twins.load()
        assert twins.same("a page that fetched nothing") == NOT_LIVE
        assert twins.fetch(page) == (1, 1)
        assert twins.same("a picture fetched and not acknowledged") == NOT_LIVE
        assert twins.shown(page, 1) == (204, 204)
        # Live: the picture's 0.05 s since its exposure, sent at the Thor's
        # monotonic clock of this moment.
        live = twins.same("a picture drawn in a visible tab")
        assert live == ViewStatus(live=True, age_s=live.age_s, sent_mono_s=504.0)
        assert live.age_s == pytest.approx(0.05, abs=1e-6)
        # The same picture a little later, with a ROBOT_STATE of that moment.
        clock.now = 1004.4
        twins.state(3)
        older = twins.same("the same picture 0.4 s later")
        assert older.live and older.age_s == pytest.approx(0.45, abs=1e-6)
        assert older.sent_mono_s == pytest.approx(504.4)
        # The live picture leaves the screen, or the tab is hidden.
        assert twins.shown(page, 1, visible=False) == (204, 204)
        assert twins.same("the picture shows no longer") == NOT_LIVE
        assert twins.shown(page, 1) == (403, 403)
        assert twins.same("'it shows' for the picture from before") == NOT_LIVE
        twins.picture(2)
        assert twins.fetch(page) == (2, 2)
        assert twins.shown(page, 2) == (204, 204)
        assert twins.same("the next picture, drawn and acknowledged").live
        # The page is loaded again: the page before counts no longer.
        other = twins.load()
        assert twins.same("a newer load of the page") == NOT_LIVE
        assert twins.shown(page, 2) == (403, 403)
        assert twins.fetch(page) == (None, None)
        assert twins.same("a word from the page that was not loaded last") == NOT_LIVE
        assert twins.fetch(other) == (2, 2)
        assert twins.shown(other, 2) == (204, 204)
        assert twins.same("the page loaded last draws and acknowledges").live
        # A picture whose times nobody measured is never live.
        twins.picture(3, invented_times=True)
        assert twins.fetch(other) == (3, 3)
        assert twins.shown(other, 3) == (204, 204)
        assert twins.same("a picture with invented times") == NOT_LIVE
        twins.picture(4)
        assert twins.fetch(other) == (4, 4)
        assert twins.shown(other, 4) == (204, 204)
        assert twins.same("a measured picture again").live
        # The camera gives nothing more and the page says nothing more: past
        # the acknowledgement's time limit the view is not live.
        clock.now = 1005.5
        twins.state(4)
        assert twins.same("no word for 1.1 s") == NOT_LIVE
        # No ROBOT_STATE for longer than its limit: the Thor's clock is unknown.
        twins.picture(5)
        assert twins.fetch(other) == (5, 5)
        assert twins.shown(other, 5) == (204, 204)
        assert twins.same("a picture acknowledged again").live
        clock.now = 1006.6
        assert twins.same("no ROBOT_STATE for 1.1 s") == NOT_LIVE
    finally:
        twins.close()
    # Both end with the word that the view is over.
    assert twins.old_wire.sent[-1] == twins.new_wire.sent[-1]
    assert decode(*twins.new_wire.sent[-1]) == NOT_LIVE
    # Nothing but VIEW_STATUS went out on either side.
    assert {kind for kind, _ in twins.old_wire.sent + twins.new_wire.sent} == {
        ViewStatus.TYPE
    }
    assert sum(status.live for _, status in twins.compared) == 6


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

    It runs the experiment's viewer on a port the system picks, on a stand-in
    camera, with --link-socket at the test's robot link service; `attach`
    stands in for the attachment to one. end() ends it as Ctrl+C does.
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
    ) -> None:
        self.lines: list[str] = []
        self.code: int | None = None
        self._attach = attach
        servers = self._servers = []

        class Server(viewer.ThreadingHTTPServer):  # type: ignore[misc, name-defined]
            def __init__(inner, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                servers.append(inner)

        # The viewer's own server class, kept so that the test can end it.
        monkeypatch.setattr(viewer, "ThreadingHTTPServer", Server)
        argv = ["--experiment-dir", str(experiment_dir), "--link-socket", link_socket]
        argv += ["--runs", str(runs), "--port", "0", "--no-auth"]
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


# The launcher's settings and its camera frame bridge


def test_the_first_values_are_the_live_views_and_the_camera_receivers() -> None:
    assert live_module.DEFAULT_FPS == 30.0 and live_module.FPS_RANGE == (1.0, 60.0)
    assert live_module.DEFAULT_WIDTH_PX == 960 and live_module.JPEG_QUALITY == 80
    assert live_module.WIDTH_RANGE_PX == (64, 4096)
    receiver = yaml.safe_load((CONFIGS / "camera_receiver_rebot.yaml").read_text())
    view = yaml.safe_load((CONFIGS / "view.yaml").read_text())
    assert live_module.RECEIVER_SOCKET == receiver["ipc"]["socket_path"]
    assert live_module.RECEIVER_SOCKET == view["camera"]["receiver_socket"]
    assert view["picture"]["jpeg_quality"] == live_module.JPEG_QUALITY
    assert view["picture"]["width_px"] == live_module.DEFAULT_WIDTH_PX
    parser = live_main._parser()
    args = parser.parse_args([])
    assert (args.live_fps, args.live_width) == (30.0, 960)
    assert args.receiver_socket == live_module.RECEIVER_SOCKET


def test_the_launchers_bridge_is_the_live_views_on_gpu_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere")
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
    # What the live view starts for --source link, with the shipped view.yaml.
    settings = view_main.ViewSettings(
        bridge_python=str(python),
        camera_pr=str(camera_pr),
        receiver_socket="/tmp/r.sock",
        state_dir=tmp_path,
    )
    views, _ = view_main.make_source(settings, "link", None, time)
    for name in ("_command", "_receiver_socket", "_streams", "_max_skew_ms", "_size"):
        assert getattr(source, name) == getattr(views, name), name
    assert (source._devices, source._rotation) == (views._devices, views._rotation)
    assert source._command == [str(python), "-m", frame_source.BRIDGE_MODULE]
    env = source._environment()
    assert env == views._environment()
    assert (env["CUDA_DEVICE_ORDER"], env["CUDA_VISIBLE_DEVICES"]) == (
        "PCI_BUS_ID",
        "0",
    )
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
    low, high = live_module.FPS_RANGE
    fps = f"{low:g} to {high:g}. Default: {live_module.DEFAULT_FPS:g}"
    assert f"#   --live-fps N       the most live pictures a second, {fps}" in script
    low, high = live_module.WIDTH_RANGE_PX
    width = f"{low} to {high}. Default: {live_module.DEFAULT_WIDTH_PX}"
    assert f"the live picture's greatest width, {width}" in script
    assert f"-m {MODULE} \\\n" in script
    receiver = yaml.safe_load((CONFIGS / "camera_receiver_rebot.yaml").read_text())
    assert receiver["ipc"]["socket_path"] == live_module.RECEIVER_SOCKET


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
