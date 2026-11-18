# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live picture: the camera's newest picture for the experiment's viewer.

LivePicture reads stereo pairs from a camera source on one thread; a second
thread packs the left one as JPEG, once for every browser, and keeps it as the
newest picture. live_handler() adds to the viewer's request handler:

GET /api/live.mjpg    the newest picture again and again, as one
                      multipart/x-mixed-replace response
GET /api/live.json    whether there is a live picture, how many a second, how
                      old, and what the robot is told of the live view
GET /api/live.jpg     the newest picture once it is newer than ?after=N, for
                      the page that sends the valid view token as X-View-Token;
                      its number in X-Picture-Seq (204 without a newer one in
                      PICTURE_WAIT_S, 403 without the valid token). With
                      &watch=1 it needs no view token and counts for nothing
POST /api/live/shown  {"seq": N, "visible": true|false, "token": "..."}: the
                      page has drawn picture N (403 when it does not count)
GET /                 the viewer's own page with page_block.html before </body>

The page with its live picture showing is the live view. The view token is the
secret a page is served with: each GET / from this machine that gives the page
makes a new one the only valid one, so only the page opened last on this
machine counts. What that page drew and acknowledged goes into the ViewModel of
view/model.py, from which view_link.py sends the robot VIEW_STATUS.

A page opened from another machine is a watch-only page: it is served without
a view token and fetches its pictures with &watch=1. It replaces no page, and a
request from another machine for a counted picture or with an acknowledgement
is refused, whatever token it names.

Every live path follows the viewer's access rule. No picture is written to disk
and none but the newest is kept.
"""

from __future__ import annotations

import contextlib
import http
import io
import ipaddress
import json
import logging
import os
import re
import select
import socket
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..agent.camera.frame_source import (
    CAMERA_PR_ENV,
    BridgeSource,
    CameraError,
    PairSource,
    StereoPair,
)
from ..view.__main__ import ViewSettings, encode_picture
from ..view.model import ViewModel
from ..view.server import MAX_SHOWN_BYTES, TOKEN_HEADER, TOKEN_PLACE

_log = logging.getLogger("isaaccapture.examples.agentic_remote.viewer_live")

PROGRAM = "viewer"
STREAM_PATH = "/api/live.mjpg"
STATE_PATH = "/api/live.json"
PICTURE_PATH = "/api/live.jpg"
SHOWN_PATH = "/api/live/shown"
BOUNDARY = "livepicture"
PAGE_BLOCK_FILE = Path(__file__).with_name("page_block.html")
# The page's end, before which the block goes.
PAGE_END = b"</body>"

_VIEW = ViewSettings()
DEFAULT_FPS = 30.0
FPS_RANGE = (1.0, 60.0)
DEFAULT_WIDTH_PX = _VIEW.width_px
WIDTH_RANGE_PX = (64, 4096)
JPEG_QUALITY = _VIEW.jpeg_quality
# The camera receiver's socket, as view.yaml and camera_receiver_rebot.yaml name it.
RECEIVER_SOCKET = _VIEW.receiver_socket
# Pictures a second while no browser has the stream open or asks for pictures.
IDLE_FPS = 1.0
# How long GET /api/live.jpg waits for a newer picture.
PICTURE_WAIT_S = 1.0
# A page counts as asking for pictures this long after its last request.
FETCH_WATCH_S = 2.0
# A longer body of POST /api/live/shown than MAX_SHOWN_BYTES is read and
# dropped up to this size; above it the connection is closed instead.
_DRAIN_BYTES = 1 << 20
# A camera source that failed is opened again after a pause: the first of these
# after pictures arrived, the next after each further failed try, then the last.
RETRY_PAUSES_S = (5.0, 10.0, 20.0, 40.0, 60.0)
# There is no live picture when the newest one is older than this.
STALE_S = 2.0
# How long one request for a stereo pair may wait.
PAIR_WAIT_S = 1.0
# A browser that takes no bytes of the stream for this long is dropped.
CLIENT_SEND_TIMEOUT_S = 10.0
# How often a waiting stream looks whether its browser is still there.
_CLIENT_CHECK_S = 0.5
# The stream's send buffer: a few pictures. A larger one would hold pictures
# back for a browser that reads slowly instead of skipping them.
_SEND_BUFFER_BYTES = 128 * 1024
# What /api/live.json says of the live view when nothing reports it to the robot.
NOT_REPORTED: Mapping[str, Any] = {
    "attached": False,
    "why": "this viewer does not report the live view",
    "live": None,
    "age_s": None,
}
# How long close() waits for the camera thread.
_CLOSE_S = 30.0
# The newest picture's arrival counts as its time unless it is no past time.
_MAX_ARRIVAL_AGE_S = 60.0

Say = Callable[[str], None]
SourceFactory = Callable[[], PairSource]
# Whether the browser at the other end of a connection runs on this machine.
ThisMachine = Callable[[Any], bool]


def _say(text: str) -> None:
    print(text, flush=True)


def use_one_encoder_thread() -> None:
    """Keep OpenCV on the calling thread.

    Its worker threads spin between two pictures: on a machine with many cores
    one encode then costs many times the CPU time it takes on one thread.
    """
    import cv2

    cv2.setNumThreads(1)


def bridge_source_factory(
    receiver_socket: str,
    run_dir: str | os.PathLike[str],
    environ: Mapping[str, str] | None = None,
) -> SourceFactory:
    """Makes the camera frame bridge the live view starts for --source link.

    The bridge runs under the camera consumer's Python, in the folder
    AGENTIC_REMOTE_CAMERA_PR names; its log and its socket go into run_dir.
    """
    environ = os.environ if environ is None else environ

    def make() -> PairSource:
        camera_pr = environ.get(CAMERA_PR_ENV, "")
        if not camera_pr:
            raise CameraError(f"{CAMERA_PR_ENV} is not set")
        python = Path(camera_pr) / "cuda_ipc_consumer" / ".venv" / "bin" / "python"
        if not python.is_file():
            raise CameraError(f"the camera consumer's Python is not there: {python}")
        return BridgeSource(
            python=python,
            receiver_socket=receiver_socket,
            run_dir=run_dir,
            camera_pr=camera_pr,
            streams=_VIEW.streams,
            max_skew_ms=_VIEW.max_skew_ms,
        )

    return make


class LivePicture:
    """The camera's newest picture as JPEG, encoded once for every browser.

    A source that does not open, or that ends, is made and opened again after a
    pause: retry_pauses_s one after the other, the last one from then on, and
    the first one again once pictures arrived. Each time the live picture goes
    away, and each time the reason or the pause of a failed try changes, `say`
    gets one line with both. While no browser has the stream open or asks for
    pictures the camera is asked for one picture a second only. Each picture
    goes to `model` with its number, its exposure and its arrival.
    """

    def __init__(
        self,
        make_source: SourceFactory,
        *,
        fps: float = DEFAULT_FPS,
        width_px: int = DEFAULT_WIDTH_PX,
        quality: int = JPEG_QUALITY,
        say: Say = _say,
        retry_pauses_s: Sequence[float] = RETRY_PAUSES_S,
        model: ViewModel | None = None,
    ) -> None:
        self._make_source = make_source
        self.model = (
            ViewModel(
                ack_timeout_s=_VIEW.ack_timeout_s,
                state_max_age_s=_VIEW.state_max_age_s,
            )
            if model is None
            else model
        )
        self.fps = float(fps)
        self.width_px = int(width_px)
        self._quality = quality
        self._say = say
        self._retry_pauses_s = tuple(retry_pauses_s)
        if not self._retry_pauses_s:
            raise ValueError("retry_pauses_s names no pause")
        self._failed_tries = 0  # in a row, since pictures arrived
        self._changed = threading.Condition()
        self._ended = False
        self._source: PairSource | None = None
        self._jpeg = b""
        self._seq = 0
        self._width = 0
        self._arrived = 0.0  # time.monotonic() of the newest picture's arrival
        self._encoded: deque[float] = deque()  # when the last pictures were made
        self._clients = 0
        self._pages: list[str] = []  # the page of each open stream that named one
        self._fetched: float | None = None  # when a page last asked for a picture
        self._why = "the camera source is being opened"
        self._told: bool | None = None  # what the last line said: live or not
        self._told_try = ""  # the last line's reason and pause, if a failed try's
        self._pair: StereoPair | None = None  # read and not yet encoded
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start the thread that reads the camera and the one that encodes."""
        for name, work in (("camera", self._run), ("encoder", self._encode)):
            thread = threading.Thread(
                target=work, name=f"viewer-live-{name}", daemon=True
            )
            self._threads.append(thread)
            thread.start()

    def close(self) -> None:
        """End the threads and the camera source; safe to call twice."""
        with self._changed:
            self._ended = True
            self._changed.notify_all()
        deadline = time.monotonic() + _CLOSE_S
        # Again and again: the camera thread may be starting the source right now.
        while True:
            with self._changed:
                source = self._source
            if source is not None:
                source.close()
            if not any(thread.is_alive() for thread in self._threads):
                return
            if time.monotonic() > deadline:
                _log.warning("the live picture's threads did not end")
                return
            self._threads[0].join(0.2)

    def state(self, page: str = "") -> dict[str, Any]:
        """What /api/live.json answers; `page` is the asking page's own name."""
        now = time.monotonic()
        with self._changed:
            age = now - self._arrived if self._seq else None
            available = age is not None and age <= STALE_S and not self._ended
            if available:
                reason = ""
            elif self._ended:
                reason = "the viewer is ending"
            else:
                reason = self._why or "the camera gave no picture yet"
            return {
                "available": available,
                "reason": reason,
                "fps_asked": self.fps,
                "fps_measured": float(sum(now - t <= 1.0 for t in self._encoded)),
                "width": self._width or None,
                "width_asked": self.width_px,
                "age_s": None if age is None else round(age, 3),
                "clients": self._clients,
                # Whether the page that asks has a stream open.
                "streaming": page in self._pages,
                # Whether a page asked for a numbered picture just now.
                "fetching": self._fetching(now),
            }

    def note_fetch(self) -> None:
        """A page asks for a numbered picture: the camera is read at the asked rate."""
        with self._changed:
            now = time.monotonic()
            idle = not self._watched(now)
            self._fetched = now
            if idle:
                self._changed.notify_all()

    @contextlib.contextmanager
    def watched(self, page: str = "") -> Iterator[None]:
        """Counts one browser that has the stream open, for its page."""
        with self._changed:
            self._clients += 1
            self._pages.append(page)
            self._changed.notify_all()
        try:
            yield
        finally:
            with self._changed:
                self._clients -= 1
                self._pages.remove(page)

    def next_picture(self, after: int, timeout_s: float) -> tuple[int, bytes] | None:
        """(number, JPEG) of the newest picture once its number is not `after`.

        None after timeout_s without one, and when the live picture has ended.
        """
        deadline = time.monotonic() + timeout_s
        with self._changed:
            while not self._ended:
                if self._seq and self._seq != after:
                    return self._seq, self._jpeg
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._changed.wait(left)
        return None

    @property
    def ended(self) -> bool:
        return self._ended

    def _run(self) -> None:
        while not self._ended:
            source = None
            try:
                source = self._make_source()
                with self._changed:
                    if self._ended:
                        return
                    self._source = source
                source.open()
                why = self._read(source)
            except (CameraError, OSError) as problem:
                why = str(problem)
            except Exception as problem:
                _log.exception("the live picture's camera thread failed")
                why = f"{type(problem).__name__}: {problem}"
            finally:
                if source is not None:
                    source.close()
                with self._changed:
                    self._source = None
            if self._ended:
                return
            pauses = self._retry_pauses_s
            pause_s = pauses[min(self._failed_tries, len(pauses) - 1)]
            self._failed_tries += 1
            self._tell(False, f"{why}; trying again in {pause_s:g} s", failed_try=True)
            self._pause(time.monotonic() + pause_s, watched=True)

    def _read(self, source: PairSource) -> str:
        """Hand pairs to the encoder until the source fails; why it did.

        "" when told to end. The encoder works on its own thread, so the next
        pair is asked for while the last one is encoded.
        """
        due = time.monotonic()
        watched_before = False
        while not self._ended:
            try:
                pair = source.next_pair(PAIR_WAIT_S)
            except CameraError as problem:
                return f"the camera source ended: {problem}"
            now = time.monotonic()
            if pair is None:
                with self._changed:
                    stale = not self._seq or now - self._arrived > STALE_S
                if stale:
                    why = getattr(source, "last_none", "") or "no new stereo pair"
                    self._tell(
                        False, f"the camera gives no picture ({why}); waiting for one"
                    )
                continue
            self._failed_tries = 0
            with self._changed:
                self._pair = pair  # one the encoder did not take yet is skipped
                watched = self._watched(now)
                self._changed.notify_all()
            if watched and not watched_before:
                due = now  # a browser came: its first period starts here
            watched_before = watched
            # The next pair is due one period after this one was; a late one is
            # not made up for.
            due = max(due + 1.0 / (self.fps if watched else IDLE_FPS), now)
            self._pause(due, watched)
        return ""

    def _encode(self) -> None:
        """Make each pair the reader hands over the newest picture."""
        while True:
            with self._changed:
                while self._pair is None and not self._ended:
                    self._changed.wait()
                if self._ended:
                    return
                pair, self._pair = self._pair, None
            try:
                jpeg = encode_picture(pair.left, self.width_px, self._quality)
            except Exception as problem:
                self._tell(
                    False, f"a camera frame could not be made a picture: {problem}"
                )
                continue
            now = time.monotonic()
            arrived = float(pair.receiver_rx_mono_s)
            if not 0.0 <= now - arrived <= _MAX_ARRIVAL_AGE_S:
                arrived = now
            with self._changed:
                self._jpeg = jpeg
                self._seq += 1
                self._width = min(self.width_px, pair.left.shape[1])
                self._arrived = arrived
                # As the live view program of view/ hands its pictures over.
                self.model.put_picture(
                    self._seq,
                    jpeg,
                    min(pair.sof_wall_ns) / 1e9,
                    pair.receiver_rx_mono_s,
                    bool(pair.check.get("invented_times")),
                )
                self._encoded.append(now)
                while self._encoded and now - self._encoded[0] > 1.0:
                    self._encoded.popleft()
                self._changed.notify_all()
            self._tell(True, "")

    def _fetching(self, now: float) -> bool:
        return self._fetched is not None and now - self._fetched < FETCH_WATCH_S

    def _watched(self, now: float) -> bool:
        """Whether a browser has the stream open or asks for pictures; under the lock."""
        return self._clients > 0 or self._fetching(now)

    def _pause(self, until: float, watched: bool) -> None:
        """Wait to `until`; not past the end, nor past the first browser's arrival."""
        with self._changed:
            while not self._ended:
                if not watched and self._watched(time.monotonic()):
                    return
                left = until - time.monotonic()
                if left <= 0:
                    return
                self._changed.wait(left)

    def _tell(self, live: bool, why: str, *, failed_try: bool = False) -> None:
        """Say a change between live and not once, and each new `why` of a failed try."""
        with self._changed:
            self._why = why
            if self._ended:
                return
            if self._told is live and not (failed_try and why != self._told_try):
                return
            self._told = live
            self._told_try = why if failed_try else ""
        if live:
            self._say(f"{PROGRAM}: live picture: the camera's pictures arrive")
        else:
            self._say(
                f"{PROGRAM}: no live picture: {why}. The recorded runs are served "
                "as before"
            )


def _page_name(query: str) -> str:
    """The page=NAME of a query, when it is a plain short name; else ""."""
    found = re.search(r"(?:^|&)page=([A-Za-z0-9]{1,32})(?:&|$)", query)
    return found.group(1) if found else ""


def page_block() -> bytes:
    """What goes into the viewer's page before its </body>."""
    return PAGE_BLOCK_FILE.read_bytes()


def with_page_block(response: bytes, block: bytes) -> bytes:
    """A whole HTTP response with `block` before the page's </body>.

    Any response but a 200 with an HTML page that has a </body> stays as it is.
    """
    head, separator, body = response.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status = lines[0].split()
    at = body.rfind(PAGE_END)
    is_page = any(
        line.lower().startswith(b"content-type: text/html") for line in lines[1:]
    )
    if not separator or status[1:2] != [b"200"] or not is_page or at < 0:
        return response
    body = body[:at] + block + body[at:]
    lines = [
        b"Content-Length: " + str(len(body)).encode()
        if line.lower().startswith(b"content-length:")
        else line
        for line in lines
    ]
    return b"\r\n".join(lines) + separator + body


def _after(query: str) -> int:
    """The after=N of a query: the number of the picture the page drew last."""
    found = re.search(r"(?:^|&)after=([0-9]{1,18})(?:&|$)", query)
    return int(found.group(1)) if found else 0


def _watch(query: str) -> bool:
    """Whether a query holds watch=1: a picture to watch, which counts for nothing."""
    return re.search(r"(?:^|&)watch=1(?:&|$)", query) is not None


def _address(name: Any) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """The IP address of a socket's name; ValueError when it has none."""
    if not isinstance(name, tuple) or not name:
        raise ValueError("not the name of an IP socket")
    address = ipaddress.ip_address(str(name[0]).partition("%")[0])
    # An IPv4 client of an IPv6 socket.
    return getattr(address, "ipv4_mapped", None) or address


def on_this_machine(connection: Any) -> bool:
    """Whether the browser at the other end of this connection runs on this machine.

    It does when the client's address is a loopback address or the address the
    request arrived on. Both are read off the connection: no header counts.
    """
    try:
        client = _address(connection.getpeername())
        arrived_on = _address(connection.getsockname())
    except (OSError, ValueError):
        return False
    return client.is_loopback or client == arrived_on


def live_handler(
    base: type,
    token: str | None,
    live: LivePicture,
    authorized: Callable[[Any, str], bool],
    view: Callable[[], dict[str, Any]] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    this_machine: ThisMachine | None = None,
) -> type:
    """The viewer's request handler class `base` with the live picture added.

    token and authorized(headers, token) are the viewer's own: None serves
    everyone, else a request needs what authorized() accepts. view() is what
    the robot is told of the live view, for /api/live.json; None: nothing is.
    monotonic() dates what a page loads, fetches and acknowledges.
    this_machine(connection) says whether a request comes from this machine;
    None: on_this_machine().
    """
    block = page_block()
    model = live.model
    own_paths = (STREAM_PATH, STATE_PATH, PICTURE_PATH)
    if this_machine is None:
        this_machine = on_this_machine

    class LiveHandler(base):  # type: ignore[misc, valid-type]
        def do_GET(self) -> None:
            self._as_the_viewer()
            url = urlsplit(self.path)
            path = url.path
            if path == "/":
                self._page()
            elif path not in own_paths:
                super().do_GET()
            elif token is not None and not authorized(self.headers, token):
                self.send_error(401, "Open the viewer using its private access link")
            elif path == STATE_PATH:
                self._state(_page_name(url.query))
            elif path == PICTURE_PATH:
                self._picture(_after(url.query), _watch(url.query))
            else:
                self._stream(_page_name(url.query))

        def do_POST(self) -> None:
            self._as_the_viewer()
            if urlsplit(self.path).path != SHOWN_PATH:
                # The viewer alone takes no POST: its answer, in its words.
                self.send_error(
                    http.HTTPStatus.NOT_IMPLEMENTED,
                    f"Unsupported method ({self.command!r})",
                )
            elif token is not None and not authorized(self.headers, token):
                self.send_error(401, "Open the viewer using its private access link")
            else:
                self._shown()

        def _as_the_viewer(self) -> None:
            """Answer as the viewer alone does until a live path says otherwise.

            A connection a live path kept open ends with the viewer's own answer.
            """
            if self.__dict__.pop("protocol_version", None) is not None:
                self.close_connection = True

        def _keep(self) -> None:
            """A live path's short answer leaves its connection open for the next."""
            self.protocol_version = "HTTP/1.1"
            self.close_connection = (
                self.request_version < "HTTP/1.1"
                or self.headers.get("Connection", "").lower() == "close"
            )

        def _page(self) -> None:
            """The viewer's own answer to GET /, with the block in its page.

            For a browser on this machine the block carries a new view token:
            the page before counts no longer. For any other it carries none:
            a watch-only page, which replaces no page.
            """
            real, answer = self.wfile, io.BytesIO()
            self.wfile = answer
            try:
                super().do_GET()
            finally:
                self.wfile = real
            response = answer.getvalue()
            if with_page_block(response, b"") is not response:
                view_token = ""
                if this_machine(self.connection):
                    view_token = model.new_page(monotonic())
                response = with_page_block(
                    response, block.replace(TOKEN_PLACE, view_token.encode())
                )
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                real.write(response)

        def _headers(self, kind: str, length: int | None = None) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()

        def _short(
            self,
            status: int,
            body: bytes = b"",
            kind: str = "",
            headers: Mapping[str, str] | None = None,
            *,
            close: bool = False,
        ) -> None:
            """A live path's whole answer, on a connection that stays open.

            With `close` the connection ends: its request was not read whole.
            """
            self._keep()
            self.close_connection = self.close_connection or close
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.send_response(status)
                if kind:
                    self.send_header("Content-Type", kind)
                    self.send_header("X-Content-Type-Options", "nosniff")
                if status != 204:  # an answer without content names no length
                    self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "DENY")
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                if self.close_connection:
                    self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

        def _state(self, page: str) -> None:
            told = live.state(page)
            told["view"] = dict(NOT_REPORTED) if view is None else view()
            body = json.dumps(told, allow_nan=False).encode()
            self._short(200, body, "application/json")

        def _picture(self, after: int, watch: bool) -> None:
            """The newest picture once it is not number `after`.

            For the page opened last on this machine, which may acknowledge
            it; with `watch` for any page, and the model hears nothing of it.
            """
            view_token = self.headers.get(TOKEN_HEADER, "")
            counted = not watch
            if counted and not (
                this_machine(self.connection) and model.is_current_page(view_token)
            ):
                self._short(403)
                return
            live.note_fetch()
            picture = live.next_picture(after, PICTURE_WAIT_S)
            if picture is None:
                self._short(204)
            elif not counted or model.note_served(picture[0], view_token, monotonic()):
                seq, jpeg = picture
                self._short(200, jpeg, "image/jpeg", {"X-Picture-Seq": str(seq)})
            else:  # the page was loaded again meanwhile
                self._short(403)

        def _shown(self) -> None:
            """The page's word that it has drawn a picture; as POST /shown of view/."""
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length < 0 or length > _DRAIN_BYTES:
                self._short(411 if length < 0 else 413, close=True)
                return
            body = self.rfile.read(length)
            if length > MAX_SHOWN_BYTES:
                self._short(413)
                return
            # A page of another site may not speak for this one.
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{self.headers.get('Host')}":
                self._short(403)
                return
            # Nor does a page opened from another machine: it only watches.
            if not this_machine(self.connection):
                self._short(403)
                return
            try:
                told = json.loads(body)
            except ValueError:
                told = None
            if not isinstance(told, dict):
                told = {}
            seq, visible = told.get("seq"), told.get("visible")
            # A body that names no token is well formed: the model refuses it.
            view_token = told.get("token", "")
            if (
                type(seq) is not int
                or type(visible) is not bool
                or type(view_token) is not str
            ):
                self._short(400)
                return
            counted = model.acknowledge(seq, visible, monotonic(), view_token)
            self._short(204 if counted else 403)

        def _stream(self, page: str) -> None:
            state = live.state()
            if not state["available"]:
                # The reason goes into the body only: a status line is Latin-1.
                self.send_error(503, "No live picture", state["reason"])
                return
            try:
                self.connection.settimeout(CLIENT_SEND_TIMEOUT_S)
                self.connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, _SEND_BUFFER_BYTES
                )
                self._headers(f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                with live.watched(page):
                    self._pictures()
            except OSError:  # the browser closed the stream, or takes no bytes
                pass
            self.close_connection = True

        def _pictures(self) -> None:
            """Send each new picture until the browser or the live picture is gone.

            A browser that reads slowly gets the newest picture after each
            write: the ones made meanwhile are skipped.
            """
            seq = 0
            while not live.ended:
                picture = live.next_picture(seq, _CLIENT_CHECK_S)
                if picture is None:
                    if self._browser_gone():
                        return
                    continue
                seq, jpeg = picture
                self.wfile.write(
                    b"--%b\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                    % (BOUNDARY.encode(), len(jpeg))
                    + jpeg
                    + b"\r\n"
                )

        def _browser_gone(self) -> bool:
            """Whether the browser closed its end; it sends nothing else."""
            if not select.select([self.connection], [], [], 0)[0]:
                return False
            try:
                return not self.connection.recv(1, socket.MSG_PEEK)
            except OSError:
                return True

    return LiveHandler
