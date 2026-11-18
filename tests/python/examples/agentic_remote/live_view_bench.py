# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A live view for tests that run one beside other programs.

The live view is the viewer's page with its live picture showing. LiveView is
the viewer's own parts on threads of a test: its live picture on a stand-in
camera, its request handler over a stand-in for the experiment's viewer, and
its attachment to the robot link service of the test. Browser is what the
viewer's page does, over one HTTP connection. No experiment folder is needed.
"""

from __future__ import annotations

import http.client
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import (
    CameraError,
    StereoPair,
    bgr_to_nv12,
)

# The live picture is packed with OpenCV; without it a file that imports this
# one is skipped.
cv2 = pytest.importorskip("cv2")

from isaaccapture_examples.agentic_remote.viewer_live import settings  # noqa: E402
from isaaccapture_examples.agentic_remote.viewer_live.live import (  # noqa: E402
    PICTURE_PATH,
    ROBOT_PATH,
    SHOWN_PATH,
    STATE_PATH,
    TOKEN_HEADER,
    LivePicture,
    live_handler,
)
from isaaccapture_examples.agentic_remote.viewer_live.robot import (  # noqa: E402
    LiveRobot,
)
from isaaccapture_examples.agentic_remote.viewer_live.view_link import (  # noqa: E402
    ViewLink,
)
from isaaccapture_examples.agentic_remote.viewer_live.view_model import (  # noqa: E402
    ViewModel,
)

WAIT_S = 10.0  # real seconds a test gives another thread before it fails
SIZE = (128, 96)  # width and height of the stand-in camera frames
RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)  # as BGR
# What the stand-in for the experiment's viewer serves as its page.
PAGE = b'<!doctype html><html><body><img id="photo">the page</body></html>'
# How a served page names its view token: the secret of that load of the page.
VIEW_TOKEN = re.compile(rb"const TOKEN = '([A-Za-z0-9_-]{32})';")
# Why a LiveView without a camera has no live picture.
NO_CAMERA = "this test has no camera"


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


class _Viewer(BaseHTTPRequestHandler):
    """A stand-in for the experiment's viewer's request handler: its page."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.partition("?")[0] != "/":
            self.send_error(400, "Unknown endpoint")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)


def _no_camera() -> Any:
    raise CameraError(NO_CAMERA)


class LiveView:
    """The viewer's live view, attached to the robot link service at socket_path.

    `source` gives the camera's stereo pairs; None: no camera, and so no live
    picture. The page is served on 127.0.0.1, on .port, to anyone: the
    stand-in viewer has no token of its own. ack_timeout_s is how long an
    acknowledgement of the page counts; None: the viewer's own value. The
    live arm has no arm model here: /api/live/robot.json gives the readings.
    close() ends everything; the robot is then told that the view is over.
    """

    def __init__(
        self,
        socket_path: str,
        source: Any = None,
        *,
        fps: float = 15.0,
        period_s: float = 0.1,
        ack_timeout_s: float | None = None,
    ) -> None:
        self.lines: list[str] = []
        self.model = ViewModel(
            ack_timeout_s=settings.ACK_TIMEOUT_S
            if ack_timeout_s is None
            else ack_timeout_s,
            state_max_age_s=settings.STATE_MAX_AGE_S,
        )
        make = _no_camera if source is None else (lambda: source)
        self.picture = LivePicture(
            make, fps=fps, say=self.lines.append, model=self.model
        )
        self.robot = LiveRobot()
        self.link = ViewLink(
            self.model,
            socket_path,
            say=self.lines.append,
            period_s=period_s,
            retry_pauses_s=(0.2,),
            robot=self.robot,
        )
        handler = live_handler(
            _Viewer,
            None,
            self.picture,
            lambda headers, token: True,
            self.link.state,
            robot=self.robot,
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self._thread.start()
        self.picture.start()
        self.link.start()

    def close(self) -> None:
        self.link.close()
        self.robot.close()
        self._server.shutdown()
        self._thread.join(WAIT_S)
        self._server.server_close()
        self.picture.close()


class Browser:
    """What the viewer's page does, over one HTTP connection.

    `token` is the view token: the secret the viewer serves a page with, valid
    until the page is loaded again by anyone. The Browser loads the page on
    its first frame(), so one that only reads replaces no page.
    """

    def __init__(self, port: int) -> None:
        self.http = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT_S)
        self.token: str | None = None
        self.seq = 0  # the picture fetched last
        self._frame_status: int | None = None

    def reload(self) -> None:
        """Load the page again: a new token, and the one before counts no more."""
        self.http.request("GET", "/")
        reply = self.http.getresponse()
        page = reply.read()
        assert reply.status == 200
        found = VIEW_TOKEN.search(page)
        assert found is not None, "the served page holds no view token"
        self.token = found.group(1).decode()

    def frame(self) -> tuple[int | None, bytes]:
        """A picture newer than the last one; (None, b"") without one or when refused.

        The viewer waits up to a second for a newer picture.
        """
        if self.token is None:
            self.reload()
        self.http.request(
            "GET",
            f"{PICTURE_PATH}?after={self.seq}",
            headers={TOKEN_HEADER: self.token},
        )
        reply = self.http.getresponse()
        self._frame_status = reply.status
        seq = reply.getheader("X-Picture-Seq")
        body = reply.read()
        if reply.status != 200 or seq is None:
            return None, b""
        self.seq = int(seq)
        return self.seq, body

    def first_frame(self) -> tuple[int, bytes]:
        """The first camera picture; asks at most 20 times."""
        for _ in range(20):
            seq, jpeg = self.frame()
            if seq is not None:
                return seq, jpeg
            time.sleep(0.1)
        raise AssertionError(
            f"no camera picture was served; the last reply was {self._frame_status}"
        )

    def shown(self, seq: int, visible: bool = True) -> None:
        """Acknowledge a picture this page fetched; the viewer must take it."""
        code = self.shown_raw(
            {"seq": seq, "visible": visible, "token": self.token or ""}
        )
        assert code == 204, f"the acknowledgement of picture {seq} got {code}"

    def shown_raw(self, body: Any) -> int:
        """POST the acknowledgement with that body; the status code."""
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        self.http.request("POST", SHOWN_PATH, body=data, headers=headers)
        reply = self.http.getresponse()
        reply.read()
        return reply.status

    def live(self) -> dict[str, Any]:
        """What the viewer says of its live picture and of what the robot is told."""
        return self._json(STATE_PATH)

    def robot(self) -> dict[str, Any]:
        """The viewer's newest ROBOT_STATE, as its page reads it."""
        return self._json(ROBOT_PATH)

    def _json(self, path: str) -> dict[str, Any]:
        self.http.request("GET", path)
        reply = self.http.getresponse()
        body = reply.read()
        assert reply.status == 200, (path, reply.status)
        return json.loads(body)

    def close(self) -> None:
        self.http.close()
