# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live view's HTTP side, for a browser on this machine.

GET /            the page (page.html beside this file), with a new view token
GET /frame.jpg   the newest camera picture, for the page that sends the valid
                 view token as X-View-Token; its number in X-Picture-Seq (204
                 without a picture, 403 without the valid token)
GET /status.json what the page shows over and under the picture; needs no token
POST /shown      {"seq": N, "visible": true|false, "token": "..."}: the page has
                 drawn picture N (403 when it does not count)

The view token is the secret a page is served with: each GET / makes a new one
the only valid one, so one page at a time counts. It is never logged.

Nothing here can reach the robot: /shown only tells the model what the page drew.
"""

from __future__ import annotations

import http.server
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .model import ViewModel

_log = logging.getLogger("isaaccapture.examples.agentic_remote.view_server")

LOOPBACK = "127.0.0.1"
PAGE_FILE = Path(__file__).with_name("page.html")
# The text in page.html that each GET / replaces with that page's view token.
TOKEN_PLACE = b"__VIEW_TOKEN__"
# The request header in which the page sends its view token with GET /frame.jpg.
TOKEN_HEADER = "X-View-Token"
# The methods and paths the log names; any other request line is logged as
# _OTHER_REQUEST.
_METHODS = ("GET", "POST")
_PATHS = ("/", "/frame.jpg", "/status.json", "/shown")
_OTHER_REQUEST = "another request"
# The longest body of POST /shown.
MAX_SHOWN_BYTES = 4096
# A longer body than MAX_SHOWN_BYTES is read and dropped up to this size; above
# it the connection is closed instead.
_DRAIN_BYTES = 1 << 20
# The page may run its own inline script and style, draw pictures it fetched
# from this server and ask this server; nothing else.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src 'self' blob:; connect-src 'self'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AgenticRemoteView"
    sys_version = ""
    # A browser that keeps a connection open and silent is dropped after this.
    timeout = 30.0

    server: _Server

    def do_GET(self) -> None:
        if not self._from_here():
            return
        path = self.path.partition("?")[0]
        if path == "/":
            token = self.server.model.new_page(self.server.monotonic())
            _log.info("the page was loaded; a page loaded before counts no longer")
            self._reply(
                200,
                self.server.page.replace(TOKEN_PLACE, token.encode()),
                "text/html; charset=utf-8",
                {"Content-Security-Policy": CONTENT_SECURITY_POLICY},
            )
        elif path == "/frame.jpg":
            model, now = self.server.model, self.server.monotonic()
            token = self.headers.get(TOKEN_HEADER, "")
            picture = model.picture()
            if picture is None:
                self._reply(204 if model.is_current_page(token) else 403)
            elif model.note_served(picture[0], token, now):
                seq, jpeg = picture
                self._reply(200, jpeg, "image/jpeg", {"X-Picture-Seq": str(seq)})
            else:
                self._reply(403)
        elif path == "/status.json":
            model, now = self.server.model, self.server.monotonic()
            told = model.status_json(now, model.thor_wall_now(now))
            told["page"] = {"picture_period_s": self.server.picture_period_s}
            self._reply(200, json.dumps(told).encode(), "application/json")
        else:
            self._reply(404)

    def do_POST(self) -> None:
        if not self._from_here():
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0:
            self.close_connection = True
            self._reply(411)
            return
        if length > _DRAIN_BYTES:
            self.close_connection = True
            self._reply(413)
            return
        body = self.rfile.read(length)
        if self.path.partition("?")[0] != "/shown":
            self._reply(404)
            return
        if length > MAX_SHOWN_BYTES:
            self._reply(413)
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.origins:
            self._reply(403)
            return
        try:
            told = json.loads(body)
        except ValueError:
            told = None
        if not isinstance(told, dict):
            told = {}
        seq, visible = told.get("seq"), told.get("visible")
        # A body that names no token is well formed: the model refuses it.
        token = told.get("token", "")
        if type(seq) is not int or type(visible) is not bool or type(token) is not str:
            self._reply(400)
            return
        now = self.server.monotonic()
        counted = self.server.model.acknowledge(seq, visible, now, token)
        self._reply(204 if counted else 403)

    def _from_here(self) -> bool:
        """Whether the request names this server; another host name gets 403."""
        if self.headers.get("Host", "") in self.server.hosts:
            return True
        self.close_connection = True
        self._reply(403)
        return False

    def _reply(
        self,
        status: int,
        body: bytes = b"",
        content_type: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        # http.server's own texts quote the request line. The standard words
        # of the status code go out and into the log instead.
        super().send_error(code)

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        # Of a request line the log names only what this server serves: a view
        # token could stand anywhere else in it.
        line = _OTHER_REQUEST
        if self.command in _METHODS:
            path = self.path.partition("?")[0]
            if path in _PATHS:
                line = f"{self.command} {path} {self.request_version}"
        self.log_message('"%s" %s', line, code)

    def log_message(self, format: str, *args: Any) -> None:
        _log.debug("%s %s", self.address_string(), format % args)


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    model: ViewModel
    monotonic: Callable[[], float]
    page: bytes
    picture_period_s: float
    hosts: frozenset[str]
    origins: frozenset[str]

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A browser that closed its tab in the middle of a reply.
        _log.debug("a request from %s ended early", client_address, exc_info=True)


def make_server(
    model: ViewModel,
    address: str,
    port: int,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    picture_period_s: float = 0.2,
) -> http.server.ThreadingHTTPServer:
    """A server for the live view, bound and not yet serving.

    Only 127.0.0.1 is taken as the address (ValueError otherwise); OSError when
    the port cannot be had. Port 0 lets the system pick one. The caller runs
    serve_forever() and later shutdown() and server_close().
    """
    if address != LOOPBACK:
        raise ValueError(
            f"the live view serves only {LOOPBACK}, not {address!r}: the page is "
            "for a browser on this machine"
        )
    page = PAGE_FILE.read_bytes()
    server = _Server((LOOPBACK, port), _Handler)
    bound = server.server_address[1]
    server.model = model
    server.monotonic = monotonic
    server.page = page
    server.picture_period_s = float(picture_period_s)
    server.hosts = frozenset({f"{LOOPBACK}:{bound}", f"localhost:{bound}"})
    server.origins = frozenset(f"http://{host}" for host in server.hosts)
    return server
