# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live view's HTTP side, for a browser on this machine.

GET /            the page (page.html beside this file)
GET /frame.jpg   the newest camera picture; its number in X-Picture-Seq (204
                 without a picture)
GET /status.json what the page shows under the picture
POST /shown      {"seq": N, "visible": true|false}: the page has drawn picture N

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
            self._reply(
                200,
                self.server.page,
                "text/html; charset=utf-8",
                {"Content-Security-Policy": CONTENT_SECURITY_POLICY},
            )
        elif path == "/frame.jpg":
            picture = self.server.model.picture()
            if picture is None:
                self._reply(204)
            else:
                seq, jpeg = picture
                self._reply(200, jpeg, "image/jpeg", {"X-Picture-Seq": str(seq)})
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
        seq = told.get("seq") if isinstance(told, dict) else None
        visible = told.get("visible") if isinstance(told, dict) else None
        if type(seq) is not int or type(visible) is not bool:
            self._reply(400)
            return
        self.server.model.acknowledge(seq, visible, self.server.monotonic())
        self._reply(204)

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
