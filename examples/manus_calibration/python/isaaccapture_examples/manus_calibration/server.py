# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small HTTP server for the calibration page, using only the standard library.

Static files and a server-sent-event stream per browser tab carry the controller's view out; three
POST endpoints carry the operator's actions in. The page can drive hardware and has no login, so
every request is checked for the page's own origin and every action must carry a per-tab client id.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from isaaccapture_manus_calibration import protocol as proto
from .controller import CalibrationController, NotOwner, View
from .wire import view_to_dict
from isaaccapture_manus_calibration.worker import (
    AttemptError,
    CalibrationWorker,
    OutcomeKind,
)

logger = logging.getLogger("isaaccapture.examples.manus_calibration.server")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_TYPES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
_DOWNLOAD_PREFIX = "/download/"
_CLIENT_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
_MAX_HEADER_BYTES = 16 * 1024
_MAX_BODY_BYTES = 4 * 1024
_REQUEST_TIMEOUT_S = 10.0
_KEEPALIVE_S = 15.0
# A reload closes the old stream before the new one opens; this window keeps it from cancelling.
_DISCONNECT_GRACE_S = 3.0
_LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
_REASONS = {
    200: "OK",
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    413: "Payload Too Large",
    500: "Internal Server Error",
}


class _BadRequest(Exception):
    """The request cannot be parsed or breaks a limit; carries the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class _Request:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes


@dataclass
class _Tab:
    """One browser tab's server-sent-event state; its id is chosen by the page."""

    connections: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    grace: asyncio.TimerHandle | None = None


class PageServer:
    """Serves the page and runs the controller on the caller's event loop."""

    def __init__(
        self,
        worker: CalibrationWorker,
        out_dir: Path,
        host: str,
        port: int,
        default_sides: str = "both",
    ) -> None:
        self._continuous_pacing = worker.continuous_pacing
        self._out_dir = str(out_dir)
        self._default_sides = default_sides
        self._host = host
        self._requested_port = port
        self._controller = CalibrationController(worker, self._on_change)
        self._tabs: dict[str, _Tab] = {}
        # Strong references: the loop keeps only weak ones, and a dropped task would skip the cancel.
        self._background: set[asyncio.Task[None]] = set()
        self._server: asyncio.Server | None = None
        self._allowed_hosts: set[str] = set()
        self._static = {
            path: (name, ctype, (_STATIC_DIR / name).read_bytes())
            for path, (name, ctype) in _STATIC_TYPES.items()
        }

    @property
    def controller(self) -> CalibrationController:
        return self._controller

    @property
    def port(self) -> int:
        """The bound port; differs from the requested one when that was 0."""
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> None:
        """Begin listening."""
        self._server = await asyncio.start_server(
            self._handle, self._host, self._requested_port
        )
        names = (
            set(_LOOPBACK_NAMES)
            if self._host in ("127.0.0.1", "localhost", "::1")
            else {self._host}
        )
        self._allowed_hosts = {f"{name}:{self.port}" for name in names}

    async def aclose(self) -> None:
        """Cancel any active attempt, wait for the worker to settle, then stop listening."""
        await self._controller.aclose()
        if self._server is not None:
            self._server.close()
            for tab in self._tabs.values():
                tab.changed.set()
            await self._server.wait_closed()

    def _on_change(self, _view: View) -> None:
        """Wake every open stream; each sends the controller's current view, so bursts coalesce."""
        for tab in self._tabs.values():
            tab.changed.set()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """One connection: parse one request, answer it, close. A stream request stays open."""
        try:
            try:
                request = await asyncio.wait_for(
                    self._read_request(reader), _REQUEST_TIMEOUT_S
                )
                self._check_origin(request)
                await self._route(request, reader, writer)
            except _BadRequest as exc:
                await self._reply_json(
                    writer, exc.status, {"ok": False, "error": str(exc)}
                )
            except (asyncio.TimeoutError, ConnectionError):
                # The client gave up or went away mid-request: nobody is left to answer.
                pass
        except Exception:
            # Detail stays in the log; the browser gets no stack trace.
            logger.exception("request failed")
            with contextlib.suppress(Exception):
                await self._reply_json(
                    writer, 500, {"ok": False, "error": "internal error"}
                )
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _read_request(self, reader: asyncio.StreamReader) -> _Request:
        """Parse the request line, headers and a bounded body."""
        try:
            block = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError:
            raise _BadRequest(413, "request headers too large") from None
        except asyncio.IncompleteReadError:
            raise ConnectionError from None
        if len(block) > _MAX_HEADER_BYTES:
            raise _BadRequest(413, "request headers too large")
        lines = block.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
            raise _BadRequest(400, "malformed request line")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            name, sep, value = line.partition(":")
            if not sep:
                raise _BadRequest(400, "malformed header")
            headers[name.strip().lower()] = value.strip()
        length_text = headers.get("content-length", "0")
        if not length_text.isdigit() or int(length_text) > _MAX_BODY_BYTES:
            raise _BadRequest(413, "request body too large")
        try:
            body = await reader.readexactly(int(length_text))
        except asyncio.IncompleteReadError:
            raise ConnectionError from None
        split = urlsplit(parts[1])
        return _Request(parts[0], split.path, parse_qs(split.query), headers, body)

    def _check_origin(self, request: _Request) -> None:
        """Refuse a foreign Host (DNS rebinding) or a cross-origin page; a missing Origin is a same-origin GET."""
        host = request.headers.get("host", "")
        if host not in self._allowed_hosts:
            raise _BadRequest(403, "unexpected Host header")
        origin = request.headers.get("origin")
        if origin is not None and origin != f"http://{host}":
            raise _BadRequest(403, "cross-origin request refused")

    async def _route(
        self,
        request: _Request,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Dispatch on method and path."""
        if request.method == "GET" and request.path in self._static:
            _name, ctype, body = self._static[request.path]
            await self._reply(writer, 200, ctype, body)
        elif request.method == "GET" and request.path == "/events":
            await self._stream(request, reader, writer)
        elif request.method == "GET" and request.path.startswith(_DOWNLOAD_PREFIX):
            await self._download(request.path[len(_DOWNLOAD_PREFIX) :], writer)
        elif request.method == "POST" and request.path in (
            "/api/start",
            "/api/confirm",
            "/api/cancel",
        ):
            await self._action(request, writer)
        elif request.path in self._static or request.path in (
            "/events",
            "/api/start",
            "/api/confirm",
            "/api/cancel",
        ):
            raise _BadRequest(405, "method not allowed")
        else:
            raise _BadRequest(404, "not found")

    async def _download(self, name: str, writer: asyncio.StreamWriter) -> None:
        """Send one file of the last completed calibration to the browser as a download.

        Only names the last completed attempt reported are served, so the route can never read anything else
        in the folder.
        """
        outcome = self._controller.view().outcome
        offered = (
            {f.file for f in outcome.files}
            if outcome is not None and outcome.kind is OutcomeKind.COMPLETED
            else set()
        )
        if name not in offered:
            raise _BadRequest(404, "not found")
        try:
            data = await asyncio.to_thread((Path(self._out_dir) / name).read_bytes)
        except OSError:
            raise _BadRequest(404, "the file is no longer available") from None
        await self._reply(
            writer,
            200,
            "application/octet-stream",
            data,
            {"Content-Disposition": f'attachment; filename="{name}"'},
        )

    async def _action(self, request: _Request, writer: asyncio.StreamWriter) -> None:
        """An operator action; the custom header and JSON type make a cross-site form post impossible."""
        client = request.headers.get("x-client-id", "")
        if not _CLIENT_ID_RE.fullmatch(client):
            raise _BadRequest(400, "missing or invalid X-Client-Id")
        if (
            request.headers.get("content-type", "").split(";")[0].strip()
            != "application/json"
        ):
            raise _BadRequest(400, "content type must be application/json")
        try:
            data: Any = json.loads(request.body or b"{}")
        except ValueError:
            raise _BadRequest(400, "body is not valid JSON") from None
        if not isinstance(data, dict):
            raise _BadRequest(400, "body must be a JSON object")
        try:
            if request.path == "/api/start":
                # Without a live stream no disconnect could ever cancel the attempt this tab would own.
                tab = self._tabs.get(client)
                if tab is None or tab.connections == 0:
                    raise _BadRequest(409, "the page's event stream is not connected")
                await self._controller.start(client, _sides_of(data))
            elif request.path == "/api/confirm":
                token = data.get("token")
                if not isinstance(token, str) or not token:
                    raise _BadRequest(400, "token is required")
                await self._controller.confirm(client, token)
            else:
                await self._controller.cancel(client)
        except NotOwner as exc:
            raise _BadRequest(403, str(exc)) from None
        except AttemptError as exc:
            raise _BadRequest(409, str(exc)) from None
        except proto.ProtocolError as exc:
            raise _BadRequest(400, str(exc)) from None
        await self._reply_json(writer, 200, {"ok": True})

    async def _stream(
        self,
        request: _Request,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Server-sent events for one tab until it disconnects or the server closes."""
        ids = request.query.get("client", [])
        if len(ids) != 1 or not _CLIENT_ID_RE.fullmatch(ids[0]):
            raise _BadRequest(400, "missing or invalid client id")
        client = ids[0]
        tab = self._tabs.setdefault(client, _Tab())
        if tab.grace is not None:
            tab.grace.cancel()
            tab.grace = None
        tab.connections += 1
        head = {
            "Content-Type": "text/event-stream",
            "Connection": "close",
            **_SECURITY_HEADERS,
        }
        writer.write(_head(200, head))
        # EOF on the request side is the only portable sign that the tab went away.
        closed = asyncio.ensure_future(reader.read())
        try:
            while self._server is not None and self._server.is_serving():
                # Cleared before the snapshot, so a change that lands while writing wakes the next wait.
                tab.changed.clear()
                payload = json.dumps(
                    view_to_dict(
                        self._controller.view(),
                        client,
                        self._out_dir,
                        self._default_sides,
                        self._continuous_pacing,
                    )
                )
                writer.write(f"data: {payload}\n\n".encode())
                await writer.drain()
                waiter = asyncio.ensure_future(tab.changed.wait())
                done, _pending = await asyncio.wait(
                    {waiter, closed},
                    timeout=_KEEPALIVE_S,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                waiter.cancel()
                if closed in done:
                    break
                if not done:
                    writer.write(b": keepalive\n\n")
                    await writer.drain()
        except ConnectionError:
            # The tab closed mid-write; the cleanup below treats it as a normal disconnect.
            pass
        finally:
            closed.cancel()
            tab.connections -= 1
            if tab.connections == 0:
                self._schedule_disconnect(client, tab)

    def _schedule_disconnect(self, client: str, tab: _Tab) -> None:
        """Treat the tab as gone only if it does not come back within the grace window."""
        loop = asyncio.get_running_loop()
        # Bound now to the attempt this tab owns, so a late timer can never cancel a newer attempt.
        attempt = self._controller.owner_attempt(client)

        def _gone() -> None:
            tab.grace = None
            if tab.connections != 0:
                return
            self._tabs.pop(client, None)
            if attempt is None:
                return
            task = loop.create_task(
                self._controller.client_disconnected(client, attempt)
            )
            self._background.add(task)
            task.add_done_callback(self._background.discard)

        tab.grace = loop.call_later(_DISCONNECT_GRACE_S, _gone)

    async def _reply(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        ctype: str,
        body: bytes,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """A complete response with the security headers."""
        headers = {
            "Content-Type": ctype,
            "Content-Length": str(len(body)),
            "Connection": "close",
            **_SECURITY_HEADERS,
            **(extra_headers or {}),
        }
        writer.write(_head(status, headers) + body)
        await writer.drain()

    async def _reply_json(
        self, writer: asyncio.StreamWriter, status: int, data: dict[str, Any]
    ) -> None:
        await self._reply(writer, status, "application/json", json.dumps(data).encode())


def _head(status: int, headers: dict[str, str]) -> bytes:
    """Status line and headers."""
    lines = [f"HTTP/1.1 {status} {_REASONS.get(status, 'Status')}"]
    lines += [f"{name}: {value}" for name, value in headers.items()]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def _sides_of(data: dict[str, Any]) -> list[proto.Side]:
    """The sides a start request names; anything but a nonempty list of left/right is a bad request."""
    raw = data.get("sides")
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) for item in raw)
    ):
        raise _BadRequest(400, "sides must be a nonempty list")
    try:
        sides = [proto.Side(item) for item in raw]
    except ValueError:
        raise _BadRequest(400, "sides must be left or right") from None
    if len(set(sides)) != len(sides):
        raise _BadRequest(400, "sides must not repeat")
    return sides
