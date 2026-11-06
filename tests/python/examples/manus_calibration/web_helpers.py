# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A raw-socket HTTP client, an SSE reader and a server harness for the page tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from isaaccapture_examples.manus_calibration.server import PageServer

CLIENT_A = "client-a-0000001"
CLIENT_B = "client-b-0000002"
IO_TIMEOUT_S = 10.0


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


def _parse(raw: bytes) -> Response:
    """Split a ``Connection: close`` response into status, headers and body."""
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ")[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return Response(status, headers, body)


async def raw_request(port: int, data: bytes) -> Response:
    """Send exactly ``data`` and read the response until the server closes."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(data)
        await writer.drain()
        return _parse(await asyncio.wait_for(reader.read(), IO_TIMEOUT_S))
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


async def request(
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
    host: str | None = None,
) -> Response:
    """One request with a correct Host header unless ``host`` overrides it."""
    sent = {"Host": host if host is not None else f"127.0.0.1:{port}"}
    sent.update(headers or {})
    if body and "Content-Length" not in sent:
        sent["Content-Length"] = str(len(body))
    lines = [f"{method} {path} HTTP/1.1"] + [f"{k}: {v}" for k, v in sent.items()]
    return await raw_request(port, ("\r\n".join(lines) + "\r\n\r\n").encode() + body)


async def api(
    port: int,
    path: str,
    client: str | None,
    body: Any = None,
    *,
    headers: dict[str, str] | None = None,
) -> Response:
    """POST JSON the way the page does, with the per-tab client id header."""
    sent = {"Content-Type": "application/json"}
    if client is not None:
        sent["X-Client-Id"] = client
    sent.update(headers or {})
    payload = b"" if body is None else json.dumps(body).encode()
    return await request(port, "POST", path, headers=sent, body=payload)


class Sse:
    """One open ``/events`` stream; reads complete ``data:`` events and comment lines."""

    def __init__(self, reader, writer, status: int, headers: dict[str, str]) -> None:
        self._reader = reader
        self._writer = writer
        self.status = status
        self.headers = headers
        self.payloads: list[str] = []

    @classmethod
    async def open(cls, port: int, client: str, *, host: str | None = None) -> Sse:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        request_head = (
            f"GET /events?client={client} HTTP/1.1\r\n"
            f"Host: {host if host is not None else f'127.0.0.1:{port}'}\r\n\r\n"
        )
        writer.write(request_head.encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), IO_TIMEOUT_S)
        parsed = _parse(head)
        return cls(reader, writer, parsed.status, parsed.headers)

    async def line(self, timeout: float = IO_TIMEOUT_S) -> str | None:
        """The next raw line, or None at end of stream."""
        data = await asyncio.wait_for(self._reader.readline(), timeout)
        return data.decode().rstrip("\n") if data else None

    async def event(self, timeout: float = IO_TIMEOUT_S) -> dict[str, Any]:
        """The next ``data:`` event, skipping comments; fails at end of stream."""
        async with asyncio.timeout(timeout):
            while True:
                line = await self.line(timeout)
                assert line is not None, "stream ended"
                if line.startswith("data: "):
                    self.payloads.append(line[6:])
                    return json.loads(line[6:])

    async def until(
        self, predicate: Callable[[dict[str, Any]], bool], timeout: float = 20.0
    ) -> dict[str, Any]:
        """Read events until ``predicate`` holds for one of them."""
        async with asyncio.timeout(timeout):
            while True:
                state = await self.event(timeout)
                if predicate(state):
                    return state

    async def at_end(self, timeout: float = IO_TIMEOUT_S) -> bool:
        """True when the server closes the stream within ``timeout``."""
        async with asyncio.timeout(timeout):
            while await self.line(timeout) is not None:
                pass
        return True

    def close(self) -> None:
        self._writer.close()


@contextlib.asynccontextmanager
async def serving(
    make_worker, scenario: dict | None = None, **settings
) -> AsyncIterator[tuple[PageServer, Any]]:
    """A running PageServer on an ephemeral port over a real fake-worker subprocess."""
    worker, store = make_worker(scenario or {}, **settings)
    server = PageServer(worker, store.directory, "127.0.0.1", 0)
    await server.start()
    try:
        yield server, store
    finally:
        await server.aclose()


async def run_flow(
    port: int,
    sse: Sse,
    client: str,
    sides: list[str],
    *,
    timeout: float = 30.0,
    seen: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Start an attempt and confirm every prompt the owner is offered until it finishes.

    Every state read from the stream is appended to ``seen`` when it is given.
    """
    started = await api(port, "/api/start", client, {"sides": sides})
    assert started.status == 200, started.body
    confirmed: set[str] = set()
    async with asyncio.timeout(timeout):
        while True:
            state = await sse.event()
            if seen is not None:
                seen.append(state)
            if state["stage"] == "finished":
                return state
            prompt = state["prompt"]
            if (
                state["stage"] == "awaiting_confirmation"
                and prompt
                and prompt["token"]
                and prompt["token"] not in confirmed
                and not state["confirm_pending"]
            ):
                confirmed.add(prompt["token"])
                reply = await api(
                    port, "/api/confirm", client, {"token": prompt["token"]}
                )
                assert reply.status == 200, reply.body
