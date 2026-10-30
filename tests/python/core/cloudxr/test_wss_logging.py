# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the WSS proxy's handshake behavior with real connections."""

import asyncio
import contextlib
import logging

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from cloudxr_py_test_ns import wss


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"GET /",
        b"GET / HTTP/1.1\r\n",
        b"invalid\r\n\r\n",
        b"OPTIONS / HTTP/1.1\r\nHost: localhost\r\n"
        b"Access-Control-Request-Method: GET\r\n\r\n",
    ],
)
async def test_proxy_handshakes_are_quiet_and_credentials_stay_private(
    payload, monkeypatch, caplog
):
    """Accept clients after probes and preflights without tracing credentials."""
    caplog.set_level(logging.DEBUG)
    listening = asyncio.Event()
    stop = asyncio.get_running_loop().create_future()
    port = None

    @contextlib.asynccontextmanager
    async def serve_proxy(handler, **kwargs):
        """Bind the production server to an ephemeral loopback port."""
        nonlocal port
        kwargs.update(host="127.0.0.1", port=0)
        async with serve(handler, **kwargs) as server:
            port = server.sockets[0].getsockname()[1]
            yield

    async def echo(ws, *_):
        """Stand in for the CloudXR backend without starting its runtime."""
        await ws.send(await ws.recv())

    monkeypatch.setattr(wss, "default_cert_paths", lambda: None)
    monkeypatch.setattr(wss, "ensure_certificate", lambda *_: None)
    monkeypatch.setattr(wss, "build_ssl_context", lambda *_: None)
    monkeypatch.setattr(wss, "ws_serve", serve_proxy)
    monkeypatch.setattr(wss, "proxy_handler", echo)
    # run() configures this shared logger; restore it after each test.
    monkeypatch.setattr(logging.getLogger("websockets"), "level", logging.DEBUG)
    task = asyncio.create_task(
        wss.run(None, stop, proxy_port=0, on_listening=listening.set)
    )
    try:
        await asyncio.wait_for(listening.wait(), timeout=5)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(payload)
            await writer.drain()
            if not payload.startswith(b"OPTIONS"):
                writer.write_eof()
            response = await asyncio.wait_for(reader.read(), timeout=5)
        finally:
            writer.close()
            await writer.wait_closed()

        if payload.startswith(b"OPTIONS"):
            assert response.startswith(b"HTTP/1.1 200 OK\r\n")
            assert b"Access-Control-Allow-Origin: *\r\n" in response

        async with connect(
            f"ws://127.0.0.1:{port}/?token=query-secret-1353",
            additional_headers={"X-Control-Token": "header-secret-1353"},
        ) as client:
            await client.send("offer")
            assert await asyncio.wait_for(client.recv(), timeout=5) == "offer"
    finally:
        if not stop.done():
            stop.set_result(None)
        await asyncio.wait_for(task, timeout=5)

    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert "query-secret-1353" not in caplog.text
    assert "header-secret-1353" not in caplog.text
    assert not any(r.exc_info for r in caplog.records)
