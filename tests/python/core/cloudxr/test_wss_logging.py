# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise handshake logging with real WebSocket server connections."""

import asyncio
import logging

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from cloudxr_py_test_ns.wss import _WSSServerLogger


@pytest.mark.parametrize(
    ("payload", "benign"),
    [
        (b"", True),
        (b"GET /", False),
        (b"GET / HTTP/1.1\r\n", False),
        (b"invalid\r\n\r\n", False),
    ],
)
async def test_handshake_logging(payload, benign, caplog):
    logger = logging.getLogger("isaaccapture.cloudxr.wss.test")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    handled = []

    async def handler(ws):
        handled.append(ws)

    async with serve(
        handler, "127.0.0.1", 0, logger=_WSSServerLogger(logger, {})
    ) as server:
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(payload)
        await writer.drain()
        writer.write_eof()
        await asyncio.wait_for(reader.read(), timeout=5)
        writer.close()
        await writer.wait_closed()

        assert not handled
        # A rejected probe must not prevent subsequent clients from connecting.
        async with connect(f"ws://127.0.0.1:{port}"):
            pass

    records = [record for record in caplog.records if record.name == logger.name]
    if benign:
        assert not any(record.levelno >= logging.WARNING for record in records)
        record = next(r for r in records if "before sending" in r.getMessage())
        assert record.levelno == logging.DEBUG
        assert record.exc_info is None
    else:
        record = next(
            r for r in records if r.getMessage() == "opening handshake failed"
        )
        assert record.levelno == logging.ERROR
        assert record.exc_info is not None
    assert len(handled) == 1
