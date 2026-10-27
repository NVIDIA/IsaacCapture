# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise handshake logging with real WebSocket server connections."""

import asyncio
import logging
import sys

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidMessage
from websockets.server import ServerProtocol

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
@pytest.mark.parametrize("explicit_exc_info", [False, True])
def test_handshake_logging(payload, benign, explicit_exc_info, caplog):
    """Classify real parser errors independently of server logging versions."""
    logger = logging.getLogger("isaaccapture.cloudxr.wss.test")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    protocol = ServerProtocol()
    protocol.receive_data(payload)
    protocol.receive_eof()
    exc = protocol.handshake_exc
    assert isinstance(exc, InvalidMessage)
    try:
        raise exc
    except InvalidMessage:
        _WSSServerLogger(logger, {}).error(
            "opening handshake failed",
            exc_info=sys.exc_info() if explicit_exc_info else True,
        )

    record = next(r for r in caplog.records if r.name == logger.name)
    if benign:
        assert (
            record.getMessage() == "Client disconnected before sending an HTTP request"
        )
        assert record.levelno == logging.DEBUG
        assert record.exc_info is None
    else:
        assert record.getMessage() == "opening handshake failed"
        assert record.levelno == logging.ERROR
        assert record.exc_info[1] is exc


@pytest.mark.parametrize(
    "payload", [b"", b"GET /", b"GET / HTTP/1.1\r\n", b"invalid\r\n\r\n"]
)
async def test_live_connections_do_not_log_credentials(payload, caplog):
    """Keep wire credentials private and accept clients after rejected probes."""
    logger = logging.getLogger("isaaccapture.cloudxr.wss.test")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    handled = []

    async def handler(ws):
        """Record accepted connections."""
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
        async with connect(
            f"ws://127.0.0.1:{port}/?token=query-secret-1353",
            additional_headers={"X-Control-Token": "header-secret-1353"},
        ):
            pass

    records = [record for record in caplog.records if record.name == logger.name]
    assert records
    assert all("query-secret-1353" not in r.getMessage() for r in records)
    assert all("header-secret-1353" not in r.getMessage() for r in records)
    if not payload:
        assert not any(record.levelno >= logging.WARNING for record in records)
    # websockets 17+ already handles parser failures without an ERROR record.
    assert all(r.exc_info is None for r in records if r.levelno == logging.DEBUG)
    assert len(handled) == 1
