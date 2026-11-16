# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""send_raw_stop: a STOP for a robot link service that takes no further client.

A raw STOP is a LOCAL_HELLO and a STOP written to the local socket in one go,
with no LocalCarrier. The service forwards it even when it turns the client
away.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from isaaccapture_examples.agentic_remote.link_service import local
from isaaccapture_examples.agentic_remote.link_service.local import (
    LocalRefused,
    RawStop,
    connect_local,
    send_raw_stop,
)
from isaaccapture_examples.agentic_remote.link_service.service import (
    STOP_UNDELIVERED,
)
from isaaccapture_examples.agentic_remote.protocol.messages import Stop
from link_bench import Bench

FULL = "too many local clients are attached"


@pytest.fixture
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short private folder: a socket path holds 107 bytes at most."""
    return tmp_path_factory.mktemp("rs")


@pytest.fixture
def full(folder: Path) -> Iterator[Bench]:
    """A service with one client slot, taken by a viewer."""
    made = Bench(folder, max_clients=1)
    yield made
    made.close()


def test_a_raw_stop_goes_out_when_every_client_slot_is_taken(full: Bench) -> None:
    thor = full.robot()
    full.client("viewer")
    with pytest.raises(LocalRefused, match=FULL):
        connect_local(full.path, "console", "console", 2.0)
    raw = send_raw_stop(full.path, reason="typed stop", sender="console")
    assert raw == RawStop(written=True, notice=FULL)
    assert thor.next_message() == Stop(reason="typed stop", sender="console")
    thor.nothing()  # one STOP, and nothing else of this client


def test_a_raw_stop_reports_the_undelivered_notice_without_a_robot_connection(
    folder: Path,
) -> None:
    bench = Bench(folder)  # no robot connection, and room for the client
    try:
        raw = send_raw_stop(
            bench.path, reason="typed stop", sender="console", timeout_s=0.5
        )
        assert raw == RawStop(written=True, notice=STOP_UNDELIVERED)
    finally:
        bench.close()


def test_a_raw_stop_to_no_service_returns_not_written(folder: Path) -> None:
    raw = send_raw_stop(str(folder / "none.sock"), reason="r", sender="console")
    assert raw.written is False
    assert raw.notice.startswith("no robot link service at ")
    # A socket file nobody listens on is the same end.
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(folder / "stale.sock"))
    stale.close()
    raw = send_raw_stop(str(folder / "stale.sock"), reason="r", sender="console")
    assert raw.written is False and "stale.sock" in raw.notice


def test_a_raw_stop_makes_one_attempt(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dials: list[str] = []
    real = local._dial

    def dial(sock: socket.socket, path: str) -> None:
        dials.append(path)
        real(sock, path)

    monkeypatch.setattr(local, "_dial", dial)
    path = str(folder / "none.sock")
    began = time.monotonic()
    assert not send_raw_stop(path, reason="r", sender="console").written
    assert dials == [path]  # no second try
    assert time.monotonic() - began < 1.0
    # With a service there, it is one connection and one STOP as well.
    bench = Bench(folder / "b")
    try:
        thor = bench.robot()
        raw = send_raw_stop(bench.path, reason="r", sender="console", timeout_s=0.3)
        assert raw == RawStop(written=True, notice="")
        assert dials == [path, bench.path]
        assert thor.next_message() == Stop(reason="r", sender="console")
        thor.nothing()
    finally:
        bench.close()
