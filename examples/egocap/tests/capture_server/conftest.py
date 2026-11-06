# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixture for a real, running CaptureServer with mock acquisition.
Normal CaptureClient integration tests use this instead of reimplementing
server behavior. A deliberate protocol violation the real server cannot be
configured to produce uses its own narrow, test-local peer instead (see
test_client.py's lifecycle-ordering/replay section and its two
never-confirms-Stop fixtures)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from tests.support.capture_server import start_capture_server, stop_capture_server


@pytest.fixture
async def real_mock_server(unused_tcp_port: int) -> AsyncIterator[str]:
    """Yields the ws:// URL of a running production CaptureServer with mock acquisition."""
    server = await start_capture_server(unused_tcp_port)
    try:
        yield f"ws://127.0.0.1:{unused_tcp_port}"
    finally:
        await stop_capture_server(server)


@pytest.fixture
async def make_station(tmp_path, monkeypatch, unused_tcp_port_factory):
    """Starts a CaptureServer with the fake calibration worker; every station is stopped on teardown."""
    from tests.capture_server.calibration_helpers import start_station, stop_station

    stations = []

    async def make(*, scenario=None, calibration=True, config=None):
        station = await start_station(
            tmp_path,
            monkeypatch,
            unused_tcp_port_factory(),
            scenario=scenario,
            calibration=calibration,
            config=config,
        )
        stations.append(station)
        return station

    try:
        yield make
    finally:
        for station in stations:
            await stop_station(station)
