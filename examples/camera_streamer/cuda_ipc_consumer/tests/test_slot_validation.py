# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CudaIpcClient's slot write-generation handshake against a fake publisher (no GPU)."""

import json
import socket
import threading

import pytest

from consumer import CudaIpcClient
from ipc_protocol import SlotMarkers

FEATURES = ["slot_generation", "slot_write_generation"]


class _FakePublisher:
    def __init__(self, path, hello, subscribed):
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(path)
        self.server.listen(1)
        self.thread = threading.Thread(
            target=self._serve, args=(hello, subscribed), daemon=True
        )
        self.thread.start()

    def _serve(self, hello, subscribed):
        conn, _ = self.server.accept()
        with conn:
            conn.sendall(json.dumps(hello).encode() + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
            if subscribed is not None:
                conn.sendall(json.dumps(subscribed).encode() + b"\n")
            conn.settimeout(2)
            try:
                conn.recv(1)
            except OSError:
                pass

    def close(self):
        self.server.close()
        self.thread.join(timeout=5)


@pytest.fixture
def markers():
    region = SlotMarkers.create("/consumer_test", [2, 1])
    yield region
    region.close()


def _descriptor(region, streams=None):
    return {
        "name": region.name,
        "size": region.size,
        "layout_version": region.layout_version,
        "epoch": region.epoch,
        "stream_indexes": streams or {"left": 0, "right": 1},
    }


@pytest.fixture
def client_for(tmp_path, monkeypatch):
    created = []

    def make(hello, subscribed):
        path = str(tmp_path / "pub.sock")
        publisher = _FakePublisher(path, hello, subscribed)
        client = CudaIpcClient(path)
        monkeypatch.setattr(client, "_open_streams", lambda _subscribed: None)
        created.append((publisher, client))
        return client

    yield make
    for publisher, client in created:
        client.close()
        publisher.close()


def _hello(features=FEATURES):
    return {"type": "hello", "version": 2, "features": features, "streams": []}


def test_refuses_a_hello_without_slot_write_generation(client_for, markers):
    client = client_for(
        _hello(["slot_generation"]),
        {"type": "subscribed", "slot_markers": _descriptor(markers), "streams": []},
    )
    with pytest.raises(RuntimeError, match="slot write generations"):
        client.connect()
    assert client.markers is None


def test_refuses_a_subscribed_without_slot_markers(client_for):
    client = client_for(_hello(), {"type": "subscribed", "streams": []})
    with pytest.raises(RuntimeError, match="slot write generations"):
        client.connect()
    assert client.markers is None


def test_refuses_a_subscribed_with_null_slot_markers(client_for):
    client = client_for(
        _hello(), {"type": "subscribed", "slot_markers": None, "streams": []}
    )
    with pytest.raises(RuntimeError, match="slot write generations"):
        client.connect()


def test_refuses_markers_of_a_stale_epoch(client_for, markers):
    descriptor = _descriptor(markers)
    descriptor["epoch"] ^= 1
    client = client_for(
        _hello(), {"type": "subscribed", "slot_markers": descriptor, "streams": []}
    )
    with pytest.raises(RuntimeError):
        client.connect()
    assert client.markers is None


def test_slot_unchanged_follows_the_shared_marker(client_for, markers):
    client = client_for(
        _hello(),
        {"type": "subscribed", "slot_markers": _descriptor(markers), "streams": []},
    )
    client.connect()
    generation = markers.begin_write(0, 1)
    assert client.slot_unchanged("left", 1, generation)
    markers.begin_write(0, 1)
    assert not client.slot_unchanged("left", 1, generation)
    # Streams map through stream_indexes, not by name order.
    other = markers.begin_write(1, 0)
    assert client.slot_unchanged("right", 0, other)
    assert not client.slot_unchanged("right", 0, other - 1)
    with pytest.raises(KeyError):
        client.slot_unchanged("missing", 0, 0)


def test_close_closes_the_region(client_for, markers):
    client = client_for(
        _hello(),
        {"type": "subscribed", "slot_markers": _descriptor(markers), "streams": []},
    )
    client.connect()
    attached = client.markers
    assert attached is not None
    client.close()
    assert client.markers is None
    with pytest.raises(RuntimeError, match="closed"):
        attached.load(0, 0)
    client.close()  # idempotent
