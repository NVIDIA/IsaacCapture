# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CudaIpcFramePublisher's slot write-generation markers with real CuPy buffers."""

import json
from pathlib import Path
import socket
import sys

import pytest

cp = pytest.importorskip("cupy", reason="CuPy is not installed")
try:
    _GPU_COUNT = cp.cuda.runtime.getDeviceCount()
except Exception as error:  # pragma: no cover - depends on the machine
    pytest.skip(f"no usable CUDA device: {error}", allow_module_level=True)
if _GPU_COUNT < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cuda_ipc import CudaIpcFramePublisher  # noqa: E402
from ipc_protocol import SlotMarkers  # noqa: E402

SLOT_COUNTS = {"left": 3, "right": 2}


def _register(publisher, counts=None):
    for name, slots in (counts or SLOT_COUNTS).items():
        publisher.register_stream(
            name,
            4,
            4,
            "rgba8",
            0,
            [cp.zeros((4, 4, 4), dtype=cp.uint8) for _ in range(slots)],
        )


@pytest.fixture
def socket_path(tmp_path):
    return str(tmp_path / "ipc.sock")


@pytest.fixture
def publisher(socket_path):
    pub = CudaIpcFramePublisher(socket_path=socket_path)
    _register(pub)
    pub.start()
    yield pub
    pub.stop()


class _Client:
    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(path)
        self.buf = b""

    def read(self):
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            assert chunk, "publisher closed the connection"
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)

    def subscribe(self):
        hello = self.read()
        self.sock.sendall(b'{"type":"subscribe","streams":[]}\n')
        return hello, self.read()

    def close(self):
        self.sock.close()


def _connect(path):
    client = _Client(path)
    try:
        return client, *client.subscribe()
    except BaseException:
        client.close()
        raise


def test_hello_lists_slot_write_generation(publisher, socket_path):
    client, hello, _ = _connect(socket_path)
    client.close()
    assert "slot_write_generation" in hello["features"]


def test_subscribed_carries_an_attachable_descriptor_in_registration_order(
    publisher, socket_path
):
    client, _, subscribed = _connect(socket_path)
    client.close()
    info = subscribed["slot_markers"]
    assert info["stream_indexes"] == {"left": 0, "right": 1}
    reader = SlotMarkers.attach(
        info["name"], info["size"], info["layout_version"], info["epoch"]
    )
    try:
        assert reader.stream_count == 2
        assert [reader.slot_count(0), reader.slot_count(1)] == [3, 2]
    finally:
        reader.close()


def test_begin_write_increases_per_stream_and_slot_without_publishing(
    publisher, socket_path
):
    client, _, subscribed = _connect(socket_path)
    info = subscribed["slot_markers"]
    reader = SlotMarkers.attach(
        info["name"], info["size"], info["layout_version"], info["epoch"]
    )
    try:
        assert [publisher.begin_write("left", 0) for _ in range(3)] == [1, 2, 3]
        assert publisher.begin_write("left", 1) == 1
        assert publisher.begin_write("right", 0) == 1
        assert reader.load(0, 0) == 3
        assert reader.load(0, 1) == 1
        assert reader.load(1, 0) == 1
        assert reader.load(1, 1) == 0
        client.sock.settimeout(0.5)
        with pytest.raises(socket.timeout):
            client.sock.recv(1)
    finally:
        reader.close()
        client.close()


def test_failed_attempt_leaves_a_gap_the_next_publish_carries(publisher, socket_path):
    client, _, subscribed = _connect(socket_path)
    info = subscribed["slot_markers"]
    reader = SlotMarkers.attach(
        info["name"], info["size"], info["layout_version"], info["epoch"]
    )
    try:
        first = publisher.begin_write("left", 0)
        publisher.publish("left", 0, first, 1, 10)
        notice = client.read()
        assert notice["slot_generation"] == first
        assert reader.unchanged(0, 0, notice["slot_generation"])

        publisher.begin_write("left", 0)  # failed attempt, never published
        assert not reader.unchanged(0, 0, notice["slot_generation"])

        last = publisher.begin_write("left", 0)
        assert last == first + 2
        publisher.publish("left", 0, last, 2, 20)
        later = client.read()
        assert later["slot_generation"] == last
        assert reader.unchanged(0, 0, later["slot_generation"])
        assert not reader.unchanged(0, 0, notice["slot_generation"])
    finally:
        reader.close()
        client.close()


def test_begin_write_before_start_raises(socket_path):
    pub = CudaIpcFramePublisher(socket_path=socket_path)
    _register(pub)
    with pytest.raises(RuntimeError, match="not started"):
        pub.begin_write("left", 0)


def test_begin_write_after_stop_raises(socket_path):
    pub = CudaIpcFramePublisher(socket_path=socket_path)
    _register(pub)
    pub.start()
    pub.stop()
    with pytest.raises(RuntimeError, match="not started"):
        pub.begin_write("left", 0)


def test_begin_write_unknown_stream_raises(publisher):
    with pytest.raises(ValueError):
        publisher.begin_write("missing", 0)


def test_registering_a_stream_after_start_raises(publisher):
    with pytest.raises(RuntimeError, match="before start"):
        publisher.register_stream(
            "late", 4, 4, "rgba8", 0, [cp.zeros((4, 4, 4), dtype=cp.uint8)]
        )


def test_restart_advertises_a_new_region_and_epoch(socket_path):
    descriptors = []
    for _ in range(2):
        pub = CudaIpcFramePublisher(socket_path=socket_path)
        _register(pub)
        pub.start()
        client, _, subscribed = _connect(socket_path)
        client.close()
        descriptors.append(subscribed["slot_markers"])
        if len(descriptors) == 1:
            pub.stop()
    try:
        old, new = descriptors
        assert new["name"] != old["name"]
        assert new["epoch"] != old["epoch"]
        with pytest.raises(RuntimeError):
            SlotMarkers.attach(
                old["name"], old["size"], old["layout_version"], old["epoch"]
            )
        with pytest.raises(RuntimeError):
            SlotMarkers.attach(
                new["name"], new["size"], new["layout_version"], old["epoch"]
            )
        SlotMarkers.attach(
            new["name"], new["size"], new["layout_version"], new["epoch"]
        ).close()
    finally:
        pub.stop()


def test_stop_removes_the_region(publisher, socket_path):
    client, _, subscribed = _connect(socket_path)
    client.close()
    info = subscribed["slot_markers"]
    path = Path("/dev/shm") / info["name"].lstrip("/")
    assert path.exists()
    publisher.stop()
    assert not path.exists()
    with pytest.raises(RuntimeError):
        SlotMarkers.attach(
            info["name"], info["size"], info["layout_version"], info["epoch"]
        )
