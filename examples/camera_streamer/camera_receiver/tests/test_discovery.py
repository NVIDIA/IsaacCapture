# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream discovery: the receiver builds its streams from the sender's bundles."""

import csv
import json
import socket
import sys

import pytest

import camera_receiver
from avs_wire import HEADER, SAMPLE, STREAM, parse_avs1_record
from camera_receiver import (
    Avs1TcpReceiver,
    LatencyCsv,
    ReceivedRecord,
    ReceiveFpsMonitor,
    ReceiveFpsWarningConfig,
    discovered_stream_configs,
)
from cuda_ipc import CudaIpcFramePublisher
from fake_gpu import FakeDecoder, fake_cupy

UNSPECIFIED = 0xFFFFFFFF
SEND_NS = 5_000_000_000
RECEIVE_NS = 5_002_500_000


def record(streams, present, *, sequence=1, session=7, stream_sequence=1):
    """One sender bundle. streams: (position, stereo, width, height, fps);
    present: index of the stream that carries a frame."""
    payload = b"\x00\x00\x00\x01frame"
    stream_descriptors, sample_descriptors, body = [], [], b""
    for index, (position, stereo, width, height, fps) in enumerate(streams):
        here = index == present
        count = 2 if stereo else 1
        stream_descriptors.append(
            STREAM.pack(
                index,
                (1 if here else 0) | (2 if stereo else 0),
                1,
                1,
                0,
                width,
                height,
                fps,
                1,
                stream_sequence if here else 0,
                len(sample_descriptors),
                count,
                0,
            )
        )
        for eye in range(count):
            sample_descriptors.append(
                SAMPLE.pack(
                    index,
                    eye,
                    (eye + 1) if stereo else 0,
                    0x9 if here else 0,
                    0,
                    position,
                    UNSPECIFIED,
                    40 + sequence if here else 0,
                    0,
                    0,
                    SEND_NS - 3000 if here else 0,
                    SEND_NS - 2000 if here else 0,
                    SEND_NS - 1000 if here else 0,
                    len(body),
                    len(payload) if here else 0,
                )
            )
            if here:
                body += payload
    descriptors = b"".join(stream_descriptors) + b"".join(sample_descriptors)
    header = HEADER.pack(
        b"AVS1",
        1,
        1,
        HEADER.size,
        len(stream_descriptors),
        len(sample_descriptors),
        sequence,
        session,
        SEND_NS - 3000,
        SEND_NS,
        len(descriptors),
        len(body),
        0,
    )
    return parse_avs1_record(header + descriptors + body)


RECEIVED = ReceivedRecord(
    payload=b"",
    receiver_rx_prefix_done_ns=RECEIVE_NS - 10,
    receiver_rx_payload_done_ns=RECEIVE_NS,
    receiver_rx_prefix_done_mono_ns=1,
    receiver_rx_payload_done_mono_ns=2,
    receiver_clock_domain="realtime",
)
THREE_LEFT_EYES = [
    (0, False, 320, 180, 15),
    (1, False, 320, 180, 15),
    (2, False, 640, 360, 15),
]


def names(configs):
    return [name for config in configs for name in config.ipc_names]


def test_names_follow_the_sender_config_position():
    configs = discovered_stream_configs(record(THREE_LEFT_EYES, 0).streams)
    assert names(configs) == ["cam0", "cam1", "cam2"]
    assert [(c.width, c.height, c.stereo) for c in configs] == [
        (320, 180, False),
        (320, 180, False),
        (640, 360, False),
    ]


def test_a_disabled_camera_does_not_rename_the_others():
    layout = [(0, False, 320, 180, 15), (2, True, 640, 360, 15)]
    configs = discovered_stream_configs(record(layout, 0).streams)
    assert names(configs) == ["cam0", "cam2.left", "cam2.right"]
    assert [c.index for c in configs] == [0, 1]


def test_names_follow_the_wire_index_without_positions():
    layout = [(UNSPECIFIED, True, 1280, 720, 60), (UNSPECIFIED, False, 640, 480, 60)]
    configs = discovered_stream_configs(record(layout, 0).streams)
    assert names(configs) == ["stream0.left", "stream0.right", "stream1"]


def test_latency_csv_has_send_and_receive_time(tmp_path):
    path = tmp_path / "latency.csv"
    log = LatencyCsv(path)
    bundle = record(THREE_LEFT_EYES, 2, sequence=4, stream_sequence=9)
    log.write("cam2", bundle, bundle.streams[2], RECEIVED)
    log.close()
    (row,) = list(csv.DictReader(path.open()))
    assert row["stream"] == "cam2"
    assert (row["bundle_sequence"], row["stream_sequence"], row["frame_number"]) == (
        "4",
        "9",
        "44",
    )
    assert int(row["sender_send_ns"]) == SEND_NS
    assert int(row["receiver_receive_ns"]) == RECEIVE_NS
    assert float(row["receive_minus_send_ms"]) == pytest.approx(2.5)


def test_fps_monitor_uses_each_streams_own_target(caplog):
    now = [0.0]
    monitor = ReceiveFpsMonitor(
        ["cam0", "cam1", "cam2"],
        ReceiveFpsWarningConfig(threshold_fps=0.0, interval_s=1.0, fraction=0.9),
        clock=lambda: now[0],
        thresholds={"cam0": 13.5, "cam1": 13.5, "cam2": 27.0},
    )
    monitor.start()
    for _ in range(15):
        monitor.observe(["cam0"])
    for _ in range(15):
        monitor.observe(["cam2"])
    now[0] = 1.0
    with caplog.at_level("WARNING"):
        monitor.poll()
    message = caplog.text
    assert "cam0" not in message
    assert "cam1=0.0fps(target 13.5)" in message
    assert "cam2=15.0fps(target 27.0)" in message


class Consumer:
    """A CUDA IPC consumer that only follows the control messages."""

    def __init__(self, path, streams=()):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(path)
        self.buffer = b""
        self.hello = self.read()
        self.sock.sendall(
            json.dumps({"type": "subscribe", "streams": list(streams)}).encode() + b"\n"
        )
        self.subscribed = self.read()

    def read(self):
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(65536)
            assert chunk, "publisher closed the connection"
            self.buffer += chunk
        line, _, self.buffer = self.buffer.partition(b"\n")
        return json.loads(line)

    def read_until(self, predicate):
        while True:
            message = self.read()
            if predicate(message):
                return message

    def close(self):
        self.sock.close()


def stream_names(message):
    return [stream["name"] for stream in message["streams"]]


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy())
    monkeypatch.setattr(camera_receiver, "NvVideoDecoder", FakeDecoder)
    publisher = CudaIpcFramePublisher(
        socket_path=str(tmp_path / "ipc.sock"), dynamic_streams=True
    )
    receiver = Avs1TcpReceiver(
        host="127.0.0.1",
        port=1,
        gpu_id=0,
        max_record_bytes=1 << 20,
        stats_log_interval_s=0,
        receive_fps_warning=ReceiveFpsWarningConfig(0.0, 10.0, 0.9),
        ipc_publisher=publisher,
        streams=[],
        discover=True,
        latency_csv=LatencyCsv(tmp_path / "latency.csv"),
    )
    # Run the receiver without its network thread; bundles are fed in directly.
    receiver._resize_sample_pool()
    publisher.start()
    yield receiver
    receiver.stop()


def test_consumer_sees_every_camera_before_it_delivers(receiver):
    consumer = Consumer(receiver.socket_path)
    assert stream_names(consumer.subscribed) == []
    assert "dynamic_streams" in consumer.hello["features"]

    receiver._dispatch_bundle(RECEIVED, record(THREE_LEFT_EYES, 0))
    layout = consumer.read_until(lambda m: m["type"] == "streams" and m["streams"])
    # cam1 has sent nothing, yet it is announced with its size.
    assert stream_names(layout) == ["cam0", "cam1", "cam2"]
    assert [(s["width"], s["height"]) for s in layout["streams"]] == [
        (320, 180),
        (320, 180),
        (640, 360),
    ]
    frame = consumer.read_until(lambda m: m["type"] == "frame")
    assert frame["stream"] == "cam0"
    assert frame["metadata"]["sender_tcp_enqueue_ns"] == SEND_NS
    assert frame["metadata"]["receiver_rx_payload_done_ns"] == RECEIVE_NS
    consumer.close()


def test_layout_change_keeps_unchanged_streams_and_replaces_the_rest(receiver):
    consumer = Consumer(receiver.socket_path)
    receiver._dispatch_bundle(RECEIVED, record(THREE_LEFT_EYES, 0))
    first = consumer.read_until(lambda m: m["type"] == "streams" and m["streams"])
    cam0 = receiver._runtimes[0]
    epochs = {s["name"]: s["epoch"] for s in first["streams"]}

    # The sender restarts with cam1 disabled and cam2 sending both eyes.
    changed = [(0, False, 320, 180, 15), (2, True, 640, 360, 15)]
    receiver._dispatch_bundle(RECEIVED, record(changed, 1, session=8))
    second = consumer.read_until(
        lambda m: m["type"] == "streams" and "cam2.left" in stream_names(m)
    )
    assert stream_names(second) == ["cam0", "cam2.left", "cam2.right"]
    assert second["streams"][0]["epoch"] == epochs["cam0"]
    assert receiver._runtimes[0] is cam0
    assert sorted(rt.config.name for rt in receiver._retired) == ["cam1", "cam2"]
    frame = consumer.read_until(lambda m: m["type"] == "frame")
    assert frame["stream"] in ("cam2.left", "cam2.right")

    # A stream that returns with a new size keeps its name and gets a new epoch.
    resized = [(0, False, 640, 360, 15)]
    receiver._dispatch_bundle(RECEIVED, record(resized, 0, session=9))
    third = consumer.read_until(
        lambda m: m["type"] == "streams" and stream_names(m) == ["cam0"]
    )
    assert third["streams"][0]["width"] == 640
    assert third["streams"][0]["epoch"] != epochs["cam0"]
    consumer.close()


def test_streams_are_listed_in_sender_order_after_a_change(receiver):
    consumer = Consumer(receiver.socket_path)
    two_left_eyes = [(0, False, 320, 180, 15), (2, False, 640, 360, 15)]
    receiver._dispatch_bundle(RECEIVED, record(two_left_eyes, 0))
    consumer.read_until(lambda m: m["type"] == "streams" and m["streams"])
    # cam0 switches to both eyes; cam2 is unchanged and was registered earlier.
    cam0_stereo = [(0, True, 320, 180, 15), (2, False, 640, 360, 15)]
    receiver._dispatch_bundle(RECEIVED, record(cam0_stereo, 1, session=8))
    layout = consumer.read_until(
        lambda m: m["type"] == "streams" and "cam0.left" in stream_names(m)
    )
    assert stream_names(layout) == ["cam0.left", "cam0.right", "cam2"]
    consumer.close()


def test_consumer_may_ask_for_a_stream_before_it_exists(receiver):
    consumer = Consumer(receiver.socket_path, streams=["cam2"])
    assert stream_names(consumer.subscribed) == []
    receiver._dispatch_bundle(RECEIVED, record(THREE_LEFT_EYES, 2))
    layout = consumer.read_until(lambda m: m["type"] == "streams" and m["streams"])
    assert stream_names(layout) == ["cam2"]
    assert consumer.read_until(lambda m: m["type"] == "frame")["stream"] == "cam2"
    consumer.close()


def test_configured_streams_still_reject_unknown_names(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy())
    publisher = CudaIpcFramePublisher(socket_path=str(tmp_path / "ipc.sock"))
    publisher.register_stream(
        "head.left", 64, 64, "nv12", 0, [fake_cupy().empty((96, 64))]
    )
    publisher.start()
    try:
        consumer = Consumer(str(tmp_path / "ipc.sock"), streams=["nope"])
        assert consumer.subscribed["type"] == "error"
        consumer.close()
    finally:
        publisher.stop()


def test_discovery_config_has_no_streams_list():
    assert camera_receiver._configured_streams({"receiver": {}}) == []
    with pytest.raises(ValueError):
        camera_receiver._configured_streams({"streams": []})
