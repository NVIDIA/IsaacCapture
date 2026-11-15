# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for the camera receiver, for the tests of the camera frame bridge.

FakeReceiver speaks the receiver's notification protocol on a Unix socket. Its
slots are files in a folder instead of GPU buffers, so that a bridge in another
process can open them without CuPy.
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    import isaaccapture_examples.agentic_remote.protocol  # noqa: F401
except ImportError:
    _root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "VERSION").is_file() and (parent / "CMakeLists.txt").is_file()
    )
    sys.path.insert(0, str(_root / "examples" / "agentic_remote" / "python"))

from isaaccapture_examples.agentic_remote.agent.camera.frame_bridge import (
    SlotReader,
)

STREAMS = ("head.left", "head.right")
SOF_NS = 1_790_000_000_000_000_000  # a Thor wall clock time, in ns

# A bridge program without CuPy: the real main() on FakeReceiver's file slots.
BRIDGE_CHILD = f"""
import sys

sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
import camera_bench
from isaaccapture_examples.agentic_remote.agent.camera import frame_bridge

sys.exit(frame_bridge.main(open_slots=camera_bench.open_file_slots))
"""


def notification(
    stream: str,
    seq: int,
    slot: int,
    timestamp_ns: int,
    *,
    session: int = 41,
    generation: int = 1,
    sof: tuple[int | None, int | None] = (SOF_NS, SOF_NS + 1000),
    devices: tuple[int, int] = (5, 7),
    rotation: int = 0,
    width: int = 2560,
    height: int = 1984,
    clock: str | None = "realtime",
    rx_mono_ns: int | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """One "frame" notification with the metadata keys the receiver sends."""
    eye = STREAMS.index(stream)
    delta = None if None in sof else sof[1] - sof[0]
    meta = {
        "source_format": "avs1",
        "session_id": session,
        "stream_sequence": seq,
        "stream_rotation": rotation,
        "camera_index": eye,
        "role": ("left", "right")[eye],
        "device_index": devices[eye],
        "width": width,
        "height": height,
        "ipc_pixel_format": "nv12",
        "sender_wall_clock": clock,
        "sensor_sof_ns": sof[eye],
        "left_sensor_sof_ns": sof[0],
        "right_sensor_sof_ns": sof[1],
        "tsc_delta_ns": delta,
        "receiver_rx_payload_done_mono_ns": (
            timestamp_ns - 4_000_000 if rx_mono_ns is None else rx_mono_ns
        ),
        "receiver_ipc_publish_mono_ns": timestamp_ns,
    }
    meta.update(metadata)
    return {
        "type": "frame",
        "stream": stream,
        "slot": slot,
        "slot_generation": generation,
        "seq": seq,
        "timestamp_ns": timestamp_ns,
        "metadata": meta,
    }


def open_file_slots(
    subscribed: Mapping[str, Any], camera_pr: Path, gpu_id: int
) -> SlotReader:
    """What frame_bridge.open_cuda_slots does, for FakeReceiver's file slots."""
    slots = {}
    for stream in subscribed["streams"]:
        for slot in stream["slots"]:
            path = base64.b64decode(slot["handle"]).decode()
            slots[(stream["name"], slot["slot"])] = np.memmap(
                path, dtype=np.uint8, mode="r", shape=tuple(slot["shape"])
            )
    return SlotReader(slots, lambda array: np.array(array))


class FakeReceiver:
    """The camera receiver's publisher with files as slots; publish() by hand."""

    def __init__(
        self,
        folder: Path,
        *,
        width: int = 64,
        height: int = 48,
        ring_depth: int = 4,
        gpu_id: int = 0,
        **meta: Any,
    ) -> None:
        self.socket_path = str(folder / "rx.sock")
        assert len(self.socket_path) < 100, self.socket_path
        self.width, self.height, self.ring_depth = width, height, ring_depth
        self.gpu_id = gpu_id
        self.meta = meta  # goes into every notification()
        self.shape = (height * 3 // 2, width)
        self.slots = [
            [
                np.memmap(
                    folder / f"slot_{eye}_{index}.bin",
                    dtype=np.uint8,
                    mode="w+",
                    shape=self.shape,
                )
                for index in range(ring_depth)
            ]
            for eye in range(2)
        ]
        self.write_index = 0
        self.seq = 0
        self.session = 41
        self.published: dict[int, int] = {}  # seq -> slot
        self.subscriptions: list[list[str]] = []
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        self._server.listen(4)
        self._server.settimeout(0.1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
        self._server.close()
        with self._lock:
            for client in self._clients:
                client.close()
            self._clients.clear()

    def wait_client(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not self._clients:
            assert time.monotonic() < deadline, "no bridge subscribed"
            time.sleep(0.01)

    @staticmethod
    def pixel(seq: int, eye: int) -> int:
        """The byte every pixel of that eye has after publish(seq)."""
        return (seq * 2 + eye) % 251

    def scribble(self) -> None:
        """Write into the next slot without publishing: a decode that failed."""
        for eye in range(2):
            self.slots[eye][self.write_index][:] = 255

    def publish(self, *, step: int = 1, **meta: Any) -> int:
        """Fill the next slot of both eyes and announce it; the pair's seq."""
        self.seq += step
        seq, slot = self.seq, self.write_index
        for eye in range(2):
            self.slots[eye][slot][:] = self.pixel(seq, eye)
        timestamp_ns = time.monotonic_ns()
        fields = {
            "session": self.session,
            "width": self.width,
            "height": self.height,
            **self.meta,
            **meta,
        }
        for stream in STREAMS:
            self._send(notification(stream, seq, slot, timestamp_ns, **fields))
        self.published[seq] = slot
        self.write_index = (slot + 1) % self.ring_depth
        return seq

    def _catalog(self, name: str, eye: int, *, slots: bool) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": name,
            "width": self.width,
            "height": self.height,
            "pixel_format": "nv12",
            "dtype": "uint8",
            "gpu_id": self.gpu_id,
            "slot_count": self.ring_depth,
            "layout": {},
        }
        if slots:
            entry["slots"] = [
                {
                    "slot": index,
                    "handle": base64.b64encode(
                        str(self.slots[eye][index].filename).encode()
                    ).decode(),
                    "nbytes": self.shape[0] * self.shape[1],
                    "shape": list(self.shape),
                    "strides": [self.width, 1],
                }
                for index in range(self.ring_depth)
            ]
        return entry

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            client.settimeout(2.0)
            hello = {
                "type": "hello",
                "version": 2,
                "streams": [
                    self._catalog(name, eye, slots=False)
                    for eye, name in enumerate(STREAMS)
                ],
            }
            try:
                client.sendall(json.dumps(hello).encode() + b"\n")
                line = b""
                while not line.endswith(b"\n"):
                    line += client.recv(4096)
                wanted = json.loads(line)["streams"]
                self.subscriptions.append(wanted)
                subscribed = {
                    "type": "subscribed",
                    "version": 2,
                    "streams": [
                        self._catalog(name, eye, slots=True)
                        for eye, name in enumerate(STREAMS)
                        if name in wanted
                    ],
                }
                client.sendall(json.dumps(subscribed).encode() + b"\n")
            except OSError:
                client.close()
                continue
            with self._lock:
                self._clients.append(client)

    def _send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        with self._lock:
            for client in list(self._clients):
                try:
                    client.sendall(line)
                except OSError:
                    self._clients.remove(client)
                    client.close()
