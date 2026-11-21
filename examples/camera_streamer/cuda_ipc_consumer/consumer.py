#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example CUDA IPC consumer for camera_receiver AVS1 streams."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import math
import select
import signal
import socket
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("isaaccapture.camera_streamer.consumer")

# Streams up to this width are shown at native size unless --scale is given.
_NATIVE_SCALE_MAX_WIDTH = 1280
_LARGE_STREAM_SCALE = 0.35
_WAITING_TILE_SIZE = (640, 360)
_REDRAW_INTERVAL_S = 0.25


@dataclass
class IpcSlot:
    stream: str
    slot: int
    array: object
    owner: "IpcMemory"


class IpcMemory:
    def __init__(self, ptr: int) -> None:
        self.ptr = ptr
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        import cupy as cp

        cp.cuda.runtime.ipcCloseMemHandle(self.ptr)
        self.closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CudaIpcClient:
    def __init__(self, socket_path: str, streams: Optional[list[str]] = None) -> None:
        self.socket_path = socket_path
        self.requested_streams = list(streams or [])
        self.sock: Optional[socket.socket] = None
        self.recv_buf = bytearray()
        self.slots: dict[tuple[str, int], IpcSlot] = {}
        self.stream_order: list[str] = []
        self.stream_info: dict[str, dict] = {}

    def connect(self) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        self.sock = sock
        hello = self._read_json_blocking()
        if hello is None or hello.get("type") != "hello":
            raise RuntimeError(f"expected hello message, got {hello!r}")
        self._send_json({"type": "subscribe", "streams": self.requested_streams})
        subscribed = self._read_json_blocking()
        if subscribed is None:
            raise RuntimeError("publisher disconnected before subscription")
        if subscribed.get("type") == "error":
            raise RuntimeError(str(subscribed.get("error", "subscription failed")))
        if subscribed.get("type") != "subscribed":
            raise RuntimeError(f"expected subscribed message, got {subscribed!r}")
        self._open_streams(subscribed)
        sock.setblocking(False)
        return {"hello": hello, "subscribed": subscribed}

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                logger.debug(
                    "CUDA IPC: failed to close publisher socket", exc_info=True
                )
            self.sock = None
        for slot in self.slots.values():
            slot.owner.close()
        self.slots.clear()
        self.recv_buf.clear()

    def read_messages(self, timeout_s: float = 0.05) -> Optional[list[dict]]:
        """Return all currently available messages, or None on socket EOF."""
        if self.sock is None:
            return None

        messages = self._pop_complete_messages()
        if messages:
            return messages

        readable, _, _ = select.select([self.sock], [], [], timeout_s)
        if not readable:
            return []

        while True:
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                break
            if not chunk:
                return None
            self.recv_buf.extend(chunk)
        return self._pop_complete_messages()

    def _send_json(self, msg: dict) -> None:
        assert self.sock is not None
        self.sock.sendall(
            json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n"
        )

    def _read_json_blocking(self) -> Optional[dict]:
        assert self.sock is not None
        while b"\n" not in self.recv_buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                return None
            self.recv_buf.extend(chunk)
        line = self._pop_line()
        return json.loads(line.decode("utf-8"))

    def _pop_complete_messages(self) -> list[dict]:
        messages: list[dict] = []
        while b"\n" in self.recv_buf:
            line = self._pop_line()
            messages.append(json.loads(line.decode("utf-8")))
        return messages

    def _pop_line(self) -> bytes:
        idx = self.recv_buf.index(ord("\n"))
        line = bytes(self.recv_buf[:idx])
        del self.recv_buf[: idx + 1]
        return line

    def _open_streams(self, subscribed: dict) -> None:
        self.apply_streams(subscribed.get("streams", []))

    def _close_stream(self, name: str) -> None:
        for key in [key for key in self.slots if key[0] == name]:
            self.slots.pop(key).owner.close()
        self.stream_info.pop(name, None)

    def apply_streams(self, streams: list[dict]) -> set[str]:
        """Match the publisher's current streams. Returns the names that were
        opened, reopened after a change, or removed."""
        import cupy as cp

        changed: set[str] = set()
        names = [str(stream["name"]) for stream in streams]
        for name in [name for name in self.stream_order if name not in names]:
            self._close_stream(name)
            changed.add(name)
        self.stream_order = names
        for stream in streams:
            name = str(stream["name"])
            known = self.stream_info.get(name)
            # A publisher bumps the epoch when it registers a name again.
            if known is not None and known.get("epoch") == stream.get("epoch"):
                continue
            self._close_stream(name)
            changed.add(name)
            gpu_id = int(stream.get("gpu_id", 0))
            self.stream_info[name] = dict(stream)
            with cp.cuda.Device(gpu_id):
                for slot in stream["slots"]:
                    handle = base64.b64decode(slot["handle"])
                    ptr = cp.cuda.runtime.ipcOpenMemHandle(
                        handle, cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess
                    )
                    owner = IpcMemory(ptr)
                    mem = cp.cuda.UnownedMemory(ptr, int(slot["nbytes"]), owner)
                    memptr = cp.cuda.MemoryPointer(mem, 0)
                    arr = cp.ndarray(
                        tuple(slot["shape"]),
                        dtype=cp.uint8,
                        memptr=memptr,
                        strides=tuple(slot["strides"]),
                    )
                    self.slots[(name, int(slot["slot"]))] = IpcSlot(
                        stream=name, slot=int(slot["slot"]), array=arr, owner=owner
                    )
        return changed


def _to_bgr_cpu(array, stream_info: dict):
    import cupy as cp
    import cv2

    pixel_format = str(stream_info.get("pixel_format", "rgba8")).lower()
    img = cp.asnumpy(array)
    if pixel_format == "nv12":
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
    if pixel_format == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"unsupported IPC pixel_format {pixel_format!r}")


def _resize_for_display(img, scale: float):
    if scale == 1.0:
        return img
    import cv2

    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _compose_canvas(images: list[object]):
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    import cv2

    max_h = max(int(img.shape[0]) for img in images)
    padded = []
    for img in images:
        h = int(img.shape[0])
        if h < max_h:
            img = cv2.copyMakeBorder(
                img,
                0,
                max_h - h,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        padded.append(img)
    return cv2.hconcat(padded)


def _display_scale(requested: Optional[float], stream_info: dict[str, dict]) -> float:
    if requested is not None:
        return requested
    widest = max(
        (int(info.get("width", 0)) for info in stream_info.values()), default=0
    )
    return 1.0 if widest <= _NATIVE_SCALE_MAX_WIDTH else _LARGE_STREAM_SCALE


def _draw_label(img, text: str):
    import cv2

    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(
            img,
            text,
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return img


def _placeholder_tile(width: int, height: int, lines: list[str]):
    """A tile that cannot be mistaken for live video."""
    import cv2
    import numpy as np

    tile = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (0, 0, 200), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, min(width / 420.0, height / 240.0))
    line_height = int(28 * scale) + 4
    y = (height - line_height * len(lines)) // 2 + line_height - 4
    for index, text in enumerate(lines):
        (text_width, _), _ = cv2.getTextSize(text, font, scale, 1)
        color = (0, 0, 255) if index == 1 else (220, 220, 220)
        cv2.putText(
            tile,
            text,
            (max(2, (width - text_width) // 2), y),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return tile


def _compose_grid(tiles: list[object], columns: int = 0):
    """Rows of up to `columns` tiles; 0 picks a near-square grid."""
    if not tiles:
        return None
    import cv2

    columns = columns if columns > 0 else math.ceil(math.sqrt(len(tiles)))
    rows = [
        _compose_canvas(tiles[start : start + columns])
        for start in range(0, len(tiles), columns)
    ]
    if len(rows) == 1:
        return rows[0]
    width = max(int(row.shape[1]) for row in rows)
    padded = [
        cv2.copyMakeBorder(
            row,
            0,
            0,
            0,
            width - int(row.shape[1]),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        for row in rows
    ]
    return cv2.vconcat(padded)


def _stream_tiles(
    stream_order: list[str],
    stream_info: dict[str, dict],
    latest_cpu: dict[str, object],
    last_frame_at: dict[str, float],
    now: float,
    scale: float,
    stale_after_s: float,
    labels: bool,
) -> list[object]:
    """One tile per stream: the live image, or a placeholder once it goes quiet."""
    if not stream_order:
        return [
            _placeholder_tile(
                *_WAITING_TILE_SIZE,
                [
                    "camera_receiver",
                    "WAITING FOR STREAMS",
                    "no layout from a sender yet",
                ],
            )
        ]
    tiles = []
    for name in stream_order:
        seen_at = last_frame_at.get(name)
        if (
            name in latest_cpu
            and seen_at is not None
            and now - seen_at <= stale_after_s
        ):
            tile = _resize_for_display(latest_cpu[name], scale)
            tiles.append(_draw_label(tile.copy(), name) if labels else tile)
            continue
        info = stream_info.get(name, {})
        width = max(96, round(int(info.get("width", _WAITING_TILE_SIZE[0])) * scale))
        height = max(54, round(int(info.get("height", _WAITING_TILE_SIZE[1])) * scale))
        quiet = (
            "no frames received"
            if seen_at is None
            else f"no frames for {now - seen_at:.0f} s"
        )
        tiles.append(
            _placeholder_tile(
                width, height, [name, "NO SIGNAL", quiet, "check sender log"]
            )
        )
    return tiles


def _format_ns(value: object, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        ns = int(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{ns:+d} ns" if signed else f"{ns} ns"


def _read_wall_clock_ns() -> tuple[int, str]:
    # Use the chrony-disciplined Unix wall clock for cross-host latency.
    return int(time.time_ns()), "realtime"


def _ms_delta(later_ns: object, earlier_ns: object) -> Optional[float]:
    if later_ns is None or earlier_ns is None:
        return None
    try:
        return (int(later_ns) - int(earlier_ns)) / 1.0e6
    except (TypeError, ValueError):
        return None


def _format_ms(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f} ms"
    except (TypeError, ValueError):
        return "n/a"


def _capture_latency_status(metadata: dict) -> str:
    if metadata.get("sender_capture_ready_to_consumer_ms") is not None:
        return _format_ms(metadata.get("sender_capture_ready_to_consumer_ms"))
    source_format = metadata.get("source_format")
    sender_clock = metadata.get("sender_wall_clock")
    consumer_clock = metadata.get("consumer_clock")
    if source_format == "avs1" and metadata.get("sender_capture_ready_ns") is None:
        return "n/a(no_sender_capture_ts)"
    if sender_clock != consumer_clock:
        return f"n/a(clock:{sender_clock}->{consumer_clock})"
    return "n/a"


def _annotate_consumer_receive(msg: dict) -> dict:
    metadata = msg.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    consumer_recv_mono_ns = time.monotonic_ns()
    consumer_recv_ns, consumer_clock = _read_wall_clock_ns()
    metadata["consumer_clock"] = consumer_clock
    metadata["consumer_ipc_msg_recv_ns"] = consumer_recv_ns
    metadata["consumer_ipc_msg_recv_mono_ns"] = consumer_recv_mono_ns

    if metadata.get("sender_wall_clock") == consumer_clock:
        for source_key, output_key in (
            ("sender_capture_ready_ns", "sender_capture_ready_to_consumer_ms"),
            ("sender_process_submit_ns", "sender_process_submit_to_consumer_ms"),
            ("sender_process_done_ns", "sender_process_done_to_consumer_ms"),
            ("sender_tcp_enqueue_ns", "sender_tcp_enqueue_to_consumer_ms"),
            ("sof_wall_clock_ns", "sof_to_consumer_ms"),
        ):
            metadata[output_key] = _ms_delta(consumer_recv_ns, metadata.get(source_key))

    if metadata.get("receiver_clock") == consumer_clock:
        metadata["receiver_rx_to_consumer_ms"] = _ms_delta(
            consumer_recv_ns, metadata.get("receiver_rx_payload_done_ns")
        )
        metadata["receiver_ipc_publish_to_consumer_ms"] = _ms_delta(
            consumer_recv_ns, metadata.get("receiver_ipc_publish_ns")
        )

    if metadata.get("receiver_rx_payload_done_mono_ns") is not None:
        metadata["receiver_rx_to_consumer_mono_ms"] = _ms_delta(
            consumer_recv_mono_ns, metadata.get("receiver_rx_payload_done_mono_ns")
        )
    if metadata.get("receiver_ipc_publish_mono_ns") is not None:
        metadata["receiver_ipc_publish_to_consumer_mono_ms"] = _ms_delta(
            consumer_recv_mono_ns, metadata.get("receiver_ipc_publish_mono_ns")
        )
    msg["metadata"] = metadata
    return msg


def _latest_metadata(
    latest_metadata_by_stream: dict[str, dict], stream_order: list[str]
) -> Optional[dict]:
    for stream in reversed(stream_order):
        metadata = latest_metadata_by_stream.get(stream)
        if metadata:
            return metadata
    return None


def _format_summary_metadata(metadata: Optional[dict]) -> str:
    if not metadata:
        return ""
    fields = []
    stream_name = metadata.get("stream_name")
    if stream_name is not None:
        fields.append(f"stream={stream_name}")
    sender_sequence = metadata.get("sender_sequence")
    if sender_sequence is not None:
        fields.append(f"sender_seq={sender_sequence}")
    if "tsc_delta_ns" in metadata:
        fields.append(
            f"sof_delta={_format_ns(metadata.get('tsc_delta_ns'), signed=True)}"
        )
    if sender_sequence is not None or metadata.get("source_format") == "avs1":
        fields.append(f"capture_to_consumer={_capture_latency_status(metadata)}")
    latency_fields = (
        ("sender_tcp_enqueue_to_consumer_ms", "txq_to_consumer"),
        ("receiver_rx_to_consumer_mono_ms", "rx_to_consumer"),
        ("receiver_ipc_publish_to_consumer_mono_ms", "ipc_to_consumer"),
        ("sof_to_consumer_ms", "sof_to_consumer"),
    )
    for key, label in latency_fields:
        if metadata.get(key) is not None:
            fields.append(f"{label}={_format_ms(metadata.get(key))}")
    return ", ".join(fields)


def _format_stream_metadata(metadata: Optional[dict]) -> str:
    if not metadata:
        return ""
    fields = []
    sender_sequence = metadata.get("sender_sequence")
    if sender_sequence is not None:
        fields.append(f"sender_seq={sender_sequence}")
    role = metadata.get("role")
    if role is not None:
        fields.append(f"role={role}")
    fields.append(f"sof_tsc={_format_ns(metadata.get('sof_tsc_ns'))}")
    if "tsc_delta_ns" in metadata:
        fields.append(
            f"sof_delta={_format_ns(metadata.get('tsc_delta_ns'), signed=True)}"
        )
    if sender_sequence is not None or metadata.get("source_format") == "avs1":
        fields.append(f"capture_to_consumer={_capture_latency_status(metadata)}")
    if metadata.get("sender_tcp_enqueue_to_consumer_ms") is not None:
        fields.append(
            "txq_to_consumer="
            f"{_format_ms(metadata.get('sender_tcp_enqueue_to_consumer_ms'))}"
        )
    if metadata.get("receiver_rx_to_consumer_mono_ms") is not None:
        fields.append(
            "rx_to_consumer="
            f"{_format_ms(metadata.get('receiver_rx_to_consumer_mono_ms'))}"
        )
    return " " + " ".join(fields)


_LATENCY_CSV_FIELDS = [
    "consumer_ipc_msg_recv_ns",
    "consumer_clock",
    "consumer_ipc_msg_recv_mono_ns",
    "stream",
    "slot",
    "slot_generation",
    "ipc_seq",
    "source_format",
    "avs1_version",
    "avs1_flags",
    "session_id",
    "bundle_sequence",
    "stream_index",
    "stream_name",
    "stream_sequence",
    "stream_stereo",
    "stream_codec",
    "stream_rotation",
    "camera_index",
    "role",
    "device_index",
    "sensor_mode_index",
    "frame_number",
    "width",
    "height",
    "source_pixel_format",
    "ipc_pixel_format",
    "sender_wall_clock",
    "sof_tsc_ns",
    "tsc_delta_ns",
    "frame_timestamp_ns",
    "left_sof_tsc_ns",
    "right_sof_tsc_ns",
    "left_frame_timestamp_ns",
    "right_frame_timestamp_ns",
    "sender_capture_ready_ns",
    "sender_process_submit_ns",
    "sender_process_done_ns",
    "sender_tcp_enqueue_ns",
    "receiver_clock",
    "receiver_rx_prefix_done_ns",
    "receiver_rx_payload_done_ns",
    "receiver_process_submit_ns",
    "receiver_process_done_ns",
    "receiver_ipc_publish_ns",
    "receiver_rx_prefix_done_mono_ns",
    "receiver_rx_payload_done_mono_ns",
    "receiver_process_submit_mono_ns",
    "receiver_process_done_mono_ns",
    "receiver_ipc_publish_mono_ns",
    "rx_to_process_submit_ms",
    "rx_to_process_done_ms",
    "rx_to_ipc_publish_ms",
    "sender_capture_ready_to_rx_ms",
    "sender_process_submit_to_rx_ms",
    "sender_process_done_to_rx_ms",
    "sender_tcp_enqueue_to_rx_ms",
    "sender_tcp_enqueue_to_ipc_publish_ms",
    "sender_capture_ready_to_ipc_publish_ms",
    "sender_capture_ready_to_consumer_ms",
    "sender_process_submit_to_consumer_ms",
    "sender_process_done_to_consumer_ms",
    "sender_tcp_enqueue_to_consumer_ms",
    "receiver_rx_to_consumer_ms",
    "receiver_ipc_publish_to_consumer_ms",
    "receiver_rx_to_consumer_mono_ms",
    "receiver_ipc_publish_to_consumer_mono_ms",
    "sof_to_consumer_ms",
]


def _csv_row(msg: dict) -> dict[str, object]:
    metadata = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
    row = {field: metadata.get(field, "") for field in _LATENCY_CSV_FIELDS}
    row["stream"] = msg.get("stream", "")
    row["slot"] = msg.get("slot", "")
    row["slot_generation"] = msg.get("slot_generation", "")
    row["ipc_seq"] = msg.get("seq", "")
    return row


def _parse_stream_filter(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def run(args: argparse.Namespace) -> int:
    import cupy as cp

    requested_streams = _parse_stream_filter(args.streams)
    client = CudaIpcClient(args.socket, requested_streams)
    stop = False

    def on_sigint(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_sigint)

    handshake = client.connect()
    subscribed = handshake["subscribed"]

    def describe(streams: list[dict]) -> str:
        return (
            ", ".join(
                f"{s['name']} {s['width']}x{s['height']} "
                f"{s.get('pixel_format', 'unknown')} slots={len(s['slots'])}"
                for s in streams
            )
            or "no streams yet"
        )

    print(
        f"connected to {args.socket}: {describe(subscribed.get('streams', []))}",
        flush=True,
    )

    cv2 = None
    if not args.no_display:
        try:
            import cv2 as _cv2

            cv2 = _cv2
        except ImportError:
            print(
                "OpenCV not available; falling back to --no-display stats", flush=True
            )

    latest_cpu: dict[str, object] = {}
    latest_metadata_by_stream: dict[str, dict] = {}
    latest_generation_by_slot: dict[tuple[str, int], int] = {}
    last_frame_at: dict[str, float] = {}
    last_draw = 0.0
    counts: dict[str, int] = dict.fromkeys(client.stream_order, 0)
    stats_interval = max(1, args.stats_every)
    last_report = time.monotonic()
    last_counts = dict(counts)
    frames_seen = 0
    overwritten = 0
    csv_file = None
    csv_writer = None
    if args.latency_csv:
        csv_file = open(args.latency_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=_LATENCY_CSV_FIELDS)
        csv_writer.writeheader()

    try:
        while not stop:
            messages = client.read_messages()
            if messages is None:
                print("publisher disconnected", flush=True)
                break
            latest_msg_by_stream: dict[str, dict] = {}
            sampled_messages: list[dict] = []
            for msg in messages:
                if msg.get("type") == "streams":
                    # The receiver's layout changed; forget what the old rings held.
                    for name in client.apply_streams(msg.get("streams", [])):
                        latest_cpu.pop(name, None)
                        latest_msg_by_stream.pop(name, None)
                        latest_metadata_by_stream.pop(name, None)
                        last_frame_at.pop(name, None)
                        for key in [
                            k for k in latest_generation_by_slot if k[0] == name
                        ]:
                            del latest_generation_by_slot[key]
                    for name in client.stream_order:
                        counts.setdefault(name, 0)
                    print(f"streams: {describe(msg.get('streams', []))}", flush=True)
                    continue
                if msg.get("type") != "frame":
                    continue
                msg = _annotate_consumer_receive(msg)
                stream = str(msg["stream"])
                last_frame_at[stream] = time.monotonic()
                slot_id = int(msg["slot"])
                generation = int(msg.get("slot_generation", 0))
                key = (stream, slot_id)
                previous = latest_generation_by_slot.get(key)
                if previous is not None and generation > previous + 1:
                    overwritten += generation - previous - 1
                latest_generation_by_slot[key] = generation
                counts[stream] = counts.get(stream, 0) + 1
                frames_seen += 1
                latest_msg_by_stream[stream] = msg
                if cv2 is None and counts[stream] % stats_interval == 0:
                    sampled_messages.append(msg)
                metadata = msg.get("metadata")
                if isinstance(metadata, dict):
                    latest_metadata_by_stream[stream] = metadata
                if csv_writer is not None:
                    csv_writer.writerow(_csv_row(msg))

            updated_display = False
            for stream in client.stream_order:
                msg = latest_msg_by_stream.get(stream)
                if msg is None:
                    continue
                slot_id = int(msg["slot"])
                slot = client.slots.get((stream, slot_id))
                if slot is None:
                    print(f"unknown frame slot: {stream}[{slot_id}]", flush=True)
                    continue
                updated_display = True
                if cv2 is not None:
                    latest_cpu[stream] = _to_bgr_cpu(
                        slot.array, client.stream_info.get(stream, {})
                    )

            if cv2 is None:
                for msg in sampled_messages:
                    stream = str(msg["stream"])
                    slot_id = int(msg["slot"])
                    slot = client.slots.get((stream, slot_id))
                    if slot is None:
                        print(f"unknown frame slot: {stream}[{slot_id}]", flush=True)
                        continue
                    metadata_text = _format_stream_metadata(msg.get("metadata"))
                    mean_text = ""
                    if args.sample_mean:
                        mean = float(cp.mean(slot.array).get())
                        mean_text = f" mean={mean:.2f}"
                    print(
                        f"{stream}[{slot_id}] gen={msg.get('slot_generation')} "
                        f"seq={msg.get('seq')}{mean_text}{metadata_text}",
                        flush=True,
                    )

            if cv2 is not None:
                # Also redraw on a timer so a stream that stops turns into a placeholder.
                draw_now = time.monotonic()
                if updated_display or draw_now - last_draw >= _REDRAW_INTERVAL_S:
                    canvas = _compose_grid(
                        _stream_tiles(
                            client.stream_order,
                            client.stream_info,
                            latest_cpu,
                            last_frame_at,
                            draw_now,
                            _display_scale(args.scale, client.stream_info),
                            args.stale_after,
                            not args.no_labels,
                        ),
                        args.columns,
                    )
                    if canvas is not None:
                        cv2.imshow(args.window_name, canvas)
                    last_draw = draw_now
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            now = time.monotonic()
            if now - last_report >= 1.0:
                rates = []
                for name in client.stream_order:
                    current = counts.get(name, 0)
                    prev = last_counts.get(name, 0)
                    rates.append(f"{name}: {current - prev} fps")
                    last_counts[name] = current
                if overwritten:
                    rates.append(f"overwritten_slots={overwritten}")
                    overwritten = 0
                metadata_text = _format_summary_metadata(
                    _latest_metadata(latest_metadata_by_stream, client.stream_order)
                )
                if metadata_text:
                    rates.append(metadata_text)
                print(", ".join(rates), flush=True)
                last_report = now
            if args.frames > 0 and frames_seen >= args.frames:
                break
    finally:
        if csv_file is not None:
            csv_file.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
        client.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--socket",
        default="/tmp/camera_receiver_avs1_cuda_ipc.sock",
        help="Unix socket path exposed by camera_receiver ipc.socket_path",
    )
    ap.add_argument(
        "--streams",
        help="comma-separated IPC stream names to subscribe to; default subscribes to all",
    )
    ap.add_argument(
        "--scale",
        type=float,
        help=(
            "display scale; default 1 when every stream is at most "
            f"{_NATIVE_SCALE_MAX_WIDTH} wide, otherwise {_LARGE_STREAM_SCALE}"
        ),
    )
    ap.add_argument(
        "--columns",
        type=int,
        default=0,
        help="tiles per row; 0 picks a near-square grid",
    )
    ap.add_argument(
        "--stale-after",
        type=float,
        default=1.0,
        help="seconds without a frame before a stream shows NO SIGNAL",
    )
    ap.add_argument(
        "--no-labels", action="store_true", help="do not draw stream names on tiles"
    )
    ap.add_argument("--window-name", default="camera_receiver CUDA IPC")
    ap.add_argument("--no-display", action="store_true", help="print stats only")
    ap.add_argument(
        "--stats-every",
        type=int,
        default=60,
        help="print one detailed sample per stream every N frames (default: 60)",
    )
    ap.add_argument(
        "--sample-mean",
        action="store_true",
        help="compute a GPU mean for occasional no-display frame samples",
    )
    ap.add_argument("--frames", type=int, default=0, help="exit after this many frames")
    ap.add_argument(
        "--latency-csv",
        help="write one CSV row per received IPC frame notification",
    )
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
