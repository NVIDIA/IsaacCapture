#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example CUDA IPC consumer for camera_receiver AVS1 streams."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import select
import signal
import socket
import time
from dataclasses import dataclass
from typing import Optional


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
            # CUDA/Python modules may already be torn down during finalization.
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
        self.sock = sock
        try:
            sock.connect(self.socket_path)
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
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                # A socket close failure must not prevent releasing CUDA mappings.
                pass
            self.sock = None
        for slot in self.slots.values():
            slot.owner.close()
        self.slots.clear()
        self.recv_buf.clear()
        self.stream_order.clear()
        self.stream_info.clear()

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
        import cupy as cp

        for stream in subscribed.get("streams", []):
            name = str(stream["name"])
            gpu_id = int(stream.get("gpu_id", 0))
            if name not in self.stream_order:
                self.stream_order.append(name)
            self.stream_info[name] = dict(stream)
            with cp.cuda.Device(gpu_id):
                for slot in stream["slots"]:
                    handle = base64.b64decode(slot["handle"])
                    ptr = cp.cuda.runtime.ipcOpenMemHandle(
                        handle, cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess
                    )
                    owner = IpcMemory(ptr)
                    try:
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
                    except Exception:
                        # A mapping is owned before its array can be registered in slots.
                        owner.close()
                        raise


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
    print(
        f"connected to {args.socket}: "
        + ", ".join(
            f"{s['name']} {s['width']}x{s['height']} "
            f"{s.get('pixel_format', 'unknown')} slots={len(s['slots'])}"
            for s in subscribed.get("streams", [])
        ),
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
        if cv2 is not None:
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
            print(
                "Close a preview window, press 'q' or Esc in the preview, or Ctrl+C in this terminal to quit.",
                flush=True,
            )
        while not stop:
            if cv2 is not None:
                # Process window-close events before displaying another frame.
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                try:
                    if (
                        cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE)
                        < 1
                    ):
                        break
                except cv2.error:
                    break
            messages = client.read_messages()
            if messages is None:
                print("publisher disconnected", flush=True)
                break
            latest_msg_by_stream: dict[str, dict] = {}
            sampled_messages: list[dict] = []
            for msg in messages:
                if msg.get("type") != "frame":
                    continue
                msg = _annotate_consumer_receive(msg)
                stream = str(msg["stream"])
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
                if updated_display:
                    images = [
                        _resize_for_display(latest_cpu[name], args.scale)
                        for name in client.stream_order
                        if name in latest_cpu
                    ]
                    if images:
                        canvas = _compose_canvas(images)
                        if canvas is not None:
                            cv2.imshow(args.window_name, canvas)

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
        default=1.0,
        help="OpenCV display scale (default: 1.0, full resolution)",
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
