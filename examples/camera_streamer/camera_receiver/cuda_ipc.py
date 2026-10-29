# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC frame publisher for camera_receiver.

Protocol v3 uses a newline-delimited JSON control socket:

1. Publisher -> client: ``hello`` stream catalog, no CUDA handles yet.
2. Client -> publisher: ``subscribe`` with stream names, or omitted/empty for all.
3. Publisher -> client: ``subscribed`` with CUDA IPC handles for selected streams.
4. Publisher -> client: ``frame`` notifications for selected streams only.
5. Client locks the slot's guard file, checks its generation, and completes GPU
   reads before unlocking. Stale notifications are discarded.

Image bytes stay in same-host CUDA IPC memory. The socket only carries control
messages and per-frame metadata.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import signal
import socket
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from slot_guard import commit_slot, read_slot, write_slot

logger = logging.getLogger("isaaccapture.camera_streamer." + __name__)

_CLIENT_SEND_TIMEOUT_S = 2.0
_CLIENT_SUBSCRIBE_TIMEOUT_S = 5.0
_MAX_CONTROL_BYTES = 1024 * 1024
_SEND_FLAGS = getattr(socket, "MSG_NOSIGNAL", 0)


@dataclass(frozen=True)
class _SlotInfo:
    slot: int
    handle_b64: str
    nbytes: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    guard_path: str


@dataclass(frozen=True)
class _StreamInfo:
    name: str
    width: int
    height: int
    pixel_format: str
    dtype: str
    gpu_id: int
    layout: dict[str, Any]
    slots: tuple[_SlotInfo, ...]


@dataclass
class _ClientState:
    sock: socket.socket
    streams: frozenset[str]


class CudaIpcFramePublisher:
    """Publish decoded GPU frames to same-host CUDA IPC consumers."""

    def __init__(
        self,
        socket_path: str = "/tmp/camera_receiver_cuda_ipc.sock",
        owner_name: str = "camera_receiver",
        queue_size: int = 512,
    ) -> None:
        self._socket_path = str(socket_path)
        self._owner_name = owner_name
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=queue_size)
        self._streams: dict[str, _StreamInfo] = {}
        self._clients: list[_ClientState] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._broadcast_thread: Optional[threading.Thread] = None
        self._started = False
        self._guard_dir = tempfile.TemporaryDirectory(prefix="camera-ipc-slots-")

    @property
    def socket_path(self) -> str:
        return self._socket_path

    def register_stream(
        self,
        name: str,
        width: int,
        height: int,
        pixel_format: str,
        gpu_id: int,
        buffers: list[Any],
        layout: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a ring of CuPy-like GPU buffers and export IPC handles."""
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError("CudaIpcFramePublisher requires CuPy") from e

        slots: list[_SlotInfo] = []
        with cp.cuda.Device(gpu_id):
            for idx, buf in enumerate(buffers):
                iface = getattr(buf, "__cuda_array_interface__", None)
                if iface is None:
                    raise TypeError(
                        f"CUDA IPC stream {name!r} slot {idx} does not expose "
                        "__cuda_array_interface__"
                    )
                ptr = int(iface["data"][0])
                handle = cp.cuda.runtime.ipcGetMemHandle(ptr)
                strides = iface.get("strides") or getattr(buf, "strides", None)
                if strides is None:
                    raise TypeError(
                        f"CUDA IPC stream {name!r} slot {idx} has no strides"
                    )
                guard_path = str(
                    Path(self._guard_dir.name) / f"{len(self._streams)}-{idx}"
                )
                Path(guard_path).write_bytes(bytes(8))
                slots.append(
                    _SlotInfo(
                        slot=idx,
                        guard_path=guard_path,
                        handle_b64=base64.b64encode(handle).decode("ascii"),
                        nbytes=int(buf.nbytes),
                        shape=tuple(int(v) for v in iface["shape"]),
                        strides=tuple(int(v) for v in strides),
                    )
                )

        with self._lock:
            if name in self._streams:
                raise ValueError(f"CUDA IPC stream {name!r} registered twice")
            self._streams[name] = _StreamInfo(
                name=name,
                width=int(width),
                height=int(height),
                pixel_format=str(pixel_format),
                dtype="uint8",
                gpu_id=int(gpu_id),
                layout=dict(layout or {}),
                slots=tuple(slots),
            )
        logger.info(
            "CUDA IPC: registered stream '%s' %s with %d slots on GPU %d",
            name,
            pixel_format,
            len(slots),
            gpu_id,
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
        self._ignore_sigpipe()
        self._stop.clear()
        try:
            self._open_server()
        except Exception:
            with self._lock:
                self._started = False
            raise
        with self._lock:
            self._started = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="camera_receiver_cuda_ipc_accept",
            daemon=False,
        )
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            name="camera_receiver_cuda_ipc_broadcast",
            daemon=False,
        )
        self._accept_thread.start()
        self._broadcast_thread.start()
        logger.info("CUDA IPC: listening on %s", self._socket_path)

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                logger.debug("CUDA IPC: socket close failed", exc_info=True)
            self._server = None
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
            self._started = False
        for client in clients:
            try:
                client.sock.close()
            except OSError:
                logger.debug("CUDA IPC: socket close failed", exc_info=True)
        for thread in (self._accept_thread, self._broadcast_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self._accept_thread = None
        self._broadcast_thread = None
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        except OSError:
            logger.debug(
                "CUDA IPC: failed to unlink %s", self._socket_path, exc_info=True
            )

    def write_slot(self, stream_name: str, slot: int, stop: threading.Event):
        return write_slot(self._streams[stream_name].slots[slot].guard_path, stop)

    def _current_frame(self, msg):
        return read_slot(
            self._streams[msg["stream"]].slots[msg["slot"]].guard_path,
            msg["slot_generation"],
        )

    def publish(
        self,
        stream_name: str,
        slot: int,
        slot_generation: int,
        seq: int,
        timestamp_ns: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self._started:
            return
        commit_slot(self._streams[stream_name].slots[slot].guard_path, slot_generation)
        msg = {
            "type": "frame",
            "stream": stream_name,
            "slot": int(slot),
            "slot_generation": int(slot_generation),
            "seq": int(seq),
            "timestamp_ns": int(timestamp_ns),
        }
        if metadata is not None:
            msg["metadata"] = metadata
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                # The broadcaster already drained the notification queue.
                pass
            try:
                self._queue.put_nowait(msg)
            except queue.Full:
                logger.debug(
                    "CUDA IPC: dropping frame notification for %s", stream_name
                )

    def _open_server(self) -> None:
        path = Path(self._socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.1)
                probe.connect(str(path))
                raise RuntimeError(
                    f"CUDA IPC socket {self._socket_path} is already in use"
                )
            except OSError:
                path.unlink()
            finally:
                probe.close()

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self._socket_path)
        os.chmod(self._socket_path, 0o600)
        srv.listen(8)
        srv.settimeout(0.5)
        self._server = srv

    def _stream_catalog(self, stream: _StreamInfo) -> dict[str, Any]:
        return {
            "name": stream.name,
            "width": stream.width,
            "height": stream.height,
            "pixel_format": stream.pixel_format,
            "dtype": stream.dtype,
            "gpu_id": stream.gpu_id,
            "slot_count": len(stream.slots),
            "layout": stream.layout,
        }

    def _stream_with_slots(self, stream: _StreamInfo) -> dict[str, Any]:
        msg = self._stream_catalog(stream)
        msg["slots"] = [
            {
                "slot": slot.slot,
                "guard_path": slot.guard_path,
                "handle": slot.handle_b64,
                "nbytes": slot.nbytes,
                "shape": list(slot.shape),
                "strides": list(slot.strides),
            }
            for slot in stream.slots
        ]
        return msg

    def _hello_message(self) -> dict[str, Any]:
        with self._lock:
            streams = list(self._streams.values())
        return {
            "type": "hello",
            "version": 3,
            "features": [
                "subscriptions",
                "slot_read_locks",
                "frame_metadata",
                "avs1_metadata",
                "latency_metadata",
                "slot_generation",
                "nv12",
            ],
            "owner": self._owner_name,
            "socket_path": self._socket_path,
            "streams": [self._stream_catalog(s) for s in streams],
        }

    def _subscribed_message(
        self, requested: list[str]
    ) -> tuple[dict[str, Any], frozenset[str]]:
        with self._lock:
            all_streams = dict(self._streams)
        selected_names = requested or list(all_streams)
        unknown = [name for name in selected_names if name not in all_streams]
        if unknown:
            raise ValueError(f"unknown CUDA IPC stream(s): {', '.join(unknown)}")
        selected = frozenset(selected_names)
        return (
            {
                "type": "subscribed",
                "version": 3,
                "streams": [
                    self._stream_with_slots(all_streams[name])
                    for name in selected_names
                ],
            },
            selected,
        )

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                client.settimeout(_CLIENT_SEND_TIMEOUT_S)
                self._send_json(client, self._hello_message())
                client.settimeout(_CLIENT_SUBSCRIBE_TIMEOUT_S)
                subscribe = self._recv_json(client)
                if subscribe.get("type") != "subscribe":
                    raise ValueError(f"expected subscribe message, got {subscribe!r}")
                if subscribe.get("version") != 3:
                    raise ValueError(
                        "CUDA IPC requires protocol 3 slot read locks; update the consumer"
                    )
                requested_raw = subscribe.get("streams") or []
                if not isinstance(requested_raw, list):
                    raise ValueError("subscribe.streams must be a list")
                requested = [str(name) for name in requested_raw]
                msg, selected = self._subscribed_message(requested)
                client.settimeout(_CLIENT_SEND_TIMEOUT_S)
                self._send_json(client, msg)
            except Exception as exc:
                logger.info("CUDA IPC: rejecting consumer (%s)", exc)
                try:
                    self._send_json(client, {"type": "error", "error": str(exc)})
                except OSError:
                    logger.debug(
                        "CUDA IPC: cannot notify rejected client", exc_info=True
                    )
                client.close()
                continue
            with self._lock:
                self._clients.append(_ClientState(client, selected))
            logger.info(
                "CUDA IPC: consumer connected (%s)",
                ", ".join(sorted(selected)) if selected else "no streams",
            )

    def _broadcast_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._current_frame(msg) as current:
                if not current:
                    continue
            stream_name = str(msg.get("stream", ""))
            with self._lock:
                clients = list(self._clients)
            if not clients:
                continue
            dead: list[_ClientState] = []
            for client in clients:
                if stream_name not in client.streams:
                    continue
                try:
                    self._send_json(client.sock, msg)
                except socket.timeout:
                    logger.warning(
                        "CUDA IPC: dropping slow consumer after send timeout"
                    )
                    dead.append(client)
                except OSError as e:
                    logger.info("CUDA IPC: consumer disconnected (%s)", e)
                    dead.append(client)
            if dead:
                with self._lock:
                    self._clients = [c for c in self._clients if c not in dead]
                for client in dead:
                    try:
                        client.sock.close()
                    except OSError:
                        logger.debug("CUDA IPC: client close failed", exc_info=True)

    @staticmethod
    def _send_json(sock: socket.socket, msg: dict[str, Any]) -> None:
        payload = json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n"
        sock.sendall(payload, _SEND_FLAGS)

    @staticmethod
    def _recv_json(sock: socket.socket) -> dict[str, Any]:
        data = bytearray()
        while b"\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                raise EOFError("consumer disconnected before subscribe")
            data.extend(chunk)
            if len(data) > _MAX_CONTROL_BYTES:
                raise ValueError("control message too large")
        line = bytes(data[: data.index(ord("\n"))])
        msg = json.loads(line.decode("utf-8"))
        if not isinstance(msg, dict):
            raise ValueError("control message must be a JSON object")
        return msg

    @staticmethod
    def _ignore_sigpipe() -> None:
        sigpipe = getattr(signal, "SIGPIPE", None)
        if sigpipe is None:
            return
        try:
            signal.signal(sigpipe, signal.SIG_IGN)
        except (ValueError, OSError):
            # Worker threads cannot change signal handlers; sends use MSG_NOSIGNAL.
            pass
