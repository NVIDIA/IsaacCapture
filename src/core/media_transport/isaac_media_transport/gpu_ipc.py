# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Same-user Linux CUDA IPC handoff with acknowledged, bounded GPU leases.

Each response holds one complete mono/stereo sample. The importer copies on GPU,
closes every mapping, then acknowledges. Unacknowledged slots are quarantined.
This protocol is independent of the Sensing receiver's unacknowledged v2 ring.
"""

import base64
from dataclasses import asdict
import json
import logging
import os
import select
from pathlib import Path
import socket
import struct
import threading
import time

from .transport import FrameMetadata, ReceivedFrame, ReceivedPair, VideoFrame
from .latency import get_recorder

log = logging.getLogger("isaaccapture.remote_assistance.gpu_ipc")
PROTOCOL = "ra-cuda-1"


def send(sock, message):
    data = json.dumps(message).encode()
    if len(data) > 16384:
        raise ValueError("IPC descriptor exceeds limit")
    if sock.send(data) != len(data):
        raise OSError("Incomplete IPC descriptor")


def receive(sock):
    data, _, flags, _ = sock.recvmsg(16384)
    if not data:
        raise EOFError("IPC peer disconnected")
    if flags & socket.MSG_TRUNC:
        raise ValueError("Truncated IPC descriptor")
    return json.loads(data)


def socket_path(settings):
    return Path(
        settings.get("decoded_socket", "~/remote-assistance-gpu.sock")
    ).expanduser()


def tracks_of(track):
    tracks = [track] if isinstance(track, str) else list(track)
    if len(tracks) not in (1, 2) or len(set(tracks)) != len(tracks):
        raise ValueError("IPC requires one track or two distinct stereo tracks")
    return tracks


class IpcServer:
    def __init__(self, path, receiver, tracks, gpu=0):
        import cupy as cp

        self.cp, self.receiver = cp, receiver
        self.timing = get_recorder(getattr(receiver, "settings", {}))
        self.path, self.tracks, self.gpu = (
            Path(path).expanduser(),
            tracks_of(tracks),
            gpu,
        )
        self.quarantine = []
        self.slots = None
        self.sent = 0
        self.ack_ms = 0.0

    def _export(self, sample, seq):
        begin_ns = time.monotonic_ns() if self.timing.enabled else None
        cp = self.cp
        frames = (
            [sample.left, sample.right]
            if isinstance(sample, ReceivedPair)
            else [sample]
        )
        if len(frames) != len(self.tracks):
            raise ValueError("IPC stereo selection differs from receiver")
        shapes = [(f.frame.height, f.frame.width, 4) for f in frames]
        if self.slots is None:
            # Dedicated cudaMalloc allocations: never export a suballocated CuPy pool.
            self.slots = [
                cp.ndarray(
                    shape,
                    cp.uint8,
                    memptr=cp.cuda.MemoryPointer(
                        cp.cuda.Memory(shape[0] * shape[1] * 4), 0
                    ),
                )
                for shape in shapes
            ]
        if [a.shape for a in self.slots] != shapes:
            raise ValueError("IPC dimensions changed; restart receiver and viewer")
        entries = []
        for track, source, dest in zip(self.tracks, frames, self.slots):
            frame = source.frame
            if frame.memory != "cuda" or frame.device_id != self.gpu:
                raise ValueError("IPC requires CUDA frames on the configured GPU")
            cp.copyto(dest, frame.data)
            entries.append(
                dict(
                    track=track,
                    width=frame.width,
                    height=frame.height,
                    handle=base64.b64encode(
                        cp.cuda.runtime.ipcGetMemHandle(dest.data.ptr)
                    ).decode(),
                    received_ns=source.received_ns,
                    metadata=asdict(source.metadata) if source.metadata else None,
                )
            )
        cp.cuda.get_current_stream().synchronize()
        if self.timing.enabled:
            ready_ns = time.monotonic_ns()
            for track, source in zip(self.tracks, frames):
                if source.metadata:
                    stamp = source.metadata.user_timestamp
                    self.timing.record("ipc_export_begin", track, stamp, at_ns=begin_ns)
                    self.timing.record("ipc_export_ready", track, stamp, at_ns=ready_ns)
        return dict(
            protocol=PROTOCOL,
            seq=seq,
            gpu=self.gpu,
            frames=entries,
            paired_ns=sample.paired_ns if isinstance(sample, ReceivedPair) else None,
        )

    def run(self, stop):
        if not hasattr(socket, "SOCK_SEQPACKET") or os.name != "posix":
            raise RuntimeError("CUDA IPC handoff currently requires Linux")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        bound = False
        try:
            listener.bind(
                str(self.path)
            )  # Never unlink another running receiver's socket.
            bound = True
            self.path.chmod(0o600)
            listener.listen(1)
            listener.settimeout(0.2)
            log.info(
                "CUDA IPC receiver listening at %s; GPU=%d tracks=%s",
                self.path,
                self.gpu,
                self.tracks,
            )
            with self.cp.cuda.Device(self.gpu):
                while not stop.is_set():
                    try:
                        conn, _ = listener.accept()
                    except TimeoutError:
                        # Surface upstream errors even when no viewer is attached.
                        self.receiver.latest()
                        continue
                    with conn:
                        conn.settimeout(5)
                        _, uid, _ = struct.unpack(
                            "3i",
                            conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                        )
                        if uid != os.getuid():
                            continue
                        outstanding = False
                        try:
                            hello = receive(conn)
                            if hello != dict(
                                protocol=PROTOCOL, tracks=self.tracks, gpu=self.gpu
                            ):
                                raise ValueError(
                                    "IPC subscription tracks/GPU/protocol mismatch"
                                )
                            while not stop.is_set():
                                if select.select([conn], [], [], 0)[0]:
                                    if receive(conn) != {"bye": True}:
                                        raise ValueError(
                                            "Unexpected IPC client message"
                                        )
                                    break
                                sample = self.receiver.latest()
                                if sample is None:
                                    stop.wait(0.002)
                                    continue
                                self.sent += 1
                                message = self._export(sample, self.sent)
                                outstanding = True
                                start = time.monotonic()
                                send(conn, message)
                                if receive(conn) != {"ack": self.sent}:
                                    raise ValueError(
                                        "Invalid CUDA lease acknowledgment"
                                    )
                                outstanding = False
                                self.ack_ms = (time.monotonic() - start) * 1000
                                if self.sent % 150 == 0 or self.ack_ms > 30:
                                    log.info(
                                        "CUDA IPC frames=%d copy/ack=%.2f ms receiver_overwritten=%d quarantined=%d",
                                        self.sent,
                                        self.ack_ms,
                                        self.receiver.overwritten,
                                        len(self.quarantine),
                                    )
                        except (OSError, EOFError, ValueError) as exc:
                            log.warning("CUDA IPC client ended: %s", exc)
                        finally:
                            if outstanding:
                                # A disconnected client may still have a mapped GPU allocation.
                                self.quarantine.append(self.slots)
                                self.slots = None
                                log.warning(
                                    "CUDA IPC unacknowledged buffers quarantined (%d/4); restart both processes to reclaim",
                                    len(self.quarantine),
                                )
                                if len(self.quarantine) >= 4:
                                    raise RuntimeError(
                                        "IPC unacknowledged lease limit reached; stop viewer then restart receiver"
                                    )
        finally:
            listener.close()
            if bound:
                self.path.unlink(missing_ok=True)


class IpcReceiver:
    overwritten = 0

    def __init__(self, settings, *, identity, publisher, track, pairing=None):
        self.timing = get_recorder(settings)
        self.path = socket_path(settings)
        self.tracks = tracks_of(track)
        self.gpu = int(settings.get("decode_gpu", 0))
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.thread = None
        self.sock = None

    def start(self):
        import cupy as cp

        self.cp = cp
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _copy(self, msg):
        begin_ns = time.monotonic_ns() if self.timing.enabled else None
        cp = self.cp
        if msg.get("protocol") != PROTOCOL or msg.get("gpu") != self.gpu:
            raise ValueError("IPC protocol/GPU mismatch")
        entries = msg["frames"]
        if [e["track"] for e in entries] != self.tracks:
            raise ValueError("IPC eye order mismatch")
        frames = []
        for entry in entries:
            w, h = int(entry["width"]), int(entry["height"])
            if min(w, h) <= 0 or max(w, h) > 8192:
                raise ValueError("Invalid CUDA IPC dimensions")
            handle = base64.b64decode(entry["handle"], validate=True)
            if len(handle) != 64:
                raise ValueError("Invalid CUDA IPC handle")
            pointer = cp.cuda.runtime.ipcOpenMemHandle(handle)
            try:
                owner = cp.cuda.UnownedMemory(
                    pointer, w * h * 4, self, device_id=self.gpu
                )
                mapped = cp.ndarray(
                    (h, w, 4), cp.uint8, cp.cuda.MemoryPointer(owner, 0)
                )
                image = mapped.copy()
                cp.cuda.get_current_stream().synchronize()
                del mapped, owner
            finally:
                cp.cuda.runtime.ipcCloseMemHandle(pointer)
            frames.append(
                ReceivedFrame(
                    VideoFrame(w, h, image, "cuda", self.gpu),
                    int(entry["received_ns"]),
                    FrameMetadata(**entry["metadata"]) if entry["metadata"] else None,
                )
            )
        if self.timing.enabled:
            ready_ns = time.monotonic_ns()
            for track, source in zip(self.tracks, frames):
                if source.metadata:
                    stamp = source.metadata.user_timestamp
                    self.timing.record("ipc_import_begin", track, stamp, at_ns=begin_ns)
                    self.timing.record("ipc_copy_ready", track, stamp, at_ns=ready_ns)
        return (
            ReceivedPair(*frames, int(msg["paired_ns"]))
            if len(frames) == 2
            else frames[0]
        )

    def _run(self):
        try:
            with self.cp.cuda.Device(self.gpu):
                while not self.stop_event.is_set():
                    try:
                        with socket.socket(
                            socket.AF_UNIX, socket.SOCK_SEQPACKET
                        ) as sock:
                            self.sock = sock
                            sock.settimeout(0.5)
                            sock.connect(str(self.path))
                            send(
                                sock,
                                dict(
                                    protocol=PROTOCOL, tracks=self.tracks, gpu=self.gpu
                                ),
                            )
                            while not self.stop_event.is_set():
                                try:
                                    msg = receive(sock)
                                except TimeoutError:
                                    continue
                                frame = self._copy(msg)
                                send(sock, {"ack": msg["seq"]})
                                with self.lock:
                                    self.overwritten += self.frame is not None
                                    self.frame = frame
                            send(sock, {"bye": True})
                    except (ConnectionError, FileNotFoundError, EOFError):
                        self.stop_event.wait(0.25)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
                log.exception("CUDA IPC receive failed")
        finally:
            self.sock = None

    def latest(self):
        if self.error:
            raise RuntimeError("CUDA IPC receiver failed") from self.error
        with self.lock:
            result, self.frame = self.frame, None
            return result

    def pairing_stats(self):
        return {}  # Pairing and its hold/drop reporting happen in the exporter.

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(6)
            if self.thread.is_alive():
                raise TimeoutError("CUDA IPC importer failed to stop")
