# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded same-user metadata access; no camera pixels or network listener."""

from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import uuid


def endpoint(run_id, role, runtime_dir=None):
    root = Path(runtime_dir or f"/tmp/isaac-latency-{os.getuid()}").expanduser()
    digest = hashlib.sha256(f"{run_id}:{role}".encode()).hexdigest()[:24]
    return root / (digest + ".sock")


class EventBuffer:
    def __init__(self, path, capacity=8192):
        self.path = Path(path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat = self.path.parent.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise ValueError(
                "Latency runtime directory must be private to the current user"
            )
        self._owner = open(str(self.path) + ".lock", "a")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise ValueError(
                "Another process owns this latency run/role endpoint"
            ) from None
        self._events = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0
        self._instance = uuid.uuid4().hex
        self._stop = threading.Event()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.path.unlink(missing_ok=True)
            self._socket.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self._socket.listen(4)
            self._socket.settimeout(0.2)
        except BaseException:
            self._socket.close()
            self._owner.close()
            raise
        self._thread = threading.Thread(
            target=self._serve, daemon=True, name="latency-events"
        )
        self._thread.start()

    def append(self, event):
        with self._lock:
            self._seq += 1
            self._events.append((self._seq, event))

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                try:
                    conn.settimeout(0.5)
                    raw = bytearray()
                    while not raw.endswith(b"\n") and len(raw) < 1024:
                        chunk = conn.recv(1024 - len(raw))
                        if not chunk:
                            break
                        raw.extend(chunk)
                    request = json.loads(raw)
                    offset = int(request.get("offset", 0))
                    with self._lock:
                        selected = [(n, e) for n, e in self._events if n > offset][:256]
                        last = next(
                            (
                                e
                                for _, e in reversed(self._events)
                                if e.get("record_type") == "stage"
                            ),
                            {},
                        )
                        answer = dict(
                            instance=self._instance,
                            sequence=self._seq,
                            gap=bool(
                                self._events
                                and offset
                                and offset < self._events[0][0] - 1
                            ),
                            next=selected[-1][0] if selected else offset,
                            events=[e for _, e in selected],
                            last=last,
                        )
                    conn.sendall(json.dumps(answer).encode() + b"\n")
                except (OSError, ValueError, TypeError):
                    continue

    def close(self):
        self._stop.set()
        self._socket.close()
        self._thread.join(1)
        self.path.unlink(missing_ok=True)
        self._owner.close()


class SocketTail:
    """Cursor API shared with the file monitor; suitable for the SSH stdlib helper."""

    def __init__(self, path):
        self.path = str(Path(path).expanduser())
        self.offset = 0
        self.instance = None

    def _request(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(0.5)
            conn.connect(self.path)
            conn.sendall(json.dumps({"offset": self.offset}).encode() + b"\n")
            result = bytearray()
            while not result.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    raise ConnectionError("Latency endpoint closed")
                result.extend(chunk)
                if len(result) > 4 * 1024 * 1024:
                    raise ValueError("Latency response exceeds limit")
        value = json.loads(result)
        if self.instance is not None and value["instance"] != self.instance:
            raise ValueError("Latency process restarted; discarding old frame joins")
        self.instance = value["instance"]
        return value

    def follow(self):
        self.offset = self._request()["sequence"]
        return self.offset

    def read(self):
        value = self._request()
        if value["gap"]:
            raise ValueError(
                "Latency memory buffer overrun; discarding incomplete joins"
            )
        self.offset = value["next"]
        return value["events"]

    def status(self):
        value = self._request()
        event = value["last"]
        return dict(
            path=self.path,
            size=value["sequence"],
            run_id=event.get("run_id"),
            stage_monotonic_ns=event.get("monotonic_ns"),
        )
