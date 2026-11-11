# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless UDP clock responder of the sender: the PC measures the sender's CLOCK_MONOTONIC with it.

Request, 13 bytes little-endian: 'Q', u32 seq, i64 t1. Reply, 29 bytes: 'R', seq, t1, t2, t3, with t2 taken
at receipt and t3 just before the send, both on the sender's monotonic clock (the one sensor_capture_sm_ns is
on). Anything else is ignored. The responder lives for the whole process, independent of media connections.
"""

from __future__ import annotations

import logging
import random
import socket
import struct
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

REQUEST = struct.Struct("<cIq")
REPLY = struct.Struct("<cIqqq")


class ClockResponder:
    """Answers clock requests on one UDP port. Faults are for tests: mute windows, loss, reorder."""

    def __init__(
        self,
        bind: str,
        port: int,
        now_ns: Callable[[], int],
        muted: Callable[[], bool] | None = None,
        loss: float = 0.0,
        reorder: float = 0.0,
        seed: int = 1,
    ) -> None:
        self._now_ns = now_ns
        self._muted = muted or (lambda: False)
        self._loss = loss
        self._reorder = reorder
        self._rng = random.Random(seed)
        self._held: tuple[bytes, tuple] | None = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind, port))
        self._sock.settimeout(0.1)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="clock-responder", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                data, peer = self._sock.recvfrom(2048)
            except TimeoutError:
                continue
            except OSError:
                return
            t2 = self._now_ns()
            if len(data) != REQUEST.size or data[:1] != b"Q" or self._muted():
                continue
            if self._rng.random() < self._loss:
                continue
            _, sequence, t1 = REQUEST.unpack(data)
            reply = REPLY.pack(b"R", sequence, t1, t2, self._now_ns())
            if self._held is not None:
                # A held reply goes out after the newer one: reordered, as a lossy network may do.
                self._sock.sendto(reply, peer)
                self._sock.sendto(*self._held)
                self._held = None
            elif self._rng.random() < self._reorder:
                self._held = (reply, peer)
            else:
                self._sock.sendto(reply, peer)


def parse_window(text: str) -> tuple[float, float]:
    start, _, duration = text.partition("+")
    return float(start), float(duration)


def mute_predicate(windows: list[tuple[float, float]]) -> Callable[[], bool]:
    started = time.monotonic()
    return lambda: any(a <= time.monotonic() - started < a + d for a, d in windows)
