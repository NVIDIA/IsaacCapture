#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the receiver's half-open-link detection.

If the AGX reboots mid-stream the PC socket stays ESTABLISHED but never delivers
another byte. `_recv_exact` must give up after `_STALL_TIMEOUT_S` so the produce
loop reconnects, instead of retrying `recv` on the dead socket forever (which
left the camera black until someone killed the receiver by hand)."""

from __future__ import annotations

import socket
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import camera_receiver  # noqa: E402
from camera_receiver import Avs1TcpReceiver  # noqa: E402


def _bare_receiver() -> Avs1TcpReceiver:
    """An instance with only the attributes `_recv_exact` touches, skipping the
    real __init__ (which needs a config, decoders and a GPU)."""
    recv = Avs1TcpReceiver.__new__(Avs1TcpReceiver)
    recv._stop = threading.Event()
    recv._receive_fps_monitor = None
    recv._host = "6.6.7.100"
    recv._port = 5000
    recv._link_warm = True  # warmed stream by default; cold-start tests flip this
    return recv


class _AlwaysTimeoutSocket:
    def recv_into(self, *_args):
        raise socket.timeout()


class _DataThenBlockSocket:
    """Delivers `chunks` (bytes) on successive recv_into calls, then times out."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def recv_into(self, view, _nbytes=None):
        if not self._chunks:
            raise socket.timeout()
        chunk = self._chunks.pop(0)
        view[: len(chunk)] = chunk
        return len(chunk)


class RecvExactHalfOpenTest(unittest.TestCase):
    def test_gives_up_after_stall_timeout(self):
        recv = _bare_receiver()
        # monotonic starts at 0 (last_progress), then jumps past the deadline on
        # the first in-loop check so the stall is detected without a real wait.
        with patch.object(
            camera_receiver.time,
            "monotonic",
            side_effect=[0.0, camera_receiver._STALL_TIMEOUT_S + 1.0],
        ):
            ok = recv._recv_exact(_AlwaysTimeoutSocket(), memoryview(bytearray(16)))
        self.assertFalse(
            ok,
            "a half-open (never-delivering) socket must return False to force a reconnect",
        )

    def test_returns_true_when_data_arrives(self):
        recv = _bare_receiver()
        sock = _DataThenBlockSocket([b"abcd", b"efgh"])
        ok = recv._recv_exact(sock, memoryview(bytearray(8)))
        self.assertTrue(ok)

    def test_tolerates_a_brief_gap_then_data(self):
        recv = _bare_receiver()

        # one timeout (within the deadline), then the record's bytes arrive.
        class _GapThenData:
            def __init__(self):
                self._sent = False

            def recv_into(self, view, _n=None):
                if not self._sent:
                    self._sent = True
                    raise socket.timeout()
                view[:4] = b"wxyz"
                return 4

        with patch.object(
            camera_receiver.time, "monotonic", side_effect=[0.0, 1.0, 1.0]
        ):
            ok = recv._recv_exact(_GapThenData(), memoryview(bytearray(4)))
        self.assertTrue(ok, "a brief sub-deadline gap must not trigger a reconnect")


class RecvExactColdStartTest(unittest.TestCase):
    def test_cold_start_does_not_give_up_before_first_bundle(self):
        # Fresh, not-yet-warm link: the sender's NVENC warmup can take a few
        # seconds before the first byte. The 3s stall must NOT fire yet.
        recv = _bare_receiver()
        recv._link_warm = False  # no bundle received yet
        # times out a few times (~4s elapsed, past _STALL_TIMEOUT_S=3 but under
        # _COLD_START_TIMEOUT_S=15), then the first bytes arrive.
        _DataThenBlockSocket([])  # will raise timeout first

        class _WarmupSock:
            def __init__(self):
                self.calls = 0

            def recv_into(self, view, _n=None):
                self.calls += 1
                if self.calls < 3:
                    raise socket.timeout()
                view[:4] = b"cold"
                return 4

        with patch.object(
            camera_receiver.time, "monotonic", side_effect=[0.0, 2.0, 4.0, 4.0]
        ):
            ok = recv._recv_exact(_WarmupSock(), memoryview(bytearray(4)))
        self.assertTrue(
            ok, "cold start must wait past _STALL_TIMEOUT_S for the first bundle"
        )

    def test_warm_link_still_gives_up_at_stall_timeout(self):
        recv = _bare_receiver()  # warm by default
        with patch.object(
            camera_receiver.time,
            "monotonic",
            side_effect=[0.0, camera_receiver._STALL_TIMEOUT_S + 1.0],
        ):
            ok = recv._recv_exact(_AlwaysTimeoutSocket(), memoryview(bytearray(16)))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
