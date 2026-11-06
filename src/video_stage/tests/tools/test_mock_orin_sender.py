# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for mock_orin_sender.py. No GPU needed; the stream tests need ffmpeg.

Run: python3 -m unittest discover -s src/video_stage/tests/tools -p 'test_*.py'
"""

from __future__ import annotations

import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_orin_sender as m  # noqa: E402

START = b"\x00\x00\x00\x01"


def nal(nal_type: int, payload: bytes = b"\x80") -> bytes:
    return START + bytes([0x60 | nal_type]) + payload


# first_mb_in_slice ue(v) codes: 0 -> '1', 1 -> '010', 2 -> '011', 3 -> '00100'
def slice_nal(nal_type: int, first_mb: int) -> bytes:
    code = {0: 0b1, 1: 0b010, 2: 0b011, 3: 0b00100}[first_mb]
    width = {0: 1, 1: 3, 2: 3, 3: 5}[first_mb]
    value = (code << (8 - width)) & 0xFF
    return START + bytes([0x60 | nal_type, value, 0xAA, 0xBB, 0xCC, 0xDD])


class SplitterTest(unittest.TestCase):
    def test_first_mb_in_slice_decoding(self) -> None:
        for mb in (0, 1, 2, 3):
            data = slice_nal(1, mb)
            self.assertEqual(m._first_mb_in_slice(data, 4), mb)

    def test_aud_less_stream_splits_at_parameter_sets_and_slices(self) -> None:
        idr = nal(7) + nal(8) + slice_nal(5, 0)
        p1 = slice_nal(1, 0)
        p2 = slice_nal(1, 0)
        data = idr + p1 + p2 + idr
        units = m.split_access_units(data)
        self.assertEqual([u for u, _ in units], [idr, p1, p2, idr])
        self.assertEqual([k for _, k in units], [True, False, False, True])
        self.assertEqual(b"".join(u for u, _ in units), data)

    def test_filler_attaches_to_the_preceding_unit(self) -> None:
        p1 = slice_nal(1, 0) + nal(12, b"\xff\xff")
        p2 = slice_nal(1, 0)
        units = m.split_access_units(p1 + p2)
        self.assertEqual([u for u, _ in units], [p1, p2])

    def test_multi_slice_picture_is_one_unit(self) -> None:
        pic = slice_nal(1, 0) + slice_nal(1, 1) + slice_nal(1, 2)
        nxt = slice_nal(1, 0)
        units = m.split_access_units(pic + nxt)
        self.assertEqual([u for u, _ in units], [pic, nxt])

    def test_sei_before_a_picture_starts_a_new_unit(self) -> None:
        a = slice_nal(1, 0)
        b = nal(6) + slice_nal(1, 0)
        units = m.split_access_units(a + b)
        self.assertEqual([u for u, _ in units], [a, b])

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg and ffprobe"
    )
    def test_unit_count_matches_ffprobe_and_units_concatenate_to_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.h264"
            m.make_stream(path, 320, 240, 30, seconds=7)
            data = path.read_bytes()
            units = m.split_access_units(data)
            self.assertEqual(b"".join(u for u, _ in units), data)
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v", "-show_entries",
                 "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=True,
            ).stdout  # fmt: skip
            self.assertEqual(len(units), int(out.strip()))
            keys = [i for i, (_, k) in enumerate(units) if k]
            self.assertEqual(keys[0], 0)
            self.assertTrue(all(b == a + 150 for a, b in zip(keys, keys[1:])))
            self.assertTrue(
                units[0][0].startswith(START) and units[0][0][4] & 0x1F == 7
            )  # SPS first
            self.assertFalse(
                any(t == 9 for u, _ in units for _, t, _ in m._nals(u))
            )  # no AUDs


class RecordTest(unittest.TestCase):
    def test_layout(self) -> None:
        au = b"\x00\x00\x00\x01\x65abc"
        rec = m.record(
            au,
            sequence=7,
            frame=14,
            stamp_ns=123456789,
            width=2560,
            height=1984,
            fps=30,
        )
        (length,) = struct.unpack("!I", rec[:4])
        self.assertEqual(length, len(rec) - 4)
        body = rec[4:]
        header = m.HEADER.unpack(body[: m.HEADER.size])
        self.assertEqual(header[0], b"AVS1")
        self.assertEqual(header[3], m.HEADER.size)
        self.assertEqual(header[6], 7)  # sequence
        descriptors = body[
            m.HEADER.size : m.HEADER.size + m.STREAM.size + m.SAMPLE.size
        ]
        self.assertEqual(header[10], len(descriptors))
        self.assertEqual(header[11], len(au))  # payload bytes
        stream = m.STREAM.unpack(descriptors[: m.STREAM.size])
        self.assertEqual(
            (stream[2], stream[5], stream[6], stream[7]), (m.CODEC_H264, 2560, 1984, 30)
        )
        sample = m.SAMPLE.unpack(descriptors[m.STREAM.size :])
        self.assertEqual(sample[7], 14)  # frame number
        self.assertEqual(sample[9], 123456789)  # stamp
        self.assertEqual(sample[-1], len(au))
        self.assertEqual(body[-len(au) :], au)


class FakeWorker:
    """A TCP server accepting DATA then CTRL connections, like the worker's receiver."""

    def __init__(self) -> None:
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(8)
        self.port = self.server.getsockname()[1]
        self.hellos: list[bytes] = []
        self.sessions: list[tuple[socket.socket, socket.socket]] = []
        self.records: list[list[bytes]] = []
        self._threads: list[threading.Thread] = []
        self._stop = False
        t = threading.Thread(target=self._accept, daemon=True)
        t.start()

    def _accept(self) -> None:
        while not self._stop:
            try:
                a, _ = self.server.accept()
                b, _ = self.server.accept()
            except OSError:
                return
            ha, hb = a.recv(4), b.recv(4)
            self.hellos += [ha, hb]
            data, ctrl = (a, b) if ha == b"DATA" else (b, a)
            self.sessions.append((data, ctrl))
            self.records.append([])
            threading.Thread(
                target=self._read, args=(data, len(self.records) - 1), daemon=True
            ).start()

    def _read(self, data: socket.socket, index: int) -> None:
        buf = b""
        try:
            while True:
                chunk = data.recv(65536)
                if not chunk:
                    return
                buf += chunk
                while len(buf) >= 4:
                    (n,) = struct.unpack("!I", buf[:4])
                    if len(buf) < 4 + n:
                        break
                    self.records[index].append(buf[4 : 4 + n])
                    buf = buf[4 + n :]
        except OSError:
            return

    def close_session(self, index: int) -> None:
        for s in self.sessions[index]:
            s.close()

    def close(self) -> None:
        self._stop = True
        self.server.close()
        for pair in self.sessions:
            for s in pair:
                s.close()


def wait_for(cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def frame_numbers(records: list[bytes]) -> list[int]:
    off = m.HEADER.size + m.STREAM.size
    return [m.SAMPLE.unpack(r[off : off + m.SAMPLE.size])[7] for r in records]


IDR = b"\x00\x00\x00\x01\x65" + b"I" * 20
DELTA = b"\x00\x00\x00\x01\x41" + b"P" * 20


class DirectSenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = FakeWorker()
        self.clock = m.OrinClock(5_000_000_000, 0.0)
        self.senders: list[m.DirectSender] = []

    def tearDown(self) -> None:
        for s in self.senders:
            s.stop()
        self.worker.close()

    def sender(self, max_queued: int = 1 << 20) -> m.DirectSender:
        s = m.DirectSender(
            "127.0.0.1", self.worker.port, 64, 48, 30, self.clock, max_queued
        )
        self.senders.append(s)
        return s

    def test_hello_order_is_data_then_ctrl(self) -> None:
        self.sender()
        self.assertTrue(wait_for(lambda: len(self.worker.hellos) == 2))
        self.assertEqual(self.worker.hellos, [b"DATA", b"CTRL"])

    def test_deltas_are_ignored_until_a_keyframe_after_connect(self) -> None:
        s = self.sender()
        self.assertTrue(
            wait_for(
                lambda: s.connects == 1 and s._need_keyframe and s._keyframe_requested
            )
        )
        s.submit(DELTA, False, 2, self.clock.now_ns())
        s.submit(DELTA, False, 4, self.clock.now_ns())
        s.submit(IDR, True, 6, self.clock.now_ns())
        s.submit(DELTA, False, 8, self.clock.now_ns())
        self.assertTrue(
            wait_for(
                lambda: (
                    len(self.worker.records) == 1 and len(self.worker.records[0]) == 2
                )
            )
        )
        self.assertEqual(frame_numbers(self.worker.records[0]), [6, 8])
        self.assertTrue(s.take_keyframe_request())
        self.assertFalse(s.take_keyframe_request())

    def test_reconnects_after_the_server_closes_and_restarts_at_a_keyframe(
        self,
    ) -> None:
        s = self.sender()
        self.assertTrue(wait_for(lambda: s.connects == 1))
        s.submit(IDR, True, 2, self.clock.now_ns())
        self.assertTrue(
            wait_for(lambda: self.worker.records and len(self.worker.records[0]) == 1)
        )
        self.worker.close_session(0)
        self.assertTrue(wait_for(lambda: s.connects == 2, timeout=8))
        self.assertEqual(self.worker.hellos, [b"DATA", b"CTRL", b"DATA", b"CTRL"])
        self.assertTrue(wait_for(lambda: s._need_keyframe and s._keyframe_requested))
        s.submit(
            DELTA, False, 4, self.clock.now_ns()
        )  # ignored: the new connection starts at a keyframe
        s.submit(IDR, True, 6, self.clock.now_ns())
        self.assertTrue(
            wait_for(
                lambda: (
                    len(self.worker.records) == 2 and len(self.worker.records[1]) == 1
                )
            )
        )
        self.assertEqual(frame_numbers(self.worker.records[1]), [6])

    def test_queue_is_cleared_on_reconnect(self) -> None:
        s = self.sender()
        self.assertTrue(wait_for(lambda: s.connects == 1))
        self.worker.close_session(0)
        self.assertTrue(wait_for(lambda: s.connects == 2, timeout=8))
        self.assertTrue(wait_for(lambda: s._need_keyframe))
        self.assertEqual(len(s._queue), 0)
        self.assertEqual(s._queued_bytes, 0)

    def test_overflow_drops_then_ignores_deltas_until_the_next_keyframe(self) -> None:
        s = self.sender()
        self.assertTrue(wait_for(lambda: s.connects == 1))
        # Stop the sender thread from draining: hold the lock while submitting.
        rec_size = len(m.record(IDR, 1, 2, 0, 64, 48, 30))
        s.max_queued_bytes = rec_size * 2 + 1
        with s._cond:
            pass
        s._queue.clear()
        s._need_keyframe = False
        s._queued_bytes = 0
        # Submit while holding the condition's lock so the sender thread cannot pop between submits.
        with s._cond:
            for i, (au, key) in enumerate(
                [(IDR, True), (DELTA, False), (DELTA, False), (DELTA, False)]
            ):
                self._submit_locked(s, au, key, 2 * (i + 1))
            self.assertEqual(
                s.dropped, 1
            )  # the third unit did not fit, the fourth was ignored
            self.assertTrue(s._need_keyframe)
            self.assertTrue(s._keyframe_requested)
            self.assertEqual(len(s._queue), 2)
        # Until the next keyframe, deltas are ignored; the keyframe is taken.
        s.submit(DELTA, False, 20, self.clock.now_ns())
        self.assertEqual(s.dropped, 1)
        self.assertTrue(wait_for(lambda: len(s._queue) == 0))
        s.submit(IDR, True, 22, self.clock.now_ns())
        self.assertFalse(s._need_keyframe)

    @staticmethod
    def _submit_locked(s: m.DirectSender, au: bytes, key: bool, frame: int) -> None:
        # Condition wraps an RLock, so submit() can re-enter it while the test holds it.
        s.submit(au, key, frame, 0)

    def test_k_sets_keyframe_requested(self) -> None:
        s = self.sender()
        self.assertTrue(wait_for(lambda: s.connects == 1))
        self.assertTrue(
            wait_for(lambda: s.take_keyframe_request())
        )  # the connect's own request
        _, ctrl = self.worker.sessions[0]
        ctrl.sendall(b"K")
        self.assertTrue(wait_for(lambda: s.keyframe_requests == 1))
        self.assertTrue(s.take_keyframe_request())

    def test_q_is_answered_with_r_echoing_seq_and_t1_on_the_offset_clock(self) -> None:
        s = self.sender()
        self.assertTrue(wait_for(lambda: s.connects == 1))
        _, ctrl = self.worker.sessions[0]
        t1 = time.monotonic_ns()
        ctrl.settimeout(5)
        ctrl.sendall(struct.pack("<cIq", b"Q", 41, t1))
        buf = b""
        while len(buf) < 29:
            buf += ctrl.recv(29 - len(buf))
        tag, seq, e1, t2, t3 = struct.unpack("<cIqqq", buf)
        self.assertEqual((tag, seq, e1), (b"R", 41, t1))
        self.assertLessEqual(t2, t3)
        local = time.monotonic_ns()
        # On the offset clock: about 5 s ahead of this machine's monotonic clock.
        self.assertTrue(4_900_000_000 < t2 - local < 5_100_000_000)
        self.assertEqual(s.clock_answers, 1)

    def test_k_and_q_in_one_burst_are_both_handled(self) -> None:
        s = self.sender()
        self.assertTrue(wait_for(lambda: s.connects == 1))
        _, ctrl = self.worker.sessions[0]
        ctrl.settimeout(5)
        ctrl.sendall(b"K" + struct.pack("<cIq", b"Q", 1, 5))
        buf = b""
        while len(buf) < 29:
            buf += ctrl.recv(29 - len(buf))
        self.assertEqual(buf[:1], b"R")
        self.assertTrue(wait_for(lambda: s.keyframe_requests == 1))

    def test_drift_moves_the_clock(self) -> None:
        clock = m.OrinClock(0, 1000.0)  # 1000 ppm fast
        a_local, a = time.monotonic_ns(), clock.now_ns()
        time.sleep(0.3)
        b_local, b = time.monotonic_ns(), clock.now_ns()
        extra = (b - a) - (b_local - a_local)
        self.assertGreater(extra, 0.3e9 * 1000e-6 * 0.5)
        flat = m.OrinClock(0, 0.0)
        c = flat.now_ns()
        time.sleep(0.1)
        d = flat.now_ns()
        self.assertAlmostEqual((d - c) / 1e9, 0.1, delta=0.03)


if __name__ == "__main__":
    unittest.main()
