# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU acceptance tests: an accepted picture always matches its notice's frame identity.

The writer (``CudaIpcFramePublisher`` plus test kernels) runs in a spawned child process, because
CUDA IPC handles cannot be opened in the exporting process; the test process is the reader
(``CudaIpcClient``). Pixels are a pure function of the frame id, recomputed in numpy here, and
GPU gates (stream host functions) hold a reader's copy in flight while the writer overwrites a
slot.
"""

from __future__ import annotations

import collections
import faulthandler
import multiprocessing
import os
import queue
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
_STREAMER_ROOT = _HERE.parents[2]
_REPO_ROOT = _HERE.parents[4]
for _p in (
    _STREAMER_ROOT,
    _STREAMER_ROOT / "camera_receiver",
    _STREAMER_ROOT / "cuda_ipc_consumer",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import cupy as cp

    _GPU_REASON = (
        None if cp.cuda.runtime.getDeviceCount() > 0 else "no CUDA device present"
    )
except Exception as _exc:  # noqa: BLE001 - any import/driver failure means skip
    cp = None
    _GPU_REASON = f"CuPy/CUDA unavailable: {_exc}"

try:
    import ipc_protocol  # noqa: F401

    _MARKER_REASON = None
except ImportError as _exc:
    _MARKER_REASON = f"ipc_protocol not built for this interpreter: {_exc}"

pytestmark = pytest.mark.skipif(
    _GPU_REASON is not None or _MARKER_REASON is not None,
    reason=_GPU_REASON or _MARKER_REASON or "",
)

ROWS, COLS = 96, 128  # 128x64 luma rows + 32 chroma rows
BANDS = 4
CMD_TIMEOUT_S = 90.0
GATE_TIMEOUT_S = 10.0


def expected_pixels(fid: int, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """Pixel pattern of frame ``fid``; every row of every plane carries the id."""
    r = np.arange(rows, dtype=np.int64)[:, None]
    c = np.arange(cols, dtype=np.int64)[None, :]
    return ((fid * 37 + (fid >> 8) * 11 + r * 5 + c) % 251).astype(np.uint8)


def expected_box(fid: int) -> np.ndarray:
    """What the pair-averaging kernel reader produces for frame ``fid``."""
    v = expected_pixels(fid).astype(np.int32).reshape(ROWS, COLS // 2, 2)
    avg = ((v[:, :, 0] + v[:, :, 1] + 1) >> 1).astype(np.uint8)
    return np.repeat(avg, 2, axis=1)


def timestamp_for(seq: int) -> int:
    return 1_000_000_000 + seq * 33_333_333


# --------------------------------------------------------------------------- writer process


def _writer_main(cmd_q, rsp_q, cfg) -> None:
    """Writer process: publisher plus test write kernels, driven by commands."""
    try:
        import cupy as wcp

        from cuda_ipc import CudaIpcFramePublisher

        wcp.cuda.set_allocator(
            None
        )  # raw cudaMalloc so every slot is its own IPC allocation
        write = wcp.ElementwiseKernel(
            "int64 fid, int32 W",
            "uint8 out",
            "int r = i / W; int c = i % W;"
            "out = (unsigned char)((fid * 37 + (fid >> 8) * 11 + (long long)r * 5 + c) % 251);",
            "test_write_pattern",
        )
        stream = wcp.cuda.Stream(non_blocking=True)
        pub = CudaIpcFramePublisher(socket_path=cfg["socket"], queue_size=512)
        bufs: dict[str, list] = {}
        for name, rows, cols, ring in cfg["streams"]:
            bufs[name] = [wcp.zeros((rows, cols), dtype=wcp.uint8) for _ in range(ring)]
            pub.register_stream(name, cols, rows * 2 // 3, "nv12", 0, bufs[name])
        pub.start()
        rsp_q.put(("ok", {"socket": cfg["socket"]}))
    except BaseException:  # noqa: BLE001
        rsp_q.put(("err", traceback.format_exc()))
        return

    def write_slot(name, slot, fid, rows=None):
        arr = bufs[name][slot]
        view = arr if rows is None else arr[:rows]
        with stream:
            write(fid, arr.shape[1], view)
        stream.synchronize()

    try:
        while True:
            cmd, args = cmd_q.get()
            try:
                if cmd == "stop":
                    break
                if cmd == "write_many":
                    # items: (stream, slot, fid, rows_or_None, publish)
                    gens = [pub.begin_write(it[0], it[1]) for it in args["items"]]
                    for it in args["items"]:
                        write_slot(it[0], it[1], it[2], it[3])
                    for it, g in zip(args["items"], gens):
                        if it[4]:
                            pub.publish(it[0], it[1], g, it[2], timestamp_for(it[2]))
                    rsp_q.put(("ok", gens))
                elif cmd == "burst":
                    name, ring = args["stream"], len(bufs[args["stream"]])
                    t0 = time.perf_counter()
                    for fid in range(args["first"], args["first"] + args["n"]):
                        slot = fid % ring
                        g = pub.begin_write(name, slot)
                        write_slot(name, slot, fid)
                        pub.publish(name, slot, g, fid, timestamp_for(fid))
                    rsp_q.put(("ok", {"elapsed_s": time.perf_counter() - t0}))
                elif cmd == "bench":
                    name, ring = args["stream"], len(bufs[args["stream"]])
                    lat = []
                    t0 = time.perf_counter()
                    for fid in range(args["n"]):
                        t1 = time.perf_counter()
                        slot = fid % ring
                        g = pub.begin_write(name, slot) if args["begin"] else 0
                        write_slot(name, slot, fid)
                        pub.publish(name, slot, g, fid, timestamp_for(fid))
                        lat.append(time.perf_counter() - t1)
                    total = time.perf_counter() - t0
                    lat.sort()
                    rsp_q.put(
                        (
                            "ok",
                            {
                                "fps": args["n"] / total,
                                "mean_us": 1e6 * sum(lat) / len(lat),
                                "p50_us": 1e6 * lat[len(lat) // 2],
                                "p99_us": 1e6 * lat[int(len(lat) * 0.99)],
                            },
                        )
                    )
                else:
                    rsp_q.put(("err", f"unknown command {cmd}"))
            except BaseException:  # noqa: BLE001
                rsp_q.put(("err", traceback.format_exc()))
    finally:
        pub.stop()


class Writer:
    """Parent-side handle on a writer child process."""

    def __init__(self, streams, workdir: Path) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.cmd_q = ctx.Queue()
        self.rsp_q = ctx.Queue()
        self.socket = str(Path(tempfile.mkdtemp(dir=workdir)) / "s.sock")
        self.proc = ctx.Process(
            target=_writer_main,
            args=(self.cmd_q, self.rsp_q, {"socket": self.socket, "streams": streams}),
            daemon=True,
        )
        self.proc.start()
        self.result("startup")

    def result(self, what: str, timeout: float = CMD_TIMEOUT_S):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status, payload = self.rsp_q.get(timeout=0.25)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f"writer died during {what}") from None
                continue
            if status != "ok":
                raise RuntimeError(f"writer failed during {what}:\n{payload}")
            return payload
        raise TimeoutError(f"writer timed out during {what}")

    def poll_result(self):
        """Non-blocking: the pending response or None."""
        try:
            status, payload = self.rsp_q.get_nowait()
        except queue.Empty:
            return None
        if status != "ok":
            raise RuntimeError(f"writer failed:\n{payload}")
        return payload

    def send(self, cmd: str, **args) -> None:
        self.cmd_q.put((cmd, args))

    def call(self, cmd: str, **args):
        self.send(cmd, **args)
        return self.result(cmd)

    def write_many(self, *items):
        return self.call("write_many", items=list(items))

    def close(self) -> None:
        try:
            if self.proc.is_alive():
                self.cmd_q.put(("stop", {}))
                self.proc.join(10)
        finally:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(5)
            if self.proc.is_alive():
                self.proc.kill()
            shutil.rmtree(Path(self.socket).parent, ignore_errors=True)


# --------------------------------------------------------------------------- reader side


class Reader:
    """Test-process reader: a CudaIpcClient plus a notice buffer."""

    def __init__(self, writer: Writer, streams=None) -> None:
        from consumer import CudaIpcClient

        self.client = CudaIpcClient(writer.socket, streams)
        self.client.connect()
        self.notices: collections.deque = collections.deque()
        self.stream = cp.cuda.Stream(non_blocking=True)

    def pump(self, timeout_s: float = 0.05) -> None:
        msgs = self.client.read_messages(timeout_s)
        assert msgs is not None, "publisher closed the socket"
        self.notices.extend(m for m in msgs if m.get("type") == "frame")

    def next_notice(self, stream: str | None = None, timeout_s: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for i, n in enumerate(self.notices):
                if stream is None or n["stream"] == stream:
                    del self.notices[i]
                    return n
            self.pump(0.05)
        raise TimeoutError("no frame notice arrived")

    def expect_no_notice(self, wait_s: float = 0.4) -> None:
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            self.pump(0.05)
        assert not self.notices, f"unexpected notices {list(self.notices)}"

    def array(self, notice: dict):
        return self.client.slots[(notice["stream"], notice["slot"])].array

    def unchanged(self, notice: dict) -> bool:
        return self.client.slot_unchanged(
            notice["stream"], notice["slot"], notice["slot_generation"]
        )

    def copy_picture(self, notice: dict) -> np.ndarray:
        """Copy into reader-owned memory on the reader stream and wait for all GPU work."""
        with self.stream:
            dst = cp.empty_like(self.array(notice))
            dst[...] = self.array(notice)
        self.stream.synchronize()
        return cp.asnumpy(dst)

    def read_checked(self, notice: dict) -> tuple[bool, np.ndarray]:
        pic = self.copy_picture(notice)
        return self.unchanged(notice), pic

    def close(self) -> None:
        self.client.close()


class Gate:
    """Holds a CUDA stream behind a host function until the test opens it."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.opened = threading.Event()
        self.after_ran = threading.Event()

    def _block(self, _arg) -> None:
        self.reached.set()
        self.opened.wait(GATE_TIMEOUT_S)

    def _after(self, _arg) -> None:
        self.after_ran.set()

    def enqueue(self, stream) -> None:
        stream.launch_host_func(self._block, None)

    def enqueue_probe(self, stream) -> None:
        """Queue a flag behind everything already queued; it stays unset while closed."""
        stream.launch_host_func(self._after, None)

    def wait_reached(self) -> None:
        assert self.reached.wait(GATE_TIMEOUT_S), "gate never reached"

    def assert_closed(self, stream) -> None:
        time.sleep(0.15)
        assert not self.after_ran.is_set(), "work behind a closed gate already ran"
        assert not stream.done, "stream finished while its gate was closed"

    def open(self) -> None:
        self.opened.set()


@pytest.fixture(autouse=True)
def _watchdog():
    faulthandler.dump_traceback_later(240, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def workdir():
    root = _REPO_ROOT / "tmp"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="gpu_slot_", dir=root))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def make_writer(workdir):
    made: list = []

    def factory(streams):
        w = Writer(streams, workdir)
        made.append(w)
        return w

    yield factory
    for w in made:
        w.close()


@pytest.fixture
def make_reader():
    made: list = []

    def factory(writer, streams=None):
        r = Reader(writer, streams)
        made.append(r)
        return r

    yield factory
    for r in made:
        r.close()


# --------------------------------------------------------------------------- scenarios


def _check_identity(notice: dict, pic: np.ndarray) -> None:
    assert notice["timestamp_ns"] == timestamp_for(notice["seq"])
    np.testing.assert_array_equal(pic, expected_pixels(notice["seq"]))


def test_baseline_no_overlap(make_writer, make_reader):
    ring = 6
    w = make_writer([("cam", ROWS, COLS, ring)])
    r = make_reader(w)
    accepted = 0
    for fid in range(1, 2 * ring + 1):  # two laps, every slot rewritten
        slot = fid % ring
        w.write_many(("cam", slot, fid, None, True))
        n = r.next_notice("cam")
        assert (n["seq"], n["slot"]) == (fid, slot)
        ok, pic = r.read_checked(n)
        assert ok
        _check_identity(n, pic)
        accepted += 1
    assert accepted == 2 * ring


def _gated_full_copy(r: Reader, w: Writer, notice, overwrite_fid, check=True):
    """Copy ``notice``'s slot behind a gate; overwrite the slot (no notice) while it is held."""
    gate = Gate()
    gate.enqueue(r.stream)
    with r.stream:
        dst = cp.empty_like(r.array(notice))
        dst[...] = r.array(notice)
    gate.enqueue_probe(r.stream)
    gate.wait_reached()
    gate.assert_closed(r.stream)
    if overwrite_fid is not None:
        w.write_many(("cam", notice["slot"], overwrite_fid, None, False))
    gate.open()
    r.stream.synchronize()
    assert gate.after_ran.wait(GATE_TIMEOUT_S)
    pic = cp.asnumpy(dst)
    unchanged = r.unchanged(notice)
    return (unchanged if check else True), pic


@pytest.mark.parametrize("check", [True, False], ids=["checked", "negative_control"])
def test_overwrite_before_copy_completes(make_writer, make_reader, check):
    w = make_writer([("cam", ROWS, COLS, 3)])
    r = make_reader(w)
    # Control: same gated path with no overwrite is accepted and exact.
    w.write_many(("cam", 0, 100, None, True))
    n0 = r.next_notice("cam")
    accepted, pic = _gated_full_copy(r, w, n0, None)
    assert accepted
    _check_identity(n0, pic)

    w.write_many(("cam", 1, 200, None, True))
    n = r.next_notice("cam")
    accepted, pic = _gated_full_copy(r, w, n, 201, check=check)
    r.expect_no_notice()  # B's notice is withheld
    np.testing.assert_array_equal(pic, expected_pixels(201))  # the copy holds B
    if check:
        assert not accepted, "overwritten slot was accepted"
    else:
        # Negative control: with the marker check off the wrong picture is accepted.
        assert accepted
        assert not np.array_equal(pic, expected_pixels(n["seq"]))


def _banded_reader(r: Reader, w: Writer, notice, overwrite_fid, kernel: bool):
    """Read the slot in BANDS row bands; overwrite it between band 2 and 3."""
    gate = Gate()
    src = r.array(notice)
    band = ROWS // BANDS
    box = cp.ElementwiseKernel(
        "raw uint8 x",
        "uint8 y",
        "y = (unsigned char)(((int)x[i] + (int)x[i ^ 1] + 1) >> 1);",
        "test_box_pairs",
    )
    with r.stream:
        dst = cp.empty_like(src)
        for b in range(BANDS):
            if b == 2:
                gate.enqueue(r.stream)
            s, d = src[b * band : (b + 1) * band], dst[b * band : (b + 1) * band]
            if kernel:
                box(s, d)
            else:
                d[...] = s
    gate.enqueue_probe(r.stream)
    gate.wait_reached()
    gate.assert_closed(r.stream)
    if overwrite_fid is not None:
        w.write_many(("cam", notice["slot"], overwrite_fid, None, False))
    gate.open()
    r.stream.synchronize()
    assert gate.after_ran.wait(GATE_TIMEOUT_S)
    return cp.asnumpy(dst)


@pytest.mark.parametrize("kernel", [False, True], ids=["copy", "box_kernel"])
@pytest.mark.parametrize("check", [True, False], ids=["checked", "negative_control"])
def test_genuine_overlap_mixes_frames(make_writer, make_reader, kernel, check):
    ref = expected_box if kernel else expected_pixels
    w = make_writer([("cam", ROWS, COLS, 3)])
    r = make_reader(w)
    # Control: kernel/copy path without overlap is exact (validates expected values).
    w.write_many(("cam", 0, 10, None, True))
    n0 = r.next_notice("cam")
    pic0 = _banded_reader(r, w, n0, None, kernel)
    assert r.unchanged(n0)
    np.testing.assert_array_equal(pic0, ref(10))

    w.write_many(("cam", 1, 20, None, True))
    n = r.next_notice("cam")
    pic = _banded_reader(r, w, n, 21, kernel)
    r.expect_no_notice()
    band = ROWS // BANDS
    from_a = [
        np.array_equal(
            pic[b * band : (b + 1) * band], ref(20)[b * band : (b + 1) * band]
        )
        for b in range(BANDS)
    ]
    from_b = [
        np.array_equal(
            pic[b * band : (b + 1) * band], ref(21)[b * band : (b + 1) * band]
        )
        for b in range(BANDS)
    ]
    print(f"\n[overlap kernel={kernel}] bands from A={from_a} from B={from_b}")
    assert from_a == [True, True, False, False]
    assert from_b == [False, False, True, True]  # a real mixture of two frames
    assert not np.array_equal(pic, ref(20)) and not np.array_equal(pic, ref(21))
    accepted = r.unchanged(n) if check else True
    if check:
        assert not accepted
    else:
        assert accepted  # defect reproduced when the check is disabled


def test_single_stream_partial_failure_rejects_earlier_notice(make_writer, make_reader):
    w = make_writer([("cam", ROWS, COLS, 3)])
    r = make_reader(w)
    w.write_many(("cam", 2, 50, None, True))
    a = r.next_notice("cam")
    gens = w.write_many(
        ("cam", 2, 51, ROWS // 3, False)
    )  # touches rows, never completes
    assert gens[0] > a["slot_generation"]
    r.expect_no_notice()
    ok, pic = r.read_checked(a)
    assert not ok
    third = ROWS // 3
    assert np.array_equal(
        pic[:third], expected_pixels(51)[:third]
    )  # the slot really is torn
    assert np.array_equal(pic[third:], expected_pixels(50)[third:])
    # A later completed write gets a fresh generation and is accepted; A stays rejected.
    w.write_many(("cam", 2, 52, None, True))
    c = r.next_notice("cam")
    ok, pic = r.read_checked(c)
    assert ok
    _check_identity(c, pic)
    assert not r.unchanged(a)
    assert c["slot_generation"] > gens[0]


def test_stereo_pair_one_eye_fails(make_writer, make_reader):
    w = make_writer([("left", ROWS, COLS, 3), ("right", ROWS, COLS, 3)])
    r = make_reader(w)
    w.write_many(("left", 0, 60, None, True), ("right", 0, 61, None, True))
    a_l, a_r = r.next_notice("left"), r.next_notice("right")
    for n in (a_l, a_r):
        ok, pic = r.read_checked(n)
        assert ok
        _check_identity(n, pic)
    # One attempt writes both eyes into slot 0: left completes, right fails partway.
    w.write_many(("left", 0, 70, None, True), ("right", 0, 71, ROWS // 2, False))
    b_l = r.next_notice("left")
    r.expect_no_notice()  # no notice for the failed eye
    ok, pic = r.read_checked(b_l)
    assert ok
    _check_identity(b_l, pic)
    for old in (a_l, a_r):
        ok, _pic = r.read_checked(old)
        assert not ok, f"earlier {old['stream']} notice accepted after the attempt"


def _process(r: Reader, n: dict, tally: dict) -> None:
    pic = r.copy_picture(n)
    if not r.unchanged(n):
        tally["discarded"] += 1
        return
    tally["accepted"] += 1
    tally["accepted_seqs"].append(n["seq"])
    assert n["slot"] == n["seq"] % 3
    assert n["timestamp_ns"] == timestamp_for(n["seq"])
    if not np.array_equal(pic, expected_pixels(n["seq"])):
        tally["mismatch"] += 1


def _tally() -> dict:
    return dict(accepted=0, discarded=0, mismatch=0, dropped=0, accepted_seqs=[])


def test_burst_with_stalled_reader(make_writer, make_reader):
    w = make_writer([("cam", ROWS, COLS, 3)])
    r = make_reader(w)
    n_frames = 240
    # Reader stalled (never reads the socket); writer is unpaced and must not wait.
    t0 = time.monotonic()
    res = w.call("burst", stream="cam", first=1, n=n_frames)
    wall = time.monotonic() - t0
    print(
        f"\n[burst] {n_frames} frames, writer {res['elapsed_s']:.3f}s "
        f"({n_frames / res['elapsed_s']:.0f} fps) with a stalled reader, wall {wall:.2f}s"
    )
    assert res["elapsed_s"] < 30.0
    time.sleep(0.5)  # stall
    tally = _tally()
    seen = 0
    while seen < n_frames:
        n = r.next_notice("cam")
        seen += 1
        _process(r, n, tally)
    print(f"[burst late drain] {tally}")
    assert tally["mismatch"] == 0
    assert tally["accepted"] == 3  # only the latest picture of each slot survives
    assert sorted(tally["accepted_seqs"]) == [n_frames - 2, n_frames - 1, n_frames]
    assert tally["accepted"] + tally["discarded"] == n_frames


def test_burst_with_delayed_and_dropped_notices(make_writer, make_reader):
    w = make_writer([("cam", ROWS, COLS, 3)])
    r = make_reader(w)
    rng = random.Random(1234)
    tally = _tally()
    n_frames = 900
    w.send("burst", stream="cam", first=1, n=n_frames)
    res = None
    live = 0
    deadline = time.monotonic() + 60
    while res is None and time.monotonic() < deadline:
        r.pump(0.01)
        while r.notices and res is None:
            n = r.notices.popleft()
            live += 1
            roll = rng.random()
            if roll < 0.3:
                tally["dropped"] += 1
            elif roll < 0.5:
                time.sleep(0.003)  # delayed past further rewrites
                _process(r, n, tally)
            else:
                _process(r, n, tally)
            res = w.poll_result()
    if res is None:
        res = w.result("burst")
    live_accepted = tally["accepted"]
    time.sleep(0.5)  # stall, then drain the rest late with no drops
    while True:
        r.pump(0.2)
        if not r.notices:
            break
        while r.notices:
            _process(r, r.notices.popleft(), tally)
    print(
        f"\n[delayed/dropped] writer {res['elapsed_s']:.3f}s; live={live} "
        f"live_accepted={live_accepted} total={tally}"
    )
    assert tally["mismatch"] == 0
    assert (
        tally["accepted"] > live_accepted
    )  # final late drain accepts the newest pictures
    assert res["elapsed_s"] < 60.0


# --------------------------------------------------------------------------- measurements

# Manual measurements, opt-in: SLOT_MARKER_MEASUREMENTS=1 pytest -s <this file> -k measure
_measure = pytest.mark.skipif(
    os.environ.get("SLOT_MARKER_MEASUREMENTS") != "1",
    reason="manual measurement; set SLOT_MARKER_MEASUREMENTS=1 to run",
)


@_measure
def test_measure_marker_call_cost():
    from ipc_protocol import SlotMarkers

    m = SlotMarkers.create("/slotit_bench", [1])
    try:
        n = 1_000_000
        t = time.perf_counter_ns()
        for _ in range(n):
            m.begin_write(0, 0)
        bw = (time.perf_counter_ns() - t) / n
        t = time.perf_counter_ns()
        for _ in range(n):
            m.load(0, 0)
        ld = (time.perf_counter_ns() - t) / n
        t = time.perf_counter_ns()
        for _ in range(n):
            m.unchanged(0, 0, 1)
        un = (time.perf_counter_ns() - t) / n
        t = time.perf_counter_ns()
        for _ in range(n):
            pass
        base = (time.perf_counter_ns() - t) / n
        print(
            f"\n[marker ns/call incl. python call overhead] begin_write={bw:.0f} "
            f"load={ld:.0f} unchanged={un:.0f} empty-loop={base:.0f}"
        )
        assert m.load(0, 0) == n
    finally:
        m.close()


@_measure
@pytest.mark.parametrize(
    "rows,cols,n,label",
    [(ROWS, COLS, 3000, "128x64 NV12"), (1984 * 3 // 2, 2560, 300, "2560x1984 NV12")],
)
def test_measure_writer_cost_with_and_without_begin_write(
    make_writer, make_reader, rows, cols, n, label
):
    w = make_writer([("cam", rows, cols, 3)])
    r = make_reader(w)  # attached and stalled: never reads
    w.call("bench", stream="cam", n=20, begin=True)  # warm-up
    out = {}
    for rep in range(2):
        for begin in (False, True):
            out[(begin, rep)] = w.call("bench", stream="cam", n=n, begin=begin)
    for (begin, rep), v in out.items():
        print(
            f"\n[bench {label} begin_write={begin} rep={rep}] {v['fps']:.0f} fps  "
            f"mean={v['mean_us']:.1f}us p50={v['p50_us']:.1f}us p99={v['p99_us']:.1f}us"
        )
    del r
    mean_without = statistics.mean(v["mean_us"] for (b, _), v in out.items() if not b)
    mean_with = statistics.mean(v["mean_us"] for (b, _), v in out.items() if b)
    print(
        f"[bench {label}] mean per-frame: without={mean_without:.1f}us with={mean_with:.1f}us"
    )
