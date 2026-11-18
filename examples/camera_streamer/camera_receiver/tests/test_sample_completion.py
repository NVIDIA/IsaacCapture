# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from camera_receiver import Avs1TcpReceiver


class _Recorder:
    """Thread-safe ordered log of what the fake publisher, workers and CUDA did."""

    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def add(self, *event):
        with self._lock:
            self.events.append(event)

    def kinds(self):
        return [event[0] for event in self.events]


def _fake_cupy(recorder):
    """The little of CuPy that the write path touches, recording who synchronizes what."""
    cupy = ModuleType("cupy")

    class Device:
        def __init__(self, _id=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def synchronize(self):
            recorder.add("device_sync", threading.get_ident())

    class Stream:
        def synchronize(self):
            recorder.add("stream_sync", threading.get_ident())

    cupy.cuda = SimpleNamespace(Device=Device, get_current_stream=Stream)
    return cupy


def _receiver(recorder, *, process_sample, generations=None, stereo=True):
    receiver = Avs1TcpReceiver.__new__(Avs1TcpReceiver)
    receiver._validate_stream_descriptor = Mock()
    receiver._gpu_device_id = 0
    receiver._stats = None
    receiver._frame_metadata = Mock(return_value={})
    receiver._receiver_latencies = Mock(return_value={})
    counters = {}

    def begin_write(name, slot):
        counters[(name, slot)] = counters.get((name, slot), 0) + 1
        generation = (generations or {}).get(name, 100) + counters[(name, slot)]
        recorder.add("begin_write", name, slot, generation)
        return generation

    receiver._ipc_publisher = Mock()
    receiver._ipc_publisher.begin_write.side_effect = begin_write
    receiver._process_sample = process_sample
    eyes = 2 if stereo else 1
    runtime = SimpleNamespace(
        write_index=0,
        config=SimpleNamespace(
            name="head", ring_depth=3, ipc_names=["head_left", "head_right"][:eyes]
        ),
    )
    stream = SimpleNamespace(
        samples=tuple(SimpleNamespace(camera_index=i) for i in range(eyes)),
        sequence=7,
        tsc_delta_ns=0,
    )
    return receiver, runtime, stream


def _run(receiver, runtime, stream):
    with ThreadPoolExecutor(max_workers=2) as sample_pool:
        receiver._sample_pool = sample_pool
        receiver._process_stream(None, None, stream, runtime)


@pytest.mark.parametrize("failure_index", [0, 1])
@pytest.mark.parametrize("raises", [False, True])
def test_failed_stereo_bundle_finishes_both_eyes_before_next_operation(
    monkeypatch, failure_index, raises
):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))
    blocked_eye_started = threading.Event()
    release_eye = threading.Event()
    next_operation = threading.Event()

    def process_sample(_runtime, _stream, sample, _slot):
        if sample.camera_index == failure_index:
            if raises:
                raise RuntimeError("decode failed")
            return False
        blocked_eye_started.set()
        if not release_eye.wait(timeout=5):
            raise TimeoutError("test did not release the other eye")
        return True

    receiver, runtime, stream = _receiver(recorder, process_sample=process_sample)
    with (
        ThreadPoolExecutor(max_workers=2) as sample_pool,
        ThreadPoolExecutor(max_workers=1) as stream_pool,
    ):
        receiver._sample_pool = sample_pool
        processing = stream_pool.submit(
            receiver._process_stream, None, None, stream, runtime
        )
        following = stream_pool.submit(next_operation.set)
        try:
            assert blocked_eye_started.wait(timeout=2)
            assert not next_operation.wait(timeout=0.1)
            assert not processing.done()
        finally:
            release_eye.set()
        if raises:
            with pytest.raises(RuntimeError, match="decode failed"):
                processing.result(timeout=2)
        else:
            assert processing.result(timeout=2) is None
        following.result(timeout=2)

    assert runtime.write_index == 0
    receiver._ipc_publisher.publish.assert_not_called()


def test_every_eye_marker_advances_before_any_sample_starts(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))

    def process_sample(_runtime, _stream, sample, slot):
        recorder.add("process", sample.camera_index, slot)
        return True

    receiver, runtime, stream = _receiver(recorder, process_sample=process_sample)
    runtime.write_index = 2
    _run(receiver, runtime, stream)

    kinds = recorder.kinds()
    assert kinds[:2] == ["begin_write", "begin_write"]
    assert [e[1:3] for e in recorder.events[:2]] == [
        ("head_left", 2),
        ("head_right", 2),
    ]
    assert kinds.count("process") == 2
    assert kinds.index("process") > 1


def test_failing_eye_leaves_markers_advanced_and_publishes_nothing(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))
    receiver, runtime, stream = _receiver(
        recorder, process_sample=lambda _r, _s, sample, _slot: sample.camera_index == 0
    )
    _run(receiver, runtime, stream)

    assert recorder.kinds().count("begin_write") == 2
    receiver._ipc_publisher.publish.assert_not_called()
    assert runtime.write_index == 0
    assert "device_sync" in recorder.kinds()
    # The next attempt on the same slot gets a newer generation than the failed one.
    first = [e[3] for e in recorder.events if e[0] == "begin_write"]
    recorder.events.clear()
    receiver._process_sample = lambda *_: True
    _run(receiver, runtime, stream)
    second = [e[3] for e in recorder.events if e[0] == "begin_write"]
    assert all(b > a for a, b in zip(first, second))
    assert runtime.write_index == 1


def test_exception_in_one_eye_waits_for_the_other_drains_and_reraises(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))
    other_started = threading.Event()
    release_other = threading.Event()

    def process_sample(_runtime, _stream, sample, _slot):
        if sample.camera_index == 0:
            other_started.wait(timeout=5)
            raise RuntimeError("eye 0 exploded")
        other_started.set()
        assert release_other.wait(timeout=5)
        recorder.add("other_eye_done")
        return True

    receiver, runtime, stream = _receiver(recorder, process_sample=process_sample)
    with (
        ThreadPoolExecutor(max_workers=2) as sample_pool,
        ThreadPoolExecutor(max_workers=1) as stream_pool,
    ):
        receiver._sample_pool = sample_pool
        processing = stream_pool.submit(
            receiver._process_stream, None, None, stream, runtime
        )
        assert other_started.wait(timeout=2)
        assert not processing.done()
        release_other.set()
        with pytest.raises(RuntimeError, match="eye 0 exploded"):
            processing.result(timeout=5)

    kinds = recorder.kinds()
    assert kinds.index("other_eye_done") < kinds.index("device_sync")
    receiver._ipc_publisher.publish.assert_not_called()


def test_first_exception_wins_when_both_eyes_raise(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))

    def process_sample(_runtime, _stream, sample, _slot):
        raise RuntimeError(f"eye {sample.camera_index}")

    receiver, runtime, stream = _receiver(recorder, process_sample=process_sample)
    with pytest.raises(RuntimeError, match="eye 0"):
        _run(receiver, runtime, stream)
    assert "device_sync" in recorder.kinds()


def test_process_stream_safe_logs_the_reraised_exception(monkeypatch, caplog):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))

    def process_sample(*_):
        raise RuntimeError("decode failed")

    receiver, runtime, stream = _receiver(recorder, process_sample=process_sample)
    with ThreadPoolExecutor(max_workers=2) as sample_pool:
        receiver._sample_pool = sample_pool
        with caplog.at_level(logging.WARNING):
            receiver._process_stream_safe(None, None, stream, runtime)
    assert any(
        "processing failed" in r.getMessage() and "decode failed" in r.getMessage()
        for r in caplog.records
    )
    receiver._ipc_publisher.publish.assert_not_called()


def test_success_publishes_exactly_the_generation_begin_write_returned(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))
    receiver, runtime, stream = _receiver(
        recorder,
        process_sample=lambda *_: True,
        generations={"head_left": 40, "head_right": 900},
    )
    runtime.write_index = 1
    _run(receiver, runtime, stream)

    assert receiver._ipc_publisher.begin_write.call_count == 2
    published = {
        call.args[0]: call.kwargs
        for call in receiver._ipc_publisher.publish.call_args_list
    }
    assert published["head_left"]["slot_generation"] == 41
    assert published["head_right"]["slot_generation"] == 901
    assert all(kw["slot"] == 1 for kw in published.values())
    assert receiver._ipc_publisher.publish.call_count == 2
    assert runtime.write_index == 2
    # A second frame on the same slot is a fresh attempt: exactly one more begin_write per eye.
    runtime.write_index = 1
    _run(receiver, runtime, stream)
    published = {
        call.args[0]: call.kwargs
        for call in receiver._ipc_publisher.publish.call_args_list[2:]
    }
    assert published["head_left"]["slot_generation"] == 42
    assert published["head_right"]["slot_generation"] == 902
    assert receiver._ipc_publisher.begin_write.call_count == 4


def test_mono_stream_begins_one_write(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))
    receiver, runtime, stream = _receiver(
        recorder, process_sample=lambda *_: True, stereo=False
    )
    _run(receiver, runtime, stream)
    receiver._ipc_publisher.begin_write.assert_called_once_with("head_left", 0)
    receiver._ipc_publisher.publish.assert_called_once()


def test_each_worker_waits_for_its_own_cuda_stream_after_its_sample(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))
    barrier = threading.Barrier(2, timeout=5)

    def process_sample(_runtime, _stream, sample, _slot):
        barrier.wait()  # force the two eyes onto two different worker threads
        recorder.add("process_end", sample.camera_index, threading.get_ident())
        return True

    receiver, runtime, stream = _receiver(recorder, process_sample=process_sample)
    _run(receiver, runtime, stream)

    ends = [e for e in recorder.events if e[0] == "process_end"]
    assert len({e[2] for e in ends}) == 2
    for _, _, thread in ends:
        idx = recorder.events.index(next(e for e in ends if e[2] == thread))
        later = [
            e
            for e in recorder.events[idx + 1 :]
            if e[0] == "stream_sync" and e[1] == thread
        ]
        assert later, "worker returned without synchronizing its own stream"


@pytest.mark.parametrize("raises", [False, True])
def test_worker_waits_for_its_stream_even_when_the_sample_fails_softly(
    monkeypatch, raises
):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(recorder))

    def process_sample(*_):
        if raises:
            raise RuntimeError("boom")
        return False

    receiver, runtime, stream = _receiver(
        recorder, process_sample=process_sample, stereo=False
    )
    try:
        _run(receiver, runtime, stream)
    except RuntimeError:
        pass
    # A soft failure returns False after the work it launched; that work must still be waited on
    # before anything is reported. (An exception path is covered by the device-wide drain.)
    assert "device_sync" in recorder.kinds()
    if not raises:
        assert "stream_sync" in recorder.kinds()
