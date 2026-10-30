# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from camera_receiver import _StreamRuntime
from cuda_ipc import CudaIpcFramePublisher


@pytest.fixture
def runtime():
    runtime = _StreamRuntime.__new__(_StreamRuntime)
    runtime.inflight = threading.BoundedSemaphore(1)
    runtime.executor = ThreadPoolExecutor(max_workers=1)
    yield runtime
    runtime.executor.shutdown(wait=True)


def test_full_queue_blocks_and_preserves_order(runtime):
    stop = threading.Event()
    release = threading.Event()
    submitted = threading.Event()
    order = []

    def first():
        release.wait(5)
        order.append(1)

    assert runtime.submit(stop, first)

    def submit_second():
        runtime.submit(stop, lambda: order.append(2))
        submitted.set()

    thread = threading.Thread(target=submit_second)
    thread.start()
    try:
        assert not submitted.wait(0.1)
    finally:
        release.set()
        thread.join(2)
    assert submitted.is_set()
    runtime.executor.shutdown(wait=True)
    assert order == [1, 2]


def test_stop_interrupts_full_queue(runtime):
    stop = threading.Event()
    runtime.inflight.acquire()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(runtime.submit(stop, lambda: None))
    )
    thread.start()
    stop.set()
    thread.join(2)
    runtime.inflight.release()
    assert not thread.is_alive()
    assert result == [False]


def test_submit_failure_returns_capacity(runtime):
    runtime.executor.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        runtime.submit(threading.Event(), lambda: None)
    assert runtime.inflight.acquire(blocking=False)


def test_worker_failure_returns_capacity(runtime):
    def fail():
        raise ValueError("decode failed")

    assert runtime.submit(threading.Event(), fail)
    assert runtime.inflight.acquire(timeout=2)


def test_ipc_socket_is_owner_only(tmp_path):
    path = tmp_path / "frames.sock"
    publisher = CudaIpcFramePublisher(str(path))
    try:
        publisher._open_server()
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        publisher.stop()


@pytest.mark.parametrize("raises", [False, True])
def test_stream_waits_for_other_eye_after_failure(monkeypatch, raises):
    import sys
    from concurrent.futures import Future, wait
    from types import SimpleNamespace

    import camera_receiver as module

    first, second = Future(), Future()
    if raises:
        first.set_exception(RuntimeError("left decode failed"))
    else:
        first.set_result(False)
    futures = iter([first, second])
    receiver = module.Avs1TcpReceiver.__new__(module.Avs1TcpReceiver)
    receiver._sample_pool = SimpleNamespace(submit=lambda *args: next(futures))
    receiver._validate_stream_descriptor = lambda *args: None
    runtime = SimpleNamespace(write_index=0, config=SimpleNamespace(name="head"))
    waiting, finished = threading.Event(), threading.Event()

    def observed_wait(tasks):
        waiting.set()
        return wait(tasks)

    def process():
        try:
            receiver._process_stream_safe(
                None, None, SimpleNamespace(samples=[object(), object()]), runtime
            )
        finally:
            finished.set()

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(module, "wait", observed_wait)
    thread = threading.Thread(target=process)
    thread.start()
    try:
        assert waiting.wait(2)
        assert not finished.is_set()
    finally:
        second.set_result(False)
        thread.join(2)
    assert finished.is_set()
