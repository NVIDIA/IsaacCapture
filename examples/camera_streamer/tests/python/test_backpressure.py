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
