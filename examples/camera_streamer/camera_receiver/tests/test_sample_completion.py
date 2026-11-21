# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from camera_receiver import Avs1TcpReceiver


@pytest.mark.parametrize("failure_index", [0, 1])
@pytest.mark.parametrize("raises", [False, True])
def test_failed_stereo_bundle_finishes_both_eyes_before_next_operation(
    monkeypatch, failure_index, raises
):
    monkeypatch.setitem(sys.modules, "cupy", ModuleType("cupy"))
    blocked_eye_started = threading.Event()
    release_eye = threading.Event()
    next_operation = threading.Event()
    receiver = Avs1TcpReceiver.__new__(Avs1TcpReceiver)
    receiver._validate_stream_descriptor = Mock()
    receiver._ipc_publisher = Mock()
    runtime = SimpleNamespace(write_index=0, config=SimpleNamespace(name="head"))
    stream = SimpleNamespace(
        samples=tuple(SimpleNamespace(camera_index=i) for i in range(2))
    )

    def process_sample(_runtime, _stream, sample, _slot):
        if sample.camera_index == failure_index:
            if raises:
                raise RuntimeError("decode failed")
            return False
        blocked_eye_started.set()
        if not release_eye.wait(timeout=5):
            raise TimeoutError("test did not release the other eye")
        return True

    receiver._process_sample = process_sample
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
