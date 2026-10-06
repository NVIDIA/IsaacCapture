# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Producer failures reach the consumer instead of leaving an empty mailbox."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sources.synthetic import SyntheticSource, SyntheticStereoSource


class _Buffer(np.ndarray):
    @property
    def device(self):
        return SimpleNamespace(id=0)


@pytest.fixture(params=[SyntheticSource, SyntheticStereoSource])
def source(request, monkeypatch):
    cp = SimpleNamespace(
        zeros=lambda shape, dtype: np.zeros(shape, dtype=dtype).view(_Buffer),
        arange=np.arange,
        sin=np.sin,
        uint8=np.uint8,
        float32=np.float32,
        cuda=SimpleNamespace(
            Device=lambda _: nullcontext(),
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)),
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    instance = request.param("test-camera", 8, 4)
    yield instance
    instance.stop()


@pytest.mark.parametrize("at_sync", [False, True])
def test_cuda_failure_reaches_latest_and_survives_stop(source, at_sync):
    failure = RuntimeError("CUDA_ERROR_NO_BINARY_FOR_GPU")

    def fail(*args, **kwargs):
        raise failure

    if at_sync:
        source._cp.cuda.Stream.null.synchronize = fail
    else:
        source._cp.arange = fail
    source.start()
    source._thread.join(timeout=2)
    assert not source._thread.is_alive(), "producer did not terminate"
    for _ in range(2):
        with pytest.raises(
            RuntimeError, match=r"test-camera.*CUDA_ERROR_NO_BINARY_FOR_GPU"
        ) as error:
            source.latest()
        assert error.value.__cause__ is failure
        source.stop()


def test_restart_clears_failure_and_stale_frame(source):
    failure = RuntimeError("producer failed after publishing")

    def fail():
        with source._lock:
            source._publish_idx = 0
        raise failure

    source._produce_frames = fail
    source.start()
    source._thread.join(timeout=2)
    assert not source._thread.is_alive()
    with pytest.raises(RuntimeError, match="producer failed after publishing"):
        source.latest()
    source.stop()
    source._produce_frames = source._stop.wait
    source.start()
    assert source.latest() is None
