# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from camera_receiver import Avs1TcpReceiver, CODEC_RAW_NV12


@pytest.mark.parametrize(
    "pixel_format,rotation", [("nv12", 0), ("nv12", 180), ("rgba8", 0), ("rgba8", 180)]
)
@pytest.mark.parametrize("fails", [False, True])
def test_raw_work_completes_on_sample_thread_before_slot_release(
    monkeypatch, pixel_format, rotation, fails
):
    pending = threading.local()
    coordinator = threading.get_ident()
    sync_threads = []

    def submit(*args):
        pending.busy = True
        if fails:
            raise RuntimeError("submitted work failed")

    def synchronize():
        sync_threads.append(threading.get_ident())
        pending.busy = False

    @contextlib.contextmanager
    def slot(*args):
        try:
            yield
        finally:
            assert not pending.busy

    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda _: contextlib.nullcontext(),
            get_current_stream=lambda: SimpleNamespace(synchronize=synchronize),
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    output = SimpleNamespace(shape=(3, 4), set=submit)
    runtime = SimpleNamespace(
        config=SimpleNamespace(ipc_names=["eye"], output_format=pixel_format),
        output_buffers=[[output]],
        raw_buffers=[SimpleNamespace(set=submit)],
        raw_kernel=submit,
        grid=None,
        block=None,
    )
    receiver = Avs1TcpReceiver.__new__(Avs1TcpReceiver)
    receiver._gpu_device_id = 0
    receiver._stop = threading.Event()
    receiver._ipc_publisher = SimpleNamespace(write_slot=slot)
    sample = SimpleNamespace(camera_index=0, payload=bytes(12))
    stream = SimpleNamespace(codec=CODEC_RAW_NV12, width=4, height=2, rotation=rotation)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(receiver._process_sample, runtime, stream, sample, 0)
        if fails:
            with pytest.raises(RuntimeError, match="submitted"):
                future.result()
        else:
            assert future.result()
    assert len(sync_threads) == 1 and sync_threads[0] != coordinator
