# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import threading
from types import SimpleNamespace

import pytest

from isaaccapture.ovxr_session_manager.camera_mailbox import FrameOutputs

from isaaccapture.ovxr_session_manager import owned_mirror as mirror
from isaaccapture.ovxr_session_manager.desktop_mirror import DesktopMirrorConfig
from test_passive_mirror import LEFT, RIGHT, outputs


class FakeMap:
    width = height = 1
    pitch = 4
    device = 0

    def __init__(self, handle):
        self.pointer = handle
        self.closed = False
        self.owner = threading.get_ident()

    def close(self):
        assert threading.get_ident() == self.owner
        assert not self.closed
        self.closed = True


def mailbox():
    mapped = []

    def map_image(_renderer, handle, _limit):
        m = FakeMap(handle)
        mapped.append(m)
        return m

    box = mirror.OwnedMirrorMailbox(30, gpu_mapper=map_image)
    box.set_products((LEFT, RIGHT))
    return box, mapped


def publish(box, handle):
    box._next_ns = 0
    box.on_outputs(FrameOutputs(1, outputs(Left=handle, Right=handle + 100)))


def test_slow_gpu_consumer_retains_map_and_drops_only_unused_pending_maps():
    box, maps = mailbox()
    publish(box, 1)
    first = box.acquire(0)
    publish(box, 2)
    publish(box, 3)
    assert len(box._mapped) == 2
    assert not maps[0].closed and maps[1].closed and not maps[2].closed
    assert box.snapshot().gpu.pointer == 3 and box.snapshot().rgba is None
    box.select(RIGHT)
    assert box.snapshot() is None and not maps[0].closed
    publish(box, 4)
    assert maps[2].closed and box.snapshot().gpu.pointer == 104
    assert box.snapshot().product == RIGHT
    with pytest.raises(RuntimeError, match="unfinished"):
        box.close()
    box.release(first)
    box.close()
    assert all(m.closed for m in maps) and not box._mapped


def test_gpu_selection_change_during_map_does_not_relabel_or_leak():
    box, maps = mailbox()
    mapper = box._gpu_mapper

    def switch_during_map(*args):
        value = mapper(*args)
        box.select(RIGHT)
        return value

    box._gpu_mapper = switch_during_map
    publish(box, 1)
    assert box.snapshot() is None
    box.close()
    assert all(m.closed for m in maps)


def test_gui_device_bootstrap_omits_copy_without_a_spurious_failure():
    from isaaccapture.ovxr_session_manager.gpu_output import OwnedCudaImagePool

    pool = OwnedCudaImagePool()
    box = mirror.OwnedMirrorMailbox(30, gpu_mapper=pool)
    box.set_products((LEFT, RIGHT))
    publish(box, 1)
    assert box.snapshot() is None and box.view()[4] is None
    assert not pool._images and pool._source is None
    box.close()


@pytest.mark.parametrize(
    "upload_error,wait_error",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_gpu_upload_release_is_after_completion_and_failed_wait_keeps_map(
    monkeypatch, upload_error, wait_error, mock_omni_ui, mock_cuda
):
    box, maps = mailbox()
    publish(box, 1)
    events = []
    maps[0].device = 1
    current_device = [1]

    def shutdown():
        assert current_device[0] == 0
        events.append("shutdown")

    worker = mirror.OwnedDesktopMirrorWorker(
        box.snapshot, DesktopMirrorConfig(), products=box
    )

    class Context:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def destroy(self):
            assert current_device[0] == 0
            events.append("destroy")

        def set_build_fn(self, _fn):
            pass

        def rebuild(self):
            pass

    Context.frame = Context()

    class Provider:
        def set_bytes_data_from_gpu(self, pointer, sizes, _format, pitch):
            assert pointer == 1 and sizes == [1, 1] and pitch == 4
            assert box._in_use is not None and not maps[0].closed
            current_device[0] = 0  # OVUI selects its Vulkan upload device.
            events.append("upload")
            worker._stop.set()
            if upload_error:
                raise RuntimeError("upload failed")

        def set_data_array(self, *_args):
            pytest.fail("GPU image made a CPU round trip")

    def wait(stream):
        assert current_device[0] == 0 and stream == mock_cuda.cudaStreamLegacy
        assert box._in_use is not None and not maps[0].closed
        events.append("complete")
        return (int(wait_error),)

    monkeypatch.setattr(mock_cuda, "cudaStreamSynchronize", wait)

    ui = SimpleNamespace(
        init=lambda *_a, **_k: None,
        shutdown=shutdown,
        ByteImageProvider=Provider,
        Window=Context,
        Frame=Context,
        ZStack=Context,
        VStack=Context,
        Label=lambda text, **_k: SimpleNamespace(text=text),
        Rectangle=lambda **_k: None,
        ImageWithProvider=lambda *_a, **_k: SimpleNamespace(visible=False),
        WINDOW_FLAGS_NO_TITLE_BAR=1,
        RasterPolicy=SimpleNamespace(NEVER=1),
        IwpFillPolicy=SimpleNamespace(IWP_PRESERVE_ASPECT_FIT=1),
        TextureFormat=SimpleNamespace(RGBA8_UNORM=1),
    )
    mock_omni_ui(ui)
    worker._run()
    assert (
        events == ["upload", "complete", "destroy", "shutdown"] and worker.status.closed
    )
    if wait_error:
        assert (
            box._in_use is not None
            and "input-copy completion failed" in worker.status.error
        )
        with pytest.raises(RuntimeError, match="unfinished"):
            box.close()
        assert not maps[0].closed
    else:
        assert box._in_use is None
        assert worker.status.error == ("upload failed" if upload_error else None)
        box.close()
        assert maps[0].closed


def test_retirement_failure_is_contained_and_stops_mapping():
    box, maps = mailbox()
    publish(box, 1)

    def fail():
        raise RuntimeError("unmap failed")

    maps[0].close = fail
    publish(box, 2)
    assert box.snapshot() is None and "unmap failed" in box.view()[4]
    publish(box, 3)
    assert len(maps) == 1 and box._mapped == maps
    with pytest.raises(RuntimeError, match="unmap failed"):
        box.close()


def test_failed_map_cleanup_is_retained_and_prevents_additional_maps():
    box, maps = mailbox()

    def fail(*_args):
        image = FakeMap(1)
        maps.append(image)
        raise mirror.OutputMapCleanupError(image, "invalid tensor", "unmap failed")

    box._gpu_mapper = fail
    publish(box, 1)
    publish(box, 2)
    assert len(maps) == 1 and box._mapped == maps
    assert box.snapshot() is None and "cleanup failed" in box.view()[4]
    box.close()
    assert maps[0].closed


@pytest.mark.parametrize("stream,event", [(0, 7), (5, 0), (1, 7), (0, 0)])
def test_producer_event_and_stream_are_both_synchronized(stream, event):
    from isaaccapture.ovxr_session_manager.gpu_output import _wait_ready

    calls = []
    cuda = SimpleNamespace(
        cudaStreamLegacy=999,
        cudaEventSynchronize=lambda value: (calls.append(("event", value)) or 0,),
        cudaStreamSynchronize=lambda value: (calls.append(("stream", value)) or 0,),
    )
    _wait_ready(cuda, SimpleNamespace(stream=stream, wait_event=event))
    expected = ([("event", event)] if event else []) + (
        [("stream", 999 if stream == 1 else stream)] if stream else []
    )
    assert calls == expected


@pytest.mark.parametrize("copy_failure", [False, True])
@pytest.mark.parametrize("source_device", [0, 1])
def test_owned_mirror_copy_releases_frame_maps_and_reuses_two_allocations(
    monkeypatch, copy_failure, source_device, mock_cuda
):
    cuda = mock_cuda
    from isaaccapture.ovxr_session_manager import gpu_output as gpu

    events = []
    allocations = []
    sources = []

    def mapped(*args):
        source = SimpleNamespace(
            width=2,
            height=3,
            device=source_device,
            pointer=42,
            busy=False,
            closed=False,
        )

        def close():
            assert not source.busy
            source.closed = True
            events.append("unmap")

        source.close = close
        sources.append(source)
        return source

    def allocate(*args):
        allocations.append(len(allocations) + 100)
        return 0, allocations[-1], 8

    monkeypatch.setattr(gpu, "map_cuda_array_rgba", mapped)
    monkeypatch.setattr(cuda, "cudaSetDevice", lambda device: (0,))
    monkeypatch.setattr(cuda, "cudaMallocPitch", allocate)
    monkeypatch.setattr(cuda, "cudaFree", lambda pointer: (events.append("free") or 0,))
    monkeypatch.setattr(
        cuda,
        "cudaMemcpy2DFromArrayAsync",
        lambda *args: (events.append("copy") or int(copy_failure),),
    )

    monkeypatch.setattr(
        cuda, "cudaStreamSynchronize", lambda *args: (events.append("complete") or 0,)
    )
    pool = gpu.OwnedCudaImagePool()
    pool.enable()
    if copy_failure:
        with pytest.raises(RuntimeError, match="CUDA"):
            pool(1, 2, 100)
        assert sources[-1].closed
        pool.finish_frame()
    else:
        first = pool(1, 2, 100)
        assert first.device == source_device
        second = pool(1, 3, 100)
        assert first.pointer != second.pointer and all(
            source.closed for source in sources
        )
        assert events[:3] == ["copy", "complete", "unmap"]
        with pytest.raises(RuntimeError, match="both owned"):
            pool(1, 4, 100)
        first.close()
        replacement = pool(1, 5, 100)
        assert replacement.pointer == first.pointer and len(allocations) == 2
        replacement.close()
        second.close()
    pool.close()
    assert events.count("free") == len(allocations)


def test_oversized_pitched_allocation_stays_rejected_on_reuse(monkeypatch, mock_cuda):
    cuda = mock_cuda
    from isaaccapture.ovxr_session_manager import gpu_output as gpu

    events = []
    source = SimpleNamespace(width=1, height=2, device=0, pointer=42, busy=False)
    source.close = lambda: events.append("unmap")
    monkeypatch.setattr(gpu, "map_cuda_array_rgba", lambda *args: source)
    monkeypatch.setattr(cuda, "cudaSetDevice", lambda *args: (0,))
    monkeypatch.setattr(cuda, "cudaMallocPitch", lambda *args: (0, 100, 256))
    monkeypatch.setattr(cuda, "cudaFree", lambda *args: (events.append("free") or 0,))
    monkeypatch.setattr(
        cuda, "cudaMemcpy2DFromArrayAsync", lambda *args: pytest.fail("oversized copy")
    )
    pool = gpu.OwnedCudaImagePool()
    pool.enable()
    for handle in (1, 2):
        with pytest.raises(RuntimeError, match="byte bound"):
            pool(1, handle, 100)
        pool.finish_frame()
    pool.close()
    assert events == ["unmap", "unmap", "free"]


def test_unproved_owned_copy_keeps_borrowed_source_and_blocks_consumption(
    monkeypatch, mock_cuda
):
    cuda = mock_cuda
    from isaaccapture.ovxr_session_manager import gpu_output as gpu

    source = SimpleNamespace(width=1, height=1, device=0, pointer=42, busy=False)
    source.close = lambda: pytest.fail("unproved source read was unmapped")
    monkeypatch.setattr(gpu, "map_cuda_array_rgba", lambda *args: source)
    monkeypatch.setattr(cuda, "cudaSetDevice", lambda *args: (0,))
    monkeypatch.setattr(cuda, "cudaMallocPitch", lambda *args: (0, 100, 4))
    monkeypatch.setattr(cuda, "cudaMemcpy2DFromArrayAsync", lambda *args: (0,))
    monkeypatch.setattr(cuda, "cudaStreamSynchronize", lambda *args: (1,))
    pool = gpu.OwnedCudaImagePool()
    pool.enable()
    with pytest.raises(RuntimeError, match="CUDA"):
        pool(1, 2, 100)
    assert source.busy
    with pytest.raises(RuntimeError, match="borrowed frame"):
        pool.finish_frame()
    with pytest.raises(RuntimeError, match="borrowed frame"):
        pool.close()


def test_unconfigured_pool_needs_no_optional_providers(monkeypatch):
    from isaaccapture.ovxr_session_manager.gpu_output import OwnedCudaImagePool

    for name in ("cuda", "cuda.bindings", "cuda.bindings.runtime", "ovxr"):
        monkeypatch.setitem(sys.modules, name, None)
    pool = OwnedCudaImagePool()
    assert pool(1, 2, 100) is None
    pool.close()


def test_empty_pool_still_retains_unproved_borrowed_sources(monkeypatch):
    from isaaccapture.ovxr_session_manager.gpu_output import OwnedCudaImagePool

    monkeypatch.setitem(sys.modules, "cuda", None)
    pool = OwnedCudaImagePool()
    borrowed = object()
    pool._source = borrowed
    with pytest.raises(RuntimeError, match="retain"):
        pool.close()
    assert pool._source is borrowed
