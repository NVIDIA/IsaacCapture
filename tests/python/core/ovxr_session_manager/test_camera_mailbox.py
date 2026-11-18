# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure callback lifetime and output-selection tests for the borrowed camera step."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaccapture.ovxr_session_manager import camera_mailbox
from isaaccapture.ovxr_session_manager.camera_mailbox import FrameOutputs

from isaaccapture.ovxr_session_manager.camera_mailbox import CompletedCameraMailbox


def test_cpu_unmap_failure_retains_handle_and_blocks_more_maps(monkeypatch):
    """Exercise the real CPU copier; failed unmap must not lose its native lease."""
    b = pytest.importorskip("ovrtx._src.bindings")
    from isaaccapture.ovxr_session_manager import gpu_output

    gpu_output._api.cache_clear()

    def map_output(_renderer, _handle, _description, _timeout, output):
        mapped = ctypes.cast(
            output, ctypes.POINTER(b.ovrtx_render_var_output_t)
        ).contents
        mapped.map_handle = 71
        mapped.num_tensors = 0  # Validation fails after a successful native map.
        mapped.status = b.EventStatus.COMPLETED
        return SimpleNamespace(status=b.OVRTX_API_SUCCESS)

    native = SimpleNamespace(
        ovrtx_map_render_var_output=Mock(side_effect=map_output),
        ovrtx_unmap_render_var_output=Mock(
            side_effect=[
                SimpleNamespace(status=1),
                SimpleNamespace(status=b.OVRTX_API_SUCCESS),
            ]
        ),
        ovrtx_fetch_results=Mock(),
    )
    monkeypatch.setattr(b, "ovrtx_loaded_bin_dir", lambda: "/declared/provider")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: native)
    try:
        mailbox = CompletedCameraMailbox("/Camera", 4)
        mailbox.on_outputs(FrameOutputs(1, _outputs(_frame(1.0, 1))))
        assert mailbox.snapshot() is None and mailbox.error() is not None
        mailbox.on_outputs(FrameOutputs(1, _outputs(_frame(2.0, 2))))
        assert native.ovrtx_map_render_var_output.call_count == 1
        mailbox.close()
        assert native.ovrtx_unmap_render_var_output.call_count == 2
        assert [
            call.args[1] for call in native.ovrtx_unmap_render_var_output.call_args_list
        ] == [71, 71]
        mailbox.close()
        assert native.ovrtx_unmap_render_var_output.call_count == 2
    finally:
        gpu_output._api.cache_clear()


def _string(text):
    encoded = text.encode()
    return SimpleNamespace(ptr=ctypes.c_char_p(encoded), length=len(encoded))


def _frame(time, handle, *, color=True):
    vars_ = [
        SimpleNamespace(
            source_name=_string("LdrColor" if color else "XRDepth"),
            output_handle=handle,
        )
    ]
    return SimpleNamespace(
        frame_end_time=time, render_var_count=len(vars_), output_render_vars=vars_
    )


def _outputs(*frames):
    product = SimpleNamespace(
        render_product_path=_string("/Camera"),
        output_frame_count=len(frames),
        output_frames=list(frames),
    )
    return SimpleNamespace(contents=SimpleNamespace(output_count=1, outputs=[product]))


def test_newest_completed_image_and_zero_output_retention(monkeypatch):
    seen = []

    def copy(_renderer, handle, _limit):
        seen.append(handle)
        return 1, 1, bytes((handle, 2, 3, 255))

    monkeypatch.setattr(camera_mailbox, "_copy_host_rgba", copy)
    mailbox = CompletedCameraMailbox("/Camera", 4)
    mailbox.on_outputs(FrameOutputs(1, _outputs(_frame(3.0, 3), _frame(4.0, 4))))
    image = mailbox.snapshot()
    assert image is not None
    assert (image.sensor_time, image.revision, image.rgba) == (
        4.0,
        1,
        b"\x04\x02\x03\xff",
    )
    mailbox.on_outputs(FrameOutputs(1, _outputs()))
    assert mailbox.snapshot() is image
    assert seen == [4]


def test_produced_but_invalid_image_is_omitted_and_error_is_contained(monkeypatch):
    monkeypatch.setattr(camera_mailbox, "_copy_host_rgba", lambda *_: (1, 1, b"rgba"))
    mailbox = CompletedCameraMailbox("/Camera", 4)
    mailbox.on_outputs(FrameOutputs(1, _outputs(_frame(1.0, 1))))
    assert mailbox.snapshot() is not None
    mailbox.on_outputs(FrameOutputs(1, _outputs(_frame(2.0, 2, color=False))))
    assert mailbox.snapshot() is None
    assert "without LdrColor" in mailbox.error()


def test_mapping_failure_is_contained_without_reusing_stale_pixels(monkeypatch):
    def fail(*_args):
        raise RuntimeError("map failed")

    monkeypatch.setattr(camera_mailbox, "_copy_host_rgba", fail)
    mailbox = CompletedCameraMailbox("/Camera", 4)
    mailbox.on_outputs(FrameOutputs(1, _outputs(_frame(1.0, 1))))
    assert mailbox.snapshot() is None
    assert mailbox.error() == "map failed"


def test_gpu_camera_retains_cadence_gaps_and_retires_on_owner():
    import threading

    class FakeMap:
        width = height = 1

        def __init__(self, handle):
            self.pointer = handle
            self.closed = False
            self.owner = threading.get_ident()

        def close(self):
            assert threading.get_ident() == self.owner
            assert not self.closed
            self.closed = True

    def outputs(**handles):
        return _outputs(*[_frame(float(h), h) for h in handles.values()])

    made = []

    def mapper(_renderer, handle, _limit):
        image = FakeMap(handle)
        made.append(image)
        return image

    mailbox = CompletedCameraMailbox("/Render/Left", 64, gpu_mapper=mapper)
    mailbox.product_path = "/Camera"
    mailbox.on_outputs(FrameOutputs(1, outputs(Left=1)))
    first = mailbox.snapshot()
    assert first.gpu is made[0] and first.rgba is None
    mailbox.on_outputs(FrameOutputs(1, outputs()))
    assert mailbox.snapshot() is first and not made[0].closed
    mailbox.on_outputs(FrameOutputs(1, outputs(Left=2)))
    assert made[0].closed and mailbox.snapshot().gpu is made[1]
    mailbox.close()
    assert made[1].closed
