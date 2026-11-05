# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest

from isaaccapture.ovxr_session_manager import camera_mailbox
from isaaccapture.ovxr_session_manager.camera_mailbox import FrameOutputs

from isaaccapture.ovxr_session_manager.owned_mirror import (
    MirrorProduct,
    OwnedMirrorMailbox,
)
from test_camera_mailbox import _frame, _string

LEFT = MirrorProduct("/Left", "Left eye")
RIGHT = MirrorProduct("/Right", "Right eye")
CAMERA = MirrorProduct("/Camera", "Camera")


def outputs(**handles):
    products = [
        SimpleNamespace(
            render_product_path=_string("/" + path),
            output_frame_count=int(handle is not None),
            output_frames=[] if handle is None else [_frame(float(handle), handle)],
        )
        for path, handle in handles.items()
    ]
    return SimpleNamespace(
        contents=SimpleNamespace(output_count=len(products), outputs=products)
    )


def mailbox(monkeypatch):
    calls = []

    def copy(_renderer, handle, _limit):
        calls.append(handle)
        return 1, 1, bytes([handle]) * 4

    monkeypatch.setattr(camera_mailbox, "_copy_host_rgba", copy)
    m = OwnedMirrorMailbox(30)
    m.set_products((LEFT, RIGHT, CAMERA))
    return m, calls


def test_selection_only_copies_selected_output_without_changing_production_set(
    monkeypatch,
):
    m, calls = mailbox(monkeypatch)
    source = outputs(Left=1, Right=2, Camera=3)
    m.on_outputs(FrameOutputs(1, source))
    assert calls == [1] and m.snapshot().product == LEFT
    m.select(RIGHT)
    assert m.snapshot() is None
    m.on_outputs(FrameOutputs(1, source))
    assert calls == [1, 2] and m.snapshot().rgba == b"\2" * 4
    assert m.view()[0] == (LEFT, RIGHT, CAMERA)
    assert [p.render_product_path.length for p in source.contents.outputs] == [5, 6, 7]


@pytest.mark.parametrize("mirror_first", [False, True])
def test_camera_cpu_copy_is_shared_in_either_consumer_order(monkeypatch, mirror_first):
    from isaaccapture.ovxr_session_manager.camera_mailbox import (
        CompletedCameraMailbox,
    )

    m, calls = mailbox(monkeypatch)
    m.select(CAMERA)
    copies = []
    monkeypatch.setattr(
        camera_mailbox,
        "_copy_host_rgba",
        lambda *_args: copies.append(1) or (1, 1, b"RGBA"),
    )
    source = outputs(Left=1, Right=2, Camera=3)
    frame = FrameOutputs(1, source)
    camera = CompletedCameraMailbox(CAMERA.path, 4)
    operations = [
        lambda: camera.on_outputs(frame),
        lambda: m.on_outputs(frame),
    ]
    for operation in operations[:: (-1 if mirror_first else 1)]:
        operation()
    held = m.snapshot()
    assert held.rgba is camera.snapshot().rgba and copies == [1] and calls == []
    bounded_camera = CompletedCameraMailbox(CAMERA.path, 3)
    bounded_camera.on_outputs(frame)
    assert bounded_camera.snapshot() is None
    assert "bounded mailbox size" in bounded_camera.error()
    assert m.snapshot() is held and copies == [1]
    m._next_ns = 0
    m.on_outputs(FrameOutputs(1, outputs(Left=5, Right=6, Camera=None)))
    assert m.snapshot() is held and m.view()[0] == (LEFT, RIGHT, CAMERA)


@pytest.mark.parametrize("change", ["selection", "remove", "generation", "reset"])
def test_inflight_copy_is_discarded_after_identity_change(change, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def copy(*_):
        entered.set()
        assert release.wait(2)
        return 1, 1, b"old!"

    monkeypatch.setattr(camera_mailbox, "_copy_host_rgba", copy)
    m = OwnedMirrorMailbox(30)
    m.set_products((LEFT, RIGHT))
    worker = threading.Thread(
        target=m.on_outputs, args=(FrameOutputs(1, outputs(Left=1, Right=2)),)
    )
    worker.start()
    try:
        assert entered.wait(2)
        if change == "selection":
            m.select(RIGHT)
        elif change == "remove":
            m.set_products((RIGHT,))
        elif change == "generation":
            m.set_products((MirrorProduct(LEFT.path, LEFT.label, 1), RIGHT))
        else:
            m.reset()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and m.snapshot() is None


def test_add_remove_is_not_arrival_driven_and_stale_dropdown_event_is_ignored(
    monkeypatch,
):
    m, _calls = mailbox(monkeypatch)
    m.on_outputs(FrameOutputs(1, outputs(Left=1)))
    held = m.snapshot()
    m.set_products((LEFT, RIGHT))
    assert m.snapshot() is held
    m.select(CAMERA)
    assert m.view()[1] == LEFT
    m.set_products((RIGHT,))
    assert m.view()[1] == RIGHT and m.snapshot() is None
    m.set_products(())
    assert m.view()[1] is None


def test_invalid_selected_image_clears_stale_pixels_and_error_is_contained(monkeypatch):
    m, _calls = mailbox(monkeypatch)
    m.on_outputs(FrameOutputs(1, outputs(Left=1)))
    m._next_ns = 0
    monkeypatch.setattr(camera_mailbox, "_copy_host_rgba", lambda *_: (1, 1, b"bad"))
    m.on_outputs(FrameOutputs(1, outputs(Left=2)))
    assert m.snapshot() is None and "RGBA8" in m.view()[4]


def test_mirror_cadence_does_not_copy_another_output_or_reset_on_gap(monkeypatch):
    m, calls = mailbox(monkeypatch)
    m.on_outputs(FrameOutputs(1, outputs(Left=1, Right=2)))
    m.on_outputs(FrameOutputs(1, outputs(Left=3, Right=4)))
    assert calls == [1]
    m._next_ns = 0
    m.on_outputs(FrameOutputs(1, outputs(Left=None, Right=4)))
    assert calls == [1] and m.snapshot().rgba == b"\1" * 4


def test_product_paths_must_be_unique_and_absolute(monkeypatch):
    m, _ = mailbox(monkeypatch)
    with pytest.raises(ValueError):
        m.set_products((LEFT, LEFT))
    with pytest.raises(ValueError):
        m.set_products((MirrorProduct("relative", "bad"),))
