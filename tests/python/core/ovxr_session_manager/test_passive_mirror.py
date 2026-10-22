# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest

from isaaccapture.ovxr_session_manager.camera_mailbox import CameraImage
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


def mailbox():
    calls = []

    def copy(_renderer, handle, _limit):
        calls.append(handle)
        return 1, 1, bytes([handle]) * 4

    m = OwnedMirrorMailbox(30, copier=copy)
    m.set_products((LEFT, RIGHT, CAMERA))
    return m, calls


def test_selection_only_copies_selected_output_without_changing_production_set():
    m, calls = mailbox()
    source = outputs(Left=1, Right=2, Camera=3)
    m.on_outputs(1, source)
    assert calls == [1] and m.snapshot().product == LEFT
    m.select(RIGHT)
    assert m.snapshot() is None
    m.on_outputs(1, source)
    assert calls == [1, 2] and m.snapshot().rgba == b"\2" * 4
    assert m.view()[0] == (LEFT, RIGHT, CAMERA)
    assert [p.render_product_path.length for p in source.contents.outputs] == [5, 6, 7]


def test_camera_reuses_independent_pip_bytes_and_cadence_gap_retains_choice():
    m, calls = mailbox()
    m.select(CAMERA)
    image = CameraImage(1, 1, b"RGBA", 3.0, 4)
    m.on_outputs(1, outputs(Left=1, Right=2, Camera=3), {CAMERA.path: image})
    held = m.snapshot()
    assert held.rgba is image.rgba and calls == []
    m._next_ns = 0
    m.on_outputs(1, outputs(Left=5, Right=6, Camera=None), {CAMERA.path: image})
    assert m.snapshot() is held and m.view()[0] == (LEFT, RIGHT, CAMERA)
    assert m.counters() == {"mirror_copies": 0, "shared_camera_reuses": 1}


@pytest.mark.parametrize("change", ["selection", "remove", "generation", "reset"])
def test_inflight_copy_is_discarded_after_identity_change(change):
    entered, release = threading.Event(), threading.Event()

    def copy(*_):
        entered.set()
        assert release.wait(2)
        return 1, 1, b"old!"

    m = OwnedMirrorMailbox(30, copier=copy)
    m.set_products((LEFT, RIGHT))
    worker = threading.Thread(target=m.on_outputs, args=(1, outputs(Left=1, Right=2)))
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


def test_add_remove_is_not_arrival_driven_and_stale_dropdown_event_is_ignored():
    m, _calls = mailbox()
    m.on_outputs(1, outputs(Left=1))
    held = m.snapshot()
    m.set_products((LEFT, RIGHT))
    assert m.snapshot() is held
    m.select(CAMERA)
    assert m.view()[1] == LEFT
    m.set_products((RIGHT,))
    assert m.view()[1] == RIGHT and m.snapshot() is None
    m.set_products(())
    assert m.view()[1] is None


def test_invalid_selected_image_clears_stale_pixels_and_error_is_contained():
    m, _calls = mailbox()
    m.on_outputs(1, outputs(Left=1))
    m._next_ns = 0
    m._copier = lambda *_: (1, 1, b"bad")
    m.on_outputs(1, outputs(Left=2))
    assert m.snapshot() is None and "RGBA8" in m.view()[4]


def test_mirror_cadence_does_not_copy_another_output_or_reset_on_gap():
    m, calls = mailbox()
    m.on_outputs(1, outputs(Left=1, Right=2))
    m.on_outputs(1, outputs(Left=3, Right=4))
    assert calls == [1]
    m._next_ns = 0
    m.on_outputs(1, outputs(Left=None, Right=4))
    assert calls == [1] and m.snapshot().rgba == b"\1" * 4


def test_product_paths_must_be_unique_and_absolute():
    m, _ = mailbox()
    with pytest.raises(ValueError):
        m.set_products((LEFT, LEFT))
    with pytest.raises(ValueError):
        m.set_products((MirrorProduct("relative", "bad"),))
