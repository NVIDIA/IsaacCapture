# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from isaaccapture.ovxr_session_manager.desktop_mirror import (
    DesktopMirrorConfig,
    DesktopMirrorStatus,
)
import isaaccapture.ovxr_session_manager.owned_mirror as mirror


def _prepared_mirror(*, render_products=(), auxiliary_products=()):
    owner = mirror.OwnedDesktopMirror(DesktopMirrorConfig())
    binding = SimpleNamespace(
        selected_auxiliary_paths=tuple(path for path, _ in auxiliary_products)
    )
    config = SimpleNamespace(
        render_products=render_products,
        selected_auxiliary_rasters=tuple(
            SimpleNamespace(label=label) for _, label in auxiliary_products
        ),
    )
    owner.on_session_started(binding, config, lambda: None)
    return owner


@pytest.mark.parametrize(
    "eye_paths",
    [(), ("/Render/CustomLeft", "/Render/CustomRight")],
)
def test_mirror_menu_includes_eyes_and_camera_labels(monkeypatch, eye_paths):
    captured = []

    def create_worker(_acquire, _config, _on_close, *, products):
        captured.extend(products.view()[0])
        return SimpleNamespace(status=DesktopMirrorStatus(), close=lambda: None)

    monkeypatch.setattr(mirror, "OwnedDesktopMirrorWorker", create_worker)
    owner = _prepared_mirror(
        render_products=eye_paths,
        auxiliary_products=(("/Camera/Head", "Head camera"), ("/Camera/Wrist", None)),
    )
    try:
        expected_eyes = eye_paths or (
            "/OVXR/Render/Products/Left",
            "/OVXR/Render/Products/Right",
        )
        assert [(product.path, product.label) for product in captured] == [
            (expected_eyes[0], "Left eye"),
            (expected_eyes[1], "Right eye"),
            ("/Camera/Head", "Head camera"),
            ("/Camera/Wrist", "Wrist"),
        ]
    finally:
        owner.close()


@pytest.mark.parametrize("failure_site", ["ui", "images"])
def test_mirror_close_preserves_failed_retirement(monkeypatch, failure_site):
    events = []

    def close_at(site):
        events.append(site)
        if site == failure_site:
            raise RuntimeError(f"{site} retirement failed")

    worker = SimpleNamespace(
        status=DesktopMirrorStatus(closed=True), close=lambda: close_at("ui")
    )
    monkeypatch.setattr(mirror, "OwnedDesktopMirrorWorker", lambda *a, **kw: worker)
    owner = _prepared_mirror(render_products=(), auxiliary_products=())
    monkeypatch.setattr(owner._mailbox, "close", lambda: close_at("images"))
    with pytest.raises(RuntimeError, match=f"{failure_site} retirement failed"):
        owner.close()
    assert events == (["ui"] if failure_site == "ui" else ["ui", "images"])
    assert owner.status == worker.status


def test_mirror_start_failure_is_optional_and_attempted_once(monkeypatch):
    worker = Mock(status=DesktopMirrorStatus())
    worker.start.side_effect = RuntimeError("UI thread unavailable")
    monkeypatch.setattr(mirror, "OwnedDesktopMirrorWorker", lambda *a, **kw: worker)
    owner = _prepared_mirror(render_products=(), auxiliary_products=())
    try:
        owner.on_frame_submitted()
        owner.on_frame_submitted()
        worker.start.assert_called_once_with()
        worker.record_render_frame.assert_not_called()
    finally:
        owner.close()


def test_optional_consumer_error_is_visible_without_invalidating_owned_image():
    mailbox = mirror.OwnedMirrorMailbox(30)
    image = mirror.OwnedMirrorFrame(1, 1, 1, b"\0" * 4, 1)
    mailbox._latest = image
    consumer = mailbox.acquire(0)
    mailbox.report_error(RuntimeError("optional upload failed"))
    assert mailbox.view()[-1] == "optional upload failed"
    assert mailbox.snapshot() is image and consumer is image
    mailbox.release(consumer)
    mailbox.close()


@pytest.mark.parametrize(
    "configured_backend,expected_backend",
    [(None, "opengl"), ("opengl", "opengl"), ("vulkan", "vulkan")],
)
def test_missing_optional_ui_records_failure_and_closes_worker(
    monkeypatch, mock_omni_ui, configured_backend, expected_backend
):
    # Register restoration even when the original environment has no selection.
    monkeypatch.setenv("OMNIUI_BACKEND", configured_backend or "")
    if configured_backend is None:
        monkeypatch.delenv("OMNIUI_BACKEND")
    mock_omni_ui(None)
    callbacks = []
    worker = mirror.OwnedDesktopMirrorWorker(
        lambda: None, DesktopMirrorConfig(), lambda: callbacks.append("user-close")
    )
    assert os.environ.get("OMNIUI_BACKEND") == configured_backend
    worker._run()
    assert os.environ["OMNIUI_BACKEND"] == expected_backend
    assert worker.status.closed and "omni.ui" in worker.status.error
    assert callbacks == []


@pytest.mark.parametrize(
    "width,height,rgba,expected_error",
    [
        (1, 1, b"\0" * 4, "owned upload failed"),
        (1, 1, b"\0" * 3, "mirror snapshot is not tightly packed RGBA8"),
        (1, 1, b"\0" * 8, "mirror snapshot is not tightly packed RGBA8"),
        (0, 1, b"", "mirror snapshot is not tightly packed RGBA8"),
        (1, -1, b"", "mirror snapshot is not tightly packed RGBA8"),
    ],
)
def test_mirror_async_upload_failure_is_not_reported_as_user_close(
    monkeypatch, width, height, rgba, expected_error, mock_omni_ui
):
    class Context:
        frame = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def destroy(self):
            pass

    Context.frame = Context()

    class RaisingProvider:
        def set_data_array(self, pixels, sizes):
            assert isinstance(pixels, np.ndarray)
            assert pixels.dtype == np.uint8 and pixels.flags.c_contiguous
            assert pixels.ndim == 1 and pixels.size == width * height * 4
            assert pixels.base is rgba and not pixels.flags.writeable
            assert sizes == [width, height]
            raise RuntimeError("owned upload failed")

    ui = SimpleNamespace(
        init=lambda *_args, **_kwargs: None,
        shutdown=lambda: None,
        ByteImageProvider=RaisingProvider,
        Window=lambda *_args, **_kwargs: Context(),
        ZStack=Context,
        VStack=Context,
        Label=lambda text, **_kwargs: SimpleNamespace(text=text),
        Rectangle=lambda **_kwargs: None,
        ImageWithProvider=lambda *_args, **_kwargs: SimpleNamespace(visible=True),
        WINDOW_FLAGS_NO_TITLE_BAR=1,
        RasterPolicy=SimpleNamespace(NEVER=1),
        IwpFillPolicy=SimpleNamespace(IWP_PRESERVE_ASPECT_FIT=1),
        TextureFormat=SimpleNamespace(RGBA8_UNORM=1),
    )
    mock_omni_ui(ui)
    callbacks = []
    image = mirror.OwnedMirrorFrame(1, width, height, rgba, 1)
    worker = mirror.OwnedDesktopMirrorWorker(
        lambda: image, DesktopMirrorConfig(), lambda: callbacks.append("user-close")
    )
    worker._run()
    assert worker.status.closed
    assert worker.status.error == expected_error
    assert worker.status.presented_frames == 0
    assert callbacks == []


@pytest.mark.parametrize(
    "invalidate_during_upload,report_after_present,replacement_gap",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (False, False, "upload"),
        (False, False, "after"),
    ],
)
def test_inprocess_slow_upload_keeps_latest_snapshot_and_closes(
    monkeypatch,
    invalidate_during_upload,
    report_after_present,
    replacement_gap,
    mock_omni_ui,
):
    entered, release, consumed_latest = (threading.Event() for _ in range(3))
    uploaded = []
    image_widget = SimpleNamespace(visible=False)
    labels = []

    class Label:
        def __init__(self, text, **_kwargs):
            self.owner = threading.get_native_id()
            self.text = text
            labels.append(self)

        @property
        def text(self):
            return self._text

        @text.setter
        def text(self, value):
            assert threading.get_native_id() == self.owner
            self._text = value

    class Context:
        def __init__(self, **_kwargs):
            pass

        def set_build_fn(self, fn):
            pass

        def rebuild(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def destroy(self):
            pass

    Context.frame = Context()

    class Provider:
        def set_data_array(self, pixels, sizes):
            assert isinstance(pixels, np.ndarray)
            assert pixels.dtype == np.uint8 and pixels.flags.c_contiguous
            assert pixels.ndim == 1 and pixels.size == 4 and sizes == [1, 1]
            assert isinstance(pixels.base, bytes) and not pixels.flags.writeable
            # Keep the supplied buffer alive while the producer replaces its
            # latest snapshot. No mailbox lock may cover this slow consumer.
            if not uploaded:
                entered.set()
                assert release.wait(2)
            uploaded.append(pixels.tobytes()[0])
            if uploaded[-1] == 3 and replacement_gap == "upload":
                publish(4)
                mailbox._drain_gpu(replace_latest=True)
                assert mailbox.snapshot() is None
            if uploaded[-1] == 3 or invalidate_during_upload:
                consumed_latest.set()

    ui = SimpleNamespace(
        init=lambda *_args, **_kwargs: None,
        shutdown=lambda: None,
        ByteImageProvider=Provider,
        Window=lambda *_args, **_kwargs: Context(),
        Frame=Context,
        ZStack=Context,
        VStack=Context,
        Label=Label,
        Rectangle=lambda **_kwargs: None,
        ImageWithProvider=lambda *_args, **_kwargs: image_widget,
        WINDOW_FLAGS_NO_TITLE_BAR=1,
        RasterPolicy=SimpleNamespace(NEVER=1),
        IwpFillPolicy=SimpleNamespace(IWP_PRESERVE_ASPECT_FIT=1),
        TextureFormat=SimpleNamespace(RGBA8_UNORM=1),
    )
    mock_omni_ui(ui)
    from test_camera_mailbox import _outputs, _frame

    mailbox = mirror.OwnedMirrorMailbox(
        30, copier=lambda _r, h, _m: (1, 1, bytes([h]) * 4)
    )
    mailbox.set_products((mirror.MirrorProduct("/Camera", "Camera"),))

    def publish(sequence):
        mailbox._next_ns = 0  # exercise ownership, not the independent rate cap
        mailbox.on_outputs(1, _outputs(_frame(sequence, sequence)))

    publish(1)
    callbacks = []
    worker = mirror.OwnedDesktopMirrorWorker(
        mailbox.snapshot,
        DesktopMirrorConfig(),
        lambda: callbacks.append("user-close"),
        products=mailbox
        if invalidate_during_upload or report_after_present or replacement_gap
        else None,
    )
    worker.set_status_text("before")
    worker.start()
    try:
        assert entered.wait(2)
        worker.set_status_text("after: Physics 0/120 Hz (0.00× RT)")
        if invalidate_during_upload:
            mailbox._next_ns = 0
            mailbox._copier = lambda *_: (1, 1, b"bad")
            mailbox.on_outputs(1, _outputs(_frame(2, 2)))
            assert mailbox.snapshot() is None
        else:
            publish(2)
            publish(3)
            assert mailbox.snapshot().sequence == 3
        release.set()
        assert consumed_latest.wait(2)
        deadline = time.monotonic() + 2
        while labels[0].text != "after: Physics 0/120 Hz (0.00× RT)":
            assert time.monotonic() < deadline
            time.sleep(0.005)
        if replacement_gap:
            deadline = time.monotonic() + 2
            while worker.status.last_sequence != 3:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            if replacement_gap == "after":
                mailbox._drain_gpu(replace_latest=True)
            assert mailbox.snapshot() is None
            time.sleep(0.1)  # Observe several UI polls during the producer gap.
            assert image_widget.visible
        if report_after_present:
            assert image_widget.visible and worker.status.last_sequence == 3
            mailbox.report_error(RuntimeError("optional readback failed"))
            deadline = time.monotonic() + 2
            while labels[-1].text != "optional readback failed":
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert not image_widget.visible
            assert mailbox.snapshot().sequence == 3

    finally:
        release.set()
        worker.close(2)
    assert uploaded == ([1] if invalidate_during_upload else [1, 3])
    assert worker.status.closed and worker.status.error is None
    assert worker.status.last_sequence == (0 if invalidate_during_upload else 3)
    if invalidate_during_upload:
        assert not image_widget.visible
        assert worker.status.presented_frames == 0
    assert callbacks == []


@pytest.mark.parametrize(
    "run_error,callback_raises", [(False, False), (False, True), (True, False)]
)
def test_ui_exit_finishes_async_tasks_before_cleanup(
    monkeypatch, run_error, callback_raises, mock_omni_ui
):
    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def destroy(self):
            assert loops
            assert not any(asyncio.all_tasks(loop) for loop in loops)

    Context.frame = Context()
    loops = []

    async def run_async():
        loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        if run_error:
            raise RuntimeError("native mirror failed")

    ui = SimpleNamespace(
        init=lambda *_args, **_kwargs: None,
        shutdown=lambda: None,
        run_async=run_async,
        ByteImageProvider=lambda: None,
        Window=lambda *_args, **_kwargs: Context(),
        ZStack=Context,
        VStack=Context,
        Label=lambda text, **_kwargs: SimpleNamespace(text=text),
        Rectangle=lambda **_kwargs: None,
        ImageWithProvider=lambda *_args, **_kwargs: SimpleNamespace(visible=True),
        WINDOW_FLAGS_NO_TITLE_BAR=1,
        RasterPolicy=SimpleNamespace(NEVER=1),
        IwpFillPolicy=SimpleNamespace(IWP_PRESERVE_ASPECT_FIT=1),
        TextureFormat=SimpleNamespace(RGBA8_UNORM=1),
    )
    mock_omni_ui(ui)
    callbacks = []

    def on_close():
        callbacks.append("user-close")
        if callback_raises:
            raise RuntimeError("application close failed")

    worker = mirror.OwnedDesktopMirrorWorker(
        lambda: None, DesktopMirrorConfig(), on_close
    )
    worker._run()
    assert worker.status.closed
    assert callbacks == ([] if run_error else ["user-close"])
    assert worker.status.error == (
        "native mirror failed"
        if run_error
        else "application close failed"
        if callback_raises
        else None
    )


def test_rates_measure_host_and_submitted_render_events_not_gui_uploads(monkeypatch):
    now = [0]
    monkeypatch.setattr(mirror.time, "perf_counter_ns", lambda: now[0])
    worker = mirror.OwnedDesktopMirrorWorker(lambda: None, DesktopMirrorConfig())
    for n in range(21):
        now[0] = n * 50_000_000
        worker.record_host_frame()
        if n % 2 == 0:
            worker.record_render_frame()
    worker.set_status_text("Physics 120 Hz / 120 Hz  1.00x")
    assert worker.status.main_thread_fps == pytest.approx(20)
    assert worker.status.render_thread_fps == pytest.approx(10)
    assert worker.status.presented_frames == 0
    assert worker.status.status_text == "Physics 120 Hz / 120 Hz  1.00x"
    now[0] += 1_000_000_001
    assert worker.status.main_thread_fps == worker.status.render_thread_fps == 0
    assert worker.status.status_text == "Physics 120 Hz / 120 Hz  1.00x"
    worker.close()


@pytest.mark.parametrize("level, expected_records", [("WARNING", 0), ("INFO", 1)])
def test_mirror_retirement_respects_named_logger_level(
    monkeypatch, caplog, capsys, level, expected_records
):
    worker = SimpleNamespace(
        status=DesktopMirrorStatus(closed=True), close=lambda: None
    )
    monkeypatch.setattr(
        mirror, "OwnedDesktopMirrorWorker", lambda *args, **kwargs: worker
    )
    with caplog.at_level(level, logger=mirror.__name__):
        owner = _prepared_mirror(render_products=(), auxiliary_products=())
        owner.close()
    records = [
        record
        for record in caplog.records
        if record.name == mirror.__name__
        and record.getMessage().startswith("OVUI desktop mirror stopped:")
    ]
    assert len(records) == expected_records
    if level == "WARNING":
        assert "OVUI desktop mirror stopped:" not in capsys.readouterr().out
