# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke for camera_viz with the synthetic source.

Drives the full stack: build_local_camera → producer thread → QuadLayer
submit → VizSession render → readback. Skips cleanly without a GPU or
without cupy installed (the synthetic source is cupy-resident).
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pytest

import isaaccapture.viz as viz

from sources import build_local_camera


def _gpu_and_cupy_available() -> bool:
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        return False
    # Cheap Vulkan + CUDA probe via VizSession construction; same shape
    # as the existing tests/python/viz skipif.
    cfg = viz.VizSessionConfig()
    cfg.mode = viz.DisplayMode.kOffscreen
    cfg.window_width = 64
    cfg.window_height = 64
    s = None
    try:
        s = viz.VizSession.create(cfg)
    except RuntimeError:
        return False
    finally:
        if s is not None:
            s.destroy()
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return False
    except cp.cuda.runtime.CUDARuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gpu_and_cupy_available(), reason="no Vulkan/CUDA GPU or cupy missing"
)


# Bounded wait helper so a stuck producer fails fast instead of hanging CI.
def _wait_for_frame(source, timeout_s: float = 3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        f = source.latest()
        if f is not None:
            return f
        time.sleep(0.005)
    raise TimeoutError(f"no frame from {source.spec.name} within {timeout_s}s")


@contextmanager
def _make_session(width: int, height: int):
    cfg = viz.VizSessionConfig()
    cfg.mode = viz.DisplayMode.kOffscreen
    cfg.window_width = width
    cfg.window_height = height
    cfg.clear_color = (
        0.0,
        0.0,
        0.0,
        1.0,
    )  # black baseline so any non-black confirms content
    session = viz.VizSession.create(cfg)
    try:
        yield session
    finally:
        session.destroy()


def test_synthetic_mono_renders_non_black():
    """build_local_camera({"type": "synthetic"}) → producer publishes →
    QuadLayer.submit → render → readback. Confirms the in-tree
    synthetic source actually drives the full pipeline."""
    spec = {
        "name": "synth",
        "type": "synthetic",
        "width": 64,
        "height": 64,
        "fps": 120,  # fast so we get a publish quickly
        "hue_speed_hz": 1.0,
    }
    [source] = build_local_camera(spec)

    with _make_session(64, 64) as session, source:
        layer_cfg = viz.QuadLayerConfig()
        layer_cfg.name = "synth"
        layer_cfg.resolution = viz.Resolution(64, 64)
        layer = session.add_quad_layer(layer_cfg)

        frame = _wait_for_frame(source)
        layer.submit(frame.image)
        session.render()
        arr = np.asarray(session.readback_to_host())
        assert arr.shape == (64, 64, 4)
        # Synthetic emits a hue-cycling sinusoid; nothing should be black.
        rgb = arr[..., :3]
        assert rgb.any(), "readback is all zero — synthetic source produced nothing"
        # Sanity: at least two distinct values somewhere (it's a pattern, not a flat fill).
        assert len(np.unique(rgb)) > 1


def test_synthetic_stereo_renders_left_eye_in_offscreen():
    """Stereo QuadLayer in single-view offscreen renders the LEFT eye
    (per the documented fallback). Pairs the cache-and-emit logic in
    PairedFrameSource's nearest cousin (the synthetic stereo source's
    one-thread producer) with the stereo submit path."""
    spec = {
        "name": "synth_stereo",
        "type": "synthetic",
        "stereo": True,
        "width": 64,
        "height": 64,
        "fps": 120,
        "disparity_px": 8,
    }
    [source] = build_local_camera(spec)

    with _make_session(64, 64) as session, source:
        layer_cfg = viz.QuadLayerConfig()
        layer_cfg.name = "synth_stereo"
        layer_cfg.resolution = viz.Resolution(64, 64)
        layer_cfg.stereo = True
        layer = session.add_quad_layer(layer_cfg)

        frame = _wait_for_frame(source)
        # Stereo source must populate image_right; otherwise the
        # synthetic-stereo dispatch wiring is broken.
        assert frame.image_right is not None, (
            "stereo source did not produce a right buffer"
        )
        layer.submit(frame.image, frame.image_right)
        session.render()
        arr = np.asarray(session.readback_to_host())
        assert arr.shape == (64, 64, 4)
        assert arr[..., :3].any(), "readback all zero — stereo render produced nothing"
