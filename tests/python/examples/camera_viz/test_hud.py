# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HUD bitmap + wrapping. Drawing is deliberately session-free, so it can be
checked without a GPU or a headset."""

from __future__ import annotations

import numpy as np

from isaaccapture_examples.camera_viz.controls import hud
from isaaccapture_examples.camera_viz.controls.hud import (
    _BODY,
    _H,
    _TITLE,
    _W,
    _fit,
    _render,
    split_message,
)


def test_render_shape_and_opacity():
    img = _render(["shape: cylinder"])
    assert img.shape == (_H, _W, 4)
    assert img.dtype == np.uint8
    assert img.flags["C_CONTIGUOUS"]  # submit() needs a contiguous buffer
    # Fully opaque is what keeps the frame eligible for CloudXR's
    # colour-only streaming.
    assert (img[..., 3] == 255).all()


def test_render_draws_text_and_the_accent_bar():
    assert not np.array_equal(_render([""]), _render(["shape: cylinder"]))
    # Left edge is the accent, not the background.
    assert tuple(_render([""])[_H // 2, 0][:3]) == hud._ACCENT


def test_channel_order_is_rgb():
    """A red-ish background must come back red-first; a swapped buffer would
    ship silently otherwise."""
    original = hud._BG
    try:
        hud._BG = (200, 0, 0)
        px = hud._render([""])[_H // 2, _W // 2]
    finally:
        hud._BG = original
    assert tuple(int(c) for c in px) == (200, 0, 0, 255)


def test_only_two_lines_are_drawn():
    assert np.array_equal(_render(["one", "two"]), _render(["one", "two", "three"]))


def test_long_line_is_ellipsized_to_the_panel():
    fitted = _fit("x" * 500, _TITLE)
    assert fitted.endswith("...")
    assert _TITLE.getlength(fitted) <= hud._TEXT_W
    assert _fit("shape: cylinder", _TITLE) == "shape: cylinder"


def test_wrapped_lines_fit_the_font_each_will_be_drawn_in():
    """Line one renders in the larger title face; measuring it with the body
    font let it overflow and get ellipsized."""
    lines = split_message("zed radius=2.50m arc=90deg")
    assert _TITLE.getlength(lines[0]) <= hud._TEXT_W
    if len(lines) > 1:
        assert _BODY.getlength(lines[1]) <= hud._TEXT_W
    # And nothing was silently dropped.
    assert " ".join(lines) == "zed radius=2.50m arc=90deg"


def test_split_message_keeps_words_whole_and_caps_at_two_lines():
    lines = split_message(" ".join(f"cam{i}=52.0mm" for i in range(40)))
    assert len(lines) == 2
    assert all(not w.startswith("=") for line in lines for w in line.split())


def test_short_message_is_one_line():
    assert split_message("mono") == ["mono"]


def test_single_line_is_positioned_differently_from_a_pair():
    """A lone line centres instead of sitting under an empty second row."""
    assert not np.array_equal(_render(["mono"]), _render(["mono", ""]))
