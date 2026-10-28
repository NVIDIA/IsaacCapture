# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-headset readout for the controller bindings.

Televiz has no screen-space overlay layer yet, so this is an ordinary
``QuadLayer`` head-locked below the operator's eyeline, fed a text bitmap.
Opaque on purpose: an alpha-blended layer drops the frame out of CloudXR's
colour-only reconstructed streaming path, which costs bandwidth all session
to light up a few glyphs. The panel hides itself a couple of seconds after
the last change.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..placements import HeadLocked, PlacementConfig

_W, _H = 1280, 160
_PAD, _BAR = 36, 12  # left text inset, accent-bar width

_BG = (18, 19, 23)
_ACCENT = (118, 185, 0)  # NVIDIA green
_PRIMARY = (242, 244, 246)
_SECONDARY = (150, 158, 168)

_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        # No size argument: it arrived in Pillow 10.1 and the floor here is
        # 10.0. The fallback bitmap font ignores size anyway, so asking for
        # one only risks a TypeError on the version we claim to support.
        return ImageFont.load_default()


# One size for both lines: a smaller continuation line reads as a different
# message rather than the same one wrapping. Hierarchy comes from colour.
_TITLE = _font(_BOLD, 44)
_BODY = _font(_REGULAR, 44)
_TEXT_W = _W - _PAD - 24  # usable width for one line


def _render(lines: Sequence[str]) -> np.ndarray:
    """Panel bitmap as RGBA8. Session- and GPU-free, so directly testable."""
    img = Image.new("RGB", (_W, _H), _BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, _BAR, _H), fill=_ACCENT)
    # A lone line sits centred; a pair straddles the middle.
    top = 20 if len(lines) > 1 else 52
    styles = ((_TITLE, _PRIMARY, top), (_BODY, _SECONDARY, top + 62))
    for line, (font, colour, y) in zip(lines[:2], styles):
        draw.text((_PAD, y), _fit(line, font), font=font, fill=colour)
    # The layer wants RGBA8 and the panel is opaque, so tack on a solid alpha.
    return np.ascontiguousarray(
        np.dstack([np.asarray(img), np.full((_H, _W), 255, np.uint8)])
    )


def _fit(line: str, font) -> str:
    """Ellipsize rather than let a long multi-camera message run off the edge."""
    if font.getlength(line) <= _TEXT_W:
        return line
    while line and font.getlength(line + "...") > _TEXT_W:
        line = line[:-1]
    return line + "..."


def split_message(message: str, max_lines: int = 2) -> List[str]:
    """Wrap a control message at word boundaries.

    Each line is measured with the font it will actually be drawn in --
    the first is the larger title face, so measuring everything with the
    body font would let line one overflow and get ellipsized.
    """
    lines: List[str] = []
    current = ""
    for word in message.split():
        candidate = f"{current} {word}".strip()
        font = _TITLE if not lines else _BODY
        if current and font.getlength(candidate) > _TEXT_W:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
        else:
            current = candidate
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines[:max_lines]


class Hud:
    """One head-locked panel. Not thread-safe: the render thread owns it."""

    def __init__(
        self,
        session,
        *,
        distance_m: float = 1.1,
        offset_y_m: float = -0.32,
        width_m: float = 0.9,
        hold_s: float = 2.5,
    ) -> None:
        import isaaccapture.viz as viz

        self._viz = viz
        cfg = viz.QuadLayerConfig()
        cfg.name = "controls_hud"
        cfg.resolution = viz.Resolution(_W, _H)
        cfg.format = viz.PixelFormat.kRGBA8
        # Runtime-composited quads sample the texture directly, so a mip chain
        # would be allocated and never read.
        cfg.generate_mipmaps = False
        cfg.alpha_blend = False
        self._layer = session.add_quad_layer(cfg)
        self._layer.set_visible(False)

        self._hold_s = hold_s
        self._since_show = 0.0
        self._visible = False
        self._placement = HeadLocked(
            PlacementConfig(
                size_meters=(width_m, width_m * _H / _W),
                distance=distance_m,
                offset_y=offset_y_m,
            )
        )

    def show(self, lines: Sequence[str]) -> None:
        """Draw ``lines`` and restart the hold timer."""
        import cupy as cp

        self._layer.submit(cp.asarray(_render(lines)))
        self._layer.set_visible(True)
        self._visible = True
        self._since_show = 0.0

    def step(self, dt: float, head_pose) -> None:
        """Render-thread tick: hold the panel in front of the head, then hide
        it once the hold expires."""
        if not self._visible:
            return
        self._since_show += dt
        if self._since_show >= self._hold_s:
            self._layer.set_visible(False)
            self._visible = False
            return
        if head_pose is None:
            return
        placed = self._placement.update(head_pose.position, head_pose.orientation)
        self._layer.set_placement(
            self._viz.QuadLayerPlacement(
                self._viz.Pose3D(placed.position, placed.orientation),
                placed.size_meters,
            )
        )


def make_hud(session, enabled: bool) -> Optional[Hud]:
    """``None`` when disabled, so callers can stay branch-light."""
    return Hud(session) if enabled else None
