# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-shaped OGLO tactile heatmap renderer for the IsaacTeleop headset overlay.

Produces an RGBA8 ``numpy`` frame (H, W, 4) ready for ``isaaccapture.viz`` ``QuadLayer.submit``.
The layout, colormap (YlOrRd) and normalization replicate the OGLO bench heatmap viewer
so the in-headset view matches the bench tool an operator already trusts.

Taxel index convention (firmware ``sample_order = "finger,row,col"``):
    idx = finger * 16 + row * 4 + col            (5 fingers x 4 rows x 4 cols = 80)
Row 0 is the fingertip (distal); rows are flipped so the fingertip draws at the top.

Standalone preview (writes a PNG you can open):
    python oglo_heatmap.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except (
    ImportError
) as exc:  # pragma: no cover - PIL is a hard dependency of this renderer
    raise ImportError("oglo_heatmap requires Pillow: pip install pillow") from exc

# --- geometry (from the reference viewer) ------------------------------------
NUM_FINGERS = 5
ROWS = 4
COLS = 4
TAXELS_PER_FINGER = ROWS * COLS  # 16
NUM_TAXELS = NUM_FINGERS * TAXELS_PER_FINGER  # 80

_CELL = 0.36
_BLK = COLS * _CELL  # 1.44

# Finger block anchor positions in "cell units" (right hand). Left hand mirrors X.
_RIGHT_POS = {
    "thumb": (1.5, 2.5),
    "index": (3.05, 5.3),
    "middle": (4.6, 6.05),
    "ring": (6.15, 5.3),
    "pinky": (7.7, 4.4),
}
_XS = [p[0] for p in _RIGHT_POS.values()]
_MIRROR = min(_XS) + max(_XS) + _BLK

CHANNELS_RIGHT = ["thumb", "index", "middle", "ring", "pinky"]
CHANNELS_LEFT = ["pinky", "ring", "middle", "index", "thumb"]

# YlOrRd 9-stop colormap (ColorBrewer), identical to the viewer.
_STOPS = np.array(
    [
        [255, 255, 255],
        [255, 255, 204],
        [255, 237, 160],
        [254, 217, 118],
        [253, 141, 60],
        [252, 78, 42],
        [227, 26, 28],
        [189, 0, 38],
        [128, 0, 38],
    ],
    dtype=np.float32,
)
_IDLE_RGB = (244, 244, 244)  # "#f4f4f4" for ~zero pressure


def ylord(t: np.ndarray) -> np.ndarray:
    """Map t in [0, 1] (array) to RGB uint8 via the YlOrRd 9-stop ramp."""
    t = np.clip(t, 0.0, 1.0) * (len(_STOPS) - 1)
    i = np.floor(t).astype(np.int32)
    i = np.minimum(i, len(_STOPS) - 2)
    f = (t - i)[..., None]
    rgb = _STOPS[i] * (1.0 - f) + _STOPS[i + 1] * f
    return rgb.astype(np.uint8)


def _hand_positions(side: str) -> dict:
    if side == "left":
        return {k: (_MIRROR - p[0] - _BLK, p[1]) for k, p in _RIGHT_POS.items()}
    return _RIGHT_POS


@dataclass
class Normalizer:
    """Baseline subtraction + threshold + scale, matching the viewer.

    ``v = clip((raw - baseline - threshold) / scale, 0, 1)``. The baseline is the
    median of the first ``baseline_frames`` samples (captured per hand). RAW mode
    bypasses baseline/threshold for absolute-ADC display.
    """

    scale: float = 500.0
    threshold: float = 30.0
    baseline_frames: int = 18
    raw: bool = False
    _buf: list = field(default_factory=list)
    _baseline: np.ndarray | None = None

    def rebaseline(self) -> None:
        self._buf.clear()
        self._baseline = None

    def normalize(self, taxels: np.ndarray) -> np.ndarray:
        taxels = np.asarray(taxels, dtype=np.float32).reshape(NUM_TAXELS)
        if self.raw:
            return np.clip(taxels / max(self.scale, 1.0), 0.0, 1.0)
        if self._baseline is None:
            self._buf.append(taxels.copy())
            if len(self._buf) >= self.baseline_frames:
                self._baseline = np.median(np.stack(self._buf), axis=0)
            base = self._buf[0] if self._baseline is None else self._baseline
        else:
            base = self._baseline
        return np.clip((taxels - base - self.threshold) / self.scale, 0.0, 1.0)


@dataclass
class _CellGeom:
    finger: int
    row: int
    col: int
    taxel: int
    x: float  # left, viewer cell-space
    y: float  # top, viewer cell-space (y-up before pixel flip)


def _hand_cells(side: str, channels: list) -> tuple:
    """Precompute per-taxel cell rects (cell-space) and the hand bounding box."""
    pos = _hand_positions(side)
    cells: list = []
    xs: list = []
    ys: list = []
    for f in range(NUM_FINGERS):
        name = channels[f] if f < len(channels) else CHANNELS_RIGHT[f]
        bx, by = pos[name]
        for r in range(ROWS):
            dr = ROWS - 1 - r  # FLIP_ROWS: fingertip (row 0) at the top
            for c in range(COLS):
                taxel = f * TAXELS_PER_FINGER + r * COLS + c
                # Viewer: px=ofx+(bx+c*CELL)*s, py=ofy-(by+BLK-dr*CELL)*s.
                x = bx + c * _CELL
                y = -(by + _BLK - dr * _CELL)
                cells.append(_CellGeom(f, r, c, taxel, x, y))
                xs += [x, x + _CELL]
                ys += [y, y + _CELL]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    label_anchors = []
    for f in range(NUM_FINGERS):
        name = channels[f] if f < len(channels) else CHANNELS_RIGHT[f]
        bx, by = pos[name]
        label_anchors.append((name, bx + _BLK / 2.0, -(by + _BLK + 0.2)))
    return cells, bbox, label_anchors


class TactileHeatmapRenderer:
    """Renders both gloves into one RGBA8 HUD panel for the headset overlay."""

    def __init__(
        self,
        width: int = 1024,
        height: int = 512,
        bg_rgba: tuple = (16, 18, 22, 200),
        channels_left: list | None = None,
        channels_right: list | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.bg_rgba = bg_rgba
        self._left = _hand_cells("left", channels_left or CHANNELS_LEFT)
        self._right = _hand_cells("right", channels_right or CHANNELS_RIGHT)
        try:
            self._font = ImageFont.truetype(
                "DejaVuSans-Bold.ttf", max(11, height // 36)
            )
        except OSError:
            self._font = ImageFont.load_default()

    def render(
        self, left_norm: np.ndarray | None, right_norm: np.ndarray | None
    ) -> np.ndarray:
        """Return an (H, W, 4) uint8 RGBA frame. Pass normalized taxels in [0, 1]."""
        img = Image.new("RGBA", (self.width, self.height), self.bg_rgba)
        draw = ImageDraw.Draw(img, "RGBA")

        half = self.width // 2
        margin = int(self.height * 0.06)
        self._draw_hand(
            draw,
            self._left,
            left_norm,
            "LEFT",
            (margin, margin, half - margin, self.height - margin),
        )
        self._draw_hand(
            draw,
            self._right,
            right_norm,
            "RIGHT",
            (half + margin, margin, self.width - margin, self.height - margin),
        )
        self._draw_legend(draw)
        return np.asarray(img, dtype=np.uint8)

    def _draw_hand(self, draw, hand, norm, title, region) -> None:
        cells, bbox, labels = hand
        rx0, ry0, rx1, ry1 = region
        rw, rh = rx1 - rx0, ry1 - ry0
        x0, y0, x1, y1 = bbox
        span_x, span_y = (x1 - x0) or 1.0, (y1 - y0) or 1.0
        s = 0.86 * min(rw / span_x, rh / span_y)
        ofx = rx0 + (rw - span_x * s) / 2.0 - x0 * s
        ofy = ry0 + (rh - span_y * s) / 2.0 - y0 * s
        sz = _CELL * s

        values = (
            None
            if norm is None
            else np.clip(
                np.asarray(norm, dtype=np.float32).reshape(NUM_TAXELS), 0.0, 1.0
            )
        )
        for cell in cells:
            px = ofx + cell.x * s
            py = ofy + cell.y * s
            if values is None:
                color = (90, 92, 96, 255)  # disconnected: neutral gray
            else:
                v = float(values[cell.taxel])
                color = (
                    (*_IDLE_RGB, 255)
                    if v < 0.015
                    else (*tuple(int(x) for x in ylord(np.array(v))), 255)
                )
            draw.rectangle(
                [px + 0.6, py + 0.6, px + sz - 1.2, py + sz - 1.2],
                fill=color,
                outline=(60, 62, 66, 255),
            )

        for name, lx, ly in labels:
            px = ofx + lx * s
            py = ofy + ly * s
            draw.text(
                (px, py),
                name.capitalize(),
                fill=(210, 212, 216, 255),
                font=self._font,
                anchor="mb",
            )

        draw.text(
            (rx0, ry0 - int(self.height * 0.05)),
            title,
            fill=(235, 236, 240, 255),
            font=self._font,
            anchor="lt",
        )

    def _draw_legend(self, draw) -> None:
        bar_w = int(self.width * 0.22)
        bar_h = max(8, int(self.height * 0.018))
        x0 = (self.width - bar_w) // 2
        y0 = self.height - bar_h - int(self.height * 0.035)
        ramp = ylord(np.linspace(0.0, 1.0, bar_w))
        for i in range(bar_w):
            r, g, b = (int(ramp[i][0]), int(ramp[i][1]), int(ramp[i][2]))
            draw.line([(x0 + i, y0), (x0 + i, y0 + bar_h)], fill=(r, g, b, 255))
        draw.text(
            (x0, y0 - 2), "low", fill=(200, 200, 205, 255), font=self._font, anchor="rb"
        )
        draw.text(
            (x0 + bar_w, y0 - 2),
            "high",
            fill=(200, 200, 205, 255),
            font=self._font,
            anchor="lb",
        )


def _demo() -> None:
    """Render a synthetic frame (a press on the right index fingertip) to PNG."""
    rng = np.random.default_rng(0)
    left = rng.integers(40, 90, NUM_TAXELS).astype(np.float32)  # idle noise
    right = rng.integers(40, 90, NUM_TAXELS).astype(np.float32)
    # Press: right index (finger 1), fingertip rows, strong center taxels.
    for r in range(2):
        for c in range(1, 3):
            right[1 * 16 + r * 4 + c] = 3500
    # Press: left thumb (finger 4 in left channel order) lighter.
    left[4 * 16 + 0 * 4 + 1] = 1800

    norm = Normalizer(raw=True, scale=4000.0)  # raw mode for a deterministic demo
    renderer = TactileHeatmapRenderer(width=1024, height=512)
    frame = renderer.render(norm.normalize(left), norm.normalize(right))
    assert frame.shape == (512, 1024, 4) and frame.dtype == np.uint8
    out = "oglo_heatmap_demo.png"
    Image.fromarray(frame, "RGBA").save(out)
    nonzero = int((frame[..., :3] > 0).any(axis=-1).sum())
    print(f"wrote {out}  shape={frame.shape}  non-background px={nonzero}")


if __name__ == "__main__":
    _demo()
