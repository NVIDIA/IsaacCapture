# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Square-pixel frame/bbox correction, duplicated from ``inference/camera/square_pixel.py``.

Duplicated rather than imported so this folder can run without the rest of
the repository, including ``inference.camera``. See
``inference/hand_detector_2d/scripts/run_hand_detector.py``'s own module
docstring on why it stays self-contained (the same reasoning
``BBOX_FILE_PATTERN`` there is duplicated for). Only the two pieces actually
needed here are reproduced: computing the resize from a calibration's
``fx``/``fy``, and mapping a ``[cx, cy, size]`` box between the source and
corrected frames. The math matches
``inference.camera.square_pixel.compute_square_pixel_resize_params`` /
``SquarePixelResizeParams`` exactly -- keep the two in sync if either changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "SquarePixelResizeParams",
    "compute_square_pixel_resize_params",
    "load_fx_fy",
]


@dataclass(frozen=True)
class SquarePixelResizeParams:
    """Resize parameters for correcting non-square camera pixels.

    See ``inference.camera.square_pixel.SquarePixelResizeParams`` for the
    full rationale; this is the same object with only the fields this folder
    needs (no ``K``, since nothing here conditions a model on intrinsics).
    """

    width: int
    height: int
    scale_x: float
    scale_y: float
    source_size: tuple[int, int]

    @property
    def is_identity(self) -> bool:
        return (self.width, self.height) == self.source_size

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize one frame into the square-pixel frame."""
        if self.is_identity:
            return frame
        return cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )

    def to_original_bbox(self, bbx_xys) -> np.ndarray:
        """Map a ``[cx, cy, size]`` box from the corrected frame back to the source frame."""
        bbox = np.asarray(bbx_xys, dtype=np.float64)
        return np.array(
            [
                (bbox[0] + 0.5) / self.scale_x - 0.5,
                (bbox[1] + 0.5) / self.scale_y - 0.5,
                bbox[2] / max(self.scale_x, self.scale_y),
            ],
            dtype=np.float32,
        )


def compute_square_pixel_resize_params(
    fx: float, fy: float, width: int, height: int, tol: float = 1e-3
) -> SquarePixelResizeParams:
    """Compute parameters for resizing a frame to square camera pixels.

    Args:
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        width: Decoded frame width in pixels.
        height: Decoded frame height in pixels.
        tol: Relative ``|fy/fx - 1|`` at or below which the pixels count as
            already square and identity parameters are returned.

    Returns:
        Parameters whose corrected size satisfies ``fx == fy`` up to the
        rounding of the output size to whole pixels. Identity parameters when
        the pixels are already square, so callers can apply them
        unconditionally.

    Raises:
        ValueError: If either focal is non-positive.
    """
    if not (fx > 0 and fy > 0):
        raise ValueError(
            f"square-pixel resize needs positive focals, got fx={fx}, fy={fy}"
        )

    source_size = (int(width), int(height))
    if abs(fy / fx - 1.0) <= tol:
        return SquarePixelResizeParams(
            width=source_size[0],
            height=source_size[1],
            scale_x=1.0,
            scale_y=1.0,
            source_size=source_size,
        )

    f_target = max(fx, fy)
    new_width = max(1, round(width * f_target / fx))
    new_height = max(1, round(height * f_target / fy))
    scale_x, scale_y = new_width / float(width), new_height / float(height)
    return SquarePixelResizeParams(
        width=new_width,
        height=new_height,
        scale_x=scale_x,
        scale_y=scale_y,
        source_size=source_size,
    )


def load_fx_fy(path, width: int, height: int) -> tuple[float, float]:
    """Read ``fx``/``fy`` out of a calibration JSON, rescaled to ``width``/``height``.

    Accepts the same pinhole layouts as the pipeline's ``load_calibration``
    (flat ``{fx, fy, cx, cy}``, a bare 3x3 matrix, ``{matrix_3x3: [...]}}``,
    or ``{camera_matrix: {...}}`` in either of those forms) plus its
    ``image_width``/``image_height`` rescaling when the calibration's stated
    size doesn't match the video. Only ``fx``/``fy`` are returned -- nothing
    here needs ``cx``/``cy`` -- and FISHEYE624/KB4 calibrations are out of
    scope (rectification only makes sense for a pinhole camera; use
    ``--no-rectify_pixel_aspect_ratio`` with those instead).

    Raises:
        ValueError: Unrecognized JSON, or a FISHEYE624/KB4 calibration.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and (
        "projection_model_type" in data or "format_version" in data
    ):
        raise ValueError(
            f"{path}: FISHEYE624/KB4 calibrations aren't supported by "
            "run_hand_detector.py's --rectify_pixel_aspect_ratio -- pass "
            "--no-rectify_pixel_aspect_ratio for these cameras"
        )

    calib_wh = None
    if isinstance(data, dict) and "image_width" in data and "image_height" in data:
        calib_wh = (float(data["image_width"]), float(data["image_height"]))

    matrix = None
    if isinstance(data, list):
        matrix = data
    elif isinstance(data, dict):
        for key in ("K", "matrix_3x3", "intrinsics"):
            if isinstance(data.get(key), list):
                matrix = data[key]
                break
        camera_matrix = data.get("camera_matrix", data)
        if matrix is None and isinstance(camera_matrix, dict):
            if isinstance(camera_matrix.get("matrix_3x3"), list):
                matrix = camera_matrix["matrix_3x3"]
            elif "fx" in camera_matrix:
                fx, fy = float(camera_matrix["fx"]), float(camera_matrix["fy"])
                if calib_wh is not None and calib_wh != (float(width), float(height)):
                    fx *= width / calib_wh[0]
                    fy *= height / calib_wh[1]
                return fx, fy

    if matrix is None:
        raise ValueError(
            f"{path}: no recognizable intrinsics (expected fx/fy or a 3x3)"
        )
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"{path}: intrinsics matrix must be 3x3, got {m.shape}")
    fx, fy = float(m[0, 0]), float(m[1, 1])
    if calib_wh is not None and calib_wh != (float(width), float(height)):
        fx *= width / calib_wh[0]
        fy *= height / calib_wh[1]
    return fx, fy
