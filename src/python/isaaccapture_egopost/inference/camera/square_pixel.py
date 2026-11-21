# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Square-pixel frame correction for non-square-pixel (fx != fy) calibrations."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SquarePixelResizeParams:
    """Parameters for rescaling a frame so its camera has square pixels.

    A calibration with ``fx != fy`` describes non-square pixels: a physically
    square patch lands on the sensor as a rectangle. EgoHand hand checkpoints are
    trained only on square-pixel cameras, so the square crop reaching the model
    shows the hand stretched along one axis -- out of distribution, and not
    something the model was trained to undo.

    Rescaling by ``diag(sx, sy) = diag(f/fx, f/fy)`` with ``f = max(fx, fy)``
    produces the image a square-pixel sensor of focal ``f`` would have captured
    through the same optics, and transforms ``K`` to match. Targeting ``max``
    rather than ``min`` upsamples the axis with the smaller focal, so the
    resize never discards detail the model could have used.

    Resampling only reparameterizes the image plane: pixel ``u`` of the
    corrected frame back-projects along the same ray as ``u / sx`` of the
    original, so the camera frame is untouched and 3D outputs -- translation,
    joints, vertices -- are already in the true camera frame. Only pixel-valued
    quantities need mapping back, via :meth:`to_original_bbox` and
    :meth:`undo_frame`.

    Pixel indices follow OpenCV's half-pixel convention, ``u' = sx*(u + 0.5) -
    0.5``, which is what ``cv2.resize`` implements. Mapping ``K`` with the
    naive ``u' = sx*u`` instead leaves a systematic ``0.5*(sx - 1)`` px offset
    between where the intrinsics say a point lands and where the resampler put
    it.

    Attributes:
        width: Corrected frame width in pixels.
        height: Corrected frame height in pixels.
        scale_x: Realized horizontal scale, ``width / source_width``.
        scale_y: Realized vertical scale, ``height / source_height``.
        K: ``(3, 3)`` float32 intrinsics of the corrected frame.
        source_size: The original ``(width, height)``.

    Both scales are the *realized* ones, recomputed from the integer output
    size rather than the requested ``f/fx``; mapping back with the requested
    value would leave the rounding as a systematic offset.
    """

    width: int
    height: int
    scale_x: float
    scale_y: float
    K: np.ndarray
    source_size: tuple[int, int]

    @property
    def is_identity(self) -> bool:
        """True when the pixels are already square and nothing is resized."""
        return (self.width, self.height) == self.source_size

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize one frame into the square-pixel frame."""
        if self.is_identity:
            return frame
        return cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )

    def undo_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a rendered frame back to the original size."""
        if self.is_identity:
            return frame
        return cv2.resize(frame, self.source_size, interpolation=cv2.INTER_AREA)

    def to_corrected_bbox(self, bbx_xys) -> np.ndarray:
        """Map a ``[cx, cy, size]`` square box from the original frame to the corrected one.

        Needed only for boxes produced against the original frame -- saved JSON
        or projected ground truth. Detectors that look at the frames handed to
        them already report in the corrected frame.

        The square side grows by the larger scale so the box still covers the
        hand on the stretched axis; sizing by the smaller one would crop it.
        """
        bbox = np.asarray(bbx_xys, dtype=np.float64)
        return np.array(
            [
                self.scale_x * (bbox[0] + 0.5) - 0.5,
                self.scale_y * (bbox[1] + 0.5) - 0.5,
                bbox[2] * max(self.scale_x, self.scale_y),
            ],
            dtype=np.float32,
        )

    def to_original_bbox(self, bbx_xys) -> np.ndarray:
        """Inverse of :meth:`to_corrected_bbox`, for boxes leaving the pipeline."""
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
    K, width, height, tol: float = 1e-3
) -> SquarePixelResizeParams:
    """Compute parameters for resizing a frame to square camera pixels.

    Args:
        K: ``(3, 3)`` intrinsics of the decoded frame.
        width: Decoded frame width in pixels.
        height: Decoded frame height in pixels.
        tol: Relative ``|fy/fx - 1|`` at or below which the pixels count as
            already square and identity parameters are returned.

    Returns:
        :class:`SquarePixelResizeParams` whose ``K`` satisfies ``fx == fy`` up
        to the rounding of the output size to whole pixels. Identity parameters
        when the pixels are already square, so callers can apply them
        unconditionally.

    Raises:
        ValueError: If either focal is non-positive.
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
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
            K=K.astype(np.float32),
            source_size=source_size,
        )

    f_target = max(fx, fy)
    new_width = max(1, round(width * f_target / fx))
    new_height = max(1, round(height * f_target / fy))
    scale_x, scale_y = new_width / float(width), new_height / float(height)

    K_new = np.eye(3, dtype=np.float32)
    K_new[0, 0] = scale_x * fx
    K_new[1, 1] = scale_y * fy
    K_new[0, 2] = scale_x * (cx + 0.5) - 0.5
    K_new[1, 2] = scale_y * (cy + 0.5) - 0.5
    return SquarePixelResizeParams(
        width=new_width,
        height=new_height,
        scale_x=scale_x,
        scale_y=scale_y,
        K=K_new,
        source_size=source_size,
    )


def rewrite_bbox_json_in_original_frame(out_dir, hands, resize_params) -> None:
    """Rewrite any persisted bbox JSON from the corrected frame to the original.

    Detectors run on the corrected frames and persist what they saw, but the
    JSON is consumed against the source video. Mapping once here, on the way
    out, keeps a single convention inside the pipeline -- everything is the
    corrected frame -- instead of spreading the conversion over each use site.

    Args:
        out_dir: Per-video output folder holding ``<name>_<hand>_bbox.json``.
        hands: Hand sides whose JSON should be rewritten.
        resize_params: Parameters that were applied, or ``None``/identity for a
            run that did not resize, in which case this is a no-op.
    """
    if resize_params is None or resize_params.is_identity:
        return
    out_dir = Path(out_dir)
    for hand in hands:
        path = out_dir / f"{out_dir.name}_{hand}_bbox.json"
        if not path.exists():
            continue
        boxes = json.loads(path.read_text())
        path.write_text(
            json.dumps(
                {
                    frame_id: resize_params.to_original_bbox(bbx).tolist()
                    for frame_id, bbx in boxes.items()
                }
            )
        )
