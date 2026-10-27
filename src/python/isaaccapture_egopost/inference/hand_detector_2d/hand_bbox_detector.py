# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""2D hand bbox detector Protocol, coordinate wrapper, and bbox padding utility."""

from typing import Protocol

import numpy as np

__all__ = ["HandBBoxDetector", "pad_hand_bboxes"]


class HandBBoxDetector(Protocol):
    """Interface for detectors that run a hand-detection model."""

    def run(self, image_ids, images_bgr, hands, output_paths_per_hand=None) -> dict:
        """Return padded square hand boxes indexed by image ID and then hand."""
        ...


class SquarePixelFrameBBoxDetector:
    """Adapt a file-backed detector to a run that corrected the pixel aspect ratio.

    Live detectors report boxes in the frames handed to them. Detectors that
    read boxes from a file (saved JSON, projected parquet ground truth) report in the
    original frame, which would crop the wrong region once the frames have been
    resized to square pixels.
    """

    def __init__(self, detector, resize_params):
        self._detector = detector
        self._resize_params = resize_params

    def __getattr__(self, name):
        # Forward per-video setup calls: set_bbox_dir, set_intrinsics, ...
        return getattr(self._detector, name)

    def run(self, image_ids, images_bgr, hands, output_paths_per_hand=None):
        bboxes = self._detector.run(
            image_ids=image_ids,
            images_bgr=images_bgr,
            hands=hands,
            output_paths_per_hand=output_paths_per_hand,
        )
        return {
            image_id: {
                hand: self._resize_params.to_corrected_bbox(bbx)
                for hand, bbx in per_hand.items()
            }
            for image_id, per_hand in bboxes.items()
        }


def pad_hand_bboxes(
    bboxes_per_image: dict[str, dict[str, np.ndarray]],
    padding: float = 0.3,
) -> dict[str, dict[str, np.ndarray]]:
    """Pad the ground-truth hand boxes for each image and return them keyed by image ID.

    Args:
        bboxes_per_image: Unpadded hand boxes from dataset ground-truth annotations, shaped as
            ``{image_id: {hand: [center_x, center_y, size]}}``.
        padding: Fractional increase applied to each square box's size.

    Returns:
        A new mapping keyed by image ID and then hand, with padded box sizes.
    """
    padded_bboxes_per_image: dict[str, dict[str, np.ndarray]] = {}
    for image_id, bboxes_per_hand in bboxes_per_image.items():
        padded_bboxes_per_image[image_id] = {}
        for hand, bbox in bboxes_per_hand.items():
            padded_bbox = np.asarray(bbox, dtype=np.float32).copy()
            padded_bbox[2] *= 1.0 + padding
            padded_bboxes_per_image[image_id][hand] = padded_bbox
    return padded_bboxes_per_image
