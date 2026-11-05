# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Shared interfaces for frame/camera data providers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from inference.camera.camera_models import CameraModel


@dataclass(frozen=True)
class ImageData:
    """Data that a dataset provider can load for one image identifier.

    Attributes:
        bgr: Image decoded by OpenCV in BGR channel order, or ``None``
            when the corresponding RGB file is missing or unreadable.
        intrinsics: Full-image 3-by-3 pinhole camera intrinsic matrix, a
            CameraModel for perspective cropping, or ``None``
            when unavailable.
        camera_to_world_transform: 4-by-4 homogeneous transformation matrix
            that maps camera-space points to world-space points, or ``None``
            when unavailable.
        gt_2d_bboxes: Unpadded ground-truth hand boxes keyed by hand side.
            Each box is ``[center_x, center_y, size]`` in image pixels.
            ``None`` means the hands annotation file is missing, unreadable,
            or invalid. An empty dictionary means the file was loaded but
            neither hand has a usable box. If only one hand has a usable box,
            only that hand's key is present.

    Example:
        >>> image_data = ImageData(
        ...     bgr=np.zeros((1408, 1408, 3), dtype=np.uint8),
        ...     intrinsics=np.array(
        ...         [[600.0, 0.0, 704.0], [0.0, 600.0, 704.0], [0.0, 0.0, 1.0]],
        ...         dtype=np.float32,
        ...     ),
        ...     camera_to_world_transform=np.eye(4),
        ...     gt_2d_bboxes={"left": np.array([704.0, 704.0, 256.0], dtype=np.float32)},
        ... )
    """

    bgr: np.ndarray | None
    intrinsics: np.ndarray | CameraModel | None
    camera_to_world_transform: np.ndarray | None
    gt_2d_bboxes: dict[str, np.ndarray] | None
