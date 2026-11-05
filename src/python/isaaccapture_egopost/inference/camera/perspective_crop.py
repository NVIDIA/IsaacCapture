# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Perspective cropping for detected hands in calibrated distorted images."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from inference.camera.camera_models import CameraModel, create_perspective_remap


def perspective_crop(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    camera: CameraModel,
    crop_side_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a square perspective crop of a detected hand region.

    The camera turns toward the box-center ray using the shortest rotation from
    the source optical axis, following UmeTrack with ``camera_angle=0``.
    Its focal length fits valid directions densely sampled along the box edges.
    Each crop pixel is sampled from the source image; unavailable samples are black.

    Args:
        image_bgr: ``(H, W, 3)`` source image in OpenCV BGR channel order.
        bbox: ``(3,)`` square box ordered ``center_x``, ``center_y``, ``side_length``,
            in source image pixels.
        camera: CameraModel describing ``image_bgr`` pixel coordinates.
        crop_side_pixels: Side length of the square crop to produce, in
            pixels. The caller sets it from the resolution its model consumes.

    Returns:
        Tuple of the square RGB crop (float32, pixel values divided by 255),
        virtual pinhole intrinsics, full-crop bbox ``[center_x, center_y,
        side_length]`` in crop pixels, and the rotation from crop-camera to
        source-camera coordinates, in that order.
    """
    bbox_center_x, bbox_center_y, bbox_side = np.asarray(bbox, dtype=np.float64)

    # The curved perimeter, including each corner, determines the full angular
    # extent; considering only four corners can cut into a fisheye hand region.
    bbox_left = bbox_center_x - bbox_side / 2
    bbox_right = bbox_center_x + bbox_side / 2
    bbox_top = bbox_center_y - bbox_side / 2
    bbox_bottom = bbox_center_y + bbox_side / 2
    # Sample each edge at four times the crop resolution, including both endpoints.
    edge_sample_count = 4 * crop_side_pixels + 1
    # Row 0 is the box center; the rest walk its four edges.
    bbox_sample_pixels = np.concatenate(
        [
            [[bbox_center_x, bbox_center_y]],
            np.linspace(
                [bbox_left, bbox_top], [bbox_right, bbox_top], edge_sample_count
            ),
            np.linspace(
                [bbox_left, bbox_bottom], [bbox_right, bbox_bottom], edge_sample_count
            ),
            np.linspace(
                [bbox_left, bbox_top], [bbox_left, bbox_bottom], edge_sample_count
            ),
            np.linspace(
                [bbox_right, bbox_top], [bbox_right, bbox_bottom], edge_sample_count
            ),
        ]
    )
    bbox_sample_rays = camera.unproject(bbox_sample_pixels)
    center_ray = bbox_sample_rays[0]
    source_perimeter_rays = bbox_sample_rays[1:]
    source_perimeter_rays = source_perimeter_rays[
        np.isfinite(source_perimeter_rays).all(axis=1)
    ]
    if not np.isfinite(center_ray).all() or len(source_perimeter_rays) == 0:
        raise ValueError("Cannot undistort this bounding box")

    # Match UmeTrack's make_look_at_matrix with camera_angle=0: rotate the source
    # optical axis onto the box-center ray by the shortest rotation.
    # https://github.com/facebookresearch/UmeTrack/blob/main/lib/common/affine.py
    crop_rotation, _ = Rotation.align_vectors(center_ray, [0.0, 0.0, 1.0])
    crop_to_source_rotation = crop_rotation.as_matrix()

    # The edge rays set the crop camera's focal length: each lands on its z=1 plane, and
    # the one furthest off axis is put at the crop's edge.
    virtual_perimeter_rays = source_perimeter_rays @ crop_to_source_rotation
    if np.any(virtual_perimeter_rays[:, 2] <= 0):
        raise ValueError("ROI perimeter reaches behind the virtual pinhole camera")
    normalized_perimeter_coords = (
        virtual_perimeter_rays[:, :2] / virtual_perimeter_rays[:, 2, None]
    )
    crop_center = (crop_side_pixels - 1) / 2.0
    max_normalized_extent = np.max(np.abs(normalized_perimeter_coords))
    focal_length = (crop_side_pixels / 2.0) / max_normalized_extent

    # Each crop pixel is a ray. Rotated into source camera coordinates and projected, it
    # gives the source pixel that crop pixel samples.
    remap_pixels = create_perspective_remap(
        camera, crop_to_source_rotation, focal_length, crop_side_pixels
    )
    crop_bgr = cv2.remap(
        image_bgr,
        remap_pixels,
        None,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = crop_rgb.astype(np.float32) / 255.0
    K_virtual = np.array(
        [
            [focal_length, 0, crop_center],
            [0, focal_length, crop_center],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    bbox_virtual = np.array(
        [crop_center, crop_center, crop_side_pixels],
        dtype=np.float64,
    )
    return crop_rgb, K_virtual, bbox_virtual, crop_to_source_rotation
