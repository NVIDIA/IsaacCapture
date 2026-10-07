# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Hand-crop preprocessing and crop-frame projection helpers."""

import cv2
import numpy as np
import torch

from egohand.utils.cam_utils import perspective_projection
from egohand.utils.crop_transform import get_transform

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 256


def crop_and_preprocess(img_bgr, bbx_xys):
    """Crop a hand bbox and normalize it for EgoHand inference.

    Args:
        img_bgr: Full image loaded by OpenCV in BGR channel order.
        bbx_xys: Bbox as ``[center_x, center_y, size]`` in full-image pixels.

    Returns:
        Normalized ``CHW`` tensor and the unnormalized RGB crop.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cx, cy, size = float(bbx_xys[0]), float(bbx_xys[1]), float(bbx_xys[2])
    trans = get_transform(
        center=np.array([cx, cy]), scale=size, res=(IMG_SIZE, IMG_SIZE), rot=0
    )
    img_crop = cv2.warpAffine(
        img_rgb, trans, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR
    )
    img_f = img_crop.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img_f.transpose(2, 0, 1))
    return img_tensor, img_crop


def project_camera_points_to_crop_with_validity(
    camera_points: torch.Tensor,
    K_fullimg: torch.Tensor,
    bbx_xys: torch.Tensor,
    *,
    min_depth: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-frame points without publishing inert placeholders.

    Projection arithmetic uses finite positive-depth placeholders internally,
    then exposes every invalid or non-finite projected point as ``NaN`` plus
    an explicit boolean validity mask. This prevents a behind-camera joint
    from silently appearing at the camera principal point.
    """
    with torch.autocast(
        device_type=camera_points.device.type,
        enabled=False,
    ):
        camera_points_f = camera_points.float()
        projection_valid = torch.isfinite(camera_points_f).all(dim=-1) & (
            camera_points_f[..., 2] > float(min_depth)
        )
        inert_points = torch.zeros_like(camera_points_f)
        inert_points[..., 2] = 1.0
        safe_camera_points = torch.where(
            projection_valid.unsqueeze(-1),
            camera_points_f,
            inert_points,
        )
        pixels = perspective_projection(
            safe_camera_points,
            K_fullimg.float(),
        )
        keypoints_2d = (pixels - bbx_xys[:, None, :2].float()) / bbx_xys[
            :, None, 2:3
        ].float()
        projection_valid = projection_valid & torch.isfinite(keypoints_2d).all(dim=-1)
        keypoints_2d = keypoints_2d.masked_fill(
            ~projection_valid.unsqueeze(-1),
            float("nan"),
        )
    return keypoints_2d, projection_valid.detach()
