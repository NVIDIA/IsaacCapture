# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Right-hand to canonical-left mirroring of rasters, cameras and predictions."""

import numpy as np
import torch

from inference.preprocessing.crop import (
    crop_and_preprocess,
    project_camera_points_to_crop_with_validity,
)


def mirror_full_image_camera_horizontal(
    bbx_xys: np.ndarray,
    K_fullimg: np.ndarray,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a bbox and calibrated pinhole camera about pixel centers.

    A horizontal image flip maps full-image pixels as
    ``u' = (W - 1) - u``.  Canonicalizing a right hand for the left-hand
    model also mirrors camera-frame X, so the complete intrinsic transform is
    ``K' = F K M`` rather than only replacing the principal point.  The full
    matrix form remains exact for off-center cameras and nonzero skew.

    Args:
        bbx_xys: Square crop ``[center_x, center_y, size]`` in full-image
            pixels.
        K_fullimg: Calibrated ``3 x 3`` full-image intrinsic matrix.
        image_width: Full-image width in pixels.

    Returns:
        Independent float32 copies ``(mirrored_bbox, mirrored_K)``.
    """
    bbox = np.asarray(bbx_xys, dtype=np.float32)
    intrinsics = np.asarray(K_fullimg, dtype=np.float32)
    if bbox.shape != (3,):
        raise ValueError(f"bbx_xys must have shape (3,), got {bbox.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"K_fullimg must have shape (3, 3), got {intrinsics.shape}")
    width = int(image_width)
    if width <= 0:
        raise ValueError("image_width must be positive")

    flip_limit = float(width - 1)
    mirrored_bbox = bbox.copy()
    mirrored_bbox[0] = flip_limit - mirrored_bbox[0]
    pixel_flip = np.array(
        [
            [-1.0, 0.0, flip_limit],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_mirror = np.diag(np.array([-1.0, 1.0, 1.0], dtype=np.float32))
    mirrored_intrinsics = pixel_flip @ intrinsics @ camera_mirror
    return mirrored_bbox, mirrored_intrinsics


def mirror_legacy_crop_camera_horizontal(
    bbx_xys: np.ndarray,
    K_fullimg: np.ndarray,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the historical crop-then-flip camera convention.

    Legacy weak/perspective hand data crops the right hand first and flips the
    crop afterward. Its continuous-coordinate metadata uses ``W - x`` and
    changes only the principal point. This is intentionally different from
    :func:`mirror_full_image_camera_horizontal`; keep it for old checkpoints
    whose training distribution followed that convention.
    """
    bbox = np.asarray(bbx_xys, dtype=np.float32)
    intrinsics = np.asarray(K_fullimg, dtype=np.float32)
    if bbox.shape != (3,):
        raise ValueError(f"bbx_xys must have shape (3,), got {bbox.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"K_fullimg must have shape (3, 3), got {intrinsics.shape}")
    width = int(image_width)
    if width <= 0:
        raise ValueError("image_width must be positive")

    mirrored_bbox = bbox.copy()
    mirrored_intrinsics = intrinsics.copy()
    mirrored_bbox[0] = float(width) - mirrored_bbox[0]
    mirrored_intrinsics[0, 2] = float(width) - mirrored_intrinsics[0, 2]
    return mirrored_bbox, mirrored_intrinsics


def prepare_right_hand_model_input(
    img_bgr: np.ndarray,
    bbx_xys: np.ndarray,
    K_fullimg: np.ndarray,
    *,
    calibrated_full_image: bool,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Canonicalize a right-hand raster and camera for the left-hand model.

    The calibrated camera contract mirrors the full raster about pixel
    centers, transforms the complete camera, and only then crops at the
    mirrored bbox.  Cropping first and flipping the resized crop differs by
    one output pixel under the affine sampling convention and would not match
    training.  Legacy checkpoints intentionally retain their historical
    crop-then-flip raster and ``W - x`` metadata contract.

    Returns:
        ``(normalized_crop, model_bbox, model_K)`` in the canonical-left
        model input frame.
    """

    image = np.asarray(img_bgr)
    if image.ndim != 3:
        raise ValueError(f"img_bgr must have shape (H, W, C), got {image.shape}")
    image_width = int(image.shape[1])
    if calibrated_full_image:
        model_bbox, model_K = mirror_full_image_camera_horizontal(
            bbx_xys,
            K_fullimg,
            image_width,
        )
        flipped_full_image = np.ascontiguousarray(image[:, ::-1])
        crop_tensor, _ = crop_and_preprocess(
            flipped_full_image,
            model_bbox,
        )
    else:
        model_bbox, model_K = mirror_legacy_crop_camera_horizontal(
            bbx_xys,
            K_fullimg,
            image_width,
        )
        crop_tensor, _ = crop_and_preprocess(image, bbx_xys)
        crop_tensor = crop_tensor.flip(-1)
    return crop_tensor, model_bbox, model_K


def mirror_left_prediction_to_right(
    model,
    poses,
    transl,
    identity_coeffs,
    global_scale,
    K_fullimg,
    bbx_xys,
    scale_params=None,
    pose_mirror=None,
):
    """Convert left-canonicalized right-hand predictions back to right hands (batched).

    All inputs share a leading batch dim B over the right-hand samples in the
    current EgoHand forward pass.

    Args:
        model: Loaded EgoHand model with left and right SOMAHand layers.
        poses: (B, 25, 3, 3) left-canonicalized SOMA hand rotations.
        transl: (B, 3) left-canonicalized camera translation in meters.
        identity_coeffs: (B, K) SOMAHand identity coefficients.
        global_scale: (B,) per-sample uniform scale.
        K_fullimg: (B, 3, 3) full-image camera intrinsics.
        bbx_xys: (B, 3) per-sample bbox as ``[center_x, center_y, size]``.
        scale_params: Optional (B, S) per-sample MHR hand scale parameters.
        pose_mirror: Optional pre-constructed ``PoseMirror_SOMA`` matching the
            paired ``["Root"] + left_joint_names + right_joint_names`` rig.
            The predictor caches this once at init so subsequent batches do
            not pay the per-call setup. When ``None``, a fresh instance is
            built on each call (correct, but slower).

    Returns:
        Dict of right-hand predictions matching EgoHand's saved schema, all with
        leading batch dim B: ``pred_vertices`` ``(B, V, 3)``, ``transl``
        ``(B, 3)``, ``pred_keypoints_3d`` ``(B, 25, 3)``, ``pred_keypoints_2d``
        ``(B, 25, 2)`` (bbox-relative, normalized by bbox size), ``poses``
        ``(B, 25, 3, 3)``.
    """
    from soma.geometry.rig_utils import (
        PoseMirror_SOMA,
        joint_world_to_local,
        precompute_joint_orient,
        remove_joint_orient_local,
    )

    left_layer = model.soma_hand.left
    right_layer = model.soma_hand.right
    batch_size = poses.shape[0]

    # 1. Forward-kinematics through the left layer in one batched call.
    left_layer.prepare_identity(
        identity_coeffs,
        global_scale=global_scale,
        scale_params=scale_params,
    )
    left_pose_out = left_layer.pose(
        poses=poses,
        pose2rot=False,
        absolute_pose=False,
        global_translation=transl,
        fk_only=True,
    )
    left_transforms = left_pose_out.transforms  # (B, 25, 4, 4)

    # 2. Apply the left<->right paired mirror once for the entire sub-batch.
    if pose_mirror is None:
        left_joint_names = list(left_layer.rig_data["joint_names"])
        right_joint_names = list(right_layer.rig_data["joint_names"])
        paired_names = ["Root"] + left_joint_names + right_joint_names
        pose_mirror = PoseMirror_SOMA(paired_names, root_name="Root")
    num_joints = left_transforms.shape[1]
    paired_transforms = (
        torch.eye(
            4,
            device=left_transforms.device,
            dtype=left_transforms.dtype,
        )
        .expand(batch_size, num_joints * 2 + 1, 4, 4)
        .clone()
    )
    paired_transforms[:, 1 : 1 + num_joints] = left_transforms
    right_transforms = pose_mirror(paired_transforms)[
        :, 1 + num_joints :
    ]  # (B, 25, 4, 4)

    # 3. Decompose mirrored cam-space transforms back into local rotations
    #    + wrist translation.
    right_local = joint_world_to_local(right_transforms, right_layer.joint_parent_ids)
    right_transl = right_transforms[
        :, 0, :3, 3
    ].clone()  # wrist (joint 0) world translation
    t_pose_world = right_layer.t_pose_world
    orient, orient_parent_T = precompute_joint_orient(
        t_pose_world, right_layer.joint_parent_ids
    )
    right_poses = remove_joint_orient_local(
        right_local[:, :, :3, :3], orient, orient_parent_T
    )

    # 4. Run the right SOMAHand forward and project.
    right_out = model.soma_hand(
        poses=right_poses,
        identity_coeffs=identity_coeffs,
        hand_type="right",
        transl=None,
        pose2rot=False,
        global_scale=global_scale,
        scale_params=scale_params,
    )
    joints_full = right_out["joints"] + right_transl.unsqueeze(1)
    keypoints_2d, projection_valid = project_camera_points_to_crop_with_validity(
        joints_full,
        K_fullimg,
        bbx_xys,
        min_depth=float(getattr(model, "kp2d_min_depth", 1e-4)),
    )
    return {
        "pred_vertices": right_out["vertices"],
        "transl": right_transl,
        "pred_keypoints_3d": right_out["joints"],
        "pred_keypoints_2d": keypoints_2d,
        "pred_projection_valid": projection_valid,
        "poses": right_poses,
    }
