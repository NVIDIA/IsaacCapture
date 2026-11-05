# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Camera calibration loading and intrinsics resolution for decoded videos."""

import numpy as np
import torch

from inference.camera.camera_models import (
    CameraModel,
    load_camera_calibration,
    load_fisheye_camera,
)


def estimate_K(width, height, focal: float | None = None):
    """Create the full-image camera matrix used by EgoHand inference.

    The model consumes this as ``K_fullimg`` together with the hand bbox. Inside
    ``EgoHand.forward_step()``, it is used for CLIFF bbox conditioning, converting
    the predicted weak-perspective camera to ``pred_cam_t``, and projecting the
    predicted 3D hand joints back to image/crop coordinates.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        focal: Focal length in pixels. When ``None`` (default), falls back to
            the image-size heuristic ``max(width, height)``. Prefer a real
            calibrated focal length when one is available instead of this
            fallback.

    Returns:
        ``(3, 3)`` torch camera intrinsics tensor with the chosen focal and
        principal point at the image center.
    """
    if focal is None:
        focal = float(max(width, height))
    K = torch.eye(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def _require_parsed_calibration(calibration):
    """Return ``calibration`` if it is parsed JSON data, else raise ``TypeError``.

    A JSON file whose top-level value is a string must not be treated as a path
    to read, so only already parsed ``dict``/``list`` data is accepted.
    """
    if not isinstance(calibration, (dict, list)):
        raise TypeError(
            "Calibration must be parsed JSON (dict or list), got "
            f"{type(calibration).__name__}; read the JSON file before calling"
        )
    return calibration


def load_calibration(calibration, camera_side=None):
    """Read shared camera JSON, legacy pinhole intrinsics, or dataset calibration.

    Args:
        calibration: Already parsed calibration JSON (a dict or a 3x3 list).
            Paths are rejected: callers read the file themselves, so JSON
            content is never reinterpreted as a filesystem path.
        camera_side: KB4 camera entry, ``left`` or ``right``; unused for other models.

    Returns:
        The pinhole tuple ``(fx, fy, cx, cy, calib_wh)``. ``calib_wh`` is the
        calibration's stated ``(width, height)`` so the caller can rescale
        intrinsics; ``None`` means they are assumed to match the video.

        Shared camera JSON and legacy FISHEYE624/KB4 calibration return a
        ``CameraModel`` with ``K``, pixel dimensions, and project/unproject methods.
    """
    parsed_calibration = _require_parsed_calibration(calibration)
    if isinstance(parsed_calibration, dict) and (
        "projection_model_type" in parsed_calibration
        or "format_version" in parsed_calibration
    ):
        return load_fisheye_camera(parsed_calibration, camera_side=camera_side)

    calib_wh = None
    if (
        isinstance(parsed_calibration, dict)
        and "image_width" in parsed_calibration
        and "image_height" in parsed_calibration
    ):
        calib_wh = (
            float(parsed_calibration["image_width"]),
            float(parsed_calibration["image_height"]),
        )

    matrix = None
    if isinstance(parsed_calibration, list):
        matrix = parsed_calibration
    elif isinstance(parsed_calibration, dict):
        for key in ("K", "matrix_3x3", "intrinsics"):
            if isinstance(parsed_calibration.get(key), list):
                matrix = parsed_calibration[key]
                break
        camera_matrix = parsed_calibration.get("camera_matrix", parsed_calibration)
        if matrix is None and isinstance(camera_matrix, dict):
            if isinstance(camera_matrix.get("matrix_3x3"), list):
                matrix = camera_matrix["matrix_3x3"]
            elif "fx" in camera_matrix:
                return (
                    float(camera_matrix["fx"]),
                    float(camera_matrix["fy"]),
                    float(camera_matrix["cx"]),
                    float(camera_matrix["cy"]),
                    calib_wh,
                )

    if matrix is None:
        raise ValueError("No recognizable intrinsics (expected fx/fy/cx/cy or a 3x3)")
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"Intrinsics matrix must be 3x3, got {m.shape}")
    return float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2]), calib_wh


def resolve_K(
    width,
    height,
    intrinsics_json=None,
    perspective_cropping=False,
    calibration_source=None,
):
    """Resolve shared calibration with opt-in perspective cropping for a decoded video.

    Args:
        width: Decoded frame width in pixels.
        height: Decoded frame height in pixels.
        intrinsics_json: Parsed shared camera JSON (not a path); legacy pinhole K
            layouts from :func:`load_calibration` remain accepted with standard bbox crops.
        perspective_cropping: Return a camera even for PINHOLE; defaults to false.
        calibration_source: Source description for logging, e.g. the JSON file path.

    Returns:
        ``(K, source)`` where ``K`` is a pinhole ``(3, 3)`` float32 array or a
        CameraModel, and ``source`` describes its origin for logging. Non-pinhole
        cameras retain their projection for full-image undistortion or perspective
        cropping. Shared camera dimensions must match the video; legacy K inputs may be rescaled.
        With no calibration and standard bbox crops, estimate focal as ``max(width, height)``.
    """
    if perspective_cropping and intrinsics_json is None:
        raise ValueError("--perspective_cropping requires --intrinsics_json")
    if intrinsics_json is not None:
        parsed = _require_parsed_calibration(intrinsics_json)
        if perspective_cropping or (
            isinstance(parsed, dict)
            and parsed.keys()
            & {
                "model",
                "distortion_coefficients",
                "projection_model_type",
                "format_version",
            }
        ):
            calibration = load_camera_calibration(parsed)
        else:
            calibration = load_calibration(parsed)
        source = calibration_source or "calibration JSON"
        if isinstance(calibration, CameraModel):
            if (calibration.width, calibration.height) != (width, height):
                raise ValueError(
                    f"Video dimensions {width}x{height} do not match the "
                    f"{calibration.model_name} camera parameters {calibration.width}x{calibration.height}"
                )
            if perspective_cropping or calibration.model_name != "PINHOLE":
                return calibration, source
            return calibration.K.astype(np.float32), source
        fx, fy, cx, cy, calib_wh = calibration
        if calib_wh is not None and (calib_wh[0] != width or calib_wh[1] != height):
            sx, sy = width / calib_wh[0], height / calib_wh[1]
            fx, cx = fx * sx, cx * sx
            fy, cy = fy * sy, cy * sy
            source += (
                f" (rescaled {calib_wh[0]:.0f}x{calib_wh[1]:.0f} -> {width}x{height}, "
                f"sx={sx:.4f}, sy={sy:.4f})"
            )
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
        return K, source

    K = estimate_K(width, height).numpy().astype(np.float32)
    source = "estimate_K heuristic max(W, H)"
    return K, source
