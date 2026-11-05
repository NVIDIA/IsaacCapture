# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Camera utility functions."""

import math
from contextlib import nullcontext

import torch

CAMERA_DEPTH_MODES = ("s", "log_Z", "log_lambda")


def canonical_camera_depth_mode(mode: str) -> str:
    """Return the canonical camera-depth parameterization name.

    ``log_Z`` keeps its original spelling, while a few lower-case aliases
    are accepted so CLI/config callers do not accidentally create
    semantically distinct modes.
    """

    normalized = str(mode).strip()
    aliases = {
        "s": "s",
        "log_Z": "log_Z",
        "log_z": "log_Z",
        "logz": "log_Z",
        "log_depth": "log_Z",
        "log-depth": "log_Z",
        "log_lambda": "log_lambda",
        "loglambda": "log_lambda",
        "log-lambda": "log_lambda",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unknown camera_depth_mode={mode!r}; expected one of {CAMERA_DEPTH_MODES}"
        ) from error


def _autocast_disabled(device: torch.device):
    """Keep calibrated pinhole arithmetic out of lower-precision autocast."""

    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def perspective_projection(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Standard pinhole camera projection performed in FP32.

    Full-image pixel coordinates can legitimately exceed the FP16 finite
    range for points close to the camera plane.  ``torch.einsum`` is an
    autocast-eligible operation, so merely passing FP32 inputs is not enough
    to prevent an enclosing mixed-precision context from downcasting it.
    Keep both the perspective divide and homogeneous matrix multiply in an
    explicitly autocast-disabled FP32 block.

    Args:
        points: (*, J, 3)
        K: (*, 3, 3)
    Returns:
        p2d: (*, J, 2)
    """
    with _autocast_disabled(points.device):
        points_f = points.float()
        intrinsics_f = K.float()
        projected = points_f / points_f[..., -1:].clamp(min=1e-6)
        projected = torch.einsum(
            "...ij,...kj->...ki",
            intrinsics_f,
            projected,
        )
        return projected[..., :2]


def compute_transl_full_cam(
    pred_cam: torch.Tensor, bbx_xys: torch.Tensor, K_fullimg: torch.Tensor
) -> torch.Tensor:
    """Convert weak-perspective camera parameters to full-perspective translation.

    Args:
        pred_cam: (*, 3) [s, tx, ty]
        bbx_xys: (*, 3) [cx, cy, size]
        K_fullimg: (*, 3, 3)
    Returns:
        cam_t: (*, 3)
    """
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)
    return torch.stack([tx + cx, ty + cy, tz], dim=-1)


def camera_translation_for_intrinsics(
    pred_cam: torch.Tensor,
    pred_cam_t: torch.Tensor,
    bbx_xys: torch.Tensor,
    K_fullimg: torch.Tensor,
    camera_mode: str,
) -> torch.Tensor:
    """Select the translation contract for a projection/eval consumer.

    A weak-perspective prediction is a crop camera that must be decoded under
    the intrinsics used by the consumer.  In every full-perspective mode,
    ``pred_cam_t`` is authoritative: its depth may come from ``log_Z`` or
    ``log_lambda`` and its lateral values may have been replaced by the direct
    2D-keypoint solver.  Re-decoding public ``pred_cam`` would silently discard
    those semantics.

    This distinction is also important for alternate-intrinsics diagnostics.
    Such panels reinterpret the weak crop camera, but for a full-perspective
    model they vary only the projection intrinsics and preserve the predicted
    physical translation.
    """

    if str(camera_mode) == "weak_perspective":
        return compute_transl_full_cam(
            pred_cam,
            bbx_xys,
            K_fullimg,
        )
    return pred_cam_t


def legacy_camera_to_depth_parameter(
    pred_cam: torch.Tensor,
    bbx_xys: torch.Tensor,
    K_fullimg: torch.Tensor,
    camera_depth_mode: str,
    *,
    min_positive: float = 1e-8,
) -> torch.Tensor:
    """Encode legacy ``[s, dx, dy]`` camera values in a depth mode.

    The calibrated EgoHand camera convention is

    ``Z = 2 fx / (s b)`` and ``lambda = b Z / fx = 2 / s``,

    where ``b`` is the square crop size in full-image pixels.  Lateral
    channels retain their historical metric-offset semantics in every mode.

    Args:
        pred_cam: ``[..., 3]`` legacy camera ``[s, dx, dy]``.
        bbx_xys: ``[..., 3]`` square boxes ``[cx, cy, b]``.
        K_fullimg: ``[..., 3, 3]`` calibrated intrinsics.
        camera_depth_mode: ``s``, ``log_Z``, or ``log_lambda``.
        min_positive: Numerical floor used before logarithms.

    Returns:
        ``[..., 3]`` mode-space camera parameters.
    """

    mode = canonical_camera_depth_mode(camera_depth_mode)
    if min_positive <= 0 or not math.isfinite(min_positive):
        raise ValueError("min_positive must be finite and positive")
    if pred_cam.shape[-1:] != (3,):
        raise ValueError(f"pred_cam must end in 3 values, got {tuple(pred_cam.shape)}")
    batch_shape = pred_cam.shape[:-1]
    if bbx_xys.shape != batch_shape + (3,):
        raise ValueError(
            f"bbx_xys must have shape {batch_shape + (3,)}, got {tuple(bbx_xys.shape)}"
        )
    if K_fullimg.shape != batch_shape + (3, 3):
        raise ValueError(
            f"K_fullimg must have shape {batch_shape + (3, 3)}, got "
            f"{tuple(K_fullimg.shape)}"
        )
    if not (pred_cam.device == bbx_xys.device == K_fullimg.device):
        raise ValueError("pred_cam, bbx_xys, and K_fullimg must share a device")
    if mode == "s":
        return pred_cam

    with _autocast_disabled(pred_cam.device):
        camera = pred_cam.float()
        boxes = bbx_xys.float()
        intrinsics = K_fullimg.float()
        scale = camera[..., 0]
        safe_scale = scale.clamp_min(float(min_positive))
        if mode == "log_Z":
            fx = intrinsics[..., 0, 0]
            safe_box_size = boxes[..., 2].clamp_min(float(min_positive))
            depth = 2.0 * fx / (safe_scale * safe_box_size)
            scalar = torch.log(depth.clamp_min(float(min_positive)))
        else:
            scalar = math.log(2.0) - torch.log(safe_scale)
        return torch.cat(
            [scalar.unsqueeze(-1), camera[..., 1:]],
            dim=-1,
        )


def depth_parameter_to_legacy_camera(
    camera_parameter: torch.Tensor,
    bbx_xys: torch.Tensor,
    K_fullimg: torch.Tensor,
    camera_depth_mode: str,
    *,
    log_value_bounds: tuple[float, float] = (-10.0, 10.0),
) -> torch.Tensor:
    """Decode a camera-depth parameter while preserving the public API.

    The returned tensor always follows the historical ``[s, dx, dy]``
    contract.  This is deliberate: callbacks, renderers, inference wrappers,
    and old analysis scripts can continue to consume ``output['pred_cam']``
    without knowing how the network internally predicts depth.

    For log modes, a broad straight-through clamp is applied before ``exp``.
    It has no effect in the physically relevant range, prevents exponential
    overflow during early optimization, and retains a nonzero gradient if a
    raw prediction temporarily leaves the range.
    """

    mode = canonical_camera_depth_mode(camera_depth_mode)
    if camera_parameter.shape[-1:] != (3,):
        raise ValueError(
            "camera_parameter must end in 3 values, got "
            f"{tuple(camera_parameter.shape)}"
        )
    batch_shape = camera_parameter.shape[:-1]
    if bbx_xys.shape != batch_shape + (3,):
        raise ValueError(
            f"bbx_xys must have shape {batch_shape + (3,)}, got {tuple(bbx_xys.shape)}"
        )
    if K_fullimg.shape != batch_shape + (3, 3):
        raise ValueError(
            f"K_fullimg must have shape {batch_shape + (3, 3)}, got "
            f"{tuple(K_fullimg.shape)}"
        )
    if not (camera_parameter.device == bbx_xys.device == K_fullimg.device):
        raise ValueError("camera_parameter, bbx_xys, and K_fullimg must share a device")
    if mode == "s":
        return camera_parameter

    with _autocast_disabled(camera_parameter.device):
        parameter = effective_camera_depth_parameter(
            camera_parameter,
            mode,
            log_value_bounds=log_value_bounds,
        )
        safe_log_value = parameter[..., 0]
        if mode == "log_Z":
            depth = torch.exp(safe_log_value)
            fx = K_fullimg.float()[..., 0, 0]
            box_size = bbx_xys.float()[..., 2]
            scale = 2.0 * fx / (depth * box_size)
        else:
            scale = 2.0 * torch.exp(-safe_log_value)
        return torch.cat(
            [scale.unsqueeze(-1), parameter[..., 1:]],
            dim=-1,
        )


def effective_camera_depth_parameter(
    camera_parameter: torch.Tensor,
    camera_depth_mode: str,
    *,
    log_value_bounds: tuple[float, float] = (-10.0, 10.0),
) -> torch.Tensor:
    """Return the exact mode-space value used by physical decoding.

    ``camera_parameter`` is the unconstrained/native head output. For log
    modes this function replaces its scalar value by a broad STE-bounded
    value. Exposing both raw and effective tensors prevents diagnostics from
    applying ``exp(raw)`` when projection actually used ``exp(effective)``.
    """

    mode = canonical_camera_depth_mode(camera_depth_mode)
    if camera_parameter.shape[-1:] != (3,):
        raise ValueError(
            "camera_parameter must end in 3 values, got "
            f"{tuple(camera_parameter.shape)}"
        )
    if mode == "s":
        return camera_parameter
    if len(log_value_bounds) != 2:
        raise ValueError("log_value_bounds must contain exactly two values")
    log_min, log_max = (
        float(log_value_bounds[0]),
        float(log_value_bounds[1]),
    )
    if not math.isfinite(log_min) or not math.isfinite(log_max) or log_min >= log_max:
        raise ValueError("log_value_bounds must contain two finite increasing values")
    parameter = camera_parameter.float()
    raw_log_value = parameter[..., 0:1]
    finite_raw = torch.isfinite(raw_log_value)
    finite_value = torch.nan_to_num(
        raw_log_value,
        nan=0.0,
        posinf=log_max,
        neginf=log_min,
    )
    clamped_log_value = finite_value.clamp(log_min, log_max)
    # Forward is exactly the bounded value, including for very large FP32
    # inputs. The common ``raw + (clamped - raw).detach()`` formulation
    # suffers catastrophic cancellation (e.g. 1e20 can decode as zero).
    # Non-finite raw predictions decode to a finite inert value but retain no
    # gradient here; the raw camera loss/health check still fails loudly.
    differentiable_raw = torch.where(
        finite_raw,
        raw_log_value,
        clamped_log_value.detach(),
    )
    effective_log_value = clamped_log_value.detach() + (
        differentiable_raw - differentiable_raw.detach()
    )
    return torch.cat(
        [effective_log_value, parameter[..., 1:]],
        dim=-1,
    )


def get_a_pred_cam(
    transl: torch.Tensor, bbx_xys: torch.Tensor, K_fullimg: torch.Tensor
) -> torch.Tensor:
    """Inverse of compute_transl_full_cam: full translation -> weak-perspective params.

    Args:
        transl: (*, 3)
        bbx_xys: (*, 3)
        K_fullimg: (*, 3, 3)
    Returns:
        pred_cam: (*, 3) [s, tx, ty]
    """
    f = K_fullimg[..., 0, 0]
    cx = K_fullimg[..., 0, 2]
    cy = K_fullimg[..., 1, 2]
    gt_s = 2 * f / (transl[..., 2] * bbx_xys[..., 2])
    gt_x = transl[..., 0] - transl[..., 2] / f * (bbx_xys[..., 0] - cx)
    gt_y = transl[..., 1] - transl[..., 2] / f * (bbx_xys[..., 1] - cy)
    return torch.stack([gt_s, gt_x, gt_y], dim=-1)
