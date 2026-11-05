# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Image, landmark, metric-size and temporal fitting constraints."""

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def target_distance(mask, weights):
    binary = (mask > 0.5).astype(np.uint8)
    edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge *= (weights > 0.5).astype(np.uint8)
    if edge.sum() == 0:
        return np.zeros_like(mask, np.float32)
    return cv2.distanceTransform(1 - edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(
        np.float32
    )


def silhouette_loss(
    pred,
    target,
    weight,
    distance,
    balance=None,
    coverage_weight=2.0,
    wrist_region=None,
    wrist_weight=2.0,
):
    """Multiscale overlap plus observed-boundary distance; all terms use trusted pixels."""
    trust = weight
    if balance is not None:
        weight = weight * balance
    intersection = (pred * target * weight).sum((-2, -1))
    denom = ((pred + target) * weight).sum((-2, -1))
    dice = 1 - (2 * intersection + 1) / (denom + 1)
    error = ((pred - target).square() * weight).sum((-2, -1)) / weight.sum(
        (-2, -1)
    ).clamp_min(1)
    p = pred[:, None]
    # A boundary is trusted only if its whole neighborhood is observed.
    edge_weight = 1 - F.max_pool2d(1 - trust[:, None], 3, 1, 1)[:, 0]
    if balance is not None:
        edge_weight = edge_weight * balance
    edge = (F.max_pool2d(p, 3, 1, 1) + F.max_pool2d(-p, 3, 1, 1))[:, 0] * edge_weight
    boundary = (
        (edge * distance).sum((-2, -1))
        / edge.sum((-2, -1)).clamp_min(1)
        / pred.shape[-1]
    )
    coarse = torch.zeros_like(dice)
    for kernel in (4, 8):
        ww = F.avg_pool2d(weight[:, None], kernel)[:, 0]
        pp = F.avg_pool2d((pred * weight)[:, None], kernel)[:, 0] / ww.clamp_min(1e-6)
        tt = F.avg_pool2d((target * weight)[:, None], kernel)[:, 0] / ww.clamp_min(1e-6)
        coarse += ((pp - tt).square() * ww).sum((-2, -1)) / ww.sum((-2, -1)).clamp_min(
            1
        )
    # Joint distances do not constrain surface width. Normalize missed visible
    # hand area by foreground area, so background and the large palm cannot
    # dilute under-coverage. Existing overlap/background terms resist expansion.
    foreground_mass = (target * trust).sum((-2, -1))
    missing = (F.relu(target - pred) * trust).sum((-2, -1)) / foreground_mass.clamp_min(
        1
    )
    wrist_loss = torch.zeros_like(dice)
    if wrist_region is not None and wrist_weight > 0:
        region = wrist_region * trust
        mass = (target * region).sum((-2, -1))
        missed = (F.relu(target - pred) * region).sum((-2, -1))
        spill = (F.relu(pred - target) * region).sum((-2, -1))
        # GrabCut can retain a few isolated pixels while missing the proximal
        # palm. Dividing by that tiny foreground mass makes background spill
        # dominate valid landmarks and pulls the mesh away from the wrist.
        # A floor relative to the trusted band bounds this amplification. The
        # objective and gradients are identical for adequately supported masks.
        normalizer = torch.maximum(mass, 0.1 * region.sum((-2, -1))).clamp_min(1)
        wrist_loss = (missed + 0.5 * spill) / normalizer * (mass >= 3)
    return (
        dice
        + error
        + 2 * boundary
        + coarse
        + coverage_weight * missing
        + wrist_weight * wrist_loss
    )


def landmark_loss(pred, target, confidence, tolerance=0.008, scale=0.02):
    finite = torch.isfinite(target).all(-1)
    safe = torch.nan_to_num(target)
    residual = torch.linalg.vector_norm(pred - safe, dim=-1)
    outside = F.relu(residual - tolerance) / scale
    robust = F.smooth_l1_loss(
        outside, torch.zeros_like(outside), reduction="none", beta=1.0
    )
    weights = confidence * finite
    return (robust * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)


def reprojection_loss(pred, target, confidence, K, tolerance=2.0, scale=10.0):
    """Robust image-plane correspondence, excluding missing/behind-camera points."""
    from .geometry import project

    finite = torch.isfinite(target).all(-1) & (target[..., 2] > 0)
    safe = torch.where(finite[..., None], target, torch.ones_like(target))
    residual = torch.linalg.vector_norm(project(pred, K) - project(safe, K), dim=-1)
    outside = F.relu(residual - tolerance) / scale
    robust = F.smooth_l1_loss(
        outside, torch.zeros_like(outside), reduction="none", beta=1.0
    )
    weights = confidence * finite
    return (robust * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)


# Metric hand dimensions and bounded native wrist placement.

# Four palm radii, palm width, and three segments of each finger. No regressor
# or other hand model is involved; these are distances between input indices.
PAIRS = (
    [(0, i) for i in (5, 9, 13, 17)]
    + [(5, 17)]
    + [(i, i + 1) for start in (1, 5, 9, 13, 17) for i in range(start, start + 3)]
)
DIMENSION_NAMES = [
    "wrist_index",
    "wrist_middle",
    "wrist_ring",
    "wrist_pinky",
    "palm_width",
] + [
    f"{finger}_segment_{i + 1}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for i in range(3)
]


def dimension_targets(points, confidence, valid):
    """Confidence-weighted sequence medians; unreliable/missing edges stay inactive."""
    values, weights = [], []
    for a, b in PAIRS:
        length = np.linalg.norm(points[:, a] - points[:, b], axis=-1)
        weight = np.minimum(confidence[:, a], confidence[:, b])
        good = (
            valid
            & np.isfinite(length)
            & (length > 0.001)
            & np.isfinite(weight)
            & (weight > 0)
        )
        if not good.any():
            values.append(1.0)
            weights.append(0.0)
            continue
        x, w = length[good], weight[good]
        order = np.argsort(x)
        x, w = x[order], w[order]
        middle = x[np.searchsorted(np.cumsum(w), w.sum() / 2)]
        # Attenuate unstable dimensions; do not let outliers redefine shared size.
        spread = np.median(np.abs(x - middle)) / middle
        reliability = np.mean(w) / (1 + (spread / 0.05) ** 2)
        values.append(float(middle))
        weights.append(float(reliability))
    return np.asarray(values, np.float32), np.asarray(weights, np.float32)


def dimensions(points):
    a, b = zip(*PAIRS)
    return torch.linalg.vector_norm(
        points[..., list(a), :] - points[..., list(b), :], dim=-1
    )


def metric_size_loss(points, target, weights, tolerance=0.02, shrink_weight=3.0):
    """Balanced relative dimension penalty: short phalanges cannot hide a small palm."""
    relative = dimensions(points) / target.clamp_min(1e-6) - 1
    residual = F.relu(relative.abs() - tolerance) / 0.05
    loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
    loss = loss * torch.where(relative < -tolerance, shrink_weight, 1.0)
    # Balance palm and fingers instead of allowing 15 finger segments to outweigh
    # all five palm measurements. Each frame contributes equally.
    groups = []
    for lo, hi in ((0, 5), (5, len(PAIRS))):
        w = weights[lo:hi]
        groups.append((loss[..., lo:hi] * w).sum(-1) / w.sum().clamp_min(1e-6))
    return (groups[0] + groups[1]) * 0.5


@torch.no_grad()
def constrain_wrist_(translation, wrist, confidence, radius):
    """Project native root translation onto a metric ball around a valid wrist.

    A negative radius disables the constraint for explicit legacy comparisons.
    The projection operates on parameters during optimization, never on exports.
    """
    if radius < 0:
        return
    good = torch.isfinite(wrist).all(-1) & (confidence > 0) & (wrist[..., 2] > 0)
    safe = torch.where(good[..., None], wrist, translation)
    delta = translation - safe
    ratio = (radius / torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-12)).clamp(
        max=1
    )
    translation.copy_(
        torch.where(good[..., None], safe + delta * ratio[..., None], translation)
    )


def preservation_report(
    predicted, native_roots, translations, points, confidence, valid
):
    """Audit metric dimensions and root preservation without re-aligning outputs."""
    result = {}
    for side, name in enumerate(("left", "right")):
        target, weight = dimension_targets(
            points[:, side], confidence[:, side], valid[:, side]
        )
        ids = valid[:, side]
        good_root = (
            ids & (confidence[:, side, 0] > 0) & np.isfinite(points[:, side, 0]).all(-1)
        )
        error = (
            np.linalg.norm(
                native_roots[good_root, side] - points[good_root, side, 0], axis=-1
            )
            * 1000
        )
        q = predicted[ids, side]
        a, b = zip(*PAIRS)
        lengths = np.linalg.norm(q[:, a] - q[:, b], axis=-1)
        dims = {}
        for i, label in enumerate(DIMENSION_NAMES):
            if weight[i] <= 0 or not len(q):
                continue
            ratios = lengths[:, i] / target[i]
            dims[label] = {
                "target_mm": float(target[i] * 1000),
                "median_mm": float(np.median(lengths[:, i]) * 1000),
                "median_ratio": float(np.median(ratios)),
                "p05_ratio": float(np.percentile(ratios, 5)),
                "p95_ratio": float(np.percentile(ratios, 95)),
            }
        root_consistency = np.abs(native_roots[ids, side] - translations[ids, side])
        result[name] = {
            "wrist_mean_mm": float(error.mean()) if len(error) else None,
            "wrist_max_mm": float(error.max()) if len(error) else None,
            "native_root_translation_max_error_m": float(root_consistency.max())
            if root_consistency.size
            else None,
            "dimensions": dims,
        }
    return result


def native_palm_floor_loss(points, target, weights, tolerance=0.02):
    """Prevent palm collapse when source wrist 0 is not a native-root target.

    Target radii come from the native neutral rig scaled by observed MCP width.
    Source wrist 0 must not define this target when its correspondence is ignored.
    A lower bound permits longer subject-specific native palm geometry.
    """
    relative = dimensions(points)[..., :4] / target[:4].clamp_min(1e-6)
    deficit = F.relu(1 - tolerance - relative) / 0.05
    penalty = F.smooth_l1_loss(deficit, torch.zeros_like(deficit), reduction="none")
    return (penalty * weights[:4]).sum(-1) / weights[:4].sum().clamp_min(1e-6)


# Bounded observation targets for the temporal objective.


def smooth_observation_targets(
    points, confidence, timestamps, ids, max_gap, window=0.067, max_correction=0.002
):
    """Local quadratic regression without coupling gaps or missing observations.

    Only the temporal objective uses this target. Spatial/image observations are
    untouched. Centered time windows avoid a causal filter's delay; a quadratic
    preserves constant acceleration. Bound every correction in camera-space
    metres so abrupt motion cannot be replaced by an unconstrained smooth path.
    """
    points = np.asarray(points)
    result = points.copy()
    if window <= 0 or max_correction <= 0 or len(points) < 4:
        return result
    times = np.asarray(timestamps, dtype=np.float64)
    dt = np.diff(times)
    connected = (np.diff(ids) == 1) & (dt > 0) & (dt <= max_gap)
    valid = (
        np.isfinite(points).all(-1)
        & (points[..., 2] > 0)
        & np.isfinite(confidence)
        & (confidence > 0)
    )
    for joint in range(points.shape[1]):
        links = connected & valid[:-1, joint] & valid[1:, joint]
        boundaries = np.r_[0, np.flatnonzero(~links) + 1, len(points)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop - start < 4:
                continue
            segment_times = times[start:stop]
            for frame in range(start, stop):
                lo = start + np.searchsorted(
                    segment_times, times[frame] - window - 1e-9
                )
                hi = start + np.searchsorted(
                    segment_times, times[frame] + window + 1e-9, side="right"
                )
                if hi - lo < 4:
                    continue
                t = (times[lo:hi] - times[frame]) / window
                design = np.stack((np.ones_like(t), t, t * t), axis=-1)
                weights = np.sqrt(confidence[lo:hi, joint].clip(0, 1))
                # Relative coordinates avoid losing precision at large camera translations.
                values = points[lo:hi, joint] - points[frame, joint]
                correction = np.linalg.lstsq(
                    design * weights[:, None], values * weights[:, None], rcond=None
                )[0][0]
                length = np.linalg.norm(correction)
                correction *= min(1.0, max_correction / max(length, 1e-12))
                result[frame, joint] += correction
    return result
