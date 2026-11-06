# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit a continuous thumb-only sensor map against human fingertip geometry."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

CHANNELS = (0, 1, 2, 4)
POSES = (
    "open",
    "fist",
    "thumb_curl",
    "thumb_index",
    "thumb_middle",
    "thumb_ring",
    "thumb_little",
)
FORMAT = "realhand-ffg-thumb-quadratic-v1"
FIT_MODEL = "noise-regularized-contact-surface-v6"
SENSOR_GAIN_SCALE = 2.0
OPEN_REPEAT_TOLERANCE_M = 0.015
CONTACT_MAX_DISTANCE_M = 0.035
CURVATURE_SCALE = 0.3
SENSOR_NOISE_RAD = np.deg2rad(1.0)
LOWER = np.array((-1.25, -1.2, 0.0, 0.0))
UPPER = np.array((1.0, 1.2, 1.8, 1.8))
BASE = np.array((0.025, -0.005, -0.020))
LENGTHS = np.array((0.035, 0.030, 0.026))
OPEN_ANGLES = np.array((-0.85, 0.0, 0.0, 0.0))


class ThumbFitError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def quadratic_features(inputs):
    inputs = np.asarray(inputs)
    products = [
        inputs[..., first] * inputs[..., second]
        for first in range(4)
        for second in range(first, 4)
    ]
    return np.concatenate((inputs, np.stack(products, axis=-1)), axis=-1)


def quadratic_feature_jacobian(inputs):
    inputs = np.asarray(inputs, dtype=float)
    linear = np.broadcast_to(np.eye(4), (*inputs.shape[:-1], 4, 4))
    products = []
    for first in range(4):
        for second in range(first, 4):
            gradient = np.zeros_like(inputs)
            gradient[..., first] += inputs[..., second]
            gradient[..., second] += inputs[..., first]
            products.append(gradient)
    return np.concatenate((linear, np.stack(products, axis=-2)), axis=-2)


def angle_delta(values, origin):
    return (np.asarray(values) - origin + np.pi) % (2 * np.pi) - np.pi


def calibration_fingerprint(poses, side="left"):
    payload = {key: poses[key] for key in (*POSES, "thumb_opposition", "spread")}
    if "open_repeat" in poses:
        payload["open_repeat"] = poses["open_repeat"]
    if "thumb_observations" in poses:
        payload["thumb_observations"] = poses["thumb_observations"]
    payload["fit_model"] = FIT_MODEL
    payload["sensor_gain_scale"] = SENSOR_GAIN_SCALE
    payload["sensor_noise_rad"] = float(SENSOR_NOISE_RAD)
    payload["curvature_scale"] = CURVATURE_SCALE
    if side == "right":
        payload["lateral_sensor_convention"] = "right-mirrored-v1"
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _rotation(angle, axis):
    vector = np.zeros(3)
    vector[axis] = angle
    return Rotation.from_rotvec(vector).as_matrix()


def _response(value, origin, endpoint):
    span = angle_delta(endpoint, origin)
    center = span * 0.5
    delta = center + angle_delta(angle_delta(value, origin), center)
    return np.clip(delta / np.where(np.abs(span) >= 0.05, span, 1), 0, 1) * (
        np.abs(span) >= 0.05
    )


def finger_joints(poses, values, finger, side="left"):
    """Canonical left-hand FK, kept equivalent to the native finger reconstruction."""
    sensor = 5 + 4 * finger
    origin = np.asarray(poses["open"])
    fist = np.asarray(poses["fist"])
    span = abs(angle_delta(poses["spread"][sensor], origin[sensor]))
    direction = 1 if side == "left" else -1
    spread = np.clip(
        direction * angle_delta(values[sensor], origin[sensor]) / max(span, 0.10), -1, 1
    )
    spread *= (0.25, 0.08, 0.12, 0.30)[finger]
    base = _response(values[sensor + 1], origin[sensor + 1], fist[sensor + 1]) * 1.5
    distal = _response(values[sensor + 3], origin[sensor + 3], fist[sensor + 3]) * 1.1
    point = np.array(((0.033, 0.011, -0.011, -0.031)[finger], 0, -0.015))
    rotation = _rotation(spread, 1)
    point += rotation @ np.array((0, 0, -(0.070, 0.075, 0.070, 0.060)[finger] + 0.015))
    lengths = (
        (0.045, 0.026, 0.018),
        (0.050, 0.030, 0.019),
        (0.046, 0.027, 0.018),
        (0.034, 0.020, 0.016),
    )[finger]
    points = [point.copy()]
    for angle, length in zip((base, distal * 1.5, distal), lengths):
        rotation = rotation @ _rotation(-angle, 0)
        point += rotation @ np.array((0, 0, -length))
        points.append(point.copy())
    return np.asarray(points)


def finger_tip(poses, values, finger, side="left"):
    return finger_joints(poses, values, finger, side)[-1]


def thumb_joints(angles, base=BASE, lengths=LENGTHS):
    rotation = _rotation(angles[0], 1) @ _rotation(angles[1], 2)
    point = np.asarray(base, dtype=float).copy()
    points = [point.copy()]
    for bend, length in zip((0, angles[2], angles[3]), lengths):
        rotation = rotation @ _rotation(-bend, 0)
        point += rotation @ np.array((0, 0, -length))
        points.append(point.copy())
    return np.asarray(points)


def thumb_tip_positions(angles, lengths=LENGTHS):
    """Batched equivalent of thumb_joints(...)[-1]."""
    a, b, c, d = np.asarray(angles).T
    y = -lengths[1] * np.sin(c) - lengths[2] * np.sin(c + d)
    z = -lengths[0] - lengths[1] * np.cos(c) - lengths[2] * np.cos(c + d)
    x = -np.sin(b) * y
    return BASE + np.column_stack(
        (np.cos(a) * x + np.sin(a) * z, np.cos(b) * y, -np.sin(a) * x + np.cos(a) * z)
    )


def mapped_angles(mapping, values):
    raw = np.asarray(values)[list(CHANNELS)]
    minimum = np.asarray(mapping["input_min"])
    maximum = np.asarray(mapping["input_max"])
    center = (minimum + maximum) * 0.5
    delta = center + angle_delta(
        angle_delta(raw, np.asarray(mapping["input_origin"])), center
    )
    inputs = np.clip(delta, minimum, maximum) / mapping["input_scale"]
    return np.clip(
        np.asarray(mapping["coefficients"]) @ np.r_[1.0, quadratic_features(inputs)],
        LOWER,
        UPPER,
    )


def fit_thumb_mapping(poses, side="left"):
    if side not in ("left", "right"):
        raise ValueError("Thumb calibration side must be left or right")
    values = np.asarray([poses[key] for key in POSES], dtype=float)
    if values.shape != (7, 21) or not np.all(np.isfinite(values)):
        raise ValueError("Thumb fitting requires seven finite 21-angle captures")
    observations = [{"pose": key, "values": poses[key]} for key in POSES]
    if "open_repeat" in poses:
        observations.append({"pose": "open", "values": poses["open_repeat"]})
    observations.extend(poses.get("thumb_observations", ()))
    if any(s.get("pose") not in POSES for s in observations):
        raise ValueError("Thumb observations must use the calibration pose names")
    for sample in observations:
        row = np.asarray(sample["values"], dtype=float)
        if row.shape != (21,) or not np.all(np.isfinite(row)):
            raise ValueError("Thumb observations must contain 21 finite angles")
    raw = np.asarray([s["values"] for s in observations], dtype=float)
    if raw.shape != (len(observations), 21) or not np.all(np.isfinite(raw)):
        raise ValueError("Thumb observations must contain 21 finite angles")
    names = np.asarray([s["pose"] for s in observations])
    origin = values[0, list(CHANNELS)]
    # Canonical relative inputs keep numerical optimizers invariant to constant sensor zero offsets.
    # Twelve decimals are well below the native float32 sensor resolution.
    deltas = np.round(angle_delta(raw[:, list(CHANNELS)], origin), 12)
    if np.any(np.ptp(deltas, axis=0) >= np.pi):
        raise ValueError(
            "Thumb captures span an ambiguous angular arc; repeat calibration"
        )
    minimum = np.min(deltas, axis=0) - 2 * SENSOR_NOISE_RAD
    maximum = np.max(deltas, axis=0) + 2 * SENSOR_NOISE_RAD
    if np.any(maximum - minimum >= np.pi):
        raise ValueError("Thumb input uncertainty spans an ambiguous angular arc")
    # Native loaders require the normalization scale to cover the full clipped arc, including uncertainty.
    scales = np.maximum(np.maximum(np.abs(minimum), np.abs(maximum)), 0.10)
    normalized = deltas / scales
    features = quadratic_features(normalized)
    anchor_features = features[3:7]
    if (
        min(
            np.linalg.norm(a - b)
            for i, a in enumerate(anchor_features)
            for b in anchor_features[i + 1 :]
        )
        < 0.08
    ):
        raise ValueError(
            "Thumb contact captures are indistinguishable; distinct thumb readings are required"
        )
    is_contact = np.isin(names, POSES[3:])
    contact_raw = raw[is_contact]
    contact_names = names[is_contact]
    targets = np.round(
        np.asarray(
            [
                finger_tip(poses, v, POSES[3:].index(n), side)
                for v, n in zip(contact_raw, contact_names)
            ]
        ),
        12,
    )
    directions = BASE - targets
    weak_goals = (
        targets + directions / np.linalg.norm(directions, axis=1)[:, None] * 0.012
    )
    is_open = names == "open"
    is_endpoint = np.isin(names, ("fist", "thumb_curl"))
    prior = np.column_stack(
        (
            -0.85 + 1.2 * _response(raw[:, 0], origin[0], poses["thumb_opposition"][0]),
            -0.7 * _response(raw[:, 1], origin[1], poses["thumb_opposition"][1]),
            1.1 * _response(raw[:, 2], origin[2], poses["thumb_curl"][2]),
            1.2 * _response(raw[:, 4], origin[3], poses["thumb_curl"][4]),
        )
    )
    prior[is_open] = OPEN_ANGLES
    prior = np.round(prior, 12)
    linear = np.linalg.lstsq(normalized, prior - OPEN_ANGLES, rcond=None)[0].T
    initial = np.r_[
        np.column_stack((np.clip(linear, -1.7, 1.7), np.zeros((4, 10)))).ravel(), 1.1
    ]
    n = int(np.sum(is_contact))
    noise = np.random.default_rng(29).normal(0, SENSOR_NOISE_RAD, (n, 4, 4))
    noisy_inputs = normalized[is_contact, None, :] + noise / scales
    noisy_features = quadratic_features(noisy_inputs.reshape(-1, 4))
    sensor_jacobians = quadratic_feature_jacobian(normalized[::3]) / scales
    # Weak curvature and cross-channel shrinkage limit correlated-sensor overfitting.
    features[:, 4:] *= CURVATURE_SCALE
    noisy_features[:, 4:] *= CURVATURE_SCALE
    sensor_jacobians[:, 4:] *= CURVATURE_SCALE

    def unpack(p):
        weights = p[:56].reshape(4, 14)
        q = OPEN_ANGLES + features @ weights.T
        return weights, q, LENGTHS * p[56]

    def residual(p):
        weights, q, lengths = unpack(p)
        tips = thumb_tip_positions(np.clip(q[is_contact], LOWER, UPPER), lengths)
        noisy_q = np.clip(OPEN_ANGLES + noisy_features @ weights.T, LOWER, UPPER)
        jac = np.einsum("ij,njk->nik", weights, sensor_jacobians)
        return np.r_[
            # Contact fixes surface distance, not an unmeasured normal toward the thumb root.
            (
                (np.linalg.norm(tips - targets, axis=1) - 0.012) / 0.0015 / np.sqrt(n)
            ).ravel(),
            ((tips - weak_goals) / 0.04 / np.sqrt(n)).ravel(),
            (
                (thumb_tip_positions(noisy_q, lengths) - np.repeat(tips, 4, axis=0))
                / 0.003
                / np.sqrt(n * 4)
            ).ravel(),
            ((q[is_open] - prior[is_open]) / 0.12 / np.sqrt(np.sum(is_open))).ravel(),
            (
                (q[is_endpoint] - prior[is_endpoint])
                / 0.10
                / np.sqrt(np.sum(is_endpoint))
            ).ravel(),
            (weights / np.r_[np.full(4, 2.0), np.full(10, 0.6)] / np.sqrt(56)).ravel(),
            (jac / SENSOR_GAIN_SCALE / np.sqrt(len(jac))).ravel(),
            (jac * (1 - np.eye(4)) / np.sqrt(len(jac))).ravel(),
            (np.maximum(LOWER - q, 0) / 0.05 / np.sqrt(len(q))).ravel(),
            (np.maximum(q - UPPER, 0) / 0.05 / np.sqrt(len(q))).ravel(),
            (p[56] - 1) / 0.2,
        ]

    bounds = (np.r_[np.full(56, -1.75), 0.85], np.r_[np.full(56, 1.75), 1.15])
    result = least_squares(residual, initial, bounds=bounds, max_nfev=500, ftol=1e-6)
    curl_features = quadratic_features(
        np.round(
            angle_delta(np.asarray(poses["thumb_curl"])[list(CHANNELS)], origin), 12
        )
        / scales
    )
    curl_features[4:] *= CURVATURE_SCALE

    def curl_quality(p):
        return (OPEN_ANGLES + curl_features @ p[:56].reshape(4, 14).T)[2:] - (
            0.92,
            1.02,
        )

    repeat_features = None
    if "open_repeat" in poses:
        repeated = (
            np.round(
                angle_delta(np.asarray(poses["open_repeat"])[list(CHANNELS)], origin),
                12,
            )
            / scales
        )
        repeat_features = quadratic_features(repeated)
        repeat_features[4:] *= CURVATURE_SCALE

    def quality(p):
        constraints = curl_quality(p).tolist()
        if repeat_features is not None:
            q = np.clip(
                OPEN_ANGLES + repeat_features @ p[:56].reshape(4, 14).T, LOWER, UPPER
            )
            length = LENGTHS * p[56]
            shift = np.linalg.norm(
                thumb_tip_positions(q[None], length)[0]
                - thumb_tip_positions(OPEN_ANGLES[None], length)[0]
            )
            constraints.append(1 - (shift / (OPEN_REPEAT_TOLERANCE_M * 0.95)) ** 2)
        return np.asarray(constraints)

    if np.all(np.isfinite(result.x)) and np.min(quality(result.x)) < 0:
        result = minimize(
            lambda p: np.mean(residual(p) ** 2),
            result.x,
            method="SLSQP",
            bounds=list(zip(*bounds)),
            constraints={"type": "ineq", "fun": quality},
            options={"maxiter": 250, "ftol": 1e-8},
        )
    if not np.all(np.isfinite(result.x)):
        raise ThumbFitError(
            f"{side.upper()} thumb fit produced non-finite coefficients", {"side": side}
        )
    weights, _, lengths = unpack(result.x)
    weights = weights.copy()
    weights[:, 4:] *= CURVATURE_SCALE
    mapping = {
        "format": FORMAT,
        "fit_model": FIT_MODEL,
        "sensor_gain_scale": SENSOR_GAIN_SCALE,
        "calibration_sha256": calibration_fingerprint(poses, side),
        "input_origin": origin.tolist(),
        "input_scale": scales.tolist(),
        "input_min": minimum.tolist(),
        "input_max": maximum.tolist(),
        "coefficients": np.column_stack((OPEN_ANGLES, weights)).tolist(),
        "base_position": BASE.tolist(),
        "lengths": lengths.tolist(),
    }
    gaps = np.linalg.norm(
        thumb_tip_positions(
            np.asarray([mapped_angles(mapping, v) for v in contact_raw]), lengths
        )
        - targets,
        axis=1,
    )
    anchors = []
    for name in POSES[3:]:
        v = poses[name]
        anchors.append(
            float(
                np.linalg.norm(
                    thumb_joints(mapped_angles(mapping, v), BASE, lengths)[-1]
                    - finger_tip(poses, v, POSES[3:].index(name), side)
                )
            )
        )
    report = {
        "side": side,
        "solver": "noise_regularized_contact_surface",
        "solver_converged": bool(result.success),
        "solver_message": str(result.message),
        "observation_count": len(raw),
        "tip_center_distances_mm": (np.asarray(anchors) * 1000).tolist(),
        "contact_surface_error_mm": (
            np.abs(np.asarray(anchors) - 0.012) * 1000
        ).tolist(),
        "contact_surface_error_before_mm": (
            np.abs(
                np.linalg.norm(thumb_tip_positions(prior[3:7]) - targets[:4], axis=1)
                - 0.012
            )
            * 1000
        ).tolist(),
        "contact_gap_p95_mm": float(np.percentile(gaps, 95) * 1000),
        "training_only": True,
    }
    if "open_repeat" in poses:
        p = thumb_joints(mapped_angles(mapping, poses["open_repeat"]), BASE, lengths)[
            -1
        ]
        report["repeat_open_training_tip_mm"] = float(
            np.linalg.norm(p - thumb_joints(OPEN_ANGLES, BASE, lengths)[-1]) * 1000
        )
    mapping["fit_report"] = report
    if (
        np.percentile(gaps, 95) > CONTACT_MAX_DISTANCE_M
        or np.min(quality(result.x)) < -0.005
    ):
        raise ThumbFitError(
            f"{side.upper()} thumb fit fails repeated-contact/curl quality checks; existing calibration is unchanged",
            report,
        )
    return mapping
