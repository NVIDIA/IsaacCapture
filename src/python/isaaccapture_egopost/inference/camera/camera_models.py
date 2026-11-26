# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Shared camera protocol and JSON loader, with pinhole, rational, and fisheye projection.

KB4 and FISHEYE624 distort ray angle with an odd polynomial. FISHEYE624 adds tangential
and thin-prism offsets; KB4 uses only four radial coefficients. The FISHEYE624
offset equations follow Project Aria's FisheyeRadTanThinPrism model.
"""

from functools import lru_cache
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import warp as wp
from numpy.polynomial import Polynomial
from scipy.optimize import newton

# Iteration limit for inverse distortion. The FISHEYE624 offset tolerance follows
# Project Aria camera_projections/Common.h in normalized image coordinates.
_MAX_INVERSE_ITERATIONS = 50
_FISHEYE624_TOLERANCE = 1e-7
# Numerical tolerance for KB4 inversion, expressed in source-image pixels.
_KB4_MAX_PIXEL_ERROR = 1e-6
# OpenCV has less overhead for the small batches used by inverse-solver retries.
_CPU_KERNEL_MIN_RAYS = 256


@runtime_checkable
class CameraModel(Protocol):
    """Source-image projection with camera axes x right, y down, and z forward."""

    K: np.ndarray
    width: int
    height: int
    model_name: str

    def project(self, rays: np.ndarray) -> np.ndarray:
        """Project source-camera vectors into source pixels.

        Args:
            rays: Vectors shaped (..., 3).

        Returns:
            Pixels shaped (..., 2), with NaN outside the physical projection domain.
        """
        ...

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Convert source pixels into source-camera directions.

        Args:
            pixels: Source-image coordinates shaped (..., 2).

        Returns:
            Unit forward rays shaped (..., 3), with NaN for invalid inversions.
        """
        ...


class _FisheyeCamera:
    """Currently supports KB4 and FISHEYE624 fisheye cameras.

    Support can be expanded to other fisheye cameras by extending this class.
    """

    def __init__(
        self,
        K: np.ndarray,
        coefficients: np.ndarray,
        width: int,
        height: int,
        model_name: str,
    ) -> None:
        """Store validated calibration and initialize the angular polynomials.

        Args:
            K: Validated, copied zero-skew 3x3 intrinsics from the shared loader.
            coefficients: Four KB4 radial terms or all twelve FISHEYE624 terms.
            width: Image width in pixels.
            height: Image height in pixels.
            model_name: Camera model, ``KB4`` or ``FISHEYE624``.
        """
        self.K = K
        self.width, self.height, self.model_name = width, height, model_name
        self.radial = coefficients[:6] if model_name == "FISHEYE624" else coefficients
        self.tangential_prism = (
            coefficients[6:] if model_name == "FISHEYE624" else np.zeros(6)
        )
        self._radial_polynomial = np.concatenate(([1.0], self.radial))
        self._radial_derivative = self._radial_polynomial * np.arange(
            1, 2 * len(self._radial_polynomial), 2
        )

    def project(self, rays: np.ndarray) -> np.ndarray:
        """Project camera rays through radial and optional tangential/prism distortion.

        Args:
            rays: Camera-coordinate vectors shaped (..., 3), with positive z forward.

        Returns:
            Pixel coordinates shaped (..., 2); nonfinite and rear-facing rays produce NaN.
        """
        rays = np.asarray(rays, dtype=np.float64)
        flat_rays = rays.reshape(-1, 3)
        ray_xy_norm = np.hypot(flat_rays[:, 0], flat_rays[:, 1])
        theta = np.arctan2(ray_xy_norm, flat_rays[:, 2])
        distorted_radii = theta * np.polynomial.polynomial.polyval(
            theta * theta, self._radial_polynomial
        )
        xy_directions = np.divide(
            flat_rays[:, :2],
            ray_xy_norm[:, None],
            out=np.zeros_like(flat_rays[:, :2]),
            where=(ray_xy_norm[:, None] > 0) & np.isfinite(ray_xy_norm[:, None]),
        )
        distorted_normalized_coords = xy_directions * distorted_radii[:, None]
        if np.any(self.tangential_prism):
            # Add FISHEYE624 tangential/thin-prism offsets. KB4 has no such terms.
            distorted_normalized_coords = (
                distorted_normalized_coords
                + self._fisheye624_tangential_prism_offsets(distorted_normalized_coords)
            )
        pixel_coords = (
            distorted_normalized_coords * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]
        )
        pixel_coords[
            (flat_rays[:, 2] <= 0)
            | ~np.isfinite(flat_rays).all(axis=1)
            | ~np.isfinite(ray_xy_norm)
        ] = np.nan
        return pixel_coords.reshape(*rays.shape[:-1], 2)

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Invert optional tangential/prism offsets and the shared radial polynomial.

        Args:
            pixels: Source image coordinates shaped (..., 2).

        Returns:
            Unit forward rays shaped (..., 3), with NaN outside the lens domain
            or when inversion fails. KB4 inversion covers radii beyond pi/2.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        flat_pixels = pixels.reshape(-1, 2)
        valid_pixels = np.isfinite(flat_pixels).all(axis=1)
        distorted_normalized_coords = (
            np.where(valid_pixels[:, None], flat_pixels, self.K[:2, 2]) - self.K[:2, 2]
        )
        distorted_normalized_coords /= [self.K[0, 0], self.K[1, 1]]
        radial_coords = distorted_normalized_coords.copy()
        if np.any(self.tangential_prism):
            # Undo FISHEYE624 tangential/thin-prism offsets before radial inversion.
            p0, p1, s0, s1, s2, s3 = self.tangential_prism
            for _ in range(_MAX_INVERSE_ITERATIONS):
                offset_residual = (
                    distorted_normalized_coords
                    - radial_coords
                    - self._fisheye624_tangential_prism_offsets(radial_coords)
                )
                x, y = radial_coords.T
                radius_squared = x * x + y * y
                jacobian_xx = (
                    1
                    + 6 * p0 * x
                    + 2 * p1 * y
                    + 2 * s0 * x
                    + 4 * s1 * radius_squared * x
                )
                jacobian_xy = (
                    2 * p0 * y + 2 * p1 * x + 2 * s0 * y + 4 * s1 * radius_squared * y
                )
                jacobian_yx = (
                    2 * p1 * x + 2 * p0 * y + 2 * s2 * x + 4 * s3 * radius_squared * x
                )
                jacobian_yy = (
                    1
                    + 6 * p1 * y
                    + 2 * p0 * x
                    + 2 * s2 * y
                    + 4 * s3 * radius_squared * y
                )
                jacobian_determinant = (
                    jacobian_xx * jacobian_yy - jacobian_xy * jacobian_yx
                )
                offset_step_numerator = np.column_stack(
                    (
                        jacobian_yy * offset_residual[:, 0]
                        - jacobian_xy * offset_residual[:, 1],
                        jacobian_xx * offset_residual[:, 1]
                        - jacobian_yx * offset_residual[:, 0],
                    )
                )
                offset_step = np.divide(
                    offset_step_numerator,
                    jacobian_determinant[:, None],
                    out=np.zeros_like(radial_coords),
                    where=jacobian_determinant[:, None] != 0,
                )
                radial_coords += offset_step
                if np.all(np.abs(offset_step) < _FISHEYE624_TOLERANCE):
                    break
            # Recompute offsets after the last update to check the inverse result.
            offset_residual = (
                radial_coords
                + self._fisheye624_tangential_prism_offsets(radial_coords)
                - distorted_normalized_coords
            )
            valid_pixels &= (
                np.max(np.abs(offset_residual), axis=1) < _FISHEYE624_TOLERANCE
            )

        distorted_radii = np.hypot(radial_coords[:, 0], radial_coords[:, 1])
        max_distorted_radius = (np.pi / 2) * np.polynomial.polynomial.polyval(
            (np.pi / 2) ** 2, self._radial_polynomial
        )
        valid_pixels &= np.isfinite(distorted_radii) & (
            distorted_radii < max_distorted_radius
        )
        target_radii = distorted_radii[valid_pixels]
        # Solve the radial polynomial with its analytic derivative.
        theta = np.minimum(target_radii, np.pi / 2)
        if target_radii.size:
            try:
                theta = newton(
                    lambda theta, radius: (
                        theta
                        * np.polynomial.polynomial.polyval(
                            theta * theta, self._radial_polynomial
                        )
                        - radius
                    ),
                    theta,
                    fprime=lambda theta, radius: np.polynomial.polynomial.polyval(
                        theta * theta, self._radial_derivative
                    ),
                    args=(target_radii,),
                    maxiter=_MAX_INVERSE_ITERATIONS,
                    disp=False,
                )
            except RuntimeError:
                theta.fill(np.nan)
        radius_residual = (
            theta
            * np.polynomial.polynomial.polyval(theta * theta, self._radial_polynomial)
            - target_radii
        )
        if self.model_name == "KB4":
            radius_tolerance = _KB4_MAX_PIXEL_ERROR / max(self.K[0, 0], self.K[1, 1])
        else:
            radius_tolerance = _FISHEYE624_TOLERANCE
        pixel_ray_angles = np.zeros_like(distorted_radii)
        pixel_ray_angles[valid_pixels] = theta
        valid_pixels[valid_pixels] = (
            (theta >= 0)
            & (theta < np.pi / 2)
            & (np.abs(radius_residual) < radius_tolerance)
        )
        xy_directions = np.divide(
            radial_coords,
            distorted_radii[:, None],
            out=np.zeros_like(radial_coords),
            where=distorted_radii[:, None] > 0,
        )
        rays = np.column_stack(
            (
                xy_directions * np.sin(pixel_ray_angles)[:, None],
                np.cos(pixel_ray_angles),
            )
        )
        rays[~valid_pixels] = np.nan
        return rays.reshape(*pixels.shape[:-1], 3)

    def _fisheye624_tangential_prism_offsets(
        self, radial_coords: np.ndarray
    ) -> np.ndarray:
        """Compute FISHEYE624 tangential and thin-prism displacements.

        Args:
            radial_coords: Normalized image coordinates after radial distortion,
                shaped (N, 2).

        Returns:
            Displacements (dx, dy) to add in normalized coordinates, shaped (N, 2).
        """
        x, y = radial_coords.T
        radius_squared = x * x + y * y
        p0, p1, s0, s1, s2, s3 = self.tangential_prism
        return np.column_stack(
            (
                p0 * (3 * x * x + y * y)
                + 2 * p1 * x * y
                + s0 * radius_squared
                + s1 * radius_squared * radius_squared,
                p1 * (x * x + 3 * y * y)
                + 2 * p0 * x * y
                + s2 * radius_squared
                + s3 * radius_squared * radius_squared,
            )
        )


def load_fisheye_camera(
    calibration: dict,
    camera_side: str | None = None,
) -> CameraModel:
    """Convert dataset-specific FISHEYE624 or KB4 calibration through the shared loader.

    FISHEYE624 camera parameters are rotated 90 degrees clockwise to match
    the upright input video.

    Args:
        calibration: Parsed calibration JSON object.
        camera_side: Required KB4 camera entry, ``left`` or ``right``.
            Unused for FISHEYE624.

    Returns:
        Validated CameraModel configured with the FISHEYE624 or KB4 model.
    """
    if calibration.get("format_version") == "2.0-stereo":
        camera_calibration = calibration["cameras"][camera_side]
        if camera_calibration["distortion"]["model"] != "opencv_equidistant_fisheye":
            raise ValueError(
                "KB4 calibration requires opencv_equidistant_fisheye distortion"
            )
        width, height = camera_calibration["image_size_px"]
        model = "KB4"
        K = camera_calibration["intrinsics"]["K"]
        coefficients = camera_calibration["distortion"]["coefficients"]
    elif calibration.get("projection_model_type") == "CameraModelType.FISHEYE624":
        # Convert the native calibration to upright pixels and camera axes.
        projection_params = np.asarray(
            calibration["projection_params"], dtype=np.float64
        )
        f, cx, cy = projection_params[:3]
        p0, p1, s0, s1, s2, s3 = projection_params[9:]
        width, height = calibration["image_height"], calibration["image_width"]
        model = "FISHEYE624"
        K = [
            [f, 0, width - 1 - cy],
            [0, f, cx],
            [0, 0, 1],
        ]
        coefficients = np.r_[projection_params[3:9], -p1, p0, -s2, -s3, s0, s1]
    else:
        raise ValueError("Unsupported camera model or distortion calibration schema")
    return load_camera_calibration(
        {
            "model": model,
            "image_width": width,
            "image_height": height,
            "camera_matrix": {"matrix_3x3": K},
            "distortion_coefficients": coefficients,
        }
    )


@wp.func
def _project_rational_ray(
    ray: wp.vec3d,
    intrinsics: wp.vec4d,
    coefficients: wp.array(dtype=wp.float64),
    tilt: wp.vec4d,
    max_radius_squared: wp.float64,
):
    """Project one ray, returning a NaN pair outside the physical lens domain.

    Intrinsics are (fx, fy, cx, cy); coefficients follow OpenCV's 14-term order.
    Tilt stores (sin(tx), sin(ty), cos(tx), cos(ty)). Both CPU kernels share
    these projection equations and validity checks.
    """
    zero, one, two = wp.float64(0.0), wp.float64(1.0), wp.float64(2.0)
    three, four, six = wp.float64(3.0), wp.float64(4.0), wp.float64(6.0)
    invalid = wp.vec2d(wp.float64(wp.nan), wp.float64(wp.nan))
    ray_x, ray_y, z = ray[0], ray[1], ray[2]
    if (
        not wp.isfinite(ray_x)
        or not wp.isfinite(ray_y)
        or not wp.isfinite(z)
        or z <= zero
    ):
        return invalid
    x, y = ray_x / z, ray_y / z
    r2 = x * x + y * y
    if wp.isfinite(max_radius_squared) and not (r2 < max_radius_squared):
        return invalid
    k1, k2, p1, p2 = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
    k3, k4, k5, k6 = coefficients[4], coefficients[5], coefficients[6], coefficients[7]
    s1, s2, s3, s4 = (
        coefficients[8],
        coefficients[9],
        coefficients[10],
        coefficients[11],
    )
    sin_x, sin_y, cos_x, cos_y = tilt[0], tilt[1], tilt[2], tilt[3]
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    r4, r6 = r2 * r2, r2 * r2 * r2
    numerator = one + k1 * r2 + k2 * r4 + k3 * r6
    denominator = one + k4 * r2 + k5 * r4 + k6 * r6
    radial = numerator / denominator
    radial_derivative = (
        (k1 + two * k2 * r2 + three * k3 * r4) * denominator
        - numerator * (k4 + two * k5 * r2 + three * k6 * r4)
    ) / (denominator * denominator)
    xd = x * radial + two * p1 * x * y + p2 * (r2 + two * x * x) + s1 * r2 + s2 * r4
    yd = y * radial + p1 * (r2 + two * y * y) + two * p2 * x * y + s3 * r2 + s4 * r4

    # The analytic Jacobian rejects folds without OpenCV's full parameter Jacobian.
    cross = two * x * y * radial_derivative + two * p1 * x + two * p2 * y
    prism_x, prism_y = two * s1 + four * s2 * r2, two * s3 + four * s4 * r2
    jxx = (
        radial
        + two * x * x * radial_derivative
        + two * p1 * y
        + six * p2 * x
        + prism_x * x
    )
    jxy, jyx = cross + prism_x * y, cross + prism_y * x
    jyy = (
        radial
        + two * y * y * radial_derivative
        + six * p1 * y
        + two * p2 * x
        + prism_y * y
    )
    tilt_z = sin_y * xd - cos_y * sin_x * yd + cos_y * cos_x
    if tilt_z == zero:
        return invalid
    u = (cos_x * xd / tilt_z) * fx + cx
    v = ((cos_y * yd - sin_x * sin_y * xd) / tilt_z) * fy + cy
    # Sensor tilt contributes det(T) / tilt_z³ to the translation-x/y determinant.
    determinant = (
        (jxx * jyy - jxy * jyx)
        * ((cos_x * cos_y) * (cos_x * cos_y))
        / (tilt_z * tilt_z * tilt_z)
        * (fx / z)
        * (fy / z)
    )
    if (
        not wp.isfinite(determinant)
        or determinant <= zero
        or not wp.isfinite(u)
        or not wp.isfinite(v)
    ):
        return invalid
    return wp.vec2d(u, v)


@wp.kernel(enable_backward=False)
def _project_rational(
    rays: wp.array2d(dtype=wp.float64),
    intrinsics: wp.vec4d,
    coefficients: wp.array(dtype=wp.float64),
    tilt: wp.vec4d,
    max_radius_squared: wp.float64,
    pixels: wp.array2d(dtype=wp.float64),
):
    """Project source-camera rays with physical-domain checks on the CPU."""
    i = wp.tid()
    pixel = _project_rational_ray(
        wp.vec3d(rays[i, 0], rays[i, 1], rays[i, 2]),
        intrinsics,
        coefficients,
        tilt,
        max_radius_squared,
    )
    pixels[i, 0] = pixel[0]
    pixels[i, 1] = pixel[1]


@wp.kernel(enable_backward=False)
def _perspective_remap(
    side: int,
    focal_length: wp.float64,
    rotation: wp.mat33d,
    intrinsics: wp.vec4d,
    coefficients: wp.array(dtype=wp.float64),
    tilt: wp.vec4d,
    max_radius_squared: wp.float64,
    pixels: wp.array2d(dtype=wp.float32),
):
    """Fuse crop rays, rotation, projection and float32 remap conversion.

    Rotation maps crop-camera rays into source-camera axes. Calculations stay
    in float64 until the final cast to OpenCV's float32 map coordinates.
    """
    i = wp.tid()
    center = wp.float64(side - 1) / wp.float64(2.0)
    x = (wp.float64(i % side) - center) / focal_length
    y = (wp.float64(i // side) - center) / focal_length
    ray = wp.vec3d(
        x * rotation[0, 0] + y * rotation[0, 1] + rotation[0, 2],
        x * rotation[1, 0] + y * rotation[1, 1] + rotation[1, 2],
        x * rotation[2, 0] + y * rotation[2, 1] + rotation[2, 2],
    )
    projected = _project_rational_ray(
        ray, intrinsics, coefficients, tilt, max_radius_squared
    )
    u, v = wp.float32(projected[0]), wp.float32(projected[1])
    pixels[i, 0] = wp.where(wp.isfinite(u), u, wp.float32(-1.0))
    pixels[i, 1] = wp.where(wp.isfinite(v), v, wp.float32(-1.0))


# Preserve the float64 operation order, including the projection validity checks.
wp.set_module_options({"fast_math": False, "fuse_fp": False})


class _PerspectiveCamera:
    """Pinhole or rational perspective projection with prism and sensor-tilt terms."""

    def __init__(self, K, coefficients, width, height, model_name):
        """Store validated calibration and find the radial branch connected to the axis.

        Args:
            K: Validated, copied source-image intrinsics from the shared loader.
            coefficients: Empty for PINHOLE, or OpenCV's fourteen distortion terms.
            width: Source-image width.
            height: Source-image height.
            model_name: Registered source-camera model identifier.
        """
        self.K = K
        self.distortion_coefficients = (
            coefficients if len(coefficients) else np.zeros(14)
        )
        self.width, self.height, self.model_name = width, height, model_name
        d = self.distortion_coefficients
        numerator = Polynomial(np.r_[1, d[[0, 1, 4]]])
        denominator = Polynomial(np.r_[1, d[[5, 6, 7]]])
        # For s=r², the first pole or zero of d(r*N(s)/D(s))/dr ends the physical branch.
        derivative = numerator * denominator + Polynomial([0, 2]) * (
            numerator.deriv() * denominator - numerator * denominator.deriv()
        )
        roots = np.r_[denominator.roots(), derivative.roots()]
        physical = (roots.real > 0) & (
            abs(roots.imag) <= 1e-10 * np.maximum(1, abs(roots.real))
        )
        self._max_radius_squared = np.min(roots.real[physical], initial=np.inf)
        self._max_inverse_radius = np.inf
        # Bound the image of the physical radial branch before attempting inversion.
        if np.isfinite(self._max_radius_squared):
            r2 = self._max_radius_squared
            denom = denominator(r2)
            if denom > 1e-12:
                # The radial image radius increases up to this branch endpoint.
                radial_radius = np.sqrt(r2) * numerator(r2) / denom
                # Tangential offsets have norm at most 3 * ||(p1, p2)|| * r².
                # Bound prism offsets componentwise over the same physical disk.
                tangential_bound = 3 * np.hypot(d[2], d[3]) * r2
                prism_x = abs(d[8]) * r2 + abs(d[9]) * r2 * r2
                prism_y = abs(d[10]) * r2 + abs(d[11]) * r2 * r2
                bound = radial_radius + tangential_bound + np.hypot(prism_x, prism_y)
                if np.any(d[-2:]) and np.isfinite(bound) and bound > 0:
                    # Bound the tilted image of the disk. If the disk can reach
                    # the tilt horizon, no finite radius bound is safe.
                    sin_x, sin_y = np.sin(d[-2:])
                    cos_x, cos_y = np.cos(d[-2:])
                    min_depth = cos_x * cos_y - np.hypot(sin_y, cos_y * sin_x) * bound
                    if min_depth > 1e-12:
                        tilt_xy = np.array([[cos_x, 0.0], [-sin_x * sin_y, cos_y]])
                        bound *= np.linalg.norm(tilt_xy, ord=2) / min_depth
                    else:
                        bound = np.inf
                if np.isfinite(bound) and bound > 0:
                    # Include the solver's per-axis pixel tolerance and roundoff.
                    tolerance = 1e-6 * np.linalg.norm(1 / K.diagonal()[:2])
                    self._max_inverse_radius = (
                        bound + tolerance + 1e-6 * max(1.0, bound)
                    )
        self._projection_parameters = None
        if np.any(d):
            self._projection_parameters = [
                wp.vec4d(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
                wp.array(d, dtype=wp.float64, device="cpu", copy=False),
                wp.vec4d(*np.r_[np.sin(d[-2:]), np.cos(d[-2:])]),
                wp.float64(self._max_radius_squared),
            ]
            # Compile both CPU kernels before crop workers start.
            wp.load_module(module=_project_rational.module, device="cpu", block_dim=1)

    def project(self, rays: np.ndarray) -> np.ndarray:
        """Project forward rays, rejecting folded or disconnected distortion branches.

        Args:
            rays: Source-camera vectors shaped (..., 3).

        Returns:
            Source pixels shaped (..., 2); invalid or rear-facing rays become NaN.
        """
        rays = np.asarray(rays, dtype=np.float64)
        if rays.ndim == 0 or rays.shape[-1] != 3:
            raise ValueError("Camera rays must have shape (..., 3)")
        flat = rays.reshape(-1, 3)
        if (
            len(flat) >= _CPU_KERNEL_MIN_RAYS
            and self._projection_parameters is not None
        ):
            pixels = np.empty((len(flat), 2), dtype=np.float64)
            wp.launch(
                _project_rational,
                dim=len(flat),
                inputs=[
                    wp.array(flat, dtype=wp.float64, device="cpu", copy=False),
                    *self._projection_parameters,
                ],
                outputs=[wp.array(pixels, dtype=wp.float64, device="cpu", copy=False)],
                device="cpu",
                record_tape=False,
            )
            return pixels.reshape(*rays.shape[:-1], 2)
        pixels = np.full((len(flat), 2), np.nan)
        indices = np.flatnonzero(np.isfinite(flat).all(axis=1) & (flat[:, 2] > 0))
        if np.isfinite(self._max_radius_squared):
            radii = np.sum((flat[indices, :2] / flat[indices, 2, None]) ** 2, axis=1)
            indices = indices[radii < self._max_radius_squared]
        if len(indices):
            if self._projection_parameters is None:
                pixels[indices] = (
                    flat[indices, :2] / flat[indices, 2, None]
                ) * self.K.diagonal()[:2] + self.K[:2, 2]
            else:
                projected, jacobian = cv2.projectPoints(
                    flat[indices],
                    np.zeros(3),
                    np.zeros(3),
                    self.K,
                    self.distortion_coefficients,
                )
                projected = projected.reshape(-1, 2)
                determinant = np.linalg.det(jacobian[:, 3:5].reshape(-1, 2, 2))
                projected[~np.isfinite(determinant) | (determinant <= 0)] = np.nan
                pixels[indices] = projected
        pixels[~np.isfinite(pixels).all(axis=1)] = np.nan
        return pixels.reshape(*rays.shape[:-1], 2)

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Invert perspective distortion with a safeguarded Newton fallback.

        Args:
            pixels: Source-image coordinates shaped (..., 2).

        Returns:
            Unit forward rays shaped (..., 3); invalid inversions become NaN.
            Distorted rays must reproject within 1e-6 pixels on the physical branch.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim == 0 or pixels.shape[-1] != 2:
            raise ValueError("Camera pixels must have shape (..., 2)")
        flat = pixels.reshape(-1, 2)
        finite = np.isfinite(flat).all(axis=1)
        if np.isfinite(self._max_inverse_radius):
            normalized = (flat - self.K[:2, 2]) / self.K.diagonal()[:2]
            finite &= (
                np.hypot(normalized[:, 0], normalized[:, 1]) <= self._max_inverse_radius
            )
        target = flat[finite]
        rays = np.full((len(flat), 3), np.nan)
        xy = (target - self.K[:2, 2]) / self.K.diagonal()[:2]
        distorted = self._projection_parameters is not None
        if distorted and len(target):
            xy = cv2.undistortPointsIter(
                target[:, None],
                self.K,
                self.distortion_coefficients,
                None,
                None,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10),
            ).reshape(-1, 2)
            error = np.max(
                abs(self.project(np.c_[xy, np.ones(len(xy))]) - target), axis=1
            )
            active = np.flatnonzero(~np.isfinite(error) | (error > 1e-6))
            # Rays that already passed stay unchanged; recheck only initial failures.
            inverse_failures = active.copy()
            # Restart failed fixed-point inversions near the pinhole ray, inside the branch.
            xy[active] = (target[active] - self.K[:2, 2]) / self.K.diagonal()[:2]
            radius_squared = np.sum(xy[active] ** 2, axis=1)
            outside = radius_squared >= self._max_radius_squared
            xy[active[outside]] *= np.sqrt(
                0.9 * self._max_radius_squared / radius_squared[outside]
            )[:, None]
            for _ in range(_MAX_INVERSE_ITERATIONS):
                if not len(active):
                    break
                projected, jacobian = cv2.projectPoints(
                    np.c_[xy[active], np.ones(len(active))],
                    np.zeros(3),
                    np.zeros(3),
                    self.K,
                    self.distortion_coefficients,
                )
                residual = projected.reshape(-1, 2) - target[active]
                error = np.max(abs(residual), axis=1)
                # At z=1, translation-x/y derivatives equal derivatives with respect to x/y.
                jacobian = jacobian[:, 3:5].reshape(-1, 2, 2)
                solvable = (
                    np.isfinite(residual).all(axis=1)
                    & np.isfinite(jacobian).all(axis=(1, 2))
                    & (np.linalg.det(jacobian) > np.finfo(np.float64).eps)
                    & (error > 1e-6)
                )
                active = active[solvable]
                if not len(active):
                    break
                step = np.linalg.solve(jacobian[solvable], residual[solvable, :, None])[
                    :, :, 0
                ]
                scale = np.ones(len(active))
                for _ in range(20):
                    candidate = xy[active] - step * scale[:, None]
                    trial = self.project(np.c_[candidate, np.ones(len(candidate))])
                    trial_error = np.max(abs(trial - target[active]), axis=1)
                    improved = np.isfinite(trial_error) & (
                        trial_error < error[solvable]
                    )
                    if np.all(improved):
                        break
                    scale[~improved] *= 0.5
                xy[active[improved]] = candidate[improved]
                active = active[improved]
        forward = np.c_[xy, np.ones(len(xy))]
        if distorted and len(target) and len(inverse_failures):
            error = np.max(
                abs(self.project(forward[inverse_failures]) - target[inverse_failures]),
                axis=1,
            )
            forward[inverse_failures[~np.isfinite(error) | (error > 1e-6)]] = np.nan
        forward[~np.isfinite(forward).all(axis=1)] = np.nan
        forward /= np.max(abs(forward), axis=1, keepdims=True)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        rays[finite] = forward
        return rays.reshape(*pixels.shape[:-1], 3)


CAMERA_MODELS = {
    "PINHOLE": (0, _PerspectiveCamera),
    "KB4": (4, _FisheyeCamera),
    "FISHEYE624": (12, _FisheyeCamera),
    "RATIONAL_POLYNOMIAL": (14, _PerspectiveCamera),
}


def load_camera_calibration(calibration: dict) -> CameraModel:
    """Load the common camera JSON without selecting, rotating, or resizing its pixels.

    Args:
        calibration: Mapping containing model, image_width, image_height,
            camera_matrix.matrix_3x3, and distortion_coefficients. PINHOLE has [],
            KB4 has four radial terms, FISHEYE624 has six radial/two tangential/four
            prism terms, and RATIONAL_POLYNOMIAL has all fourteen OpenCV perspective terms.

    Returns:
        CameraModel calibrated for exactly the supplied source-image dimensions.
    """
    if (
        not isinstance(calibration, dict)
        or not isinstance(calibration.get("model"), str)
        or calibration["model"] not in CAMERA_MODELS
    ):
        raise ValueError("Convert calibration to the shared camera JSON schema")
    model = calibration["model"]
    count, factory = CAMERA_MODELS[model]
    width, height = calibration.get("image_width"), calibration.get("image_height")
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError(
            "Camera image_width and image_height must be positive integers"
        )
    matrix = calibration.get("camera_matrix")
    if not isinstance(matrix, dict) or "matrix_3x3" not in matrix:
        raise ValueError("Camera JSON requires camera_matrix.matrix_3x3")
    K = np.array(matrix["matrix_3x3"], dtype=np.float64, copy=True)
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or K[0, 1] != 0
        or K[1, 0] != 0
        or not np.array_equal(K[2], [0, 0, 1])
    ):
        raise ValueError(
            "Camera intrinsics must be finite, zero-skew, and have positive focals"
        )
    coefficients = np.array(
        calibration.get("distortion_coefficients"), dtype=np.float64
    )
    if coefficients.shape != (count,) or not np.isfinite(coefficients).all():
        raise ValueError(
            f"{model} requires exactly {count} finite distortion coefficients"
        )
    return factory(K, coefficients, width, height, model_name=model)


@lru_cache(maxsize=8)
def _crop_pixel_grid(side: int) -> np.ndarray:
    """Cache read-only pixel coordinates that crop workers can safely share."""
    grid = np.indices((side, side), dtype=np.float64)
    grid.setflags(write=False)
    return grid


def create_perspective_remap(
    camera: CameraModel,
    crop_to_source_rotation: np.ndarray,
    focal_length: float,
    side: int,
) -> np.ndarray:
    """Build a perspective crop's float32 source-pixel map for OpenCV.

    Rational cameras fuse ray generation, rotation and projection in a CPU
    kernel. Other camera models retain their own projection implementation.
    Invalid coordinates become -1 so OpenCV samples the constant black border.

    Args:
        camera: Source camera with intrinsics in source-image pixel coordinates.
        crop_to_source_rotation: 3x3 rotation from crop to source camera axes.
        focal_length: Virtual pinhole focal length in crop pixels.
        side: Width and height of the square output crop, in pixels.

    Returns:
        A (side, side, 2) float32 array of source x/y pixel coordinates.
    """
    if (
        isinstance(camera, _PerspectiveCamera)
        and camera._projection_parameters is not None
    ):
        pixels = np.empty((side * side, 2), dtype=np.float32)
        wp.launch(
            _perspective_remap,
            dim=side * side,
            inputs=[
                side,
                wp.float64(focal_length),
                wp.mat33d(crop_to_source_rotation),
                *camera._projection_parameters,
            ],
            outputs=[wp.array(pixels, dtype=wp.float32, device="cpu", copy=False)],
            device="cpu",
            record_tape=False,
        )
        return pixels.reshape(side, side, 2)
    grid_y, grid_x = _crop_pixel_grid(side)
    center = (side - 1) / 2.0
    rays = np.stack(
        (
            (grid_x - center) / focal_length,
            (grid_y - center) / focal_length,
            np.ones_like(grid_x),
        ),
        axis=-1,
    )
    source_pixels = camera.project(rays @ crop_to_source_rotation.T)
    with np.errstate(over="ignore", invalid="ignore"):
        pixels = source_pixels.astype(np.float32)
    pixels[~np.isfinite(pixels)] = -1.0
    return pixels
