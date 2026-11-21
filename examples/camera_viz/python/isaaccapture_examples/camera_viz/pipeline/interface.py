# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source contract for camera_viz.

Sources own their own producer thread (or rely on the vendor SDK's
callback thread) and expose ``latest()`` as a non-blocking mailbox.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SourceSpec:
    """Static description of what a source produces.

    Layers are sized to ``(width, height)`` at construction; the source
    contract is that every ``Frame.image`` it later emits matches this
    shape + format exactly.
    """

    name: str
    width: int
    height: int
    pixel_format: str = "rgba8"  # Phase 1: only RGBA8


@dataclass
class Frame:
    """One produced frame, GPU-resident.

    ``image`` is anything that exposes ``__cuda_array_interface__`` —
    CuPy / PyTorch / Numba arrays all work. ``stream`` is the producer's
    CUDA stream so the consumer can synchronize when it's not 0/default.

    Stereo: ``image`` is the left eye; ``image_right`` carries the
    right eye when paired (both from the same capture instant).
    """

    image: Any
    timestamp_ns: int
    source_id: str
    stream: int = 0
    image_right: Optional[Any] = None


#: The lens models a calibration can name. ``brown_conrady`` is OpenCV's
#: standard one: radial and tangential terms, with its optional rational,
#: thin-prism and tilt terms. ``fisheye`` is OpenCV's fisheye model.
DISTORTION_MODELS = ("brown_conrady", "fisheye")


def _check_lens_model(model: str) -> None:
    if model not in DISTORTION_MODELS:
        raise ValueError(
            f"unknown distortion_model {model!r} (known: {', '.join(DISTORTION_MODELS)})"
        )


def _freeze(*arrays: Any) -> None:
    """A source hands the same record to every caller, so its arrays are
    made read-only."""
    for array in arrays:
        if hasattr(array, "setflags"):
            array.setflags(write=False)


@dataclass(frozen=True, eq=False)
class StereoCalibration:
    """The lenses of a stereo camera and how its two eyes sit.

    Matrices are NumPy arrays in OpenCV's conventions. ``left_K`` and
    ``right_K`` are 3x3 pinhole matrices for ``width`` x ``height`` images,
    ``left_D`` and ``right_D`` are distortion coefficients in
    ``distortion_model``, and a point ``p`` in the left eye's frame is
    ``R @ p + T`` in the right eye's, with ``T`` in metres.

    ``rectified`` says the images are already rectified: both eyes then
    share ``left_K``, there is no distortion, ``R`` is the identity and
    ``T`` is ``(-baseline, 0, 0)``.

    ``R_left_from_imu`` rotates a vector from the IMU's frame into the left
    eye's; None when the camera has no IMU.
    """

    width: int
    height: int
    left_K: Any
    left_D: Any
    right_K: Any
    right_D: Any
    R: Any
    T: Any
    rectified: bool = False
    R_left_from_imu: Optional[Any] = None
    distortion_model: str = "brown_conrady"

    def __post_init__(self) -> None:
        _check_lens_model(self.distortion_model)
        _freeze(
            self.left_K,
            self.left_D,
            self.right_K,
            self.right_D,
            self.R,
            self.T,
            self.R_left_from_imu,
        )


@dataclass(frozen=True)
class ImuSamples:
    """IMU samples in the order they were taken.

    ``timestamps_ns`` is (N,) int64, on the clock of the camera's frame
    timestamps. ``accel`` is (N, 3) float64 specific force in m/s^2, in the
    IMU's frame: at rest it reads about 9.81 upward. ``gyro`` is kept for
    the rotation rate and stays None until a source fills it.
    """

    timestamps_ns: Any
    accel: Any
    gyro: Optional[Any] = None


class FrameSource(ABC):
    """Pull-based GPU-resident frame source."""

    @property
    @abstractmethod
    def spec(self) -> SourceSpec: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def latest(self) -> Optional[Frame]:
        """Return the freshest frame, or None if no new frame since the
        last call. Must be non-blocking — the render loop polls this
        every frame and skips submission on None. Raise on a fatal
        producer failure so the runner can stop and report it."""

    @abstractmethod
    def stop(self) -> None: ...

    def calibration(self) -> Optional[StereoCalibration]:
        """The stereo calibration, from a source that knows it. None until
        the camera is open, and always None from a source that has none."""
        return None

    def imu(self) -> Optional[ImuSamples]:
        """The IMU samples taken since the last call, from a source that was
        asked to read them. None when there are none."""
        return None

    # Convenience context manager so app code can `with source:`.
    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
