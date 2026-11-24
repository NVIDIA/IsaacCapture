# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OAK-D as a stereo camera: both eyes of each capture, in one frame.

- The mono pair (CAM_B left, CAM_C right) is the stereo pair, as captured:
  unrectified, with the calibration stored on the device.
- With a picture, the colour camera (CAM_A) is delivered too, as the image
  to look at. That takes a colour sensor wired to the pair's frame sync; a
  device without one is refused.
- The left sensor drives the sync line and the others follow. Frames
  exposed together are matched by device timestamp, not by sequence
  number: the colour camera's count starts at a different frame.
- The accelerometer is read when asked for.
- A device that is not there yet is waited for. Once it has delivered, an
  error is final: reopening restarts the device's clock, which its frames
  are stamped with, and a reader that follows the camera could not tell.

Frames arrive in host memory over USB and are uploaded to the GPU.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import deque
from datetime import timedelta
from typing import Optional

import numpy as np

from ..pipeline import (
    Frame,
    FrameSource,
    ImuSamples,
    PictureCalibration,
    SourceSpec,
    StereoCalibration,
)
from ._helpers import CaptureMailbox, alloc_pinned_host, notify

#: Colour sensors that share the stereo pair's frame sync.
SYNCED_COLOR_SENSORS = ("OV9782",)

#: Frames exposed together are stamped within microseconds of each other.
MATCH_TOLERANCE_NS = 2_000_000

#: How long frames may arrive without one matching set before the device
#: is given up on.
NO_MATCH_TIMEOUT_S = 3.0

RECONNECT_DELAY_S = 5.0

# How long stop() waits for the producer.
_STOP_S = 5.0

_ACCEL_RATE_HZ = 200
# Unread IMU samples kept for a reader that fell behind.
_IMU_BACKLOG = 4096

# The lens models depthai names, as StereoCalibration names them.
_LENS_MODELS = {"Perspective": "brown_conrady", "Fisheye": "fisheye"}


class Refused(RuntimeError):
    """This device cannot be, or can no longer be, the camera that was
    asked for. Reopening it would not help, so the reader is told instead."""


def _notify(msg: str, level: int = logging.INFO) -> None:
    notify("oakd", msg, level)


def check_sensors(sensors: dict, picture: bool) -> None:
    """Refuse a device that cannot deliver what was asked for.

    ``sensors`` maps a socket name (CAM_A, CAM_B, CAM_C) to its sensor's
    name.
    """
    wanted = ("CAM_B", "CAM_C") + (("CAM_A",) if picture else ())
    missing = [socket for socket in wanted if socket not in sensors]
    if missing:
        raise Refused(f"the device has no camera on {', '.join(missing)}")
    if picture and sensors["CAM_A"] not in SYNCED_COLOR_SENSORS:
        raise Refused(
            f"the colour camera is a {sensors['CAM_A']}, which is not "
            "frame-synced with the stereo pair. Supported colour sensors: "
            f"{', '.join(SYNCED_COLOR_SENSORS)} (the OAK-D models ending in 97). "
            "`mode: stereo` gives the pair alone"
        )


def _lens_model(calib, socket) -> str:
    name = calib.getDistortionModel(socket).name
    if name not in _LENS_MODELS:
        raise Refused(f"the device calibrates a lens as {name}, which is not handled")
    return _LENS_MODELS[name]


def _coefficients(calib, socket, model: str) -> np.ndarray:
    """A lens's distortion coefficients. The device pads every model to
    one length; OpenCV's fisheye model takes the first four."""
    values = np.asarray(calib.getDistortionCoefficients(socket), dtype=np.float64)
    return values[:4] if model == "fisheye" else values


def imu_rotation(calib, left):
    """The rotation that takes a vector from the IMU's frame into the left
    eye's, from the device's calibration."""
    left_from_imu = np.asarray(calib.getImuToCameraExtrinsics(left), dtype=np.float64)
    return left_from_imu[:3, :3]


def output_intrinsics(calib, socket, size) -> np.ndarray:
    """The intrinsics of ``socket``'s image at ``size``.

    The camera scales its calibrated image until it covers ``size`` and
    crops what is left over evenly from both sides (``ImgResizeMode.CROP``).
    The handler's own resize scales by width, which is another image when
    ``size`` is narrower than the sensor.
    """
    matrix, native_width, native_height = calib.getDefaultIntrinsics(socket)
    K = np.array(matrix, dtype=np.float64)
    width, height = size
    scale = max(width / native_width, height / native_height)
    K[:2] *= scale
    K[0, 2] -= (native_width * scale - width) / 2.0
    K[1, 2] -= (native_height * scale - height) / 2.0
    return K


def stereo_calibration(calib, sockets, size, picture_size=None) -> StereoCalibration:
    """The device's calibration of its stereo pair, and of its colour
    camera when ``picture_size`` is given.

    ``calib`` is the device's calibration handler and ``sockets`` its
    ``(left, right, colour)`` socket identifiers. The device stores lengths
    in centimetres.
    """
    left, right, color = sockets
    width, height = size
    right_from_left = np.asarray(
        calib.getCameraExtrinsics(left, right), dtype=np.float64
    )
    model = _lens_model(calib, left)
    if _lens_model(calib, right) != model:
        raise Refused("the device calibrates its two eyes with different lens models")
    picture = None
    if picture_size is not None:
        left_from_color = np.asarray(
            calib.getCameraExtrinsics(color, left), dtype=np.float64
        )
        picture_model = _lens_model(calib, color)
        picture = PictureCalibration(
            width=picture_size[0],
            height=picture_size[1],
            K=output_intrinsics(calib, color, picture_size),
            D=_coefficients(calib, color, picture_model),
            R_left_from_picture=left_from_color[:3, :3],
            distortion_model=picture_model,
        )
    return StereoCalibration(
        width=width,
        height=height,
        left_K=output_intrinsics(calib, left, size),
        left_D=_coefficients(calib, left, model),
        right_K=output_intrinsics(calib, right, size),
        right_D=_coefficients(calib, right, model),
        R=right_from_left[:3, :3],
        T=right_from_left[:3, 3] / 100.0,
        rectified=False,
        distortion_model=model,
        picture=picture,
    )


class FrameMatcher:
    """Groups the frames the sensors exposed together.

    ``names`` are the sensors, the first being the one the others are
    matched to.
    """

    def __init__(
        self, names, tolerance_ns: int = MATCH_TOLERANCE_NS, keep: int = 8
    ) -> None:
        self._names = tuple(names)
        self._tolerance_ns = tolerance_ns
        self._keep = keep
        self._frames: dict = {name: {} for name in self._names}

    def add(self, name: str, timestamp_ns: int, frame) -> None:
        frames = self._frames[name]
        frames[timestamp_ns] = frame
        while len(frames) > self._keep:
            del frames[min(frames)]

    def take_next(self) -> Optional[tuple]:
        """The oldest complete set as ``(timestamp of the first sensor,
        {name: frame})``, or None. Frames arrive in bursts, so sets are
        handed out in order instead of skipping to the newest. Taking a set
        drops it and everything older."""
        first, others = self._names[0], self._names[1:]
        for stamp in sorted(self._frames[first]):
            partners = {name: self._nearest(name, stamp) for name in others}
            if None in partners.values():
                continue
            found = {first: self._frames[first][stamp]}
            found.update(
                {name: self._frames[name][at] for name, at in partners.items()}
            )
            for frames in self._frames.values():
                for old in [t for t in frames if t <= stamp + self._tolerance_ns]:
                    del frames[old]
            return stamp, found
        return None

    def _nearest(self, name: str, stamp: int) -> Optional[int]:
        frames = self._frames[name]
        if not frames:
            return None
        best = min(frames, key=lambda t: abs(t - stamp))
        return best if abs(best - stamp) <= self._tolerance_ns else None


def _device_ns(stamp: timedelta) -> int:
    """A device timestamp as integer nanoseconds, without a float round trip."""
    return (stamp // timedelta(microseconds=1)) * 1000


class OakdStereoSource(FrameSource):
    """Both eyes of each capture of an OAK-D and, with ``picture``, its
    colour camera's image of the same instant.

    A frame carries the left sensor's device timestamp. The eyes are the
    mono sensors' images as gray RGBA.
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        fps: int = 30,
        device_id: str = "",
        picture: bool = False,
        picture_width: int = 0,
        picture_height: int = 0,
        exposure_limit_ms: float = 0.0,
        imu: bool = False,
    ) -> None:
        try:
            import cupy as cp
            import depthai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "OakdStereoSource requires CuPy + depthai. Install camera_viz "
                "with its `oakd` extra, or run `camera_viz.sh setup`, which "
                "also installs the CuPy matching your CUDA toolkit."
            ) from e

        self._cp = cp
        self._spec = SourceSpec(
            name=name, width=width, height=height, pixel_format="rgba8"
        )
        self._fps = int(fps)
        self._device_id = device_id
        self._exposure_limit_us = int(exposure_limit_ms * 1000)
        self._read_imu = imu
        self._picture_size = (
            (picture_width or width, picture_height or height) if picture else None
        )

        self._stream = cp.cuda.Stream(non_blocking=True)

        def rgba(w: int, h: int) -> list:
            buffers = [cp.empty((h, w, 4), dtype=cp.uint8) for _ in range(3)]
            # On the producer's stream, so its uploads come after.
            with self._stream:
                for buffer in buffers:
                    buffer[..., 3] = 255
            return buffers

        # Per sensor: the buffers a reader is handed, a pinned host copy of
        # the frame as it arrived, and that copy on the GPU.
        self._buffers = {"left": rgba(width, height), "right": rgba(width, height)}
        self._staging = {
            eye: alloc_pinned_host((height, width), np.uint8)
            for eye in ("left", "right")
        }
        if self._picture_size is not None:
            pw, ph = self._picture_size
            self._buffers["picture"] = rgba(pw, ph)
            self._staging["picture"] = alloc_pinned_host((ph, pw, 3), np.uint8)
        self._landing = {
            sensor: cp.empty(staging.shape, dtype=cp.uint8)
            for sensor, staging in self._staging.items()
        }
        self._captures = CaptureMailbox(3)

        self._lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._calibration: Optional[StereoCalibration] = None
        self._imu_samples: deque = deque(maxlen=_IMU_BACKLOG)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._device = None
        self._pipeline = None
        self._queues: dict = {}
        self._imu_queue = None
        self._matcher: Optional[FrameMatcher] = None
        self._connected = False
        # A capture or IMU samples were handed over since start().
        self._delivered = False
        self._waiting = False  # the "not visible" notice is out
        self._last_attempt_s = 0.0
        self._last_match_s = 0.0

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    @property
    def dropped_captures(self) -> int:
        """Captures that were replaced before they were read."""
        return self._captures.dropped

    def calibration(self) -> Optional[StereoCalibration]:
        return self._calibration

    def imu(self) -> Optional[ImuSamples]:
        with self._lock:
            samples = list(self._imu_samples)
            self._imu_samples.clear()
            if samples:
                # Stamped with this device's clock, as a capture is.
                self._delivered = True
        if not samples:
            return None
        return ImuSamples(
            timestamps_ns=np.array([s[0] for s in samples], dtype=np.int64),
            accel=np.array([s[1:] for s in samples], dtype=np.float64),
        )

    def start(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError(f"OAK-D {self._spec.name!r} is still stopping")
                return
            # stop() gave up waiting for this producer; it has ended since.
            self._thread = None
        self._stop.clear()
        # Nothing of the last session is handed out in this one.
        self._captures.reset()
        with self._lock:
            self._error = None
            self._imu_samples.clear()
        self._calibration = None
        self._delivered = False
        self._thread = threading.Thread(
            target=self._produce_loop,
            name=f"oakd_stereo_{self._spec.name}",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_S)
            if self._thread.is_alive():
                # The producer is inside a depthai call, such as the search
                # for a device. Closing the device under it could crash, so
                # the producer closes it when the call returns.
                _notify(
                    f"the producer did not stop within {_STOP_S:.0f} s; it closes "
                    "the device when it ends",
                    logging.WARNING,
                )
                return
            self._thread = None

    def latest(self) -> Optional[Frame]:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(
                    f"OAK-D {self._spec.name!r}: {self._error}"
                ) from self._error
        taken = self._captures.take()
        if taken is None:
            return None
        index, timestamp_ns = taken
        picture = self._buffers.get("picture")
        return Frame(
            image=self._buffers["left"][index],
            image_right=self._buffers["right"][index],
            picture=None if picture is None else picture[index],
            timestamp_ns=timestamp_ns,
            source_id=self._spec.name,
            stream=0,
        )

    # ── Device open / close (producer thread) ─────────────────────────

    def _visible(self, dai) -> bool:
        """Whether the device is on USB. Opening one that is not blocks for
        about ten seconds; this answers at once."""
        try:
            available = dai.Device.getAllAvailableDevices()
        except Exception:
            return False
        if not self._device_id:
            return bool(available)
        return any(self._device_id in str(device) for device in available)

    def _open_device(self, dai) -> None:
        """Open the device and start its streams. Raises Refused for a
        device that is there but cannot be set up as asked."""
        device = (
            dai.Device(dai.DeviceInfo(self._device_id))
            if self._device_id
            else dai.Device()
        )
        self._device = device
        try:
            # depthai would reopen a device it lost by itself, and the
            # clock its frames are stamped with would start again.
            device.setMaxReconnectionAttempts(0)
            self._set_up(dai, device)
        except Refused:
            raise
        except Exception as e:
            # Its calibration lacks something, or it will not stream this
            # size or rate: the next try would fail the same way.
            raise Refused(f"the device could not be set up: {e}") from e

    def _set_up(self, dai, device) -> None:
        sockets = dai.CameraBoardSocket
        left, right, color = sockets.CAM_B, sockets.CAM_C, sockets.CAM_A
        with_picture = self._picture_size is not None
        check_sensors(
            {f.socket.name: f.sensorName for f in device.getConnectedCameraFeatures()},
            with_picture,
        )
        speed = getattr(device.getUsbSpeed(), "name", "?")
        if speed not in ("SUPER", "SUPER_PLUS"):
            _notify(f"USB {speed} (not USB 3): the streams will drop frames")

        size = (self._spec.width, self._spec.height)
        calib = device.readCalibration()
        calibration = stereo_calibration(
            calib, (left, right, color), size, self._picture_size
        )
        read_imu = self._read_imu
        if read_imu and device.getConnectedIMU() in ("", "NONE"):
            _notify("this device has no IMU")
            read_imu = False
        if read_imu:
            try:
                calibration = dataclasses.replace(
                    calibration, R_left_from_imu=imu_rotation(calib, left)
                )
            except Exception as e:
                _notify(f"the device's IMU is not calibrated ({e}); it is not read")
                read_imu = False
        self._calibration = calibration

        pipeline = dai.Pipeline(device)
        pipeline.setXLinkChunkSize(0)  # send each frame as one transfer
        sync = dai.CameraControl.FrameSyncMode
        streams = [
            ("left", left, dai.ImgFrame.Type.GRAY8, size),
            ("right", right, dai.ImgFrame.Type.GRAY8, size),
        ]
        if with_picture:
            streams.append(
                ("picture", color, dai.ImgFrame.Type.NV12, self._picture_size)
            )
        self._queues = {}
        for name, socket, frame_type, stream_size in streams:
            camera = pipeline.create(dai.node.Camera).build(socket)
            if self._exposure_limit_us > 0:
                camera.initialControl.setAutoExposureLimit(self._exposure_limit_us)
            if with_picture:
                # The left sensor drives the sync line and the others
                # follow. Left to itself, the colour camera trails the pair
                # by several milliseconds; the pair alone keeps the device's
                # own sync.
                camera.initialControl.setFrameSyncMode(
                    sync.OUTPUT if name == "left" else sync.INPUT
                )
            # CROP is the geometry output_intrinsics describes.
            out = camera.requestOutput(
                stream_size,
                type=frame_type,
                resizeMode=dai.ImgResizeMode.CROP,
                fps=self._fps,
            )
            self._queues[name] = out.createOutputQueue(maxSize=8, blocking=False)

        self._imu_queue = None
        if read_imu:
            imu = pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, _ACCEL_RATE_HZ)
            # Batched: one message per sample crowds the frames off the link.
            imu.setBatchReportThreshold(10)
            imu.setMaxBatchReports(20)
            self._imu_queue = imu.out.createOutputQueue(maxSize=50, blocking=False)

        pipeline.start()
        self._pipeline = pipeline
        self._matcher = FrameMatcher(self._queues)
        self._last_match_s = time.monotonic()

    def _close_device(self) -> None:
        for handle, close in ((self._pipeline, "stop"), (self._device, "close")):
            if handle is not None:
                try:
                    getattr(handle, close)()
                except Exception:
                    pass
        self._pipeline = self._device = self._imu_queue = None
        self._queues = {}
        self._connected = False

    # ── Producer loop ─────────────────────────────────────────────────

    def _produce_loop(self) -> None:
        import depthai as dai

        cp = self._cp
        try:
            # Pinned to the GPU the buffers are on, which need not be this
            # thread's default.
            with cp.cuda.Device(int(self._buffers["left"][0].device.id)):
                while not self._stop.is_set():
                    self._produce_once(dai)
        except Refused as exc:
            # latest() tells the reader: this device will not do.
            with self._lock:
                self._error = exc
        finally:
            # By the thread that used it, also after stop() gave up waiting.
            self._close_device()

    def _produce_once(self, dai) -> None:
        if not self._connected:
            now = time.monotonic()
            if now - self._last_attempt_s < RECONNECT_DELAY_S:
                self._stop.wait(timeout=0.1)
                return
            self._last_attempt_s = now
            if not self._visible(dai):
                if not self._waiting:
                    _notify("device not visible on USB; waiting")
                    self._waiting = True
                return
            self._waiting = False
            _notify("opening...")
            try:
                self._open_device(dai)
            except Refused:
                self._close_device()
                raise
            except Exception as e:
                # Not there yet, or busy: worth another try.
                _notify(f"open failed ({e})", logging.ERROR)
                self._close_device()
                return
            self._connected = True
            _notify("connected")
            return
        try:
            self._pipeline.processTasks()
            self._collect()
            matched = self._matcher.take_next()
            if matched is not None:
                self._upload(*matched)
                self._last_match_s = time.monotonic()
                self._delivered = True
                return
        except Refused:
            self._close_device()
            raise
        except Exception as e:
            self._close_device()
            with self._lock:
                final = self._delivered
                if not final:
                    # The reopened device's clock starts again.
                    self._imu_samples.clear()
            if final:
                raise Refused(
                    f"the device stopped ({e}). Reopening it would restart the "
                    "clock its frames are stamped with"
                ) from e
            _notify(f"frame error ({e}); reopening", logging.ERROR)
            return
        if time.monotonic() - self._last_match_s > NO_MATCH_TIMEOUT_S:
            self._close_device()
            raise Refused(
                "its sensors delivered no set of frames exposed together for "
                f"{NO_MATCH_TIMEOUT_S:.0f} s: it stopped, or its frames are not "
                "synchronised"
            )
        self._stop.wait(timeout=0.001)

    def _collect(self) -> None:
        """Move what the device has sent into the matcher and the IMU
        backlog."""
        for name, queue in self._queues.items():
            while queue.has():
                frame = queue.get()
                self._matcher.add(name, _device_ns(frame.getTimestampDevice()), frame)
        if self._imu_queue is not None:
            while self._imu_queue.has():
                for packet in self._imu_queue.get().packets:
                    a = packet.acceleroMeter
                    sample = (_device_ns(a.getTimestampDevice()), a.x, a.y, a.z)
                    with self._lock:
                        self._imu_samples.append(sample)

    def _upload(self, timestamp_ns: int, frames: dict) -> None:
        """One matched set onto the GPU, into buffers no reader holds."""
        index = self._captures.write_index()
        try:
            with self._stream:
                for sensor, frame in frames.items():
                    # NV12 becomes packed BGR on the host; gray stays as it is.
                    pixels = (
                        frame.getCvFrame() if sensor == "picture" else frame.getFrame()
                    )
                    staging = self._staging[sensor]
                    if pixels.shape != staging.shape:
                        raise Refused(
                            f"the device sent {sensor} frames of {pixels.shape[1]}x"
                            f"{pixels.shape[0]}, not the {staging.shape[1]}x"
                            f"{staging.shape[0]} that was asked for"
                        )
                    np.copyto(staging, pixels)
                    landing = self._landing[sensor]
                    landing.set(staging)
                    buffer = self._buffers[sensor][index]
                    if sensor == "picture":
                        buffer[..., :3] = landing[..., ::-1]  # BGR to RGB
                    else:
                        buffer[..., :3] = landing[..., None]
        finally:
            # Also after a failed upload: the next one reuses the staging
            # buffers. And the reader's copy runs on another stream.
            self._stream.synchronize()
        self._captures.publish(index, timestamp_ns)
