# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ZED source for camera_viz (mono / stereo).

Zero H2D in the steady state — pyzed's ``retrieve_image(MEM.GPU)`` hands
back a pitched BGRA8 buffer already in GPU memory. We wrap it as a
zero-copy CuPy view via ``UnownedMemory`` and GPU channel-swap directly
into our pre-allocated contiguous RGBA8 output. No staging through host
memory at any point.

Architecture mirrors :mod:`sources.oakd`: a single ``_ZedCamera`` owns
the ``sl.Camera`` + producer thread; ``ZedSource`` is a reference-
counted handle per eye. Stereo cameras (ZED 2 / ZED X Mini / ZED Mini)
share one underlying camera open; mono (ZED X One) just gets a single
``left`` slot.

Reconnect logic: classify grab errors as fatal (camera removed, init
failure) vs transient (one bad grab), count consecutive transients,
force a reopen past the threshold.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from ..pipeline import Frame, FrameSource, ImuSamples, SourceSpec, StereoCalibration

from ._helpers import CaptureMailbox, notify, notify_verbose


def _notify(msg: str, level: int = logging.INFO) -> None:
    notify("zed", msg, level)


# Periodic capture-rate breadcrumb interval (verbose mode).
_STATS_PERIOD_S = 5.0


# Open() blocks inside pyzed's C extension; print a USB-3 hint after this
# long so the user has something to look at on a USB-2 hang.
_OPEN_HINT_AFTER_S = 8.0


RECONNECT_DELAY_S = 2.0
MAX_CONSECUTIVE_FAILURES = 10

# pyzed.sl.RESOLUTION enum mirror.
_RESOLUTION_DIMS = {
    "HD2K": (2208, 1242),
    "HD1080": (1920, 1080),
    "HD720": (1280, 720),
    "VGA": (672, 376),
}
_DIMS_TO_RESOLUTION = {dims: name for name, dims in _RESOLUTION_DIMS.items()}

#: Camera settings an entry may set, with the range of a manual value. An
#: absent key leaves the camera as the last application set it.
VIDEO_SETTINGS = {
    "exposure": (0, 100),
    "gain": (0, 100),
    "white_balance": (2800, 6500),
    "brightness": (0, 8),
    "contrast": (0, 8),
    "saturation": (0, 8),
    "sharpness": (0, 8),
    "gamma": (1, 9),
}
#: The ones that also take ``auto``, which hands them back to the camera.
_AUTO_SETTINGS = ("exposure", "gain", "white_balance")

# Unread IMU samples kept for a reader that fell behind.
_IMU_BACKLOG = 4096


def video_settings(spec: dict) -> dict:
    """The camera settings a ``type: zed`` entry sets, checked. ``auto``
    becomes None."""
    settings = {}
    for key, (low, high) in VIDEO_SETTINGS.items():
        if key not in spec:
            continue
        value = spec[key]
        takes_auto = key in _AUTO_SETTINGS
        if value == "auto" and takes_auto:
            settings[key] = None
        elif isinstance(value, bool) or not isinstance(value, int):
            wanted = f"auto or {low}..{high}" if takes_auto else f"{low}..{high}"
            raise ValueError(f"ZedSource: {key} must be {wanted}, got {value!r}")
        elif not low <= value <= high:
            raise ValueError(
                f"ZedSource: {key} must be within {low}..{high}, got {value}"
            )
        else:
            settings[key] = value
    shared = [settings[key] for key in ("exposure", "gain") if key in settings]
    if len(shared) == 2 and (shared[0] is None) != (shared[1] is None):
        raise ValueError(
            "ZedSource: exposure and gain share one automatic mode: give both "
            "as numbers, both as auto, or leave one out"
        )
    return settings


def stereo_calibration(info, width: int, height: int) -> StereoCalibration:
    """The calibration of the rectified pair the SDK hands out, from the
    camera information of an open camera."""
    import numpy as np

    params = info.camera_configuration.calibration_parameters
    left = params.left_cam
    K = np.array(
        [[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    baseline_m = float(params.get_camera_baseline())
    # The SDK reports it in its length unit, millimetres unless told
    # otherwise. A stereo baseline is a few centimetres, so a value above 1
    # is in millimetres.
    if baseline_m > 1.0:
        baseline_m *= 1e-3
    try:
        transform = info.sensors_configuration.camera_imu_transform
        R_left_from_imu = np.asarray(transform.m, dtype=np.float64)[:3, :3]
    except Exception:  # a camera without an IMU
        R_left_from_imu = None
    return StereoCalibration(
        width=width,
        height=height,
        left_K=K,
        left_D=np.zeros(5),
        right_K=K.copy(),
        right_D=np.zeros(5),
        R=np.eye(3),
        T=np.array([-baseline_m, 0.0, 0.0]),
        rectified=True,
        R_left_from_imu=R_left_from_imu,
    )


def _resolution_for_dims(width: int, height: int) -> str:
    """YAML width × height → SDK preset. Raises with the valid list on mismatch."""
    key = (int(width), int(height))
    if key in _DIMS_TO_RESOLUTION:
        return _DIMS_TO_RESOLUTION[key]
    valid = ", ".join(f"{n} = {w}x{h}" for n, (w, h) in _RESOLUTION_DIMS.items())
    raise ValueError(
        f"ZedSource: {width}x{height} doesn't match any ZED SDK preset. "
        f"Valid (per-eye): {valid}."
    )


@dataclass
class _EyeSlot:
    """Per-eye GPU output + publish state.

    No host staging — the H2D never happens. ``gpu_buffers`` are the only
    long-lived GPU allocations the eye owns; ``cu_stream`` is the producer
    stream the D2D channel-swap runs on.
    """

    eye: str  # "left" | "right"
    width: int
    height: int
    gpu_buffers: list  # 3 × cupy HxWx4 uint8 RGBA (triple-buffer mailbox)
    cu_stream: object
    write_idx: int = 0
    publish_idx: int = -1
    consumed_idx: int = -2
    lock: threading.Lock = field(default_factory=threading.Lock)
    zed_mat: object = None  # pyzed Mat owning the GPU side

    def upload_and_convert(self, bgra_view) -> None:
        """GPU channel-swap BGRA (pitched view of ZED memory) → contiguous
        RGBA. Alpha column was pre-set to 255 at slot construction so we
        only touch the first three output channels."""
        buf = self.gpu_buffers[self.write_idx]
        with self.cu_stream:
            # bgra_view[..., 2::-1] is (R, G, B) from BGRA's first three slots.
            buf[..., :3] = bgra_view[..., 2::-1]

    def publish(self) -> None:
        with self.lock:
            self.publish_idx = self.write_idx
        self.write_idx = (self.write_idx + 1) % len(self.gpu_buffers)

    def latest(self) -> Optional[Frame]:
        with self.lock:
            if self.publish_idx < 0 or self.publish_idx == self.consumed_idx:
                return None
            idx = self.publish_idx
            self.consumed_idx = idx
        return Frame(
            image=self.gpu_buffers[idx],
            timestamp_ns=time.monotonic_ns(),
            source_id=self.eye,
            stream=0,
        )


class _ZedCamera:
    """Owns the pyzed sl.Camera + producer thread.

    Reference-counted via :meth:`acquire` / :meth:`release`. The first
    acquire starts the producer thread (which opens the camera lazily so
    a missing camera doesn't crash the runner); the last release stops
    it and closes the camera.
    """

    def __init__(
        self,
        serial_number: int,
        bus_type: str,
        width: int,
        height: int,
        fps: int,
        stereo: bool,
        settings: Optional[dict] = None,
        pairs: bool = False,
        imu: bool = False,
    ) -> None:
        try:
            import cupy as cp  # noqa: F401
            import pyzed.sl  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "ZedSource requires CuPy + pyzed. "
                "Run `camera_viz.sh setup --with-zed` with the ZED SDK installed "
                "to install pyzed and CuPy matching your CUDA toolkit."
            ) from e

        self._resolution_name = _resolution_for_dims(width, height)

        if bus_type.lower() not in ("usb", "gmsl"):
            raise ValueError(
                f"ZedSource: unknown bus_type {bus_type!r} (expected usb | gmsl)"
            )

        self._serial_number = serial_number
        self._bus_type = bus_type.lower()
        self._fps = fps
        self._stereo = stereo
        self._width, self._height = _RESOLUTION_DIMS[self._resolution_name]

        self._slots: dict[str, _EyeSlot] = {}
        self._build_slots()

        self._lock = threading.Lock()
        self._refcount = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._camera = None
        self._runtime_params = None
        self._connected = False
        self._last_reconnect_attempt_s = 0.0
        self._consecutive_failures = 0
        self._reconnect_count = 0

        self._settings = dict(settings or {})
        self._calibration: Optional[StereoCalibration] = None
        # For a reader of pairs: both eyes of a capture go into the same
        # buffer index, handed over as one. The per-eye slots work without
        # it.
        self._pairs = pairs and stereo
        self._captures = CaptureMailbox(
            len(next(iter(self._slots.values())).gpu_buffers)
        )
        self._read_imu = imu
        self._imu_lock = threading.Lock()
        # Bounded: a reader that falls behind keeps the newest samples.
        self._imu_samples: deque = deque(maxlen=_IMU_BACKLOG)
        # The reader thread of the open camera and the event that ends it.
        self._imu_reader: Optional[tuple] = None

    def _build_slots(self) -> None:
        import cupy as cp

        eyes = ["left", "right"] if self._stereo else ["left"]
        for eye in eyes:
            gpu_buffers = [
                cp.empty((self._height, self._width, 4), dtype=cp.uint8)
                for _ in range(3)
            ]
            for b in gpu_buffers:
                b[..., 3] = 255
            self._slots[eye] = _EyeSlot(
                eye=eye,
                width=self._width,
                height=self._height,
                gpu_buffers=gpu_buffers,
                cu_stream=cp.cuda.Stream(non_blocking=True),
            )

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ── Lifecycle ─────────────────────────────────────────────────────

    def acquire(self) -> None:
        with self._lock:
            self._refcount += 1
            if self._refcount > 1:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._produce_loop,
                name=f"zed_{self._serial_number or 'auto'}",
                daemon=False,
            )
            self._thread.start()

    def release(self) -> None:
        with self._lock:
            # VizRunner.stop() runs from both the SIGINT handler and the
            # finally-block, so source.stop() (and therefore release()) is
            # called more than once per acquire. Guard the underflow so a
            # second release doesn't drive the refcount negative + re-run
            # the close path on an already-torn-down camera.
            if self._refcount <= 0:
                return
            self._refcount -= 1
            if self._refcount > 0:
                return
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join()
        self._close_camera()

    def latest(self, eye: str) -> Optional[Frame]:
        slot = self._slots.get(eye)
        return slot.latest() if slot is not None else None

    def latest_pair(self, source_id: str) -> Optional[Frame]:
        """Both eyes of the newest capture not handed out yet, stamped with
        its capture time. The frame stays intact until the next call."""
        taken = self._captures.take()
        if taken is None:
            return None
        index, timestamp_ns = taken
        return Frame(
            image=self._slots["left"].gpu_buffers[index],
            image_right=self._slots["right"].gpu_buffers[index],
            timestamp_ns=timestamp_ns,
            source_id=source_id,
            stream=0,
        )

    @property
    def calibration(self) -> Optional[StereoCalibration]:
        return self._calibration

    @property
    def pairs_dropped(self) -> int:
        return self._captures.dropped

    def drain_imu(self) -> Optional[ImuSamples]:
        with self._imu_lock:
            samples = list(self._imu_samples)
            self._imu_samples.clear()
        if not samples:
            return None
        import numpy as np

        return ImuSamples(
            timestamps_ns=np.array([s[0] for s in samples], dtype=np.int64),
            accel=np.array([s[1:] for s in samples], dtype=np.float64),
        )

    # ── Open / close (producer thread only) ───────────────────────────

    def _open_camera(self) -> bool:
        import pyzed.sl as sl

        camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, self._resolution_name)
        init_params.camera_fps = self._fps
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.sdk_verbose = 0
        # Bind the ZED SDK's internal CUDA context to the same GPU our
        # pre-allocated buffers + producer streams live on. Without this
        # pyzed defaults to GPU 0 and retrieve_image(MEM.GPU) → GPU
        # channel-swap fails the device-match check on multi-GPU hosts
        # where VizSession picked a non-default Vulkan adapter.
        first_slot = next(iter(self._slots.values()))
        init_params.sdk_gpu_id = int(first_slot.gpu_buffers[0].device.id)

        bus_enum = sl.BUS_TYPE.USB if self._bus_type == "usb" else sl.BUS_TYPE.GMSL
        if self._serial_number > 0:
            init_params.set_from_serial_number(self._serial_number, bus_enum)
        elif self._bus_type == "gmsl":
            init_params.input.setFromCameraID(-1, bus_enum)

        _notify("opening...")

        # Watchdog hint while pyzed.open() blocks in C. USB-2 is the usual cause.
        opened = threading.Event()

        def _hint():
            if not opened.wait(timeout=_OPEN_HINT_AFTER_S):
                _notify("camera not responding — check USB 3 connection")

        threading.Thread(target=_hint, name="zed_open_hint", daemon=True).start()

        try:
            err = camera.open(init_params)
        finally:
            opened.set()

        if err != sl.ERROR_CODE.SUCCESS:
            _notify(f"open failed ({err})", logging.ERROR)
            try:
                camera.close()
            except Exception:
                pass
            return False

        info = camera.get_camera_information()
        actual_w = info.camera_configuration.resolution.width
        actual_h = info.camera_configuration.resolution.height
        if actual_w != self._width or actual_h != self._height:
            _notify(
                f"resolution mismatch (expected {self._width}x{self._height}, "
                f"got {actual_w}x{actual_h})",
                logging.ERROR,
            )
            camera.close()
            return False

        # The SDK silently grants a lower rate than requested when the
        # resolution/fps combination or the link can't sustain it (USB 2,
        # shared bandwidth). Surface it — a 60 fps config quietly running
        # at 30 is otherwise invisible until someone measures.
        granted_fps = int(getattr(info.camera_configuration, "fps", 0) or 0)
        if granted_fps and granted_fps != self._fps:
            _notify(
                f"SDK granted {granted_fps} fps (requested {self._fps}) — "
                "unsupported resolution/fps combination or USB bandwidth cap"
            )

        # Pyzed Mats are pitched GPU buffers — one per eye, reused every
        # grab() / retrieve_image() pair.
        for slot in self._slots.values():
            slot.zed_mat = sl.Mat()

        self._camera = camera
        self._runtime_params = sl.RuntimeParameters()
        self._consecutive_failures = 0
        self._apply_settings(camera, sl)
        if self._pairs:
            self._calibration = stereo_calibration(info, self._width, self._height)
            if self._read_imu:
                if self._calibration.R_left_from_imu is None:
                    _notify("this camera has no IMU")
                else:
                    self._start_imu(camera, sl)
        return True

    def _apply_settings(self, camera, sl) -> None:
        """Write the entry's camera settings. The camera keeps what the last
        application set, so every listed setting is written at each open."""
        settings = self._settings
        if not settings:
            return
        names = sl.VIDEO_SETTINGS

        def write(setting, value, label: str) -> None:
            result = camera.set_camera_settings(setting, int(value))
            if result is not None and result != sl.ERROR_CODE.SUCCESS:
                _notify(
                    f"setting {label} to {value} failed ({result})", logging.WARNING
                )

        # Exposure and gain share one automatic mode: it is on unless one
        # of them is a number.
        if "exposure" in settings or "gain" in settings:
            manual = {
                key: settings[key]
                for key in ("exposure", "gain")
                if settings.get(key) is not None
            }
            write(names.AEC_AGC, 0 if manual else 1, "auto exposure and gain")
            for key, value in manual.items():
                write(getattr(names, key.upper()), value, key)
        if "white_balance" in settings:
            kelvin = settings["white_balance"]
            write(
                names.WHITEBALANCE_AUTO,
                1 if kelvin is None else 0,
                "auto white balance",
            )
            if kelvin is not None:
                write(names.WHITEBALANCE_TEMPERATURE, kelvin, "white_balance")
        for key in ("brightness", "contrast", "saturation", "sharpness", "gamma"):
            if key in settings:
                write(getattr(names, key.upper()), settings[key], key)

    def _start_imu(self, camera, sl) -> None:
        # An event of its own: a reader that outlives its camera's close
        # must not be woken by the next camera's reader starting.
        stop = threading.Event()
        thread = threading.Thread(
            target=self._imu_loop, args=(camera, sl, stop), name="zed_imu", daemon=True
        )
        self._imu_reader = (thread, stop)
        thread.start()

    def _stop_imu(self) -> bool:
        """End the IMU reader. False when it did not end in time."""
        reader, self._imu_reader = self._imu_reader, None
        if reader is None:
            return True
        thread, stop = reader
        stop.set()
        thread.join(timeout=2.0)
        return not thread.is_alive()

    def _imu_loop(self, camera, sl, stop: threading.Event) -> None:
        """Poll the IMU, which runs several times faster than the camera:
        reading it once per grab would lose most samples. The SDK allows
        this alongside grab()."""
        data = sl.SensorsData()
        last_ns = 0
        while not stop.is_set():
            status = camera.get_sensors_data(data, sl.TIME_REFERENCE.CURRENT)
            if status == sl.ERROR_CODE.SUCCESS:
                imu = data.get_imu_data()
                stamp_ns = int(imu.timestamp.get_nanoseconds())
                if stamp_ns != last_ns:
                    last_ns = stamp_ns
                    a = imu.get_linear_acceleration()
                    with self._imu_lock:
                        self._imu_samples.append((stamp_ns, a[0], a[1], a[2]))
            stop.wait(0.001)

    def _choose_buffers(self) -> None:
        """Point both eyes' next write at a buffer the reader of pairs
        cannot be looking at."""
        index = self._captures.write_index()
        for slot in self._slots.values():
            slot.write_idx = index

    def _publish_pair(self, sl) -> None:
        """Hand over the capture whose two eyes were just published."""
        stamp = self._camera.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        self._captures.publish(
            self._slots["left"].publish_idx, int(stamp.get_nanoseconds())
        )

    def _close_camera(self) -> None:
        imu_ended = self._stop_imu()
        if self._camera is not None:
            if imu_ended:
                try:
                    self._camera.close()
                except Exception:
                    pass
            else:
                # The reader is still inside an SDK call on this camera, and
                # closing it under that call could crash. The handle is
                # leaked instead.
                _notify(
                    "the IMU reader did not stop; leaving the camera open",
                    logging.WARNING,
                )
            self._camera = None
        for slot in self._slots.values():
            slot.zed_mat = None
        self._connected = False

    # ── Producer loop ─────────────────────────────────────────────────

    def _produce_loop(self) -> None:
        import cupy as cp
        import pyzed.sl as sl

        # Pin to the GPU our per-eye buffers + CUDA streams were allocated
        # on at __init__ time. Multi-GPU hosts where VizSession picks a
        # non-default Vulkan adapter otherwise default this thread to GPU 0
        # and break stream/buffer device matching.
        first_slot = next(iter(self._slots.values()))
        device_id = int(first_slot.gpu_buffers[0].device.id)
        with cp.cuda.Device(device_id):
            self._produce_loop_inner(sl)

    def _produce_loop_inner(self, sl) -> None:
        view_enum = {"left": sl.VIEW.LEFT, "right": sl.VIEW.RIGHT}
        fatal_errors = {
            sl.ERROR_CODE.CAMERA_NOT_DETECTED,
            sl.ERROR_CODE.CAMERA_REBOOTING,
            sl.ERROR_CODE.FAILURE,
            sl.ERROR_CODE.CAMERA_NOT_INITIALIZED,
        }

        first_frame_seen = False
        grab_count = 0
        stats_t0 = time.monotonic()
        while not self._stop.is_set():
            if not self._connected:
                now = time.monotonic()
                if now - self._last_reconnect_attempt_s < RECONNECT_DELAY_S:
                    self._stop.wait(timeout=0.1)
                    continue
                self._last_reconnect_attempt_s = now
                try:
                    self._connected = self._open_camera()
                except Exception as e:
                    _notify(f"open failed ({e})", logging.ERROR)
                    self._close_camera()
                    self._reconnect_count += 1
                    continue
                if not self._connected:
                    self._reconnect_count += 1
                    continue
                _notify("connected")
                # Reset the first-frame breadcrumb after each (re)connect.
                first_frame_seen = False

            err = self._camera.grab(self._runtime_params)
            if err != sl.ERROR_CODE.SUCCESS:
                self._consecutive_failures += 1
                if (
                    err in fatal_errors
                    or self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                ):
                    _notify(f"grab failed ({err}); reconnecting", logging.ERROR)
                    self._close_camera()
                continue
            self._consecutive_failures = 0

            try:
                retrieve_failed = False
                if self._pairs:
                    self._choose_buffers()
                for eye, slot in self._slots.items():
                    err = self._camera.retrieve_image(
                        slot.zed_mat, view_enum[eye], sl.MEM.GPU
                    )
                    if err != sl.ERROR_CODE.SUCCESS:
                        retrieve_failed = True
                        continue
                    bgra_view = _zed_mat_as_cupy(slot.zed_mat, sl)
                    if bgra_view is None:
                        retrieve_failed = True
                        continue
                    slot.upload_and_convert(bgra_view)
                    slot.cu_stream.synchronize()
                    slot.publish()
                # Otherwise a consistently-failing retrieve_image with
                # successful grab() would spin forever with no reconnect
                # (the existing _consecutive_failures only tracks grab()).
                if retrieve_failed:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        _notify(
                            "camera connection problem; reconnecting", logging.ERROR
                        )
                        self._close_camera()
                else:
                    self._consecutive_failures = 0
                    if self._pairs:
                        self._publish_pair(sl)
                    if not first_frame_seen:
                        first_frame_seen = True
                        _notify("streaming")
                    # Measured capture rate + the SDK's own estimate.
                    # get_current_fps() exposes auto-exposure throttling
                    # (low light halves the sensor rate even when the
                    # grant was 60) that a wall-clock count alone hides.
                    grab_count += 1
                    now = time.monotonic()
                    if now - stats_t0 >= _STATS_PERIOD_S:
                        measured = grab_count / (now - stats_t0)
                        try:
                            sdk_fps = float(self._camera.get_current_fps())
                        except Exception:
                            sdk_fps = 0.0
                        notify_verbose(
                            "zed",
                            f"capture {measured:.1f} fps"
                            + (f" (SDK current {sdk_fps:.1f})" if sdk_fps else ""),
                        )
                        grab_count = 0
                        stats_t0 = now
            except Exception as e:
                _notify(f"frame error ({e}); reconnecting", logging.ERROR)
                self._close_camera()
                self._reconnect_count += 1
                continue


def _zed_mat_as_cupy(zed_mat, sl):
    """Wrap a pyzed GPU Mat as a CuPy ndarray (zero-copy, pitched).

    Returns None if the Mat has no GPU backing. Strides are explicit to
    account for ZED's row-pitch alignment, which is wider than width*bpp
    on most resolutions."""
    import cupy as cp

    h = zed_mat.get_height()
    w = zed_mat.get_width()
    pixel_bytes = zed_mat.get_pixel_bytes()  # 4 for BGRA8
    step = zed_mat.get_step(sl.MEM.GPU)
    # SDK quirk: get_step occasionally returns the count in pixels
    # instead of bytes. Detect + fix.
    if step == w:
        step = w * pixel_bytes
    ptr = zed_mat.get_pointer(sl.MEM.GPU)
    if ptr is None:
        return None
    mem = cp.cuda.UnownedMemory(ptr, h * step, owner=zed_mat)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    # BGRA8 with explicit pitched strides.
    return cp.ndarray(
        shape=(h, w, 4),
        dtype=cp.uint8,
        memptr=memptr,
        strides=(step, pixel_bytes, 1),
    )


class ZedSource(FrameSource):
    """Per-eye handle over a shared :class:`_ZedCamera`.

    Build via :meth:`ZedSource.build`. Mono cameras (ZED X One) produce a
    single ``left`` source; stereo cameras produce ``left`` and ``right``.
    """

    def __init__(self, camera: _ZedCamera, eye: str, spec: SourceSpec) -> None:
        self._camera = camera
        self._eye = eye
        self._spec = spec

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    @property
    def eye(self) -> str:
        return self._eye

    def start(self) -> None:
        self._camera.acquire()

    def stop(self) -> None:
        self._camera.release()

    def latest(self) -> Optional[Frame]:
        return self._camera.latest(self._eye)

    @classmethod
    def build(
        cls,
        base_name: str,
        width: int,
        height: int,
        fps: int = 30,
        serial_number: int = 0,
        bus_type: str = "usb",
        stereo: bool = True,
        settings: Optional[dict] = None,
    ) -> List["ZedSource"]:
        camera = _ZedCamera(
            serial_number, bus_type, width, height, fps, stereo, settings=settings
        )
        eyes = ["left", "right"] if stereo else ["left"]
        return [
            cls(
                camera,
                eye,
                SourceSpec(f"{base_name}_{eye}", camera.width, camera.height, "rgba8"),
            )
            for eye in eyes
        ]


class ZedStereoSource(FrameSource):
    """Both eyes of each capture of a stereo ZED, in one frame.

    The frame carries the capture's own timestamp, on the SDK's clock.
    Build via :meth:`ZedStereoSource.build`.
    """

    def __init__(self, camera: _ZedCamera, name: str) -> None:
        self._camera = camera
        self._spec = SourceSpec(name, camera.width, camera.height, "rgba8")

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    @property
    def dropped_captures(self) -> int:
        """Captures that were replaced before they were read."""
        return self._camera.pairs_dropped

    def start(self) -> None:
        self._camera.acquire()

    def stop(self) -> None:
        self._camera.release()

    def latest(self) -> Optional[Frame]:
        return self._camera.latest_pair(self._spec.name)

    def calibration(self) -> Optional[StereoCalibration]:
        return self._camera.calibration

    def imu(self) -> Optional[ImuSamples]:
        return self._camera.drain_imu()

    @classmethod
    def build(
        cls,
        name: str,
        width: int,
        height: int,
        fps: int = 30,
        serial_number: int = 0,
        bus_type: str = "usb",
        settings: Optional[dict] = None,
        imu: bool = False,
    ) -> "ZedStereoSource":
        camera = _ZedCamera(
            serial_number,
            bus_type,
            width,
            height,
            fps,
            stereo=True,
            settings=settings,
            pairs=True,
            imu=imu,
        )
        return cls(camera, name)
