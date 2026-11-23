# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OAK-D as a stereo camera: the pair of one capture and, on a model
that can, its colour camera's picture of the same instant.

What the source reads from a device and how it matches frames is tested
everywhere. The source itself runs against a stand-in for depthai; that
needs cupy + a CUDA device and skips cleanly without one.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from collections import deque
from datetime import timedelta
from typing import ClassVar

import numpy as np
import pytest

from isaaccapture_examples.camera_viz.sources import (
    build_stereo_camera,
    check_stereo_camera,
    describe_stereo_camera,
)
from isaaccapture_examples.camera_viz.sources import oakd_stereo as oakd

PAIR = {"name": "oak", "type": "oakd", "stereo": True, "width": 64, "height": 40}
WITH_PICTURE = {
    "name": "oak",
    "type": "oakd",
    "mode": "stereo_rgb",
    "width": 64,
    "height": 40,
    "rgb_width": 128,
    "rgb_height": 80,
    "fps": 45,
    "exposure_limit_ms": 8,
}


def _cuda_available() -> bool:
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not _cuda_available(), reason="needs cupy + a CUDA device"
)


# ── the entry (no GPU needed) ───────────────────────────────────────


def test_the_pair_is_a_stereo_camera_with_or_without_the_colour_picture():
    for entry, picture in ((PAIR, False), (WITH_PICTURE, True)):
        check_stereo_camera(entry)
        traits = describe_stereo_camera(entry)
        # It stores its calibration, of images that are not rectified.
        assert traits.calibrated and not traits.rectified
        assert traits.picture is picture


@pytest.mark.parametrize(
    "entry, message",
    [
        ({**PAIR, "stereo": False}, r"opens one camera unless it sets `stereo: true`"),
        ({**PAIR, "mode": "mono"}, r"opens one camera unless"),
        ({**PAIR, "mode": "depth"}, r"unknown oakd mode 'depth'"),
        ({**PAIR, "fps": "fast"}, r"'oak': width, height, fps, rgb_width"),
        ({**PAIR, "exposure_limit_ms": -1}, r"exposure_limit_ms is a finite number"),
        ({**PAIR, "exposure_limit_ms": float("nan")}, r"exposure_limit_ms is a finite"),
        ({**PAIR, "width": -1}, r"'oak': width, height and fps must be above 0"),
        ({**PAIR, "fps": 0}, r"'oak': width, height and fps must be above 0"),
        ({**PAIR, "rgb_width": -1}, r"rgb_width and rgb_height cannot be negative"),
        ({**PAIR, "height": float("inf")}, r"'oak': width, height, fps, rgb_width"),
        ({k: v for k, v in PAIR.items() if k != "width"}, r"needs width"),
    ],
)
def test_an_oakd_entry_that_cannot_be_a_stereo_camera_is_refused(entry, message):
    with pytest.raises(ValueError, match=message):
        check_stereo_camera(entry)


# ── which devices will do ───────────────────────────────────────────

SENSORS = {"CAM_A": "OV9782", "CAM_B": "OV9282", "CAM_C": "OV9282"}


def test_a_model_with_a_synced_colour_camera_gives_a_picture():
    oakd.check_sensors(SENSORS, picture=True)


def test_a_colour_camera_that_is_not_synced_is_refused_by_name():
    with pytest.raises(oakd.Refused, match=r"IMX378.*not frame-synced.*`mode: stereo`"):
        oakd.check_sensors({**SENSORS, "CAM_A": "IMX378"}, picture=True)


def test_the_pair_alone_needs_no_colour_camera():
    pair = {"CAM_B": "OV9282", "CAM_C": "OV9282"}
    oakd.check_sensors(pair, picture=False)
    oakd.check_sensors({**SENSORS, "CAM_A": "IMX378"}, picture=False)
    with pytest.raises(oakd.Refused, match=r"no camera on CAM_A"):
        oakd.check_sensors(pair, picture=True)
    with pytest.raises(oakd.Refused, match=r"no camera on CAM_C"):
        oakd.check_sensors({"CAM_B": "OV9282"}, picture=False)


# ── the device's calibration ────────────────────────────────────────


class Socket:
    def __init__(self, name):
        self.name = name


LEFT, RIGHT, COLOR = Socket("CAM_B"), Socket("CAM_C"), Socket("CAM_A")

R_LEFT_FROM_COLOR = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R_LEFT_FROM_IMU = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class Calibration:
    """A device's calibration handler: lengths in centimetres."""

    models: ClassVar[dict] = {}
    imu_calibrated: ClassVar[bool] = True

    def getCameraExtrinsics(self, source, target):
        matrix = np.eye(4)
        if (source, target) == (LEFT, RIGHT):
            matrix[:3, 3] = [-7.5, 0.1, 0.0]
        elif (source, target) == (COLOR, LEFT):
            matrix[:3, :3] = R_LEFT_FROM_COLOR
        else:
            raise AssertionError("asked for another pair of cameras")
        return matrix.tolist()

    def getCameraIntrinsics(self, socket, width, height):
        focal = {LEFT: 300.0, RIGHT: 301.0, COLOR: 600.0}[socket]
        return [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]

    def getDistortionCoefficients(self, socket):
        return [{LEFT: 0.1, RIGHT: 0.2, COLOR: 0.3}[socket]] + [0.0] * 13

    def getImuToCameraExtrinsics(self, socket):
        assert socket is LEFT
        if not self.imu_calibrated:
            raise RuntimeError("IMU calibration data is not available on device yet")
        matrix = np.eye(4)
        matrix[:3, :3] = R_LEFT_FROM_IMU
        return matrix.tolist()

    def getDistortionModel(self, socket):
        return types.SimpleNamespace(name=self.models.get(socket.name, "Perspective"))


def test_the_calibration_is_handed_over_in_metres_left_to_right():
    calibration = oakd.stereo_calibration(
        Calibration(), (LEFT, RIGHT, COLOR), (640, 400), (1280, 800)
    )
    assert (calibration.width, calibration.height) == (640, 400)
    assert not calibration.rectified and calibration.distortion_model == "brown_conrady"
    assert calibration.left_K[0, 0] == 300.0 and calibration.right_K[0, 0] == 301.0
    assert calibration.left_K[0, 2] == 320.0 and calibration.left_K[1, 2] == 200.0
    assert calibration.left_D[0] == 0.1 and calibration.right_D[0] == 0.2
    np.testing.assert_array_equal(calibration.R, np.eye(3))
    np.testing.assert_allclose(calibration.T, [-0.075, 0.001, 0.0])
    assert calibration.R_left_from_imu is None  # read when the IMU is
    np.testing.assert_array_equal(
        oakd.imu_rotation(Calibration(), LEFT), R_LEFT_FROM_IMU
    )
    picture = calibration.picture
    assert (picture.width, picture.height) == (1280, 800)
    assert picture.K[0, 0] == 600.0 and picture.K[0, 2] == 640.0 and picture.D[0] == 0.3
    np.testing.assert_array_equal(picture.R_left_from_picture, R_LEFT_FROM_COLOR)
    assert picture.distortion_model == "brown_conrady"


def test_the_pair_alone_has_no_picture_calibration_and_no_imu_without_one():
    calibration = oakd.stereo_calibration(
        Calibration(), (LEFT, RIGHT, COLOR), (640, 400)
    )
    assert calibration.picture is None and calibration.R_left_from_imu is None


def test_the_lens_model_the_device_names_is_passed_on_or_refused(monkeypatch):
    sockets = (LEFT, RIGHT, COLOR)
    fisheye = {"CAM_B": "Fisheye", "CAM_C": "Fisheye"}
    monkeypatch.setattr(Calibration, "models", fisheye)
    calibration = oakd.stereo_calibration(
        Calibration(), sockets, (640, 400), (1280, 800)
    )
    assert calibration.distortion_model == "fisheye"
    assert calibration.picture.distortion_model == "brown_conrady"
    # OpenCV's fisheye model takes four coefficients; the other, all.
    assert calibration.left_D.shape == calibration.right_D.shape == (4,)
    assert calibration.picture.D.shape == (14,)

    monkeypatch.setattr(Calibration, "models", {"CAM_B": "Fisheye"})
    with pytest.raises(oakd.Refused, match=r"two eyes with different lens models"):
        oakd.stereo_calibration(Calibration(), sockets, (640, 400))
    monkeypatch.setattr(Calibration, "models", {"CAM_B": "Equirectangular"})
    with pytest.raises(oakd.Refused, match=r"as Equirectangular, which is not handled"):
        oakd.stereo_calibration(Calibration(), sockets, (640, 400))


# ── matching the frames of one exposure ─────────────────────────────

MS = 1_000_000
THREE = ("left", "right", "picture")


def test_frames_exposed_together_match_whatever_their_sequence_numbers():
    matcher = oakd.FrameMatcher(THREE)
    matcher.add("left", 100 * MS, "L7")
    matcher.add("right", 100 * MS + 3_000, "R7")
    matcher.add("picture", 100 * MS + 20_000, "C5")  # its count started later
    assert matcher.take_next() == (
        100 * MS,
        {"left": "L7", "right": "R7", "picture": "C5"},
    )
    assert matcher.take_next() is None


def test_a_picture_from_another_exposure_does_not_match():
    matcher = oakd.FrameMatcher(THREE)
    matcher.add("left", 100 * MS, "L")
    matcher.add("right", 100 * MS, "R")
    matcher.add("picture", 105 * MS, "C")  # unsynced: it trails by 5 ms
    assert matcher.take_next() is None


def test_sets_come_out_in_order_each_once():
    matcher = oakd.FrameMatcher(("left", "right"))
    for n in (3, 1, 2):  # a burst, out of order
        matcher.add("left", n * 22 * MS, f"L{n}")
        matcher.add("right", n * 22 * MS, f"R{n}")
    taken = [matcher.take_next()[1]["left"] for _ in range(3)]
    assert taken == ["L1", "L2", "L3"] and matcher.take_next() is None


def test_a_set_that_lost_a_frame_does_not_hold_up_the_next():
    matcher = oakd.FrameMatcher(("left", "right"))
    matcher.add("left", 22 * MS, "L1")  # its right eye never comes
    matcher.add("left", 44 * MS, "L2")
    matcher.add("right", 44 * MS, "R2")
    assert matcher.take_next()[1] == {"left": "L2", "right": "R2"}
    assert matcher.take_next() is None  # L1 went with it


def test_frames_that_never_match_do_not_pile_up():
    matcher = oakd.FrameMatcher(("left", "right"), keep=4)
    for n in range(50):
        matcher.add("left", n * 22 * MS, n)
    assert len(matcher._frames["left"]) == 4


def test_a_device_timestamp_becomes_whole_nanoseconds():
    assert oakd._device_ns(timedelta(seconds=3, microseconds=7)) == 3_000_007_000


# ── the source against a stand-in depthai (GPU) ─────────────────────

CAPTURE_PERIOD_US = 22_222  # 45 fps


class _ImgFrame:
    def __init__(self, capture, sensor, size, offset_us):
        self.capture, self.sensor, self.size = capture, sensor, size
        self._stamp = timedelta(microseconds=capture * CAPTURE_PERIOD_US + offset_us)

    def getTimestampDevice(self):
        return self._stamp

    def getFrame(self):
        width, height = self.size
        value = self.capture % 100 + (100 if self.sensor == "right" else 0)
        return np.full((height, width), value, dtype=np.uint8)

    def getCvFrame(self):  # packed BGR
        width, height = self.size
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[...] = (self.capture % 100, 7, 200)
        return frame


class _Queue:
    def __init__(self):
        self.items = deque()

    def has(self):
        return bool(self.items)

    def get(self):
        return self.items.popleft()


class _Output:
    def __init__(self, queue):
        self._queue = queue

    def createOutputQueue(self, maxSize, blocking):
        return self._queue


class _Control:
    def __init__(self):
        self.sync = None
        self.exposure_limit_us = None

    def setFrameSyncMode(self, mode):
        self.sync = mode

    def setAutoExposureLimit(self, microseconds):
        self.exposure_limit_us = microseconds


class _CameraNode:
    def build(self, socket):
        self.socket = socket
        self.initialControl = _Control()
        self.queue = _Queue()
        return self

    def requestOutput(self, size, type, fps):  # noqa: A002 - depthai's name
        if size in _Device.refused_sizes:
            raise RuntimeError(f"the sensor has no mode for {size[0]}x{size[1]}")
        self.size, self.type, self.fps = size, type, fps
        return _Output(self.queue)


class _ImuNode:
    def __init__(self):
        self.queue = _Queue()
        self.out = _Output(self.queue)
        self.sensor = self.rate = None

    def enableIMUSensor(self, sensor, rate):
        self.sensor, self.rate = sensor, rate

    def setBatchReportThreshold(self, _count):
        pass

    def setMaxBatchReports(self, _count):
        pass


class _Pipeline:
    """Sends one capture per ``processTasks`` call that comes a capture
    period after the last one."""

    def __init__(self, device):
        self.device = device
        self.cameras, self.imu = [], None
        self.capture = 0
        self._last = 0.0
        device.pipeline = self

    def setXLinkChunkSize(self, _size):
        pass

    def create(self, node_type):
        node = node_type()
        if isinstance(node, _ImuNode):
            self.imu = node
        else:
            self.cameras.append(node)
        return node

    def start(self):
        pass

    def stop(self):
        pass

    def processTasks(self):
        if time.monotonic() - self._last < 0.004:
            return
        self._last = time.monotonic()
        self.capture += 1
        fails_after = self.device.link_fails_after.get(
            _Device.opened.index(self.device)
        )
        if fails_after is not None and self.capture > fails_after:
            raise RuntimeError("X_LINK_ERROR")
        for camera in self.cameras:
            sensor = {"CAM_B": "left", "CAM_C": "right", "CAM_A": "picture"}[
                camera.socket.name
            ]
            offset_us = self.device.offsets_us[sensor]
            camera.queue.items.append(
                _ImgFrame(self.capture, sensor, camera.size, offset_us)
            )
        if self.imu is not None:
            stamp = timedelta(microseconds=self.capture * CAPTURE_PERIOD_US)
            report = types.SimpleNamespace(
                x=0.0, y=9.81, z=float(self.capture), getTimestampDevice=lambda: stamp
            )
            packet = types.SimpleNamespace(acceleroMeter=report)
            self.imu.queue.items.append(types.SimpleNamespace(packets=[packet]))


class _Device:
    opened: ClassVar[list] = []
    sensors: ClassVar[dict] = SENSORS
    imu_name: ClassVar[str] = "BNO086"
    #: Each sensor's timestamp, after the capture's own.
    offsets_us: ClassVar[dict] = {"left": 0, "right": 3, "picture": 20}
    #: How many times the device is looked for on USB before it is there.
    not_found: ClassVar[list] = [0]
    #: Sizes its sensors will not stream.
    refused_sizes: ClassVar[tuple] = ()
    #: Per opened device, in order: after how many captures its link fails.
    link_fails_after: ClassVar[dict] = {}

    @staticmethod
    def getAllAvailableDevices():
        if _Device.not_found[0] > 0:
            _Device.not_found[0] -= 1
            return []
        return ["DeviceInfo(deviceId=14442C10)"]

    def __init__(self, info=None):
        self.info = info
        self.closed = False
        self.pipeline = None
        _Device.opened.append(self)

    def getConnectedCameraFeatures(self):
        return [
            types.SimpleNamespace(socket=Socket(socket), sensorName=sensor)
            for socket, sensor in self.sensors.items()
        ]

    def setMaxReconnectionAttempts(self, attempts):
        self.reconnection_attempts = attempts

    def getUsbSpeed(self):
        return types.SimpleNamespace(name="SUPER")

    def getConnectedIMU(self):
        return self.imu_name

    def readCalibration(self):
        return Calibration()

    def close(self):
        self.closed = True


@pytest.fixture
def depthai(monkeypatch):
    """The stand-in, in place of ``depthai``. Yields the devices opened."""
    dai = types.ModuleType("depthai")
    dai.Device, dai.Pipeline = _Device, _Pipeline
    dai.DeviceInfo = lambda device_id: device_id
    dai.CameraBoardSocket = types.SimpleNamespace(CAM_A=COLOR, CAM_B=LEFT, CAM_C=RIGHT)
    dai.node = types.SimpleNamespace(Camera=_CameraNode, IMU=_ImuNode)
    dai.ImgFrame = types.SimpleNamespace(
        Type=types.SimpleNamespace(GRAY8="GRAY8", NV12="NV12")
    )
    dai.CameraControl = types.SimpleNamespace(
        FrameSyncMode=types.SimpleNamespace(OUTPUT="OUTPUT", INPUT="INPUT")
    )
    dai.IMUSensor = types.SimpleNamespace(ACCELEROMETER_RAW="ACCELEROMETER_RAW")
    monkeypatch.setitem(sys.modules, "depthai", dai)
    monkeypatch.setattr(_Device, "opened", [])
    monkeypatch.setattr(_Device, "not_found", [0])
    monkeypatch.setattr(oakd, "RECONNECT_DELAY_S", 0.01)
    return _Device.opened


def _read(source, count, timeout_s=10.0):
    """The next ``count`` frames, one at a time: a frame is the source's to
    reuse once the next is asked for."""
    deadline = time.monotonic() + timeout_s
    while count and time.monotonic() < deadline:
        frame = source.latest()
        if frame is None:
            time.sleep(0.001)
            continue
        count -= 1
        yield frame
    assert not count, f"{count} frames did not arrive"


def _wait(source, count=1):
    """Let ``count`` frames arrive. Returns the last, the only one still
    intact."""
    return list(_read(source, count))[-1]


@requires_gpu
def test_a_frame_holds_the_pair_and_the_picture_of_one_capture(depthai):
    source = build_stereo_camera(WITH_PICTURE)
    assert source.calibration() is None  # the device is not open yet
    source.start()
    try:
        seen = []
        for frame in _read(source, 8):
            capture = round(frame.timestamp_ns / (CAPTURE_PERIOD_US * 1000))
            left, right = frame.image.get(), frame.image_right.get()
            picture = frame.picture.get()
            assert left.shape == right.shape == (40, 64, 4)
            assert picture.shape == (80, 128, 4)
            # The mono eyes as gray RGBA, the colour camera's BGR as RGBA.
            assert (left == [capture % 100] * 3 + [255]).all()
            assert (right == [capture % 100 + 100] * 3 + [255]).all()
            assert (picture == [200, 7, capture % 100, 255]).all()
            seen.append(capture)
        calibration = source.calibration()
    finally:
        source.stop()
    assert seen == sorted(set(seen)), "a capture was repeated or out of order"
    assert calibration.picture.width == 128 and calibration.T[0] == pytest.approx(
        -0.075
    )
    assert depthai[0].closed


@requires_gpu
def test_the_left_sensor_drives_the_sync_and_the_exposure_is_limited(depthai):
    source = build_stereo_camera(WITH_PICTURE)
    source.start()
    try:
        _wait(source)
    finally:
        source.stop()
    cameras = {camera.socket.name: camera for camera in depthai[0].pipeline.cameras}
    assert cameras["CAM_B"].initialControl.sync == "OUTPUT"
    assert cameras["CAM_C"].initialControl.sync == "INPUT"
    assert cameras["CAM_A"].initialControl.sync == "INPUT"
    assert {c.initialControl.exposure_limit_us for c in cameras.values()} == {8000}
    assert (cameras["CAM_B"].size, cameras["CAM_B"].type) == ((64, 40), "GRAY8")
    assert (cameras["CAM_A"].size, cameras["CAM_A"].type) == ((128, 80), "NV12")
    assert {camera.fps for camera in cameras.values()} == {45}


@requires_gpu
def test_the_pair_alone_opens_two_sensors_and_gives_no_picture(depthai):
    source = build_stereo_camera(PAIR)
    source.start()
    try:
        frame = _wait(source)
        assert frame.picture is None and frame.image_right is not None
        assert source.calibration().picture is None
    finally:
        source.stop()
    sockets = [camera.socket.name for camera in depthai[0].pipeline.cameras]
    assert sockets == ["CAM_B", "CAM_C"]
    # Nothing asked for a limit or a picture: exposure and frame sync are
    # left to the device.
    for camera in depthai[0].pipeline.cameras:
        assert camera.initialControl.exposure_limit_us is None
        assert camera.initialControl.sync is None


@requires_gpu
def test_the_accelerometer_is_read_only_when_asked_for(depthai):
    quiet = build_stereo_camera(PAIR)
    quiet.start()
    try:
        _wait(quiet, 2)
        assert quiet.imu() is None
    finally:
        quiet.stop()
    assert depthai[0].pipeline.imu is None

    source = build_stereo_camera(PAIR, imu=True)
    source.start()
    try:
        _wait(source, 4)
        samples = source.imu()
    finally:
        source.stop()
    node = depthai[1].pipeline.imu
    assert (node.sensor, node.rate) == ("ACCELEROMETER_RAW", 200)
    assert samples.timestamps_ns.dtype == np.int64 and samples.gyro is None
    assert (np.diff(samples.timestamps_ns) > 0).all()
    np.testing.assert_array_equal(samples.accel[:, 1], 9.81)
    np.testing.assert_array_equal(source.calibration().R_left_from_imu, R_LEFT_FROM_IMU)


@requires_gpu
def test_a_device_without_an_imu_is_not_asked_for_one(depthai, monkeypatch):
    monkeypatch.setattr(_Device, "imu_name", "NONE")
    source = build_stereo_camera(PAIR, imu=True)
    source.start()
    try:
        _wait(source, 2)
        assert source.imu() is None
        assert source.calibration().R_left_from_imu is None
    finally:
        source.stop()
    assert depthai[0].pipeline.imu is None


@requires_gpu
def test_a_device_that_cannot_give_the_picture_tells_the_reader_why(
    depthai, monkeypatch
):
    monkeypatch.setattr(_Device, "sensors", {**SENSORS, "CAM_A": "IMX378"})
    source = build_stereo_camera(WITH_PICTURE)
    source.start()
    try:
        deadline = time.monotonic() + 5.0
        with pytest.raises(
            RuntimeError, match=r"OAK-D 'oak'.*IMX378.*not frame-synced"
        ):
            while time.monotonic() < deadline:
                source.latest()
                time.sleep(0.005)
    finally:
        source.stop()
    assert len(depthai) == 1 and depthai[0].closed  # not reopened: it would not help


@requires_gpu
def test_a_held_frame_stays_intact_while_new_captures_arrive(depthai):
    source = build_stereo_camera(WITH_PICTURE)
    source.start()
    try:
        held = _wait(source)
        before = held.image.get(), held.image_right.get(), held.picture.get()
        time.sleep(0.08)
        after = held.image.get(), held.image_right.get(), held.picture.get()
        for was, now in zip(before, after):
            np.testing.assert_array_equal(now, was)
        later = _wait(source)
        assert later.timestamp_ns > held.timestamp_ns and source.dropped_captures > 0
        assert (later.image.get() != before[0]).any()
    finally:
        source.stop()


@requires_gpu
def test_a_restarted_source_hands_out_nothing_of_its_last_session(depthai):
    source = build_stereo_camera(PAIR, imu=True)
    source.start()
    try:
        _wait(source, 6)
        time.sleep(0.08)  # captures and samples arrive and stay unread
    finally:
        source.stop()
    source.start()
    try:
        stamps = [frame.timestamp_ns for frame in _read(source, 3)]
        samples = source.imu()
    finally:
        source.stop()
    # The reopened device's clock starts again, so anything of the last
    # session would be stamped later than what follows it.
    assert stamps == sorted(stamps)
    assert (np.diff(samples.timestamps_ns) > 0).all()


@requires_gpu
def test_a_source_whose_stop_gave_up_waiting_closes_its_device_and_starts_again(
    depthai, monkeypatch
):
    monkeypatch.setattr(oakd, "_STOP_S", 0.05)
    process, stuck, release = (
        _Pipeline.processTasks,
        threading.Event(),
        threading.Event(),
    )

    def blocks(pipeline):  # a depthai call that does not return in time
        stuck.set()
        release.wait(10.0)
        process(pipeline)

    source = build_stereo_camera(PAIR)
    source.start()
    try:
        _wait(source)
        monkeypatch.setattr(_Pipeline, "processTasks", blocks)
        assert stuck.wait(10.0)
        source.stop()  # gives up: the producer is inside the call
        assert not depthai[0].closed
        with pytest.raises(RuntimeError, match="still stopping"):
            source.start()
        monkeypatch.setattr(_Pipeline, "processTasks", process)
        release.set()  # the call returns, and the producer ends
        deadline = time.monotonic() + 10.0
        while True:
            try:
                source.start()
                break
            except RuntimeError:
                assert time.monotonic() < deadline
                time.sleep(0.005)
        assert depthai[0].closed  # by the producer, once it was out of the call
        assert _wait(source).image_right is not None
    finally:
        release.set()
        monkeypatch.setattr(oakd, "_STOP_S", 5.0)  # this stop waits
        source.stop()
    assert len(depthai) == 2 and depthai[1].closed


def _error(source, timeout_s=5.0):
    """What ``latest()`` raises once the source has given up."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            source.latest()
        except RuntimeError as error:
            return str(error)
        time.sleep(0.002)
    raise AssertionError("the source did not report an error")


@requires_gpu
def test_a_picture_that_trails_the_pair_is_never_handed_out(depthai, monkeypatch):
    monkeypatch.setattr(oakd, "NO_MATCH_TIMEOUT_S", 0.05)
    # An unsynced colour camera: 5 ms behind every capture.
    monkeypatch.setattr(_Device, "offsets_us", {"left": 0, "right": 3, "picture": 5000})
    source = build_stereo_camera(WITH_PICTURE)
    source.start()
    try:
        message, deadline = None, time.monotonic() + 5.0
        while message is None:  # no frame, until the source gives up
            assert time.monotonic() < deadline, "the source did not report an error"
            try:
                assert source.latest() is None
            except RuntimeError as error:
                message = str(error)
            time.sleep(0.002)
        assert "no set of frames exposed together" in message
    finally:
        source.stop()
    assert len(depthai) == 1 and depthai[0].closed


@requires_gpu
def test_a_device_that_is_not_there_yet_is_waited_for(depthai, monkeypatch):
    monkeypatch.setattr(_Device, "not_found", [3])
    source = build_stereo_camera(PAIR)
    source.start()
    try:
        assert _wait(source).image_right is not None
    finally:
        source.stop()
    assert len(depthai) == 1  # opened once, when it was there


@requires_gpu
def test_another_device_than_the_one_named_is_not_opened(depthai):
    source = build_stereo_camera({**PAIR, "device_id": "0123456789ABCDEF"})
    source.start()
    try:
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            assert source.latest() is None
            time.sleep(0.005)
    finally:
        started = time.monotonic()
        source.stop()
    assert time.monotonic() - started < 1.0, "stop waited for a device search"
    assert depthai == []

    named = build_stereo_camera({**PAIR, "device_id": "14442C10"})
    named.start()
    try:
        assert _wait(named).image_right is not None
    finally:
        named.stop()
    assert depthai[0].info == "14442C10"


@requires_gpu
def test_a_device_that_cannot_be_set_up_as_asked_is_reported_once(depthai, monkeypatch):
    monkeypatch.setattr(_Device, "refused_sizes", ((64, 40),))
    source = build_stereo_camera(PAIR)
    source.start()
    try:
        assert "could not be set up: the sensor has no mode for 64x40" in _error(source)
        time.sleep(0.1)  # several reopen delays
    finally:
        source.stop()
    assert len(depthai) == 1 and depthai[0].closed, "it was opened again"


@requires_gpu
def test_an_error_after_the_first_capture_is_final(depthai, monkeypatch):
    """Reopening restarts the device's clock, which stamps its frames."""
    source = build_stereo_camera(PAIR)
    source.start()
    try:
        _wait(source)
        monkeypatch.setattr(_Device, "link_fails_after", {0: 0})  # from now on
        message = _error(source)
    finally:
        source.stop()
    assert (
        "the device stopped (X_LINK_ERROR)" in message
        and "restart the clock" in message
    )
    assert len(depthai) == 1 and depthai[0].closed


@requires_gpu
def test_an_error_after_imu_samples_were_read_is_final_too(depthai, monkeypatch):
    """They carry the device's clock as well, capture or no capture."""
    # The right sensor trails the left by 5 ms: no pair ever matches.
    monkeypatch.setattr(_Device, "offsets_us", {"left": 0, "right": 5000})
    source = build_stereo_camera(PAIR, imu=True)
    source.start()
    try:
        deadline = time.monotonic() + 10.0
        while source.imu() is None:
            assert time.monotonic() < deadline, "no IMU sample arrived"
            time.sleep(0.002)
        monkeypatch.setattr(_Device, "link_fails_after", {0: 0})  # from now on
        message = _error(source)
    finally:
        source.stop()
    assert "the device stopped (X_LINK_ERROR)" in message
    assert len(depthai) == 1 and depthai[0].closed


@requires_gpu
def test_the_device_is_told_not_to_reconnect_by_itself(depthai):
    """A device depthai reopened would restart its clock unnoticed."""
    source = build_stereo_camera(PAIR)
    source.start()
    try:
        _wait(source)
    finally:
        source.stop()
    assert depthai[0].reconnection_attempts == 0


@requires_gpu
def test_an_error_before_the_first_capture_reopens_the_device(depthai, monkeypatch):
    monkeypatch.setattr(_Device, "link_fails_after", {0: 0})
    source = build_stereo_camera(PAIR)
    source.start()
    try:
        assert _wait(source).image_right is not None
    finally:
        source.stop()
    assert len(depthai) == 2 and depthai[0].closed


@requires_gpu
def test_an_imu_without_a_calibration_is_not_read_and_the_camera_still_opens(
    depthai, monkeypatch
):
    monkeypatch.setattr(Calibration, "imu_calibrated", False)
    source = build_stereo_camera(PAIR, imu=True)
    source.start()
    try:
        _wait(source, 2)
        assert source.imu() is None and source.calibration().R_left_from_imu is None
    finally:
        source.stop()
    assert depthai[0].pipeline.imu is None


@requires_gpu
def test_frames_of_another_size_than_asked_for_end_the_source(depthai, monkeypatch):
    monkeypatch.setattr(
        _ImgFrame, "getFrame", lambda self: np.zeros((10, 10), dtype=np.uint8)
    )
    source = build_stereo_camera(PAIR)
    source.start()
    try:
        assert "sent left frames of 10x10, not the 64x40" in _error(source)
    finally:
        source.stop()
