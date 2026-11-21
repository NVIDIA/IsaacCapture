# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stereo cameras: ``build_stereo_camera`` hands out both eyes of one
capture per frame, or says why an entry cannot.

The checks run everywhere. The ZED tests drive the real source against a
stand-in for the ZED SDK; they need cupy + a CUDA device and skip cleanly
without one.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import ClassVar

import numpy as np
import pytest

from isaaccapture_examples.camera_viz.sources import (
    build_local_camera,
    build_stereo_camera,
    check_stereo_camera,
    describe_stereo_camera,
)
from isaaccapture_examples.camera_viz.sources import zed as zed_source
from isaaccapture_examples.camera_viz.sources._helpers import CaptureMailbox

ZED = {"name": "head", "type": "zed", "stereo": True, "width": 1280, "height": 720}


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


# ── which entries are stereo cameras (no GPU needed) ────────────────


@pytest.mark.parametrize(
    "entry",
    [
        ZED,
        {"name": "clip", "type": "video", "stereo": True, "path": "clip.mp4"},
    ],
)
def test_a_stereo_camera_passes_the_check(entry):
    check_stereo_camera(entry)


@pytest.mark.parametrize(
    "entry, reason",
    [
        ({**ZED, "stereo": False}, r"opens one eye unless it sets `stereo: true`"),
        ({"name": "clip", "type": "video", "path": "clip.mp4"}, r"it is one view"),
        ({"name": "cam", "type": "v4l2"}, r"gives one image"),
        (
            {"name": "bars", "type": "synthetic", "width": 64, "height": 48},
            r"it is one view",
        ),
        (
            {
                "name": "bars",
                "type": "synthetic",
                "stereo": True,
                "width": 8,
                "height": 8,
            },
            r"nothing behind it",
        ),
        ({"name": "head", "type": "argus", "stereo": True}, r"`sync_session: true`"),
        (
            {"name": "head", "type": "argus", "stereo": True, "sync_session": True},
            r"not offered as a stereo camera yet",
        ),
        (
            {
                "name": "oak",
                "type": "oakd",
                "stereo": True,
                "width": 640,
                "height": 400,
            },
            r"an oakd camera is not offered as a stereo camera yet",
        ),
    ],
)
def test_an_entry_that_is_not_a_stereo_camera_is_refused_with_the_reason(entry, reason):
    with pytest.raises(
        ValueError, match=rf"{entry['name']!r} is not a stereo.*{reason}"
    ):
        check_stereo_camera(entry)
    with pytest.raises(ValueError, match=reason):
        build_stereo_camera(entry)


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"name": "x", "type": "lidar"}, r"unknown camera type 'lidar'"),
        ({k: v for k, v in ZED.items() if k != "height"}, r"needs height"),
        ({**ZED, "width": 1000}, r"doesn't match any ZED SDK preset"),
        ({"name": "clip", "type": "video", "stereo": True}, r"needs path"),
        ("zed", r"a camera entry is a mapping"),
        ({**ZED, "height": None}, r"'head': width, height, fps and serial_number"),
        ({**ZED, "fps": "fast"}, r"'head': width, height, fps and serial_number"),
        ({**ZED, "bus_type": "pcie"}, r"bus_type is usb or gmsl, got 'pcie'"),
        (
            {"name": "clip", "type": "video", "stereo": True, "path": "c", "fps": "x"},
            r"'clip': width, height and fps are numbers",
        ),
    ],
)
def test_a_malformed_entry_is_refused(entry, message):
    with pytest.raises(ValueError, match=message):
        check_stereo_camera(entry)


def test_a_stereo_camera_is_described_before_it_is_opened():
    clip = {"name": "clip", "type": "video", "stereo": True, "path": "clip.mp4"}
    # (stores its calibration, hands out rectified images)
    for entry, traits in (
        (ZED, (True, True)),
        (clip, (False, False)),
    ):
        described = describe_stereo_camera(entry)
        assert (described.calibrated, described.rectified) == traits
    with pytest.raises(ValueError, match=r"is not a stereo camera"):
        describe_stereo_camera({"name": "cam", "type": "v4l2"})


def test_a_calibration_names_a_lens_model_that_is_known():
    from isaaccapture_examples.camera_viz.pipeline import (
        DISTORTION_MODELS,
        StereoCalibration,
    )

    eye, none = np.eye(3), np.zeros(5)
    for model in DISTORTION_MODELS:
        StereoCalibration(8, 8, eye, none, eye, none, eye, none, distortion_model=model)
    with pytest.raises(ValueError, match=r"unknown distortion_model 'equidistant'"):
        StereoCalibration(
            8, 8, eye, none, eye, none, eye, none, distortion_model="equidistant"
        )


def test_a_calibration_is_read_only_and_compared_by_identity():
    from isaaccapture_examples.camera_viz.pipeline import StereoCalibration

    def calibration():
        eye, none = np.eye(3), np.zeros(5)
        return StereoCalibration(8, 8, eye, none, eye.copy(), none, eye, np.zeros(3))

    one, two = calibration(), calibration()
    assert one == one and one != two and len({one, two}) == 2
    with pytest.raises(ValueError, match=r"read-only"):
        one.left_K[0, 0] = 2.0


def test_zed_settings_take_a_number_in_range_or_auto_where_the_camera_has_it():
    settings = zed_source.video_settings(
        {**ZED, "exposure": 40, "gain": 12, "white_balance": "auto", "gamma": 5}
    )
    assert settings == {"exposure": 40, "gain": 12, "white_balance": None, "gamma": 5}
    assert zed_source.video_settings({**ZED, "exposure": "auto", "gain": "auto"}) == {
        "exposure": None,
        "gain": None,
    }
    assert zed_source.video_settings(ZED) == {}
    for bad, message in [
        ({"exposure": 300}, r"exposure must be within 0\.\.100"),
        ({"gamma": "auto"}, r"gamma must be 1\.\.9"),
        ({"gain": 2.5}, r"gain must be auto or 0\.\.100"),
        ({"contrast": True}, r"contrast must be 0\.\.8"),
        # One automatic mode for both: an auto beside a number cannot hold.
        ({"exposure": 40, "gain": "auto"}, r"share one automatic mode"),
        ({"exposure": "auto", "gain": 12}, r"share one automatic mode"),
    ]:
        with pytest.raises(ValueError, match=message):
            check_stereo_camera({**ZED, **bad})


def _camera_information(baseline, with_imu=True):
    left = types.SimpleNamespace(fx=700.0, fy=701.0, cx=640.5, cy=360.25)
    info = types.SimpleNamespace(
        camera_configuration=types.SimpleNamespace(
            resolution=types.SimpleNamespace(width=1280, height=720),
            fps=60,
            calibration_parameters=types.SimpleNamespace(
                left_cam=left, get_camera_baseline=lambda: baseline
            ),
        ),
        sensors_configuration=types.SimpleNamespace(),
    )
    if with_imu:
        m = np.eye(4)
        m[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        info.sensors_configuration.camera_imu_transform = types.SimpleNamespace(m=m)
    return info


@pytest.mark.parametrize("reported", [63.0, 0.063])
def test_the_calibration_is_the_rectified_one_with_the_baseline_in_metres(reported):
    calibration = zed_source.stereo_calibration(
        _camera_information(reported), 1280, 720
    )
    assert calibration.rectified and (calibration.width, calibration.height) == (
        1280,
        720,
    )
    np.testing.assert_array_equal(
        calibration.left_K, [[700.0, 0.0, 640.5], [0.0, 701.0, 360.25], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_array_equal(calibration.right_K, calibration.left_K)
    assert not calibration.left_D.any() and not calibration.right_D.any()
    np.testing.assert_array_equal(calibration.R, np.eye(3))
    np.testing.assert_allclose(calibration.T, [-0.063, 0.0, 0.0])
    np.testing.assert_array_equal(
        calibration.R_left_from_imu,
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )


def test_a_camera_without_an_imu_has_no_imu_rotation():
    info = _camera_information(63.0, with_imu=False)
    assert zed_source.stereo_calibration(info, 1280, 720).R_left_from_imu is None


# ── the ZED source against a stand-in SDK (GPU) ─────────────────────

CAPTURE_PERIOD_S = 0.01
FIRST_CAPTURE_NS = 1_700_000_000_000_000_000
CAPTURE_PERIOD_NS = 16_666_667


class _Enum:
    """Attribute access gives the attribute's own name: enough of an enum."""

    def __getattr__(self, name):
        return name


class _Mat:
    """A GPU image of the stand-in SDK: BGRA, tightly packed."""

    def __init__(self):
        self.array = None

    def get_height(self):
        return self.array.shape[0]

    def get_width(self):
        return self.array.shape[1]

    def get_pixel_bytes(self):
        return 4

    def get_step(self, _mem):
        return self.array.shape[1] * 4

    def get_pointer(self, _mem):
        return self.array.data.ptr


class _Camera:
    """One capture per grab(). Both eyes of capture ``n`` carry ``n`` in
    their red channel; green tells the eyes apart. Captures are counted
    across cameras, as a clock is, so a camera opened after a reconnect
    carries on."""

    opened: ClassVar[list] = []
    captures: ClassVar[list] = [0]
    #: How many grabs each opened camera, in order, survives; absent: all.
    grabs_before_unplug: ClassVar[dict] = {}
    information: ClassVar = staticmethod(lambda: _camera_information(63.0))

    def __init__(self):
        self.capture = 0
        self.grabs = 0
        self.settings = []
        self.timestamp_reads = 0
        self.sensor_reads = 0
        self.closed = threading.Event()
        self._unplug_after = _Camera.grabs_before_unplug.get(len(_Camera.opened))
        _Camera.opened.append(self)

    def open(self, _init):
        return "SUCCESS"

    def close(self):
        self.closed.set()

    def get_camera_information(self):
        return _Camera.information()

    def grab(self, _runtime):
        time.sleep(CAPTURE_PERIOD_S)
        self.grabs += 1
        if self._unplug_after is not None and self.grabs > self._unplug_after:
            return "CAMERA_NOT_DETECTED"
        _Camera.captures[0] += 1
        self.capture = _Camera.captures[0]
        return "SUCCESS"

    def retrieve_image(self, mat, view, _mem):
        import cupy as cp

        bgra = cp.zeros((720, 1280, 4), dtype=cp.uint8)
        bgra[..., 2] = self.capture % 256
        bgra[..., 1] = 0 if view == "LEFT" else 255
        # The SDK has finished writing when it returns.
        cp.cuda.Stream.null.synchronize()
        mat.array = bgra
        return "SUCCESS"

    def get_timestamp(self, _reference):
        self.timestamp_reads += 1
        stamp = FIRST_CAPTURE_NS + self.capture * CAPTURE_PERIOD_NS
        return types.SimpleNamespace(get_nanoseconds=lambda: stamp)

    def get_sensors_data(self, data, _reference):
        self.sensor_reads += 1
        data.count = self.sensor_reads
        return "SUCCESS"

    def set_camera_settings(self, setting, value):
        self.settings.append((setting, value))
        return "SUCCESS"

    def get_current_fps(self):
        return 60.0


class _SensorsData:
    count = 0

    def get_imu_data(self):
        count = self.count
        return types.SimpleNamespace(
            timestamp=types.SimpleNamespace(
                get_nanoseconds=lambda: FIRST_CAPTURE_NS + count * 2_500_000
            ),
            get_linear_acceleration=lambda: (0.0, 9.81, float(count)),
        )


class _InitParameters:
    def __init__(self):
        self.input = types.SimpleNamespace(setFromCameraID=lambda *a: None)

    def set_from_serial_number(self, *_a):
        pass


@pytest.fixture
def zed_sdk(monkeypatch):
    """The stand-in SDK, in place of ``pyzed.sl``. Yields the cameras the
    source opens."""
    sl = types.ModuleType("pyzed.sl")
    sl.Camera, sl.Mat, sl.SensorsData = _Camera, _Mat, _SensorsData
    sl.InitParameters = _InitParameters
    sl.RuntimeParameters = types.SimpleNamespace
    for enum in (
        "RESOLUTION",
        "DEPTH_MODE",
        "BUS_TYPE",
        "ERROR_CODE",
        "VIEW",
        "MEM",
        "TIME_REFERENCE",
        "VIDEO_SETTINGS",
    ):
        setattr(sl, enum, _Enum())
    pyzed = types.ModuleType("pyzed")
    pyzed.sl = sl
    monkeypatch.setitem(sys.modules, "pyzed", pyzed)
    monkeypatch.setitem(sys.modules, "pyzed.sl", sl)
    monkeypatch.setattr(_Camera, "opened", [])
    monkeypatch.setattr(_Camera, "captures", [0])
    monkeypatch.setattr(_Camera, "grabs_before_unplug", {})
    return _Camera.opened


def _capture_of(frame):
    """Which capture each eye of ``frame`` shows, and each eye's green."""
    left, right = frame.image, frame.image_right
    return (
        int(left[0, 0, 0]),
        int(right[0, 0, 0]),
        int(left[0, 0, 1]),
        int(right[0, 0, 1]),
    )


def _read(source, count, timeout_s=10.0, pause_s=0.001):
    frames = []
    deadline = time.monotonic() + timeout_s
    while len(frames) < count and time.monotonic() < deadline:
        frame = source.latest()
        if frame is None:
            time.sleep(pause_s)
            continue
        frames.append((frame.timestamp_ns, *_capture_of(frame)))
    assert len(frames) == count, f"only {len(frames)} of {count} frames arrived"
    return frames


@requires_gpu
def test_every_frame_holds_both_eyes_of_one_capture_with_its_timestamp(zed_sdk):
    source = build_stereo_camera(ZED)
    assert (source.spec.name, source.spec.width, source.spec.height) == (
        "head",
        1280,
        720,
    )
    assert source.calibration() is None  # the camera is not open yet
    source.start()
    try:
        frames = _read(source, 12)
        calibration = source.calibration()
    finally:
        source.stop()
    captures = []
    for stamp, left, right, left_green, right_green in frames:
        assert left == right, "the two eyes are of different captures"
        assert (left_green, right_green) == (0, 255)  # left is left, right is right
        captures.append(round((stamp - FIRST_CAPTURE_NS) / CAPTURE_PERIOD_NS))
        assert captures[-1] % 256 == left, "the timestamp is another capture's"
    assert captures == sorted(set(captures)), "a capture was repeated or out of order"
    assert calibration.rectified and calibration.T[0] == pytest.approx(-0.063)
    assert zed_sdk[0].closed.is_set()


@requires_gpu
def test_a_reader_that_falls_behind_gets_the_newest_capture_and_a_count(zed_sdk):
    source = build_stereo_camera(ZED)
    source.start()
    try:
        (first,) = _read(source, 1)
        time.sleep(8 * CAPTURE_PERIOD_S)
        (second,) = _read(source, 1)
        dropped = source.dropped_captures
    finally:
        source.stop()
    skipped = round((second[0] - first[0]) / CAPTURE_PERIOD_NS) - 1
    assert skipped >= 3
    assert dropped == skipped


@requires_gpu
def test_the_imu_is_read_only_when_asked_for(zed_sdk):
    quiet = build_stereo_camera(ZED)
    quiet.start()
    try:
        _read(quiet, 3)
        assert quiet.imu() is None
    finally:
        quiet.stop()
    assert zed_sdk[0].sensor_reads == 0

    source = build_stereo_camera(ZED, imu=True)
    source.start()
    try:
        _read(source, 3)
        samples = source.imu()
        again = source.imu()
    finally:
        source.stop()
    assert samples.timestamps_ns.dtype == np.int64 and samples.gyro is None
    assert samples.accel.shape == (len(samples.timestamps_ns), 3)
    assert (np.diff(samples.timestamps_ns) > 0).all()
    np.testing.assert_array_equal(samples.accel[:, 1], 9.81)
    # Each sample is handed out once.
    assert again is None or again.timestamps_ns[0] > samples.timestamps_ns[-1]


@requires_gpu
def test_listed_settings_are_written_at_open_and_others_are_left_alone(zed_sdk):
    entry = {**ZED, "exposure": 40, "white_balance": "auto", "sharpness": 3}
    for build in (
        lambda: build_stereo_camera(entry),
        lambda: build_local_camera(entry)[0],
    ):
        source = build()
        source.start()
        try:
            _read_any(source)
        finally:
            source.stop()
        assert zed_sdk[-1].settings == [
            ("AEC_AGC", 0),
            ("EXPOSURE", 40),
            ("WHITEBALANCE_AUTO", 1),
            ("SHARPNESS", 3),
        ]

    untouched = build_stereo_camera(ZED)
    untouched.start()
    try:
        _read(untouched, 1)
    finally:
        untouched.stop()
    assert zed_sdk[-1].settings == []


def _read_any(source, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = source.latest()
        if frame is not None:
            return frame
        time.sleep(0.001)
    raise AssertionError("no frame arrived")


@requires_gpu
def test_a_viewing_source_reads_neither_timestamps_nor_the_imu(zed_sdk):
    (paired,) = build_local_camera(ZED)
    paired.start()
    try:
        before = time.monotonic_ns()
        frame = _read_any(paired)
        after = time.monotonic_ns()
    finally:
        paired.stop()
    assert frame.image_right is not None
    assert before <= frame.timestamp_ns <= after  # stamped when read, as before
    assert zed_sdk[0].timestamp_reads == 0 and zed_sdk[0].sensor_reads == 0


@requires_gpu
def test_a_frame_stays_intact_until_the_reader_asks_for_the_next(zed_sdk):
    source = build_stereo_camera(ZED)
    source.start()
    try:
        held = _read_any(source)
        shown = _capture_of(held)
        time.sleep(8 * CAPTURE_PERIOD_S)  # the camera writes many captures
        assert _capture_of(held) == shown, "the held frame was overwritten"
        later = _read_any(source)
        assert _capture_of(later)[0] != shown[0]
        assert _capture_of(later)[0] == _capture_of(later)[1]
    finally:
        source.stop()


@requires_gpu
def test_a_reconnected_camera_carries_on_with_new_captures_and_its_imu(
    zed_sdk, monkeypatch
):
    monkeypatch.setattr(zed_source, "RECONNECT_DELAY_S", 0.01)
    monkeypatch.setattr(_Camera, "grabs_before_unplug", {0: 5})
    source = build_stereo_camera(ZED, imu=True)
    source.start()
    try:
        frames = _read(source, 12)
        assert source.imu() is not None
    finally:
        source.stop()
    assert len(zed_sdk) == 2 and zed_sdk[0].closed.is_set()
    assert zed_sdk[1].sensor_reads > 0, "the second camera's IMU was not read"
    stamps = [frame[0] for frame in frames]
    assert stamps == sorted(set(stamps)), "a capture was repeated or out of order"
    for _stamp, left, right, _lg, _rg in frames:
        assert left == right


@requires_gpu
def test_a_restarted_source_hands_out_nothing_of_its_last_session(zed_sdk):
    source = build_stereo_camera(ZED)
    source.start()
    try:
        _read(source, 2)
        time.sleep(5 * CAPTURE_PERIOD_S)  # captures arrive and stay unread
    finally:
        source.stop()
    last_session = _Camera.captures[0]
    source.start()
    try:
        (frame,) = _read(source, 1)
    finally:
        source.stop()
    assert frame[1] > last_session, "a capture of the last session was handed out"


@requires_gpu
def test_a_camera_without_an_imu_is_not_polled_for_one(zed_sdk, monkeypatch):
    monkeypatch.setattr(
        _Camera,
        "information",
        staticmethod(lambda: _camera_information(63.0, with_imu=False)),
    )
    source = build_stereo_camera(ZED, imu=True)
    source.start()
    try:
        _read(source, 3)
        assert source.imu() is None
    finally:
        source.stop()
    assert zed_sdk[0].sensor_reads == 0


@requires_gpu
def test_a_reader_behind_on_the_imu_keeps_the_newest_samples(zed_sdk, monkeypatch):
    monkeypatch.setattr(zed_source, "_IMU_BACKLOG", 8)
    source = build_stereo_camera(ZED, imu=True)
    source.start()
    try:
        _read(source, 1)
        time.sleep(0.1)
        reads = zed_sdk[0].sensor_reads
        samples = source.imu()
    finally:
        source.stop()
    assert len(samples.timestamps_ns) == 8
    newest = (int(samples.timestamps_ns[-1]) - FIRST_CAPTURE_NS) // 2_500_000
    assert newest >= reads - 2, "the newest samples were the ones dropped"


@requires_gpu
def test_a_viewing_source_streams_from_a_camera_whose_calibration_is_unreadable(
    zed_sdk, monkeypatch
):
    def without_calibration():
        info = _camera_information(63.0)
        del info.camera_configuration.calibration_parameters
        return info

    monkeypatch.setattr(_Camera, "information", staticmethod(without_calibration))
    (paired,) = build_local_camera(ZED)
    paired.start()
    try:
        assert _read_any(paired).image_right is not None
    finally:
        paired.stop()


# ── the capture mailbox (no GPU needed) ─────────────────────────────


def test_the_mailbox_hands_out_the_newest_capture_once_and_counts_the_rest():
    box = CaptureMailbox()
    assert box.take() is None
    for stamp in (10, 20, 30):
        box.publish(box.write_index(), stamp)
    _index, stamp = box.take()
    assert stamp == 30 and box.take() is None
    assert box.dropped == 0  # nothing was read before: nothing was missed
    for stamp in (40, 50, 60):
        box.publish(box.write_index(), stamp)
    assert box.take()[1] == 60 and box.dropped == 2


def test_the_mailbox_never_writes_the_buffer_the_reader_holds():
    box = CaptureMailbox()
    box.publish(box.write_index(), 1)
    held, _ = box.take()
    for stamp in range(2, 40):
        writing = box.write_index()
        assert writing != held, "the producer was sent to the held buffer"
        box.publish(writing, stamp)
        # Nor the newest one, which the reader may take at any moment.
        assert box.write_index() != writing
    newest, _ = box.take()
    assert newest != held


def test_a_reset_mailbox_forgets_the_capture_but_not_the_buffer_that_is_held():
    box = CaptureMailbox()
    box.publish(box.write_index(), 1)
    held, _ = box.take()
    box.publish(box.write_index(), 2)
    box.reset()
    assert box.take() is None, "a capture of the last session was handed out"
    for stamp in (3, 4, 5):
        writing = box.write_index()
        assert writing != held
        box.publish(writing, stamp)
    assert box.take()[1] == 5


def test_the_mailbox_needs_a_buffer_to_publish_one_to_hold_and_one_to_write():
    with pytest.raises(ValueError, match=r"at least three buffers"):
        CaptureMailbox(2)
