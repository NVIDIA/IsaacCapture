# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LinkStereoCamera: stereo pairs from the camera connection as APAI camera frames.

Depth comes from a stand-in on the CPU, except in the tests marked "experiment",
which run the experiment's StereoDepth with its OpenCV matcher on a recorded
raw pair. No GPU and no camera.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from conftest import import_experiment
from link_bench import FakeThor, robot_state, until

from isaaccapture_examples.agentic_remote.agent.camera import stereo_camera
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import (
    CameraError,
    RecordedPairSource,
    StereoPair,
)
from isaaccapture_examples.agentic_remote.agent.camera.stereo_camera import (
    CAMERA_TYPE,
    CLOCK_STEP_QUIET_S,
    FUTURE_LIMIT_S,
    STATE_MAX_AGE_S,
    FrameStamp,
    calibration_id,
    register_camera,
    stamp_pair,
)
from isaaccapture_examples.agentic_remote.agent.checkahead import FrameTimes
from isaaccapture_examples.agentic_remote.agent.link_client import (
    AgentLink,
    StaleState,
)

# The Thor's wall clock minus its monotonic clock in these tests.
SKEW = 1.7e9
# The Thor's monotonic clock minus the workstation's in these tests.
AHEAD = 500.0
MARGIN = 0.02
# A wall time near 1.7e9 s is a float with steps of 0.24 microseconds.
CLOSE = 2e-6
WIDTH, HEIGHT = 64, 48
RAW_PAIR = ".state/thor/captures/20261106T212519Z"
STEREO_CALIBRATION = (
    ".state/thor/handeye/20261107T005443Z-live-2_v51bnx/stereo-calibration"
)
HAND_EYE = ".state/thor/mounting/20261107T005443Z/handeye.json"
CALIBRATION_ID = "stereo-c50bf5b2b899+mount-41d3ce701628"


class Now:
    """The workstation's monotonic clock, moved by the test."""

    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class StubLink:
    """The five clock calls of AgentLink, on numbers the test sets."""

    def __init__(self, now: Callable[[], float]) -> None:
        self.now = now
        self.has_state = True
        self.state_age_s = 0.01
        self.step_at: float | None = None
        self.ahead = AHEAD

    def state_arrived_s(self) -> float:
        self._need_state()
        return self.now() - self.state_age_s

    def clock_step_at_s(self) -> float | None:
        return self.step_at

    def thor_now(self) -> tuple[float, float]:
        self._need_state()
        mono = self.now() + self.ahead
        return mono, mono + SKEW

    def wall_age_s(self, thor_wall_s: float) -> float:
        return self.thor_now()[1] - thor_wall_s + MARGIN

    def wall_to_thor_mono(self, thor_wall_s: float) -> float:
        self._need_state()
        return thor_wall_s - SKEW

    def _need_state(self) -> None:
        if not self.has_state:
            raise StaleState("no ROBOT_STATE has arrived yet")


def nv12(shade: int = 90) -> np.ndarray:
    image = np.full((HEIGHT * 3 // 2, WIDTH), 128, dtype=np.uint8)
    image[:HEIGHT] = shade
    return image


def a_pair(
    link: Any,
    seq: int = 1,
    *,
    age_s: float = 0.08,
    arrived_ago_s: float = 0.03,
    session_id: int = 1,
    devices: tuple[int, int] = (5, 7),
    rotation: int = 0,
    skew_ns: int = 1_000_000,
) -> StereoPair:
    """A pair captured age_s ago by the Thor's wall clock (as the link has it)."""
    sof_ns = round((link.thor_now()[1] - age_s) * 1e9)
    return StereoPair(
        session_id=session_id,
        seq=seq,
        left=nv12(90),
        right=nv12(60),
        sof_wall_ns=(sof_ns + skew_ns, sof_ns),
        device_index=devices,
        rotation=rotation,
        receiver_rx_mono_s=link.now() - arrived_ago_s,
        check={"source": "test", "confirmed_by_seq": seq + 1},
    )


@pytest.fixture
def now() -> Now:
    return Now()


@pytest.fixture
def link(now: Now) -> StubLink:
    return StubLink(now)


# The times of a camera frame; neither APAI nor the experiment is needed


def test_a_camera_frame_is_dated_at_its_capture_with_the_clock_margin(
    link: StubLink, now: Now
) -> None:
    pair = a_pair(link, age_s=0.08, arrived_ago_s=0.03)
    stamp = stamp_pair(link, pair, now)
    assert isinstance(stamp, FrameStamp) and type(stamp.t) is float
    # The earlier eye, untouched: this is what a PERMIT quotes.
    assert stamp.evidence_wall_s == min(pair.sof_wall_ns) / 1e9
    assert stamp.capture_thor_mono_s == pytest.approx(now.t + AHEAD - 0.08, abs=CLOSE)
    assert stamp.raw_age_s == pytest.approx(0.08, abs=CLOSE)
    # 0.08 s of age plus the margin: the age errs on the old side.
    assert stamp.t == pytest.approx(now.t - 0.08 - MARGIN, abs=CLOSE)
    assert stamp.t < pair.receiver_rx_mono_s


def test_a_camera_frame_is_never_dated_later_than_it_arrived(
    link: StubLink, now: Now
) -> None:
    # The clock estimate says 10 ms old; the receiver saw it arrive 30 ms ago.
    pair = a_pair(link, age_s=-0.01, arrived_ago_s=0.03)
    stamp = stamp_pair(link, pair, now)
    assert stamp.t == pair.receiver_rx_mono_s == now.t - 0.03
    assert stamp.raw_age_s == pytest.approx(-0.01, abs=CLOSE)


def undated(stamp: FrameStamp, pair: StereoPair, says: str) -> bool:
    """The stamp carries no evidence time, says why, and is the pair's arrival."""
    assert stamp.evidence_wall_s is None and stamp.capture_thor_mono_s is None
    assert stamp.t == pair.receiver_rx_mono_s and type(stamp.t) is float
    assert says in stamp.problem, stamp.problem
    return True


def test_without_a_robot_state_a_camera_frame_is_dated_at_its_arrival_only(
    link: StubLink, now: Now
) -> None:
    pair = a_pair(link)
    link.has_state = False
    stamp = stamp_pair(link, pair, now)
    assert undated(stamp, pair, "no ROBOT_STATE has arrived yet")
    assert stamp.raw_age_s is None


def test_a_robot_state_older_than_one_second_gives_no_evidence_time(
    link: StubLink, now: Now
) -> None:
    assert STATE_MAX_AGE_S == 1.0
    pair = a_pair(link)
    link.state_age_s = 0.99
    assert stamp_pair(link, pair, now).problem == ""
    link.state_age_s = 1.01
    stamp = stamp_pair(link, pair, now)
    assert undated(stamp, pair, "newest ROBOT_STATE is 1.01 s old")
    # What the two clocks said is kept for the record.
    assert stamp.raw_age_s == pytest.approx(0.08, abs=CLOSE)


def test_a_capture_time_in_the_future_of_the_thors_clock_gives_no_evidence_time(
    link: StubLink, now: Now
) -> None:
    assert FUTURE_LIMIT_S == 0.05
    near = stamp_pair(link, a_pair(link, age_s=-0.049, arrived_ago_s=0.0), now)
    assert near.problem == "" and near.evidence_wall_s is not None
    pair = a_pair(link, age_s=-0.051)
    assert undated(stamp_pair(link, pair, now), pair, "0.051 s in the future")
    pair = a_pair(link, age_s=-3600.0)
    assert undated(stamp_pair(link, pair, now), pair, "in the future")


@pytest.mark.parametrize("arrived_ago_s", [0.5, 2.5])
def test_a_capture_time_after_the_pairs_arrival_gives_no_evidence_time(
    link: StubLink, now: Now, arrived_ago_s: float
) -> None:
    # Exposed 10 ms ago by the Thor's clock, yet here since arrived_ago_s.
    pair = a_pair(link, age_s=0.010, arrived_ago_s=arrived_ago_s)
    stamp = stamp_pair(link, pair, now)
    assert undated(stamp, pair, "s after the arrival, over 0.07 s")
    # The limit is the future limit plus the clock margin.
    edge = FUTURE_LIMIT_S + MARGIN + 0.010
    inside = a_pair(link, age_s=0.010, arrived_ago_s=edge - 0.001)
    assert stamp_pair(link, inside, now).problem == ""
    outside = a_pair(link, age_s=0.010, arrived_ago_s=edge + 0.001)
    assert undated(stamp_pair(link, outside, now), outside, "after the arrival")


def test_the_evidence_time_is_never_newer_than_the_arrival(
    link: StubLink, now: Now
) -> None:
    # Within the limit: the capture time lies 40 ms after the arrival.
    pair = a_pair(link, age_s=0.010, arrived_ago_s=0.050)
    stamp = stamp_pair(link, pair, now)
    assert stamp.t == pair.receiver_rx_mono_s
    arrival_wall = link.thor_now()[1] - 0.050
    assert stamp.evidence_wall_s == pytest.approx(arrival_wall, abs=CLOSE)
    assert stamp.evidence_wall_s < min(pair.sof_wall_ns) / 1e9
    assert stamp.capture_thor_mono_s == pytest.approx(now.t + AHEAD - 0.050, abs=CLOSE)
    # What the two clocks said is kept as it was measured.
    assert stamp.raw_age_s == pytest.approx(0.010, abs=CLOSE)


def test_a_capture_time_long_before_the_arrival_is_the_camera_frames_time(
    link: StubLink, now: Now
) -> None:
    """A late camera link is no refusal of its own: the frame is dated that old."""
    assert not hasattr(stereo_camera, "BEFORE_ARRIVAL_LIMIT_S")
    # Captured 2.5 s ago by the Thor's clock; the pair arrived 0.03 s ago.
    pair = a_pair(link, age_s=2.5, arrived_ago_s=0.03)
    stamp = stamp_pair(link, pair, now)
    assert stamp.problem == "" and stamp.evidence_wall_s == min(pair.sof_wall_ns) / 1e9
    assert stamp.t == pytest.approx(now.t - 2.5 - MARGIN, abs=CLOSE)


def test_for_three_seconds_after_a_clock_step_there_is_no_evidence_time(
    link: StubLink, now: Now
) -> None:
    assert CLOCK_STEP_QUIET_S == 3.0
    link.step_at = now.t - 2.9
    pair = a_pair(link)
    assert undated(stamp_pair(link, pair, now), pair, "wall clock stepped 2.90 s ago")
    link.step_at = now.t - 3.0
    assert stamp_pair(link, a_pair(link), now).problem == ""


def test_an_arrival_stamp_that_is_not_a_past_time_is_refused(
    link: StubLink, now: Now
) -> None:
    for arrival in (now.t + 0.5, math.nan, math.inf):
        pair = dataclasses.replace(a_pair(link), receiver_rx_mono_s=arrival)
        # With and without the robot link's clocks.
        for has_state in (True, False):
            link.has_state = has_state
            with pytest.raises(CameraError, match="arrival stamp"):
                stamp_pair(link, pair, now)
        link.has_state = True


def a_state(seq: int, mono: float, skew: float = SKEW) -> Any:
    return dataclasses.replace(
        robot_state(seq=seq), clock_mono_s=mono, clock_wall_s=mono + skew
    )


def test_the_times_come_from_the_agent_links_clock_calls() -> None:
    """The same rules on a real AgentLink with a scripted robot side."""
    now = Now(500.0)
    thor = FakeThor("test")
    link = AgentLink(thor.service_end, monotonic=now)

    def arrive(state: Any, at: float) -> None:
        now.t = at
        before = link._states
        thor.send(state)
        until(lambda: link._states > before, "the ROBOT_STATE")

    def pair_at(wall_s: float, arrival: float, seq: int = 1) -> StereoPair:
        sof_ns = round(wall_s * 1e9)
        return StereoPair(
            1, seq, nv12(), nv12(), (sof_ns, sof_ns), (5, 7), 0, arrival, {}
        )

    try:
        first = pair_at(SKEW + 1000.0, 499.0)
        assert undated(stamp_pair(link, first, now), first, "no ROBOT_STATE")
        # The Thor read its clocks at 1000.0 s; the state arrived at 500.0 here.
        arrive(a_state(1, 1000.0), at=500.0)
        now.t = 500.1  # the Thor's monotonic clock is estimated at 1000.1 s
        stamp = stamp_pair(link, pair_at(SKEW + 1000.0, 500.05), now)
        assert stamp.capture_thor_mono_s == pytest.approx(1000.0, abs=CLOSE)
        assert stamp.raw_age_s == pytest.approx(0.1, abs=CLOSE)
        assert stamp.t == pytest.approx(500.0 - MARGIN, abs=CLOSE)
        now.t = 501.2
        late = pair_at(SKEW + 1001.1, 501.15)
        assert undated(stamp_pair(link, late, now), late, "ROBOT_STATE is 1.20 s old")
        # The Thor's wall clock jumps by 5 s against its monotonic clock.
        arrive(a_state(2, 1001.2, SKEW + 5.0), at=501.2)
        assert link.clock_step_at_s() == 501.2
        stepped = pair_at(SKEW + 5.0 + 1001.1, 501.15)
        assert undated(stamp_pair(link, stepped, now), stepped, "wall clock stepped")
        arrive(a_state(3, 1004.3, SKEW + 5.0), at=504.3)
        stamp = stamp_pair(link, pair_at(SKEW + 5.0 + 1004.2, 504.25), now)
        assert stamp.capture_thor_mono_s == pytest.approx(1004.2, abs=CLOSE)
        assert stamp.t == pytest.approx(504.2 - MARGIN, abs=CLOSE)
    finally:
        link.close()
        thor.carrier.close()


def calibration_files(folder: Path) -> tuple[Path, Path]:
    directory = folder / "stereo-calibration"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({"sha256": {}}))
    hand_eye = folder / "handeye.json"
    hand_eye.write_text(json.dumps({"T_base_cam": "test"}))
    return directory, hand_eye


def test_the_calibration_id_names_both_files_by_their_hashes(tmp_path: Path) -> None:
    directory, hand_eye = calibration_files(tmp_path)
    stereo = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    mount = hashlib.sha256(hand_eye.read_bytes()).hexdigest()
    assert calibration_id(directory, hand_eye) == (
        f"stereo-{stereo[:12]}+mount-{mount[:12]}"
    )
    assert calibration_id(str(directory), str(hand_eye)) == (
        calibration_id(directory, hand_eye)
    )
    hand_eye.write_text(json.dumps({"T_base_cam": "moved"}))
    assert calibration_id(directory, hand_eye).startswith(f"stereo-{stereo[:12]}+")
    assert not calibration_id(directory, hand_eye).endswith(mount[:12])
    with pytest.raises(CameraError, match="cannot read the calibration file"):
        calibration_id(directory, tmp_path / "absent.json")
    with pytest.raises(CameraError, match=r"manifest\.json"):
        calibration_id(tmp_path / "absent", hand_eye)


# The camera class, with APAI and a stand-in for depth


class FakeDepth:
    """In place of the experiment's StereoDepth: a flat scene 1.5 m away."""

    made: ClassVar[list[FakeDepth]] = []

    def __init__(self, directory: Path, backend: str, options: Any) -> None:
        self.directory, self.backend, self.options = directory, backend, options
        self.K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])
        self.manifest = {
            "status": "test_calibration",
            "raw_rotation_degrees": 0,
            "camera_order": {
                "left": {"argus_device": 5},
                "right": {"argus_device": 7},
            },
        }
        self.matcher = types.SimpleNamespace(
            metadata={"backend": backend}, last_timings_ms={"total": 1.5}
        )
        self.images: list[np.ndarray] = []
        self.during: Callable[[], None] | None = None
        self.closed = 0
        FakeDepth.made.append(self)

    def compute(self, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        self.images = images
        if self.during is not None:
            self.during()
        depth = np.full(images[0].shape[:2], 1.5, dtype=np.float32)
        depth[:, : WIDTH // 4] = 0  # a quarter of the image has no depth
        return images[0], depth

    def close(self) -> None:
        self.closed += 1


class FakeSource:
    """A PairSource that hands out what the test scripted, in order."""

    def __init__(self, *items: Any) -> None:
        self.items = list(items)
        self.opened = self.closed = 0
        self.timeouts: list[float] = []
        self.last_none = ""
        self.open_error: Exception | None = None

    def open(self) -> None:
        if self.open_error is not None:
            raise self.open_error
        self.opened += 1

    def close(self) -> None:
        self.closed += 1

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        self.timeouts.append(timeout_s)
        if not self.items:
            self.last_none = "nothing scripted"
            return None
        item = self.items.pop(0)
        if callable(item):
            item = item()
        if isinstance(item, Exception):
            raise item
        if item is None:
            self.last_none = "scripted none"
        return item


class Bench:
    """One LinkStereoCamera on a stub link, a scripted source and FakeDepth."""

    def __init__(self, folder: Path, link: StubLink, now: Now, **profile: Any) -> None:
        from apai.config import Cfg

        directory, hand_eye = calibration_files(folder)
        self.calibration_id = calibration_id(directory, hand_eye)
        self.link, self.now = link, now
        self.frame_times = FrameTimes()
        self.source = FakeSource()
        self.profile = {
            "type": CAMERA_TYPE,
            "name": "head",
            "width": WIDTH,
            "height": HEIGHT,
            "fps": 7,
            "calibration_dir": str(directory),
            "extrinsics": {"mode": "eye_to_hand", "hand_eye_json": str(hand_eye)},
            "depth_backend": "stand_in",
            "ffs": {"iterations": 4},
            "frame_timeout_s": 0.15,
            "max_frame_age_s": 3,
            **profile,
        }
        FakeDepth.made.clear()
        self.camera = stereo_camera.LinkStereoCamera(
            Cfg(self.profile),
            link=link,
            frame_times=self.frame_times,
            source=self.source,
            depth=FakeDepth,
            monotonic=now,
        )

    @property
    def depth(self) -> FakeDepth:
        return FakeDepth.made[-1]

    def pair(self, seq: int, **fields: Any) -> StereoPair:
        return a_pair(self.link, seq, **fields)


@pytest.fixture
def bench(tmp_path: Path, link: StubLink, now: Now) -> Iterator[Bench]:
    made = Bench(tmp_path, link, now)
    made.camera.open()
    yield made
    made.camera.close()


@pytest.mark.apai
def test_the_camera_is_an_apai_camera_that_fails_at_the_first_refusal() -> None:
    from apai.perception.camera_base import CameraBase

    camera = stereo_camera.LinkStereoCamera
    assert issubclass(camera, CameraBase) and not camera.__abstractmethods__
    assert camera.MAX_CONSECUTIVE_FAILURES == 1
    assert camera is stereo_camera.LinkStereoCamera  # made once
    assert camera.__module__ == stereo_camera.__name__
    with pytest.raises(AttributeError):
        stereo_camera.NoSuchName  # noqa: B018


@pytest.mark.apai
def test_a_pair_becomes_a_camera_frame_with_depth_times_and_its_calibration(
    bench: Bench, now: Now
) -> None:
    from apai.perception import freshness
    from apai.types import Frame

    pair = bench.pair(41, age_s=0.08, arrived_ago_s=0.03)
    bench.source.items.append(pair)
    frame = bench.camera.get_frame()

    assert isinstance(frame, Frame) and frame.frame_id == 1
    assert bench.camera.has_depth and frame.has_depth
    assert frame.rgb.shape == (HEIGHT, WIDTH, 3) and frame.rgb.dtype == np.uint8
    assert frame.rgb is bench.depth.images[0]  # the left image, as depth gave it
    assert frame.depth_m.shape == (HEIGHT, WIDTH)
    assert frame.depth_m.dtype == np.float32
    assert frame.depth_source == "sensor"
    assert np.array_equal(frame.K, bench.depth.K) and frame.K is not bench.depth.K
    # Both eyes reached depth as BGR images, left first.
    left, right = bench.depth.images
    assert int(left[0, 0, 1]) > int(right[0, 0, 1])

    # The times.
    evidence = min(pair.sof_wall_ns) / 1e9
    assert type(frame.t) is float and frame.observed_t == frame.t
    assert frame.t == pytest.approx(now.t - 0.08 - MARGIN, abs=CLOSE)
    capture = frame.capture
    assert capture["backend"] == "link_stereo"
    assert capture["evidence_wall_s"] == evidence
    assert capture["capture_thor_mono_s"] == pytest.approx(
        now.t + AHEAD - 0.08, abs=CLOSE
    )
    assert capture["time_problem"] is None and bench.camera.undated == 0
    assert capture["receipt_monotonic_s"] == pair.receiver_rx_mono_s
    assert capture["clock_raw_age_s"] == pytest.approx(0.08, abs=CLOSE)
    # What a permit looks up, by exactly the float APAI's grid keeps.
    assert bench.frame_times.lookup(frame.t) == evidence
    assert len(bench.frame_times) == 1

    assert (capture["session_id"], capture["seq"]) == (1, 41)
    assert capture["sensor_ids"] == [5, 7]
    assert capture["sensor_timestamps_ns"] == list(pair.sof_wall_ns)
    assert capture["check"] == {"source": "test", "confirmed_by_seq": 42}
    assert capture["invented_times"] is False
    assert capture["calibration_id"] == bench.calibration_id
    assert capture["calibration_status"] == "test_calibration"
    assert capture["depth_backend"] == "stand_in"
    assert capture["depth_parameters"] == {"backend": "stand_in"}
    assert capture["depth_timings_ms"] == {"total": 1.5}
    assert capture["valid_fraction"] == pytest.approx(0.75)
    json.dumps(capture)  # plain values only

    # For APAI it is a local camera frame, dated on this machine's clock.
    assert freshness.capture_marker(frame) == {
        "channel": "local_frame",
        "clock": "client_monotonic",
        "t": frame.t,
    }
    assert freshness.observed_at(frame) == frame.t


@pytest.mark.apai
def test_open_builds_depth_from_the_profile_and_close_ends_both(
    tmp_path: Path, link: StubLink, now: Now
) -> None:
    made = Bench(tmp_path, link, now)
    camera, source = made.camera, made.source
    with pytest.raises(CameraError, match="not open"):
        camera.get_frame()
    assert not FakeDepth.made and source.opened == 0
    camera.open()
    depth = made.depth
    assert depth.directory == Path(made.profile["calibration_dir"])
    assert depth.backend == "stand_in"
    assert depth.options.as_dict() == {"iterations": 4}
    assert camera.calibration_id == made.calibration_id
    assert source.opened == 1
    with pytest.raises(CameraError, match="already open"):
        camera.open()
    camera.close()
    assert (source.closed, depth.closed) == (1, 1)
    camera.close()  # twice is fine
    assert (source.closed, depth.closed) == (1, 1)
    with pytest.raises(CameraError, match="not open"):
        camera.get_frame()
    camera.open()  # and it opens again
    assert source.opened == 2 and made.depth is not depth
    camera.close()


@pytest.mark.apai
def test_a_source_that_cannot_open_leaves_nothing_open(
    tmp_path: Path, link: StubLink, now: Now
) -> None:
    made = Bench(tmp_path, link, now)
    made.source.open_error = CameraError("cannot start the camera frame bridge")
    with pytest.raises(CameraError, match="cannot start the camera frame bridge"):
        made.camera.open()
    assert made.depth.closed == 1
    with pytest.raises(CameraError, match="not open"):
        made.camera.get_frame()


@pytest.mark.apai
@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"calibration_dir": ""}, "needs calibration_dir and extrinsics"),
        ({"extrinsics": {"mode": "eye_to_hand"}}, "needs calibration_dir and"),
        ({"calibration_dir": "/no/such/folder"}, "cannot read the calibration file"),
    ],
)
def test_a_profile_without_its_calibration_files_is_refused_at_open(
    tmp_path: Path, link: StubLink, now: Now, change: dict, text: str
) -> None:
    made = Bench(tmp_path, link, now, **change)
    with pytest.raises(CameraError, match=text):
        made.camera.open()
    assert not FakeDepth.made and made.source.opened == 0


@pytest.mark.apai
def test_without_the_experiment_folder_the_default_depth_is_refused(
    tmp_path: Path, link: StubLink, now: Now, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apai.config import Cfg

    monkeypatch.delenv("AGENTIC_REMOTE_EXPERIMENT_DIR", raising=False)
    made = Bench(tmp_path, link, now)
    camera = stereo_camera.LinkStereoCamera(
        Cfg(made.profile), link=link, frame_times=FrameTimes(), source=made.source
    )
    with pytest.raises(CameraError, match="AGENTIC_REMOTE_EXPERIMENT_DIR is not set"):
        camera.open()
    assert made.source.opened == 0


@pytest.mark.apai
def test_each_grab_gives_a_newer_camera_frame(bench: Bench, now: Now) -> None:
    frames = []
    for seq in (1, 2, 3):
        bench.source.items.append(bench.pair(seq))
        frames.append(bench.camera.get_frame())
        now.t += 0.14
    assert [frame.frame_id for frame in frames] == [1, 2, 3]
    assert frames[0].t < frames[1].t < frames[2].t
    assert [frame.capture["seq"] for frame in frames] == [1, 2, 3]
    assert len(bench.frame_times) == 3 and bench.camera.delivered == 3


@pytest.mark.apai
def test_the_same_pair_twice_is_refused(bench: Bench, now: Now) -> None:
    pair = bench.pair(7)
    bench.source.items += [pair, pair]
    bench.camera.get_frame()
    now.t += 0.14
    with pytest.raises(CameraError, match=r"stereo pair \(1, 7\) twice"):
        bench.camera.get_frame()
    assert len(bench.frame_times) == 1


@pytest.mark.apai
def test_a_capture_time_that_does_not_grow_gives_no_evidence_time(
    bench: Bench, now: Now
) -> None:
    first = bench.pair(7)
    bench.source.items.append(first)
    bench.camera.get_frame()
    now.t += 0.14
    # As after a step back of the Thor's wall clock: a later pair, the same time.
    same_time = dataclasses.replace(first, seq=8, receiver_rx_mono_s=now.t - 0.01)
    bench.source.items.append(same_time)
    frame = bench.camera.get_frame()
    assert frame.capture["seq"] == 8 and frame.t == same_time.receiver_rx_mono_s
    assert frame.capture["evidence_wall_s"] is None
    assert "capture time did not grow" in frame.capture["time_problem"]
    assert len(bench.frame_times) == 1 and bench.camera.undated == 1
    # The mark is the last dated camera frame's: an older capture is undated too.
    now.t += 0.14
    bench.source.items.append(bench.pair(9, age_s=0.5))
    assert bench.camera.get_frame().capture["evidence_wall_s"] is None
    # The camera frame's own time must grow, as on the Thor.
    bench.source.items.append(bench.pair(10, age_s=0.3, arrived_ago_s=0.2))
    with pytest.raises(CameraError, match="this machine's clock did not grow"):
        bench.camera.get_frame()
    # The next pair that is newer has an evidence time again.
    now.t += 0.14
    bench.source.items.append(bench.pair(11))
    frame = bench.camera.get_frame()
    assert frame.capture["seq"] == 11 and frame.capture["time_problem"] is None
    assert len(bench.frame_times) == 2 and bench.camera.undated == 2


@pytest.mark.apai
def test_a_new_camera_session_may_start_its_numbers_again(
    bench: Bench, now: Now
) -> None:
    bench.source.items.append(bench.pair(500, session_id=1))
    bench.camera.get_frame()
    now.t += 0.14
    bench.source.items.append(bench.pair(1, session_id=2))
    frame = bench.camera.get_frame()
    assert (frame.capture["session_id"], frame.capture["seq"]) == (2, 1)


@pytest.mark.apai
def test_a_new_session_whose_camera_frame_is_not_newer_is_refused(
    bench: Bench, now: Now
) -> None:
    bench.source.items.append(bench.pair(500, session_id=1, age_s=0.08))
    bench.camera.get_frame()
    now.t += 0.01
    bench.source.items.append(bench.pair(1, session_id=2, age_s=0.2))
    with pytest.raises(CameraError, match="this machine's clock did not grow"):
        bench.camera.get_frame()
    assert len(bench.frame_times) == 1


@pytest.mark.apai
def test_no_pair_in_time_is_a_camera_error_that_says_why(bench: Bench) -> None:
    bench.source.items += [None, None]
    began = time.monotonic()
    with pytest.raises(
        CameraError, match=r"no new stereo pair within 0\.15 s: nothing scripted"
    ):
        bench.camera.get_frame()
    waited = time.monotonic() - began
    assert 0.14 < waited < 2.0
    # The source is asked for no longer than the time that is left.
    assert bench.source.timeouts[0] <= 0.15
    assert all(
        later <= earlier
        for earlier, later in zip(bench.source.timeouts, bench.source.timeouts[1:])
    )
    assert len(bench.frame_times) == 0


@pytest.mark.apai
def test_a_dead_bridge_is_a_camera_error_at_once(bench: Bench) -> None:
    bench.source.items.append(CameraError("the camera frame bridge is gone: killed"))
    with pytest.raises(CameraError, match="camera frame bridge is gone: killed"):
        bench.camera.get_frame()
    assert len(bench.source.timeouts) == 1  # asked once, not until the timeout


@pytest.mark.apai
def test_a_camera_frame_the_link_cannot_date_is_handed_on_and_notes_nothing(
    bench: Bench, link: StubLink, now: Now, caplog: pytest.LogCaptureFixture
) -> None:
    """The robot link's trouble refuses no camera frame; it withholds the evidence."""
    caplog.set_level("INFO", logger=stereo_camera._log.name)
    troubles = (
        (lambda: setattr(link, "has_state", False), "no ROBOT_STATE has arrived"),
        (lambda: setattr(link, "state_age_s", 1.5), "ROBOT_STATE is 1.50 s old"),
        (lambda: setattr(link, "step_at", now.t - 1.0), "wall clock stepped"),
    )
    for seq, (trouble, says) in enumerate(troubles, start=1):
        link.has_state, link.state_age_s, link.step_at = True, 0.01, None
        pair = bench.pair(seq)
        trouble()
        bench.source.items.append(pair)
        frame = bench.camera.get_frame()
        capture = frame.capture
        assert frame.t == frame.observed_t == pair.receiver_rx_mono_s
        assert frame.t == capture["receipt_monotonic_s"]  # as on the Thor
        assert capture["evidence_wall_s"] is None
        assert capture["capture_thor_mono_s"] is None
        assert says in capture["time_problem"]
        assert frame.depth_m is not None and bench.depth.images
        json.dumps(capture)
        now.t += 0.14
    assert len(bench.frame_times) == 0
    assert (bench.camera.delivered, bench.camera.undated) == (3, 3)
    link.has_state, link.state_age_s, link.step_at = True, 0.01, None
    bench.source.items.append(bench.pair(4, age_s=-3600.0))
    assert "in the future" in bench.camera.get_frame().capture["time_problem"]
    now.t += 0.14
    # The next pair the link can date has its evidence time.
    bench.source.items.append(bench.pair(5))
    frame = bench.camera.get_frame()
    assert frame.capture["time_problem"] is None
    assert bench.frame_times.lookup(frame.t) == frame.capture["evidence_wall_s"]
    # One line when the evidence goes and one when it is back, not one per frame.
    lines = [record.getMessage() for record in caplog.records]
    assert sum("carry no evidence time" in line for line in lines) == 1
    assert sum("evidence time again" in line for line in lines) == 1


@pytest.mark.apai
def test_the_time_is_taken_again_after_depth(
    bench: Bench, link: StubLink, now: Now
) -> None:
    """A ROBOT_STATE that stops arriving while depth runs leaves no evidence time."""

    def states_stop() -> None:
        now.t += 0.4
        link.state_age_s = 1.2

    bench.depth.during = states_stop
    pair = bench.pair(1)
    bench.source.items.append(pair)
    frame = bench.camera.get_frame()
    assert "ROBOT_STATE is 1.20 s old" in frame.capture["time_problem"]
    assert frame.t == pair.receiver_rx_mono_s and len(bench.frame_times) == 0


@pytest.mark.apai
def test_a_pair_too_old_after_depth_is_dropped_for_the_next_one(
    bench: Bench, link: StubLink, now: Now
) -> None:
    def slow_depth() -> None:
        now.t += 3.0  # depth took three seconds this once
        bench.depth.during = None

    bench.depth.during = slow_depth
    bench.source.items.append(bench.pair(1, age_s=0.08, arrived_ago_s=0.03))
    bench.source.items.append(lambda: bench.pair(2))
    frame = bench.camera.get_frame()
    assert frame.capture["seq"] == 2
    assert bench.camera.dropped_old == 1 and bench.camera.delivered == 1
    assert len(bench.frame_times) == 1


@pytest.mark.apai
def test_a_pair_of_other_devices_or_rotation_than_the_calibration_is_refused(
    bench: Bench,
) -> None:
    bench.source.items.append(bench.pair(1, devices=(0, 1)))
    with pytest.raises(CameraError, match=r"devices \(0, 1\) rotated 0; .*\(5, 7\)"):
        bench.camera.get_frame()
    bench.source.items.append(bench.pair(2, devices=(7, 5)))
    with pytest.raises(CameraError, match="the calibration is for"):
        bench.camera.get_frame()
    bench.source.items.append(bench.pair(3, rotation=180))
    with pytest.raises(CameraError, match="rotated 180"):
        bench.camera.get_frame()
    assert bench.depth.images == []


@pytest.mark.apai
def test_apais_stream_gets_camera_frames_and_survives_a_refusal(
    bench: Bench, link: StubLink, now: Now, capsys: pytest.CaptureFixture
) -> None:
    from apai.perception.stream import CameraStream

    def next_pair() -> StereoPair:
        now.t += 0.05
        return bench.pair(int(now.t * 1000))

    def from_the_future() -> StereoPair:
        return dataclasses.replace(next_pair(), receiver_rx_mono_s=now.t + 5.0)

    # The first grab is refused; the stream tries again.
    bench.source.items += [from_the_future, *[next_pair] * 200]
    bench.camera.close()
    stream = CameraStream(bench.camera, name="head", rate_hz=100.0)
    stream.open()
    try:
        until(lambda: "grab failed" in capsys.readouterr().err, "the refusal")
        until(lambda: stream._latest_seq >= 3, "three camera frames")
        frame = stream._latest
    finally:
        stream.close()
    assert frame.capture["backend"] == "link_stereo"
    assert bench.frame_times.lookup(frame.t) == frame.capture["evidence_wall_s"]
    assert bench.source.closed >= 1 and bench.depth.closed == 1


@pytest.mark.apai
def test_the_factory_answers_link_stereo_and_leaves_the_other_types(
    tmp_path: Path, link: StubLink, now: Now, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apai.config import Cfg
    from apai.perception import camera_base
    from apai.perception.mock_camera import MockCamera

    original = camera_base.make_camera
    monkeypatch.setattr(camera_base, "make_camera", original)  # put back afterwards
    link = StubLink(time.monotonic)  # the factory builds cameras on the real clock
    made = Bench(tmp_path, link, now)
    frame_times = FrameTimes()
    register_camera(link, frame_times, made.source, depth=FakeDepth)
    wrapper = camera_base.make_camera
    assert wrapper is not original
    camera = wrapper(Cfg(made.profile))
    assert type(camera) is stereo_camera.LinkStereoCamera
    assert isinstance(wrapper(Cfg({"type": "mock"})), MockCamera)
    with pytest.raises(ValueError, match="unknown camera type"):
        wrapper(Cfg({"type": "no_such_camera"}))

    made.source.items.append(a_pair(link, 1))
    with camera:
        frame = camera.get_frame()
    assert frame_times.lookup(frame.t) == frame.capture["evidence_wall_s"]

    # Registering again wraps nothing twice; the next camera gets the new parts.
    other_times, other_source = FrameTimes(), FakeSource(a_pair(link, 2))
    register_camera(link, other_times, other_source, depth=FakeDepth)
    assert camera_base.make_camera is wrapper
    with wrapper(Cfg(made.profile)) as second:
        frame = second.get_frame()
    assert other_times.lookup(frame.t) is not None and len(frame_times) == 1
    assert other_source.opened == 1


@pytest.mark.apai
def test_a_profile_may_name_a_folder_of_recorded_pairs(
    tmp_path: Path, link: StubLink, now: Now
) -> None:
    import cv2
    from apai.config import Cfg

    folder = tmp_path / "pairs"
    folder.mkdir()
    image = np.zeros((1984, 2560, 3), dtype=np.uint8)
    image[:, 1280:] = 200
    for name in ("left_raw.jpg", "right_raw.jpg"):
        assert cv2.imwrite(str(folder / name), image)
    made = Bench(tmp_path, link, now, recorded_dir=str(folder))
    frame_times = FrameTimes()
    camera = stereo_camera.LinkStereoCamera(
        Cfg(made.profile),
        link=link,
        frame_times=frame_times,
        depth=FakeDepth,
        monotonic=now,
    )
    with camera:
        frame = camera.get_frame()
    assert frame.rgb.shape == (1984, 2560, 3)
    assert frame.capture["check"]["invented_times"] is True
    assert frame.capture["invented_times"] is True
    # The invented capture time is on the link's estimate of the Thor's wall clock.
    assert frame.capture["evidence_wall_s"] == pytest.approx(
        link.thor_now()[1] - 0.06, abs=CLOSE
    )
    assert frame_times.lookup(frame.t) == frame.capture["evidence_wall_s"]

    del made.profile["recorded_dir"]
    nothing = stereo_camera.LinkStereoCamera(
        Cfg(made.profile), link=link, frame_times=frame_times, depth=FakeDepth
    )
    with pytest.raises(CameraError, match="needs a stereo pair source"):
        nothing.open()
    assert FakeDepth.made[-1].closed == 1


@pytest.mark.apai
def test_apais_grid_keeps_the_noted_time_and_masks_at_the_capture_time(
    tmp_path: Path,
) -> None:
    """One camera frame through a required grid: what the permits and the mask need."""
    from apai.perception.occupancy import OccupancyMap
    from conftest import FakeGridClient
    from test_self_mask import FakeHistory, read
    from test_self_mask import a_state as joints_state

    from isaaccapture_examples.agentic_remote.agent import self_mask

    link = StubLink(time.monotonic)
    made = Bench(tmp_path, link, time.monotonic)
    feed = self_mask.JointFeed(1, history=FakeHistory(1))
    poses: list[Any] = []

    def never() -> Any:
        raise AssertionError("the newest pose was asked for")

    def mask(depth: np.ndarray, intrinsics: Any, camera_pose: Any) -> np.ndarray:
        poses.append((self_mask.FRAME.get(), self_mask.mask_pose(feed, never)))
        return np.zeros(depth.shape, dtype=bool)

    grid = OccupancyMap(
        client=FakeGridClient(),
        region_min=np.full(3, -2.0),
        region_max=np.full(3, 2.0),
        required=True,
        max_age_s=3.0,
    )
    grid.add_robot_body_mask(mask)
    made.source.items.append(made.pair(1))
    with made.camera, self_mask.frame_context():
        frame = made.camera.get_frame()
        stamp = frame.capture["capture_thor_mono_s"]
        feed.on_state(joints_state([read(0, 0.3, stamp - 0.01)]))
        feed.on_state(joints_state([read(0, 0.3, stamp + 0.01)], seq=2))
        grid.refresh(frame, np.eye(4))
    assert grid.last_error is None and grid.refreshes == 1
    # The grid keeps exactly the float the camera noted: a permit finds its
    # evidence time by it.
    assert grid._last_refresh == frame.t
    assert (
        made.frame_times.lookup(grid._last_refresh)
        == (frame.capture["evidence_wall_s"])
    )
    # The mask asked for the pose at this camera frame's capture time.
    assert poses and all(seen is frame for seen, _ in poses)
    pose = poses[-1][1]
    assert pose.t == frame.t
    assert pose.mask_alignment["capture_thor_mono_s"] == stamp
    assert self_mask.FRAME.get() is None


# The experiment's depth on a recorded raw pair, with the CPU matcher


class LiveThor:
    """A robot side that sends a ROBOT_STATE every 50 ms, on the real clocks."""

    def __init__(self) -> None:
        self.thor = FakeThor("test")
        self.link = AgentLink(self.thor.service_end)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        until(lambda: self.link._states > 0, "the first ROBOT_STATE")

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            seq += 1
            self.thor.send(a_state(seq, time.monotonic() + AHEAD))
            self._stop.wait(0.05)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(5.0)
        self.link.close()
        self.thor.carrier.close()


@pytest.fixture
def live() -> Iterator[LiveThor]:
    made = LiveThor()
    yield made
    made.close()


def recorded_camera(
    experiment_dir: Path, live: LiveThor
) -> tuple[Any, FrameTimes, list[float]]:
    """The camera on the recorded raw pair; ahead[0] moves its invented times."""
    from apai.config import Cfg

    import_experiment("thor_camera")
    frame_times = FrameTimes()
    ahead = [0.0]
    profile = {
        "type": CAMERA_TYPE,
        "name": "head",
        "width": 1280,
        "height": 992,
        "fps": 7,
        "calibration_dir": str(experiment_dir / STEREO_CALIBRATION),
        "extrinsics": {
            "mode": "eye_to_hand",
            "hand_eye_json": str(experiment_dir / HAND_EYE),
        },
        "depth_backend": "opencv_cpu",
        "frame_timeout_s": 10,
        "max_frame_age_s": 3,
    }
    source = RecordedPairSource(
        experiment_dir / RAW_PAIR,
        period_s=0.01,
        thor_wall_now=lambda: live.link.thor_now()[1] + ahead[0],
    )
    camera = stereo_camera.LinkStereoCamera(
        Cfg(profile), link=live.link, frame_times=frame_times, source=source
    )
    return camera, frame_times, ahead


@pytest.mark.apai
@pytest.mark.experiment
def test_recorded_pair_gives_depth_times_and_calibration_id(
    experiment_dir: Path, live: LiveThor
) -> None:
    camera, frame_times, _ = recorded_camera(experiment_dir, live)
    stereo_depth = import_experiment("thor_camera").StereoDepth
    with camera:
        assert isinstance(camera.depth, stereo_depth)
        frame = camera.get_frame()
        k = camera.depth.K.copy()
    capture = frame.capture
    assert capture["calibration_id"] == camera.calibration_id == CALIBRATION_ID
    assert capture["calibration_status"] == "provisional_metric_calibration"
    assert capture["depth_backend"] == "opencv_cpu"
    assert capture["depth_parameters"] == {"backend": "opencv_cpu"}
    assert capture["sensor_ids"] == [5, 7]
    assert capture["check"]["invented_times"] is True

    # The rectified left image and metric depth, at the calibration's map size.
    assert frame.rgb.shape == (992, 1280, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth_m.shape == (992, 1280) and frame.depth_m.dtype == np.float32
    assert np.array_equal(frame.K, k) and frame.K.shape == (3, 3)
    valid = frame.depth_m[frame.depth_m > 0]
    assert capture["valid_fraction"] == pytest.approx(valid.size / frame.depth_m.size)
    assert capture["valid_fraction"] > 0.2
    assert 0.2 <= valid.min() and valid.max() <= 8.0
    assert 0.3 < float(np.median(valid)) < 3.0  # a table scene in front of the arm

    # The times: the pair was "captured" 0.06 s before it was handed out.
    evidence = capture["evidence_wall_s"]
    assert frame_times.lookup(frame.t) == evidence
    assert capture["capture_thor_mono_s"] == pytest.approx(evidence - SKEW, abs=CLOSE)
    assert frame.t == pytest.approx(
        capture["receipt_monotonic_s"] - 0.06 - MARGIN, abs=0.02
    )
    age = time.monotonic() - frame.t
    assert 0.0 < age < 3.0  # what APAI's grid will measure, depth time included


@pytest.mark.apai
@pytest.mark.experiment
def test_frame_t_is_never_later_than_the_receipt(
    experiment_dir: Path, live: LiveThor
) -> None:
    camera, frame_times, ahead = recorded_camera(experiment_dir, live)
    with camera:
        first = camera.get_frame()
        # The next pair's capture time lies 40 ms after its hand-out, as with a
        # clock estimate that is off: the estimate dates it after its arrival,
        # and the arrival stamp wins.
        ahead[0] = 0.06 + 0.04
        second = camera.get_frame()
    assert first.t < first.capture["receipt_monotonic_s"]
    receipt = second.capture["receipt_monotonic_s"]
    sof_s = min(second.capture["sensor_timestamps_ns"]) / 1e9
    assert sof_s - SKEW - AHEAD > receipt + 0.03  # by the two clocks
    assert second.t == receipt
    # The evidence time is the arrival's, not the later capture time.
    assert second.capture["evidence_wall_s"] < sof_s - 0.03
    assert second.capture["capture_thor_mono_s"] - AHEAD <= receipt + 0.005
    assert second.t > first.t and second.frame_id == 2
    for frame in (first, second):
        assert frame.t <= frame.capture["receipt_monotonic_s"] <= time.monotonic()
        assert frame_times.lookup(frame.t) == frame.capture["evidence_wall_s"]
