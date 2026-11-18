# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's perception against the functional camera path.

The functional code is the experiment's on-device agent: its camera class
(thor_camera.ThorStereoCamera) and its self mask (thor_self_mask). Each test
feeds the same input to the functional routine and to this example's, and
compares what comes out; no expected number is typed here. "Undated" is a
camera frame whose capture the robot link could not date: it carries no
evidence time (the capture time a PERMIT quotes).

No GPU, no camera and no arm: the recorded raw stereo pair of the experiment
folder, its calibration files, and the OpenCV matcher on the CPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from conftest import import_experiment
from repo_paths import repo_root
from test_self_mask import a_state
from test_stereo_camera import AHEAD, MARGIN, SKEW, FakeDepth, FakeSource, StubLink

from isaaccapture_examples.agentic_remote.agent import self_mask
from isaaccapture_examples.agentic_remote.agent.camera import stereo_camera
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import (
    CameraError,
    StereoPair,
    bgr_to_nv12,
)
from isaaccapture_examples.agentic_remote.agent.checkahead import FrameTimes
from isaaccapture_examples.agentic_remote.protocol.messages import JointSample

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
EXPERIMENT_VARIABLE = "${AGENTIC_REMOTE_EXPERIMENT_DIR}"
RAW_PAIR = ".state/thor/captures/20261106T212519Z"
CALIBRATION = ".state/thor/calibration"
# The head of pairs.csv, as the experiment's capture program writes it
# (sensing_capture.cpp).
PAIRS_HEAD = (
    "pair_index,capture_id,sensor5_timestamp_ns,sensor7_timestamp_ns,"
    "receipt_monotonic_s,elapsed_s,left_file,right_file\n"
)
# What only one of the two cameras says of a camera frame, by its nature: the
# functional one names its capture folder and pair, this one its camera
# session, its clocks and its calibration.
FUNCTIONAL_ONLY = {"source", "pair_index", "capture_id"}
# The camera reads the running clock twice to date a pair.
CLOSE = 1e-4

pytestmark = [pytest.mark.apai, pytest.mark.experiment]


def shipped(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def shipped_hand_eye(experiment_dir: Path) -> Path:
    """camera.hand_eye_json of the rig's agent file, in the experiment folder."""
    text = shipped("agent_rebot.yaml")["camera"]["hand_eye_json"]
    assert text.startswith(EXPERIMENT_VARIABLE)
    return Path(text.replace(EXPERIMENT_VARIABLE, str(experiment_dir)))


@pytest.fixture(scope="module")
def functional_profile(experiment_dir: Path) -> dict[str, Any]:
    """The functional camera profile: the one beside the hand-eye file in use.

    A fit exported from the Thor brings none: then the one of the newest fit
    of this machine's copy, which names the same camera.
    """
    others = experiment_dir.glob(".state/thor/*/*/camera-candidate.yaml")
    profiles = [
        shipped_hand_eye(experiment_dir).with_name("camera-candidate.yaml"),
        *sorted(others, key=lambda path: path.parent.name, reverse=True),
    ]
    file = next(path for path in profiles if path.is_file())
    return yaml.safe_load(file.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw_pair(experiment_dir: Path) -> dict[str, list[np.ndarray]]:
    """The recorded raw pair as I420 (the functional capture) and as NV12 (ours)."""
    import cv2

    eyes: dict[str, list[np.ndarray]] = {"i420": [], "nv12": []}
    for name in ("left_raw.jpg", "right_raw.jpg"):
        bgr = cv2.imread(str(experiment_dir / RAW_PAIR / name), cv2.IMREAD_COLOR)
        assert bgr is not None, name
        eyes["i420"].append(cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420))
        eyes["nv12"].append(bgr_to_nv12(bgr))
    return eyes


class Running:
    """In place of the capture program's process: it has not ended."""

    def poll(self) -> None:
        return None


class Functional:
    """The experiment's camera on a capture folder the test writes.

    The camera's own open() starts the Argus capture program; here its folder,
    its depth and a process that still runs are set as open() leaves them, and
    publish() adds a pair as the capture program does.
    """

    def __init__(self, folder: Path, profile: dict[str, Any], depth: Any) -> None:
        from apai.config import Cfg

        self.module = import_experiment("thor_camera")
        self.folder = folder
        self.camera = self.module.ThorStereoCamera(Cfg(profile))
        self.camera.process = Running()
        self.camera.directory = folder
        self.camera.depth = depth
        (folder / "pairs.csv").write_text(PAIRS_HEAD)

    def publish(
        self,
        index: int,
        eyes: list[np.ndarray],
        receipt: float,
        sensor_ns: tuple[int, int],
        suffix: str = ".i420",
    ) -> None:
        names = [f"pair-{index}-sensor-{sensor}{suffix}" for sensor in (5, 7)]
        for name, eye in zip(names, eyes, strict=True):
            eye.tofile(self.folder / name)
        with (self.folder / "pairs.csv").open("a") as rows:
            rows.write(
                f"{index},{index},{sensor_ns[0]},{sensor_ns[1]},{receipt!r},0.0,"
                f"{names[0]},{names[1]}\n"
            )


def link_camera(
    profile: dict[str, Any], link: Any, source: Any, depth: Any = None
) -> tuple[Any, FrameTimes]:
    from apai.config import Cfg

    frame_times = FrameTimes()
    camera = stereo_camera.LinkStereoCamera(
        Cfg(profile), link=link, frame_times=frame_times, source=source, depth=depth
    )
    return camera, frame_times


def link_profile(experiment_dir: Path, functional_profile: dict[str, Any]) -> dict:
    """This example's camera profile on the functional profile's numbers."""
    return {
        "type": stereo_camera.CAMERA_TYPE,
        "name": "head",
        "calibration_dir": str(experiment_dir / CALIBRATION),
        "extrinsics": {
            "mode": "eye_to_hand",
            "hand_eye_json": str(shipped_hand_eye(experiment_dir)),
        },
        **{
            key: functional_profile[key]
            for key in ("width", "height", "fps", "frame_timeout_s", "max_frame_age_s")
            if key in functional_profile
        },
    }


def a_pair(
    eyes: list[np.ndarray],
    seq: int,
    receipt: float,
    sensor_ns: tuple[int, int],
) -> StereoPair:
    return StereoPair(1, seq, eyes[0], eyes[1], sensor_ns, (5, 7), 0, receipt, {})


def capture_ns(link: StubLink, age_s: float) -> tuple[int, int]:
    """Start of exposure age_s ago on the Thor's wall clock; the eyes 5 us apart."""
    start = round((link.thor_now()[1] - age_s) * 1e9)
    return (start + 5000, start)


# One pair through both cameras, with the experiment's depth


@dataclasses.dataclass
class Both:
    functional: Any  # the Frame of the experiment's camera
    ours: Any  # the Frame of LinkStereoCamera
    pair: StereoPair
    frame_times: FrameTimes
    depth_k: np.ndarray


@pytest.fixture(scope="module")
def both(
    experiment_dir: Path,
    functional_profile: dict[str, Any],
    raw_pair: dict[str, list[np.ndarray]],
    tmp_path_factory: pytest.TempPathFactory,
) -> Both:
    """The same raw pair, calibration and matcher through the two cameras."""
    thor_camera = import_experiment("thor_camera")
    calibration = experiment_dir / CALIBRATION
    link = StubLink(time.monotonic)
    receipt = time.monotonic() - 0.03
    sensor_ns = capture_ns(link, 0.08)
    # The matcher on the CPU is slow: the age limit has a test of its own.
    patient = {**functional_profile, "max_frame_age_s": 120.0}

    depth = thor_camera.StereoDepth(calibration, "opencv_cpu", {})
    try:
        theirs = Functional(
            tmp_path_factory.mktemp("functional"),
            {**patient, "type": "sensing_stereo"},
            depth,
        )
        theirs.publish(1, raw_pair["i420"], receipt, sensor_ns)
        functional = theirs.camera.get_frame()
        depth_k = depth.K.copy()
    finally:
        depth.close()

    profile = {
        **link_profile(experiment_dir, patient),
        "depth_backend": "opencv_cpu",
    }
    pair = a_pair(raw_pair["nv12"], 1, receipt, sensor_ns)
    camera, frame_times = link_camera(profile, link, FakeSource(pair))
    with camera:
        ours = camera.get_frame()
    return Both(functional, ours, pair, frame_times, depth_k)


def test_the_same_pair_gives_apai_the_same_camera_frame_as_the_functional_camera(
    both: Both,
) -> None:
    """Every field but the times and the capture facts: type, shape, values."""
    from apai.types import Frame

    assert type(both.functional) is type(both.ours) is Frame
    compared = []
    for field in dataclasses.fields(Frame):
        if field.name in ("t", "observed_t", "capture"):
            continue  # the next two tests
        theirs = getattr(both.functional, field.name)
        ours = getattr(both.ours, field.name)
        assert type(ours) is type(theirs), field.name
        if isinstance(theirs, np.ndarray):
            assert ours.dtype == theirs.dtype, field.name
            assert ours.shape == theirs.shape, field.name
            assert np.array_equal(ours, theirs), field.name
        else:
            assert ours == theirs, field.name
        compared.append(field.name)
    assert {"rgb", "depth_m", "K", "depth_source", "frame_id"} <= set(compared)
    assert both.ours.has_depth is both.functional.has_depth
    # The images are of the functional size, and the depth has valid pixels.
    assert both.functional.rgb.ndim == 3 and both.functional.depth_m.any()
    assert np.array_equal(both.ours.K, both.depth_k)


def test_the_time_fields_are_of_the_functional_kind_on_this_machines_clock(
    both: Both,
) -> None:
    from apai.perception import freshness

    theirs, ours, pair = both.functional, both.ours, both.pair
    for frame in (theirs, ours):
        assert type(frame.t) is float and frame.observed_t == frame.t
        # APAI dates both the same way: a local frame on the client's clock.
        assert freshness.observed_at(frame) == frame.t
        assert 0.0 <= time.monotonic() - frame.t
    their_marker = freshness.capture_marker(theirs)
    our_marker = freshness.capture_marker(ours)
    assert our_marker.keys() == their_marker.keys()
    for key in ("channel", "clock"):
        assert our_marker[key] == their_marker[key]
    # The functional camera dates a camera frame at its receipt.
    assert theirs.t == theirs.capture["receipt_monotonic_s"] == pair.receiver_rx_mono_s
    # Ours keeps that receipt, and dates the frame at its capture, which the
    # link could date here: the start of exposure less the clock margin.
    assert ours.capture["receipt_monotonic_s"] == theirs.capture["receipt_monotonic_s"]
    capture_here = min(pair.sof_wall_ns) / 1e9 - SKEW - AHEAD
    assert ours.t == pytest.approx(capture_here - MARGIN, abs=CLOSE)
    # Never later than the functional time: every age APAI measures is as old
    # or older.
    assert ours.t <= theirs.t
    assert ours.capture["time_problem"] is None
    assert both.frame_times.lookup(ours.t) == ours.capture["evidence_wall_s"]


def test_the_capture_facts_the_functional_camera_gives_are_given_alike(
    both: Both,
) -> None:
    theirs, ours = both.functional.capture, both.ours.capture
    assert theirs["backend"] != ours["backend"]  # each names its own camera
    shared = set(theirs) - FUNCTIONAL_ONLY - {"backend"}
    # The launcher adds the mounting facts to ours, as thor_runtime does to theirs.
    shared.discard("camera_to_robot_base_calibrated")
    assert shared <= set(ours), shared - set(ours)
    for key in sorted(shared):
        assert type(ours[key]) is type(theirs[key]), key
        if key == "depth_timings_ms":
            assert ours[key].keys() == theirs[key].keys()
        else:
            assert ours[key] == theirs[key], key
    assert {"depth_backend", "depth_parameters", "valid_fraction"} <= shared
    assert {"sensor_ids", "sensor_timestamps_ns", "calibration_status"} <= shared
    json.dumps(ours)


# The rules that refuse or drop a pair, with a stand-in for depth


class Pairing:
    """One pair at a time through both cameras; depth is a stand-in."""

    def __init__(
        self,
        folder: Path,
        experiment_dir: Path,
        functional_profile: dict[str, Any],
        raw_pair: dict[str, list[np.ndarray]],
    ) -> None:
        # A short wait for a pair: a dropped pair ends as "no pair in time".
        waits = {**functional_profile, "frame_timeout_s": 0.3}
        self.raw_pair = raw_pair
        self.link = StubLink(time.monotonic)
        self.theirs = Functional(
            folder, {**waits, "type": "sensing_stereo"}, FakeDepth(folder, "same", {})
        )
        self.source = FakeSource()
        self.ours, self.frame_times = link_camera(
            link_profile(experiment_dir, waits), self.link, self.source, FakeDepth
        )
        self.ours.open()
        self.index = 0

    def close(self) -> None:
        self.ours.close()

    def give(
        self, receipt: float, sensor_ns: tuple[int, int] | None = None
    ) -> tuple[Any, Any]:
        """(theirs, ours): each a Frame, or the CameraError it raised."""
        self.index += 1
        if sensor_ns is None:
            sensor_ns = capture_ns(self.link, time.monotonic() - receipt + 0.05)
        self.theirs.publish(self.index, self.raw_pair["i420"], receipt, sensor_ns)
        self.source.items.append(
            a_pair(self.raw_pair["nv12"], self.index, receipt, sensor_ns)
        )
        outcome = []
        for camera in (self.theirs.camera, self.ours):
            try:
                outcome.append(camera.get_frame())
            except CameraError as refusal:
                outcome.append(refusal)
        return outcome[0], outcome[1]


@pytest.fixture
def pairing(
    tmp_path: Path,
    experiment_dir: Path,
    functional_profile: dict[str, Any],
    raw_pair: dict[str, list[np.ndarray]],
) -> Iterator[Pairing]:
    made = Pairing(tmp_path, experiment_dir, functional_profile, raw_pair)
    yield made
    made.close()


def delivered(frame: Any) -> bool:
    return not isinstance(frame, CameraError)


def test_the_camera_fails_at_the_first_refusal_as_the_functional_camera_does() -> None:
    theirs = import_experiment("thor_camera").ThorStereoCamera
    ours = stereo_camera.LinkStereoCamera
    assert ours.MAX_CONSECUTIVE_FAILURES == theirs.MAX_CONSECUTIVE_FAILURES


def test_the_pixels_reach_depth_as_the_functional_capture_file_does(
    pairing: Pairing,
) -> None:
    """NV12 from the camera connection against the I420 file of the Thor."""
    theirs, ours = pairing.give(time.monotonic() - 0.03)
    assert delivered(theirs) and delivered(ours)
    # The stand-in for depth returns the left image as it got it.
    assert ours.rgb.dtype == theirs.rgb.dtype and ours.rgb.shape == theirs.rgb.shape
    assert np.array_equal(ours.rgb, theirs.rgb)
    rows, width = pairing.raw_pair["i420"][0].shape
    assert theirs.rgb.shape == (rows * 2 // 3, width, 3)  # the whole raw image


@pytest.mark.parametrize("share", [0.1, 0.8, 1.2])
def test_a_pair_is_handed_on_or_dropped_by_the_functional_age_limit(
    pairing: Pairing, functional_profile: dict[str, Any], share: float
) -> None:
    """The one age limit is the functional profile's max_frame_age_s."""
    limit_s = float(functional_profile["max_frame_age_s"])
    theirs, ours = pairing.give(time.monotonic() - share * limit_s)
    assert delivered(ours) is delivered(theirs)
    assert delivered(theirs) is (share < 1.0)
    assert pairing.ours.dropped_old == (0 if delivered(theirs) else 1)


def test_a_profile_without_an_age_limit_gets_the_functional_cameras_own(
    tmp_path: Path,
    experiment_dir: Path,
    functional_profile: dict[str, Any],
    raw_pair: dict[str, list[np.ndarray]],
) -> None:
    """The limit a camera profile leaves out is the same in both cameras."""
    profile = dict(functional_profile)
    del profile["max_frame_age_s"]
    made = Pairing(tmp_path, experiment_dir, profile, raw_pair)
    try:
        outcomes = []
        for age_s in (16.0, 8.0, 4.0, 2.0, 1.0, 0.5):  # the receipts grow
            theirs, ours = made.give(time.monotonic() - age_s)
            outcomes.append((delivered(theirs), delivered(ours)))
    finally:
        made.close()
    assert all(ours is theirs for theirs, ours in outcomes), outcomes
    assert {theirs for theirs, _ in outcomes} == {True, False}, outcomes


def test_a_receipt_that_is_no_past_time_or_does_not_grow_is_refused_by_both(
    pairing: Pairing,
) -> None:
    theirs, ours = pairing.give(time.monotonic() + 30.0)
    assert not delivered(theirs) and not delivered(ours)
    # Dated at the receipt, as the functional camera dates: the link has no clock.
    pairing.link.has_state = False
    receipt = time.monotonic() - 0.03
    theirs, ours = pairing.give(receipt, (1, 1))
    assert delivered(theirs) and delivered(ours)
    assert ours.t == theirs.t == receipt
    theirs, ours = pairing.give(receipt, (2, 2))  # a new pair, the same receipt
    assert not delivered(theirs) and not delivered(ours)
    theirs, ours = pairing.give(time.monotonic() - 0.01, (3, 3))
    assert delivered(theirs) and delivered(ours) and ours.t == theirs.t


def no_state(link: StubLink) -> None:
    link.has_state = False


def old_state(link: StubLink) -> None:
    link.state_age_s = 2.0 * stereo_camera.STATE_MAX_AGE_S


def clock_step(link: StubLink) -> None:
    link.step_at = time.monotonic() - 0.5 * stereo_camera.CLOCK_STEP_QUIET_S


@pytest.mark.parametrize("trouble", [no_state, old_state, clock_step])
def test_trouble_on_the_robot_link_refuses_no_pair_the_functional_camera_delivers(
    pairing: Pairing, trouble: Any
) -> None:
    """The functional camera needs no robot link; ours withholds the evidence."""
    receipt = time.monotonic() - 0.03
    sensor_ns = capture_ns(pairing.link, 0.08)
    trouble(pairing.link)
    theirs, ours = pairing.give(receipt, sensor_ns)
    assert delivered(theirs) and delivered(ours)
    # Dated as the functional camera dates it, and the same picture.
    assert ours.t == theirs.t == theirs.capture["receipt_monotonic_s"]
    assert np.array_equal(ours.rgb, theirs.rgb)
    assert np.array_equal(ours.depth_m, theirs.depth_m)
    # No PERMIT and no self mask can rest on it.
    assert ours.capture["evidence_wall_s"] is None
    assert ours.capture["capture_thor_mono_s"] is None and ours.capture["time_problem"]
    assert len(pairing.frame_times) == 0 and pairing.ours.undated == 1
    token = self_mask.FRAME.set(ours)
    try:
        from apai.perception.robot_mask import BodyMaskUnavailable

        with pytest.raises(BodyMaskUnavailable, match="capture_thor_mono_s"):
            self_mask.mask_pose(self_mask.JointFeed(1), lambda: None)
    finally:
        self_mask.FRAME.reset(token)


@pytest.mark.parametrize("captured_ago", ["future", "late"])
def test_a_capture_time_the_functional_camera_never_reads_refuses_no_pair(
    pairing: Pairing, functional_profile: dict[str, Any], captured_ago: str
) -> None:
    """The functional camera checks the receipt, never the sensor's time stamps."""
    limit_s = float(functional_profile["max_frame_age_s"])
    # An hour ahead of the Thor's clock; or long before the arrival (a late
    # camera connection), inside the functional age limit.
    age_s = -3600.0 if captured_ago == "future" else 0.6 * limit_s
    receipt = time.monotonic() - 0.03
    theirs, ours = pairing.give(receipt, capture_ns(pairing.link, age_s))
    assert delivered(theirs) and delivered(ours)
    if captured_ago == "future":
        assert ours.t == theirs.t and ours.capture["evidence_wall_s"] is None
    else:
        # Dated at its capture: APAI's ages count the late connection.
        assert ours.capture["time_problem"] is None
        assert theirs.t - ours.t == pytest.approx(age_s + MARGIN - 0.03, abs=0.01)


# The arm pose for the self mask


def functional_outcome(history: Any, stamp: float) -> tuple[Any, ...]:
    from apai.perception.robot_mask import BodyMaskUnavailable

    try:
        state = history.state_at(stamp)
    except BodyMaskUnavailable as refusal:
        return ("refused", str(refusal))
    return (
        "pose",
        [float(value) for value in state.q],
        state.gripper_valid,
        state.mask_alignment["joints"],
    )


def our_outcome(feed: Any, stamp: float) -> tuple[Any, ...]:
    from apai.perception.robot_mask import BodyMaskUnavailable

    def never() -> Any:
        raise AssertionError("the newest pose was asked for")

    frame = types.SimpleNamespace(
        t=time.monotonic(),
        capture={"backend": self_mask.LINK_STEREO, "capture_thor_mono_s": stamp},
    )
    token = self_mask.FRAME.set(frame)
    try:
        state = self_mask.mask_pose(feed, never)
    except BodyMaskUnavailable as refusal:
        return ("refused", str(refusal))
    finally:
        self_mask.FRAME.reset(token)
    assert state.t == frame.t
    assert state.mask_alignment["capture_thor_mono_s"] == stamp
    return (
        "pose",
        [float(value) for value in state.q],
        state.gripper_valid,
        state.mask_alignment["joints"],
    )


def mask_cases(history: Any) -> dict[str, tuple[list[tuple], float]]:
    """name -> (readings (joint, value, start, end), camera-frame time).

    Every time is built from the functional limits: the longest read, the
    longest gap between two readings and the longest hold of one.
    """
    read_s, gap_s, hold_s = history.MAX_READ_S, history.MAX_GAP_S, history.MAX_HOLD_S
    took = read_s / 10.0

    def at(end: float, values: tuple[float, float], took: float = took) -> list[tuple]:
        return [(joint, value, end - took, end) for joint, value in enumerate(values)]

    start = 10.0
    return {
        "between two readings": (
            at(start, (0.10, -0.30)) + at(start + gap_s / 2, (0.12, -0.30)),
            start + gap_s / 8,
        ),
        "on the last reading, young enough to hold": (
            at(start, (0.10, -0.30)),
            start + hold_s / 2,
        ),
        "on the last reading, too old to hold": (
            at(start, (0.10, -0.30)),
            start + 2 * hold_s,
        ),
        "between two readings too far apart in time": (
            at(start, (0.10, -0.30)) + at(start + 2 * gap_s, (0.10, -0.30)),
            start + gap_s,
        ),
        "between two readings of a joint that moved far": (
            at(start, (0.10, -0.30)) + at(start + gap_s / 2, (1.10, -0.30)),
            start + gap_s / 8,
        ),
        "a read that took too long is left out": (
            at(start, (0.10, -0.30))
            + at(start + gap_s / 4, (0.90, 0.90), took=2 * read_s)
            + at(start + gap_s / 2, (0.12, -0.30)),
            start + gap_s / 3,
        ),
        "before the first reading": (at(start, (0.10, -0.30)), start - 1.0),
        "readings of many moments": (
            [
                row
                for step in range(12)
                for row in at(start + step * gap_s / 20, (0.01 * step, -0.30))
            ],
            start + 5.5 * gap_s / 20,
        ),
    }


MASK_CASES = (
    "between two readings",
    "on the last reading, young enough to hold",
    "on the last reading, too old to hold",
    "between two readings too far apart in time",
    "between two readings of a joint that moved far",
    "a read that took too long is left out",
    "before the first reading",
    "readings of many moments",
)


@pytest.mark.parametrize("case", MASK_CASES)
def test_the_mask_pose_is_what_the_functional_history_says_of_the_same_readings(
    case: str,
) -> None:
    """The functional arm adds each read to JointHistory; ours gets ROBOT_STATEs."""
    history = import_experiment("thor_self_mask").JointHistory
    assert set(mask_cases(history)) == set(MASK_CASES)
    readings, stamp = mask_cases(history)[case]

    theirs = history(2)
    for joint, value, start, end in readings:
        theirs.add([joint], [value], start, end)  # as its _read_joint wrapper does

    feed = self_mask.JointFeed(2)
    assert isinstance(feed.history, history)
    samples = [JointSample(joint, value, a, b, True) for joint, value, a, b in readings]
    # Two ROBOT_STATEs: the second repeats the first's window, newest first.
    half = len(samples) // 2
    feed.on_state(a_state(samples[:half] or samples))
    feed.on_state(a_state(list(reversed(samples)), seq=2))

    assert our_outcome(feed, stamp) == functional_outcome(theirs, stamp)


def test_the_mask_cases_meet_every_rule_of_the_functional_history() -> None:
    """The cases above are worth their name: each functional outcome occurs."""
    history = import_experiment("thor_self_mask").JointHistory
    outcomes = []
    for readings, stamp in mask_cases(history).values():
        theirs = history(2)
        for joint, value, start, end in readings:
            theirs.add([joint], [value], start, end)
        outcomes.append(functional_outcome(theirs, stamp))
    modes = {
        row["mode"]
        for outcome in outcomes
        if outcome[0] == "pose"
        for row in outcome[3]
    }
    # "Joint 1: <the rule> (numbers)": the rule alone.
    refusals = {
        outcome[1].split(":", 1)[1].split("(")[0].strip()
        for outcome in outcomes
        if outcome[0] == "refused"
    }
    assert len(modes) > 1 and len(refusals) > 2, (modes, refusals)


def test_frame_context_names_the_camera_frame_where_the_functional_adapters_do() -> (
    None
):
    """The same two methods of APAI's obstacle map, the same frame in each."""
    from apai.perception.occupancy import OccupancyMap

    thor_self_mask = import_experiment("thor_self_mask")
    try:
        import apai.control.rebot_rs_arm  # noqa: F401
    except ImportError as missing:
        pytest.skip(f"the functional adapters cannot be entered here: {missing}")
    seen: list[tuple[str, Any, Any]] = []

    def mask(grid, depth, K, T_base_cam, *, frame=None):  # noqa: N803
        seen.append(("mask", thor_self_mask.FRAME.get(), self_mask.FRAME.get()))

    def points(grid, frame, T_base_cam):  # noqa: N803
        seen.append(("points", thor_self_mask.FRAME.get(), self_mask.FRAME.get()))

    def run(context: Any) -> tuple[set[str], list[tuple[str, Any, Any]]]:
        before = dict(vars(OccupancyMap))
        grid = object.__new__(OccupancyMap)
        seen.clear()
        with context():
            wrapped = {
                name
                for name, value in vars(OccupancyMap).items()
                if before.get(name) is not value
            }
            grid._robot_depth_mask("d", "k", "t", frame="frame-1")
            grid._integrate_points("frame-2", "t")
            grid._robot_depth_mask("d", "k", "t")
        return wrapped, list(seen)

    with (
        patch.object(OccupancyMap, "_robot_depth_mask", mask),
        patch.object(OccupancyMap, "_integrate_points", points),
    ):
        their_methods, their_calls = run(thor_self_mask.self_mask_adapters)
        our_methods, our_calls = run(self_mask.frame_context)
    assert our_methods == their_methods and their_methods
    # Each names the frame in its own variable, and only there.
    assert [(kind, frame) for kind, frame, _ in their_calls] == (
        [(kind, frame) for kind, _, frame in our_calls]
    )
    assert any(frame is not None for _, frame, _ in their_calls)
    assert all(frame is None for _, _, frame in their_calls)


# The shipped camera settings


def test_the_rig_agent_file_names_a_hand_eye_fit_the_functional_launcher_would_take(
    experiment_dir: Path, functional_profile: dict[str, Any]
) -> None:
    """thor_runtime.load_config takes a mounting only with accepted_fit true."""
    hand_eye = shipped_hand_eye(experiment_dir)
    assert hand_eye.is_file()
    # A fit made here has result.json; one exported from the Thor summary.json.
    records = [hand_eye.with_name(name) for name in ("result.json", "summary.json")]
    record = json.loads(next(path for path in records if path.is_file()).read_text())
    assert record["accepted_fit"] is True
    own_profile = hand_eye.with_name("camera-candidate.yaml")
    if own_profile.is_file():
        # The fit's own camera profile names this very file, on the Thor's paths.
        named = Path(functional_profile["extrinsics"]["hand_eye_json"])
    else:
        # The export's record does, and names the file's SHA-256.
        named = Path(record["handeye_file"])
        assert record["sha256"] == hashlib.sha256(hand_eye.read_bytes()).hexdigest()
    assert named.parts[-2:] == hand_eye.parts[-2:]
    # The one camera number the agent file copies is the functional profile's.
    camera = shipped("agent_rebot.yaml")["camera"]
    assert camera["fps"] == functional_profile["fps"]
    assert camera["depth_backend"] == functional_profile["depth_backend"]


def test_the_viewers_bridge_starts_on_the_bridge_interpreter_of_the_agent_file(
    tmp_path: Path,
) -> None:
    from isaaccapture_examples.agentic_remote.agent.camera.frame_source import (
        CAMERA_PR_ENV,
    )
    from isaaccapture_examples.agentic_remote.viewer_live import live

    agents = shipped("agent_rebot.yaml")["camera"]["bridge_python"]
    variable = "${" + CAMERA_PR_ENV + "}"
    assert agents and agents.startswith(variable)
    python = Path(agents.replace(variable, str(tmp_path)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    make = live.bridge_source_factory(
        "/tmp/r.sock", tmp_path / "run", {CAMERA_PR_ENV: str(tmp_path)}
    )
    # What the viewer starts the camera frame bridge of its live picture under.
    assert make()._command[0] == str(python)
