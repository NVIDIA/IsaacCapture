# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arm pose at camera-frame time, from the joint samples of ROBOT_STATE."""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import EXPERIMENT_DIR_ENV, import_experiment
from link_bench import FakeThor, robot_state, until

from isaaccapture_examples.agentic_remote.agent import self_mask
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink
from isaaccapture_examples.agentic_remote.agent.self_mask import (
    FRAME,
    LINK_STEREO,
    JointFeed,
    experiment_module,
    frame_context,
    mask_pose,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    JointSample,
    RobotState,
)

PACKAGE_ROOT = Path(self_mask.__file__).resolve().parents[3]


def a_state(
    samples: list[JointSample], *, session_id: str = "s1", seq: int = 1
) -> RobotState:
    newest = {sample.joint: sample for sample in samples}
    return dataclasses.replace(
        robot_state(seq=seq, session_id=session_id),
        recent=list(samples),
        joints=list(newest.values()),
    )


def read(joint: int, value: float, end: float, took: float = 0.002) -> JointSample:
    return JointSample(joint, value, end - took, end, True)


class FakeHistory:
    """In place of the experiment's JointHistory: records what is added."""

    def __init__(self, n_joints: int) -> None:
        self.n_joints = n_joints
        self.added: list[tuple[int, float, float, float]] = []
        self.invalidated = 0

    def add(self, joints: list[int], values: list[float], start: float, end: float):
        if not all(math.isfinite(v) for v in (*values, start, end)) or end < start:
            raise ValueError("bad sample")
        self.added.append((joints[0], values[0], start, end))

    def invalidate(self) -> None:
        self.invalidated += 1
        self.added.clear()

    def latest(self) -> float:
        ends = [
            max((row[3] for row in self.added if row[0] == joint), default=-math.inf)
            for joint in range(self.n_joints)
        ]
        return min(ends)

    def state_at(self, stamp: float) -> Any:
        before = [row for row in self.added if row[3] <= stamp]
        after = [row for row in self.added if row[3] > stamp]
        return types.SimpleNamespace(
            q=[before[-1][1]] if before else [],
            t=stamp,
            gripper_valid=False,
            bracketed=bool(after),
            mask_alignment={"joints": ["evidence"], "gripper": "unknown"},
        )


# Without APAI and without the experiment


def test_the_two_modules_import_without_apai_torch_or_the_experiment() -> None:
    code = (
        "import sys\n"
        "import isaaccapture_examples.agentic_remote.agent.self_mask\n"
        "import isaaccapture_examples.agentic_remote.agent.camera.stereo_camera\n"
        "heavy = {'apai', 'torch', 'cupy', 'warp', 'cv2', 'isaaccapture'}\n"
        "heavy |= {name for name in sys.modules if name.startswith('thor_')}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", EXPERIMENT_DIR_ENV)
    }
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_the_experiment_is_imported_only_from_the_folder_the_environment_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(EXPERIMENT_DIR_ENV, raising=False)
    with pytest.raises(ImportError, match=f"{EXPERIMENT_DIR_ENV} is not set"):
        experiment_module("thor_self_mask")
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(tmp_path / "absent"))
    with pytest.raises(ImportError, match="is not a folder"):
        experiment_module("thor_self_mask")
    (tmp_path / "unit_e_probe_module.py").write_text("VALUE = 7\n")
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert experiment_module("unit_e_probe_module").VALUE == 7
    assert sys.path[-1] == str(tmp_path.resolve())  # appended: it shadows nothing
    sys.modules.pop("unit_e_probe_module", None)


def test_only_valid_samples_newer_than_the_last_of_their_joint_are_added() -> None:
    history = FakeHistory(2)
    feed = JointFeed(2, history=history)
    feed.on_state(
        a_state([read(0, 0.1, 10.0), read(1, 0.5, 10.01), read(0, 0.2, 10.02)])
    )
    assert [row[:2] for row in history.added] == [(0, 0.1), (1, 0.5), (0, 0.2)]
    assert history.added[0][2:] == (10.0 - 0.002, 10.0)  # t_start and t_end as sent
    # The next ROBOT_STATE repeats the window and brings one new reading per joint.
    feed.on_state(
        a_state(
            [
                read(0, 0.1, 10.0),
                read(1, 0.5, 10.01),
                read(0, 0.2, 10.02),
                read(1, 0.6, 10.03),
                read(0, 0.3, 10.04),
            ],
            seq=2,
        )
    )
    assert [row[:2] for row in history.added[3:]] == [(1, 0.6), (0, 0.3)]
    assert feed.added == 5 and feed.skipped == 0


def test_samples_out_of_order_in_one_state_are_added_in_time_order() -> None:
    history = FakeHistory(1)
    feed = JointFeed(1, history=history)
    feed.on_state(
        a_state([read(0, 0.3, 10.04), read(0, 0.1, 10.0), read(0, 0.2, 10.02)])
    )
    assert [row[1] for row in history.added] == [0.1, 0.2, 0.3]


def test_invalid_foreign_and_nonfinite_samples_are_left_out() -> None:
    history = FakeHistory(2)
    feed = JointFeed(2, history=history)
    feed.on_state(
        a_state(
            [
                JointSample(0, 0.1, 9.99, 10.0, False),  # a failed read
                read(2, 0.1, 10.0),  # no such joint
                read(-1, 0.1, 10.0),
                read(1, math.nan, 10.0),
                JointSample(1, 0.2, 10.2, 10.1, True),  # ends before it starts
                read(1, 0.4, 10.3),
            ]
        )
    )
    assert [row[:2] for row in history.added] == [(1, 0.4)]
    assert feed.skipped == 2


def test_a_new_session_clears_the_history() -> None:
    history = FakeHistory(1)
    feed = JointFeed(1, history=history)
    feed.on_state(a_state([read(0, 0.1, 500.0)]))
    assert history.invalidated == 0
    # The Thor's monotonic clock of the new session starts lower.
    feed.on_state(a_state([read(0, 0.7, 20.0)], session_id="s2"))
    assert history.invalidated == 1
    assert [row[:2] for row in history.added] == [(0, 0.7)]
    feed.clear()
    assert history.invalidated == 2
    feed.on_state(a_state([read(0, 0.7, 20.0)], session_id="s2", seq=2))
    assert len(history.added) == 1  # cleared means forgotten: the reading counts again


def test_a_feed_refuses_a_joint_count_below_one() -> None:
    with pytest.raises(ValueError, match="n_joints"):
        JointFeed(0, history=FakeHistory(0))


# With APAI


@pytest.fixture
def unavailable() -> type:
    from apai.perception.robot_mask import BodyMaskUnavailable

    return BodyMaskUnavailable


@pytest.mark.apai
def test_frame_context_names_the_camera_frame_during_both_grid_methods() -> None:
    from unittest.mock import patch

    from apai.perception.occupancy import OccupancyMap

    seen: list[tuple[str, Any, tuple]] = []

    def mask(grid, depth, K, T_base_cam, *, frame=None):  # noqa: N803
        seen.append(("mask", FRAME.get(), (grid, depth, K, T_base_cam, frame)))
        return "masked"

    def points(grid, frame, T_base_cam):  # noqa: N803
        seen.append(("points", FRAME.get(), (grid, frame, T_base_cam)))

    with (
        patch.object(OccupancyMap, "_robot_depth_mask", mask),
        patch.object(OccupancyMap, "_integrate_points", points),
    ):
        grid = object.__new__(OccupancyMap)
        with frame_context():
            assert FRAME.get() is None
            assert grid._robot_depth_mask("d", "k", "t", frame="frame-1") == "masked"
            assert FRAME.get() is None  # reset when the call returns
            grid._integrate_points("frame-2", "t")
            assert grid._robot_depth_mask("d", "k", "t") == "masked"
            assert OccupancyMap._robot_depth_mask is not mask
        assert OccupancyMap._robot_depth_mask is mask  # put back on exit
        assert OccupancyMap._integrate_points is points
    assert seen == [
        ("mask", "frame-1", (grid, "d", "k", "t", "frame-1")),
        ("points", "frame-2", (grid, "frame-2", "t")),
        ("mask", None, (grid, "d", "k", "t", None)),
    ]


@pytest.mark.apai
def test_frame_context_resets_the_camera_frame_when_the_grid_method_raises() -> None:
    from unittest.mock import patch

    from apai.perception.occupancy import OccupancyMap

    def points(grid, frame, T_base_cam):  # noqa: N803
        raise RuntimeError("no grid")

    with patch.object(OccupancyMap, "_integrate_points", points), frame_context():
        with pytest.raises(RuntimeError, match="no grid"):
            object.__new__(OccupancyMap)._integrate_points("frame", "t")
        assert FRAME.get() is None


@pytest.mark.apai
def test_frame_context_does_not_reach_another_thread() -> None:
    from unittest.mock import patch

    from apai.perception.occupancy import OccupancyMap

    elsewhere: list[Any] = []

    def points(grid, frame, T_base_cam):  # noqa: N803
        thread = threading.Thread(target=lambda: elsewhere.append(FRAME.get()))
        thread.start()
        thread.join(5.0)

    with patch.object(OccupancyMap, "_integrate_points", points), frame_context():
        object.__new__(OccupancyMap)._integrate_points("frame", "t")
    assert elsewhere == [None]


def a_frame(capture: dict | None, t: float = 77.0) -> Any:
    return types.SimpleNamespace(t=t, capture=capture)


@pytest.fixture
def in_frame() -> Iterator[Any]:
    tokens = []

    def enter(frame: Any) -> None:
        tokens.append(FRAME.set(frame))

    yield enter
    for token in reversed(tokens):
        FRAME.reset(token)


@pytest.mark.apai
def test_without_a_camera_frame_the_pose_is_get_states_with_the_jaws_unknown() -> None:
    from apai.types import RobotState as ApaiState

    measured = ApaiState(q=[0.1, 0.2], gripper_pos=0.4, gripper_valid=True, t=5.0)
    pose = mask_pose(None, lambda: measured)
    assert pose is not measured and measured.gripper_valid is True
    assert (list(pose.q), pose.t, pose.gripper_valid) == ([0.1, 0.2], 5.0, False)


@pytest.mark.apai
def test_a_camera_frame_of_another_backend_gets_the_newest_pose(in_frame: Any) -> None:
    measured = types.SimpleNamespace(q=[0.1], gripper_valid=True)
    for capture in (None, {}, {"backend": "mock"}, {"backend": "sensing_stereo"}):
        in_frame(a_frame(capture))
        assert mask_pose(JointFeed(1, history=FakeHistory(1)), lambda: measured) is (
            measured
        )


@pytest.mark.apai
def test_a_link_stereo_camera_frame_never_falls_back_to_the_newest_pose(
    in_frame: Any, unavailable: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never() -> Any:
        raise AssertionError("the newest pose was asked for")

    feed = JointFeed(1, history=FakeHistory(1))
    in_frame(a_frame({"backend": LINK_STEREO}))
    with pytest.raises(unavailable, match="capture_thor_mono_s"):
        mask_pose(feed, never)
    for bad in (None, "12.0", math.nan, math.inf, True):
        in_frame(a_frame({"backend": LINK_STEREO, "capture_thor_mono_s": bad}))
        with pytest.raises(unavailable, match="capture_thor_mono_s"):
            mask_pose(feed, never)
    in_frame(a_frame({"backend": LINK_STEREO, "capture_thor_mono_s": 12.0}))
    with pytest.raises(unavailable, match="no joint feed"):
        mask_pose(None, never)
    # Without the experiment folder there is no joint history to ask.
    monkeypatch.delenv(EXPERIMENT_DIR_ENV, raising=False)
    without = JointFeed(1)
    without.on_state(a_state([read(0, 0.1, 11.99)]))  # taken and dropped
    with pytest.raises(unavailable, match=f"{EXPERIMENT_DIR_ENV} is not set"):
        mask_pose(without, never)


@pytest.mark.apai
def test_pose_at_waits_for_the_state_that_brackets_the_capture_time() -> None:
    history = FakeHistory(1)
    feed = JointFeed(1, history=history)
    feed.on_state(a_state([read(0, 0.1, 10.0)]))
    timer = threading.Timer(0.05, feed.on_state, [a_state([read(0, 0.2, 10.1)], seq=2)])
    timer.start()
    try:
        began = time.monotonic()
        pose = feed.pose_at(10.05, wait_s=5.0, frame_t=3.5)
        waited = time.monotonic() - began
    finally:
        timer.join(5.0)
    assert pose.bracketed and 0.03 < waited < 2.0
    assert pose.t == 3.5 and pose.gripper_valid is False
    assert pose.mask_alignment == {
        "source": "ROBOT_STATE samples at the capture time",
        "capture_thor_mono_s": 10.05,
        "clock": "the Thor's monotonic clock",
        "joints": ["evidence"],
        "gripper": "unknown",
    }


@pytest.mark.apai
def test_pose_at_waits_no_longer_than_asked_and_then_lets_the_history_decide() -> None:
    history = FakeHistory(1)
    feed = JointFeed(1, history=history)
    feed.on_state(a_state([read(0, 0.1, 10.0)]))
    began = time.monotonic()
    pose = feed.pose_at(10.05, wait_s=0.1)
    waited = time.monotonic() - began
    assert not pose.bracketed and 0.09 < waited < 1.0
    assert pose.t == 10.05  # no camera-frame time was given
    # A later sample is already there: no wait at all.
    feed.on_state(a_state([read(0, 0.2, 10.1)], seq=2))
    began = time.monotonic()
    assert feed.pose_at(10.05, wait_s=5.0).bracketed
    assert time.monotonic() - began < 1.0


@pytest.mark.apai
def test_pose_at_refuses_a_time_that_is_not_a_finite_number(unavailable: type) -> None:
    feed = JointFeed(1, history=FakeHistory(1))
    began = time.monotonic()
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(unavailable, match="finite"):
            feed.pose_at(bad, wait_s=5.0)
    assert time.monotonic() - began < 1.0


# With the experiment's JointHistory


class Rig:
    """A JointFeed listening on an AgentLink whose robot side the test scripts."""

    def __init__(self, n_joints: int) -> None:
        self.thor = FakeThor("test")
        self.link = AgentLink(self.thor.service_end)
        self.feed = JointFeed(n_joints)
        self.link.add_state_listener(self.feed.on_state)
        self._seq = 0

    def arrive(self, samples: list[JointSample], session_id: str = "s1") -> None:
        self._seq += 1
        before = self.feed.states
        self.thor.send(a_state(samples, session_id=session_id, seq=self._seq))
        until(lambda: self.feed.states > before, "the ROBOT_STATE")

    def close(self) -> None:
        self.link.close()
        self.thor.carrier.close()


@pytest.fixture
def rig() -> Iterator[Rig]:
    import_experiment("thor_self_mask")
    made = Rig(2)
    yield made
    made.close()


@pytest.mark.apai
@pytest.mark.experiment
def test_pose_at_camera_frame_time_from_robot_state(
    rig: Rig, in_frame: Any, unavailable: type
) -> None:
    history = import_experiment("thor_self_mask").JointHistory
    assert isinstance(rig.feed.history, history)
    # Joint 0 moves at 0.5 rad/s, joint 1 stands; a reading every 20 ms.
    window = [
        read(joint, 0.5 * i * 0.02 if joint == 0 else -0.3, 100.0 + i * 0.02)
        for i in range(11)
        for joint in (0, 1)
    ]
    rig.arrive(window[:12])
    rig.arrive(window)  # overlaps the first, as the 0.5 s window of ROBOT_STATE does
    assert rig.feed.added == len(window)

    capture = {"backend": LINK_STEREO, "capture_thor_mono_s": 100.09}
    in_frame(a_frame(capture, t=41.25))

    def never() -> Any:
        raise AssertionError("the newest pose was asked for")

    pose = mask_pose(rig.feed, never)
    assert pose.q == pytest.approx([0.5 * 0.09, -0.3], abs=1e-9)
    assert pose.t == 41.25 and pose.gripper_valid is False
    record = pose.mask_alignment
    assert record["source"] == "ROBOT_STATE samples at the capture time"
    assert record["capture_thor_mono_s"] == 100.09
    assert [row["mode"] for row in record["joints"]] == [
        "interpolated_measurements"
    ] * 2
    assert "receipt_monotonic_s" not in record and "clock_limit" not in record

    # Before the first sample there is nothing to go by.
    with pytest.raises(unavailable, match="no measured feedback"):
        rig.feed.pose_at(99.9, wait_s=0.0)
    # After the newest sample: the preceding reading, while it is at most 0.15 s old.
    held = rig.feed.pose_at(100.3, wait_s=0.0)
    assert held.q == pytest.approx([0.1, -0.3])
    assert held.mask_alignment["joints"][0]["mode"] == "preceding_measurement"
    with pytest.raises(unavailable, match="lacks nearby measured feedback"):
        rig.feed.pose_at(100.4, wait_s=0.0)


@pytest.mark.apai
@pytest.mark.experiment
def test_the_limits_of_the_joint_history_are_unchanged(
    rig: Rig, unavailable: type
) -> None:
    history = import_experiment("thor_self_mask").JointHistory
    assert (history.MAX_READ_S, history.MAX_GAP_S, history.MAX_HOLD_S) == (
        0.15,
        0.4,
        0.15,
    )
    # A read that took longer than 0.15 s is not used: with it, joint 2 would be
    # refused at 10.25 (0.9 rad against 0.0 rad).
    rig.arrive([read(0, 0.0, 10.0), read(1, 0.0, 10.0), read(1, 0.9, 10.2, took=0.2)])
    rig.arrive([read(0, 0.0, 10.3), read(1, 0.0, 10.3)])
    assert rig.feed.pose_at(10.25, wait_s=0.0).q == pytest.approx([0.0, 0.0])
    # Two bracketing readings more than 0.05 rad apart are not interpolated.
    rig.arrive([read(0, 0.06, 10.4), read(1, 0.0, 10.4)])
    with pytest.raises(unavailable, match="Joint 1: feedback changed too far"):
        rig.feed.pose_at(10.35, wait_s=0.0)
    # Readings more than 0.4 s apart are not interpolated either, and the one
    # before is too old to hold.
    rig.arrive([read(0, 0.06, 10.9), read(1, 0.0, 10.9)])
    with pytest.raises(unavailable, match="lacks nearby measured feedback"):
        rig.feed.pose_at(10.6, wait_s=0.0)


@pytest.mark.apai
@pytest.mark.experiment
def test_a_new_session_forgets_the_old_sessions_samples(
    rig: Rig, unavailable: type
) -> None:
    rig.arrive([read(0, 0.1, 500.0), read(1, 0.1, 500.0)])
    assert rig.feed.pose_at(500.05, wait_s=0.0).q == pytest.approx([0.1, 0.1])
    rig.arrive([read(0, 0.7, 20.0), read(1, 0.7, 20.0)], session_id="s2")
    with pytest.raises(unavailable, match="lacks nearby measured feedback"):
        rig.feed.pose_at(500.05, wait_s=0.0)
    assert rig.feed.pose_at(20.05, wait_s=0.0).q == pytest.approx([0.7, 0.7])


@pytest.mark.apai
@pytest.mark.experiment
def test_pose_at_waits_for_the_next_robot_state_on_the_link(rig: Rig) -> None:
    rig.arrive([read(0, 0.1, 10.0), read(1, 0.1, 10.0)])
    later = a_state([read(0, 0.12, 10.1), read(1, 0.1, 10.1)], seq=99)
    timer = threading.Timer(0.05, rig.thor.send, [later])
    timer.start()
    try:
        pose = rig.feed.pose_at(10.05, wait_s=5.0)
    finally:
        timer.join(5.0)
    assert pose.q == pytest.approx([0.11, 0.1])
