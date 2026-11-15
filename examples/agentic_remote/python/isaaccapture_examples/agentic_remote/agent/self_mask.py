# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arm pose at camera-frame time, for APAI's self mask.

APAI removes the arm's own pixels from a depth image with the arm pose it gets
from the arm object, which takes no argument. frame_context() names the camera
frame being masked in FRAME, JointFeed keeps the joint samples of every
ROBOT_STATE, and mask_pose() answers with the pose at that camera frame's
capture time. All sample times stay on the Thor's monotonic clock.
"""

from __future__ import annotations

import copy
import importlib
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    import types

    from ..protocol.messages import RobotState

_log = logging.getLogger("isaaccapture.examples.agentic_remote.self_mask")

# The experiment folder (read-only); its modules are imported from there.
EXPERIMENT_DIR_ENV = "AGENTIC_REMOTE_EXPERIMENT_DIR"
# Frame.capture["backend"] of the camera frames of camera/stereo_camera.py.
LINK_STEREO = "link_stereo"
# What a pose from JointFeed says of itself in mask_alignment["source"].
ALIGNMENT_SOURCE = "ROBOT_STATE samples at the capture time"

# The camera frame APAI is masking or integrating right now, on this thread.
FRAME: ContextVar[Any] = ContextVar("agentic_remote_mask_frame", default=None)


def experiment_module(name: str) -> types.ModuleType:
    """Import a module of the experiment folder; ImportError says what is missing.

    The folder is appended to the import path, so it shadows nothing.
    """
    folder = os.environ.get(EXPERIMENT_DIR_ENV)
    if not folder:
        raise ImportError(
            f"{EXPERIMENT_DIR_ENV} is not set: the experiment's {name} cannot be "
            "imported"
        )
    if not Path(folder).is_dir():
        raise ImportError(f"{EXPERIMENT_DIR_ENV}={folder} is not a folder")
    resolved = str(Path(folder).resolve())
    if resolved not in sys.path:
        sys.path.append(resolved)
    return importlib.import_module(name)


@contextmanager
def frame_context() -> Iterator[None]:
    """While open, APAI's grid names the camera frame it works on in FRAME.

    Wraps OccupancyMap._robot_depth_mask and OccupancyMap._integrate_points for
    every grid of the process, as the experiment's self_mask_adapters() does.
    """
    from apai.perception.occupancy import OccupancyMap

    original_mask = OccupancyMap._robot_depth_mask
    original_points = OccupancyMap._integrate_points

    @wraps(original_mask)
    def robot_depth_mask(grid, depth, K, T_base_cam, *, frame=None):  # noqa: N803
        token = FRAME.set(frame)
        try:
            return original_mask(grid, depth, K, T_base_cam, frame=frame)
        finally:
            FRAME.reset(token)

    @wraps(original_points)
    def integrate_points(grid, frame, T_base_cam):  # noqa: N803
        token = FRAME.set(frame)
        try:
            return original_points(grid, frame, T_base_cam)
        finally:
            FRAME.reset(token)

    with (
        patch.object(OccupancyMap, "_robot_depth_mask", robot_depth_mask),
        patch.object(OccupancyMap, "_integrate_points", integrate_points),
    ):
        yield


class JointFeed:
    """The joint samples of ROBOT_STATE, kept in the experiment's JointHistory.

    on_state() is an AgentLink state listener and runs on the link's receive
    thread; pose_at() may be called from any thread. Without the experiment
    folder nothing is kept and pose_at() refuses.
    """

    def __init__(self, n_joints: int, *, history: Any = None) -> None:
        if n_joints < 1:
            raise ValueError(f"n_joints must be at least 1, got {n_joints!r}")
        self._n_joints = int(n_joints)
        self._unavailable = ""
        if history is None:
            try:
                history = experiment_module("thor_self_mask").JointHistory(n_joints)
            except ImportError as missing:
                self._unavailable = f"no joint history for the self mask: {missing}"
                _log.warning("%s", self._unavailable)
        self._history = history
        self._cond = threading.Condition()
        self._session: str | None = None
        self._last = [-math.inf] * self._n_joints  # t_end of the last sample added
        self.states = 0  # ROBOT_STATEs taken
        self.added = 0  # samples added to the history
        self.skipped = 0  # valid samples the history refused

    @property
    def history(self) -> Any:
        """The experiment's JointHistory; None without the experiment folder."""
        return self._history

    def on_state(self, state: RobotState) -> None:
        """Add every valid sample that is newer than the last one of its joint."""
        history = self._history
        with self._cond:
            self.states += 1
            if history is not None:
                if state.session_id != self._session:
                    if self._session is not None:
                        self._forget(history)
                    self._session = state.session_id
                samples = sorted(
                    (*state.recent, *state.joints), key=lambda sample: sample.t_end
                )
                for sample in samples:
                    joint = sample.joint
                    if (
                        not sample.valid
                        or not 0 <= joint < self._n_joints
                        or not sample.t_end > self._last[joint]
                    ):
                        continue
                    try:
                        history.add(
                            [joint], [sample.value], sample.t_start, sample.t_end
                        )
                    except ValueError:
                        self.skipped += 1
                        continue
                    self._last[joint] = sample.t_end
                    self.added += 1
            self._cond.notify_all()

    def clear(self) -> None:
        """Forget every sample."""
        with self._cond:
            if self._history is not None:
                self._forget(self._history)

    def pose_at(
        self, thor_mono_s: float, *, wait_s: float = 0.1, frame_t: float | None = None
    ) -> Any:
        """The arm pose at a time of the Thor's monotonic clock, as APAI's RobotState.

        Waits up to wait_s for a ROBOT_STATE with a later sample of every joint,
        then lets JointHistory.state_at decide with its own limits; it raises
        APAI's BodyMaskUnavailable. The state's t is frame_t, the camera frame's
        time on this machine's clock; without one it stays thor_mono_s.
        """
        from apai.perception.robot_mask import BodyMaskUnavailable

        history = self._history
        if history is None:
            raise BodyMaskUnavailable(self._unavailable)
        try:
            stamp = float(thor_mono_s)
        except (TypeError, ValueError):
            stamp = math.nan
        if not math.isfinite(stamp):
            raise BodyMaskUnavailable(
                "the camera frame's capture time on the Thor's clock is not a "
                f"finite number: {thor_mono_s!r}"
            )
        end = time.monotonic() + wait_s
        with self._cond:
            while not history.latest() > stamp:
                left = end - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
        state = history.state_at(stamp)
        if frame_t is not None:
            state.t = float(frame_t)
        state.gripper_valid = False
        earlier = getattr(state, "mask_alignment", None) or {}
        state.mask_alignment = {
            "source": ALIGNMENT_SOURCE,
            "capture_thor_mono_s": stamp,
            "clock": "the Thor's monotonic clock",
            "joints": earlier.get("joints"),
            "gripper": earlier.get("gripper"),
        }
        return state

    def _forget(self, history: Any) -> None:
        history.invalidate()
        self._last = [-math.inf] * self._n_joints


def mask_pose(feed: JointFeed | None, get_state: Callable[[], Any]) -> Any:
    """The pose for APAI's self mask: what the arm's body_mask_state() returns.

    With a camera frame of the link's stereo camera in FRAME: the pose at its
    capture time, or BodyMaskUnavailable; never a newer pose. With any other
    camera frame: get_state() as it is. Without one: a copy of get_state() with
    the jaws unknown.
    """
    frame = FRAME.get()
    if frame is None:
        state = copy.copy(get_state())
        state.gripper_valid = False
        return state
    capture = getattr(frame, "capture", None)
    if not isinstance(capture, dict) or capture.get("backend") != LINK_STEREO:
        return get_state()
    from apai.perception.robot_mask import BodyMaskUnavailable

    stamp = capture.get("capture_thor_mono_s")
    if type(stamp) not in (int, float) or not math.isfinite(stamp):
        raise BodyMaskUnavailable(
            f"the {LINK_STEREO} camera frame has no usable capture_thor_mono_s: "
            f"{stamp!r}"
        )
    if feed is None:
        raise BodyMaskUnavailable(
            f"no joint feed for the {LINK_STEREO} camera frame: no pose at its "
            "capture time"
        )
    return feed.pose_at(stamp, frame_t=frame.t)
