# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The camera APAI gets on the workstation: stereo pairs from the camera connection.

LinkStereoCamera turns each stereo pair of a PairSource into one camera frame
with depth, dated at its capture: Frame.t is the capture time on this machine's
monotonic clock, so the age APAI's obstacle grid measures counts encoding, the
network, decoding and depth. The capture time on the Thor's wall clock, the
evidence time, is noted in FrameTimes for the permits.

LinkStereoCamera subclasses APAI's CameraBase, so the class is made on first
use: read it as `stereo_camera.LinkStereoCamera`, or import it inside a function.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..link_client import StaleState
from ..self_mask import LINK_STEREO, experiment_module
from .frame_source import BridgeSource, CameraError, RecordedPairSource

if TYPE_CHECKING:
    from ..checkahead import FrameTimes
    from ..link_client import AgentLink
    from .frame_source import PairSource, StereoPair

_log = logging.getLogger("isaaccapture.examples.agentic_remote.stereo_camera")

# The camera profile type the factory answers with this camera.
CAMERA_TYPE = "link_stereo"
# A camera frame is refused when the newest ROBOT_STATE is older than this: its
# clock pair is what dates the camera frame.
STATE_MAX_AGE_S = 1.0
# A capture time this far ahead of the estimate of the Thor's wall clock is
# refused; so is one this far, plus the link's clock margin, after the arrival.
FUTURE_LIMIT_S = 0.05
# A capture time this far before the pair's arrival on this machine is refused.
BEFORE_ARRIVAL_LIMIT_S = 1.0
# After a step of the Thor's wall clock camera frames are refused for this long.
CLOCK_STEP_QUIET_S = 3.0
_NONE_PAUSE_S = 0.01

DepthFactory = Callable[[Path, str, Any], Any]


@dataclass(frozen=True)
class FrameStamp:
    """The times of one camera frame."""

    evidence_wall_s: float  # capture time, the Thor's wall clock; a PERMIT quotes it
    capture_thor_mono_s: float  # the same on the Thor's monotonic clock
    t: float  # the same on this machine's monotonic clock: Frame.t
    raw_age_s: float  # the Thor's wall clock now minus the capture time, no margin


def stamp_pair(
    link: AgentLink,
    pair: StereoPair,
    monotonic: Callable[[], float] = time.monotonic,
) -> FrameStamp:
    """Date a stereo pair on three clocks; CameraError when that cannot be trusted.

    `monotonic` must be the clock the link was built with and the clock of the
    pair's arrival stamp.
    """
    evidence = min(pair.sof_wall_ns) / 1e9
    arrival = float(pair.receiver_rx_mono_s)
    try:
        state_at = link.state_arrived_s()
        step_at = link.clock_step_at_s()
        now = monotonic()
        thor_wall = link.thor_now()[1]
        raw_age = thor_wall - evidence
        margin = link.wall_age_s(evidence) - raw_age
        converted = now - raw_age - margin
        thor_mono = link.wall_to_thor_mono(evidence)
    except StaleState:
        raise CameraError(
            "no time for the camera frame: no ROBOT_STATE has arrived yet"
        ) from None
    state_age = now - state_at
    if state_age > STATE_MAX_AGE_S:
        raise CameraError(
            f"no time for the camera frame: the newest ROBOT_STATE is "
            f"{state_age:.2f} s old, over {STATE_MAX_AGE_S:g} s"
        )
    if step_at is not None and now - step_at < CLOCK_STEP_QUIET_S:
        raise CameraError(
            f"the Thor's wall clock stepped {now - step_at:.2f} s ago; camera "
            f"frames are refused for {CLOCK_STEP_QUIET_S:g} s after a step"
        )
    if not math.isfinite(arrival) or arrival > now:
        raise CameraError(
            f"the stereo pair's arrival stamp ({arrival!r}) is not a past time of "
            "this machine's monotonic clock"
        )
    if not raw_age >= -FUTURE_LIMIT_S:
        raise CameraError(
            f"the camera frame's capture time lies {-raw_age:.3f} s in the future "
            f"of the Thor's wall clock, over {FUTURE_LIMIT_S:g} s"
        )
    # The capture time minus the arrival, both on the Thor's wall clock.
    lead = (now - arrival) - raw_age
    if lead > FUTURE_LIMIT_S + margin:
        raise CameraError(
            f"the camera frame's capture time lies {lead:.3f} s after its "
            f"arrival, over {FUTURE_LIMIT_S + margin:g} s"
        )
    if arrival - converted > BEFORE_ARRIVAL_LIMIT_S:
        raise CameraError(
            f"the camera frame's capture time is {arrival - converted:.2f} s "
            f"before its arrival, over {BEFORE_ARRIVAL_LIMIT_S:g} s"
        )
    # The evidence is never newer than the arrival.
    late = max(0.0, lead)
    return FrameStamp(
        evidence_wall_s=evidence - late,
        capture_thor_mono_s=float(thor_mono) - late,
        t=float(min(converted, arrival)),
        raw_age_s=float(raw_age),
    )


def calibration_id(calibration_dir: str | Path, hand_eye_json: str | Path) -> str:
    """Names the stereo calibration and the camera mounting by their files' hashes."""
    parts = []
    for label, path in (
        ("stereo", Path(calibration_dir) / "manifest.json"),
        ("mount", Path(hand_eye_json)),
    ):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as problem:
            raise CameraError(
                f"cannot read the calibration file {path}: {problem}"
            ) from None
        parts.append(f"{label}-{digest[:12]}")
    return "+".join(parts)


def experiment_depth(directory: Path, backend: str, options: Any) -> Any:
    """The experiment's StereoDepth; it checks the calibration files' hashes."""
    try:
        thor_camera = experiment_module("thor_camera")
    except ImportError as missing:
        raise CameraError(f"no stereo depth: {missing}") from None
    return thor_camera.StereoDepth(directory, backend, options)


class _LinkStereoInput:
    """What LinkStereoCamera does; APAI's CameraBase is added on first use."""

    MAX_CONSECUTIVE_FAILURES = 1

    def __init__(
        self,
        cfg: Any,
        *,
        link: AgentLink,
        frame_times: FrameTimes,
        source: PairSource | None = None,
        depth: DepthFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._link = link
        self._frame_times = frame_times
        self._given_source = source
        self._make_depth = depth or experiment_depth
        self._monotonic = monotonic
        self._source: PairSource | None = None
        self.depth: Any = None
        self.calibration_id = ""
        self._last_pair: tuple[int, int] | None = None
        self._last_evidence = -math.inf
        self._last_t = -math.inf
        self.delivered = 0  # camera frames handed to APAI
        self.dropped_old = 0  # stereo pairs older than max_frame_age_s after depth

    @property
    def has_depth(self) -> bool:
        return True

    def open(self) -> None:
        if self.depth is not None:
            raise CameraError("the stereo camera is already open")
        cfg = self.cfg
        directory = cfg.get("calibration_dir")
        hand_eye = (cfg.get("extrinsics") or {}).get("hand_eye_json")
        if not directory or not hand_eye:
            raise CameraError(
                f"a {CAMERA_TYPE} camera profile needs calibration_dir and "
                "extrinsics.hand_eye_json"
            )
        self.calibration_id = calibration_id(directory, hand_eye)
        backend = cfg.get("depth_backend", "opencv_cpu")
        depth = self._make_depth(Path(directory), backend, cfg.get("ffs", {}))
        try:
            source = self._given_source or self._source_from_cfg()
            source.open()
        except BaseException:
            depth.close()
            raise
        self._source, self.depth = source, depth
        self._last_pair = None
        self._last_evidence = self._last_t = -math.inf
        _log.info(
            "stereo camera open: calibration %s, depth by %s",
            self.calibration_id,
            backend,
        )

    def close(self) -> None:
        source, self._source = self._source, None
        depth, self.depth = self.depth, None
        try:
            if source is not None:
                source.close()
        finally:
            if depth is not None:
                depth.close()

    def _grab(self) -> Any:
        """The next stereo pair as a camera frame with depth; CameraError."""
        import cv2
        import numpy as np
        from apai.types import Frame

        source, depth = self._source, self.depth
        if source is None or depth is None:
            raise CameraError("the stereo camera is not open")
        timeout_s = float(self.cfg.get("frame_timeout_s", 10))
        max_age_s = float(self.cfg.get("max_frame_age_s", 3))
        deadline = time.monotonic() + timeout_s
        while True:
            left_s = deadline - time.monotonic()
            if left_s <= 0:
                why = getattr(source, "last_none", "")
                raise CameraError(
                    f"no new stereo pair within {timeout_s:g} s"
                    + (f": {why}" if why else "")
                )
            pair = source.next_pair(left_s)
            if pair is None:
                time.sleep(min(_NONE_PAUSE_S, max(0.0, deadline - time.monotonic())))
                continue
            self._check_pair(pair, depth)
            stamp_pair(self._link, pair, self._monotonic)  # refuse before the work
            # Only a pair with a believable time moves the mark the next is held to.
            self._last_pair = (pair.session_id, pair.seq)
            self._last_evidence = min(pair.sof_wall_ns)
            images = [
                cv2.cvtColor(eye, cv2.COLOR_YUV2BGR_NV12)
                for eye in (pair.left, pair.right)
            ]
            bgr, depth_m = depth.compute(images)
            stamp = stamp_pair(self._link, pair, self._monotonic)
            if self._monotonic() - stamp.t > max_age_s:
                self.dropped_old += 1
                continue
            if not stamp.t > self._last_t:
                raise CameraError(
                    "the camera frame's time on this machine's clock did not grow: "
                    f"{stamp.t:.6f} s after {self._last_t:.6f} s"
                )
            matcher = getattr(depth, "matcher", None)
            manifest = getattr(depth, "manifest", None) or {}
            capture = {
                "backend": LINK_STEREO,
                "session_id": pair.session_id,
                "seq": pair.seq,
                "evidence_wall_s": stamp.evidence_wall_s,
                "capture_thor_mono_s": stamp.capture_thor_mono_s,
                "receipt_monotonic_s": float(pair.receiver_rx_mono_s),
                "clock_raw_age_s": stamp.raw_age_s,
                "sensor_ids": list(pair.device_index),
                "sensor_timestamps_ns": list(pair.sof_wall_ns),
                "check": dict(pair.check),
                # True: the source made the times up; nothing was captured then.
                "invented_times": pair.check.get("invented_times") is True,
                "calibration_id": self.calibration_id,
                "calibration_status": manifest.get("status"),
                "absolute_accuracy_verified": False,
                "depth_backend": getattr(depth, "backend", None),
                "depth_parameters": getattr(matcher, "metadata", {}),
                "depth_timings_ms": dict(getattr(matcher, "last_timings_ms", {})),
                "valid_fraction": float((depth_m > 0).mean()),
            }
            self._last_t = stamp.t
            self.delivered += 1
            # Before the camera frame can reach the grid: a permit looks the
            # evidence time up by exactly this float.
            self._frame_times.note(stamp.t, stamp.evidence_wall_s)
            return Frame(
                rgb=bgr,
                depth_m=depth_m,
                K=np.array(depth.K, dtype=np.float64),
                t=stamp.t,
                observed_t=stamp.t,
                depth_source="sensor",
                capture=capture,
            )

    def _check_pair(self, pair: StereoPair, depth: Any) -> None:
        key = (pair.session_id, pair.seq)
        evidence = min(pair.sof_wall_ns)
        last = self._last_pair
        if last is not None and last[0] == key[0]:
            if key[1] == last[1]:
                raise CameraError(f"the source gave stereo pair {key} twice")
            if not evidence > self._last_evidence:
                raise CameraError(
                    f"the capture time of stereo pair {key} did not grow: "
                    f"{evidence} ns after {self._last_evidence} ns"
                )
        manifest = getattr(depth, "manifest", None) or {}
        order = manifest.get("camera_order")
        if order:
            devices = (order["left"]["argus_device"], order["right"]["argus_device"])
            rotation = manifest.get("raw_rotation_degrees", 0)
            if tuple(pair.device_index) != devices or pair.rotation != rotation:
                raise CameraError(
                    f"stereo pair {key} is of devices {tuple(pair.device_index)} "
                    f"rotated {pair.rotation}; the calibration is for {devices} "
                    f"rotated {rotation}"
                )

    def _source_from_cfg(self) -> PairSource:
        """The source a camera profile names, when none was handed in."""
        recorded = self.cfg.get("recorded_dir")
        if recorded:
            return RecordedPairSource(
                recorded,
                thor_wall_now=lambda: self._link.thor_now()[1],
                monotonic=self._monotonic,
            )
        bridge = self.cfg.get("bridge")
        if bridge:
            options = bridge.as_dict() if hasattr(bridge, "as_dict") else dict(bridge)
            for name in ("streams", "devices"):
                if name in options:
                    options[name] = tuple(options[name])
            return BridgeSource(**options)
        raise CameraError(
            f"a {CAMERA_TYPE} camera needs a stereo pair source: none was handed "
            "in and the camera profile has neither bridge nor recorded_dir"
        )


_class_lock = threading.Lock()
_camera_class: type | None = None


def _camera_type() -> type:
    global _camera_class
    with _class_lock:
        if _camera_class is None:
            from apai.perception.camera_base import CameraBase

            class LinkStereoCamera(_LinkStereoInput, CameraBase):
                """APAI's camera on stereo pairs from the camera connection.

                LinkStereoCamera(cfg, *, link, frame_times, source=None,
                depth=None, monotonic=time.monotonic). _grab() runs on the
                thread of APAI's camera stream.
                """

            LinkStereoCamera.__module__ = __name__
            _camera_class = LinkStereoCamera
        return _camera_class


def __getattr__(name: str) -> Any:
    if name == "LinkStereoCamera":
        return _camera_type()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_camera(
    link: AgentLink,
    frame_times: FrameTimes,
    source: PairSource | None = None,
    *,
    depth: DepthFactory | None = None,
) -> None:
    """Make APAI's camera factory answer type "link_stereo" with LinkStereoCamera.

    Calling it again does not wrap the factory twice; the newest arguments are
    the ones the next camera is built with.
    """
    from apai.perception import camera_base

    previous = camera_base.make_camera
    binding = {"link": link, "frame_times": frame_times, "source": source}
    binding["depth"] = depth
    registered = getattr(previous, "_link_stereo", None)
    if registered is not None:
        registered.update(binding)
        return

    def make_camera(cfg: Any) -> Any:
        if cfg.type == CAMERA_TYPE:
            return _camera_type()(cfg, **binding)
        return previous(cfg)

    make_camera._link_stereo = binding
    camera_base.make_camera = make_camera
