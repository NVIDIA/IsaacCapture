# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The camera APAI gets on the workstation: stereo pairs from the camera connection.

LinkStereoCamera turns each stereo pair of a PairSource into one camera frame
with depth, as the experiment's camera (thor_camera.ThorStereoCamera) does on
the Thor: the same depth class, the same Frame fields, and no refusal but its
own (an arrival stamp that is no past time, a time that does not grow, a pair
older than max_frame_age_s after depth, no pair within frame_timeout_s).

Frame.t is a time of this machine's monotonic clock, which every age check of
APAI reads. While the robot link can date the pair, it is the capture time (the
start of exposure), so those ages count encoding, the network, decoding and
depth; the capture time on the Thor's wall clock, the evidence time, is then
noted in FrameTimes for the permits. While the link cannot, Frame.t is the
pair's arrival, as on the Thor, and the camera frame carries no evidence time:
it is seen, and neither a PERMIT nor the self mask can rest on it.

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

from ..link_client import LinkLost, StaleState
from ..self_mask import LINK_STEREO, experiment_module
from .frame_source import BridgeSource, CameraError, RecordedPairSource

if TYPE_CHECKING:
    from ..checkahead import FrameTimes
    from ..link_client import AgentLink
    from .frame_source import PairSource, StereoPair

_log = logging.getLogger("isaaccapture.examples.agentic_remote.stereo_camera")

# The camera profile type the factory answers with this camera.
CAMERA_TYPE = "link_stereo"
# The three rules below decide whether a camera frame gets an evidence time (its
# capture time on the Thor's wall clock, which a PERMIT quotes) and a mask time.
# None of them refuses the camera frame.
# The newest ROBOT_STATE is at most this old: its clock pair dates the capture.
STATE_MAX_AGE_S = 1.0
# The capture time lies at most this far ahead of the estimate of the Thor's
# wall clock, and at most this far, plus the link's clock margin, after the
# arrival.
FUTURE_LIMIT_S = 0.05
# The Thor's wall clock did not step within this long.
CLOCK_STEP_QUIET_S = 3.0
_NONE_PAUSE_S = 0.01

DepthFactory = Callable[[Path, str, Any], Any]


@dataclass(frozen=True)
class FrameStamp:
    """The times of one camera frame.

    Dated (problem is empty): t is the capture time. Undated: problem says why
    the link could not date the capture, the two Thor times are None and t is
    the pair's arrival.
    """

    # Capture time, the Thor's wall clock; a PERMIT quotes it.
    evidence_wall_s: float | None
    capture_thor_mono_s: float | None  # the same on the Thor's monotonic clock
    t: float  # Frame.t, this machine's monotonic clock
    # The Thor's wall clock now minus the capture time, no margin.
    raw_age_s: float | None
    problem: str = ""


def stamp_pair(
    link: AgentLink,
    pair: StereoPair,
    monotonic: Callable[[], float] = time.monotonic,
) -> FrameStamp:
    """The times of a stereo pair, dated when the link's clocks allow it.

    CameraError only by the rule of the experiment's camera: an arrival stamp
    that is no past time of this machine's clock. `monotonic` must be the clock
    the link was built with and the clock of the pair's arrival stamp.
    """
    evidence = min(pair.sof_wall_ns) / 1e9
    arrival = float(pair.receiver_rx_mono_s)
    raw_age: float | None = None

    def undated(problem: str) -> FrameStamp:
        return FrameStamp(None, None, arrival, raw_age, problem)

    try:
        state_at: float | None = link.state_arrived_s()
        step_at = link.clock_step_at_s()
        missing = ""
    except (StaleState, LinkLost) as problem:
        state_at, step_at, missing = None, None, str(problem)
    now = monotonic()
    if not math.isfinite(arrival) or arrival > now:
        raise CameraError(
            f"the stereo pair's arrival stamp ({arrival!r}) is not a past time of "
            "this machine's monotonic clock"
        )
    if state_at is None:
        return undated(missing)
    try:
        raw_age = float(link.thor_now()[1] - evidence)
        margin = link.wall_age_s(evidence) - raw_age
        thor_mono = link.wall_to_thor_mono(evidence)
    except (StaleState, LinkLost) as problem:
        return undated(str(problem))
    converted = now - raw_age - margin
    state_age = now - state_at
    if state_age > STATE_MAX_AGE_S:
        return undated(
            f"the newest ROBOT_STATE is {state_age:.2f} s old, over "
            f"{STATE_MAX_AGE_S:g} s"
        )
    if step_at is not None and now - step_at < CLOCK_STEP_QUIET_S:
        return undated(
            f"the Thor's wall clock stepped {now - step_at:.2f} s ago, within "
            f"{CLOCK_STEP_QUIET_S:g} s"
        )
    if not raw_age >= -FUTURE_LIMIT_S:
        return undated(
            f"the capture time lies {-raw_age:.3f} s in the future of the Thor's "
            f"wall clock, over {FUTURE_LIMIT_S:g} s"
        )
    # The capture time minus the arrival, both on the Thor's wall clock.
    lead = (now - arrival) - raw_age
    if lead > FUTURE_LIMIT_S + margin:
        return undated(
            f"the capture time lies {lead:.3f} s after the arrival, over "
            f"{FUTURE_LIMIT_S + margin:g} s"
        )
    # The evidence is never newer than the arrival.
    late = max(0.0, lead)
    return FrameStamp(
        evidence_wall_s=evidence - late,
        capture_thor_mono_s=float(thor_mono) - late,
        t=float(min(converted, arrival)),
        raw_age_s=raw_age,
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
        # (camera session, capture time in ns) of the last dated camera frame.
        self._last_dated: tuple[int, int] | None = None
        self._undated_now = False  # the last camera frame had no evidence time
        self._last_t = -math.inf
        self.delivered = 0  # camera frames handed to APAI
        self.undated = 0  # of those, the ones without an evidence time
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
        self._last_pair = self._last_dated = None
        self._undated_now = False
        self._last_t = -math.inf
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
            # Before the work: an arrival stamp that is no past time is refused.
            stamp_pair(self._link, pair, self._monotonic)
            self._last_pair = (pair.session_id, pair.seq)
            images = [
                cv2.cvtColor(eye, cv2.COLOR_YUV2BGR_NV12)
                for eye in (pair.left, pair.right)
            ]
            bgr, depth_m = depth.compute(images)
            stamp = self._stamp(pair)
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
                # Why the two times above are None; None when they are there.
                "time_problem": stamp.problem or None,
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
            if stamp.evidence_wall_s is None:
                if not self._undated_now:
                    _log.warning(
                        "camera frames carry no evidence time: %s; no PERMIT and "
                        "no self mask can rest on them",
                        stamp.problem,
                    )
                self._undated_now = True
                self.undated += 1
            else:
                if self._undated_now:
                    _log.info("camera frames carry an evidence time again")
                self._undated_now = False
                self._last_dated = (pair.session_id, min(pair.sof_wall_ns))
                # Before the camera frame can reach the grid: a permit looks
                # the evidence time up by exactly this float.
                self._frame_times.note(stamp.t, stamp.evidence_wall_s)
            return Frame(
                rgb=bgr,
                depth_m=depth_m,
                K=np.array(depth.K),
                t=stamp.t,
                observed_t=stamp.t,
                depth_source="sensor",
                capture=capture,
            )

    def _stamp(self, pair: StereoPair) -> FrameStamp:
        """stamp_pair(); undated too when the capture time went back."""
        stamp = stamp_pair(self._link, pair, self._monotonic)
        last = self._last_dated
        evidence = min(pair.sof_wall_ns)
        if (
            stamp.evidence_wall_s is not None
            and last is not None
            and last[0] == pair.session_id
            and not evidence > last[1]
        ):
            return FrameStamp(
                None,
                None,
                float(pair.receiver_rx_mono_s),
                stamp.raw_age_s,
                f"the capture time did not grow: {evidence} ns after {last[1]} ns",
            )
        return stamp

    def _check_pair(self, pair: StereoPair, depth: Any) -> None:
        key = (pair.session_id, pair.seq)
        if key == self._last_pair:
            raise CameraError(f"the source gave stereo pair {key} twice")
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
