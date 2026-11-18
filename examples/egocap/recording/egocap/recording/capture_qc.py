# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture-quality checks over a sealed, recorded episode, and the keep/flag/reject decision they imply.

This is separate from ``dataset_check`` (storage integrity: is the episode structurally valid?). Here the
question is whether the capture itself was good enough, judged only from the recorded timestamps and values.
Each check is opt-in: it runs only for the streams the configuration declares, and a declaration that does not
match the recorded sources is a configuration error before recording, never a silent skip.

Gates that reject the take (a camera problem):

* ``camera_min_fps``: each camera must average at least ``camera_min_fps_ratio`` of its expected record rate.
* ``camera_frame_parity`` (opt-in): the listed cameras' frame counts may differ by at most a configured amount.

Gates that keep the data but flag the episode bad (a tracking problem):

* ``sustained_freeze``: a pose stream whose per-frame change stays at or below ``freeze_eps`` for at least
  ``freeze_seconds``.
* ``avatar_hand_liveness``: each hand of an Avatar glove needs at least ``avatar_min_states`` rows and a stream
  that is not frozen for the whole episode (a dead or disconnected glove).
* ``pico_body_joints``: Pico full-body tracking must have the expected shape, finite values, and no more than the
  allowed invalid or all-identity frames.

Advisory (typed results and warnings only, never deciding): ``flat_motion``, ``dead_stream`` and ``frame_drops``.

Nothing here imports capture, acquisition or device code.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import zarr

from .capture_measure import (
    average_fps,
    longest_still_seconds,
    pico_body_problem,
    stream_liveness,
    total_frame_drops,
    windowed_frame_drops,
)
from .dataset_check import CheckResult, Outcome, QualityReport, report_to_dict
from .episode_layout import FINALIZATION_KEEP, FINALIZATION_REMOVE, FINALIZATION_RETAIN
from .episode_store import FieldSpec, NumericSourceSpec, VideoSourceSpec

__all__ = [
    "ALGORITHM_VERSION",
    "CaptureQcConfig",
    "PicoBodyConfig",
    "QcDecision",
    "check_capture_quality",
    "decide",
    "finalization_for",
    "policy_to_dict",
    "quality_attributes",
]

#: Bump when a check's measurement or decision rule changes, so a stored decision names the rules that made it.
ALGORITHM_VERSION = 2

CAMERA_MIN_FPS = "camera_min_fps"
CAMERA_FRAME_PARITY = "camera_frame_parity"
SUSTAINED_FREEZE = "sustained_freeze"
AVATAR_HAND_LIVENESS = "avatar_hand_liveness"
PICO_BODY_JOINTS = "pico_body_joints"
FLAT_MOTION = "flat_motion"
DEAD_STREAM = "dead_stream"
FRAME_DROPS = "frame_drops"
#: Result-name prefixes that decide the outcome, by what a failure means.
CAMERA_GATES = (CAMERA_MIN_FPS, CAMERA_FRAME_PARITY)
TRACKING_GATES = (SUSTAINED_FREEZE, AVATAR_HAND_LIVENESS, PICO_BODY_JOINTS)

#: The native Avatar glove streams per hand; a hand is alive if any one of them is not frozen.
AVATAR_HAND_STREAMS = {
    "left": ("left_hand_keypoints", "left_hand_joints_robot"),
    "right": ("right_hand_keypoints", "right_hand_joints_robot"),
}


class QcDecision(Enum):
    """What capture quality means for the take (not storage integrity, not the service disposition)."""

    PASSED = "PASSED"  # capture checks ran and found no fatal problem (advisories may remain as warnings)
    SKIPPED = "SKIPPED"  # nothing was measured: capture QC is not configured, or the integrity check failed first
    ERROR = "ERROR"  # a gate could not run: the service retains the take for recovery, never as unflagged data
    FLAGGED_BAD = "FLAGGED_BAD"  # measured fatal tracking problem: the data is kept but marked bad
    REJECTED_CAMERA = "REJECTED_CAMERA"  # measured camera failure: the take is rejected


@dataclass(frozen=True)
class PicoBodyConfig:
    """Pico full-body tracking: where it is recorded and how much invalid or all-identity data is tolerated."""

    source: str = "pico_controller"
    field: str = "body_joints_pose"  # float (T, 24, 4, 4)
    valid_field: str | None = "body_joints_valid"  # per-frame validity, or None when the device records none
    max_invalid_frames: int = 0
    max_identity_frames: int = 0
    identity_atol: float = 1e-6


@dataclass(frozen=True)
class CaptureQcConfig:
    """Thresholds and the streams each check reads.

    The defaults document the values in ``data_check_config.yaml``: a camera must
    average at least 0.86 of its record rate (51.6 Hz at 60 Hz, the value tuned for the ego OAK), a pose may be
    still for at most 1.0 s (``1e-6`` tolerance), and 2.0 s at ``1e-3`` is a flat-motion advisory. A failed camera
    gate deletes the take, so a different ratio is a deliberate choice, not a default.
    """

    # (video source, expected record rate in Hz); None means the source's own configured fps (see ``resolved``)
    camera_expected_fps: tuple[tuple[str, float | None], ...] = ()
    camera_min_fps_ratio: float = 0.86
    # Cameras whose frame counts must agree (opt-in; at least two). Off by default for the same reason it is off in
    # the native configuration: a camera with a hardware ceiling (the ego OAK, about 57-58 Hz against 60) fails
    # every take longer than about 20 s, and this gate rejects (removes) the take.
    camera_parity: tuple[str, ...] = ()
    camera_parity_max_diff: int = 50
    pose_streams: tuple[tuple[str, str], ...] = ()  # (numeric source, float field) checked for freezing
    freeze_seconds: float = 1.0
    freeze_eps: float = 1e-6
    flat_seconds: float = 2.0
    flat_eps: float = 1e-3
    avatar_hand_sources: tuple[str, ...] = ()  # numeric sources that record an Avatar glove's hand streams
    avatar_min_states: int = 2
    dead_std_eps: float = 1e-6
    dead_frac_identical: float = 0.95
    pico_body: PicoBodyConfig | None = None
    max_dropped_frames: int = 6  # advisory: per camera in camera_expected_fps, within any drop_window_s
    drop_window_s: float = 2.0

    def resolved(self, video: tuple[VideoSourceSpec, ...]) -> CaptureQcConfig:
        """This policy with every default camera rate replaced by that source's own configured fps.

        An unknown camera keeps its ``None`` so :meth:`problems` reports it. The service resolves once, so the
        policy stored with a take (and the configuration digest) always holds concrete rates.
        """
        own = {v.name: v.fps for v in video}
        rates = tuple((name, own.get(name) if fps is None else fps) for name, fps in self.camera_expected_fps)
        return replace(self, camera_expected_fps=rates)

    def problems(self, numeric: tuple[NumericSourceSpec, ...], video: tuple[VideoSourceSpec, ...]) -> list[str]:
        """Everything wrong with this configuration against the recorded sources (empty when usable)."""
        problems: list[str] = []
        if not (
            self.camera_expected_fps
            or self.camera_parity
            or self.pose_streams
            or self.avatar_hand_sources
            or self.pico_body
        ):
            problems.append("capture QC names no camera, parity, pose stream, Avatar hand or Pico body check")
        video_names = {v.name for v in video}
        fields = {s.name: {f.name: f for f in s.fields} for s in numeric}
        for name, fps in self.camera_expected_fps:
            if name not in video_names:
                problems.append(f"camera_expected_fps names unknown video source {name!r}")
            if fps is not None and not 0 < fps < math.inf:
                problems.append(f"camera_expected_fps for {name!r} must be a finite number > 0")
        if self.camera_parity:
            unknown = sorted(set(self.camera_parity) - video_names)
            if len(self.camera_parity) < 2 or len(set(self.camera_parity)) != len(self.camera_parity) or unknown:
                problems.append(f"camera_parity needs two or more distinct video sources (unknown: {unknown})")
            if not isinstance(self.camera_parity_max_diff, int) or self.camera_parity_max_diff < 0:
                problems.append("camera_parity_max_diff must be an integer >= 0")
        for source, field in self.pose_streams:
            if source not in fields:
                problems.append(f"pose_streams names unknown numeric source {source!r}")
            elif field not in fields[source] or np.dtype(fields[source][field].dtype).kind != "f":
                problems.append(f"pose_streams {source}/{field} is not a float field")
        for source in self.avatar_hand_sources:
            have = fields.get(source, {})
            present = [side for side, names in AVATAR_HAND_STREAMS.items() if any(f in have for f in names)]
            if not present:
                problems.append(f"avatar_hand_sources {source!r} is not a numeric source with Avatar hand fields")
            elif len(present) < len(AVATAR_HAND_STREAMS):  # a missing side would flag every take bad
                problems.append(f"avatar_hand_sources {source!r} records only the {present} hand; both are needed")
        if self.pico_body is not None:
            problems += self._pico_problems(fields)
        if not 0 < self.camera_min_fps_ratio <= 1:
            problems.append("camera_min_fps_ratio must be > 0 and <= 1")
        if not 0 < self.dead_frac_identical <= 1:
            problems.append("dead_frac_identical must be > 0 and <= 1")
        for label in ("freeze_seconds", "freeze_eps", "flat_seconds", "flat_eps", "dead_std_eps", "drop_window_s"):
            if not 0 < getattr(self, label) < math.inf:
                problems.append(f"{label} must be a finite number > 0")
        if not isinstance(self.avatar_min_states, int) or self.avatar_min_states < 1:
            problems.append("avatar_min_states must be an integer >= 1")
        if not isinstance(self.max_dropped_frames, int) or self.max_dropped_frames < 0:
            problems.append("max_dropped_frames must be an integer >= 0")
        return problems

    def _pico_problems(self, fields: dict[str, dict[str, FieldSpec]]) -> list[str]:
        """Problems with the Pico body declaration: a float (24, 4, 4) pose field, a per-frame validity field."""
        body = self.pico_body
        problems = []
        if body.source not in fields:
            return [f"pico_body names unknown numeric source {body.source!r}"]
        pose = fields[body.source].get(body.field)
        if pose is None or np.dtype(pose.dtype).kind != "f" or tuple(pose.shape) != (24, 4, 4):
            problems.append(f"pico_body {body.source}/{body.field} must be a float field of shape (24, 4, 4)")
        if body.valid_field is not None:
            valid = fields[body.source].get(body.valid_field)
            if valid is None or tuple(valid.shape) != () or np.dtype(valid.dtype).kind not in "biu":
                problems.append(f"pico_body validity {body.source}/{body.valid_field} must be a per-frame flag")
        for label in ("max_invalid_frames", "max_identity_frames"):
            if not isinstance(getattr(body, label), int) or getattr(body, label) < 0:
                problems.append(f"pico_body {label} must be an integer >= 0")
        if not 0 <= body.identity_atol < math.inf:
            problems.append("pico_body identity_atol must be a finite number >= 0")
        return problems


def policy_to_dict(config: CaptureQcConfig) -> dict:
    """The concrete, serializable policy (thresholds, mappings and algorithm version) stored with each take."""
    return {"algorithm_version": ALGORITHM_VERSION, **asdict(config)}


def _measure(name: str, measure: Callable[[], tuple[list[str], str, dict]]) -> CheckResult:
    """One item's result. ``measure`` returns (issues, evidence, measurements); an exception is its own ERROR.

    Every camera, stream and hand is measured separately, so a measured failure on one can never be erased by a
    read error on another.
    """
    try:
        issues, evidence, measurements = measure()
    except Exception as exc:  # noqa: BLE001 - a check that cannot run is reported, not raised
        return CheckResult(name, Outcome.ERROR, (f"{name} could not run: {type(exc).__name__}: {exc}",))
    return CheckResult(
        name, Outcome.FAILED if issues else Outcome.PASSED, tuple(issues), (evidence,), measurements
    )


def check_capture_quality(pending_dir: Path, config: CaptureQcConfig) -> QualityReport:
    """Run the configured capture checks on a sealed episode. Never raises: each item reports its own result.

    Gating result names are ``<check>:<item>`` (see the module docstring); advisory results keep their typed
    outcome, evidence and measurements, and their findings become the report's warnings.
    """
    try:
        root = zarr.open_group(str(pending_dir), mode="r")
    except Exception as exc:  # noqa: BLE001 - every item is then unmeasured, so every gate is an ERROR
        names = [
            *[f"{CAMERA_MIN_FPS}:{name}" for name, _ in config.camera_expected_fps],
            *([CAMERA_FRAME_PARITY] if config.camera_parity else []),
            *[f"{SUSTAINED_FREEZE}:{s}/{f}" for s, f in config.pose_streams],
            *[
                f"{AVATAR_HAND_LIVENESS}:{s}/{side}"
                for s in config.avatar_hand_sources
                for side in AVATAR_HAND_STREAMS
            ],
            *([f"{PICO_BODY_JOINTS}:{config.pico_body.source}"] if config.pico_body else []),
        ]
        errors = tuple(
            CheckResult(n, Outcome.ERROR, (f"{n}: cannot open the sealed episode: {exc}",)) for n in names
        )
        return QualityReport(Outcome.ERROR, tuple(i for r in errors for i in r.issues), (), errors)

    def times(source: str) -> np.ndarray:
        return root[source]["capture_time"][:]

    def camera_min_fps(name: str, expected: float | None):
        if expected is None:
            raise ValueError("the expected rate is unresolved; resolve the policy with CaptureQcConfig.resolved()")
        fps, minimum = average_fps(times(name)), config.camera_min_fps_ratio * expected
        issue = (
            f"{name}: {fps:.2f} fps average is below {config.camera_min_fps_ratio} x {expected} = {minimum:.2f}"
        )
        return ([issue] if fps < minimum else []), f"{fps:.2f} fps average (minimum {minimum:.2f})", {"fps": fps}

    def camera_frame_parity():
        counts = {name: len(times(name)) for name in config.camera_parity}
        spread = max(counts.values()) - min(counts.values())
        shown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        issue = f"camera frame-count spread {spread} exceeds {config.camera_parity_max_diff}: {shown}"
        return (
            ([issue] if spread > config.camera_parity_max_diff else []),
            f"spread {spread} ({shown})",
            dict(counts),
        )

    def still(source: str, field: str, seconds: float, eps: float):
        longest = longest_still_seconds(times(source), root[source][field][:], eps)
        issue = f"{source}/{field}: no motion for {longest:.2f}s (limit {seconds}s, eps {eps})"
        return (
            ([issue] if longest >= seconds else []),
            f"longest still window {longest:.2f}s (limit {seconds}s)",
            {"longest_still_s": longest},
        )

    def avatar_hand(source: str, side: str):
        group, streams = root[source], [s for s in AVATAR_HAND_STREAMS[side] if s in root[source]]
        rows, live = 0, False
        for name in streams:
            arr = group[name][:]
            rows = max(rows, arr.shape[0])
            live = live or (
                arr.shape[0] >= config.avatar_min_states
                and not stream_liveness(arr, config.dead_std_eps, config.dead_frac_identical)["frozen"]
            )
        issue = f"{source} {side} hand: 0 live states (dead or disconnected glove)"
        evidence = f"streams {streams or 'none recorded'}, {rows} rows, live {live}"
        return ([] if streams and rows >= config.avatar_min_states and live else [issue]), evidence, {"rows": rows}

    def pico_body():
        body = config.pico_body
        group = root[body.source]
        valid = group[body.valid_field][:] if body.valid_field else None
        problem = pico_body_problem(
            group[body.field][:], valid, body.max_invalid_frames, body.max_identity_frames, body.identity_atol
        )
        return ([f"{body.source}: {problem}"] if problem else []), problem or "body-joint tracking valid", {}

    def dead_stream(source: str, field: str):
        live = stream_liveness(root[source][field][:], config.dead_std_eps, config.dead_frac_identical)
        issue = f"{source}/{field}: frozen for the whole episode" + (
            " at identity (sensor dropout)" if live["at_identity"] else ""
        )
        return (
            ([issue] if live["frozen"] else []),
            f"std {live['std_mean']:.3g}, identical {live['frac_identical']}",
            {
                "std_mean": live["std_mean"],
                "frac_identical": live["frac_identical"],
            },
        )

    def drops(name: str, expected: float):
        stamps = times(name)
        window = windowed_frame_drops(stamps, expected, config.drop_window_s)
        total = total_frame_drops(stamps, expected)
        issue = f"{name}: {window} dropped frames in {config.drop_window_s:.0f}s (> {config.max_dropped_frames})"
        measured = {
            "window_max": window,
            "total": total,
            "window_s": config.drop_window_s,
            "expected_fps": expected,
        }
        return (
            ([issue] if window > config.max_dropped_frames else []),
            f"{window} in window, {total} total",
            measured,
        )

    results = [
        _measure(f"{CAMERA_MIN_FPS}:{n}", lambda n=n, e=e: camera_min_fps(n, e))
        for n, e in config.camera_expected_fps
    ]
    if config.camera_parity:
        results.append(_measure(CAMERA_FRAME_PARITY, camera_frame_parity))
    results += [
        _measure(
            f"{SUSTAINED_FREEZE}:{s}/{f}", lambda s=s, f=f: still(s, f, config.freeze_seconds, config.freeze_eps)
        )
        for s, f in config.pose_streams
    ]
    results += [
        _measure(f"{AVATAR_HAND_LIVENESS}:{s}/{side}", lambda s=s, side=side: avatar_hand(s, side))
        for s in config.avatar_hand_sources
        for side in AVATAR_HAND_STREAMS
    ]
    if config.pico_body:
        results.append(_measure(f"{PICO_BODY_JOINTS}:{config.pico_body.source}", pico_body))
    advisories = [
        _measure(f"{FLAT_MOTION}:{s}/{f}", lambda s=s, f=f: still(s, f, config.flat_seconds, config.flat_eps))
        for s, f in config.pose_streams
    ]
    advisories += [
        _measure(f"{DEAD_STREAM}:{s}/{f}", lambda s=s, f=f: dead_stream(s, f)) for s, f in config.pose_streams
    ]
    advisories += [
        _measure(f"{FRAME_DROPS}:{n}", lambda n=n, e=e: drops(n, e)) for n, e in config.camera_expected_fps
    ]
    warnings = tuple(i for r in advisories if r.outcome is not Outcome.PASSED for i in r.issues)
    failed = [r for r in results if r.outcome is Outcome.FAILED]
    overall = (
        Outcome.ERROR
        if any(r.outcome is Outcome.ERROR for r in results)
        else Outcome.FAILED
        if failed
        else Outcome.PASSED
    )
    issues = tuple(i for r in results if r.outcome is not Outcome.PASSED for i in r.issues)
    return QualityReport(overall, issues, warnings, tuple(results + advisories))


def decide(report: QualityReport | None) -> QcDecision:
    """The keep/flag/reject decision for a capture report (``None`` means capture QC was not configured).

    Only a measured camera failure outranks known bad tracking, so the order is:
    camera FAILED (reject) > tracking FAILED (keep, flag bad) > a gate that could not run (ERROR: the service
    retains the take for recovery rather than keeping it as checked, with the per-check results showing which one)
    > PASSED. A gate that could not be measured therefore never makes a measured failure of another look
    unflagged. Advisory results are ignored.
    """
    if report is None:
        return QcDecision.SKIPPED
    outcomes = {"camera": set(), "tracking": set()}
    for r in report.check_results:
        kind = r.name.split(":", 1)[0]
        if kind in CAMERA_GATES:
            outcomes["camera"].add(r.outcome)
        elif kind in TRACKING_GATES:
            outcomes["tracking"].add(r.outcome)
    camera, tracking = outcomes["camera"], outcomes["tracking"]
    if Outcome.FAILED in camera:
        return QcDecision.REJECTED_CAMERA
    if Outcome.FAILED in tracking:
        return QcDecision.FLAGGED_BAD
    if Outcome.ERROR in camera or Outcome.ERROR in tracking:
        return QcDecision.ERROR
    return QcDecision.PASSED


def finalization_for(integrity: Outcome, decision: QcDecision) -> str:
    """What finalization does with a checked take: failed integrity retains it, a camera rejection removes it."""
    if integrity is not Outcome.PASSED:
        return FINALIZATION_RETAIN
    return FINALIZATION_REMOVE if decision is QcDecision.REJECTED_CAMERA else FINALIZATION_KEEP


def quality_attributes(decision: QcDecision, report: QualityReport | None, policy: dict | None) -> dict:
    """The dataset-level metadata for a kept episode's capture quality, in the names the native readers use.

    ``domain_checks`` says whether capture QC ran; ``bad_episode`` marks kept data it flagged. The native
    ``freeze_check_passed``/``freeze_check_issues`` are the episode's overall tracking verdict, as in the native
    recorder: every tracking gate (freeze, Avatar hand liveness, Pico body joints) counts, ``False`` with the
    failed gates' issues if any failed, ``True`` only if every tracking gate that ran passed. Nothing is written
    when no tracking gate established a verdict (a camera-only policy, or tracking checks that could not run), so
    it never reads as passed. Per-camera drop figures are the native ``frame_drops`` evidence.
    """
    attributes = {
        "domain_checks": "SKIPPED" if decision is QcDecision.SKIPPED else "EXECUTED",
        "capture_qc_decision": decision.value,
        "capture_qc_issues": list(report.issues) if report else [],
        "capture_qc_warnings": list(report.warnings) if report else [],
        "capture_qc_results": report_to_dict(report)["check_results"] if report else [],
        "capture_qc_policy": policy,
        "bad_episode": decision is QcDecision.FLAGGED_BAD,
    }
    if report is None:
        return attributes
    tracking = [r for r in report.check_results if r.name.split(":", 1)[0] in TRACKING_GATES]
    failed = [r for r in tracking if r.outcome is Outcome.FAILED]
    if failed or (tracking and all(r.outcome is Outcome.PASSED for r in tracking)):
        attributes["freeze_check_passed"] = not failed
        attributes["freeze_check_issues"] = [issue for r in failed for issue in r.issues]
    attributes["quality_warnings"] = list(report.warnings)
    drops = {
        r.name.split(":", 1)[1]: r.measurements
        for r in report.check_results
        if r.name.startswith(f"{FRAME_DROPS}:") and r.measurements
    }
    if drops:
        attributes["frame_drops"] = drops
    return attributes
