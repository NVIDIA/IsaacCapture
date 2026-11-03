# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture user-specific RealHand FFG Glove calibration through DeviceIO."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_trackers import PluginDeviceStatusTracker
from isaaccapture.retargeting_engine.deviceio_source_nodes import JointStateSource
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)

_COLLECTIONS = {"left": "realhand_ffg_glove_left", "right": "realhand_ffg_glove_right"}
_SENSOR_NAMES = tuple(f"sensor_{index}" for index in range(21))
_FORMAT = "realhand-ffg-glove-human-calibration-v1"
_DEFAULT_PREPARE_SECONDS = 8
_DEFAULT_HOLD_SECONDS = 6.0
_DEFAULT_SAMPLES = 120
# CLI and PC previews share the same device calibration, independent of robot model.
_POSES = (
    (
        "open palm: straighten all digits with the four fingers together and thumb pointing sideways",
        "open",
    ),
    ("full fist: curl the four fingers fully; keep the thumb outside the fist", "fist"),
    (
        "thumb-only curl: bend the thumb MCP and IP fully; keep the other digits straight",
        "thumb_curl",
    ),
    (
        "thumb opposition: sweep the thumb across the palm toward the little-finger base; keep fingers straight",
        "thumb_opposition",
    ),
    (
        "finger spread: keep the fingers straight and spread them sideways as far as comfortable",
        "spread",
    ),
)
_CONTACTS = ("index", "middle", "ring", "little")
_CAPTURE_POSES = (
    *_POSES,
    *(
        (f"touch the thumb tip lightly to the {finger} fingertip", f"thumb_{finger}")
        for finger in _CONTACTS
    ),
    *(
        (
            f"release, then repeat thumb to {finger} fingertip contact",
            f"repeat_thumb_{finger}",
        )
        for finger in _CONTACTS
    ),
    (_POSES[0][0], "open_repeat"),
)


def _build_raw_pipeline(sides: tuple[str, ...]):
    outputs = {}
    for side in sides:
        source = JointStateSource(
            name=f"realhand_ffg_glove_{side}",
            collection_id=_COLLECTIONS[side],
            joint_names=list(_SENSOR_NAMES),
        )
        outputs[side] = source.output(JointStateSource.JOINTS)
    return OutputCombiner(outputs)


def _sample_group(result, side: str) -> np.ndarray | None:
    group = result[side]
    if group.is_none:
        return None
    values = np.asarray([float(group[index]) for index in range(len(_SENSOR_NAMES))])
    if not np.all(np.isfinite(values)):
        return None
    return values


def _wait_for_sides(
    session: TeleopSession, sides: tuple[str, ...], timeout: float, status_tracker=None
) -> None:
    pending = set(sides)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        result = session.step()
        for side in tuple(pending):
            if _current_sample(session, result, side, status_tracker) is not None:
                pending.remove(side)
        time.sleep(0.02)
    if pending:
        names = ", ".join(sorted(pending))
        raise RuntimeError(f"No RealHand FFG Glove data received for: {names}")


def _current_sample(session, result, side, status_tracker):
    if status_tracker is not None:
        report = status_tracker.get_device_status_snapshot(session.deviceio_session)
        if (
            report is None
            or report.schema_version != 1
            or not 0 <= time.monotonic_ns() - report.report_time_ns < 2_000_000_000
            or not any(
                device.path == f"/glove/realhand_ffg_glove_{side}"
                and device.state.name == "CONNECTED"
                for device in report.devices
            )
        ):
            return None
    return _sample_group(result, side)


def _capture_pose(
    session: TeleopSession,
    sides: tuple[str, ...],
    sample_count: int,
    timeout: float,
    *,
    hold_seconds: float | None = None,
    status_tracker=None,
    return_samples: bool = False,
) -> dict:
    samples: dict[str, list[np.ndarray]] = {side: [] for side in sides}
    deadline = time.monotonic() + (timeout if hold_seconds is None else hold_seconds)
    printed = None
    while time.monotonic() < deadline and (
        hold_seconds is not None
        or any(len(samples[side]) < sample_count for side in sides)
    ):
        result = session.step()
        for side in sides:
            if hold_seconds is None and len(samples[side]) >= sample_count:
                continue
            values = _current_sample(session, result, side, status_tracker)
            if values is None and hold_seconds is not None:
                raise RuntimeError(
                    f"{side}: glove data lost during HOLD; calibration unchanged"
                )
            if values is not None:
                samples[side].append(values)
        if hold_seconds is not None:
            remaining = max(1, int(np.ceil(deadline - time.monotonic())))
            if remaining != printed:
                print(f"  HOLD: {remaining} s", flush=True)
                printed = remaining
        time.sleep(0.01)
    missing = {
        side: sample_count - len(values)
        for side, values in samples.items()
        if len(values) < sample_count
    }
    if missing:
        raise RuntimeError(f"Calibration capture timed out; missing samples: {missing}")
    if return_samples:
        return samples
    return {
        # Median raw phases across 0/2pi can invent a nonexistent half-turn pose.
        side: np.median(np.unwrap(np.asarray(values), axis=0), axis=0).tolist()
        for side, values in samples.items()
    }


def _prepare(session, seconds):
    for remaining in range(seconds, 0, -1):
        print(f"  PREPARE: {remaining} s", flush=True)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            session.step()
            time.sleep(0.01)


def _store_capture(poses, key, samples):
    values = np.asarray(samples, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1:] != (21,)
        or len(values) < 30
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Each hold needs at least 30 finite 21-angle samples")
    poses[key] = np.median(np.unwrap(values, axis=0), axis=0).tolist()
    canonical = key.removeprefix("repeat_")
    if canonical == "open_repeat":
        canonical = "open"
    if canonical in {"open", "fist", "thumb_curl", *(f"thumb_{f}" for f in _CONTACTS)}:
        indices = list(range(0, len(values), max(1, len(values) // 16)))[:16]
        poses.setdefault("thumb_observations", []).extend(
            {"pose": canonical, "values": values[i].tolist()} for i in indices
        )


def _fit_and_save(path, captured):
    from .realhand_ffg_thumb_calibration import fit_thumb_mapping

    candidate = deepcopy(_load_output(path))
    for side, poses in captured.items():
        # Replace the selected side in full: old coefficients cannot survive fresh sensor zeros.
        candidate["hands"][side] = deepcopy(poses)
        mapping = fit_thumb_mapping(poses, side)
        if (
            "open_repeat" in poses
            and mapping["fit_report"]["repeat_open_training_tip_mm"] > 15
        ):
            raise ValueError(
                f"{side}: repeated-open thumb displacement exceeds 15 mm; repeat calibration"
            )
        candidate["hands"][side]["thumb_mapping"] = mapping
    _validate_calibration(candidate, tuple(captured))
    candidate["timestamp"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(candidate, stream, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return candidate


def _validate_calibration(calibration: dict, sides: tuple[str, ...]) -> None:
    for side in sides:
        poses = calibration["hands"][side]
        for _, key in _POSES:
            values = np.asarray(poses[key], dtype=float)
            if values.shape != (21,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{side}.{key} must contain 21 finite angles")
        spans = [(0, "thumb_opposition"), (2, "thumb_curl"), (4, "thumb_curl")]
        if "thumb_mapping" in poses:
            mapping = poses["thumb_mapping"]
            minimum = np.asarray(mapping.get("input_min"), dtype=float)
            maximum = np.asarray(mapping.get("input_max"), dtype=float)
            if (
                mapping.get("format") != "realhand-ffg-thumb-quadratic-v1"
                or minimum.shape != (4,)
                or maximum.shape != (4,)
                or not np.all(np.isfinite((minimum, maximum)))
                or np.any(minimum > 0)
                or np.any(maximum < 0)
                or np.any(maximum - minimum >= np.pi)
                or np.max(maximum - minimum) < 0.05
            ):
                raise ValueError(f"{side}: invalid thumb mapping input ranges")
            # Full multivariate captures replace the legacy single-pose thumb span check.
            spans = []
        spans.extend(
            (sensor, "fist")
            for finger in range(4)
            for sensor in (6 + 4 * finger, 8 + 4 * finger)
        )
        for sensor, key in spans:
            delta = poses[key][sensor] - poses["open"][sensor]
            delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
            if abs(delta) < 0.05:
                raise ValueError(
                    f"{side} sensor_{sensor} has insufficient calibration motion; capture the poses again"
                )


def _load_output(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"format": _FORMAT, "hands": {}}
    with path.open("r", encoding="utf-8") as stream:
        existing = yaml.safe_load(stream) or {}
    if not isinstance(existing, dict):
        raise ValueError(f"Existing calibration must be a YAML mapping: {path}")
    if existing.get("format") != _FORMAT or not isinstance(existing.get("hands"), dict):
        raise ValueError(
            "Existing file is not a human device calibration; select a new output file"
        )
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--prepare-seconds", type=int, default=_DEFAULT_PREPARE_SECONDS)
    parser.add_argument("--hold-seconds", type=float, default=_DEFAULT_HOLD_SECONDS)
    parser.add_argument(
        "--plugin-path",
        type=Path,
        help="Directory containing realhand_ffg_glove/plugin.yaml",
    )
    parser.add_argument(
        "--realhand-ffg-glove-ports",
        help="Comma-separated verified glove ports; requires --plugin-path",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()
    if args.realhand_ffg_glove_ports and args.plugin_path is None:
        parser.error("--realhand-ffg-glove-ports requires --plugin-path")
    if args.samples < 30:
        parser.error("--samples must be at least 30")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if (
        args.prepare_seconds < 0
        or not np.isfinite(args.hold_seconds)
        or args.hold_seconds <= 0
    ):
        parser.error(
            "--prepare-seconds must be nonnegative and --hold-seconds finite and positive"
        )

    sides = ("left", "right") if args.side == "both" else (args.side,)
    pipeline = _build_raw_pipeline(sides)
    plugins = []
    if args.plugin_path is not None:
        plugins.append(
            PluginConfig(
                plugin_name="realhand_ffg_glove_plugin",
                plugin_root_id="realhand_ffg_glove",
                search_paths=[args.plugin_path],
                # Raw capture must not inherit a personal calibration from the manifest.
                plugin_args=["--calibration="]
                + (
                    [f"--ports={args.realhand_ffg_glove_ports}"]
                    if args.realhand_ffg_glove_ports
                    else []
                ),
                required=True,
            )
        )
    tracker = PluginDeviceStatusTracker("realhand_ffg_glove/device_status")
    config = TeleopSessionConfig(
        app_name="FFGGloveCalibration",
        trackers=[tracker],
        pipeline=pipeline,
        plugins=plugins,
    )
    _load_output(args.output)
    captured = {side: {} for side in sides}

    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        print(f"Waiting for RealHand FFG Glove stream: {', '.join(sides)}")
        _wait_for_sides(session, sides, args.timeout, tracker)
        for index, (label, key) in enumerate(_CAPTURE_POSES, 1):
            input(
                f"[{index}/{len(_CAPTURE_POSES)}] {label}. Press Enter, then release the keyboard..."
            )
            _prepare(session, args.prepare_seconds)
            samples = _capture_pose(
                session,
                sides,
                args.samples,
                args.timeout,
                hold_seconds=args.hold_seconds,
                status_tracker=tracker,
                return_samples=True,
            )
            for side, values in samples.items():
                _store_capture(captured[side], key, values)
            print(f"Captured {label}")

    _fit_and_save(args.output, captured)
    print(f"Calibration saved to {args.output}")


if __name__ == "__main__":
    main()
