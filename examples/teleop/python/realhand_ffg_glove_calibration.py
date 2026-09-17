# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture user-specific RealHand FFG Glove calibration through DeviceIO."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters.realhand import REALHAND_FFG_GLOVE_SENSOR_NAMES
from isaaccapture.retargeting_engine.deviceio_source_nodes import JointStateSource
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)

_COLLECTIONS = {"left": "realhand_ffg_glove_left", "right": "realhand_ffg_glove_right"}
_POSES = {
    "l6": (
        ("open palm", "jointangleoriginal"),
        ("full fist", "jointanglefist"),
        ("thumb-only curl", "jointanglethumb_curl"),
        ("thumb touching index fingertip", "jointangleopose_index"),
        ("thumb touching middle fingertip", "jointangleopose_middle"),
    ),
    "o6": (
        ("open palm", "jointangleoriginal"),
        ("full fist", "jointanglefist"),
        ("thumb-only curl", "jointanglethumb_curl"),
        ("thumb touching index fingertip", "jointangleopose_index"),
        ("thumb touching middle fingertip", "jointangleopose_middle"),
    ),
    "l20": (
        ("open palm", "jointangleoriginal"),
        ("full fist", "jointanglefist"),
        ("thumb-only curl", "jointanglethumb_curl"),
        ("thumb touching index fingertip", "jointangleopose_index"),
        ("thumb touching middle fingertip", "jointangleopose_middle"),
        ("thumb touching ring fingertip", "jointangleopose_ring"),
        ("thumb touching pinky fingertip", "jointangleopose_pinky"),
    ),
}


def _build_raw_pipeline(sides: tuple[str, ...]):
    outputs = {}
    for side in sides:
        source = JointStateSource(
            name=f"realhand_ffg_glove_{side}",
            collection_id=_COLLECTIONS[side],
            joint_names=list(REALHAND_FFG_GLOVE_SENSOR_NAMES),
        )
        outputs[side] = source.output(JointStateSource.JOINTS)
    return OutputCombiner(outputs)


def _sample_group(result, side: str) -> np.ndarray | None:
    group = result[side]
    if group.is_none:
        return None
    values = np.asarray(
        [float(group[index]) for index in range(len(REALHAND_FFG_GLOVE_SENSOR_NAMES))]
    )
    if not np.all(np.isfinite(values)):
        return None
    return values


def _wait_for_sides(
    session: TeleopSession, sides: tuple[str, ...], timeout: float
) -> None:
    pending = set(sides)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        result = session.step()
        for side in tuple(pending):
            if _sample_group(result, side) is not None:
                pending.remove(side)
        time.sleep(0.02)
    if pending:
        names = ", ".join(sorted(pending))
        raise RuntimeError(f"No RealHand FFG Glove data received for: {names}")


def _capture_pose(
    session: TeleopSession,
    sides: tuple[str, ...],
    sample_count: int,
    timeout: float,
) -> dict[str, list[float]]:
    samples: dict[str, list[np.ndarray]] = {side: [] for side in sides}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(
        len(samples[side]) < sample_count for side in sides
    ):
        result = session.step()
        for side in sides:
            if len(samples[side]) >= sample_count:
                continue
            values = _sample_group(result, side)
            if values is not None:
                samples[side].append(values)
        time.sleep(0.01)
    missing = {
        side: sample_count - len(values)
        for side, values in samples.items()
        if len(values) < sample_count
    }
    if missing:
        raise RuntimeError(f"Calibration capture timed out; missing samples: {missing}")
    return {
        side: np.median(np.asarray(values), axis=0).tolist()
        for side, values in samples.items()
    }


def _load_output(path: Path, model: str) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        existing = yaml.safe_load(stream) or {}
    if not isinstance(existing, dict):
        raise ValueError(f"Existing calibration must be a YAML mapping: {path}")
    existing_model = existing.get("model")
    if existing_model not in (None, model):
        raise ValueError(
            f"Existing calibration is for {existing_model!r}, not requested model {model!r}"
        )
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hand-model", choices=tuple(_POSES), required=True)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--plugin-path",
        type=Path,
        help="Directory containing realhand_ffg_glove/plugin.yaml",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()
    if args.samples < 10:
        parser.error("--samples must be at least 10")

    sides = ("left", "right") if args.side == "both" else (args.side,)
    pipeline = _build_raw_pipeline(sides)
    plugins = []
    if args.plugin_path is not None:
        plugins.append(
            PluginConfig(
                plugin_name="realhand_ffg_glove_plugin",
                plugin_root_id="realhand_ffg_glove",
                search_paths=[args.plugin_path],
                required=True,
            )
        )
    config = TeleopSessionConfig(
        app_name="FFGGloveCalibration",
        trackers=[],
        pipeline=pipeline,
        plugins=plugins,
    )
    calibration = _load_output(args.output, args.hand_model)
    calibration.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.hand_model,
            "format": "realhand-ffg-glove-raw-calibration-v2",
        }
    )

    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        print(f"Waiting for RealHand FFG Glove stream: {', '.join(sides)}")
        _wait_for_sides(session, sides, args.timeout)
        for label, key in _POSES[args.hand_model]:
            input(f"Hold {label}. Press Enter when ready, then remain still...")
            captured = _capture_pose(session, sides, args.samples, args.timeout)
            for side, values in captured.items():
                suffix = "l" if side == "left" else "r"
                calibration[f"{key}_{suffix}"] = values
                if key == "jointangleopose_index":
                    calibration[f"jointangleopose_{suffix}"] = values
            print(f"Captured {label}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(calibration, stream, sort_keys=False)
    print(f"Calibration saved to {args.output}")


if __name__ == "__main__":
    main()
