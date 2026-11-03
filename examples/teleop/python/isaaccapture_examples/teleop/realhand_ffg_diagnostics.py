# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record raw FFG sensors, OpenXR hands and controllers without robot retargeting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio import McapRecordingConfig
from isaaccapture.deviceio_trackers import PluginDeviceStatusTracker
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
    JointStateSource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)

from .realhand_ffg_glove_calibration import _POSES
from .realhand_ffg_hand_demo_bridge import (
    load_calibration_example,
    sample_glove_inputs,
)

SIDES = ("left", "right")
POSES = (
    *_POSES,
    *(
        (f"touch the thumb tip lightly to the {finger} fingertip", f"thumb_{finger}")
        for finger in ("index", "middle", "ring", "little")
    ),
    (_POSES[0][0], "open_end"),
)


def build_pipeline():
    # Sources must remain in the output graph for their native MCAP channels to be recorded.
    hands = HandsSource(name="diagnostic_hands")
    controllers = ControllersSource(name="diagnostic_controllers")
    outputs = {}
    for side in SIDES:
        outputs[f"hand_{side}"] = hands.output(getattr(HandsSource, side.upper()))
        outputs[f"controller_{side}"] = controllers.output(
            getattr(ControllersSource, side.upper())
        )
        raw = JointStateSource(
            name=f"diagnostic_raw_{side}",
            collection_id=f"realhand_ffg_glove_{side}",
            joint_names=[f"sensor_{index}" for index in range(21)],
        )
        outputs[f"raw_{side}"] = raw.output(JointStateSource.JOINTS)
    return OutputCombiner(outputs)


def summarize_samples(samples):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (21,) or len(values) < 30:
        raise ValueError("Need at least 30 complete raw frames for each pose and side")
    if not np.all(np.isfinite(values)):
        raise ValueError("Raw capture contains non-finite sensor angles")
    values = np.unwrap(values, axis=0)
    median = np.median(values, axis=0)
    return {
        "frames": len(values),
        "median_rad": median.tolist(),
        "robust_sigma_rad": (
            1.4826 * np.median(np.abs(values - median), axis=0)
        ).tolist(),
        "p05_rad": np.percentile(values, 5, axis=0).tolist(),
        "p95_rad": np.percentile(values, 95, axis=0).tolist(),
    }


def pump(session, tracker, example, seconds, collect=False):
    samples = {side: [] for side in SIDES}
    missing = dict.fromkeys(SIDES, 0)
    start = time.monotonic_ns()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = session.step()
        raw, _ = sample_glove_inputs(session, result, example, tracker)
        if collect:
            for side in SIDES:
                if raw[side] is None:
                    missing[side] += 1
                else:
                    samples[side].append(raw[side].tolist())
        time.sleep(1 / 120)
    return start, time.monotonic_ns(), samples, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--plugin-path", type=Path, required=True)
    parser.add_argument(
        "--ports", required=True, help="Previously identified glove serial ports"
    )
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--prepare-seconds", type=float, default=3.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()
    if args.cycles < 1 or not all(
        np.isfinite(x) and x >= 1 for x in (args.prepare_seconds, args.hold_seconds)
    ):
        parser.error(
            "Cycles must be positive; preparation and hold times must be finite and >= 1 s"
        )
    if not args.calibration.is_file():
        parser.error("An existing human calibration file is required")
    metadata = args.output.with_suffix(".diagnostics.json")
    if args.output.exists() or metadata.exists():
        parser.error("Output already exists; select a new recording path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tracker = PluginDeviceStatusTracker("realhand_ffg_glove/device_status")
    config = TeleopSessionConfig(
        app_name="RealHandFFGDiagnostics",
        trackers=[tracker],
        pipeline=build_pipeline(),
        plugins=[
            PluginConfig(
                plugin_name="realhand_ffg_glove_plugin",
                plugin_root_id="realhand_ffg_glove",
                search_paths=[args.plugin_path],
                plugin_args=[
                    f"--ports={args.ports}",
                    f"--calibration={args.calibration.resolve()}",
                    "--wrist-source=controller",
                ],
                required=True,
            )
        ],
        mcap_config=McapRecordingConfig(str(args.output)),
    )
    report = {
        "purpose": "Raw sensor diagnosis, not the official CTS or an acceptance result",
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "clock_domain": "system CLOCK_MONOTONIC, nanoseconds",
        "completed": False,
        "windows": [],
    }
    example = load_calibration_example()
    try:
        with CloudXRLauncher.launch_context(args):
            input(
                "Connect the headset and controllers. Press Enter to begin the guided diagnostic... "
            )
            with TeleopSession(config) as session:
                print("Waiting for current raw data from both gloves...", flush=True)
                deadline = time.monotonic() + 20
                while True:
                    result = session.step()
                    raw, errors = sample_glove_inputs(session, result, example, tracker)
                    if all(raw[s] is not None for s in SIDES):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"Both gloves must have current data: {errors}"
                        )
                    time.sleep(0.01)
                for cycle in range(args.cycles):
                    for instruction, label in POSES:
                        print(
                            f"\nCycle {cycle + 1}/{args.cycles}: {instruction}.",
                            flush=True,
                        )
                        print(
                            f"Prepare for {args.prepare_seconds:g} seconds...",
                            flush=True,
                        )
                        pump(session, tracker, example, args.prepare_seconds)
                        print(f"\aHOLD for {args.hold_seconds:g} seconds.", flush=True)
                        start, end, samples, missing = pump(
                            session, tracker, example, args.hold_seconds, True
                        )
                        window = {
                            "cycle": cycle + 1,
                            "pose": label,
                            "start_ns": start,
                            "end_ns": end,
                            "missing_frames": missing,
                            "raw": {},
                        }
                        report["windows"].append(window)
                        for side in SIDES:
                            window["raw"][side] = summarize_samples(samples[side])
                        if any(missing.values()):
                            raise RuntimeError(
                                "Glove data was lost during a hold; the partial recording is retained"
                            )
                report["completed"] = True
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted; partial data retained.", flush=True)
    finally:
        metadata.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            f"Diagnostic recording: {args.output}\nPose summary: {metadata}", flush=True
        )


if __name__ == "__main__":
    main()
