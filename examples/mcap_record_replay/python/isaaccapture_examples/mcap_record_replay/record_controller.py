# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Record a live OpenXR controller-tracking session to an MCAP file.

``CloudXRLauncher`` starts the CloudXR runtime and WSS proxy automatically —
no separate terminal or pre-running headset daemon is needed. The pipeline in
``common.py`` wires only ``ControllersSource``, so ``TeleopSession`` records
exactly the ``controllers`` channel — no head, no hands.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.record_controller [duration_seconds] [output.mcap] [--accept-eula]

Defaults: 5 seconds → ./recordings/controllers_<timestamp>.mcap

See: https://nvidia.github.io/IsaacCapture/main/references/mcap_record_replay.html
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio import McapRecordingConfig
from isaaccapture.retargeting_engine.tensor_types.indices import ControllerInputIndex
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

from .common import build_controller_pipeline


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "duration", nargs="?", type=float, default=5.0, help="Recording duration (s)"
    )
    parser.add_argument("output", nargs="?", help="Output .mcap path")
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])

    duration_s: float = args.duration

    if args.output:
        mcap_path = Path(args.output)
        mcap_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path.cwd() / "recordings"
        out_dir.mkdir(exist_ok=True)
        mcap_path = out_dir / f"controllers_{datetime.now():%Y%m%d_%H%M%S}.mcap"

    print(f"[record] writing {mcap_path} for {duration_s:.1f}s")

    config = TeleopSessionConfig(
        app_name="McapControllerRecordExample",
        pipeline=build_controller_pipeline(),
        mcap_config=McapRecordingConfig(str(mcap_path)),
    )

    with CloudXRLauncher.launch_context(args) as launcher:
        if launcher.owns_runtime:
            print(
                f"[record] CloudXR runtime started (WSS log: {launcher.wss_log_path})"
            )
        with TeleopSession(config) as session:
            start = time.time()
            while time.time() - start < duration_s:
                result = session.step()
                if session.frame_count % 60 == 0:
                    left_ctrl = result["controller_left"]
                    right_ctrl = result["controller_right"]
                    left = not left_ctrl.is_none and bool(
                        left_ctrl[ControllerInputIndex.AIM_IS_VALID]
                    )
                    right = not right_ctrl.is_none and bool(
                        right_ctrl[ControllerInputIndex.AIM_IS_VALID]
                    )
                    print(
                        f"[record] t={time.time() - start:5.2f}s  "
                        f"frame={session.frame_count}  L={'Y' if left else '-'} "
                        f"R={'Y' if right else '-'}"
                    )
                time.sleep(1 / 60)

    print(f"[record] done — {mcap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
