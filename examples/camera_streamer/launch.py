# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone build and foreground launch commands."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SENDER = ROOT / "televiz/argus_sender"
RECEIVER = ROOT / "televiz/camera_receiver"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["build-sender", "build-receiver", "sender", "receiver", "consumer"],
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", help="Robot Site IP; required for receiver")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    build = (args.build_dir or SENDER / "build").expanduser().absolute()
    if args.action == "build-sender":
        commands = [
            [
                "cmake",
                "-S",
                str(SENDER),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                *extra,
            ],
            ["cmake", "--build", str(build), "--parallel", str(args.jobs)],
        ]
    elif args.action == "build-receiver":
        build = (args.build_dir or RECEIVER / "codec/build").expanduser().absolute()
        commands = [
            [
                "cmake",
                "-S",
                str(RECEIVER / "codec"),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython3_EXECUTABLE={sys.executable}",
                *extra,
            ],
            ["cmake", "--build", str(build), "--parallel", str(args.jobs)],
        ]
    elif args.action == "sender":
        config = args.config or ROOT / "configs/s56-stereo-2560x1984-30.yaml"
        command = [str(build / "argus_sender")]
        if "--list-modes" not in extra and "--help" not in extra:
            command += ["--config", str(config.expanduser().absolute())]
        commands = [[*command, *extra]]
    elif args.action == "receiver":
        if not args.host:
            parser.error(
                "receiver requires --host ROBOT_IP (do not use an example address)"
            )
        config = args.config or ROOT / "configs/s56-stereo-2560x1984-30.yaml"
        commands = [
            [
                sys.executable,
                str(RECEIVER / "camera_receiver.py"),
                "--config",
                str(config.expanduser().absolute()),
                "--host",
                args.host,
                *extra,
            ]
        ]
    else:
        commands = [
            [
                sys.executable,
                str(ROOT / "televiz/cuda_ipc_consumer/consumer.py"),
                "--socket",
                "/tmp/isaac-camera-streamer.sock",
                *extra,
            ]
        ]
    for command in commands:
        print("Command:", command, flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
