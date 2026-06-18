#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import time

from vehicle_teleop.command_log import McapCommandLogReader


class CommandMcapReplay:
    def __init__(self, args: argparse.Namespace) -> None:
        self._reader = McapCommandLogReader(args.path)
        self._realtime = args.realtime

    def run(self) -> None:
        previous_sample_time_ns = None
        for record in self._reader.records():
            if self._realtime and previous_sample_time_ns is not None:
                delay_ns = max(0, record.sample.timestamp_ns - previous_sample_time_ns)
                time.sleep(delay_ns / 1_000_000_000)
            previous_sample_time_ns = record.sample.timestamp_ns
            command = record.command
            print(
                f"seq={command.sequence} t={command.timestamp_ns} "
                f"accel={command.accel:+.3f} steer={command.steer:+.3f} "
                f"throttle={command.throttle:.3f} brake={command.brake:.3f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay an MCAP vehicle command log.")
    parser.add_argument("path", help="Path to MCAP command log.")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between records using sample timestamps.",
    )
    return parser


def main() -> None:
    CommandMcapReplay(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()
