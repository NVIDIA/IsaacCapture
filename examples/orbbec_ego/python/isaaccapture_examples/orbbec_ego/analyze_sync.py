# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write an EGO timing audit and media/sample association index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaccapture_examples.orbbec_ego import analyze_recording_sync


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit selected EGO streams without decoding or playing media."
    )
    parser.add_argument(
        "recording", type=Path, help="Recording directory or completed MCAP"
    )
    parser.add_argument(
        "--required-streams",
        nargs="+",
        required=True,
        choices=("left_rgb", "right_rgb", "accel", "gyro", "audio"),
        help="Streams required by this use; include the reference stream",
    )
    parser.add_argument(
        "--alignment", type=Path, help="Explicit recording-bound alignment JSON"
    )
    parser.add_argument(
        "--reference-stream", choices=("left_rgb", "right_rgb"), default="left_rgb"
    )
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--start-time-ns", type=int)
    parser.add_argument("--end-time-ns", type=int)
    parser.add_argument(
        "--output", required=True, type=Path, help="New JSON audit/index path"
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite audit output: {args.output}")
        report = analyze_recording_sync(
            args.recording,
            required_streams=args.required_streams,
            alignment=args.alignment,
            reference_stream=args.reference_stream,
            epoch=args.epoch,
            start_time_ns=args.start_time_ns,
            end_time_ns=args.end_time_ns,
        )
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as output:
            output.write(encoded)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        parser.error(str(error))
    coverage = report["coverage"]
    status = report["alignment"]["status"]
    print(
        f"EGO alignment status={status}; windows={coverage['window_count']}; "
        f"complete={coverage['complete_windows']}; output={args.output}"
    )
    return (
        0
        if status == "validated"
        and coverage["window_count"] > 0
        and coverage["incomplete_windows"] == 0
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
