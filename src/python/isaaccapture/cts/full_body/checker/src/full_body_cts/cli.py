# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the CTS checks over one MCAP recording.

Usage:
    python -m full_body_cts.cli RECORDING.mcap [--json] [--check NAME ...]

Exit status is 0 for a pass, 1 for a fail, 2 for a retake and 3 when there was not
enough data to conclude.
"""

from __future__ import annotations

import argparse
import sys

from .checks import build, build_all
from .labels import StepTimeline
from .mcap_source import McapFrameSource
from .report import Verdict, run

EXIT_STATUS = {
    Verdict.PASS: 0,
    Verdict.FAIL: 1,
    Verdict.RETAKE: 2,
    Verdict.INSUFFICIENT_DATA: 3,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m full_body_cts.cli", description=__doc__
    )
    parser.add_argument("recording", nargs="?", help="MCAP file to check")
    parser.add_argument(
        "--json", action="store_true", help="emit the machine-readable report"
    )
    parser.add_argument(
        "--check",
        action="append",
        dest="checks",
        metavar="NAME",
        help="run only this check; repeatable",
    )
    parser.add_argument(
        "--labels",
        metavar="SIDECAR",
        help="motion-step labels; defaults to RECORDING.labels.json beside the file",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="print the checks, with no recording, and exit",
    )
    args = parser.parse_args(argv)

    if args.list_checks:
        for check in build_all():
            print(
                f"{check.name:<45} {check.group:<9} {check.severity:<9} {check.summary}"
            )
        return 0

    if args.recording is None:
        parser.error("a recording is required unless --list-checks is given")

    source = McapFrameSource(args.recording)
    timeline = (
        StepTimeline.load(args.labels)
        if args.labels
        else StepTimeline.beside(args.recording)
    )
    checks = build(args.checks, timeline) if args.checks else build_all(timeline)
    report = run(source, checks, timeline)
    print(report.to_json() if args.json else report.to_text())
    return EXIT_STATUS[report.verdict]


if __name__ == "__main__":
    sys.exit(main())
