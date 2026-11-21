# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the CTS checks over one MCAP recording.

Usage:
    python -m full_body_cts.cli RECORDING.mcap [--json] [--check NAME ...]

Exit status is 0 for a pass, 1 for a fail, 2 for a retake, 3 when there was not
enough data to conclude, and 4 when the check could not run: a usage error, an unknown
check, or a recording or label sidecar that cannot be read.
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from cts_common.labels import UnreadableLabels
from cts_common.mcap_reader import UnreadableRecording

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
EXIT_ERROR = 4


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        """One line on stderr and ``EXIT_ERROR``, never a verdict's status."""
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(prog="python -m full_body_cts.cli", description=__doc__)
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

    try:
        timeline = (
            StepTimeline.load(args.labels)
            if args.labels
            else StepTimeline.beside(args.recording)
        )
    except UnreadableLabels as unreadable:
        parser.error(str(unreadable))
    if args.checks:
        try:
            checks = build(args.checks, timeline)
        except KeyError as unknown:
            parser.error(f"{unknown.args[0]}; --list-checks prints the names")
    else:
        checks = build_all(timeline)
    try:
        report = run(McapFrameSource(args.recording), checks, timeline)
    except UnreadableRecording as unreadable:
        parser.error(str(unreadable))
    print(report.to_json() if args.json else report.to_text())
    return EXIT_STATUS[report.verdict]


if __name__ == "__main__":
    sys.exit(main())
