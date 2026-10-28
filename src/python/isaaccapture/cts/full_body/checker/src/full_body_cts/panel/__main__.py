# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m full_body_cts.panel RECORDING.mcap`` — the CTS panel.

Runs every check over the recording in one pass, prints the same text report the CLI
prints, then serves the skeleton and the result list until interrupted.

Needs the optional viser extra: ``./setup_env.sh --panel``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..labels import StepTimeline
from ..mcap_source import McapFrameSource
from ..report import run
from .track import TeeSource

try:
    from .app import serve
except ModuleNotFoundError as missing:
    raise SystemExit(
        f"{missing}. The panel's renderer is an optional extra: run "
        f"`./setup_env.sh --panel` to install it."
    ) from missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m full_body_cts.panel", description=__doc__
    )
    parser.add_argument("recording", help="MCAP file to view")
    parser.add_argument(
        "--labels",
        metavar="SIDECAR",
        help="motion-step labels; defaults to RECORDING.labels.json beside the file",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; 0.0.0.0 to reach the panel from another machine",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    timeline = (
        StepTimeline.load(args.labels)
        if args.labels
        else StepTimeline.beside(args.recording)
    )
    source = TeeSource(McapFrameSource(args.recording), timeline)
    report = run(source, None, timeline)
    print(report.to_text())
    serve(
        report,
        source.track(),
        Path(args.recording),
        timeline,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
