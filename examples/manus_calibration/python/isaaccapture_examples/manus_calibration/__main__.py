# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line for the standalone Manus calibration page."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from .server import PageServer
from .worker import CalibrationWorker

logger = logging.getLogger("isaaccapture.examples.manus_calibration")


def build_parser() -> argparse.ArgumentParser:
    """Arguments for the page; the output directory is a launch setting, never browser input."""
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.manus_calibration",
        description="Calibrate Manus gloves from a browser and save the MCAL files on this host.",
    )
    parser.add_argument(
        "--worker",
        required=True,
        type=Path,
        help="path to the manus_glove_calibrator executable",
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="calibration directory on this host"
    )
    parser.add_argument(
        "--sides",
        choices=("left", "right", "both"),
        default="both",
        help="gloves preselected in the page (default: both)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to serve on; the page has no authentication, so keep loopback and use SSH port forwarding",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port to serve on (default: 8080)"
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=15.0,
        help="seconds the worker waits for the requested gloves (default: 15)",
    )
    parser.add_argument(
        "--continuous-pacing",
        choices=("operator", "window"),
        default="operator",
        help="how a continuous step is paced: 'operator' (move, then confirm; the step ends at once) or "
        "'window' (confirm, then move for the SDK's estimate); default: operator",
    )
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fail early, with a usage error, on a worker path that cannot run."""
    if not args.worker.is_file() or not os.access(args.worker, os.X_OK):
        parser.error(f"--worker {args.worker} is not an executable file")
    if not 0 <= args.port < 65536:
        parser.error("--port must be between 0 and 65535")
    if args.discovery_timeout <= 0:
        parser.error("--discovery-timeout must be positive")


async def _serve(args: argparse.Namespace) -> int:
    """Serve until interrupted, then cancel any active attempt and wait for the worker to settle."""
    out_dir = args.out_dir.resolve()
    worker = CalibrationWorker(
        args.worker,
        out_dir,
        discovery_timeout_s=args.discovery_timeout,
        continuous_pacing=args.continuous_pacing,
    )
    server = PageServer(worker, out_dir, args.host, args.port, args.sides)
    await server.start()
    logger.info(
        "serving on http://%s:%d, calibration files go to %s",
        args.host,
        server.port,
        out_dir,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    # SIGHUP: a dropped terminal session must cancel an active calibration rather than kill the page mid-write.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    logger.info("shutting down; cancelling any active calibration")
    await server.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(args, parser)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return asyncio.run(_serve(args))


if __name__ == "__main__":
    sys.exit(main())
