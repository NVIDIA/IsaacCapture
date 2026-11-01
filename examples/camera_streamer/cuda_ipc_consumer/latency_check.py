#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump camera_receiver CUDA IPC frame latency metadata as CSV."""

from __future__ import annotations

import argparse
import csv
import signal
import sys
from typing import Optional

from consumer import (
    CudaIpcClient,
    _LATENCY_CSV_FIELDS,
    _annotate_consumer_receive,
    _csv_row,
    _parse_stream_filter,
)


def run(args: argparse.Namespace) -> int:
    client = CudaIpcClient(args.socket, _parse_stream_filter(args.streams))
    stop = False

    def on_sigint(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_sigint)

    output = sys.stdout
    close_output = False
    if args.output != "-":
        output = open(args.output, "w", newline="", encoding="utf-8")
        close_output = True

    frames_seen = 0
    try:
        handshake = client.connect()
        subscribed = handshake["subscribed"]
        print(
            "connected to "
            f"{args.socket}: "
            + ", ".join(
                f"{s['name']} {s['width']}x{s['height']} "
                f"{s.get('pixel_format', 'unknown')} slots={len(s['slots'])}"
                for s in subscribed.get("streams", [])
            ),
            file=sys.stderr,
            flush=True,
        )
        writer = csv.DictWriter(output, fieldnames=_LATENCY_CSV_FIELDS)
        writer.writeheader()
        output.flush()

        while not stop:
            messages = client.read_messages(timeout_s=args.poll_timeout)
            if messages is None:
                print("publisher disconnected", file=sys.stderr, flush=True)
                break
            for msg in messages:
                if msg.get("type") != "frame":
                    continue
                msg = _annotate_consumer_receive(msg)
                writer.writerow(_csv_row(msg))
                frames_seen += 1
                if args.frames > 0 and frames_seen >= args.frames:
                    return 0
            output.flush()
    finally:
        client.close()
        if close_output:
            output.close()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--socket",
        default="/tmp/camera_receiver_avs1_cuda_ipc.sock",
        help="Unix socket path exposed by camera_receiver ipc.socket_path",
    )
    ap.add_argument(
        "--streams",
        help="comma-separated IPC stream names to subscribe to; default subscribes to all",
    )
    ap.add_argument(
        "--output",
        default="-",
        help="CSV output path, or '-' for stdout",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=0,
        help="exit after this many frame notifications; 0 means run until stopped",
    )
    ap.add_argument("--poll-timeout", type=float, default=0.05)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
