#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample the hand as a consumer sees it (gr00t XRClient) into a CSV for latency_breakdown.py.

Runs inside the gr00t venv. mono_ns is time.monotonic_ns(), i.e. CLOCK_MONOTONIC, the clock of the
bridge's --stats-csv. curl is |slot 9 - slot 0| of get_hand_fingertips(): XRSnapshot slot = OpenXR
joint - 1 (OXR_TO_FINGERTIP_SLOT), so these are XR_HAND_JOINT_WRIST_EXT (1) and
XR_HAND_JOINT_INDEX_TIP_EXT (10), the joints of the bridge's index_curl_m.
"""

from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np

WRIST_SLOT = 0
INDEX_TIP_SLOT = 9


def fingertips_curl(joints: np.ndarray | None) -> tuple[float, int]:
    """(curl in m or NaN, number of non-identity slots) of one get_hand_fingertips() array.

    An identity slot is one the server did not fill. The wrist is the exception: without a headset
    wrist pose the Manus root is the origin, so a tracked wrist is exactly eye(4). Like the bridge's
    index_curl_m, curl then only needs the index tip.
    """
    if joints is None:
        return math.nan, 0
    is_set = ~np.all(joints == np.eye(4), axis=(1, 2))
    if not is_set[INDEX_TIP_SLOT]:
        return math.nan, int(is_set.sum())
    curl = float(
        np.linalg.norm(joints[INDEX_TIP_SLOT, :3, 3] - joints[WRIST_SLOT, :3, 3])
    )
    return curl if math.isfinite(curl) else math.nan, int(is_set.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", required=True, help="CSV path (mono_ns,side,curl,n_set)"
    )
    parser.add_argument(
        "--hz", type=float, default=200.0, help="sample rate (default %(default)s)"
    )
    parser.add_argument(
        "--seconds", type=float, default=60.0, help="duration (default %(default)s)"
    )
    parser.add_argument(
        "--sides", default="left,right", help="comma list (default %(default)s)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="xr_server host (default %(default)s)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="xr_server port (default XRClient's)"
    )
    args = parser.parse_args()

    from groot.core.robotics.teleop.xr.xr_client import XRClient

    sides = [side for side in args.sides.split(",") if side]
    client = (
        XRClient(host=args.host)
        if args.port is None
        else XRClient(host=args.host, port=args.port)
    )
    client.start_streaming()
    period_ns = int(1e9 / args.hz)
    rows = 0
    with_curl = 0
    try:
        with open(args.out, "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["mono_ns", "side", "curl", "n_set"])
            start = time.monotonic_ns()
            next_ns = start
            end = start + int(args.seconds * 1e9)
            while next_ns < end:
                time.sleep(max(0.0, (next_ns - time.monotonic_ns()) / 1e9))
                for side in sides:
                    mono_ns = time.monotonic_ns()
                    joints = client.get_hand_fingertips(side)
                    curl, n_set = fingertips_curl(joints)
                    if not math.isnan(curl):
                        with_curl += 1
                    writer.writerow(
                        [
                            mono_ns,
                            side,
                            "" if math.isnan(curl) else f"{curl:.5f}",
                            n_set,
                        ]
                    )
                    rows += 1
                next_ns += period_ns
    finally:
        client.close()
    print(f"latency_probe_xrclient: {rows} rows ({with_curl} with curl) -> {args.out}")


if __name__ == "__main__":
    main()
