#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture-to-consumer latency breakdown from a bridge --stats-csv and a latency_probe_xrclient.py CSV.

Both CSVs must come from the same box (one CLOCK_MONOTONIC). The consumer stage is the lag of the
probe's curl behind the bridge's pushed curl, so it needs a step test (fast finger flexes). (c) and (d)
are only reported when enough windows correlate (--min-xcorr) and enough step edges pair (--min-edges).
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

GRID_NS = 1_000_000  # resampling step for the curl signals
WINDOW_NS = 10_000_000_000
WINDOW_STEP_NS = 5_000_000_000
MAX_LAG_NS = 500_000_000
MIN_LAG_NS = -50_000_000
EDGE_SMOOTH_NS = 10_000_000
EDGE_MIN_GAP_NS = 200_000_000
EDGE_THRESHOLD = 0.3  # fraction of the largest |d curl/dt|
EDGE_MAD_K = 3.0  # edge pairs outside median +- K * MAD are discarded
MIN_OVERLAP_NS = 2_000_000_000
NOT_ESTIMATED = "consumer lag: not estimated (low correlation / too few edges)"


class InputError(Exception):
    """Input the breakdown cannot use (no rows, no finite curl, too little overlap)."""


def load_bridge(path: str, side: str) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["side"] == side]
    cols = ("push_mono_ns", "recv_mono_ns", "capture_sender_ns", "capture_mapped_ns")
    out = {c: np.array([int(r[c]) for r in rows], dtype=np.int64) for c in cols}
    out["curl"] = np.array([float(r["curl"]) for r in rows])
    return out


def load_probe(path: str, side: str) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["side"] == side and r["curl"] != ""]
    out = {
        "mono_ns": np.array([int(r["mono_ns"]) for r in rows], dtype=np.int64),
        "curl": np.array([float(r["curl"]) for r in rows]),
    }
    finite = np.isfinite(out["curl"])
    return {k: v[finite] for k, v in out.items()}


def check_inputs(bridge: dict[str, np.ndarray], probe: dict[str, np.ndarray]) -> None:
    if len(bridge["push_mono_ns"]) == 0:
        raise InputError("no bridge rows for this side")
    if not np.isfinite(bridge["curl"]).any():
        raise InputError("bridge curl is all NaN for this side")
    if len(probe["mono_ns"]) == 0:
        raise InputError("no probe rows with curl for this side")
    overlap_ns = min(bridge["push_mono_ns"].max(), probe["mono_ns"].max()) - max(
        bridge["push_mono_ns"].min(), probe["mono_ns"].min()
    )
    if overlap_ns < MIN_OVERLAP_NS:
        raise InputError(
            f"bridge/probe overlap is {max(overlap_ns, 0) / 1e9:.2f} s, need {MIN_OVERLAP_NS / 1e9:.0f} s"
        )


def hold(t_ns: np.ndarray, values: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    """Zero-order hold of (t, value) onto grid; grid points before the first sample take the first value."""
    order = np.argsort(t_ns, kind="stable")
    idx = np.searchsorted(t_ns[order], grid_ns, side="right") - 1
    return values[order][np.clip(idx, 0, len(order) - 1)]


def xcorr_lag_ns(ref: np.ndarray, sig: np.ndarray) -> tuple[float, float] | None:
    """(lag ns, peak score) of sig behind ref maximising normalized cross-correlation; None for a flat window."""
    lags = np.arange(MIN_LAG_NS // GRID_NS, MAX_LAG_NS // GRID_NS + 1)
    n = len(ref) - lags.max() + min(lags.min(), 0)
    if n <= 0:
        return None
    start = -min(lags.min(), 0)
    a = ref[start : start + n] - ref[start : start + n].mean()
    if a.std() == 0:
        return None
    best, best_score = None, -np.inf
    for lag in lags:
        b = sig[start + lag : start + lag + n]
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        if denom == 0:
            continue
        score = (a * b).sum() / denom
        if score > best_score:
            best, best_score = lag, score
    return None if best is None else (float(best * GRID_NS), float(best_score))


def edges(grid_ns: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Times and signs of |d curl/dt| peaks (smoothed derivative above EDGE_THRESHOLD of its max)."""
    k = max(1, EDGE_SMOOTH_NS // GRID_NS)
    deriv = (signal[k:] - signal[:-k]) / k
    mag = np.abs(deriv)
    if len(mag) == 0 or mag.max() == 0:
        return np.array([], dtype=np.int64), np.array([])
    above = mag > EDGE_THRESHOLD * mag.max()
    times, signs = [], []
    i = 0
    while i < len(mag):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(mag) and above[j]:
            j += 1
        peak = i + int(np.argmax(mag[i:j]))
        t = grid_ns[peak] + (k * GRID_NS) // 2
        if not times or t - times[-1] >= EDGE_MIN_GAP_NS:
            times.append(t)
            signs.append(np.sign(deriv[peak]))
        i = j
    return np.array(times, dtype=np.int64), np.array(signs)


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def breakdown(
    bridge: dict[str, np.ndarray],
    probe: dict[str, np.ndarray],
    min_xcorr: float = 0.6,
    min_edges: int = 3,
) -> dict:
    """Raises InputError for unusable input. consumer_p50 and total_p50 are NaN unless `estimated`."""
    check_inputs(bridge, probe)
    network_ms = (bridge["recv_mono_ns"] - bridge["capture_mapped_ns"]) / 1e6
    bridge_ms = (bridge["push_mono_ns"] - bridge["recv_mono_ns"]) / 1e6
    offset_ms = (bridge["capture_mapped_ns"] - bridge["capture_sender_ns"]) / 1e6

    t0 = max(bridge["push_mono_ns"].min(), probe["mono_ns"].min())
    t1 = min(bridge["push_mono_ns"].max(), probe["mono_ns"].max())
    grid = np.arange(t0, t1, GRID_NS, dtype=np.int64)
    finite = np.isfinite(bridge["curl"])
    ref = hold(bridge["push_mono_ns"][finite], bridge["curl"][finite], grid)
    sig = hold(probe["mono_ns"], probe["curl"], grid)

    window = WINDOW_NS // GRID_NS
    step = WINDOW_STEP_NS // GRID_NS
    window_lags, scores = [], []
    windows_total = 0
    for s in range(0, max(1, len(grid) - window + 1), step):
        windows_total += 1
        result = xcorr_lag_ns(ref[s : s + window], sig[s : s + window])
        if result is None:
            continue
        scores.append(result[1])
        if result[1] >= min_xcorr:
            window_lags.append(result[0] / 1e6)

    ref_t, ref_sign = edges(grid, ref)
    sig_t, sig_sign = edges(grid, sig)
    edge_delays = []
    for t, sign in zip(ref_t, ref_sign):
        candidates = (
            (sig_sign == sign) & (sig_t >= t + MIN_LAG_NS) & (sig_t <= t + MAX_LAG_NS)
        )
        if candidates.any():
            delays = sig_t[candidates] - t
            edge_delays.append(delays[np.argmin(np.abs(delays))] / 1e6)

    # The MAD floor is the probe period: edge times are quantized to it.
    edge_ms = np.array(edge_delays)
    edges_paired = len(edge_ms)
    if edges_paired:
        probe_period_ms = float(np.median(np.diff(np.unique(probe["mono_ns"])))) / 1e6
        med = np.median(edge_ms)
        mad = max(float(np.median(np.abs(edge_ms - med))), probe_period_ms)
        edge_ms = edge_ms[np.abs(edge_ms - med) <= EDGE_MAD_K * mad]

    estimated = len(window_lags) > 0 and len(edge_ms) >= min_edges
    consumer_ms = np.array(window_lags)
    consumer_p50 = pct(consumer_ms, 50) if estimated else float("nan")
    return {
        "network_p50": pct(network_ms, 50),
        "network_p99": pct(network_ms, 99),
        "bridge_p50": pct(bridge_ms, 50),
        "bridge_p99": pct(bridge_ms, 99),
        "consumer_p50": consumer_p50,
        "consumer_windows": len(window_lags),
        "windows_total": windows_total,
        "xcorr_median": pct(np.array(scores), 50),
        "edge_delays_ms": edge_ms.tolist(),
        "edges_paired": edges_paired,
        "edges_discarded": edges_paired - len(edge_ms),
        "edge_p50": pct(edge_ms, 50),
        "estimated": estimated,
        "offset_ms": pct(offset_ms, 50),
        "total_p50": pct(network_ms, 50) + pct(bridge_ms, 50) + consumer_p50,
        "samples": len(bridge["push_mono_ns"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bridge", required=True, help="remote_device_bridge --stats-csv output"
    )
    parser.add_argument(
        "--probe", required=True, help="latency_probe_xrclient.py output"
    )
    parser.add_argument("--side", default="right", choices=("left", "right"))
    parser.add_argument(
        "--min-xcorr",
        type=float,
        default=0.6,
        help="windows whose correlation peak is lower are not used for (c) (default 0.6)",
    )
    parser.add_argument(
        "--min-edges",
        type=int,
        default=3,
        help="step edges that must pair for (c) to be reported (default 3)",
    )
    args = parser.parse_args(argv)

    try:
        r = breakdown(
            load_bridge(args.bridge, args.side),
            load_probe(args.probe, args.side),
            args.min_xcorr,
            args.min_edges,
        )
    except InputError as e:
        print(f"latency_breakdown: side={args.side}: {e}", file=sys.stderr)
        return 2
    edges_ms = ", ".join(f"{d:.0f}" for d in r["edge_delays_ms"][:20])
    print(
        f"side={args.side} bridge samples={r['samples']} offset_ms={r['offset_ms']:.3f} (capture_mapped - capture_sender)"
    )
    print(f"{'stage':<40}{'p50 ms':>10}{'p99 ms':>10}")
    print(
        f"{'(a) network+queue  recv - capture_mapped':<40}{r['network_p50']:>10.1f}{r['network_p99']:>10.1f}"
    )
    print(
        f"{'(b) bridge         push - recv':<40}{r['bridge_p50']:>10.1f}{r['bridge_p99']:>10.1f}"
    )
    print(
        f"xcorr windows used {r['consumer_windows']}/{r['windows_total']} (peak >= {args.min_xcorr:.2f}),"
        f" median peak {r['xcorr_median']:.2f}"
    )
    print(
        f"step edges paired {r['edges_paired']}, discarded {r['edges_discarded']} (outside median +- 3 MAD),"
        f" kept {len(r['edge_delays_ms'])} (need {args.min_edges})"
    )
    if not r["estimated"]:
        print(NOT_ESTIMATED)
        return 0
    print(
        f"{'(c) consumer       xcorr probe vs bridge':<40}{r['consumer_p50']:>10.1f}{'':>10}  ({r['consumer_windows']} x 10 s windows)"
    )
    print(
        f"{'    step edges     probe - bridge':<40}{r['edge_p50']:>10.1f}{'':>10}  n={len(r['edge_delays_ms'])}: {edges_ms}"
    )
    print(f"{'(d) total          (a)+(b)+(c)':<40}{r['total_p50']:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
