# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""latency_breakdown.py recovers known stage delays from synthetic bridge/probe CSV pairs and refuses to
report the consumer stage when the signals do not correlate."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_paths import repo_root  # noqa: E402

_TOOL = (
    repo_root()
    / "src"
    / "plugins"
    / "remote_device_bridge"
    / "tools"
    / "latency_breakdown.py"
)
_spec = importlib.util.spec_from_file_location("latency_breakdown", _TOOL)
latency_breakdown = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(latency_breakdown)

MS = 1_000_000
NETWORK_MS, BRIDGE_MS, CONSUMER_MS = 70, 3, 40
OFFSET_NS = 5_000_000 * MS  # bridge clock - sender clock
BRIDGE_HEADER = [
    "push_mono_ns",
    "recv_mono_ns",
    "capture_sender_ns",
    "capture_mapped_ns",
    "age_ms",
    "side",
    "seq",
    "curl",
]


def write_csvs(
    directory: Path,
    capture_mapped: np.ndarray,
    recv: np.ndarray,
    push: np.ndarray,
    curl: np.ndarray,
    probe_t: np.ndarray,
    probe_curl: np.ndarray,
) -> tuple[Path, Path]:
    """Right-hand rows only; capture_sender = capture_mapped - OFFSET_NS."""
    directory.mkdir(parents=True, exist_ok=True)
    bridge = directory / "bridge.csv"
    with open(bridge, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BRIDGE_HEADER)
        for i in range(len(push)):
            age = (push[i] - capture_mapped[i]) / 1e6
            w.writerow(
                [
                    push[i],
                    recv[i],
                    capture_mapped[i] - OFFSET_NS,
                    capture_mapped[i],
                    f"{age:.3f}",
                    "right",
                    i,
                    f"{curl[i]:.5f}",
                ]
            )
    probe = directory / "probe.csv"
    with open(probe, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mono_ns", "side", "curl", "n_set"])
        for t, c in zip(probe_t, probe_curl):
            w.writerow([t, "right", f"{c:.5f}", 25])
    return bridge, probe


def write_case(
    directory: Path,
    network_ms: float,
    bridge_ms: float,
    consumer_ms: float,
    curl_fn,
    probe_noise: bool = False,
    seconds: float = 60.0,
) -> tuple[Path, Path]:
    """90 Hz bridge rows with 10 ms network jitter; a 200 Hz probe (0-2 ms jitter) sees curl_fn
    consumer_ms after the push, or independent noise when probe_noise."""
    rng = np.random.default_rng(35)
    capture_s = np.arange(0.0, seconds, 1 / 90)
    capture_mapped = (2_000 * 1e9 + capture_s * 1e9).astype(np.int64)
    recv = capture_mapped + (
        network_ms * MS + rng.normal(0, 10 * MS, len(capture_s))
    ).astype(np.int64)
    push = recv + int(bridge_ms * MS)
    curl = curl_fn(capture_s, rng)
    probe_t = np.arange(push.min(), push.max(), 5 * MS, dtype=np.int64)
    probe_t += rng.integers(0, 2 * MS, len(probe_t))
    if probe_noise:
        probe_curl = curl_fn(np.zeros(len(probe_t)), rng)
    else:
        probe_curl = latency_breakdown.hold(push, curl, probe_t - int(consumer_ms * MS))
    return write_csvs(directory, capture_mapped, recv, push, curl, probe_t, probe_curl)


def three_steps(t_s: np.ndarray, rng) -> np.ndarray:
    return 0.07 + 0.06 * (np.searchsorted([12.3, 27.8, 44.1], t_s) % 2)


def static_noise(t_s: np.ndarray, rng) -> np.ndarray:
    return 0.09 + rng.normal(0, 0.0005, len(t_s))  # 0.5 mm


def flat(t_s: np.ndarray, rng) -> np.ndarray:
    return np.full(len(t_s), 0.09)


def run_cli(bridge: Path, probe: Path, side: str = "right") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = latency_breakdown.main(
            ["--bridge", str(bridge), "--probe", str(probe), "--side", side]
        )
    return rc, out.getvalue(), err.getvalue()


def write_pair(directory: Path) -> tuple[Path, Path]:
    """Steps of random length between two curl levels, sampled at 90 Hz and probed at 200 Hz."""
    rng = np.random.default_rng(32)
    switch_s = np.cumsum(rng.uniform(0.4, 1.2, size=80))
    level = lambda t_s: 0.08 + 0.05 * (np.searchsorted(switch_s, t_s) % 2)  # noqa: E731

    capture_s = np.arange(0.0, 60.0, 1 / 90)
    capture_sender = (1_000 * 1e9 + capture_s * 1e9).astype(np.int64)
    capture_mapped = capture_sender + OFFSET_NS
    recv = capture_mapped + (
        NETWORK_MS * MS + rng.normal(0, 2 * MS, len(capture_s))
    ).astype(np.int64)
    push = recv + BRIDGE_MS * MS
    curl = level(capture_s)

    bridge = directory / "bridge.csv"
    with open(bridge, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "push_mono_ns",
                "recv_mono_ns",
                "capture_sender_ns",
                "capture_mapped_ns",
                "age_ms",
                "side",
                "seq",
                "curl",
            ]
        )
        for i in range(len(capture_s)):
            age = (push[i] - capture_mapped[i]) / 1e6
            w.writerow(
                [
                    push[i],
                    recv[i],
                    capture_sender[i],
                    capture_mapped[i],
                    f"{age:.3f}",
                    "right",
                    i,
                    f"{curl[i]:.5f}",
                ]
            )

    probe = directory / "probe.csv"
    probe_t = np.arange(push[0], push[-1], 5 * MS, dtype=np.int64)
    seen = latency_breakdown.hold(push, curl, probe_t - CONSUMER_MS * MS)
    with open(probe, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mono_ns", "side", "curl", "n_set"])
        for t, c in zip(probe_t, seen):
            w.writerow([t, "right", f"{c:.5f}", 25])
            w.writerow([t, "left", "", 0])
    return bridge, probe


class LatencyBreakdownTest(unittest.TestCase):
    def test_recovers_70_3_40_ms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bridge_csv, probe_csv = write_pair(Path(tmp))
            r = latency_breakdown.breakdown(
                latency_breakdown.load_bridge(str(bridge_csv), "right"),
                latency_breakdown.load_probe(str(probe_csv), "right"),
            )
        self.assertAlmostEqual(r["network_p50"], NETWORK_MS, delta=5)
        self.assertAlmostEqual(r["bridge_p50"], BRIDGE_MS, delta=5)
        self.assertAlmostEqual(r["consumer_p50"], CONSUMER_MS, delta=5)
        self.assertAlmostEqual(r["edge_p50"], CONSUMER_MS, delta=5)
        self.assertGreater(len(r["edge_delays_ms"]), 40)
        self.assertAlmostEqual(
            r["total_p50"], NETWORK_MS + BRIDGE_MS + CONSUMER_MS, delta=5
        )
        self.assertAlmostEqual(r["offset_ms"], OFFSET_NS / 1e6, delta=1e-3)
        self.assertTrue(r["estimated"])

    def test_recovers_110_2_25_ms_with_three_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bridge_csv, probe_csv = write_case(Path(tmp), 110, 2, 25, three_steps)
            r = latency_breakdown.breakdown(
                latency_breakdown.load_bridge(str(bridge_csv), "right"),
                latency_breakdown.load_probe(str(probe_csv), "right"),
            )
        self.assertTrue(r["estimated"])
        self.assertEqual(len(r["edge_delays_ms"]), 3)
        self.assertAlmostEqual(r["network_p50"], 110, delta=5)
        self.assertAlmostEqual(r["bridge_p50"], 2, delta=5)
        self.assertAlmostEqual(r["consumer_p50"], 25, delta=5)
        self.assertAlmostEqual(r["edge_p50"], 25, delta=5)
        self.assertAlmostEqual(r["total_p50"], 137, delta=5)

    def test_static_hand_with_uncorrelated_noise_is_not_estimated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _ = run_cli(
                *write_case(Path(tmp), 110, 2, 25, static_noise, probe_noise=True)
            )
        self.assertEqual(rc, 0)
        self.assertIn(latency_breakdown.NOT_ESTIMATED, out)
        self.assertIn("(a) network", out)
        self.assertNotIn("(c)", out)
        self.assertNotIn("(d)", out)

    def test_flat_curl_is_not_estimated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _ = run_cli(*write_case(Path(tmp), 110, 2, 25, flat))
        self.assertEqual(rc, 0)
        self.assertIn(latency_breakdown.NOT_ESTIMATED, out)
        self.assertNotIn("(c)", out)

    def test_unusable_input_exits_2_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cases = {
                "empty side": (
                    write_case(Path(tmp) / "a", 110, 2, 25, three_steps),
                    "left",
                ),
                "all-NaN curl": (
                    write_case(
                        Path(tmp) / "b",
                        110,
                        2,
                        25,
                        lambda t, rng: np.full(len(t), np.nan),
                    ),
                    "right",
                ),
                "short overlap": (
                    write_case(Path(tmp) / "c", 110, 2, 25, three_steps, seconds=1.5),
                    "right",
                ),
            }
            for name, ((bridge_csv, probe_csv), side) in cases.items():
                with self.subTest(name):
                    rc, out, err = run_cli(bridge_csv, probe_csv, side)
                    self.assertEqual(rc, 2)
                    self.assertEqual(out, "")
                    self.assertEqual(len(err.strip().splitlines()), 1)
                    self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
