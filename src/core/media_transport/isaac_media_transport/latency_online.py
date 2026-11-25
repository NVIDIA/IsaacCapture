# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewer-owned, clock-qualified frame joins over an authenticated SSH helper."""

import asyncio
from dataclasses import asdict
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import threading
import time

from .latency import open_trace
from .latency_bus import SocketTail, endpoint
from .latency_monitor import (
    ClockSamples,
    OnlineJoiner,
    RemoteReader,
    TOTAL_SPAN,
    clock_identity,
    serve,
)


def helper_script():
    return "\n".join(
        (
            "import hashlib, json, os, socket, sys, time\nfrom pathlib import Path",
            inspect.getsource(endpoint),
            inspect.getsource(SocketTail),
            inspect.getsource(clock_identity),
            inspect.getsource(serve),
            "JsonlTail = SocketTail",
            "sys.argv = [sys.argv[0], str(endpoint(sys.argv[1], 'sender', sys.argv[2] or None))]",
            "serve()",
        )
    )


class OnlineLatency:
    def __init__(self, config, *, ipc):
        self.config = config
        clock = config["clock"]
        self.client = clock["client"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+", self.client):
            raise ValueError("latency.clock.client must be SSH user@host")
        for key, default in (
            ("max_age_s", 2),
            ("max_uncertainty_ms", 5),
            ("drift_ppm", 1000),
        ):
            value = clock.get(key, default)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"latency.clock.{key} must be finite and positive")
        self.roles = ("receiver", "viewer") if ipc else ("viewer",)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = None
        self._error = "waiting for clock qualification and frame markers"
        self._updated = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="latency-online"
        )
        self._thread.start()

    def _run(self):
        asyncio.run(self._retry())

    async def _retry(self):
        while not self._stop.is_set():
            try:
                await self._collect()
            except (OSError, ValueError, RuntimeError, KeyError, TimeoutError) as exc:
                with self._lock:
                    self._error = str(exc) or type(exc).__name__
                    self._snapshot = None
                await asyncio.sleep(1)

    async def _collect(self):
        config, clock = self.config, self.config["clock"]
        run_id = config["run_id"]
        clocks = ClockSamples(
            clock_identity(),
            max_age_ns=int(clock.get("max_age_s", 2) * 1e9),
            uncertainty_ns=int(clock.get("max_uncertainty_ms", 5) * 1e6),
            drift_ppm=clock.get("drift_ppm", 1000),
        )
        tails = [
            SocketTail(endpoint(run_id, role, config.get("runtime_dir")))
            for role in self.roles
        ]
        command = [
            "env",
            "-u",
            "LD_PRELOAD",
            "-u",
            "FAKETIME",
            "-u",
            "FAKETIME_DONT_FAKE_MONOTONIC",
            clock.get("client_python", "python3"),
            "-u",
            "-c",
            helper_script(),
            run_id,
            config.get("runtime_dir", ""),
        ]
        environment = {
            k: v
            for k, v in os.environ.items()
            if k not in ("LD_PRELOAD", "FAKETIME", "FAKETIME_DONT_FAKE_MONOTONIC")
        }
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            self.client,
            shlex.join(command),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            limit=4 * 1024 * 1024,
        )
        remote = RemoteReader(proc)
        stream = None
        try:

            async def probe():
                h1, h4, response = await remote.request("probe")
                sample = clocks.add(
                    h1,
                    response["client_received_ns"],
                    response["client_sent_ns"],
                    h4,
                    response["clock_id"],
                )
                if stream:
                    stream.write(
                        json.dumps(dict(record_type="clock_sample", **asdict(sample)))
                        + "\n"
                    )

            for _ in range(8):
                await probe()
            await remote.request("follow")
            for tail in tails:
                await asyncio.to_thread(tail.follow)
            begin = time.monotonic_ns()
            joiner = OnlineJoiner(
                run_id,
                clocks,
                collection_start_ns=begin,
                measurement_start_ns=begin,
                window_size=300,
            )
            if config.get("file_enabled", False):
                root = Path(config.get("log_dir", "/tmp")).expanduser()
                stream = open_trace(root / f"umb-{run_id}-online.jsonl")
            next_probe = next_report = time.monotonic()
            previous_count = 0
            last_valid = time.monotonic()
            while not self._stop.is_set():
                if time.monotonic() >= next_probe:
                    await probe()
                    next_probe = time.monotonic() + 0.5
                _, _, response = await remote.request("read")
                if response["clock_id"] != clocks.source:
                    raise ValueError("Client clock identity changed")
                records = response["events"]
                for tail in tails:
                    records.extend(await asyncio.to_thread(tail.read))
                now = time.monotonic_ns()
                for event in records:
                    joiner.add(event, now)
                rows = joiner.flush(now)
                if stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
                if time.monotonic() >= next_report:
                    snapshot = joiner.summary()
                    for name, metric in snapshot["metrics"].items():
                        metric["session_mean_ms"] = metric["mean_ms"]
                        values = joiner.samples[name]
                        metric["mean_ms"] = sum(values) / len(values)
                    count = snapshot["valid_total_count"]
                    if count > previous_count:
                        last_valid = time.monotonic()
                    previous_count = count
                    with self._lock:
                        self._snapshot = snapshot
                        self._updated = last_valid
                        self._error = (
                            None
                            if count
                            else "waiting for matching capture/viewer endpoints"
                        )
                    if stream:
                        stream.write(json.dumps(snapshot) + "\n")
                        stream.flush()
                    next_report = time.monotonic() + 1
                await asyncio.sleep(0.1)
        finally:
            if stream:
                stream.close()
            await remote.close()

    def lines(self):
        with self._lock:
            snapshot, error, updated = self._snapshot, self._error, self._updated
        if error or snapshot is None or time.monotonic() - updated > 3:
            return (
                "Partial P2P: unavailable",
                error or "No fresh qualified frame totals",
            )
        totals = [
            (name.split("/")[0], metric)
            for name, metric in snapshot["metrics"].items()
            if name.endswith("/totals/" + TOTAL_SPAN)
        ]
        values = " | ".join(
            f"{name.rsplit('.', 1)[-1]} {value['mean_ms']:.1f}±{value['max_uncertainty_ms']:.1f} ms"
            for name, value in totals
        )
        return (
            "Partial P2P mean: " + values,
            "Capture available → viewer submit (clock-qualified)",
        )

    def report_text(self):
        with self._lock:
            snapshot = self._snapshot
        status = self.lines()
        if snapshot is None or "unavailable" in status[0]:
            return "\n".join(status) + "\n"
        return format_report(
            snapshot, self.config["clock"].get("max_uncertainty_ms", 5)
        )

    def close(self):
        self._stop.set()
        self._thread.join(6)


def format_report(snapshot, uncertainty_budget_ms):
    """Align per-track intervals; overlapping diagnostics never enter the total."""
    from .latency_monitor import STAGES

    tracks = {}
    for name, metric in snapshot["metrics"].items():
        track, group, span = name.split("/", 2)
        tracks.setdefault(track, {}).setdefault(group, []).append((span, metric))
    width = max([42] + [len(name.split("/", 2)[-1]) for name in snapshot["metrics"]])

    def row(label, metric):
        return (
            f"  {label:<{width}}  {metric['mean_ms']:>9.2f}  "
            f"{metric['p95_window_ms']:>9.2f}  {metric['max_uncertainty_ms']:>10.2f}  "
            f"{metric['window_count']:>6}"
        )

    def order(item):
        start, end = item[0].split("->", 1)
        return (STAGES.index(start) if start in STAGES else len(STAGES), end)

    lines = [
        "Partial P2P latency — capture available → viewer submit (clock-qualified)",
        "",
    ]
    for track, groups in sorted(tracks.items()):
        lines.extend(
            [
                track,
                f"  {'Stage interval':<{width}}  {'Mean ms':>9}  {'p95 ms':>9}  {'± clock ms':>10}  {'n':>6}",
            ]
        )
        for group, title in (
            ("contributors", "Contributors (pipeline order)"),
            ("partial_spans", "Spans across missing intermediate markers"),
            ("diagnostics", "Diagnostics (overlap; do not add to total)"),
        ):
            if groups.get(group):
                lines.append("  " + title)
                lines.extend(
                    row(label, metric)
                    for label, metric in sorted(groups[group], key=order)
                )
        # Measure the endpoints directly: summing diagnostic intervals double-counts time.
        total = dict(groups.get("totals", [])).get(TOTAL_SPAN)
        lines.append("  " + "-" * (width + 42))
        lines.append(
            row("TOTAL (capture → viewer submit)", total)
            if total
            else "  TOTAL: unavailable"
        )
        lines.append("")
    lines.append(
        f"Clock uncertainty budget ±{uncertainty_budget_ms} ms; "
        f"{snapshot['counters'].get('unqualified_spans', 0)} unqualified spans (session cumulative)"
    )
    return "\n".join(lines) + "\n"
