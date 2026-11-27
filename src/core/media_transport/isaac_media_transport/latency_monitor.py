# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join live frame traces using bounded SSH monotonic clock measurements."""

import argparse
import asyncio
from collections import Counter, OrderedDict, defaultdict, deque
from contextlib import suppress
from dataclasses import asdict
import heapq
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import sys
import time

from .latency import ClockMap, trace_log_path

STAGES = (
    "capture_available",
    "encoder_submit",
    "encoded_ready",
    "access_unit_ready",
    "publish_begin",
    "transport_submit",
    "transport_available",
    "decode_begin",
    "decode_ready",
    "pair_ready",
    "ipc_export_begin",
    "ipc_export_ready",
    "ipc_import_begin",
    "ipc_copy_ready",
    "viewer_take",
    "viewer_ready",
    "viewer_submit_begin",
    "viewer_submit_end",
)
MAX_LINE = 16384
MAX_BATCH = 256 * 1024
# Python can regain the GIL after rendering consumed the submitted native frame.
PRIMARY_STAGES = STAGES[:-1]
TOTAL_SPAN = PRIMARY_STAGES[0] + "->" + PRIMARY_STAGES[-1]
# These overlap the primary transport interval; never add them to its sum.
TRANSPORT_DIAGNOSTICS = (
    ("transport_submit", "transport_return"),
    ("transport_submit", "sdk_native_available"),
    ("sdk_native_available", "sdk_callback"),
    ("sdk_callback", "transport_available"),
)
EXTRA_STAGES = {s for pair in TRANSPORT_DIAGNOSTICS for s in pair} - set(STAGES)


class JsonlTail:
    """Read complete records with bounded work, refusing rotation/truncation."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.offset = 0
        self.identity = None

    def follow(self):
        """Start at the last complete line, leaving existing trace data intact."""
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                self.identity = (stat.st_dev, stat.st_ino)
                begin = max(0, stat.st_size - MAX_LINE - 1)
                stream.seek(begin)
                suffix = stream.read(stat.st_size - begin)
                newline = suffix.rfind(b"\n")
                if newline < 0 and begin:
                    raise ValueError("Trace record exceeds size limit")
                self.offset = begin + newline + 1
        except FileNotFoundError:
            pass  # A process may create its trace after the monitor starts.
        return self.offset

    def status(self):
        """Inspect the last complete stage without consuming the live tail."""
        status = dict(path=str(self.path), size=0, run_id=None, stage_monotonic_ns=None)
        try:
            with self.path.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                begin = max(0, size - MAX_BATCH)
                stream.seek(begin)
                lines = stream.read(MAX_BATCH).splitlines(keepends=True)
                if begin:
                    lines = lines[1:]
                status["size"] = size
                for line in reversed(lines):
                    if not line.endswith(b"\n"):
                        continue
                    if len(line) > MAX_LINE:
                        raise ValueError("Trace record exceeds size limit")
                    event = json.loads(line)
                    if event.get("record_type") == "stage":
                        status["run_id"] = event.get("run_id")
                        status["stage_monotonic_ns"] = event.get("monotonic_ns")
                        break
        except FileNotFoundError:
            pass
        return status

    def read(self):
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if self.identity not in (None, identity) or stat.st_size < self.offset:
                    raise ValueError("Trace was replaced or truncated; use a fresh run")
                self.identity = identity
                stream.seek(self.offset)
                records, consumed = [], 0
                while consumed < MAX_BATCH:
                    line = stream.readline(MAX_LINE + 1)
                    if len(line) > MAX_LINE:
                        raise ValueError("Trace record exceeds size limit")
                    if not line.endswith(b"\n"):
                        break
                    consumed += len(line)
                    self.offset += len(line)
                    records.append(json.loads(line))
                return records
        except FileNotFoundError:
            if self.identity is not None:
                raise ValueError("Trace disappeared during collection") from None
            return []


def clock_identity():
    return (
        socket.gethostname()
        + ":"
        + Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    )


def serve():
    tails = [JsonlTail(path) for path in sys.argv[1:]]
    clock_id = clock_identity()
    for line in sys.stdin:
        received = time.monotonic_ns()
        request = json.loads(line)
        response = {"seq": request["seq"], "clock_id": clock_id}
        if request["kind"] == "probe":
            response.update(client_received_ns=received)
            response["client_sent_ns"] = time.monotonic_ns()
        elif request["kind"] == "read":
            response["events"] = [event for tail in tails for event in tail.read()]
        elif request["kind"] == "follow":
            response["offsets"] = [tail.follow() for tail in tails]
        elif request["kind"] == "status":
            response["traces"] = [tail.status() for tail in tails]
        else:
            raise ValueError("Unknown monitor request")
        print(json.dumps(response, separators=(",", ":")), flush=True)


def helper_script():
    # Shipping this stdlib helper avoids installing receiver dependencies on Orin.
    return "\n".join(
        (
            "import json, os, socket, sys, time\nfrom pathlib import Path",
            f"MAX_LINE = {MAX_LINE}\nMAX_BATCH = {MAX_BATCH}",
            inspect.getsource(JsonlTail),
            inspect.getsource(clock_identity),
            inspect.getsource(serve),
            "serve()",
        )
    )


class ClockSamples:
    def __init__(
        self,
        target_clock_id,
        *,
        max_age_ns=2_000_000_000,
        drift_ppm=1000,
        uncertainty_ns=5_000_000,
    ):
        self.target = target_clock_id
        self.max_age_ns = max_age_ns
        self.drift_ppm = drift_ppm
        self.uncertainty_ns = uncertainty_ns
        self.source = None
        self.samples = deque(maxlen=16)

    def add(self, h1, c2, c3, h4, source):
        if not h1 <= h4 or not c2 <= c3 or h4 - h1 < c3 - c2:
            raise ValueError("Invalid four-timestamp ordering or negative interval")
        if self.source is not None and self.source != source:
            self.samples.clear()
            raise ValueError("Remote clock identity changed; start a fresh run")
        self.source = source
        middle = (c2 + c3) // 2
        margin = math.ceil((c3 - c2) * self.drift_ppm / 2e6) + 1000
        sample = ClockMap(
            source_clock_id=source,
            target_clock_id=self.target,
            offset_lower_ns=h1 - c2 - margin,
            offset_upper_ns=h4 - c3 + margin,
            measured_source_ns=middle,
            measured_target_ns=h4,
            max_age_ns=self.max_age_ns,
            max_relative_drift_ppm=self.drift_ppm,
        )
        for previous in self.samples:
            try:
                lo, hi = previous.interval(middle, h4)
            except ValueError:
                continue
            if hi < sample.offset_lower_ns or lo > sample.offset_upper_ns:
                self.samples.clear()
                gap_ns = max(sample.offset_lower_ns - hi, lo - sample.offset_upper_ns)
                raise ValueError(
                    "Clock samples disagree beyond the drift allowance: "
                    f"offset intervals separated by {gap_ns / 1e6:.3f} ms "
                    f"with {self.drift_ppm:g} ppm allowed"
                )
        self.samples.append(sample)
        return sample

    def best(self, source_ns, now_ns):
        candidates = []
        for sample in self.samples:
            try:
                lower, upper = sample.interval(source_ns, now_ns)
            except ValueError:
                continue
            candidates.append((upper - lower, lower, upper, sample))
        if not candidates:
            raise ValueError("No fresh clock measurement")
        _, lower, upper, sample = min(candidates, key=lambda item: item[0])
        if (upper - lower) / 2 > self.uncertainty_ns:
            raise ValueError("Clock uncertainty exceeds the configured budget")
        return lower, upper, sample


class RemoteReader:
    def __init__(self, process):
        self.process = process
        self.sequence = 0

    async def request(self, kind):
        self.sequence += 1
        request = json.dumps({"seq": self.sequence, "kind": kind}).encode() + b"\n"
        started = time.monotonic_ns()
        self.process.stdin.write(request)
        try:
            async with asyncio.timeout(3):
                await self.process.stdin.drain()
                line = await self.process.stdout.readline()
        except TimeoutError as exc:
            raise TimeoutError(
                f"SSH trace helper {kind} timed out after 3 seconds; "
                "check client power and network connectivity"
            ) from exc
        ended = time.monotonic_ns()
        if not line:
            raise ConnectionError("SSH trace helper ended")
        response = json.loads(line)
        if response.get("seq") != self.sequence:
            raise ValueError("SSH response sequence mismatch")
        return started, ended, response

    async def close(self):
        self.process.stdin.close()
        try:
            async with asyncio.timeout(2):
                await self.process.wait()
        except TimeoutError:
            self.process.terminate()
            try:
                async with asyncio.timeout(2):
                    await self.process.wait()
            except TimeoutError:
                self.process.kill()
                await self.process.wait()


def interval_between(start, end, clocks, now_ns):
    first, last = start["monotonic_ns"], end["monotonic_ns"]
    mapping = {}
    if start["clock_id"] == end["clock_id"]:
        lower = upper = last - first
    elif (start["clock_id"], end["clock_id"]) == (clocks.source, clocks.target):
        lo, hi, sample = clocks.best(first, now_ns)
        lower, upper = last - first - hi, last - first - lo
        mapping = dict(
            clock_sample_target_ns=sample.measured_target_ns,
            clock_age_ms=(now_ns - sample.measured_target_ns) / 1e6,
            relative_drift_assumption_ppm=sample.max_relative_drift_ppm,
        )
    else:
        raise ValueError("Unmapped clock domains")
    if upper < 0:
        raise ValueError("Stage timing is inconsistent with clock bounds")
    return dict(
        status="ok" if lower >= 0 else "uncertain_order",
        estimate_ms=(lower + upper) / 2e6 if lower >= 0 else None,
        lower_ms=lower / 1e6,
        upper_ms=upper / 1e6,
        uncertainty_ms=(upper - lower) / 2e6,
        **mapping,
    )


class OnlineJoiner:
    def __init__(
        self,
        run_id,
        clocks,
        *,
        join_ns=1_000_000_000,
        capacity=4096,
        collection_start_ns=0,
        measurement_start_ns=0,
        measurement_end_ns=None,
        window_size=10000,
    ):
        self.run_id, self.clocks = run_id, clocks
        self.join_ns, self.capacity = join_ns, capacity
        self.frames = OrderedDict()
        self.completed_timestamp = {}
        self.counters = Counter()
        self.track_counters = defaultdict(Counter)
        self.collection_start_ns = collection_start_ns
        self.measurement_start_ns = measurement_start_ns
        self.measurement_end_ns = measurement_end_ns
        self.telemetry = {}
        self.samples = defaultdict(lambda: deque(maxlen=window_size))
        self.uncertainties = defaultdict(lambda: deque(maxlen=window_size))
        self.totals = defaultdict(lambda: [0, 0.0])

    def add(self, event, now_ns):
        if event.get("run_id") != self.run_id:
            self.counters["other_run_events"] += 1
            return
        if event.get("record_type") == "telemetry_status":
            self.counters["telemetry_status_records"] += 1
            identity = f"{event['clock_id']}:{event['pid']}"
            self.telemetry[identity] = {
                key: event.get(key)
                for key in ("written", "dropped", "invalid", "closed")
            }
            return
        if event.get("stage", "").startswith("sdk_stats"):
            self.counters["sdk_stats_records"] += 1
            return
        if event.get("stage") not in set(STAGES) | EXTRA_STAGES:
            self.counters["unknown_stage_events"] += 1
            return
        key = (event["track"], event["source_timestamp_ns"])
        if key not in self.frames and key[1] <= self.completed_timestamp.get(
            key[0], -1
        ):
            self.counters["late_events"] += 1
            return
        if key not in self.frames:
            if len(self.frames) >= self.capacity:
                self.counters["join_capacity_drops"] += 1
                return
            self.frames[key] = (now_ns, {})
        events = self.frames[key][1]
        stage = event["stage"]
        if stage in events:
            self.counters["duplicate_stage_events"] += 1
            events[stage] = None  # Repeated submissions must not invent a frame match.
            return
        events[stage] = event

    def phase(self, events, now_ns):
        """Qualify capture time; a late viewer arrival cannot bypass warmup."""
        capture = events.get("capture_available")
        if capture is None:
            return "unqualified", "Missing capture_available", None
        at = capture["monotonic_ns"]
        if capture["clock_id"] == self.clocks.target:
            lower = upper = at
        elif capture["clock_id"] == self.clocks.source:
            try:
                lo, hi, _ = self.clocks.best(at, now_ns)
            except ValueError as exc:
                return "unqualified", str(exc), None
            lower, upper = at + lo, at + hi
        else:
            return "unqualified", "Unmapped capture clock", None
        bounds = dict(lower_ns=lower, upper_ns=upper)
        if upper < self.collection_start_ns:
            return "historical", None, bounds
        if lower >= self.collection_start_ns and upper < self.measurement_start_ns:
            return "warmup", None, bounds
        if lower >= self.measurement_start_ns and (
            self.measurement_end_ns is None or upper < self.measurement_end_ns
        ):
            return "measurement", None, bounds
        if self.measurement_end_ns is not None and lower >= self.measurement_end_ns:
            return "after_measurement", None, bounds
        return "boundary", "Capture interval overlaps a phase boundary", bounds

    def flush(self, now_ns, *, force=False):
        rows = []
        for key, (created, events) in list(self.frames.items()):
            if not force and now_ns - created < self.join_ns:
                continue
            del self.frames[key]
            self.completed_timestamp[key[0]] = max(
                key[1], self.completed_timestamp.get(key[0], -1)
            )
            available = [stage for stage in STAGES if events.get(stage)]
            phase, phase_reason, capture_bounds = self.phase(events, now_ns)
            row = dict(
                record_type="frame_result",
                run_id=self.run_id,
                track=key[0],
                source_timestamp_ns=key[1],
                phase=phase,
                phase_reason=phase_reason,
                capture_host_interval_ns=capture_bounds,
                contributors={},
                partial_spans={},
                diagnostics={},
                totals={},
                missing_stages=[
                    stage for stage in PRIMARY_STAGES if not events.get(stage)
                ],
                missing_diagnostics=[
                    stage
                    for stage in STAGES[len(PRIMARY_STAGES) :]
                    if not events.get(stage)
                ],
            )
            spans = [
                (
                    a,
                    b,
                    "diagnostics"
                    if b == "viewer_submit_end"
                    else "contributors"
                    if STAGES.index(b) == STAGES.index(a) + 1
                    else "partial_spans",
                )
                for a, b in zip(available, available[1:])
            ]
            spans.append((PRIMARY_STAGES[0], PRIMARY_STAGES[-1], "totals"))
            spans.extend(
                (a, b, "diagnostics")
                for a, b in TRANSPORT_DIAGNOSTICS
                if events.get(a) and events.get(b)
            )
            for first, last, group in spans:
                name = first + "->" + last
                try:
                    if not events.get(first) or not events.get(last):
                        raise ValueError("Missing total endpoint")
                    metric = interval_between(
                        events[first], events[last], self.clocks, now_ns
                    )
                    metric_key = f"{key[0]}/{group}/{name}"
                    if metric["estimate_ms"] is not None and phase == "measurement":
                        self.samples[metric_key].append(metric["estimate_ms"])
                        self.uncertainties[metric_key].append(metric["uncertainty_ms"])
                        total = self.totals[metric_key]
                        total[0] += 1
                        total[1] += metric["estimate_ms"]
                    elif metric["estimate_ms"] is None:
                        self.counters["unqualified_spans"] += 1
                except ValueError as exc:
                    metric = {"estimate_ms": None, "reason": str(exc)}
                    self.counters["unqualified_spans"] += 1
                row[group][name] = metric
            track_counts = self.track_counters[key[0]]
            track_counts[phase + "_frames"] += 1
            if phase == "measurement":
                qualified = row["totals"][TOTAL_SPAN]["estimate_ms"] is not None
                track_counts["valid_totals" if qualified else "unqualified_totals"] += 1
            self.counters["frames"] += 1
            self.counters[phase + "_frames"] += 1
            self.counters[
                "complete_frames" if not row["missing_stages"] else "partial_frames"
            ] += 1
            rows.append(row)
        return rows

    def summary(self):
        metrics = {}
        for name, values in self.samples.items():
            ordered = sorted(values)
            count, total = self.totals[name]
            metrics[name] = dict(
                count=count,
                mean_ms=total / count,
                p95_window_ms=ordered[math.ceil(len(ordered) * 0.95) - 1],
                window_count=len(ordered),
                max_uncertainty_ms=max(self.uncertainties[name]),
            )
        valid_totals = sum(
            counts["valid_totals"] for counts in self.track_counters.values()
        )
        availability = (
            "measured"
            if valid_totals
            else (
                "no_frame_timings"
                if not self.counters["frames"]
                else "no_qualified_totals"
            )
        )
        return dict(
            record_type="summary",
            run_id=self.run_id,
            counters=dict(self.counters),
            pending_frames=len(self.frames),
            telemetry=self.telemetry,
            availability=availability,
            valid_total_count=valid_totals,
            tracks={
                track: dict(counts) for track, counts in self.track_counters.items()
            },
            metrics=metrics,
        )


def terminal_summary(snapshot, phase):
    drops = sum(value.get("dropped") or 0 for value in snapshot["telemetry"].values())
    lines = [
        f"Latency {phase}: valid totals={snapshot['valid_total_count']} "
        f"pending={snapshot['pending_frames']} telemetry drops={drops}"
    ]
    if phase == "warmup":
        lines.append(
            f"WARMUP — measured statistics start after warmup. "
            f"Frames joined={snapshot['counters'].get('frames', 0)} "
            f"warmup={snapshot['counters'].get('warmup_frames', 0)} "
            f"unqualified={snapshot['counters'].get('unqualified_frames', 0)}"
        )
        return "\n".join(lines)
    if not snapshot["valid_total_count"]:
        lines.append("NO FRAME TIMINGS — not a latency result")
    for track, counts in sorted(snapshot["tracks"].items()):
        total = snapshot["metrics"].get(f"{track}/totals/{TOTAL_SPAN}")
        timing = (
            f"{total['mean_ms']:.2f}/{total['p95_window_ms']:.2f} ms"
            if total
            else "unavailable"
        )
        prefix = f"{track}/contributors/"
        contributors = sorted(
            (
                (name.removeprefix(prefix), metric["mean_ms"])
                for name, metric in snapshot["metrics"].items()
                if name.startswith(prefix)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top = (
            ", ".join(f"{name} {value:.2f} ms" for name, value in contributors)
            or "unavailable"
        )
        unqualified = counts.get("unqualified_totals", 0) + counts.get(
            "unqualified_frames", 0
        )
        excluded = sum(
            value
            for key, value in counts.items()
            if key.endswith("_frames") and key != "measurement_frames"
        )
        lines.append(
            f"  {track}: partial P2P mean/p95={timing} "
            f"n={counts.get('valid_totals', 0)} unqualified={unqualified} "
            f"excluded={excluded}; top 3: {top}"
        )
    return "\n".join(lines)


def write_record(stream, value):
    stream.write(json.dumps(value, separators=(",", ":")) + "\n")
    stream.flush()


def recording_schedule(warmup_ns, duration_ns):
    """Yield recording cues at offsets from the collection's monotonic start."""
    second = 1_000_000_000
    end_ns = warmup_ns + duration_ns
    if warmup_ns:
        yield (
            0,
            "warmup",
            f"PLEASE WAIT {warmup_ns / second:g} SECONDS FOR WARMUP PERIOD",
        )
    else:
        yield 0, "start_recording", "START RECORDING LATENCY VIDEO NOW"
    boundaries = {warmup_ns, end_ns}
    deadlines = sorted(
        offset
        for offset in {
            boundary - remaining * second
            for boundary in boundaries
            for remaining in range(6)
        }
        if offset > 0
    )
    previous = 0
    for offset in heapq.merge(range(5 * second, end_ns + 1, 5 * second), deadlines):
        if offset == previous:
            continue
        previous = offset
        yield offset, "elapsed", f"TIME SINCE START: {offset / second:g} SECONDS"
        if offset == warmup_ns:
            yield offset, "start_recording", "START RECORDING LATENCY VIDEO NOW"
        if offset == end_ns:
            yield (
                offset,
                "stop_recording",
                "MEASUREMENT IS DONE, STOP RECORDING LATENCY VIDEO",
            )


async def recording_guide(collection_start_ns, measurement_start_ns, end_ns, results):
    # Run separately from SSH polling so a slow read cannot hide a recording cue.
    def emit(cue, message, scheduled_ns):
        now = time.monotonic_ns()
        print(message, flush=True)
        write_record(
            results,
            dict(
                record_type="operator_cue",
                cue=cue,
                message=message,
                scheduled_monotonic_ns=scheduled_ns,
                emitted_monotonic_ns=now,
                elapsed_s=(now - collection_start_ns) / 1e9,
            ),
        )

    try:
        for offset, cue, message in recording_schedule(
            measurement_start_ns - collection_start_ns, end_ns - measurement_start_ns
        ):
            deadline = collection_start_ns + offset
            # Deadline wait, not condition polling: an early timer wake must not
            # tell the operator to record before the measured window opens.
            while (remaining := deadline - time.monotonic_ns()) > 0:  # noqa: ASYNC110
                await asyncio.sleep(remaining / 1e9)
            emit(cue, message, deadline)
    except asyncio.CancelledError:
        emit("aborted", "MEASUREMENT ABORTED, STOP RECORDING LATENCY VIDEO", None)
        raise


async def wait_for_traces(remote, host_tails, run_id, *, timeout=10):
    """Require new stage records from every selected log before the countdown."""
    first_stages = {}
    ready = set()
    deadline = time.monotonic() + timeout
    while True:
        _, _, response = await remote.request("status")
        statuses = [("client", status) for status in response["traces"]]
        statuses.extend(("host", tail.status()) for tail in host_tails)
        for site, status in statuses:
            key = (site, status["path"])
            actual = status["run_id"]
            if actual is not None and actual != run_id:
                raise ValueError(
                    f"Trace run ID mismatch: {site} {status['path']} contains "
                    f"{actual!r}, but --run-id is {run_id!r}. Use the same run ID "
                    "in both camera profiles and the monitor; restart media "
                    "processes after profile edits."
                )
            stamp = status["stage_monotonic_ns"]
            first_stage = first_stages.setdefault(key, stamp)
            if actual == run_id and stamp is not None and stamp != first_stage:
                ready.add(key)
        missing = set(first_stages) - ready
        if not missing:
            return
        if time.monotonic() >= deadline:
            paths = "; ".join(f"{site}: {path}" for site, path in sorted(missing))
            raise ValueError(
                f"No fresh frame traces for run {run_id!r}: {paths}. "
                "Start the sender, receiver and viewer with tracing enabled. "
                "Check --run-id against the Latency telemetry run= startup/log "
                "message; old trace files do not establish readiness."
            )
        await asyncio.sleep(0.2)


async def monitor(args):
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=False)
    clocks = ClockSamples(
        clock_identity(),
        max_age_ns=int(args.clock_max_age * 1e9),
        drift_ppm=args.drift_ppm,
        uncertainty_ns=int(args.max_uncertainty_ms * 1e6),
    )
    manifest = dict(
        schema="umb.latency.monitor.v1",
        run_id=args.run_id,
        client=args.client,
        client_logs=args.client_log,
        host_logs=args.host_log,
        duration_s=args.duration,
        warmup_s=args.warmup,
        target_clock_id=clocks.target,
        clock_method="ssh_4timestamp_monotonic",
        clock_max_age_s=args.clock_max_age,
        relative_drift_assumption_ppm=args.drift_ppm,
        max_uncertainty_ms=args.max_uncertainty_ms,
        stages=STAGES,
        primary_stages=PRIMARY_STAGES,
        total_span=TOTAL_SPAN,
        join_timeout_s=1,
        started_unix_ns=time.time_ns(),
        capture_startup=getattr(args, "capture_startup", False),
    )
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    joiner = OnlineJoiner(args.run_id, clocks)
    tails = [JsonlTail(path) for path in args.host_log]
    remote = None
    guide = None
    failure = None
    command = [
        "env",
        "-u",
        "LD_PRELOAD",
        "-u",
        "FAKETIME",
        "-u",
        "FAKETIME_DONT_FAKE_MONOTONIC",
        args.client_python,
        "-u",
        "-c",
        helper_script(),
        *args.client_log,
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("LD_PRELOAD", "FAKETIME", "FAKETIME_DONT_FAKE_MONOTONIC")
    }
    with (
        (output / "ssh.stderr").open("xb") as errors,
        (output / "events.jsonl").open("x") as events_log,
        (output / "results.jsonl").open("x") as results,
    ):
        try:
            print(
                "Connecting to client and checking clocks before warmup...", flush=True
            )
            process = await asyncio.create_subprocess_exec(
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                args.client,
                shlex.join(command),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=errors,
                env=environment,
                limit=MAX_BATCH * max(8, len(args.client_log) * 2),
            )
            remote = RemoteReader(process)

            async def probe():
                h1, h4, response = await remote.request("probe")
                timestamps = dict(
                    host_sent_ns=h1,
                    host_received_ns=h4,
                    client_received_ns=response["client_received_ns"],
                    client_sent_ns=response["client_sent_ns"],
                )
                try:
                    sample = clocks.add(
                        h1,
                        response["client_received_ns"],
                        response["client_sent_ns"],
                        h4,
                        response["clock_id"],
                    )
                except ValueError as exc:
                    write_record(
                        events_log,
                        dict(
                            record_type="clock_rejected",
                            source_clock_id=response["clock_id"],
                            target_clock_id=clocks.target,
                            error=str(exc),
                            **timestamps,
                        ),
                    )
                    raise
                write_record(
                    events_log,
                    dict(
                        record_type="clock_sample",
                        **asdict(sample),
                        **timestamps,
                    ),
                )

            for _ in range(8):
                await probe()
            if not getattr(args, "capture_startup", False):
                print(
                    f"Checking fresh frame traces for run {args.run_id}...", flush=True
                )
                await wait_for_traces(remote, tails, args.run_id)
            else:
                # Startup capture must begin before media, on a fresh run's files.
                _, _, status = await remote.request("status")
                if any(s["size"] for s in status["traces"]) or any(
                    t.status()["size"] for t in tails
                ):
                    raise ValueError("Startup capture requires empty/new trace files")
                print(
                    "STARTUP COLLECTION READY — start the media processes now",
                    flush=True,
                )
            # Replaying unequal per-process histories can complete current frame
            # joins before their sender records arrive. Follow both sites live.
            _, _, following = await remote.request("follow")
            write_record(
                results,
                dict(
                    record_type="trace_follow",
                    client_offsets=following["offsets"],
                    host_offsets=[tail.follow() for tail in tails],
                ),
            )
            joiner.collection_start_ns = time.monotonic_ns()
            joiner.measurement_start_ns = joiner.collection_start_ns + int(
                args.warmup * 1e9
            )
            joiner.measurement_end_ns = joiner.measurement_start_ns + int(
                args.duration * 1e9
            )
            write_record(
                results,
                dict(
                    record_type="measurement_window",
                    collection_start_ns=joiner.collection_start_ns,
                    measurement_start_ns=joiner.measurement_start_ns,
                    measurement_end_ns=joiner.measurement_end_ns,
                ),
            )
            if not getattr(args, "capture_startup", False):
                guide = asyncio.create_task(
                    recording_guide(
                        joiner.collection_start_ns,
                        joiner.measurement_start_ns,
                        joiner.measurement_end_ns,
                        results,
                    )
                )
            next_probe = next_report = time.monotonic()
            while time.monotonic_ns() < joiner.measurement_end_ns:
                if time.monotonic() >= next_probe:
                    await probe()
                    next_probe = time.monotonic() + 1
                _, _, response = await remote.request("read")
                if response["clock_id"] != clocks.source:
                    raise ValueError("Remote clock identity changed")
                records = response["events"]
                for tail in tails:
                    records.extend(tail.read())
                now = time.monotonic_ns()
                for event in records:
                    write_record(events_log, event)
                    joiner.add(event, now)
                for row in joiner.flush(now):
                    write_record(results, row)
                if time.monotonic() >= next_report:
                    snapshot = joiner.summary()
                    write_record(results, snapshot)
                    phase = (
                        "warmup" if now < joiner.measurement_start_ns else "measurement"
                    )
                    print(terminal_summary(snapshot, phase), flush=True)
                    next_report = time.monotonic() + 5
                await asyncio.sleep(0.1)
            if guide:
                await guide
        except (OSError, ValueError, TimeoutError, KeyError) as exc:
            failure = str(exc) or type(exc).__name__
        except asyncio.CancelledError:
            failure = "Interrupted"
            raise
        finally:
            if guide:
                guide.cancel()
                with suppress(asyncio.CancelledError):
                    await guide
            # Finalize before SSH teardown can age the last clock sample by seconds.
            for row in joiner.flush(time.monotonic_ns(), force=True):
                write_record(results, row)
            if remote:
                await remote.close()
            summary = dict(joiner.summary(), error=failure)
            write_record(results, summary)
            print(
                terminal_summary(summary, "aborted" if failure else "finished"),
                flush=True,
            )
            with (output / "summary.json").open("x") as stream:
                json.dump(summary, stream, indent=2)
    if failure:
        raise RuntimeError(f"Collection stopped: {failure}; partial results: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True, help="SSH user@host")
    parser.add_argument(
        "--client-log",
        action="append",
        help="Override the automatic sender log; repeat for additional files",
    )
    parser.add_argument("--client-python", default="python3")
    parser.add_argument(
        "--host-log",
        action="append",
        help="Override the automatic receiver/viewer logs; repeat for additional files",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp",
        help="Trace directory on each site; filenames derive from --run-id and the process role",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument(
        "--capture-startup",
        action="store_true",
        help="Begin before media on fresh trace files; no optical recording cues",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0,
        help="Seconds before the measured duration; retained but excluded",
    )
    parser.add_argument("--clock-max-age", type=float, default=2)
    parser.add_argument("--drift-ppm", type=float, default=1000)
    parser.add_argument("--max-uncertainty-ms", type=float, default=5)
    args = parser.parse_args()
    try:
        if not args.client_log:
            args.client_log = [str(trace_log_path(args.run_id, "sender", args.log_dir))]
        if not args.host_log:
            args.host_log = [
                str(trace_log_path(args.run_id, role, args.log_dir))
                for role in ("receiver", "viewer")
            ]
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.warmup) or args.warmup < 0:
        parser.error("Warmup must be finite and nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+", args.client):
        parser.error("--client must be SSH user@host")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            args.duration,
            args.clock_max_age,
            args.drift_ppm,
            args.max_uncertainty_ms,
        )
    ):
        parser.error("Duration and clock limits must be finite and positive")
    try:
        asyncio.run(monitor(args))
    except (OSError, RuntimeError, KeyboardInterrupt) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
