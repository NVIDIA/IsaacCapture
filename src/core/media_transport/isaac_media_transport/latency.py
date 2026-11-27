# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded stage telemetry and qualified monotonic clock differences."""

import atexit
from collections import Counter, deque
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
import uuid

logger = logging.getLogger("isaaccapture.remote_assistance.latency")
SCHEMA = "umb.latency.v1"


def clock_identity():
    """Identify a host boot; an unavailable boot ID forbids cross-process joins."""
    host = socket.gethostname()
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not boot:
            raise OSError("Empty boot identifier")
    except OSError:
        boot = f"process-{os.getpid()}-{uuid.uuid4().hex}"
    return dict(host_id=host, boot_id=boot, clock_id=f"{host}:{boot}")


@dataclass(frozen=True)
class ClockMap:
    """Target-minus-source offset bounds, conditional on the stated drift budget."""

    source_clock_id: str
    target_clock_id: str
    offset_lower_ns: int
    offset_upper_ns: int
    measured_source_ns: int
    measured_target_ns: int
    max_age_ns: int
    max_relative_drift_ppm: float
    method: str = "ssh_4timestamp"

    def __post_init__(self):
        integer_fields = (
            self.offset_lower_ns,
            self.offset_upper_ns,
            self.measured_source_ns,
            self.measured_target_ns,
            self.max_age_ns,
        )
        if (
            any(type(value) is not int for value in integer_fields)
            or not self.source_clock_id
            or not self.target_clock_id
            or self.source_clock_id == self.target_clock_id
            or not self.method
            or self.offset_lower_ns > self.offset_upper_ns
            or min(self.measured_source_ns, self.measured_target_ns) < 0
            or self.max_age_ns <= 0
            or not math.isfinite(self.max_relative_drift_ppm)
            or self.max_relative_drift_ppm < 0
        ):
            raise ValueError("Invalid monotonic clock mapping")

    def interval(self, source_ns, target_now_ns):
        """Return offset bounds expanded for age; reject stale or reversed clocks."""
        if any(
            type(value) is not int or value < 0 for value in (source_ns, target_now_ns)
        ):
            raise ValueError("Clock mapping requires nonnegative integer timestamps")
        source_age = abs(source_ns - self.measured_source_ns)
        target_age = target_now_ns - self.measured_target_ns
        if target_age < 0:
            raise ValueError("Clock mapping is from the future or target clock stepped")
        age = max(source_age, target_age)
        if age > self.max_age_ns:
            raise ValueError("Stale clock mapping")
        drift = math.ceil(age * self.max_relative_drift_ppm / 1_000_000)
        return self.offset_lower_ns - drift, self.offset_upper_ns + drift


def duration(start, end, *, clock_map=None, max_uncertainty_ns=None):
    """Measure a single stage interval, retaining invalid and missing status."""
    result = dict(
        status="missing_stage", estimate_ns=None, lower_ns=None, upper_ns=None
    )
    if start is None or end is None:
        return result
    key_fields = ("run_id", "track", "source_timestamp_ns")
    if any(event.get(field) is None for event in (start, end) for field in key_fields):
        return {**result, "status": "missing_frame_key"}
    if any(start.get(field) != end.get(field) for field in key_fields):
        return {**result, "status": "frame_mismatch"}
    if not start.get("clock_id") or not end.get("clock_id"):
        return {**result, "status": "missing_clock"}
    if any(
        type(event.get("monotonic_ns")) is not int or event["monotonic_ns"] < 0
        for event in (start, end)
    ):
        return {**result, "status": "invalid_timestamp"}
    delta = end["monotonic_ns"] - start["monotonic_ns"]
    if start["clock_id"] == end["clock_id"]:
        lower = upper = delta
    else:
        if clock_map is None:
            return {**result, "status": "unmapped_clock"}
        if (start["clock_id"], end["clock_id"]) != (
            clock_map.source_clock_id,
            clock_map.target_clock_id,
        ):
            return {**result, "status": "clock_mismatch"}
        try:
            offset_lower, offset_upper = clock_map.interval(
                start["monotonic_ns"], end["monotonic_ns"]
            )
        except ValueError as exc:
            return {**result, "status": "invalid_clock_map", "reason": str(exc)}
        lower, upper = delta - offset_upper, delta - offset_lower
        result.update(
            mapping_method=clock_map.method,
            mapping_age_ns=end["monotonic_ns"] - clock_map.measured_target_ns,
            max_relative_drift_ppm=clock_map.max_relative_drift_ppm,
        )
    uncertainty = (upper - lower) / 2
    result.update(lower_ns=lower, upper_ns=upper, uncertainty_ns=uncertainty)
    if upper < 0:
        return {**result, "status": "clock_order_error"}
    if lower < 0:
        return {**result, "status": "uncertain_order"}
    if max_uncertainty_ns is not None and uncertainty > max_uncertainty_ns:
        return {**result, "status": "uncertainty_exceeded"}
    return {**result, "status": "ok", "estimate_ns": (lower + upper) / 2}


def contributors(events, stages, **options):
    """Adjacent intervals are disjoint; the separate total must not be summed."""
    if len(stages) < 2 or len(set(stages)) != len(stages):
        raise ValueError("At least two distinct ordered stages are required")
    by_stage = {}
    duplicate = set()
    for event in events:
        stage = event.get("stage")
        if stage not in stages:
            continue
        if stage in by_stage:
            duplicate.add(stage)
        else:
            by_stage[stage] = event

    def measure(first, last):
        if first in duplicate or last in duplicate:
            return dict(status="duplicate_stage", estimate_ns=None)
        return duration(by_stage.get(first), by_stage.get(last), **options)

    return dict(
        contributors={
            f"{first}->{last}": measure(first, last)
            for first, last in zip(stages, stages[1:])
        },
        total=measure(stages[0], stages[-1]),
    )


class _DisabledRecorder:
    enabled = False
    dropped = 0
    closed = False

    def record(self, stage, track, source_timestamp_ns, *, at_ns=None, details=None):
        return False

    def close(self, timeout=5):
        return True


DISABLED = _DisabledRecorder()
_recorders = {}
_registry_lock = threading.Lock()


class Recorder:
    enabled = True

    def __init__(
        self,
        log_path,
        run_id,
        *,
        queue_capacity=4096,
        summary_interval_s=5,
        file_enabled=True,
        event_endpoint=None,
    ):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("Latency telemetry requires a nonempty run_id")
        if (
            type(queue_capacity) is not int
            or queue_capacity <= 0
            or not math.isfinite(summary_interval_s)
            or summary_interval_s <= 0
        ):
            raise ValueError(
                "Latency queue capacity and summary interval must be positive"
            )
        self.path = Path(log_path).expanduser().resolve()
        self._stream = None
        if file_enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8")
        self.events = None
        if event_endpoint is not None:
            from .latency_bus import EventBuffer

            try:
                self.events = EventBuffer(event_endpoint)
            except (OSError, ValueError):
                if self._stream is not None:
                    self._stream.close()
                raise
        self.base = dict(
            schema=SCHEMA, run_id=run_id, pid=os.getpid(), **clock_identity()
        )
        self.capacity = queue_capacity
        self.summary_interval_s = summary_interval_s
        self._pending = deque()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self.closed = False
        self.dropped = self.invalid = self.written = 0
        self.error = None
        self._counts = Counter()
        self._thread = threading.Thread(
            target=self._run, name="latency-jsonl", daemon=True
        )
        self._thread.start()

    def record(self, stage, track, source_timestamp_ns, *, at_ns=None, details=None):
        """Offer scalar metadata only; contention or capacity pressure drops telemetry."""
        if self.closed or self.error is not None:
            return False
        if (
            not isinstance(stage, str)
            or not 0 < len(stage) <= 128
            or not isinstance(track, str)
            or not 0 < len(track) <= 256
            or type(source_timestamp_ns) is not int
            or source_timestamp_ns < 0
            or (at_ns is not None and (type(at_ns) is not int or at_ns < 0))
            or not _valid_details(details)
        ):
            self.invalid += 1
            return False
        event = dict(
            record_type="stage",
            stage=stage,
            track=track,
            source_timestamp_ns=source_timestamp_ns,
            monotonic_ns=time.monotonic_ns() if at_ns is None else at_ns,
            details=dict(details) if details else {},
        )
        if not self._lock.acquire(blocking=False):
            self.dropped += 1
            return False
        try:
            if self.closed:
                return False
            if len(self._pending) >= self.capacity:
                self.dropped += 1
                return False
            self._pending.append(event)
        finally:
            self._lock.release()
        self._ready.set()
        return True

    def _write(self, event):
        event = {**self.base, **event}
        if self.events is not None:
            self.events.append(event)
        if self._stream is not None:
            self._stream.write(json.dumps(event, allow_nan=False) + "\n")

    def _status(self):
        self._write(
            dict(
                record_type="telemetry_status",
                monotonic_ns=time.monotonic_ns(),
                written=self.written,
                dropped=self.dropped,
                invalid=self.invalid,
                stages=dict(self._counts),
                closed=self.closed,
            )
        )
        if self._stream is not None:
            self._stream.flush()
        logger.debug(
            "Latency telemetry run=%s stages=%d dropped=%d invalid=%d path=%s",
            self.base["run_id"],
            self.written,
            self.dropped,
            self.invalid,
            self.path,
        )

    def _run(self):
        report_at = time.monotonic() + self.summary_interval_s
        flush_at = time.monotonic() + 0.1
        try:
            while True:
                with self._lock:
                    event = self._pending.popleft() if self._pending else None
                    done = self.closed and event is None
                    if event is None:
                        self._ready.clear()
                if done:
                    break
                if event is not None:
                    self._write(event)
                    self.written += 1
                    stage = event["stage"]
                    if stage not in self._counts and len(self._counts) >= 64:
                        stage = "other_stages"
                    self._counts[stage] += 1
                now = time.monotonic()
                if now >= report_at:
                    self._status()
                    report_at = now + self.summary_interval_s
                if now >= flush_at:
                    # Live joins cannot wait for the slower human-readable summaries.
                    if self._stream is not None:
                        self._stream.flush()
                    flush_at = now + 0.1
                if event is None:
                    self._ready.wait(min(0.1, self.summary_interval_s))
            self._status()
        except (OSError, TypeError, ValueError) as exc:
            self.error = str(exc)
            logger.exception("Latency telemetry writer failed: %s", self.path)
        finally:
            if self._stream is not None:
                self._stream.close()
            if self.events is not None:
                self.events.close()

    def close(self, timeout=5):
        """Drain off the media path; return false if the writer exceeds the deadline."""
        with self._lock:
            self.closed = True
        self._ready.set()
        self._thread.join(timeout)
        return not self._thread.is_alive() and self.error is None


def _valid_details(details):
    if details is None:
        return True
    if not isinstance(details, dict) or len(details) > 16:
        return False
    for key, value in details.items():
        if not isinstance(key, str) or len(key) > 128:
            return False
        if value is not None and type(value) not in (str, int, float, bool):
            return False
        if isinstance(value, str) and len(value) > 256:
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def for_process(settings, role):
    """Keep decode/IPC hooks in the owning process's trace, including in-process viewers."""
    if role not in ("sender", "receiver", "viewer"):
        raise ValueError("Unknown latency process role")
    config = (settings or {}).get("latency") or {}
    if not config.get("enabled", False):
        return settings
    return {**settings, "latency": {**config, "_process_role": role}}


def trace_log_path(run_id, role, log_dir="/tmp"):
    """Return a role-specific filename; expand home paths only on the owning site."""
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id
    ):
        raise ValueError(
            "Automatic trace filenames require a run_id using letters, digits, dots, underscores, or hyphens"
        )
    if role not in ("sender", "receiver", "viewer"):
        raise ValueError(
            "Automatic trace filenames require a sender, receiver, or viewer process role"
        )
    if not isinstance(log_dir, (str, Path)) or not str(log_dir):
        raise ValueError("latency.log_dir must be a nonempty directory path")
    return Path(log_dir) / f"umb-{run_id}-{role}.jsonl"


def get_recorder(settings):
    config = (settings or {}).get("latency") or {}
    if not config.get("enabled", False):
        return DISABLED
    online = bool(config.get("clock", {}).get("client"))
    if not online and not config.get("file_enabled", True):
        return DISABLED
    path = config.get("log_path") or trace_log_path(
        config.get("run_id"), config.get("_process_role"), config.get("log_dir", "/tmp")
    )
    path = Path(path).expanduser().resolve()
    key = (os.getpid(), path)
    with _registry_lock:
        recorder = _recorders.get(key)
        if recorder is not None and not recorder.closed:
            if recorder.base["run_id"] != config.get("run_id"):
                raise ValueError("A latency log_path cannot mix active run IDs")
            return recorder
        if online:
            from .latency_bus import endpoint

        recorder = Recorder(
            path,
            config.get("run_id"),
            queue_capacity=config.get("queue_capacity", 4096),
            summary_interval_s=config.get("summary_interval_s", 5),
            file_enabled=config.get("file_enabled", True),
            event_endpoint=endpoint(
                config.get("run_id"),
                config.get("_process_role"),
                config.get("runtime_dir"),
            )
            if online
            else None,
        )
        _recorders[key] = recorder
        logger.debug(
            "Latency trace: run_id=%s role=%s file=%s",
            config.get("run_id"),
            config.get("_process_role", "explicit"),
            path,
        )
        return recorder


def close_recorders():
    for (pid, _), recorder in list(_recorders.items()):
        if pid == os.getpid():
            recorder.close()


atexit.register(close_recorders)


class ViewerTiming:
    """Bounded host-local measurements; never subtract a sender clock."""

    def __init__(self, settings):
        config = (settings or {}).get("latency") or {}
        enabled = bool(config.get("enabled", False))
        self.terminal_enabled = enabled and bool(config.get("terminal_enabled", False))
        self.hud_enabled = enabled and bool(config.get("hud_enabled", False))
        self.enabled = self.terminal_enabled or self.hud_enabled
        self.interval = float(config.get("report_interval_s", 5))
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("latency.report_interval_s must be finite and positive")
        self._samples = deque(maxlen=600)
        self._lock = threading.Lock()
        self._last_report = time.monotonic()
        self._last_sample = None
        self.online = None
        if enabled and config.get("clock", {}).get("client"):
            from .latency_online import OnlineLatency

            self.online = OnlineLatency(
                config, ipc=settings.get("decoded_handoff") == "ipc"
            )

    def observe(self, received_ns, begin_ns, end_ns):
        if (
            self.online is not None
            or not self.enabled
            or not 0 <= received_ns <= begin_ns <= end_ns
        ):
            return
        # Render-side reads never make the media thread wait for telemetry.
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._samples.append(
                ((begin_ns - received_ns) / 1e6, (end_ns - begin_ns) / 1e6)
            )
            self._last_sample = time.monotonic()
        finally:
            self._lock.release()

    def lines(self):
        if self.online is not None:
            return self.online.lines()
        now = time.monotonic()
        with self._lock:
            samples = list(self._samples)
            last = self._last_sample
        if last is None or now - last > max(2, self.interval):
            return (
                "Host-local latency: waiting for frames",
                "End-to-end: unavailable (no clock mapping)",
            )
        age = sum(s[0] for s in samples) / len(samples)
        submit = sum(s[1] for s in samples) / len(samples)
        return (
            f"Host decode-to-submit: {age:.1f} ms | submit: {submit:.1f} ms",
            "Rolling means | end-to-end unavailable",
        )

    def report(self, emit=print):
        if (
            self.terminal_enabled
            and time.monotonic() - self._last_report >= self.interval
        ):
            self._last_report = time.monotonic()
            emit(
                self.online.report_text()
                if self.online is not None
                else "Latency: " + " — ".join(self.lines())
            )

    def close(self):
        if self.online is not None:
            self.online.close()
