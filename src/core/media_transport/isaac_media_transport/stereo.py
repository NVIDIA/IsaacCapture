# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded timestamp pairing; sender timestamps are compared only to each other."""

from collections import Counter, deque
import logging
import math
import time

logger = logging.getLogger("isaaccapture.remote_assistance.stereo")


def stereo_tracks(spec):
    if any(not isinstance(spec.get(eye, {}), dict) for eye in ("left", "right")):
        raise ValueError("Stereo left and right settings must be mappings")
    names = tuple(
        spec.get(eye, {}).get("track", f"{spec['name']}.{eye}")
        for eye in ("left", "right")
    )
    if (
        any(not isinstance(n, str) or not n.strip() for n in names)
        or names[0] == names[1]
    ):
        raise ValueError("Stereo tracks must have two distinct nonempty names")
    return names


def camera_specs(settings, spec):
    if not spec.get("stereo", False):
        return [spec]
    kind = spec.get("type")
    if (
        kind not in ("argus", "dexmate_zenoh")
        or settings.get("video_encoder") != "gstreamer"
    ):
        raise ValueError(
            "Encoded stereo requires type: argus or dexmate_zenoh and video_encoder: gstreamer"
        )
    result = []
    for eye, track in zip(("left", "right"), stereo_tracks(spec), strict=True):
        overrides = spec.get(eye, {})
        allowed = (
            {"track", "topic"}
            if kind == "dexmate_zenoh"
            else {"track", "sensor_id", "sensor_mode", "flip_method"}
        )
        if set(overrides) - allowed:
            raise ValueError("Unsupported per-eye camera overrides")
        child = {
            k: v
            for k, v in spec.items()
            if k not in ("left", "right", "stereo_pairing")
        }
        child.update(overrides)
        child.update(stereo=False, name=track)
        if kind == "argus":
            child["sensor_id"] = overrides.get("sensor_id", 0 if eye == "left" else 1)
        elif not isinstance(child.get("topic"), str) or not child["topic"].strip():
            raise ValueError("Each DexSensor eye requires a topic")
        result.append(child)
    if kind == "argus" and result[0]["sensor_id"] == result[1]["sensor_id"]:
        raise ValueError("Stereo Argus sensors must be distinct")
    if kind == "dexmate_zenoh" and result[0]["topic"] == result[1]["topic"]:
        raise ValueError("Stereo DexSensor topics must be distinct")
    return result


class StereoPairer:
    """Match heads immediately; expire unmatched frames instead of building latency."""

    def __init__(self, settings=None, *, clock=time.monotonic_ns):
        settings = settings or {}
        self.max_skew_ms = float(settings.get("max_skew_ms", 8))
        self.max_wait_ms = float(settings.get("max_wait_ms", 25))
        self.queue_size = settings.get("queue_size", 4)
        if not all(
            math.isfinite(v) and v > 0 for v in (self.max_skew_ms, self.max_wait_ms)
        ):
            raise ValueError(
                "Stereo max_skew_ms and max_wait_ms must be positive and finite"
            )
        if type(self.queue_size) is not int or not 1 <= self.queue_size <= 16:
            raise ValueError("Stereo queue_size must be an integer in 1..16")
        self.clock = clock
        self.queues = [deque(), deque()]
        self.last_timestamp = [0, 0]
        self.counters = Counter()
        self.interval = Counter()
        self.samples = [deque(maxlen=2048) for _ in range(3)]
        self.dropped_holds = [deque(maxlen=2048) for _ in range(2)]
        self.last_report = clock()

    def _count(self, key):
        self.counters[key] += 1
        self.interval[key] += 1

    def _drop(self, eye, reason):
        _, frame = self.queues[eye].popleft()
        self.dropped_holds[eye].append((self.clock() - frame.received_ns) / 1e6)
        self._count(f"drop_{('left', 'right')[eye]}_{reason}")

    def reset(self):
        for eye in range(2):
            while self.queues[eye]:
                self._drop(eye, "reset")
        self.last_timestamp = [0, 0]

    def push(self, eye, frame):
        self._count(f"{('left', 'right')[eye]}_frames")
        stamp = getattr(frame.metadata, "user_timestamp", 0)
        if not isinstance(stamp, int) or stamp <= 0:
            self._count(f"drop_{('left', 'right')[eye]}_missing_timestamp")
            return
        if stamp <= self.last_timestamp[eye]:
            self._count(f"drop_{('left', 'right')[eye]}_old_timestamp")
            return
        self.last_timestamp[eye] = stamp
        queue = self.queues[eye]
        if len(queue) >= self.queue_size:
            self._drop(eye, "overflow")
        queue.append((stamp, frame))

    def pop(self):
        now = self.clock()
        for eye, queue in enumerate(self.queues):
            while queue and now - queue[0][1].received_ns > self.max_wait_ms * 1e6:
                self._drop(eye, "timeout")
        left, right = self.queues
        while left and right:
            skew = left[0][0] - right[0][0]
            if abs(skew) > self.max_skew_ms * 1e6:
                self._drop(0 if skew < 0 else 1, "skew")
                continue
            _, lframe = left.popleft()
            _, rframe = right.popleft()
            wait = (rframe.received_ns - lframe.received_ns) / 1e6
            self._count("pairs")
            if wait:
                early = "left" if wait > 0 else "right"
                self._count(f"waited_{early}")
                logger.debug(
                    "Stereo pairing delayed %s by %.3f ms waiting for the other eye",
                    early,
                    abs(wait),
                )
            if abs(wait) > 1:
                self._count("wait_gt_1ms")
            self.samples[0].append((now - lframe.received_ns) / 1e6)
            self.samples[1].append((now - rframe.received_ns) / 1e6)
            self.samples[2].append(abs(skew) / 1e6)
            return lframe, rframe, now
        return None

    def stats(self):
        return {
            **self.counters,
            "queued_left": len(self.queues[0]),
            "queued_right": len(self.queues[1]),
        }

    @staticmethod
    def _summary(samples):
        values = sorted(samples)
        if not values:
            return "0.00/0.00/0.00"
        return f"{sum(values) / len(values):.2f}/{values[math.ceil(0.95 * len(values)) - 1]:.2f}/{values[-1]:.2f}"

    def report(self, name, *, force=False):
        now = self.clock()
        if not force and now - self.last_report < 5_000_000_000:
            return
        elapsed = max((now - self.last_report) / 1e9, 0.001)
        pairs = self.interval["pairs"]
        fraction = 100 * self.interval["wait_gt_1ms"] / pairs if pairs else 0
        logger.info(
            "Stereo pairing %s: pairs=%d (%.1f fps) delayed_left=%d delayed_right=%d "
            "wait_gt_1ms=%.1f%% hold_ms_left(avg/p95/max)=%s hold_ms_right=%s "
            "source_skew_ms=%s dropped_hold_ms_left=%s dropped_hold_ms_right=%s "
            "max_wait_ms=%.1f max_skew_ms=%.1f drops=%s queued=%s; "
            "timestamps=sender-GStreamer-PTS (not verified exposure sync)",
            name,
            pairs,
            pairs / elapsed,
            self.interval["waited_left"],
            self.interval["waited_right"],
            fraction,
            *(self._summary(s) for s in self.samples),
            *(self._summary(s) for s in self.dropped_holds),
            self.max_wait_ms,
            self.max_skew_ms,
            {k: v for k, v in self.interval.items() if k.startswith("drop_")},
            tuple(map(len, self.queues)),
        )
        if pairs and fraction >= 50:
            logger.warning(
                "Stereo pairing %s: %.1f%% of pairs delayed one eye by >1 ms; coordination adds latency",
                name,
                fraction,
            )
        if not pairs and sum(self.interval.values()):
            logger.warning(
                "Stereo pairing %s: no matched pairs; check both tracks, source timestamp skew and wait budget",
                name,
            )
        self.interval.clear()
        for sample in (*self.samples, *self.dropped_holds):
            sample.clear()
        self.last_report = now
