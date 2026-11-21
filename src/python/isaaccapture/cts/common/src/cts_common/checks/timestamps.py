# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..frames import Frame
from .base import Check, Outcome, Severity, Status


class Monotonic(Check):
    name = "timestamps.monotonic"
    severity = Severity.HARD
    summary = "Sample timestamps never step backwards"
    min_frames = 2

    def __init__(self) -> None:
        super().__init__()
        self.previous_sample: int | None = None
        self.previous_log: int | None = None
        self.sample_regressions = 0
        self.log_regressions = 0
        self.worst_step_ns = 0
        self.first_frame: int | None = None

    def _update(self, frame: Frame) -> None:
        if frame.sample_time_ns is not None:
            if (
                self.previous_sample is not None
                and frame.sample_time_ns < self.previous_sample
            ):
                self.sample_regressions += 1
                step = self.previous_sample - frame.sample_time_ns
                self.worst_step_ns = max(self.worst_step_ns, step)
                if self.first_frame is None:
                    self.first_frame = frame.sequence
            self.previous_sample = frame.sample_time_ns

        if self.previous_log is not None and frame.log_time_ns < self.previous_log:
            self.log_regressions += 1
        self.previous_log = frame.log_time_ns

    def _result(self) -> Outcome:
        measurements = {
            "sample_regressions": self.sample_regressions,
            "log_time_regressions": self.log_regressions,
            "worst_step_ms": self.worst_step_ns / 1e6,
            "first_frame": self.first_frame,
        }
        if self.sample_regressions == 0 and self.log_regressions == 0:
            return Outcome(Status.PASS, "monotonic", measurements)
        return Outcome(
            Status.FAIL,
            f"{self.sample_regressions} backward sample-time steps "
            f"({self.log_regressions} in logTime), worst "
            f"{self.worst_step_ns / 1e6:.1f} ms, first at frame {self.first_frame}",
            measurements,
        )


class AvailableNotBeforeSample(Check):
    name = "timestamps.available_not_before_sample"
    severity = Severity.HARD
    summary = "A sample is never available before it was taken"

    def __init__(self) -> None:
        super().__init__()
        self.samples = 0
        self.inversions = 0
        self.worst_ns = 0

    def _update(self, frame: Frame) -> None:
        if frame.available_time_ns is None or frame.sample_time_ns is None:
            return
        self.samples += 1
        deficit = frame.sample_time_ns - frame.available_time_ns
        if deficit > 0:
            self.inversions += 1
            self.worst_ns = max(self.worst_ns, deficit)

    def _result(self) -> Outcome:
        if self.samples == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no timestamps present")
        measurements = {
            "samples": self.samples,
            "inversions": self.inversions,
            "worst_ms": self.worst_ns / 1e6,
        }
        if self.inversions == 0:
            return Outcome(Status.PASS, "available >= sample", measurements)
        return Outcome(
            Status.FAIL,
            f"{self.inversions} of {self.samples} records became available up to "
            f"{self.worst_ns / 1e6:.1f} ms before the sample was taken",
            measurements,
        )


class DeviceClockDistinct(Check):
    name = "timestamps.device_clock_distinct"
    severity = Severity.ADVISORY
    summary = "The raw device clock is not a copy of the local common clock"

    # Largest fraction of records whose device and common-clock times may be identical.
    MAX_IDENTICAL_RATE = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.samples = 0
        self.identical = 0

    def _update(self, frame: Frame) -> None:
        if frame.device_time_ns is None or frame.sample_time_ns is None:
            return
        self.samples += 1
        if frame.device_time_ns == frame.sample_time_ns:
            self.identical += 1

    def _result(self) -> Outcome:
        if self.samples == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no timestamps present")
        rate = self.identical / self.samples
        measurements = {
            "samples": self.samples,
            "identical": self.identical,
            "rate": rate,
        }
        if rate <= self.MAX_IDENTICAL_RATE:
            return Outcome(
                Status.PASS, f"{rate:.1%} of records share a value", measurements
            )
        return Outcome(
            Status.FAIL,
            f"sample_time_raw_device_clock equals sample_time_local_common_clock on "
            f"{rate:.0%} of records, so there is no independent device clock for "
            f"cross-device sync",
            measurements,
        )
