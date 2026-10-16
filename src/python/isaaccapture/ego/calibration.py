# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit recording-bound clocks from annotated events and independent validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence


VALIDATION_DURATION_NS = 15 * 60 * 1_000_000_000


@dataclass(frozen=True)
class PairedTimeEvent:
    event_id: str
    source_value: int
    reference_time_ns: int
    uncertainty_ns: int
    evidence: str


@dataclass(frozen=True)
class TimeAlignmentFit:
    source_anchor: int
    reference_anchor_ns: int
    scale: float
    source_unit: str
    status: str
    max_abs_error_ns: int
    uncertainty_ns: int
    held_out_duration_ns: int
    drift_ppm: float
    absolute_error_percentiles_ns: tuple[int, int, int]
    evidence: tuple[str, ...]
    reasons: tuple[str, ...]
    validation_start_ns: int
    validation_end_ns: int

    def reference_time(self, source_value: int) -> int:
        return self.reference_anchor_ns + round(
            (source_value - self.source_anchor) * self.scale
        )

    def validation_report(self) -> dict:
        return {
            "status": self.status,
            "method": "held_out_paired_physical_events",
            "max_abs_error_ns": self.max_abs_error_ns,
            "uncertainty_ns": self.uncertainty_ns,
            "held_out_duration_ns": self.held_out_duration_ns,
            "validation_start_ns": self.validation_start_ns,
            "validation_end_ns": self.validation_end_ns,
            "drift_ppm": self.drift_ppm,
            "absolute_error_percentiles_ns": dict(
                zip(("p50", "p95", "p99"), self.absolute_error_percentiles_ns)
            ),
            "evidence": list(self.evidence),
            "reasons": list(self.reasons),
        }


def _events(values: Sequence[PairedTimeEvent], name: str) -> list[PairedTimeEvent]:
    if len(values) < 3:
        raise ValueError(f"{name} needs at least three independently annotated events")
    ids = set()
    for value in values:
        if not value.event_id or value.event_id in ids or not value.evidence:
            raise ValueError(f"{name} needs unique event IDs and evidence references")
        ids.add(value.event_id)
        for number in (
            value.source_value,
            value.reference_time_ns,
            value.uncertainty_ns,
        ):
            if type(number) is not int:
                raise ValueError(
                    "Event coordinates and uncertainty must be integer ns/samples"
                )
        if value.uncertainty_ns < 0:
            raise ValueError("Measurement uncertainty cannot be negative")
    result = sorted(values, key=lambda event: event.source_value)
    if any(
        left.source_value >= right.source_value
        or left.reference_time_ns >= right.reference_time_ns
        for left, right in zip(result, result[1:])
    ):
        raise ValueError(
            f"{name} events must advance within one continuous clock segment"
        )
    return result


def _percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    fraction = location - lower
    return round(
        ordered[lower] * (1 - fraction)
        + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    )


def fit_time_alignment(
    calibration_events: Sequence[PairedTimeEvent],
    validation_events: Sequence[PairedTimeEvent],
    *,
    fps: float,
    validation_start_ns: int,
    validation_end_ns: int,
    source_unit: str = "ns",
    sample_rate_hz: int | None = None,
) -> TimeAlignmentFit:
    """Uncertainty includes event detection, exposure and light/sound source delay."""
    training = _events(calibration_events, "Calibration")
    held_out = _events(validation_events, "Validation")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if (
        type(validation_start_ns) is not int
        or type(validation_end_ns) is not int
        or validation_start_ns >= validation_end_ns
    ):
        raise ValueError("Validation needs a nonempty integer-ns recording interval")
    if source_unit not in ("ns", "samples"):
        raise ValueError("source_unit must be ns or samples")
    if source_unit == "samples" and (
        type(sample_rate_hz) is not int or sample_rate_hz <= 0
    ):
        raise ValueError("Audio fitting needs its nominal sample rate")
    if {event.event_id for event in training} & {
        event.event_id for event in held_out
    } or {(event.source_value, event.reference_time_ns) for event in training} & {
        (event.source_value, event.reference_time_ns) for event in held_out
    }:
        raise ValueError("Calibration and validation events must be disjoint")
    if any(
        not validation_start_ns <= event.reference_time_ns <= validation_end_ns
        for event in held_out
    ):
        raise ValueError("Validation event is outside its declared recording interval")

    # Subtract integer anchors before floating point; epoch-sized ns lose precision.
    source_anchor = training[0].source_value
    reference_origin = training[0].reference_time_ns
    x = [event.source_value - source_anchor for event in training]
    y = [event.reference_time_ns - reference_origin for event in training]
    mean_x, mean_y = math.fsum(x) / len(x), math.fsum(y) / len(y)
    denominator = math.fsum((value - mean_x) ** 2 for value in x)
    scale = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Fitted clock must be finite and advance")
    reference_anchor = reference_origin + round(mean_y - scale * mean_x)
    errors = [
        abs(
            reference_anchor
            + round((event.source_value - source_anchor) * scale)
            - event.reference_time_ns
        )
        for event in held_out
    ]
    uncertainty = max(event.uncertainty_ns for event in (*training, *held_out))
    duration = validation_end_ns - validation_start_ns
    reasons = []
    if duration < VALIDATION_DURATION_NS:
        reasons.append("validation_shorter_than_15_minutes")
    relative_times = [
        (event.reference_time_ns - validation_start_ns) / duration for event in held_out
    ]
    if not (
        any(value <= 0.1 for value in relative_times)
        and any(0.4 <= value <= 0.6 for value in relative_times)
        and any(value >= 0.9 for value in relative_times)
    ):
        reasons.append("missing_beginning_middle_or_end_validation")
    if max(errors) + uncertainty > 1_000_000_000 / (2 * fps):
        reasons.append("physical_error_and_uncertainty_exceed_half_frame")
    nominal_period = 1 if source_unit == "ns" else 1_000_000_000 / sample_rate_hz
    return TimeAlignmentFit(
        source_anchor,
        reference_anchor,
        scale,
        source_unit,
        "unverified" if reasons else "validated",
        max(errors),
        uncertainty,
        duration,
        (scale / nominal_period - 1) * 1_000_000,
        tuple(_percentile(errors, probability) for probability in (0.5, 0.95, 0.99)),
        tuple(sorted({event.evidence for event in (*training, *held_out)})),
        tuple(reasons),
        validation_start_ns,
        validation_end_ns,
    )


def evidence_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_recording_alignment(
    recording: str | Path,
    *,
    model_id: str,
    identity: Mapping,
    profile: Mapping,
    epochs: Sequence[Mapping],
    media_paths: Sequence[str | Path] = (),
) -> dict:
    """Build an explicit MCAP-bound model; each nonreference stream needs a fit."""
    path = Path(recording).resolve(strict=True)
    if not path.is_file():
        raise ValueError("recording must name an MCAP file")
    if not isinstance(model_id, str) or not model_id.strip() or not epochs:
        raise ValueError("A recording model needs an ID and effective epoch intervals")
    for field in ("serial", "firmware", "sdk_version"):
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError(f"Missing identity.{field}")
    fps = profile.get("video", {}).get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("profile.video.fps must be finite and positive")
    document = {
        "version": 1,
        "scope": "recording",
        "status": "validated",
        "reference_stream": "left_rgb",
        "model_id": model_id,
        "identity": dict(identity),
        "profile": dict(profile),
        "recording_fingerprint": {
            "mcap_sha256": evidence_sha256(path),
            "media_sha256": {},
        },
        "epochs": [],
    }
    for media in media_paths:
        media_path = Path(media).resolve(strict=True)
        relative = media_path.relative_to(path.parent).as_posix()
        document["recording_fingerprint"]["media_sha256"][relative] = evidence_sha256(
            media_path
        )
    fits = []
    reports = []
    reasons = set()
    ranges = {}
    for value in epochs:
        epoch = value["epoch"]
        begin, end = value["valid_from_ns"], value["valid_to_ns"]
        if any(type(number) is not int for number in (epoch, begin, end)):
            raise ValueError("Epoch and effective interval must be integer values")
        if (
            epoch < 0
            or begin >= end
            or any(
                begin < previous_end and previous_begin < end
                for previous_begin, previous_end in ranges.get(epoch, [])
            )
        ):
            raise ValueError("Invalid epoch or effective interval")
        ranges.setdefault(epoch, []).append((begin, end))
        streams = {
            "left_rgb": {
                "source_anchor_ns": begin,
                "reference_anchor_ns": begin,
                "scale": 1.0,
            }
        }
        for name, fit in value["streams"].items():
            if name == "left_rgb":
                raise ValueError("left_rgb defines the reference and is not fitted")
            if name == "audio":
                segments = []
                seen = set()
                for segment in fit:
                    result = segment["fit"]
                    first = segment["first_sample_index"]
                    count = segment["sample_count"]
                    segment_id = segment["segment_id"]
                    if any(
                        type(number) is not int for number in (first, count, segment_id)
                    ):
                        raise ValueError("Audio segment coordinates must be integers")
                    if first < 0 or count <= 0 or segment_id < 0 or segment_id in seen:
                        raise ValueError("Invalid or duplicate audio segment")
                    seen.add(segment_id)
                    if (
                        not isinstance(result, TimeAlignmentFit)
                        or result.source_unit != "samples"
                    ):
                        raise ValueError("Audio segment needs a sample-index fit")
                    continuity = segment.get("continuity_verified", False)
                    if type(continuity) is not bool:
                        raise ValueError("continuity_verified must be a boolean")
                    if not continuity:
                        reasons.add("audio_continuity_unverified")
                    segments.append(
                        {
                            "segment_id": segment_id,
                            "first_sample_index": first,
                            "sample_count": count,
                            "reference_anchor_ns": result.reference_time(first),
                            "period_ns": result.scale,
                            "continuity_verified": continuity,
                            "validation_start_ns": result.validation_start_ns,
                            "validation_end_ns": result.validation_end_ns,
                        }
                    )
                    fits.append(result)
                    reports.append(
                        {
                            "epoch": epoch,
                            "stream": name,
                            "segment_id": segment_id,
                            **result.validation_report(),
                        }
                    )
                if not segments:
                    raise ValueError(
                        "Audio model needs at least one continuous segment"
                    )
                streams[name] = {"segments": segments}
            else:
                if name not in ("right_rgb", "accel", "gyro") or not isinstance(
                    fit, TimeAlignmentFit
                ):
                    raise ValueError("Unsupported stream or missing fitted clock")
                if fit.source_unit != "ns":
                    raise ValueError("Video/IMU model must use raw ns")
                if not fit.validation_start_ns <= begin < end <= fit.validation_end_ns:
                    raise ValueError(
                        "Effective epoch interval exceeds independent validation bounds"
                    )
                streams[name] = {
                    "source_anchor_ns": fit.source_anchor,
                    "reference_anchor_ns": fit.reference_anchor_ns,
                    "scale": fit.scale,
                    "validation_start_ns": fit.validation_start_ns,
                    "validation_end_ns": fit.validation_end_ns,
                }
                fits.append(fit)
                reports.append(
                    {
                        "epoch": epoch,
                        "stream": name,
                        **fit.validation_report(),
                    }
                )
        document["epochs"].append(
            {
                "epoch": epoch,
                "valid_from_ns": begin,
                "valid_to_ns": end,
                "streams": streams,
            }
        )
    if not fits:
        raise ValueError("Provide at least one measured nonreference stream")
    reasons.update(reason for fit in fits for reason in fit.reasons)
    error = max(fit.max_abs_error_ns for fit in fits)
    uncertainty = max(fit.uncertainty_ns for fit in fits)
    if error + uncertainty > 500_000_000 / fps:
        reasons.add("physical_error_and_uncertainty_exceed_half_frame")
    if reasons or any(fit.status != "validated" for fit in fits):
        document["status"] = "unverified"
    document["validation"] = {
        "method": "held_out_paired_physical_events",
        "max_abs_error_ns": error,
        "uncertainty_ns": uncertainty,
        "held_out_duration_ns": min(fit.held_out_duration_ns for fit in fits),
        "evidence": sorted({evidence for fit in fits for evidence in fit.evidence}),
        "reasons": sorted(reasons),
        "stream_reports": reports,
    }
    return document
