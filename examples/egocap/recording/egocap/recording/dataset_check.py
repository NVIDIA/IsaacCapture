# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Quality checks that read a staged episode's real Zarr arrays and MP4 files.

Every check opens the files written by :class:`~egocap.recording.episode_store.EpisodeStore`;
none consults the writer's own bookkeeping except to compare it against what is on disk.
Mandatory storage-integrity checks (schema, required sources, counts, timestamps, readable
video) are always run and can never be SKIPPED.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import zarr

from .episode_store import TIMESTAMP_FIELDS, NumericSourceSpec, VideoSourceSpec


class Outcome(Enum):
    """Verdict of a single check or of the whole report."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CheckResult:
    """One named check: its outcome, the issues behind a non-passing result, and measured evidence."""

    name: str
    outcome: Outcome
    issues: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    measurements: dict[str, float] = dataclasses.field(default_factory=dict)  # numbers behind the evidence


@dataclass(frozen=True)
class QualityReport:
    """All check results plus the overall outcome (ERROR beats FAILED beats PASSED)."""

    outcome: Outcome
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    check_results: tuple[CheckResult, ...]


def _open(pending_dir: Path) -> zarr.Group:
    """Open the staged episode read-only."""
    return zarr.open_group(str(pending_dir), mode="r")


def _check_schema(
    group: zarr.Group, numeric: tuple[NumericSourceSpec, ...], video: tuple[VideoSourceSpec, ...]
) -> list[str]:
    """Every declared source group exists and holds each declared array with the declared dtype and row shape."""
    issues = []
    wanted: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    for spec in numeric:
        fields = {f.name: (f.dtype, f.shape) for f in spec.fields}
        wanted[spec.name] = {**{t: ("float64", ()) for t in TIMESTAMP_FIELDS}, **fields}
    for vspec in video:
        wanted[vspec.name] = {t: ("float64", ()) for t in TIMESTAMP_FIELDS}
        if vspec.intrinsics:
            wanted[vspec.name]["intrinsics"] = ("float64", (3, 3))
    for source, arrays in wanted.items():
        if source not in group:
            issues.append(f"{source}: group missing")
            continue
        for name, (dtype, shape) in arrays.items():
            if name not in group[source]:
                issues.append(f"{source}/{name}: array missing")
                continue
            array = group[source][name]
            if array.dtype != np.dtype(dtype) or tuple(array.shape[1:]) != shape:
                issues.append(f"{source}/{name}: {array.dtype}{tuple(array.shape[1:])} != declared {dtype}{shape}")
    return issues


def _source_lengths(group: zarr.Group, sources: list[str]) -> dict[str, dict[str, int]]:
    """Row count of every array of every (existing) source."""
    return {s: {n: a.shape[0] for n, a in group[s].arrays()} for s in sources if s in group}


def _decoded_frames(path: Path) -> tuple[int, str | None]:
    """Decode every frame of an MP4, returning (frames decoded, problem or None)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0, "video cannot be opened"
    count = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    return count, None


def check_episode(
    pending_dir: Path,
    numeric: tuple[NumericSourceSpec, ...],
    video: tuple[VideoSourceSpec, ...],
    required_sources: tuple[str, ...],
    expected_counts: Mapping[str, int],
) -> QualityReport:
    """Run the mandatory storage-integrity checks on a staged episode and return the report."""
    results: list[CheckResult] = []
    try:
        group = _open(pending_dir)
    except Exception as exc:  # noqa: BLE001 - an unreadable dataset is an ERROR verdict, not a crash
        err = CheckResult("schema_readable", Outcome.ERROR, (f"cannot open staged dataset: {exc}",))
        return QualityReport(Outcome.ERROR, err.issues, (), (err,))
    names = [s.name for s in numeric] + [s.name for s in video]

    def record(name: str, issues: list[str]) -> None:
        results.append(CheckResult(name, Outcome.FAILED if issues else Outcome.PASSED, tuple(issues)))

    record("schema_readable", _check_schema(group, numeric, video))
    lengths = _source_lengths(group, names)

    record(
        "required_sources_nonempty",
        [
            f"{s}: required source has no rows"
            for s in required_sources
            if not lengths.get(s, {}).get("capture_time")
        ],
    )

    count_issues = []
    for source in names:
        per_array = lengths.get(source, {})
        if len(set(per_array.values())) > 1:
            count_issues.append(f"{source}: array lengths differ {per_array}")
        rows = per_array.get("capture_time", 0)
        if rows != expected_counts.get(source, 0):
            count_issues.append(f"{source}: {rows} rows on disk != {expected_counts.get(source, 0)} accepted")
    record("accepted_equals_written", count_issues)

    time_issues = []
    for source in names:
        if source not in group or "capture_time" not in group[source]:
            continue
        stamps = group[source]["capture_time"][:]
        if not np.all(np.isfinite(stamps)):
            time_issues.append(f"{source}: non-finite capture_time")
        elif np.any(np.diff(stamps) < 0):
            time_issues.append(f"{source}: capture_time goes backwards")
    record("timestamps_valid", time_issues)

    value_issues = []
    for spec in numeric:
        if spec.name not in group:
            continue
        for field in spec.fields:
            if field.name in group[spec.name] and np.dtype(field.dtype).kind == "f":
                if not np.all(np.isfinite(group[spec.name][field.name][:])):
                    value_issues.append(f"{spec.name}/{field.name}: non-finite values")
    record("numeric_values_finite", value_issues)

    video_issues = []
    for vspec in video:
        path = pending_dir / f"{vspec.name}.mp4"
        if not path.is_file():
            video_issues.append(f"{vspec.name}: mp4 missing")
            continue
        decoded, problem = _decoded_frames(path)
        stamped = lengths.get(vspec.name, {}).get("capture_time", 0)
        if problem:
            video_issues.append(f"{vspec.name}: {problem}")
        if decoded != stamped:
            video_issues.append(f"{vspec.name}: {decoded} decodable frames != {stamped} timestamps")
    record("video_readable_and_aligned", video_issues)

    failed = [r for r in results if r.outcome is not Outcome.PASSED]
    overall = Outcome.FAILED if failed else Outcome.PASSED
    return QualityReport(overall, tuple(i for r in failed for i in r.issues), (), tuple(results))


def report_to_dict(report: QualityReport) -> dict:
    """Serialize a report for the episode manifest."""
    return {
        "outcome": report.outcome.value,
        "issues": list(report.issues),
        "warnings": list(report.warnings),
        "check_results": [
            {
                "name": r.name,
                "outcome": r.outcome.value,
                "issues": list(r.issues),
                "evidence": list(r.evidence),
                "measurements": dict(r.measurements),
            }
            for r in report.check_results
        ],
    }


def report_from_dict(data: dict) -> QualityReport:
    """Rebuild a report persisted by :func:`report_to_dict`."""
    return QualityReport(
        Outcome(data["outcome"]),
        tuple(data["issues"]),
        tuple(data["warnings"]),
        tuple(
            CheckResult(
                r["name"],
                Outcome(r["outcome"]),
                tuple(r["issues"]),
                tuple(r.get("evidence", ())),
                dict(r.get("measurements", {})),
            )
            for r in data["check_results"]
        ),
    )
