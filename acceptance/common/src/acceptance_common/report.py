# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running checks over a frame source, and the report that comes out.

Verdict aggregation is a pure function of the results plus each check's declared
severity and attribution. ``fail`` and ``retake`` are kept apart deliberately: telling a
vendor their device is broken when the operator simply squatted too shallow is the
specific failure the acceptance process exists to avoid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite
from typing import Any, Iterable, Sequence

from .checks.base import Attribution, Check, Severity, Status
from .frames import FrameSource, SourceMetadata


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    RETAKE = "retake"
    INSUFFICIENT_DATA = "insufficient_data"


class Mark(StrEnum):
    """How one result is shown, in any renderer.

    Five states, not three. A number nobody judged and a question nobody could answer
    are both weaker claims than a pass, and drawing either one like a pass reads as an
    approval that was never given.
    """

    PASS = "pass"
    FAIL = "FAIL"
    MEAS = "meas"
    UNANSWERED = "n/a"
    NOTE = "note"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    gate: str
    severity: Severity
    attribution: Attribution
    summary: str
    status: Status
    detail: str
    measurements: dict[str, Any]
    required: bool = True
    judged: bool = True

    @property
    def counts_toward_verdict(self) -> bool:
        return self.severity is not Severity.ADVISORY

    @property
    def mark(self) -> Mark:
        if self.status is Status.INSUFFICIENT_DATA:
            return Mark.UNANSWERED
        if self.status is Status.FAIL:
            return Mark.FAIL if self.counts_toward_verdict else Mark.NOTE
        return Mark.PASS if self.judged else Mark.MEAS


@dataclass(frozen=True, slots=True)
class Report:
    source: str
    frames: int
    metadata: SourceMetadata
    results: tuple[CheckResult, ...]
    verdict: Verdict
    notes: tuple[str, ...] = ()

    @property
    def advisories(self) -> tuple[CheckResult, ...]:
        return tuple(
            r
            for r in self.results
            if not r.counts_toward_verdict and r.status is Status.FAIL
        )

    @property
    def unanswered(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.status is Status.INSUFFICIENT_DATA)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(
            r
            for r in self.results
            if r.counts_toward_verdict and r.status is Status.FAIL
        )

    def to_dict(self) -> dict[str, Any]:
        schema_data = self.metadata.schema_data
        return {
            "source": self.source,
            "verdict": str(self.verdict),
            "frames": self.frames,
            "container": {
                "schema_name": self.metadata.schema_name,
                "schema_encoding": self.metadata.schema_encoding,
                "schema_sha256": (
                    hashlib.sha256(schema_data).hexdigest() if schema_data else None
                ),
                "schema_bytes": len(schema_data) if schema_data else None,
                "message_encoding": self.metadata.message_encoding,
                "topic": self.metadata.topic,
                "profile": self.metadata.profile,
                "channel_found": self.metadata.channel_found,
            },
            "notes": list(self.notes),
            "checks": [
                {
                    "name": r.name,
                    "gate": r.gate,
                    # Derived, and carried anyway: a reader who recomputes it from
                    # status, severity and judged reimplements Mark and drifts.
                    "mark": str(r.mark),
                    "severity": str(r.severity),
                    "attribution": str(r.attribution),
                    "judged": r.judged,
                    "status": str(r.status),
                    "required": r.required,
                    "detail": r.detail,
                    "measurements": json_safe(r.measurements),
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        # allow_nan=False so a value this dict cannot represent fails here rather than
        # reaching a strict reader as a bare NaN.
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=False, allow_nan=False
        )

    def to_text(self) -> str:
        lines = [
            f"source   {self.source}",
            f"frames   {self.frames}",
            f"topic    {self.metadata.topic or '-'}  "
            f"schema {self.metadata.schema_name or '-'}",
            f"verdict  {str(self.verdict).upper()}",
            "",
        ]
        for r in self.results:
            lines.append(f"  [{r.mark:<4}] {r.name:<45} {r.detail}")
        unanswered = self.unanswered
        if unanswered:
            lines.append("")
            lines.append(f"  {len(unanswered)} checks could not be answered:")
            lines.extend(f"    {r.name}: {r.detail}" for r in unanswered)
        if self.notes:
            lines.append("")
            lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def json_safe(value: Any) -> Any:
    """Recursively replaces non-finite floats with None.

    Python's ``json`` writes a bare ``NaN`` or ``Infinity``, which every strict reader
    rejects — ``JSON.parse`` included. This is reachable, not defensive:
    the full-body ``coordinate_frame.up_axis`` divides by the second-largest axis
    component and reports an infinite dominance whenever that is exactly zero.
    """
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def suppress_dependents(
    results: Sequence[CheckResult], checks: Sequence[Check]
) -> list[CheckResult]:
    """Turns results whose preconditions failed into "cannot conclude".

    Iterates to a fixed point so suppression propagates along a chain: broken label
    windows make the step-order test unanswerable, which in turn makes every device
    measurement read through those windows unanswerable.
    """
    by_name = {result.name: result for result in results}
    blocked: dict[str, str] = {}
    dependencies = {check.name: check.depends_on for check in checks}

    changed = True
    while changed:
        changed = False
        for name, needs in dependencies.items():
            if name in blocked:
                continue
            for required_name in needs:
                upstream = by_name.get(required_name)
                if upstream is None:
                    continue
                if upstream.status is Status.FAIL or required_name in blocked:
                    blocked[name] = required_name
                    changed = True
                    break

    return [
        replace(
            result,
            status=Status.INSUFFICIENT_DATA,
            detail=(
                f"not judged: {blocked[result.name]} failed, so this measurement would "
                f"describe the wrong frames"
            ),
        )
        if result.name in blocked and result.status is not Status.INSUFFICIENT_DATA
        else result
        for result in results
    ]


def aggregate(results: Iterable[CheckResult]) -> Verdict:
    counted = [r for r in results if r.counts_toward_verdict]
    if any(
        r.status is Status.FAIL and r.attribution is Attribution.DEVICE for r in counted
    ):
        return Verdict.FAIL
    if any(
        r.status is Status.FAIL and r.attribution is Attribution.PERFORMANCE
        for r in counted
    ):
        return Verdict.RETAKE
    # Only a *required* check left unanswered means the recording itself was not good
    # enough to judge. A conditional one simply had nothing to work with, which the
    # report says out loud rather than hiding behind a verdict.
    if any(r.status is Status.INSUFFICIENT_DATA and r.required for r in counted):
        return Verdict.INSUFFICIENT_DATA
    if not any(r.status in (Status.PASS, Status.FAIL) for r in counted):
        return Verdict.INSUFFICIENT_DATA
    return Verdict.PASS


def evaluate(
    source: FrameSource, active: Sequence[Check]
) -> tuple[int, list[CheckResult]]:
    """One pass over ``source`` feeding every check; the frame count and the results."""
    frames = 0
    for frame in source:
        frames += 1
        for check in active:
            check.update(frame)

    results = []
    for check in active:
        outcome = check.result()
        results.append(
            CheckResult(
                name=check.name,
                gate=check.gate,
                severity=check.severity,
                attribution=outcome.attribution or check.attribution,
                judged=check.judged,
                summary=check.summary,
                status=outcome.status,
                detail=outcome.detail,
                measurements=dict(outcome.measurements),
                required=check.required,
            )
        )

    return frames, suppress_dependents(results, active)


def assemble(
    source: FrameSource,
    frames: int,
    results: Sequence[CheckResult],
    notes: Sequence[str] = (),
) -> Report:
    return Report(
        source=str(getattr(source, "path", source.__class__.__name__)),
        frames=frames,
        metadata=source.metadata,
        results=tuple(results),
        verdict=aggregate(results),
        notes=tuple(notes),
    )
