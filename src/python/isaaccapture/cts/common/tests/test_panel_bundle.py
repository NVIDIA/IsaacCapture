# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared submission bundle, fed what a schema package would supply."""

from __future__ import annotations

import io
import json
import zipfile

from cts_common.checks.base import Attribution, Severity, Status
from cts_common.frames import SourceMetadata
from cts_common.panel import bundle
from cts_common.panel.sample import Sample
from cts_common.panel.status import grouped
from cts_common.report import CheckResult, Report, aggregate

METADATA = SourceMetadata(
    schema_name="core.ExampleRecord",
    schema_encoding="flatbuffer",
    schema_data=None,
    message_encoding="flatbuffer",
    topic="example",
    profile=None,
    channel_found=True,
)


def result(name: str, group: str, status: Status) -> CheckResult:
    return CheckResult(
        name=name,
        group=group,
        severity=Severity.HARD,
        attribution=Attribution.DEVICE,
        summary="",
        status=status,
        detail="",
        measurements={"peak": float("inf")},
    )


def report_of(*results: CheckResult) -> Report:
    return Report(
        source="take.mcap",
        frames=3,
        metadata=METADATA,
        results=results,
        verdict=aggregate(results),
    )


def sample(t_s: float, interval_ms: float | None, valid: int) -> Sample:
    return Sample(
        sequence=0,
        t_s=t_s,
        positions=(None,) * 3,
        valid=(True,) * valid + (False,) * (3 - valid),
        interval_ms=interval_ms,
        step=None,
    )


def packed_report(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        [name] = [n for n in archive.namelist() if n.endswith("/report.json")]
        return json.loads(archive.read(name))


def test_the_series_and_groups_are_packed_as_the_schema_supplies_them(tmp_path):
    recording = tmp_path / "take.mcap"
    recording.write_bytes(b"opaque")
    report = report_of(
        result("a.first", "one", Status.PASS), result("b.second", "two", Status.FAIL)
    )
    groups = grouped(report, (("two", "Second"), ("one", "First")))
    series = {"either": {"t_ms": [0.0], "rate_hz": [None], "valid": [3]}}

    name, data = bundle.build(
        report, recording, None, groups=groups, series=series, package=tmp_path
    )

    payload = packed_report(data)
    assert name == "take.fail.zip"
    assert payload["series"] == series
    assert payload["groups"] == [
        {"code": "two", "title": "Second", "checks": ["b.second"]},
        {"code": "one", "title": "First", "checks": ["a.first"]},
    ]
    assert payload["checks"][0]["measurements"] == {"peak": None}


def test_the_series_is_columnar_and_nulls_a_rate_it_cannot_state():
    samples = (sample(0.0, None, 3), sample(0.02, 20.0, 2), sample(0.02, 0.0, 1))
    assert bundle.series(samples) == {
        "t_ms": [0.0, 20.0, 20.0],
        "rate_hz": [None, 50.0, None],
        "valid": [3, 2, 1],
    }


def test_companions_that_are_not_there_are_named_not_raised(tmp_path):
    recording = tmp_path / "take.mcap"
    recording.write_bytes(b"opaque")
    (tmp_path / "take.log").write_text("log")
    present, missing = bundle.companions(recording)
    assert present == [tmp_path / "take.log"]
    assert missing == ["take.labels.json", "take.json"]


def test_the_tool_block_is_null_where_git_cannot_answer(tmp_path):
    assert bundle.tool(4, tmp_path) == {"commit": None, "dirty": None, "checks": 4}
