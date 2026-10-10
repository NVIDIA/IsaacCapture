# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle layer: every envelope fixture, judged against ``fixtures_index.json``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DEFAULT_FIXTURES, fixtures_root
from known_deviations import BY_FIXTURE, EXPECTED_COLLATERAL

from full_body_cts import McapFrameSource, run
from full_body_cts.checks import CHECKS
from full_body_cts.report import Verdict

IMPLEMENTED = {cls.name for cls in CHECKS}


def _envelope_fixtures() -> list[dict]:
    root = fixtures_root()
    if root is None:
        return []
    index = json.loads((root / "fixtures_index.json").read_text())
    return [f for f in index["fixtures"] if f["batch"] == "envelope"]


ENVELOPE = _envelope_fixtures()

# Stated at collection time: an empty parametrisation never reaches ``fixture_dir``.
NEEDS_FIXTURES = f"no fixture set; run {DEFAULT_FIXTURES}/generate.sh"


def _identify(entry: dict) -> str:
    return Path(entry["filename"]).stem


@pytest.mark.skipif(not ENVELOPE, reason=NEEDS_FIXTURES)
@pytest.mark.parametrize("entry", ENVELOPE, ids=_identify)
def test_envelope_fixture_matches_the_index(entry: dict, fixture_dir: Path):
    report = run(McapFrameSource(fixture_dir / entry["filename"]))
    name = Path(entry["filename"]).name
    expected_check = entry.get("expected_failing_check")
    failures = [r.name for r in report.failures]
    advisories = [r.name for r in report.advisories]

    deviation = BY_FIXTURE.get(name)
    if deviation is not None:
        assert entry["expected_verdict"] == deviation.index_verdict
        assert deviation.advisory_check in IMPLEMENTED, deviation.advisory_check
        assert str(report.verdict) == deviation.checker_verdict, report.to_text()
        assert deviation.advisory_check in advisories, report.to_text()
        return

    if expected_check is None:
        assert str(report.verdict) == entry["expected_verdict"], report.to_text()
        assert failures == [], report.to_text()
        return

    assert expected_check in IMPLEMENTED, expected_check
    collateral = EXPECTED_COLLATERAL.get(name, set())
    assert collateral <= IMPLEMENTED, collateral - IMPLEMENTED
    assert set(failures) == {expected_check} | collateral, report.to_text()
    assert report.verdict is Verdict.FAIL


@pytest.mark.skipif(not ENVELOPE, reason=NEEDS_FIXTURES)
def test_every_golden_and_benign_fixture_passes(fixture_dir: Path):
    """The false-positive guard. A checker that fails everything must not score well."""
    clean = [e for e in ENVELOPE if e["category"] in ("golden", "benign")]
    assert len(clean) == 5

    for entry in clean:
        report = run(McapFrameSource(fixture_dir / entry["filename"]))
        assert report.verdict is Verdict.PASS, (
            f"{entry['filename']}\n{report.to_text()}"
        )


@pytest.mark.skipif(not ENVELOPE, reason=NEEDS_FIXTURES)
def test_schema_bytes_match_the_repo_golden(fixture_dir: Path):
    golden = (
        Path(__file__).resolve().parents[7]
        / "src"
        / "core"
        / "schema"
        / "golden"
        / "full_body.bfbs"
    ).read_bytes()

    for entry in ENVELOPE:
        source = McapFrameSource(fixture_dir / entry["filename"])
        assert source.metadata.schema_data == golden, entry["filename"]
        assert source.metadata.schema_encoding == "flatbuffer"
        assert source.metadata.message_encoding == "flatbuffer"


@pytest.mark.skipif(not ENVELOPE, reason=NEEDS_FIXTURES)
def test_reader_preserves_file_order(fixture_dir: Path):
    """``make_reader()`` would re-sort by log time and silently repair this fixture."""
    entry = next(
        e
        for e in ENVELOPE
        if e["filename"].endswith("defect_non_monotonic_timestamps.mcap")
    )
    frames = list(McapFrameSource(fixture_dir / entry["filename"]))
    regressions = sum(
        1
        for previous, current in zip(frames, frames[1:])
        if current.sample_time_ns < previous.sample_time_ns
    )
    assert regressions == 12
