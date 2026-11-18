# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every oracle fixture, judged against ``fixtures_index.json``."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import DEFAULT_FIXTURES, index_entries

from hand_cts import McapFrameSource, run
from hand_cts.checks import CHECKS

ENTRIES = index_entries()
NEEDS_FIXTURES = f"no fixture set; run {DEFAULT_FIXTURES}/generate.sh"


@pytest.mark.skipif(not ENTRIES, reason=NEEDS_FIXTURES)
@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: Path(e["filename"]).stem)
def test_fixture_matches_the_index(entry: dict, fixture_dir: Path):
    report = run(McapFrameSource(fixture_dir / entry["filename"]))
    failures = [r.name for r in report.failures]
    advisories = [r.name for r in report.advisories]

    assert str(report.verdict) == entry["expected_verdict"], report.to_text()
    if entry["expected_verdict"] == "pass":
        assert failures == [], report.to_text()
        assert [r.name for r in report.unanswered if r.required] == [], report.to_text()
    expected = entry.get("expected_failing_check")
    if expected is not None:
        assert expected in failures, report.to_text()
    advisory = entry.get("expected_advisory")
    if advisory is not None:
        assert advisory in advisories, report.to_text()
    else:
        assert advisories == [], report.to_text()


@pytest.mark.skipif(not ENTRIES, reason=NEEDS_FIXTURES)
def test_every_named_check_exists():
    implemented = {cls.name for cls in CHECKS}
    named = {
        e["expected_failing_check"] for e in ENTRIES if "expected_failing_check" in e
    }
    assert named <= implemented, named - implemented
