# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lines the hand checker must not cross, asserted mechanically."""

from __future__ import annotations

import re
from pathlib import Path

HAND = Path(__file__).resolve().parents[2]
CHECKER = HAND / "checker"
ORACLE = HAND / "oracle"
THIS = Path(__file__).resolve()


def _sources(*roots: Path, pattern: str = "*.py") -> list[Path]:
    return [
        path
        for root in roots
        for path in sorted(root.rglob(pattern))
        if path.resolve() != THIS
        and ".venv" not in path.parts
        and "generated" not in path.parts
    ]


def test_the_checker_does_not_import_isaacteleop():
    offenders = [
        p.name
        for p in _sources(CHECKER / "src", CHECKER / "tests")
        if "isaacteleop" in p.read_text()
    ]
    assert offenders == []


def test_the_checker_does_not_import_the_capture_scripts():
    capture = re.compile(
        r"^\s*(?:from|import)\s+(?:session|steps|cues|live|capture_panel|make_labels)\b",
        re.M,
    )
    offenders = [
        p.name for p in _sources(CHECKER / "src") if capture.search(p.read_text())
    ]
    assert offenders == []


def test_the_oracle_shares_no_code_with_the_checker():
    """The hand table, FK and quaternion arithmetic exist twice on purpose."""
    offenders = [
        p.name
        for p in _sources(ORACLE)
        if re.search(r"\b(hand_acceptance|acceptance_common)\b", p.read_text())
    ]
    assert offenders == []


def test_no_internal_gate_numbers_in_the_hand_tree():
    """Report groups are named, not numbered; the numbering is not part of the contract."""
    gate = re.compile(r"\bG[1-5]\b")
    offenders = [
        str(p.relative_to(HAND))
        for suffix in ("*.py", "*.md", "*.sh", "*.json")
        for p in _sources(HAND, pattern=suffix)
        if "fixtures" not in p.parts and gate.search(p.read_text())
    ]
    assert offenders == []
