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


def _sources(*roots: Path) -> list[Path]:
    return [
        path
        for root in roots
        for path in sorted(root.rglob("*.py"))
        if path.resolve() != THIS
        and ".venv" not in path.parts
        and "generated" not in path.parts
    ]


# The package and its legacy import alias.
ISAAC_PACKAGE = re.compile(r"\bisaac(?:capture|teleop)\b")


def test_the_checker_does_not_import_isaaccapture():
    offenders = [
        p.name
        for p in _sources(CHECKER / "src", CHECKER / "tests")
        if ISAAC_PACKAGE.search(p.read_text())
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
        if re.search(r"\b(hand_cts|cts_common)\b", p.read_text())
    ]
    assert offenders == []
