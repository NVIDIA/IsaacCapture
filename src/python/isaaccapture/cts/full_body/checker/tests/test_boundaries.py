# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundaries between the parts of CTS, asserted mechanically."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[7]
CTS = Path(__file__).resolve().parents[3]
CHECKER = CTS / "full_body" / "checker"
CHECKER_SOURCES = (CHECKER / "src", CHECKER / "tests")
ORACLE = CTS / "full_body" / "oracle"
CAPTURE_MODULES = (
    "session",
    "cues",
    "steps",
    "live",
    "render",
    "capture_panel",
    "make_labels",
)

# Matched as an import statement: the checker names `record.sh` and the capture layout in prose.
CAPTURE_IMPORT = re.compile(
    rf"^\s*(?:from|import)\s+(?:{'|'.join(CAPTURE_MODULES)})\b", re.MULTILINE
)


def _python_files() -> list[Path]:
    """Every hand-written module under ``checker/``, less this file."""
    this_file = Path(__file__).resolve()
    return [
        path
        for root in CHECKER_SOURCES
        for path in sorted(root.rglob("*.py"))
        if path.resolve() != this_file
    ]


# The package name and its legacy ``isaacteleop`` alias.
ISAAC_PACKAGE = re.compile(r"\bisaac(?:capture|teleop)\b")


def test_the_checker_does_not_import_isaaccapture():
    """The checker takes only the .fbs text from the repo; flatc does the rest."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in _python_files()
        if ISAAC_PACKAGE.search(path.read_text())
    ]
    assert offenders == [], f"must not depend on isaaccapture: {offenders}"


def test_the_checker_does_not_import_the_capture_scripts():
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in _python_files()
        if CAPTURE_IMPORT.search(path.read_text())
    ]
    assert offenders == [], f"the checker must not import capture: {offenders}"


def test_the_oracle_shares_no_code_with_the_checker():
    """A fault in code shared by the oracle and the checker cancels itself out."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in sorted(ORACLE.rglob("*.py"))
        if re.search(r"\b(full_body_cts|cts_common)\b", path.read_text())
    ]
    assert offenders == [], f"the oracle must not import the checker: {offenders}"
