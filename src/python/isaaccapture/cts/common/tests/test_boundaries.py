# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What keeps this package shareable, asserted mechanically."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "cts_common"
RENDERER = PACKAGE / "panel" / "skeleton.py"

IMPORTS_VISER = re.compile(r"^\s*(?:import viser|from viser)", re.MULTILINE)
SCHEMA_SPECIFIC = re.compile(r"\b(?:full_body_cts|hand_cts|FullBodyPose|HandPose)\w*")


def _modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


# The package and its legacy import alias.
ISAAC_PACKAGE = re.compile(r"\bisaac(?:capture|teleop)\b")


def test_no_isaaccapture():
    offenders = [p.name for p in _modules() if ISAAC_PACKAGE.search(p.read_text())]
    assert offenders == [], f"must not depend on isaaccapture: {offenders}"


def test_names_no_schema_package_or_record():
    offenders = [
        p.relative_to(PACKAGE).as_posix()
        for p in _modules()
        if SCHEMA_SPECIFIC.search(p.read_text())
    ]
    assert offenders == [], f"schema-specific code belongs in its package: {offenders}"


def test_only_the_skeleton_imports_viser():
    offenders = [
        p.relative_to(PACKAGE).as_posix()
        for p in _modules()
        if p != RENDERER and IMPORTS_VISER.search(p.read_text())
    ]
    assert offenders == [], f"viser belongs in panel/skeleton.py alone: {offenders}"


def test_everything_else_imports_with_viser_blocked():
    probe = (
        "import sys; sys.modules['viser'] = None;"
        "import cts_common.report, cts_common.labels,"
        " cts_common.mcap_reader, cts_common.panel.sample,"
        " cts_common.panel.track, cts_common.panel.status,"
        " cts_common.panel.render;"
        "import cts_common.checks.continuity, cts_common.checks.coverage,"
        " cts_common.checks.quaternion, cts_common.checks.rate,"
        " cts_common.checks.schema, cts_common.checks.timestamps,"
        " cts_common.checks.values"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        cwd=PACKAGE.parent,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
