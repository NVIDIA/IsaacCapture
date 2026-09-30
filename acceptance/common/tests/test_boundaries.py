# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What keeps this package shareable, asserted mechanically."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "acceptance_common"
RENDERER = PACKAGE / "panel" / "skeleton.py"

IMPORTS_VISER = re.compile(r"^\s*(?:import viser|from viser)", re.MULTILINE)
SCHEMA_SPECIFIC = re.compile(
    r"\b(?:full_body_acceptance|hand_acceptance|FullBodyPose|HandPose)\w*"
)


def _modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_no_isaacteleop():
    offenders = [p.name for p in _modules() if "isaacteleop" in p.read_text()]
    assert offenders == [], f"must not depend on isaacteleop: {offenders}"


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
        "import acceptance_common.report, acceptance_common.labels,"
        " acceptance_common.mcap_reader, acceptance_common.panel.sample,"
        " acceptance_common.panel.track, acceptance_common.panel.status,"
        " acceptance_common.panel.render;"
        "import acceptance_common.checks.continuity, acceptance_common.checks.coverage,"
        " acceptance_common.checks.quaternion, acceptance_common.checks.rate,"
        " acceptance_common.checks.schema, acceptance_common.checks.timestamps,"
        " acceptance_common.checks.values"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PACKAGE.parent,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
