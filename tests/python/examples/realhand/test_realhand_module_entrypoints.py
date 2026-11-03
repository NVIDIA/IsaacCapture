# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Examples must run as package modules, including their relative imports."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from repo_paths import repo_root


def test_retargeter_dependency_declarations_match():
    root = repo_root()
    requirements = root / "src/core/python/requirements-retargeters.txt"
    runtime = {
        Requirement(line)
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    source = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    examples = tomllib.loads(
        Path(__file__).with_name("pyproject.toml").read_text(encoding="utf-8")
    )
    assert runtime == {
        Requirement(value)
        for value in source["project"]["optional-dependencies"]["retargeters"]
    }
    assert runtime <= {
        Requirement(value)
        for value in examples["project"]["optional-dependencies"]["dev"]
    }
    assert any(requirement.name == "six" for requirement in runtime)


@pytest.mark.parametrize(
    "module",
    (
        "p7_realhand_bimanual_example",
        "realhand_ffg_glove_calibration",
        "realhand_ffg_diagnostics",
        "realhand_ffg_hand_demo",
        "realhand_ffg_hand_demo_bridge",
        "fetch_realhand_assets",
    ),
)
def test_module_help_outside_checkout(module, tmp_path):
    environment = os.environ.copy()
    namespace = str(repo_root() / "examples/teleop/python")
    environment["PYTHONPATH"] = os.pathsep.join(
        (namespace, environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-m", f"isaaccapture_examples.teleop.{module}", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # Assert below so failures include the child's stderr.
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
