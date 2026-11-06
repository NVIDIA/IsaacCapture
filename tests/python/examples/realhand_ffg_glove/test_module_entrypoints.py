# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
from repo_paths import repo_root


@pytest.mark.parametrize(
    "module",
    (
        "realhand_ffg_glove_calibration",
        "realhand_ffg_diagnostics",
        "realhand_ffg_hand_demo",
        "realhand_ffg_hand_demo_bridge",
    ),
)
def test_help_outside_checkout(module, tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(repo_root() / "examples/realhand_ffg_glove/python"),
            environment.get("PYTHONPATH", ""),
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"isaaccapture_examples.realhand_ffg_glove.{module}",
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # Assert below so failures include the child's stderr.
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
