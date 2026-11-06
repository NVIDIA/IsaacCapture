# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
import subprocess
from pathlib import Path
from repo_paths import repo_root
import sys

# Robot example packages must not mask missing imports in the glove-only suite.
sys.path.insert(0, str(repo_root() / "examples/realhand_ffg_glove/python"))


def pytest_addoption(parser):
    parser.addoption("--ffg-skeleton-executable", default=None)
    parser.addoption("--ffg-human-calibration", default=None)


@pytest.fixture(scope="session")
def calibrated_skeleton_samples(request, tmp_path_factory):
    calibration = request.config.getoption("--ffg-human-calibration")
    executable = request.config.getoption("--ffg-skeleton-executable")
    if calibration is None or executable is None:
        pytest.skip(
            "Pass --ffg-human-calibration and --ffg-skeleton-executable for local replay"
        )
    assert Path(calibration).is_file()
    assert Path(executable).is_file()
    output = tmp_path_factory.mktemp("calibrated_ffg_skeleton") / "samples.yml"
    subprocess.run(
        [executable, "--write-calibrated-skeleton-fixtures", calibration, str(output)],
        check=True,
    )
    return yaml.safe_load(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def skeleton_samples(request, tmp_path_factory):
    executable = request.config.getoption("--ffg-skeleton-executable")
    if executable is None:
        pytest.skip(
            "Build the FFG C++ test executable and pass --ffg-skeleton-executable"
        )
    assert Path(executable).is_file()
    output = tmp_path_factory.mktemp("ffg_skeleton") / "samples.yml"
    subprocess.run([executable, "--write-skeleton-fixtures", str(output)], check=True)
    return yaml.safe_load(output.read_text(encoding="utf-8"))


@pytest.fixture
def human_calibration(tmp_path):
    """Device measurements have no target robot model or joint names."""
    data = {
        "format": "realhand-ffg-glove-human-calibration-v1",
        "hands": {
            side: {
                "open": [0.0] * 21,
                "fist": [1.0] * 21,
                "thumb_curl": [1.0] * 21,
                "thumb_opposition": [1.0] * 21,
                "spread": [0.3] * 21,
            }
            for side in ("left", "right")
        },
    }
    path = tmp_path / "ffg_human.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path
