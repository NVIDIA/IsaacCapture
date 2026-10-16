# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""record.sh's argument handling, run against a stub interpreter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

RECORD = Path(__file__).resolve().parents[1] / "record.sh"


def record(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess, Path]:
    here = tmp_path / "capture"
    (here / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(RECORD, here)
    python = here / ".venv" / "bin" / "python"
    python.write_text('#!/bin/sh\necho "$@"\n')
    python.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        ["bash", str(here / "record.sh"), *args],
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, home


def test_an_option_where_the_device_goes_is_refused_before_anything_exists(
    tmp_path,
):
    result, home = record(tmp_path, "--plugin", "x")
    assert result.returncode == 2
    assert result.stderr.startswith("usage: ")
    assert list(home.iterdir()) == []


def test_no_argument_records_the_default_device(tmp_path):
    result, home = record(tmp_path)
    assert result.returncode == 0, result.stderr
    [take] = (home / "isaaccapture-cts").iterdir()
    assert take.name.startswith("pico4u_")
    provenance = json.loads((take / f"{take.name}-body.json").read_text())
    assert provenance["device"] == "pico4u"
