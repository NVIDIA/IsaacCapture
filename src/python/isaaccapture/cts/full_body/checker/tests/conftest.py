# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

DEFAULT_FIXTURES = Path(__file__).resolve().parents[2] / "oracle"


def fixtures_root() -> Path | None:
    """The fixtures are derived and not in git, so their absence is normal."""
    override = os.environ.get("FULLBODY_FIXTURES")
    candidate = Path(override) if override else DEFAULT_FIXTURES
    if not (candidate / "fixtures_index.json").is_file():
        return None
    return candidate if (candidate / "fixtures").is_dir() else None


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    root = fixtures_root()
    if root is None:
        pytest.skip(f"no fixture set; run {DEFAULT_FIXTURES}/generate.sh")
    return root


@pytest.fixture(scope="session")
def index(fixture_dir: Path) -> dict:
    return json.loads((fixture_dir / "fixtures_index.json").read_text())
