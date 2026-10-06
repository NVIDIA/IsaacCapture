# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import repo_root  # noqa: E402

sys.path.insert(0, str(repo_root() / "examples" / "soma_hand_publisher" / "python"))


def pytest_addoption(parser):
    parser.addoption("--soma-assets", type=Path, help="SOMA-X assets directory")
    parser.addoption(
        "--soma-hand-pusher", type=Path, help="Built soma_hand_pusher executable"
    )


@pytest.fixture(scope="module")
def soma_assets(request):
    path = request.config.getoption("--soma-assets")
    if path is None:
        pytest.skip("Pass --soma-assets to run pinned SOMA-X semantic tests")
    return path
