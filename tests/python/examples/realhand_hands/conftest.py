# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from repo_paths import repo_root

sys.path.insert(0, str(repo_root() / "examples/realhand_hands/python"))

from _samples import human_sample


@pytest.fixture(scope="session")
def skeleton_samples():
    return {
        side: [human_sample(side, curl) for curl in (0.0, 0.6, 1.1)]
        for side in ("left", "right")
    }
