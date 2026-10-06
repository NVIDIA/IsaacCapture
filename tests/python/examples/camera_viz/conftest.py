# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests live under tests/python/examples/camera_viz/ but import from
# examples/camera_viz/ (the example's pipeline / sources modules).
# Prepend the parent dir so those imports resolve against the in-tree
# source rather than any installed copy.

import sys
from pathlib import Path

_tests_python = Path(__file__).resolve().parents[2]
if str(_tests_python) not in sys.path:
    sys.path.insert(0, str(_tests_python))

from repo_paths import repo_root  # noqa: E402

sys.path.insert(0, str(repo_root() / "examples" / "camera_viz"))
