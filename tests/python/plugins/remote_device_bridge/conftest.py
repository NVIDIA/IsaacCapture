# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The tests import the remote_device_bridge tools/ scripts from their in-tree source.

import sys
from pathlib import Path

_tests_python = Path(__file__).resolve().parents[2]
if str(_tests_python) not in sys.path:
    sys.path.insert(0, str(_tests_python))

from repo_paths import repo_root  # noqa: E402

sys.path.insert(
    0, str(repo_root() / "src" / "plugins" / "remote_device_bridge" / "tools")
)
