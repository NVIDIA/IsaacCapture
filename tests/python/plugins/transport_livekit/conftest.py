# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

for ancestor in Path(__file__).resolve().parents:
    helper = ancestor / "repo_paths.py"
    if helper.is_file():
        sys.path.insert(0, str(ancestor))
        break
from repo_paths import repo_root  # noqa: E402 — shared helper is located above.

root = repo_root()
sys.path[:0] = [
    str(root / "src/core/media_transport"),
    str(root / "src/plugins/transport_livekit"),
]
