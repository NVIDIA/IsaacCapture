# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_paths import repo_root  # noqa: E402

sys.path.insert(0, str(repo_root() / "examples" / "vehicle_teleop" / "python"))
