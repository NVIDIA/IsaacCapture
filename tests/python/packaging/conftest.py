# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Puts tests/python on sys.path so the tests can import repo_paths.

import sys
from pathlib import Path

_tests_python = Path(__file__).resolve().parents[1]
if str(_tests_python) not in sys.path:
    sys.path.insert(0, str(_tests_python))
