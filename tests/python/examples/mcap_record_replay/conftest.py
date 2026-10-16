# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

_tests_python = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "repo_paths.py").is_file()
)
sys.path.insert(0, str(_tests_python))

from repo_paths import repo_root  # noqa: E402

sys.path.insert(0, str(repo_root() / "examples" / "mcap_record_replay" / "python"))
