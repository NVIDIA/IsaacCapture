# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locates the flatc-generated ``core`` bindings and ``.bfbs`` built by ``../checker/setup_env.sh``.

Nothing from the checker's own package is imported.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.abspath(os.path.join(_HERE, os.pardir, "checker"))
GENERATED = os.path.join(CHECKER, "generated")
BFBS_PATH = os.path.join(CHECKER, "build", "bfbs", "hand.bfbs")


def ensure() -> None:
    if not os.path.isfile(os.path.join(GENERATED, "core", "HandPoseRecord.py")):
        raise SystemExit(f"missing {GENERATED}; run {CHECKER}/setup_env.sh first")
    if not os.path.isfile(BFBS_PATH):
        raise SystemExit(f"missing {BFBS_PATH}; run {CHECKER}/setup_env.sh first")
    if GENERATED not in sys.path:
        sys.path.insert(0, GENERATED)
