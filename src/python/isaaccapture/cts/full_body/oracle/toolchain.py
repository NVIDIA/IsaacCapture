# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locates the flatc output this generator encodes with: the checker's bindings and ``.bfbs``."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.abspath(os.path.join(_HERE, os.pardir, "checker"))
GENERATED = os.path.join(CHECKER, "generated")
BFBS_PATH = os.path.join(CHECKER, "build", "bfbs", "full_body.bfbs")


def ensure() -> None:
    """Put the generated bindings on the path, so ``core`` imports as a top-level package."""
    if not os.path.isfile(os.path.join(GENERATED, "core", "FullBodyPoseRecord.py")):
        raise SystemExit(f"missing {GENERATED}; run {CHECKER}/setup_env.sh first")
    if not os.path.isfile(BFBS_PATH):
        raise SystemExit(f"missing {BFBS_PATH}; run {CHECKER}/setup_env.sh first")
    if GENERATED not in sys.path:
        sys.path.insert(0, GENERATED)
