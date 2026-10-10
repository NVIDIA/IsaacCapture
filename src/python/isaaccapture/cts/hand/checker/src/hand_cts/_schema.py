# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access to the flatc-generated bindings; the only module that touches ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

GENERATED_DIR = Path(__file__).resolve().parents[2] / "generated"

if not (GENERATED_DIR / "core" / "HandPoseRecord.py").exists():
    raise ImportError(
        f"flatc-generated bindings missing from {GENERATED_DIR}. Run setup_env.sh."
    )

if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

from core.HandJointPose import HandJointPose  # noqa: E402
from core.HandPoseRecord import HandPoseRecord  # noqa: E402
from core.Point import Point  # noqa: E402
from core.Pose import Pose  # noqa: E402
from core.Quaternion import Quaternion  # noqa: E402

__all__ = [
    "GENERATED_DIR",
    "HandJointPose",
    "HandPoseRecord",
    "Point",
    "Pose",
    "Quaternion",
]
