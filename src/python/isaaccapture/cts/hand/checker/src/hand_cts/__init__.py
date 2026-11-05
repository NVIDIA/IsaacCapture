# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CTS checks for a hand-tracking or glove integration."""

from __future__ import annotations

from .frames import Frame, FrameSource, JointPose, SourceMetadata
from .mcap_source import McapFrameSource
from .report import CheckResult, Report, Verdict, aggregate, run

__all__ = [
    "CheckResult",
    "Frame",
    "FrameSource",
    "JointPose",
    "McapFrameSource",
    "Report",
    "SourceMetadata",
    "Verdict",
    "aggregate",
    "run",
]
