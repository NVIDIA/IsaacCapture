# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The full-body script's step labels, read through ``acceptance_common.labels``."""

from __future__ import annotations

from acceptance_common.labels import SIDECAR_SUFFIX, Defect, Step
from acceptance_common.labels import StepTimeline as _StepTimeline

__all__ = ["SIDECAR_SUFFIX", "STILL_LABELS", "Defect", "Step", "StepTimeline"]

STILL_LABELS = frozenset(
    {"a_pose_still", "t_pose_hold_open", "neutral_stance", "t_pose_hold_close"}
)


class StepTimeline(_StepTimeline):
    still_labels = STILL_LABELS
