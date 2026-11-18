# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration utility functions."""

import re


def sanitize_task_name(raw_task: str) -> str:
    """Convert a free-form task name into a filename-safe slug.

    Converts spaces to underscores, strips characters that are not
    alphanumeric, underscores, or hyphens, and collapses consecutive
    underscores/hyphens.

    Examples:
        "Lego Disassembly Toss v2" -> "Lego_Disassembly_Toss_v2"
        "pick & place!" -> "pick_place"
    """
    if not raw_task:
        return "unnamed"
    # Replace whitespace with underscores
    slug = re.sub(r"\s+", "_", raw_task.strip())
    # Remove characters that are not word chars (a-z, A-Z, 0-9, _) or hyphens
    slug = re.sub(r"[^\w\-]", "", slug)
    # Collapse consecutive underscores / hyphens
    slug = re.sub(r"[_\-]{2,}", "_", slug)
    slug = slug.strip("_-")
    return slug or "unnamed"
