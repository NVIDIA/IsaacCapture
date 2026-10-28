# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The full-body panel's own HTML; the shared pieces are ``cts_common.panel.render``."""

from __future__ import annotations

from html import escape
from typing import Sequence

from cts_common.panel.render import (
    MARK_COLOURS,
    VERDICT_COLOURS,
    VERDICT_MEANING,
    badge,
    facts,
    group_body,
    group_heading,
    package_failed,
    packaged,
    result_rows,
    verdict_banner,
)

__all__ = [
    "MARK_COLOURS",
    "VERDICT_COLOURS",
    "VERDICT_MEANING",
    "badge",
    "facts",
    "group_body",
    "group_heading",
    "package_failed",
    "packaged",
    "playhead",
    "result_rows",
    "verdict_banner",
]


def playhead(
    t_s: float,
    duration_s: float,
    sequence: int,
    valid_count: int,
    total_joints: int,
    step: str | None,
    held: Sequence[str],
) -> str:
    """The live readout under the scrubber."""
    short = valid_count < total_joints
    colour = "#b0342c" if short else "#12502c"
    held_line = (
        f'<div style="font-size:11px;color:#b0342c">held: '
        f"{escape(', '.join(held))}</div>"
        if held
        else ""
    )
    return (
        f'<div style="font-family:monospace;font-size:12px">'
        f"{t_s:7.2f} s / {duration_s:.2f} s &nbsp; frame {sequence}</div>"
        f'<div style="font-size:12px">joints valid '
        f'<span style="font-weight:700;color:{colour}">{valid_count}/{total_joints}'
        f"</span></div>"
        f'<div style="font-size:11px;opacity:0.75">window: '
        f"{escape(step) if step else 'unlabelled'}</div>"
        f"{held_line}"
    )
