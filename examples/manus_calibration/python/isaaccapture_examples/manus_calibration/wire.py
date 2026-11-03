# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON the page renders, built from the controller's view for one browser."""

from __future__ import annotations

from typing import Any

from .controller import Stage, View

_LOG_LINES = 8
_ACTIVE = frozenset(
    {
        Stage.STARTING,
        Stage.DISCOVERING,
        Stage.AWAITING_CONFIRMATION,
        Stage.SAMPLING,
        Stage.SAVING,
        Stage.STOPPING,
    }
)


def view_to_dict(
    view: View, client_id: str, out_dir: str, default_sides: str = "both"
) -> dict[str, Any]:
    """Snapshot for ``client_id``; only the owning browser is offered the prompt token."""
    owner = view.owner == client_id
    active = view.stage in _ACTIVE
    prompt = None
    if view.prompt is not None and active:
        prompt = {
            "side": view.prompt.side.value,
            "index": view.prompt.index,
            "count": view.prompt.count,
            "title": view.prompt.title,
            "instruction": view.prompt.instruction,
            "mode": view.prompt.mode.value,
            "duration_s": view.prompt.duration_s,
            "continuous": view.prompt.continuous,
            "estimated_s": view.prompt.estimated_s,
            # The token lets the owner's click name the prompt it saw; others never need it.
            "token": view.prompt.token if owner else None,
        }
    return {
        "stage": view.stage.value,
        "active": active,
        "owner": owner,
        "out_dir": out_dir,
        "default_sides": default_sides,
        "gloves": [{"side": g.side.value, "glove_id": g.glove_id} for g in view.gloves],
        "prompt": prompt,
        "confirm_pending": view.confirm_pending,
        "elapsed_s": view.elapsed_s,
        "notice": view.notice,
        "outcome": _outcome(view, out_dir),
    }


def _outcome(view: View, out_dir: str) -> dict[str, Any] | None:
    """Terminal result: where files landed, or why none were saved."""
    outcome = view.outcome
    if outcome is None:
        return None
    files = [
        {"side": f.side.value, "file": f.file, "bytes": f.bytes} for f in outcome.files
    ]
    directory = out_dir if files else None
    return {
        "kind": outcome.kind.value,
        "message": outcome.message,
        "directory": directory,
        "files": files,
        "log_tail": list(outcome.log_tail[-_LOG_LINES:])
        if outcome.kind.value in ("failed", "uncertain")
        else [],
    }
