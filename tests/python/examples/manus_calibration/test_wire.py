# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JSON the page renders: who sees the prompt token and how outcomes are shaped."""

from __future__ import annotations

import json

import pytest
from isaaccapture_examples.manus_calibration import protocol as proto
from isaaccapture_examples.manus_calibration.controller import Stage, View
from isaaccapture_examples.manus_calibration.wire import view_to_dict
from isaaccapture_examples.manus_calibration.worker import (
    CalibrationOutcome,
    OutcomeKind,
)

OWNER = "owner-client-0001"
OTHER = "other-client-0002"
OUT = "/data/calibrations"

PROMPT = proto.Prompt(
    attempt="cal-1",
    token="left-0",
    side=proto.Side.LEFT,
    glove_id=11,
    index=0,
    count=2,
    title="Flat hand",
    instruction="Lay your hand flat.",
    mode=proto.StepMode.HOLD,
    duration_s=0.0,
)
GLOVES = (proto.Glove(proto.Side.LEFT, 11), proto.Glove(proto.Side.RIGHT, 22))
FILES = (
    proto.ExportedFile(proto.Side.LEFT, "cal-1_left.mcal", 640, "a" * 64),
    proto.ExportedFile(proto.Side.RIGHT, "cal-1_right.mcal", 640, "b" * 64),
)


def active_view(**changes) -> View:
    fields = {
        "stage": Stage.AWAITING_CONFIRMATION,
        "owner": OWNER,
        "sides": (proto.Side.LEFT,),
        "gloves": GLOVES,
        "prompt": PROMPT,
    }
    return View(**{**fields, **changes})


def finished_view(kind: OutcomeKind, **changes) -> View:
    outcome = CalibrationOutcome(
        kind,
        "cal-1",
        (proto.Side.LEFT, proto.Side.RIGHT),
        f"{kind.value} message",
        **changes,
    )
    return View(stage=Stage.FINISHED, outcome=outcome)


def test_the_continuous_pacing_policy_is_part_of_the_browser_state():
    view = active_view()
    assert view_to_dict(view, OWNER, OUT)["continuous_pacing"] == "operator"
    assert (
        view_to_dict(view, OWNER, OUT, "both", "window")["continuous_pacing"]
        == "window"
    )


def test_the_owner_alone_receives_the_prompt_token():
    view = active_view()
    owner = view_to_dict(view, OWNER, OUT)
    other = view_to_dict(view, OTHER, OUT)
    assert owner["owner"] is True
    assert owner["prompt"]["token"] == "left-0"
    assert other["owner"] is False
    assert other["prompt"]["token"] is None
    # Everything else about the prompt is public.
    assert {k: v for k, v in owner["prompt"].items() if k != "token"} == {
        k: v for k, v in other["prompt"].items() if k != "token"
    }
    assert "left-0" not in json.dumps(other)


def test_prompt_fields_describe_the_sdk_step():
    prompt = view_to_dict(active_view(), OWNER, OUT)["prompt"]
    assert prompt == {
        "side": "left",
        "index": 0,
        "count": 2,
        "title": "Flat hand",
        "instruction": "Lay your hand flat.",
        "mode": "hold",
        "duration_s": 0.0,
        "continuous": False,
        "estimated_s": 0.0,
        "token": "left-0",
    }


CONTINUOUS = proto.Prompt(
    attempt="cal-1",
    token="left-1",
    side=proto.Side.LEFT,
    glove_id=11,
    index=1,
    count=2,
    title="Slow circle",
    instruction="Circle your hand slowly.",
    mode=proto.StepMode.HOLD,
    duration_s=0.0,
    continuous=True,
    estimated_s=3.5,
)


def test_a_continuous_prompt_reports_its_estimate_to_every_browser():
    view = active_view(prompt=CONTINUOUS)
    for client in (OWNER, OTHER):
        prompt = view_to_dict(view, client, OUT)["prompt"]
        assert prompt["continuous"] is True
        assert prompt["estimated_s"] == 3.5
        assert prompt["mode"] == "hold"
        assert prompt["duration_s"] == 0.0
    assert view_to_dict(view, OWNER, OUT)["prompt"]["token"] == "left-1"
    other = view_to_dict(view, OTHER, OUT)
    assert other["prompt"]["token"] is None
    assert "left-1" not in json.dumps(other)


def test_an_idle_view_has_no_prompt_owner_or_outcome():
    data = view_to_dict(View(), OWNER, OUT)
    assert data["stage"] == "idle"
    assert data["active"] is False
    assert data["owner"] is False
    assert data["prompt"] is None
    assert data["outcome"] is None


@pytest.mark.parametrize(
    "stage",
    [
        Stage.STARTING,
        Stage.DISCOVERING,
        Stage.AWAITING_CONFIRMATION,
        Stage.SAMPLING,
        Stage.SAVING,
        Stage.STOPPING,
    ],
)
def test_every_working_stage_is_active(stage):
    assert view_to_dict(active_view(stage=stage), OWNER, OUT)["active"] is True


@pytest.mark.parametrize("stage", [Stage.IDLE, Stage.FINISHED])
def test_idle_and_finished_are_not_active_and_hide_a_stale_prompt(stage):
    data = view_to_dict(active_view(stage=stage, owner=None), OWNER, OUT)
    assert data["active"] is False
    assert data["prompt"] is None


def test_gloves_progress_and_notice_pass_through():
    view = active_view(
        stage=Stage.SAMPLING, elapsed_s=0.25, confirm_pending=True, notice="note"
    )
    data = view_to_dict(view, OWNER, OUT)
    assert data["gloves"] == [
        {"side": "left", "glove_id": 11},
        {"side": "right", "glove_id": 22},
    ]
    assert data["elapsed_s"] == 0.25
    assert data["confirm_pending"] is True
    assert data["notice"] == "note"
    assert data["out_dir"] == OUT


def test_default_sides_default_to_both_and_can_be_set():
    assert view_to_dict(View(), OWNER, OUT)["default_sides"] == "both"
    assert view_to_dict(View(), OWNER, OUT, "left")["default_sides"] == "left"


def test_a_completed_outcome_lists_the_saved_files_and_directory():
    view = finished_view(OutcomeKind.COMPLETED, files=FILES)
    outcome = view_to_dict(view, OWNER, OUT)["outcome"]
    assert outcome["kind"] == "completed"
    assert outcome["directory"] == OUT
    assert outcome["files"] == [
        {
            "side": "left",
            "file": "cal-1_left.mcal",
            "bytes": 640,
        },
        {
            "side": "right",
            "file": "cal-1_right.mcal",
            "bytes": 640,
        },
    ]
    assert outcome["log_tail"] == []


@pytest.mark.parametrize("kind", [OutcomeKind.CANCELLED, OutcomeKind.COMPLETED])
def test_the_log_tail_is_kept_back_for_clean_outcomes(kind):
    view = finished_view(kind, log_tail=("one", "two"))
    assert view_to_dict(view, OWNER, OUT)["outcome"]["log_tail"] == []


@pytest.mark.parametrize("kind", [OutcomeKind.FAILED, OutcomeKind.UNCERTAIN])
def test_failed_and_uncertain_outcomes_show_the_last_log_lines_and_no_files(kind):
    tail = tuple(f"line {i}" for i in range(20))
    outcome = view_to_dict(finished_view(kind, log_tail=tail), OWNER, OUT)["outcome"]
    assert outcome["kind"] == kind.value
    assert outcome["files"] == []
    assert outcome["directory"] is None
    assert outcome["log_tail"] == [f"line {i}" for i in range(12, 20)]


def test_a_short_log_tail_is_not_padded():
    view = finished_view(OutcomeKind.FAILED, log_tail=("only",))
    assert view_to_dict(view, OWNER, OUT)["outcome"]["log_tail"] == ["only"]


def test_the_snapshot_is_json_serialisable_for_every_shape():
    views = [
        View(),
        active_view(),
        active_view(stage=Stage.SAMPLING, elapsed_s=0.1),
        finished_view(OutcomeKind.COMPLETED, files=FILES),
        finished_view(OutcomeKind.UNCERTAIN, log_tail=("x",)),
    ]
    for view in views:
        json.dumps(view_to_dict(view, OWNER, OUT))
