# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from isaaccapture_examples.manus_calibration import protocol as proto

SHA = "a" * 64
ATTEMPT = "cal-abc"


def line(message: dict) -> str:
    return json.dumps({"v": 1, **message})


VALID = {
    "ready": {"type": "ready", "worker_version": "1", "sdk_version": None},
    "status": {
        "type": "status",
        "attempt": ATTEMPT,
        "phase": "discovering",
        "gloves": [{"side": "left", "glove_id": 3}],
    },
    "prompt": {
        "type": "prompt",
        "attempt": ATTEMPT,
        "token": "t0",
        "side": "left",
        "glove_id": 3,
        "index": 0,
        "count": 2,
        "title": "Flat",
        "instruction": "Lay flat",
        "mode": "hold",
        "duration_s": 0,
        "continuous": False,
        "estimated_s": 0,
    },
    "step_accepted": {"type": "step_accepted", "attempt": ATTEMPT, "token": "t0"},
    "step_progress": {
        "type": "step_progress",
        "attempt": ATTEMPT,
        "token": "t0",
        "elapsed_s": 0.5,
    },
    "rejected": {
        "type": "rejected",
        "attempt": None,
        "command": "confirm",
        "reason": "stale_token",
        "message": "no",
        "token": "t9",
    },
    "exported": {
        "type": "exported",
        "attempt": ATTEMPT,
        "side": "right",
        "glove_id": 4,
        "file": "cal-abc_right.mcal",
        "bytes": 10,
        "sha256": SHA,
    },
    "result": {
        "type": "result",
        "attempt": ATTEMPT,
        "outcome": "completed",
        "files": [{"side": "left", "file": "a.mcal", "bytes": 1, "sha256": SHA}],
        "released": True,
    },
}


@pytest.mark.parametrize("name", sorted(VALID))
def test_every_documented_event_parses(name):
    event = proto.parse_event(line(VALID[name]))
    assert type(event).__name__.lower().replace("_", "") == name.replace("_", "")


def test_failed_result_carries_an_error_and_cancelled_carries_neither():
    failed = proto.parse_event(
        line(
            {
                "type": "result",
                "attempt": None,
                "outcome": "failed",
                "error": {
                    "code": "no_gloves",
                    "message": "none",
                    "side": "left",
                    "step": 1,
                },
                "released": True,
            }
        )
    )
    assert failed.error == proto.WorkerError("no_gloves", "none", proto.Side.LEFT, 1)
    cancelled = proto.parse_event(
        line(
            {
                "type": "result",
                "attempt": ATTEMPT,
                "outcome": "cancelled",
                "released": True,
            }
        )
    )
    assert cancelled.files == () and cancelled.error is None


def mutate(name: str, **changes) -> str:
    message = dict(VALID[name])
    for key, value in changes.items():
        if value is KeyError:
            message.pop(key)
        else:
            message[key] = value
    return line(message)


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("prompt", {"extra": 1}),
        ("prompt", {"title": KeyError}),
        ("prompt", {"index": 2}),
        ("prompt", {"count": 0}),
        ("prompt", {"index": True}),
        ("prompt", {"mode": "hold", "duration_s": 1.0}),
        ("prompt", {"mode": "motion", "duration_s": 0}),
        ("prompt", {"mode": "dance"}),
        ("prompt", {"token": ""}),
        ("prompt", {"side": "middle"}),
        ("prompt", {"attempt": "has space"}),
        ("step_progress", {"elapsed_s": -1}),
        ("step_progress", {"elapsed_s": True}),
        ("step_progress", {"elapsed_s": "1"}),
        ("status", {"phase": "napping"}),
        ("status", {"gloves": "left"}),
        (
            "status",
            {
                "gloves": [
                    {"side": "left", "glove_id": 1},
                    {"side": "left", "glove_id": 2},
                ]
            },
        ),
        ("exported", {"file": "../escape.mcal"}),
        ("exported", {"file": "/abs.mcal"}),
        ("exported", {"file": "a/b.mcal"}),
        ("exported", {"sha256": "ABC"}),
        ("exported", {"bytes": -1}),
        ("exported", {"attempt": None}),
        ("rejected", {"reason": "because"}),
        ("result", {"outcome": "completed", "files": KeyError}),
        ("result", {"outcome": "cancelled"}),
        ("result", {"outcome": "failed"}),
        ("result", {"outcome": "failed", "error": {"code": "weird", "message": "x"}}),
        ("ready", {"worker_version": 1}),
        ("ready", {"v": 2}),
    ],
)
def test_malformed_events_are_protocol_errors(name, changes):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate(name, **changes))


FILE = {"side": "left", "file": "a.mcal", "bytes": 1, "sha256": SHA}
ERROR = {"code": "internal", "message": "x"}


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("status", {"gloves": ["left"]}),
        ("status", {"gloves": [{"side": "left", "glove_id": 1, "extra": 0}]}),
        ("result", {"outcome": "completed", "files": "left"}),
        ("result", {"outcome": "completed", "files": ["left"]}),
        ("result", {"outcome": "completed", "files": [{**FILE, "extra": 1}]}),
        ("result", {"outcome": "completed", "files": [{**FILE, "side": "middle"}]}),
        ("result", {"outcome": "completed", "files": [{**FILE, "sha256": "short"}]}),
        ("result", {"outcome": "completed", "error": ERROR}),
        ("result", {"files": KeyError, "outcome": "cancelled", "error": ERROR}),
        ("result", {"outcome": "cancelled", "files": []}),
        ("result", {"outcome": "failed", "error": ERROR, "files": []}),
        ("result", {"files": KeyError, "outcome": "failed", "error": "boom"}),
        (
            "result",
            {
                "files": KeyError,
                "outcome": "failed",
                "error": {"code": "weird", "message": "x"},
            },
        ),
        ("result", {"files": KeyError, "outcome": "failed", "error": {"message": "x"}}),
        (
            "result",
            {"files": KeyError, "outcome": "failed", "error": {**ERROR, "extra": 1}},
        ),
        (
            "result",
            {"files": KeyError, "outcome": "failed", "error": {**ERROR, "step": -1}},
        ),
        (
            "result",
            {"files": KeyError, "outcome": "failed", "error": {**ERROR, "step": True}},
        ),
        (
            "result",
            {"files": KeyError, "outcome": "failed", "error": {**ERROR, "step": 1.5}},
        ),
        (
            "result",
            {
                "files": KeyError,
                "outcome": "failed",
                "error": {**ERROR, "side": "middle"},
            },
        ),
        ("rejected", {"token": ""}),
        ("rejected", {"token": 5}),
        ("rejected", {"command": ""}),
    ],
)
def test_more_malformed_events_are_protocol_errors(name, changes):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate(name, **changes))


def test_a_completed_result_may_list_no_files_but_never_omit_the_field():
    event = proto.parse_event(
        line(
            {
                "type": "result",
                "attempt": ATTEMPT,
                "outcome": "completed",
                "files": [],
                "released": True,
            }
        )
    )
    assert event.files == ()
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(
            line(
                {
                    "type": "result",
                    "attempt": ATTEMPT,
                    "outcome": "completed",
                    "released": True,
                }
            )
        )


def test_every_documented_error_code_parses():
    for code in sorted(proto._ERROR_CODES):
        event = proto.parse_event(
            line(
                {
                    "type": "result",
                    "attempt": ATTEMPT,
                    "outcome": "failed",
                    "error": {"code": code, "message": "m"},
                    "released": True,
                }
            )
        )
        assert event.error.code == code


@pytest.mark.parametrize("released", [True, False])
@pytest.mark.parametrize(
    "extra",
    [
        {"outcome": "completed", "files": []},
        {"outcome": "cancelled"},
        {"outcome": "failed", "error": {"code": "internal", "message": "m"}},
    ],
)
def test_released_parses_for_every_outcome(extra, released):
    event = proto.parse_event(
        line(
            {
                "type": "result",
                "attempt": ATTEMPT,
                "released": released,
                **extra,
            }
        )
    )
    assert event.released is released


def test_a_result_without_released_is_a_protocol_error():
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate("result", released=KeyError))


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None, [], {}, 1.0])
def test_a_result_whose_released_is_not_a_boolean_is_a_protocol_error(value):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate("result", released=value))


def test_result_released_defaults_to_true_on_the_dataclass():
    result = proto.Result(attempt=None, outcome=proto.Outcome.CANCELLED)
    assert result.released is True


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "[]",
        "null",
        '{"v":1}',
        '{"v":1,"type":"nope"}',
        '{"v":"1","type":"ready"}',
        '{"v":1,"type":"step_progress","attempt":"a","token":"t","elapsed_s":NaN}',
        '{"v":1,"type":"step_progress","attempt":"a","token":"t","elapsed_s":Infinity}',
        b"\xff\xfe",
        b"x" * (proto.MAX_LINE_BYTES + 1),
    ],
)
def test_garbage_lines_are_protocol_errors(raw):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(raw)


def test_commands_round_trip_and_order_sides_left_first():
    start = json.loads(
        proto.encode_start("cal-1", (proto.Side.RIGHT, proto.Side.LEFT), "/stage")
    )
    assert start == {
        "v": 1,
        "type": "start",
        "attempt": "cal-1",
        "sides": ["left", "right"],
        "out_dir": "/stage",
    }
    assert json.loads(proto.encode_confirm("cal-1", "tok")) == {
        "v": 1,
        "type": "confirm",
        "attempt": "cal-1",
        "token": "tok",
    }
    assert json.loads(proto.encode_cancel("cal-1")) == {
        "v": 1,
        "type": "cancel",
        "attempt": "cal-1",
    }
    assert proto.encode_cancel("cal-1").endswith(b"\n")


@pytest.mark.parametrize("sides", [(), (proto.Side.LEFT, proto.Side.LEFT)])
def test_start_rejects_empty_or_duplicate_sides(sides):
    with pytest.raises(proto.ProtocolError):
        proto.encode_start("cal-1", sides, "/stage")


def test_start_rejects_a_bad_attempt_id():
    with pytest.raises(proto.ProtocolError):
        proto.encode_start("no spaces", (proto.Side.LEFT,), "/stage")


# --- prompt: continuous steps -------------------------------------------------------------


def test_a_prompt_with_an_empty_instruction_is_accepted():
    prompt = proto.parse_event(mutate("prompt", instruction=""))
    assert prompt.instruction == ""


def test_a_plain_hold_prompt_is_not_continuous():
    prompt = proto.parse_event(line(VALID["prompt"]))
    assert prompt.continuous is False
    assert prompt.estimated_s == 0.0
    assert prompt.mode is proto.StepMode.HOLD


def test_a_continuous_hold_prompt_carries_its_estimate():
    prompt = proto.parse_event(mutate("prompt", continuous=True, estimated_s=3.5))
    assert prompt.continuous is True
    assert prompt.estimated_s == 3.5
    assert prompt.mode is proto.StepMode.HOLD
    assert prompt.duration_s == 0.0


def test_an_integer_estimate_is_accepted_and_read_as_a_float():
    prompt = proto.parse_event(mutate("prompt", continuous=True, estimated_s=3))
    assert prompt.estimated_s == 3.0
    assert isinstance(prompt.estimated_s, float)


def test_a_timed_motion_prompt_is_not_continuous_and_has_no_estimate():
    prompt = proto.parse_event(
        mutate("prompt", mode="motion", duration_s=0.4, continuous=False, estimated_s=0)
    )
    assert prompt.mode is proto.StepMode.MOTION
    assert prompt.duration_s == 0.4
    assert prompt.continuous is False
    assert prompt.estimated_s == 0.0


def test_the_prompt_dataclass_defaults_to_a_non_continuous_step():
    prompt = proto.Prompt(
        attempt=ATTEMPT,
        token="t0",
        side=proto.Side.LEFT,
        glove_id=3,
        index=0,
        count=1,
        title="t",
        instruction="i",
        mode=proto.StepMode.HOLD,
        duration_s=0.0,
    )
    assert prompt.continuous is False
    assert prompt.estimated_s == 0.0
    with pytest.raises(AttributeError):
        prompt.continuous = True  # frozen


@pytest.mark.parametrize(
    "changes",
    [
        # continuous needs a positive estimate
        {"continuous": True, "estimated_s": 0},
        {"continuous": True, "estimated_s": 0.0},
        # continuous is a hold, never a timed motion
        {"continuous": True, "estimated_s": 3, "mode": "motion", "duration_s": 0.4},
        # a non-continuous step has no estimate, whatever its mode
        {"continuous": False, "estimated_s": 3},
        {"continuous": False, "estimated_s": 0.001},
        {"continuous": False, "estimated_s": 2, "mode": "motion", "duration_s": 0.4},
    ],
    ids=[
        "continuous-zero-int",
        "continuous-zero-float",
        "continuous-motion",
        "plain-estimate",
        "plain-tiny-estimate",
        "motion-estimate",
    ],
)
def test_inconsistent_continuous_and_estimate_pairs_are_protocol_errors(changes):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate("prompt", **changes))


@pytest.mark.parametrize("missing", ["continuous", "estimated_s"])
def test_a_prompt_missing_a_continuous_field_is_a_protocol_error(missing):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate("prompt", **{missing: KeyError}))


def test_a_prompt_missing_both_continuous_fields_is_a_protocol_error():
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate("prompt", continuous=KeyError, estimated_s=KeyError))


@pytest.mark.parametrize("value", [1, 0, "true", "false", None, [], {}, 1.0])
def test_continuous_must_be_a_strict_boolean(value):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate("prompt", continuous=value))


@pytest.mark.parametrize("value", ["3", None, True, False, [], {}, [3]])
def test_estimated_s_must_be_a_number(value):
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(mutate("prompt", continuous=True, estimated_s=value))


@pytest.mark.parametrize(
    "value",
    [-1, -0.001, float("nan"), float("inf"), float("-inf")],
    ids=["minus-one", "minus-tiny", "nan", "inf", "minus-inf"],
)
def test_estimated_s_must_be_finite_and_not_negative(value):
    for continuous in (True, False):
        with pytest.raises(proto.ProtocolError):
            proto.parse_event(
                mutate("prompt", continuous=continuous, estimated_s=value)
            )


def test_an_estimate_beyond_the_float_range_is_a_protocol_error():
    text = mutate("prompt", continuous=True, estimated_s=1).replace(
        '"estimated_s": 1', '"estimated_s": 1e400'
    )
    assert "1e400" in text
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(text)


def test_an_extra_key_on_a_continuous_prompt_is_still_a_protocol_error():
    with pytest.raises(proto.ProtocolError):
        proto.parse_event(
            mutate("prompt", continuous=True, estimated_s=3, extra_field=1)
        )
