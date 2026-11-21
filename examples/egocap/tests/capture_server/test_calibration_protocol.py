# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from capture_server import protocol as p

LEFT, RIGHT = p.HandSide.LEFT, p.HandSide.RIGHT


def _step(**overrides) -> p.CalibrationStep:
    fields = dict(
        token="tok-1",
        side=LEFT,
        index=0,
        count=8,
        title="Make a fist",
        instruction="Lift up your hand and make a fist.",
        mode=p.CalibrationStepMode.HOLD,
        duration_seconds=0.0,
        continuous=False,
        estimated_seconds=0.0,
    )
    fields.update(overrides)
    return p.CalibrationStep(**fields)


def _stopped(reason=p.CalibrationStopReason.CANCELLED, result=None, error_message=None) -> p.CalibrationStopped:
    return p.CalibrationStopped("cal-1", reason, result, error_message)


def _begin_json(plugins) -> dict:
    return {
        "type": 1,
        "version": 1,
        "session_id": "session-1",
        "tracking": {"sources": [1, 3], "rate_hz": 60, "headset": {"device_index": 0}},
        "plugins": plugins,
        "cameras": [],
    }


# ---------------------------------------------------------------------------
# Numeric values and enums
# ---------------------------------------------------------------------------


def test_existing_capture_message_values_are_unchanged():
    assert [int(t) for t in (p.MessageType.BEGIN, p.MessageType.STOP, p.MessageType.SESSION_INIT)] == [1, 2, 9]
    assert int(p.MessageType.REQUEST_REJECTED) == 10
    assert int(p.MessageType.CAPTURE_FRAME) == 7
    assert int(p.MessageType.CAPTURE_IMAGE) == 8


def test_calibration_message_types_take_the_next_numeric_values():
    names = (
        "START_CALIBRATION",
        "CONFIRM_CALIBRATION_STEP",
        "STOP_CALIBRATION",
        "CALIBRATION_STARTED",
        "CALIBRATION_STATUS",
        "CALIBRATION_STEP_ACCEPTED",
        "CALIBRATION_STOPPED",
    )
    assert [int(p.MessageType[name]) for name in names] == [13, 14, 15, 16, 17, 18, 19]


def test_calibration_enums_have_their_documented_values():
    assert (int(p.HandSide.LEFT), int(p.HandSide.RIGHT)) == (1, 2)
    assert [int(x) for x in p.CalibrationPhase] == [1, 2, 3, 4, 5]
    assert (int(p.CalibrationStepMode.HOLD), int(p.CalibrationStepMode.MOTION)) == (0, 1)
    assert [int(x) for x in p.CalibrationStopReason] == [0, 1, 2, 3]
    assert p.CalibrationStopReason.UNCERTAIN == 3


# ---------------------------------------------------------------------------
# Client-to-server requests
# ---------------------------------------------------------------------------


def test_start_calibration_round_trips():
    request = p.StartCalibrationRequest(session_id="cal-1", sides=(LEFT, RIGHT))
    assert request.to_json() == {"type": 13, "version": 1, "session_id": "cal-1", "sides": [1, 2]}
    assert p.parse_start_calibration(request.to_json()) == request


@pytest.mark.parametrize("sides", [(), (LEFT, LEFT)])
def test_start_calibration_request_rejects_empty_or_duplicate_sides(sides):
    with pytest.raises(ValueError):
        p.StartCalibrationRequest(session_id="cal-1", sides=sides)


def test_start_calibration_request_rejects_non_hand_side_members():
    with pytest.raises(ValueError):
        p.StartCalibrationRequest(session_id="cal-1", sides=(1,))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        {"sides": []},
        {"sides": [1, 1]},
        {"sides": [3]},
        {"sides": [True]},
        {"sides": "left"},
        {"sides": None},
        {"extra": 1},
    ],
)
def test_parse_start_calibration_rejects_bad_fields(mutation):
    golden = p.StartCalibrationRequest("cal-1", (LEFT,)).to_json()
    with pytest.raises(p.ProtocolError):
        p.parse_start_calibration({**golden, **mutation})


def test_parse_start_calibration_rejects_a_missing_sides_key_and_the_wrong_envelope_type():
    golden = p.StartCalibrationRequest("cal-1", (LEFT,)).to_json()
    without_sides = {k: v for k, v in golden.items() if k != "sides"}
    with pytest.raises(p.ProtocolError):
        p.parse_start_calibration(without_sides)
    with pytest.raises(p.ProtocolError):
        p.parse_start_calibration({**golden, "type": int(p.MessageType.STOP_CALIBRATION)})


def test_confirm_calibration_step_round_trips_and_rejects_bad_tokens():
    request = p.ConfirmCalibrationStepRequest(session_id="cal-1", step_token="left-0")
    assert request.to_json() == {"type": 14, "version": 1, "session_id": "cal-1", "step_token": "left-0"}
    assert p.parse_confirm_calibration_step(request.to_json()) == request
    for bad in ("", None, 5, "x" * 129):
        with pytest.raises(p.ProtocolError):
            p.parse_confirm_calibration_step({**request.to_json(), "step_token": bad})
    with pytest.raises(p.ProtocolError):
        p.parse_confirm_calibration_step({**request.to_json(), "extra": 1})


def test_stop_calibration_round_trips_and_rejects_extra_keys():
    request = p.StopCalibrationRequest(session_id="cal-1")
    assert request.to_json() == {"type": 15, "version": 1, "session_id": "cal-1"}
    assert p.parse_stop_calibration(request.to_json()) == request
    with pytest.raises(p.ProtocolError):
        p.parse_stop_calibration({**request.to_json(), "sides": [1]})


# ---------------------------------------------------------------------------
# Step model
# ---------------------------------------------------------------------------


def test_step_round_trips_for_hold_motion_and_continuous():
    hold = _step()
    motion = _step(mode=p.CalibrationStepMode.MOTION, duration_seconds=2.5)
    continuous = _step(continuous=True, estimated_seconds=3.0)
    for step in (hold, motion, continuous):
        assert p.CalibrationStep.from_json(step.to_json()) == step


def test_continuous_step_keeps_actual_duration_distinct_from_the_estimate():
    step = _step(continuous=True, estimated_seconds=10.0)
    data = step.to_json()
    assert data["duration_seconds"] == 0.0
    assert data["estimated_seconds"] == 10.0
    assert data["continuous"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": p.CalibrationStepMode.HOLD, "duration_seconds": 1.0},
        {"mode": p.CalibrationStepMode.MOTION, "duration_seconds": 0.0},
        {"continuous": True, "estimated_seconds": 0.0},
        {
            "continuous": True,
            "estimated_seconds": 2.0,
            "mode": p.CalibrationStepMode.MOTION,
            "duration_seconds": 1.0,
        },
        {"continuous": False, "estimated_seconds": 2.0},
        {"index": 8},
        {"index": -1},
        {"count": 0},
        {"token": ""},
    ],
)
def test_step_constructor_rejects_inconsistent_invariants(overrides):
    with pytest.raises(ValueError):
        _step(**overrides)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, "1", None])
@pytest.mark.parametrize("key", ["duration_seconds", "estimated_seconds"])
def test_step_from_json_rejects_non_finite_negative_or_mistyped_seconds(key, bad):
    data = _step().to_json()
    data[key] = bad
    with pytest.raises(p.ProtocolError):
        p.CalibrationStep.from_json(data)


def test_step_from_json_rejects_missing_extra_and_mistyped_fields():
    golden = _step().to_json()
    for key in golden:
        with pytest.raises(p.ProtocolError):
            p.CalibrationStep.from_json({k: v for k, v in golden.items() if k != key})
    with pytest.raises(p.ProtocolError):
        p.CalibrationStep.from_json({**golden, "extra": 1})
    for key, bad in (
        ("side", 3),
        ("mode", 2),
        ("index", 1.5),
        ("count", "8"),
        ("instruction", ""),
        ("title", None),
        ("continuous", 1),
    ):
        with pytest.raises(p.ProtocolError):
            p.CalibrationStep.from_json({**golden, key: bad})


def test_step_from_json_reports_a_broken_invariant_as_a_protocol_error():
    golden = _step().to_json()
    with pytest.raises(p.ProtocolError):
        p.CalibrationStep.from_json({**golden, "continuous": True, "estimated_seconds": 0.0})


def test_step_has_no_hardcoded_count_and_carries_the_sdk_text_unchanged():
    step = _step(count=3, title="SDK title", instruction="SDK instruction")
    data = step.to_json()
    assert (data["count"], data["title"], data["instruction"]) == (3, "SDK title", "SDK instruction")


# ---------------------------------------------------------------------------
# Server-to-client messages
# ---------------------------------------------------------------------------


def test_started_round_trips_and_rejects_bad_sides():
    event = p.CalibrationStarted("cal-1", (LEFT, RIGHT))
    assert event.to_json() == {"type": 16, "version": 1, "session_id": "cal-1", "sides": [1, 2]}
    assert p.CalibrationStarted.from_json(event.to_json()) == event
    for bad in ([], [1, 1], [9], None):
        with pytest.raises(p.ProtocolError):
            p.CalibrationStarted.from_json({**event.to_json(), "sides": bad})


def test_status_round_trips_with_and_without_a_step():
    gloves = (p.CalibrationGlove(LEFT, 11), p.CalibrationGlove(RIGHT, 22))
    with_step = p.CalibrationStatus("cal-1", p.CalibrationPhase.AWAITING_STEP, gloves, _step(), None)
    capturing = p.CalibrationStatus(
        "cal-1",
        p.CalibrationPhase.CAPTURING_STEP,
        gloves,
        _step(mode=p.CalibrationStepMode.MOTION, duration_seconds=2.0),
        0.5,
    )
    connecting = p.CalibrationStatus("cal-1", p.CalibrationPhase.CONNECTING, (), None, None)
    for status in (with_step, capturing, connecting):
        assert p.CalibrationStatus.from_json(status.to_json()) == status


def test_status_rejects_bad_fields():
    golden = p.CalibrationStatus("cal-1", p.CalibrationPhase.CONNECTING, (), None, None).to_json()
    for key, bad in (
        ("phase", 0),
        ("phase", 6),
        ("gloves", None),
        ("gloves", [{"side": 1}]),
        ("gloves", [{"side": 1, "glove_id": -1}]),
        ("step", {"token": "x"}),
        ("step_elapsed_seconds", -1.0),
        ("step_elapsed_seconds", math.nan),
    ):
        with pytest.raises(p.ProtocolError):
            p.CalibrationStatus.from_json({**golden, key: bad})
    with pytest.raises(p.ProtocolError):
        p.CalibrationStatus.from_json({**golden, "extra": 1})


def test_step_accepted_round_trips_and_requires_a_token():
    event = p.CalibrationStepAccepted("cal-1", "left-0")
    assert event.to_json() == {"type": 18, "version": 1, "session_id": "cal-1", "step_token": "left-0"}
    assert p.CalibrationStepAccepted.from_json(event.to_json()) == event
    with pytest.raises(p.ProtocolError):
        p.CalibrationStepAccepted.from_json({**event.to_json(), "step_token": ""})


def test_stopped_completed_requires_a_result_with_basenames():
    result = p.CalibrationResult((p.McalFile(LEFT, "set_left.mcal"), p.McalFile(RIGHT, "set_right.mcal")))
    event = _stopped(p.CalibrationStopReason.COMPLETED, result, None)
    assert p.CalibrationStopped.from_json(event.to_json()) == event
    assert event.to_json()["result"]["files"][0] == {"side": 1, "filename": "set_left.mcal"}


@pytest.mark.parametrize(
    "reason",
    [p.CalibrationStopReason.CANCELLED, p.CalibrationStopReason.FAILED, p.CalibrationStopReason.UNCERTAIN],
)
def test_stopped_other_reasons_forbid_a_result(reason):
    result = p.CalibrationResult((p.McalFile(LEFT, "set_left.mcal"),))
    with pytest.raises(ValueError):
        _stopped(reason, result, "oops")
    event = _stopped(reason, None, "something went wrong")
    assert p.CalibrationStopped.from_json(event.to_json()) == event


def test_stopped_completed_without_a_result_or_with_an_error_is_invalid():
    result = p.CalibrationResult((p.McalFile(LEFT, "set_left.mcal"),))
    with pytest.raises(ValueError):
        _stopped(p.CalibrationStopReason.COMPLETED, None, None)
    with pytest.raises(ValueError):
        _stopped(p.CalibrationStopReason.COMPLETED, result, "late warning")


def test_stopped_from_json_reports_broken_invariants_as_protocol_errors():
    golden = _stopped(p.CalibrationStopReason.FAILED, None, "failed").to_json()
    with pytest.raises(p.ProtocolError):
        p.CalibrationStopped.from_json({**golden, "result": {"files": [{"side": 1, "filename": "a.mcal"}]}})
    with pytest.raises(p.ProtocolError):
        p.CalibrationStopped.from_json({**golden, "error_message": ""})
    with pytest.raises(p.ProtocolError):
        p.CalibrationStopped.from_json({**golden, "reason": 4})


@pytest.mark.parametrize(
    "filename", ["../x.mcal", "/abs/x.mcal", "dir/x.mcal", ".hidden.mcal", "x.txt", "x", "", "a" * 130 + ".mcal"]
)
def test_mcal_file_rejects_anything_but_a_bare_mcal_basename(filename):
    with pytest.raises(p.ProtocolError):
        p.McalFile.from_json({"side": 1, "filename": filename})


def test_calibration_result_rejects_empty_and_two_files_for_one_side():
    with pytest.raises(p.ProtocolError):
        p.CalibrationResult.from_json({"files": []})
    one = {"side": 1, "filename": "a.mcal"}
    with pytest.raises(p.ProtocolError):
        p.CalibrationResult.from_json({"files": [one, {"side": 1, "filename": "b.mcal"}]})


def test_parse_server_message_dispatches_the_four_calibration_events():
    events = (
        p.CalibrationStarted("cal-1", (LEFT,)),
        p.CalibrationStatus("cal-1", p.CalibrationPhase.CONNECTING, (), None, None),
        p.CalibrationStepAccepted("cal-1", "tok"),
        _stopped(p.CalibrationStopReason.CANCELLED, None, None),
    )
    for event in events:
        assert p.parse_server_message(event.to_json()) == event


def test_parse_server_message_rejects_client_to_server_calibration_requests():
    for request in (
        p.StartCalibrationRequest("cal-1", (LEFT,)),
        p.ConfirmCalibrationStepRequest("cal-1", "tok"),
        p.StopCalibrationRequest("cal-1"),
    ):
        with pytest.raises(p.ProtocolError):
            p.parse_server_message(request.to_json())


# ---------------------------------------------------------------------------
# RequestRejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "request_type",
    [
        p.MessageType.BEGIN,
        p.MessageType.STOP,
        p.MessageType.START_CALIBRATION,
        p.MessageType.CONFIRM_CALIBRATION_STEP,
        p.MessageType.STOP_CALIBRATION,
    ],
)
def test_request_rejected_answers_every_client_request_type(request_type):
    event = p.RequestRejected("s", request_type, p.RejectionReason.BUSY, "busy")
    assert p.RequestRejected.from_json(event.to_json()) == event


@pytest.mark.parametrize(
    "request_type",
    [p.MessageType.SESSION_INIT, p.MessageType.CALIBRATION_STATUS, p.MessageType.CALIBRATION_STOPPED],
)
def test_request_rejected_refuses_server_message_types(request_type):
    with pytest.raises(ValueError, match="client request type"):
        p.RequestRejected("s", request_type, p.RejectionReason.BUSY, "busy")


def test_request_rejected_echoes_a_step_token_only_for_confirm():
    event = p.RequestRejected(
        "cal-1",
        p.MessageType.CONFIRM_CALIBRATION_STEP,
        p.RejectionReason.INVALID_CONFIG,
        "stale",
        step_token="left-0",
    )
    assert event.to_json()["step_token"] == "left-0"
    assert p.RequestRejected.from_json(event.to_json()) == event
    with pytest.raises(ValueError):
        p.RequestRejected("s", p.MessageType.BEGIN, p.RejectionReason.BUSY, "busy", step_token="x")
    golden = p.RequestRejected("s", p.MessageType.BEGIN, p.RejectionReason.BUSY, "busy").to_json()
    with pytest.raises(p.ProtocolError):
        p.RequestRejected.from_json({**golden, "step_token": "x"})


def test_begin_and_stop_rejections_keep_their_exact_earlier_wire_form():
    golden = p.RequestRejected("s", p.MessageType.BEGIN, p.RejectionReason.BUSY, "busy").to_json()
    assert golden == {
        "type": 10,
        "version": 1,
        "session_id": "s",
        "request_type": 1,
        "reason": 2,
        "error_message": "busy",
    }
