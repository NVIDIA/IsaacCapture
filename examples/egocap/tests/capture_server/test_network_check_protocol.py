# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json

import pytest

from capture_server import protocol as p


def _camera(**overrides) -> p.NetworkCheckCamera:
    fields = dict(
        stream_id=16,
        name="Head left",
        passed=True,
        expected_fps=30.0,
        usable_frames=900,
        delivered_fps=29.9,
        missing_frames=0,
        max_capture_gap_ms=40.5,
        max_arrival_gap_ms=60.0,
        start_silence_ms=33.0,
        end_silence_ms=30.0,
        age_median_ms=80.0,
        age_max_ms=120.0,
        age_growth_ms=1.5,
        reasons=(),
        track_id="TR_abc",
    )
    fields.update(overrides)
    return p.NetworkCheckCamera(**fields)


def _status(**overrides) -> p.NetworkCheckStatus:
    fields = dict(
        state=p.NetworkCheckState.READY,
        session_id="sess-1",
        generation="abc123",
        endpoint="LiveKit wss://lk, room r, publisher o",
        legs_tested=("a", "b"),
        legs_not_tested=("c",),
        window_s=30.0,
        valid_for_s=300.0,
        phase=None,
        progress_s=None,
        completed_at_unix_ns=1_700_000_000_000_000_000,
        reason=None,
        cameras=(_camera(),),
    )
    fields.update(overrides)
    return p.NetworkCheckStatus(**fields)


def _wire(obj) -> dict:
    return json.loads(json.dumps(obj.to_json()))


def test_network_check_state_values_are_wire_stable():
    assert [int(s) for s in p.NetworkCheckState] == [0, 1, 2, 3, 4]
    assert [s.name for s in p.NetworkCheckState] == ["NOT_TESTED", "CHECKING", "READY", "UNSUITABLE", "STALE"]
    assert int(p.MessageType.RUN_NETWORK_CHECK) == 20
    assert int(p.MessageType.QUERY_STATUS) == 21
    assert int(p.MessageType.SERVER_STATUS) == 22


def test_camera_round_trips_through_json():
    camera = _camera(passed=False, reasons=("a", "b"))
    assert p.NetworkCheckCamera.from_json(_wire(camera)) == camera


def test_camera_none_measurements_are_preserved_as_none_not_zero():
    camera = _camera(
        passed=None,
        usable_frames=None,
        delivered_fps=None,
        missing_frames=None,
        max_capture_gap_ms=None,
        max_arrival_gap_ms=None,
        start_silence_ms=None,
        end_silence_ms=None,
        age_median_ms=None,
        age_max_ms=None,
        age_growth_ms=None,
    )
    wire = _wire(camera)
    assert wire["missing_frames"] is None and wire["age_median_ms"] is None and wire["passed"] is None
    parsed = p.NetworkCheckCamera.from_json(wire)
    assert parsed == camera
    assert parsed.missing_frames is None and parsed.age_median_ms is None


def test_camera_zero_measurement_stays_zero():
    parsed = p.NetworkCheckCamera.from_json(_wire(_camera(missing_frames=0, age_max_ms=0.0)))
    assert parsed.missing_frames == 0 and parsed.age_max_ms == 0.0


def test_camera_integer_valued_measurement_may_arrive_as_integer_and_is_a_float():
    wire = _wire(_camera())
    wire["delivered_fps"] = 30
    assert p.NetworkCheckCamera.from_json(wire).delivered_fps == 30.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("stream_id"),
        lambda d: d.update(stream_id=-1),
        lambda d: d.update(stream_id=True),
        lambda d: d.update(stream_id="16"),
        lambda d: d.update(name=""),
        lambda d: d.update(name=3),
        lambda d: d.update(passed="yes"),
        lambda d: d.update(passed=1),
        lambda d: d.update(expected_fps=0),
        lambda d: d.update(expected_fps=True),
        lambda d: d.update(usable_frames=-1),
        lambda d: d.update(usable_frames=1.5),
        lambda d: d.update(usable_frames=True),
        lambda d: d.update(missing_frames=2.0),
        lambda d: d.update(delivered_fps=-0.1),
        lambda d: d.update(delivered_fps=True),
        lambda d: d.update(delivered_fps="30"),
        lambda d: d.update(age_max_ms=float("nan")),
        lambda d: d.update(age_growth_ms=True),
        lambda d: d.update(age_median_ms="1"),
        lambda d: d.update(age_median_ms=float("inf")),
        lambda d: d.update(reasons="slow"),
        lambda d: d.update(reasons=[1]),
        lambda d: d.update(reasons=None),
        lambda d: d.update(track_id=""),
        lambda d: d.update(track_id=7),
    ],
)
def test_camera_parsing_is_strict(mutation):
    wire = _wire(_camera())
    mutation(wire)
    with pytest.raises(p.ProtocolError):
        p.NetworkCheckCamera.from_json(wire)


def test_signed_age_measurements_round_trip():
    # Growth is last-third minus first-third, negative when latency improves; ages are negative when clocks lead.
    camera = _camera(age_growth_ms=-12.5, age_median_ms=-3.0, age_max_ms=-1.0)
    assert p.NetworkCheckCamera.from_json(_wire(camera)) == camera


@pytest.mark.parametrize(
    "key",
    [
        "usable_frames",
        "delivered_fps",
        "missing_frames",
        "max_capture_gap_ms",
        "max_arrival_gap_ms",
        "start_silence_ms",
        "end_silence_ms",
    ],
)
def test_every_other_measurement_stays_nonnegative(key):
    wire = _wire(_camera())
    wire[key] = -1
    with pytest.raises(p.ProtocolError):
        p.NetworkCheckCamera.from_json(wire)


def test_camera_rejects_a_non_object():
    with pytest.raises(p.ProtocolError):
        p.NetworkCheckCamera.from_json([])


def test_status_round_trips_through_json():
    status = _status(
        state=p.NetworkCheckState.CHECKING,
        phase="measuring",
        progress_s=7,
        completed_at_unix_ns=None,
        cameras=(_camera(), _camera(stream_id=17, name="Head right", passed=False, reasons=("lost",))),
    )
    assert p.NetworkCheckStatus.from_json(_wire(status)) == status


def test_status_with_reason_and_no_cameras_round_trips():
    status = _status(state=p.NetworkCheckState.UNSUITABLE, reason="could not join", cameras=())
    assert p.NetworkCheckStatus.from_json(_wire(status)) == status


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("cameras"),
        lambda d: d.update(state=9),
        lambda d: d.update(state="READY"),
        lambda d: d.update(state=True),
        lambda d: d.update(generation=""),
        lambda d: d.update(generation=None),
        lambda d: d.update(session_id=""),
        lambda d: d.update(session_id=5),
        lambda d: d.update(endpoint=""),
        lambda d: d.update(phase=3),
        lambda d: d.update(reason=3),
        lambda d: d.update(cameras={}),
        lambda d: d.update(cameras=None),
        lambda d: d.update(cameras=[{}]),
        lambda d: d.update(legs_tested="a"),
        lambda d: d.update(legs_not_tested=[1]),
        lambda d: d.update(window_s=-1),
        lambda d: d.update(window_s=True),
        lambda d: d.update(valid_for_s="300"),
        lambda d: d.update(progress_s=-1),
        lambda d: d.update(progress_s=1.5),
        lambda d: d.update(progress_s=True),
        lambda d: d.update(completed_at_unix_ns=-5),
        lambda d: d.update(completed_at_unix_ns=1.5),
    ],
)
def test_status_parsing_is_strict(mutation):
    wire = _wire(_status())
    mutation(wire)
    with pytest.raises(p.ProtocolError):
        p.NetworkCheckStatus.from_json(wire)


def test_status_none_optionals_stay_none():
    parsed = p.NetworkCheckStatus.from_json(_wire(_status(phase=None, progress_s=None, completed_at_unix_ns=None)))
    assert (parsed.phase, parsed.progress_s, parsed.completed_at_unix_ns, parsed.reason) == (None,) * 4


def test_status_snapshot_omits_network_check_when_none():
    wire = p.StatusSnapshot(updated_at_ns=5).to_json()
    assert "network_check" not in wire


def test_status_snapshot_round_trips_a_network_check():
    snapshot = p.StatusSnapshot(updated_at_ns=5, network_check=_status())
    wire = _wire(snapshot)
    assert wire["network_check"]["state"] == int(p.NetworkCheckState.READY)
    assert p.StatusSnapshot.from_json(wire) == snapshot


def test_status_snapshot_network_check_is_optional_and_absent_means_no_check():
    old = {"updated_at_ns": 5, "providers": [], "devices": [], "cameras": []}
    parsed = p.StatusSnapshot.from_json(old)
    assert parsed.network_check is None
    assert parsed == p.StatusSnapshot(updated_at_ns=5)


def test_status_snapshot_with_a_bad_network_check_is_rejected():
    wire = _wire(p.StatusSnapshot(updated_at_ns=5, network_check=_status()))
    wire["network_check"]["state"] = 99
    with pytest.raises(p.ProtocolError):
        p.StatusSnapshot.from_json(wire)


def test_run_network_check_request_round_trips():
    request = p.RunNetworkCheckRequest(session_id="nc-1")
    wire = _wire(request)
    assert wire == {"type": 20, "version": p.PROTOCOL_VERSION, "session_id": "nc-1"}
    assert p.parse_run_network_check(wire) == request


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("session_id"),
        lambda d: d.pop("version"),
        lambda d: d.update(type=int(p.MessageType.STOP)),
        lambda d: d.update(version=p.PROTOCOL_VERSION + 1),
        lambda d: d.update(session_id=""),
        lambda d: d.update(session_id=7),
        lambda d: d.update(session_id="x" * 500),
    ],
)
def test_run_network_check_rejects_a_bad_envelope(mutation):
    wire = _wire(p.RunNetworkCheckRequest(session_id="nc-1"))
    mutation(wire)
    with pytest.raises(p.ProtocolError):
        p.parse_run_network_check(wire)


@pytest.mark.parametrize("reason", [p.RejectionReason.BUSY, p.RejectionReason.INVALID_CONFIG])
def test_request_rejected_accepts_run_network_check(reason):
    rejected = p.RequestRejected("nc-1", p.MessageType.RUN_NETWORK_CHECK, reason, "no")
    parsed = p.parse_server_message(_wire(rejected))
    assert isinstance(parsed, p.RequestRejected)
    assert (parsed.request_type, parsed.reason) == (p.MessageType.RUN_NETWORK_CHECK, reason)


def test_request_rejected_still_refuses_a_server_message_type_as_the_request():
    with pytest.raises(ValueError):
        p.RequestRejected("x", p.MessageType.SESSION_STATUS, p.RejectionReason.BUSY, "no")


def test_parse_server_message_of_a_session_status_carrying_a_network_check():
    status = _status(state=p.NetworkCheckState.STALE)
    message = p.SessionStatus(session_id="s1", status=p.StatusSnapshot(updated_at_ns=1, network_check=status))
    parsed = p.parse_server_message(_wire(message))
    assert isinstance(parsed, p.SessionStatus)
    assert parsed.status.network_check == status


def test_session_begun_carries_the_network_check_in_its_status():
    snapshot = p.StatusSnapshot(updated_at_ns=1, network_check=_status())
    begun = p.SessionBegun(session_id="s1", status=snapshot)
    parsed = p.parse_server_message(_wire(begun))
    assert isinstance(parsed, p.SessionBegun)
    assert parsed.status.network_check == snapshot.network_check


def test_a_client_never_parses_the_request_as_a_server_message():
    with pytest.raises(p.ProtocolError):
        p.parse_server_message(_wire(p.RunNetworkCheckRequest(session_id="nc-1")))


def test_wire_json_is_not_shared_with_the_source_objects():
    status = _status()
    wire = status.to_json()
    before = copy.deepcopy(wire)
    wire["cameras"].append({})
    wire["legs_tested"].append("x")
    assert status.to_json() == before


def test_camera_track_id_round_trips_and_may_be_none():
    assert p.NetworkCheckCamera.from_json(_wire(_camera())).track_id == "TR_abc"
    none_wire = _wire(_camera(track_id=None))
    assert none_wire["track_id"] is None
    assert p.NetworkCheckCamera.from_json(none_wire).track_id is None


def test_status_session_id_round_trips_and_may_be_none():
    assert p.NetworkCheckStatus.from_json(_wire(_status())).session_id == "sess-1"
    wire = _wire(_status(session_id=None))
    assert wire["session_id"] is None
    assert p.NetworkCheckStatus.from_json(wire).session_id is None


def test_query_status_request_round_trips():
    request = p.QueryStatusRequest(request_id="q-1")
    wire = _wire(request)
    assert wire == {"type": 21, "version": p.PROTOCOL_VERSION, "request_id": "q-1"}
    assert p.parse_query_status(wire) == request


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("request_id"),
        lambda d: d.pop("version"),
        lambda d: d.update(request_id=""),
        lambda d: d.update(request_id=7),
        lambda d: d.update(type=int(p.MessageType.QUERY_DEVICES)),
        lambda d: d.update(version=p.PROTOCOL_VERSION + 1),
    ],
)
def test_query_status_rejects_a_bad_envelope(mutation):
    wire = _wire(p.QueryStatusRequest(request_id="q-1"))
    mutation(wire)
    with pytest.raises(p.ProtocolError):
        p.parse_query_status(wire)


def _reply(**overrides) -> p.ServerStatusReply:
    fields = dict(
        request_id="q-1",
        mode=p.CaptureMode.MOCK,
        session_id="sess-1",
        status=p.StatusSnapshot(updated_at_ns=5, network_check=_status()),
        network_check=_status(),
    )
    fields.update(overrides)
    return p.ServerStatusReply(**fields)


@pytest.mark.parametrize(
    "reply",
    [
        _reply(),
        _reply(session_id=None, status=None),
        _reply(network_check=None),
        _reply(session_id=None, status=None, network_check=None),
    ],
    ids=["session-and-check", "no-session", "no-check", "neither"],
)
def test_server_status_reply_round_trips(reply):
    assert p.ServerStatusReply.from_json(_wire(reply)) == reply


def test_server_status_reply_wire_carries_the_message_type():
    wire = _wire(_reply())
    assert (wire["type"], wire["version"]) == (22, p.PROTOCOL_VERSION)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=1),
        lambda d: d.update(request_id=""),
        lambda d: d.update(mode=99),
        lambda d: d.update(mode="MOCK"),
        lambda d: d.update(session_id=""),
        lambda d: d.update(session_id=3),
        lambda d: d.update(type=int(p.MessageType.SESSION_STATUS)),
        lambda d: d.update(version=p.PROTOCOL_VERSION + 1),
        lambda d: d.update(network_check={"state": 1}),
        lambda d: d.update(status={"updated_at_ns": -1}),
    ],
)
def test_server_status_reply_parsing_is_strict(mutation):
    wire = _wire(_reply())
    mutation(wire)
    with pytest.raises(p.ProtocolError):
        p.ServerStatusReply.from_json(wire)


def test_server_status_reply_is_not_a_session_message():
    # Like DevicesReply it answers a /query request and is parsed by its own from_json, never by the session
    # message dispatcher.
    wire = _wire(_reply())
    with pytest.raises(p.ProtocolError):
        p.parse_server_message(wire)
