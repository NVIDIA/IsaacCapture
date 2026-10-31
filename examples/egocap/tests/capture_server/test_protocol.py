# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from capture_server import protocol as p

# ---------------------------------------------------------------------------
# Binary header.
# ---------------------------------------------------------------------------


def test_header_round_trips():
    header = p.FrameMetadata(
        kind=p.BinaryKind.CAPTURE_FRAME,
        stream_id=0,
        sequence=7,
        capture_time_ns=123,
        receive_time_ns=456,
    )
    message = p.encode_header(header, b"payload")
    assert len(message) == p.HEADER_SIZE + len(b"payload")
    assert message[:4] == b"GTC1"

    decoded_header, payload = p.decode_header(message)
    assert decoded_header == header
    assert payload == b"payload"


def test_header_rejects_short_message():
    with pytest.raises(p.HeaderError, match="too short"):
        p.decode_header(b"short")


def test_header_rejects_bad_magic():
    header = p.FrameMetadata(p.BinaryKind.CAPTURE_IMAGE, 1, 0, 0, 0)
    message = bytearray(p.encode_header(header, b""))
    message[0:4] = b"XXXX"
    with pytest.raises(p.HeaderError, match="bad magic"):
        p.decode_header(bytes(message))


def test_header_rejects_unsupported_version():
    header = p.FrameMetadata(p.BinaryKind.CAPTURE_IMAGE, 1, 0, 0, 0)
    message = bytearray(p.encode_header(header, b""))
    message[4] = 2
    with pytest.raises(p.HeaderError, match="unsupported version"):
        p.decode_header(bytes(message))


def test_header_rejects_unknown_kind():
    header = p.FrameMetadata(p.BinaryKind.CAPTURE_IMAGE, 1, 0, 0, 0)
    message = bytearray(p.encode_header(header, b""))
    message[5] = 99
    with pytest.raises(p.HeaderError, match="unknown binary kind"):
        p.decode_header(bytes(message))


def test_header_error_is_a_protocol_error():
    """A malformed binary header IS a protocol violation -- callers that
    catch ProtocolError (e.g. CaptureClient's receive loop) must catch a
    HeaderError too, not just a plain ValueError."""
    assert issubclass(p.HeaderError, p.ProtocolError)
    with pytest.raises(p.ProtocolError, match="too short"):
        p.decode_header(b"short")


# ---------------------------------------------------------------------------
# ic_tracking_v1 payload.
# ---------------------------------------------------------------------------


def _zero_tracking_values() -> dict:
    values = {}
    for f in p.IC_TRACKING_V1_FIELDS:
        values[f.name] = np.zeros(f.shape, dtype=f.numpy_dtype)
    # Identity quaternions, per "use identity xyzw quaternions as inert
    # placeholders" for absent/unselected groups.
    values["head.orientation"] = np.array([0, 0, 0, 1], dtype=np.float32)
    for prefix in ("controller_left", "controller_right"):
        values[f"{prefix}.grip_orientation"] = np.array([0, 0, 0, 1], dtype=np.float32)
        values[f"{prefix}.aim_orientation"] = np.array([0, 0, 0, 1], dtype=np.float32)
    for side in ("hand_left", "hand_right"):
        values[f"{side}.joint_orientations"] = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (26, 1))
    return values


def test_ic_tracking_v1_payload_size_is_1995():
    assert p.IC_TRACKING_V1_PAYLOAD_SIZE == 1995


def test_ic_tracking_v1_fields_have_no_gaps_or_overlaps():
    expected_offset = 0
    for f in p.IC_TRACKING_V1_FIELDS:
        assert f.offset == expected_offset, f"{f.name} starts at {f.offset}, expected {expected_offset}"
        expected_offset += f.nbytes
    assert expected_offset == p.IC_TRACKING_V1_PAYLOAD_SIZE


def test_encode_tracking_frame_round_trips_exact_bytes():
    values = _zero_tracking_values()
    values["head.present"] = np.uint8(1)
    values["head.position"] = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    values["head.is_valid"] = np.uint8(1)
    values["head.is_tracked"] = np.uint8(1)
    values["controller_left.trigger_value"] = np.float64(0.75)

    payload = p.encode_tracking_frame(values)
    assert len(payload) == 1995

    decoded = p.decode_tracking_frame(payload)
    np.testing.assert_array_equal(decoded["head.position"], values["head.position"])
    assert decoded["controller_left.trigger_value"].dtype == np.float64
    assert float(decoded["controller_left.trigger_value"]) == pytest.approx(0.75)

    # Exact byte check for one field: head.position starts at offset 1,
    # three little-endian float32 values.
    expected_head_position = np.asarray(values["head.position"], dtype="<f4").tobytes()
    assert payload[1:13] == expected_head_position


def test_encode_tracking_frame_rejects_missing_field():
    values = _zero_tracking_values()
    del values["head.position"]
    with pytest.raises(p.TrackingPayloadError, match="missing required field"):
        p.encode_tracking_frame(values)


def test_decode_tracking_frame_rejects_wrong_size():
    with pytest.raises(p.TrackingPayloadError, match="expected 1995"):
        p.decode_tracking_frame(b"\x00" * 10)


def test_tracking_payload_error_is_a_protocol_error():
    """Same rationale as HeaderError -- a malformed tracking payload IS a
    protocol violation, not just a bare ValueError."""
    assert issubclass(p.TrackingPayloadError, p.ProtocolError)
    with pytest.raises(p.ProtocolError, match="expected 1995"):
        p.decode_tracking_frame(b"\x00" * 10)


def test_field_lookup():
    assert p.field("head.position").shape == (3,)
    with pytest.raises(p.TrackingPayloadError, match="no such ic_tracking_v1 field"):
        p.field("nonexistent")


def test_field_wire_dtype_mapping():
    assert p.field("head.present").wire_dtype == p.WireDType.UINT8
    assert p.field("head.position").wire_dtype == p.WireDType.FLOAT32_LE
    assert p.field("controller_left.trigger_value").wire_dtype == p.WireDType.FLOAT64_LE


def test_hand_joint_names_has_26_entries_in_openxr_order():
    assert len(p.HAND_JOINT_NAMES) == 26
    assert p.HAND_JOINT_NAMES[0] == "PALM"
    assert p.HAND_JOINT_NAMES[1] == "WRIST"


def test_tracking_stream_descriptor_matches_sessioninit_example():
    stream = p.tracking_stream_descriptor((p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS), 60.0)
    descriptor = stream.to_json()
    assert descriptor["stream_id"] == 0
    assert descriptor["kind"] == int(p.StreamKind.TRACKING)
    assert descriptor["schema"] == int(p.TrackingSchema.IC_TRACKING_V1)
    assert descriptor["sources"] == [1, 2, 3]
    assert descriptor["pose_representation"] == int(p.PoseRepresentation.POSITION_QUATERNION)
    assert descriptor["quaternion_order"] == int(p.QuaternionOrder.XYZW)
    assert descriptor["hand_joint_order"] == int(p.HandJointOrder.OPENXR_26)
    assert descriptor["reference_space"] == "session_floor"
    assert descriptor["layout"]["payload_size"] == 1995
    assert len(descriptor["layout"]["fields"]) == len(p.IC_TRACKING_V1_FIELDS)

    first_field = descriptor["layout"]["fields"][0]
    assert first_field == {"name": "head.present", "dtype": 1, "shape": [], "offset": 0}

    head_position = descriptor["layout"]["fields"][1]
    assert head_position == {"name": "head.position", "dtype": 2, "shape": [3], "offset": 1}

    round_tripped = p.TrackingStream.from_json(descriptor)
    assert round_tripped == stream


# ---------------------------------------------------------------------------
# Camera stream descriptor.
# ---------------------------------------------------------------------------


def test_camera_stream_descriptor_matches_doc_shape():
    camera = p.CameraStreamConfig(
        stream_id=1,
        name="camera_left",
        type=p.CameraType.OAK_D_COLOR,
        required=True,
        width=640,
        height=480,
        fps=30.0,
        jpeg_quality=90,
    )
    stream = p.camera_stream_descriptor(camera)
    descriptor = stream.to_json()
    assert descriptor == {
        "stream_id": 1,
        "kind": 2,
        "name": "camera_left",
        "type": 1,
        "required": True,
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "codec": 1,
        "jpeg_quality": 90,
        "orientation": {"rotation_degrees": 0, "mirrored": False},
        "intrinsics": None,
    }
    assert p.CameraStream.from_json(descriptor) == stream


def test_camera_stream_descriptor_with_intrinsics():
    camera = p.CameraStreamConfig(
        stream_id=2,
        name="camera_right",
        type=p.CameraType.OAK_D_COLOR,
        required=False,
        width=1920,
        height=1080,
        fps=60.0,
        jpeg_quality=80,
        intrinsics=((1.0, 0.0, 960.0), (0.0, 1.0, 540.0), (0.0, 0.0, 1.0)),
    )
    descriptor = p.camera_stream_descriptor(camera).to_json()
    assert descriptor["intrinsics"] == [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]]


# Binary event wrappers (CaptureFrame/CameraImage) now live in client.py as
# plain, undecoded delivery values -- see test_client.py's "Binary dispatch"
# section. protocol.py no longer unpacks tracking payloads or owns a public
# CameraImage type.


# ---------------------------------------------------------------------------
# Request models: local construction validation (ValueError).
# ---------------------------------------------------------------------------


def test_headset_config_default():
    assert p.HeadsetConfig().device_index == 0


def test_headset_config_rejects_nonzero_index():
    with pytest.raises(ValueError, match="device_index must be 0"):
        p.HeadsetConfig(device_index=1)


def test_tracking_config_defaults():
    config = p.TrackingConfig(sources=(p.SourceType.HEAD,))
    assert config.rate_hz == 60.0
    assert config.headset == p.HeadsetConfig()


def test_tracking_config_rejects_duplicate_sources():
    with pytest.raises(ValueError, match="duplicate"):
        p.TrackingConfig(sources=(p.SourceType.HEAD, p.SourceType.HEAD))


@pytest.mark.parametrize("rate_hz", [0, -1, 121])
def test_tracking_config_rejects_bad_rate_hz(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        p.TrackingConfig(sources=(p.SourceType.HEAD,), rate_hz=rate_hz)


def test_tracking_config_rejects_non_number_rate_hz():
    with pytest.raises(ValueError, match="rate_hz"):
        p.TrackingConfig(sources=(p.SourceType.HEAD,), rate_hz="60")


def test_tracking_config_rejects_non_enum_sources():
    with pytest.raises(ValueError, match="SourceType"):
        p.TrackingConfig(sources=(1,))  # type: ignore[arg-type]


def test_plugin_request_defaults():
    plugin = p.PluginRequest(plugin=p.PluginType.MANUS)
    assert plugin.enabled is True
    assert plugin.required is False
    assert plugin.datasets == (p.ManusDataset.HUMAN,)


def test_plugin_request_rejects_non_enum_plugin():
    with pytest.raises(ValueError, match="PluginType"):
        p.PluginRequest(plugin=1)  # type: ignore[arg-type]


def test_plugin_request_rejects_non_bool_enabled():
    with pytest.raises(ValueError, match="enabled"):
        p.PluginRequest(plugin=p.PluginType.MANUS, enabled=1)  # type: ignore[arg-type]


def test_plugin_request_rejects_non_bool_required():
    with pytest.raises(ValueError, match="required"):
        p.PluginRequest(plugin=p.PluginType.MANUS, required=1)  # type: ignore[arg-type]


def test_plugin_request_rejects_empty_datasets():
    with pytest.raises(ValueError, match="nonempty"):
        p.PluginRequest(plugin=p.PluginType.MANUS, datasets=())


def test_plugin_request_rejects_non_enum_datasets():
    with pytest.raises(ValueError, match="ManusDataset"):
        p.PluginRequest(plugin=p.PluginType.MANUS, datasets=(1,))  # type: ignore[arg-type]


def test_plugin_request_rejects_duplicate_datasets():
    with pytest.raises(ValueError, match="duplicate"):
        p.PluginRequest(plugin=p.PluginType.MANUS, datasets=(p.ManusDataset.HUMAN, p.ManusDataset.HUMAN))


def test_camera_request_rejects_empty_name():
    with pytest.raises(ValueError, match="nonempty"):
        p.CameraRequest(name="", type=p.CameraType.OAK_D_COLOR)


def test_camera_request_rejects_non_enum_type():
    with pytest.raises(ValueError, match="CameraType"):
        p.CameraRequest(name="camera_left", type=1)  # type: ignore[arg-type]


def test_headset_config_rejects_bool_device_index():
    with pytest.raises(ValueError, match="device_index"):
        p.HeadsetConfig(device_index=False)  # type: ignore[arg-type]


def test_begin_request_defaults():
    request = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    assert request.plugins == ()
    assert request.cameras == ()
    assert request.session_id is None


def test_begin_request_rejects_empty_session_id():
    with pytest.raises(ValueError, match="session_id"):
        p.BeginRequest(tracking=p.TrackingConfig(sources=()), session_id="")


def test_begin_request_rejects_duplicate_plugin_kinds():
    with pytest.raises(ValueError, match="duplicate plugin kind"):
        p.BeginRequest(
            tracking=p.TrackingConfig(sources=()),
            plugins=(
                p.PluginRequest(plugin=p.PluginType.MANUS),
                p.PluginRequest(plugin=p.PluginType.MANUS),
            ),
        )


def test_begin_request_rejects_too_many_cameras():
    cameras = tuple(p.CameraRequest(name=f"camera_{i}", type=p.CameraType.OAK_D_COLOR) for i in range(4))
    with pytest.raises(ValueError, match="at most 3"):
        p.BeginRequest(tracking=p.TrackingConfig(sources=()), cameras=cameras)


def test_begin_request_rejects_duplicate_camera_names():
    cameras = (
        p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),
        p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),
    )
    with pytest.raises(ValueError, match="duplicate name"):
        p.BeginRequest(tracking=p.TrackingConfig(sources=()), cameras=cameras)


# ---------------------------------------------------------------------------
# Begin wire parsing (ProtocolError).
# ---------------------------------------------------------------------------


def _valid_begin_dict(**overrides) -> dict:
    data = {
        "type": 1,
        "version": 1,
        "session_id": "session-1",
        "tracking": {
            "sources": [1, 2, 3],
            "rate_hz": 60,
            "headset": {"device_index": 0},
        },
        "plugins": [{"plugin": 1, "enabled": True, "required": True, "datasets": [1]}],
        "cameras": [],
    }
    data.update(overrides)
    return data


def test_parse_begin_minimal():
    msg = p.parse_begin(_valid_begin_dict())
    assert msg.session_id == "session-1"
    assert msg.tracking.sources == (p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS)
    assert msg.tracking.rate_hz == 60.0
    assert msg.tracking.headset.device_index == 0
    assert msg.plugins == (
        p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True, required=True, datasets=(p.ManusDataset.HUMAN,)),
    )
    assert msg.cameras == ()


def test_parse_begin_empty_sources_permits_camera_only():
    msg = p.parse_begin(_valid_begin_dict(tracking={"sources": [], "rate_hz": 60, "headset": None}))
    assert msg.tracking.sources == ()
    assert msg.tracking.headset == p.HeadsetConfig(device_index=0)


def test_parse_begin_defaults_plugins_and_cameras():
    data = _valid_begin_dict()
    del data["plugins"]
    del data["cameras"]
    msg = p.parse_begin(data)
    assert msg.plugins == ()
    assert msg.cameras == ()


def test_parse_begin_with_cameras():
    msg = p.parse_begin(
        _valid_begin_dict(cameras=[{"name": "camera_left", "type": 1}, {"name": "camera_right", "type": 1}])
    )
    assert msg.cameras == (
        p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),
        p.CameraRequest(name="camera_right", type=p.CameraType.OAK_D_COLOR),
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"type": "Begin"}, "must be an integer|expected type"),
        ({"type": 2}, "expected type"),
        ({"version": 2}, "expected version"),
        ({"version": True}, "expected version"),
        ({"session_id": ""}, "session_id"),
        ({"session_id": "x" * 129}, "session_id"),
        ({"session_id": 5}, "session_id"),
        ({"extra_key": 1}, "unknown key"),
    ],
)
def test_parse_begin_rejects_bad_envelope(overrides, match):
    with pytest.raises(p.ProtocolError, match=match):
        p.parse_begin(_valid_begin_dict(**overrides))


def test_parse_begin_rejects_unknown_tracking_key():
    data = _valid_begin_dict()
    data["tracking"]["extra"] = 1
    with pytest.raises(p.ProtocolError, match="unknown key"):
        p.parse_begin(data)


def test_parse_begin_rejects_sources_not_a_list():
    data = _valid_begin_dict()
    data["tracking"]["sources"] = 1
    with pytest.raises(p.ProtocolError, match="sources must be a list"):
        p.parse_begin(data)


def test_parse_begin_rejects_source_as_bool():
    data = _valid_begin_dict()
    data["tracking"]["sources"] = [True]
    with pytest.raises(p.ProtocolError, match="must be an integer"):
        p.parse_begin(data)


def test_parse_begin_rejects_source_as_string_name():
    data = _valid_begin_dict()
    data["tracking"]["sources"] = ["HEAD"]
    with pytest.raises(p.ProtocolError, match="must be an integer"):
        p.parse_begin(data)


def test_parse_begin_rejects_unknown_source_value():
    data = _valid_begin_dict()
    data["tracking"]["sources"] = [99]
    with pytest.raises(p.ProtocolError, match="must be one of"):
        p.parse_begin(data)


def test_parse_begin_rejects_duplicate_sources():
    data = _valid_begin_dict()
    data["tracking"]["sources"] = [1, 1]
    with pytest.raises(p.ProtocolError, match="duplicate"):
        p.parse_begin(data)


@pytest.mark.parametrize("rate_hz", [0, -1, 121, True, "60", float("inf"), float("nan")])
def test_parse_begin_rejects_bad_rate_hz(rate_hz):
    data = _valid_begin_dict()
    data["tracking"]["rate_hz"] = rate_hz
    with pytest.raises(p.ProtocolError, match="rate_hz"):
        p.parse_begin(data)


def test_parse_begin_rejects_non_object_headset():
    data = _valid_begin_dict()
    data["tracking"]["headset"] = False
    with pytest.raises(p.ProtocolError, match="must be a JSON object"):
        p.parse_begin(data)


@pytest.mark.parametrize("device_index", [1, -1, True, "0"])
def test_parse_begin_rejects_nonzero_device_index(device_index):
    data = _valid_begin_dict()
    data["tracking"]["headset"] = {"device_index": device_index}
    with pytest.raises(p.ProtocolError, match="device_index"):
        p.parse_begin(data)


def test_parse_begin_rejects_plugins_not_a_list():
    data = _valid_begin_dict(plugins={})
    with pytest.raises(p.ProtocolError, match="plugins must be a list"):
        p.parse_begin(data)


def test_parse_begin_rejects_duplicate_plugin_kind():
    data = _valid_begin_dict(
        plugins=[
            {"plugin": 1, "enabled": True, "required": False, "datasets": [1]},
            {"plugin": 1, "enabled": True, "required": False, "datasets": [1]},
        ]
    )
    with pytest.raises(p.ProtocolError, match="duplicate plugin kind"):
        p.parse_begin(data)


def test_parse_begin_rejects_unknown_plugin_value():
    data = _valid_begin_dict(plugins=[{"plugin": 99, "enabled": True, "required": False, "datasets": [1]}])
    with pytest.raises(p.ProtocolError, match="must be one of"):
        p.parse_begin(data)


def test_parse_begin_plugin_enabled_required_default():
    data = _valid_begin_dict(plugins=[{"plugin": 1, "datasets": [1]}])
    msg = p.parse_begin(data)
    assert msg.plugins[0].enabled is True
    assert msg.plugins[0].required is False


def test_parse_begin_rejects_plugin_enabled_as_int():
    data = _valid_begin_dict(plugins=[{"plugin": 1, "enabled": 1, "required": False, "datasets": [1]}])
    with pytest.raises(p.ProtocolError, match="must be a boolean"):
        p.parse_begin(data)


def test_parse_begin_plugin_datasets_default():
    data = _valid_begin_dict(plugins=[{"plugin": 1}])
    msg = p.parse_begin(data)
    assert msg.plugins[0].datasets == (p.ManusDataset.HUMAN,)


def test_parse_begin_rejects_empty_datasets():
    data = _valid_begin_dict(plugins=[{"plugin": 1, "datasets": []}])
    with pytest.raises(p.ProtocolError, match="nonempty list"):
        p.parse_begin(data)


def test_parse_begin_rejects_duplicate_datasets():
    data = _valid_begin_dict(plugins=[{"plugin": 1, "datasets": [1, 1]}])
    with pytest.raises(p.ProtocolError, match="duplicate"):
        p.parse_begin(data)


def test_parse_begin_rejects_cameras_not_a_list():
    with pytest.raises(p.ProtocolError, match="cameras must be a list"):
        p.parse_begin(_valid_begin_dict(cameras={}))


def test_parse_begin_rejects_too_many_cameras():
    cameras = [{"name": f"camera_{i}", "type": 1} for i in range(4)]
    with pytest.raises(p.ProtocolError, match="at most 3"):
        p.parse_begin(_valid_begin_dict(cameras=cameras))


def test_parse_begin_rejects_duplicate_camera_names():
    cameras = [{"name": "camera_left", "type": 1}, {"name": "camera_left", "type": 1}]
    with pytest.raises(p.ProtocolError, match="duplicate camera name"):
        p.parse_begin(_valid_begin_dict(cameras=cameras))


def test_parse_begin_rejects_empty_camera_name():
    with pytest.raises(p.ProtocolError, match="nonempty string"):
        p.parse_begin(_valid_begin_dict(cameras=[{"name": "", "type": 1}]))


def test_parse_begin_rejects_unknown_camera_type():
    with pytest.raises(p.ProtocolError, match="must be one of"):
        p.parse_begin(_valid_begin_dict(cameras=[{"name": "camera_left", "type": 99}]))


def test_parse_begin_rejects_unknown_camera_key():
    with pytest.raises(p.ProtocolError, match="unknown key"):
        p.parse_begin(_valid_begin_dict(cameras=[{"name": "camera_left", "type": 1, "device_id": "x"}]))


def test_effective_config_round_trips_begin():
    msg = p.parse_begin(_valid_begin_dict())
    config = p.effective_config(msg)
    assert config.to_json() == {
        "tracking": {"sources": [1, 2, 3], "rate_hz": 60.0, "headset": {"device_index": 0}},
        "plugins": [{"plugin": 1, "enabled": True, "required": True, "datasets": [1]}],
        "cameras": [],
    }
    assert p.CaptureConfiguration.from_json(config.to_json()) == config


# ---------------------------------------------------------------------------
# Stop.
# ---------------------------------------------------------------------------


def test_parse_stop():
    msg = p.parse_stop({"type": 2, "version": 1, "session_id": "session-1"})
    assert msg.session_id == "session-1"


@pytest.mark.parametrize(
    "data,match",
    [
        ({"type": 1, "version": 1, "session_id": "s"}, "expected type"),
        ({"type": 2, "version": 2, "session_id": "s"}, "expected version"),
        ({"type": 2, "version": 1, "session_id": ""}, "session_id"),
        ({"type": 2, "version": 1, "session_id": "s", "extra": 1}, "unknown key"),
        ("not a dict", "must be a JSON object"),
    ],
)
def test_parse_stop_rejects_bad_message(data, match):
    with pytest.raises(p.ProtocolError, match=match):
        p.parse_stop(data)


# ---------------------------------------------------------------------------
# Status snapshot.
# ---------------------------------------------------------------------------


def _sample_status() -> p.StatusSnapshot:
    return p.StatusSnapshot(
        updated_at_ns=124000,
        providers=(
            p.ProviderStatus(
                id="provider-1",
                type=p.ProviderType.PLUGIN,
                plugin=p.PluginType.MANUS,
                name="manus_hand_plugin",
                status=p.ProviderState.AVAILABLE,
                reason=p.StatusReason.NONE,
                error="",
                pid=1234,
            ),
        ),
        devices=(
            p.DeviceStatus(
                id="device-1",
                path="/hand/left",
                type="hand",
                description="Left hand",
                provider_id="provider-1",
                status=p.DeviceState.DISCONNECTED,
                reason=p.StatusReason.NO_HARDWARE_SIGNAL,
                error="",
            ),
        ),
        cameras=(
            p.CameraStatus(
                name="camera_left",
                stream_id=1,
                status=p.DeviceState.CONNECTED,
                reason=p.StatusReason.NONE,
                error="",
                intrinsics=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ),
        ),
    )


def test_status_snapshot_to_json_matches_doc_shape():
    status_json = _sample_status().to_json()
    assert status_json["updated_at_ns"] == 124000
    assert status_json["providers"][0] == {
        "id": "provider-1",
        "type": 1,
        "plugin": 1,
        "name": "manus_hand_plugin",
        "status": 2,
        "reason": 0,
        "error": "",
        "pid": 1234,
        "exit_code": None,
        "term_signal": None,
    }
    assert status_json["devices"][0] == {
        "id": "device-1",
        "path": "/hand/left",
        "type": "hand",
        "description": "Left hand",
        "provider_id": "provider-1",
        "status": 4,
        "reason": 8,
        "error": "",
    }
    assert status_json["cameras"][0]["intrinsics"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_status_snapshot_to_json_empty():
    status_json = p.StatusSnapshot(updated_at_ns=0).to_json()
    assert status_json == {"updated_at_ns": 0, "providers": [], "devices": [], "cameras": []}


def test_provider_plugin_null_for_openxr_runtime():
    status = p.StatusSnapshot(
        updated_at_ns=0,
        providers=(
            p.ProviderStatus(
                id="provider-2",
                type=p.ProviderType.OPENXR_RUNTIME,
                plugin=None,
                name="OpenXR",
                status=p.ProviderState.AVAILABLE,
                reason=p.StatusReason.NONE,
                error="",
            ),
        ),
    )
    assert status.to_json()["providers"][0]["plugin"] is None


# ---------------------------------------------------------------------------
# SessionInit.
# ---------------------------------------------------------------------------


def test_session_init_event_shape():
    begin = p.parse_begin(_valid_begin_dict())
    streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
    event = p.session_init_event(session_id="session-1", mode=p.CaptureMode.MOCK, begin=begin, streams=streams)
    assert event.type == p.MessageType.SESSION_INIT
    assert event.version == 1
    assert event.session_id == "session-1"
    assert event.mode == p.CaptureMode.MOCK
    assert event.effective_config == p.effective_config(begin)
    assert event.clock.to_json() == {"domain": 1, "unit": 1, "sample_time_semantics": 1}
    assert event.streams == tuple(streams)

    message = event.to_json()
    assert message["type"] == 9
    assert message["streams"] == [s.to_json() for s in streams]

    round_tripped = p.SessionInit.from_json(message)
    assert round_tripped == event


# ---------------------------------------------------------------------------
# SessionStarting.
# ---------------------------------------------------------------------------


def test_session_starting_with_null_status():
    event = p.SessionStarting(
        session_id="session-1",
        stage=p.StartupStage.INITIALIZING,
        message="Initializing capture sources",
        status=None,
    )
    assert event.to_json() == {
        "type": 3,
        "version": 1,
        "session_id": "session-1",
        "stage": 1,
        "message": "Initializing capture sources",
        "status": None,
    }


def test_session_starting_with_status():
    event = p.SessionStarting(
        session_id="session-1",
        stage=p.StartupStage.STARTING_SOURCES,
        message="Starting IC sources",
        status=_sample_status(),
    )
    assert event.to_json()["status"] == _sample_status().to_json()


def test_session_starting_rejects_empty_message():
    with pytest.raises(ValueError, match="nonempty string"):
        p.SessionStarting(session_id="session-1", stage=p.StartupStage.INITIALIZING, message="", status=None)


# ---------------------------------------------------------------------------
# SessionBegun.
# ---------------------------------------------------------------------------


def test_session_begun_shape():
    event = p.SessionBegun(session_id="session-1", status=_sample_status())
    assert event.to_json()["type"] == 4
    assert event.to_json()["session_id"] == "session-1"
    assert event.to_json()["status"] == _sample_status().to_json()


# ---------------------------------------------------------------------------
# SessionStatus.
# ---------------------------------------------------------------------------


def test_session_status_shape():
    event = p.SessionStatus(session_id="session-1", status=_sample_status())
    assert event.to_json()["type"] == 5
    assert event.to_json()["status"] == _sample_status().to_json()
    assert p.SessionStatus.from_json(event.to_json()) == event


def test_session_starting_round_trips_through_json():
    event = p.SessionStarting(
        session_id="session-1",
        stage=p.StartupStage.STARTING_SOURCES,
        message="Starting IC sources",
        status=_sample_status(),
    )
    assert p.SessionStarting.from_json(event.to_json()) == event


def test_session_begun_round_trips_through_json():
    event = p.SessionBegun(session_id="session-1", status=_sample_status())
    assert p.SessionBegun.from_json(event.to_json()) == event


# ---------------------------------------------------------------------------
# SessionStopped.
# ---------------------------------------------------------------------------


def test_session_stopped_normal_closure():
    event = p.SessionStopped(
        session_id="session-1",
        reason=p.StopReason.REQUESTED,
        error_message=None,
        status=_sample_status(),
        streams=(p.StreamCounters(stream_id=0, produced=120, sent=117, dropped=3),),
    )
    message = event.to_json()
    assert message["type"] == 6
    assert message["reason"] == 1
    assert message["error_message"] is None
    assert message["streams"] == [{"stream_id": 0, "produced": 120, "sent": 117, "dropped": 3}]
    assert p.SessionStopped.from_json(message) == event


def test_session_stopped_with_error():
    event = p.SessionStopped(
        session_id="session-1",
        reason=p.StopReason.IC_ERROR,
        error_message="camera_left failed to open",
        status=None,
    )
    message = event.to_json()
    assert message["reason"] == 7
    assert message["error_message"] == "camera_left failed to open"
    assert message["status"] is None
    assert message["streams"] == []


def test_session_stopped_rejects_empty_error_message():
    with pytest.raises(ValueError, match="error_message"):
        p.SessionStopped(
            session_id="session-1",
            reason=p.StopReason.IC_ERROR,
            error_message="",
            status=None,
        )


def test_session_stopped_rejects_counter_conservation_violation():
    with pytest.raises(ValueError, match=r"produced=sent\+dropped"):
        p.StreamCounters(stream_id=0, produced=120, sent=100, dropped=1)


# ---------------------------------------------------------------------------
# RequestRejected.
# ---------------------------------------------------------------------------


def test_request_rejected_shape():
    event = p.RequestRejected(
        session_id="session-1",
        request_type=p.MessageType.BEGIN,
        reason=p.RejectionReason.INVALID_CONFIG,
        error_message="Begin configuration differs from the active session",
    )
    assert event.to_json() == {
        "type": 10,
        "version": 1,
        "session_id": "session-1",
        "request_type": 1,
        "reason": 1,
        "error_message": "Begin configuration differs from the active session",
    }
    assert p.RequestRejected.from_json(event.to_json()) == event


def test_request_rejected_for_stop():
    event = p.RequestRejected(
        session_id="session-9",
        request_type=p.MessageType.STOP,
        reason=p.RejectionReason.UNKNOWN_SESSION,
        error_message="unknown session",
    )
    assert event.to_json()["request_type"] == 2
    assert event.to_json()["reason"] == 3


def test_request_rejected_rejects_non_begin_stop_request_type():
    with pytest.raises(ValueError, match="BEGIN or STOP"):
        p.RequestRejected(
            session_id="session-1",
            request_type=p.MessageType.SESSION_INIT,
            reason=p.RejectionReason.BUSY,
            error_message="busy",
        )


def test_request_rejected_rejects_empty_error_message():
    with pytest.raises(ValueError, match="nonempty string"):
        p.RequestRejected(
            session_id="session-1",
            request_type=p.MessageType.BEGIN,
            reason=p.RejectionReason.BUSY,
            error_message="",
        )


# ---------------------------------------------------------------------------
# Enum wire-value table, as a safeguard against accidental reordering.
# ---------------------------------------------------------------------------


def test_message_type_wire_values():
    assert [int(m) for m in p.MessageType] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_status_reason_wire_values_are_0_through_23():
    assert [int(r) for r in p.StatusReason] == list(range(24))


def test_device_state_wire_values():
    assert int(p.DeviceState.INITIALIZING) == 1
    assert int(p.DeviceState.DISABLED) == 8


def test_stop_reason_wire_values_skip_rejection_reason_numbers():
    # 2, 3 and 5 are reserved for RejectionReason (INVALID_CONFIG=1,
    # BUSY=2, UNKNOWN_SESSION=3); StopReason never reuses them.
    assert {int(r) for r in p.StopReason} == {1, 4, 6, 7, 8, 9}


# ---------------------------------------------------------------------------
# from_json rejection paths -- one mutation at a time against a golden
# valid JSON dict, to cover the strict-parsing branches of every received
# value type without hand-writing a full invalid message for each.
# ---------------------------------------------------------------------------


def _assert_rejects(from_json, golden: dict, mutations: dict):
    assert from_json(golden) is not None  # the golden fixture itself must parse
    for key, bad_value in mutations.items():
        mutated = dict(golden)
        if bad_value is _MISSING:
            del mutated[key]
        else:
            mutated[key] = bad_value
        with pytest.raises(p.ProtocolError):
            from_json(mutated)


_MISSING = object()


def test_field_descriptor_from_json_rejects_bad_values():
    golden = {"name": "head.position", "dtype": 2, "shape": [3], "offset": 1}
    _assert_rejects(
        p.FieldDescriptor.from_json,
        golden,
        {
            "name": "",
            "dtype": 99,
            "shape": "not-a-list",
            "offset": -1,
            "extra": 1,
        },
    )


def test_tracking_layout_from_json_rejects_bad_values():
    golden = p.tracking_stream_descriptor((p.SourceType.HEAD,), 60.0).to_json()["layout"]
    _assert_rejects(
        p.TrackingLayout.from_json,
        golden,
        {
            "payload_size": 0,
            "fields": "not-a-list",
        },
    )


def test_tracking_stream_from_json_rejects_bad_values():
    golden = p.tracking_stream_descriptor(
        (p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS), 60.0
    ).to_json()
    _assert_rejects(
        p.TrackingStream.from_json,
        golden,
        {
            "stream_id": 1,
            "name": "",
            "sources": "not-a-list",
            "rate_hz": True,
            "reference_space": "",
        },
    )


@pytest.mark.parametrize("bad_rate_hz", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 121.0])
def test_tracking_stream_from_json_rejects_duplicate_sources_and_bad_rate(bad_rate_hz):
    golden = p.tracking_stream_descriptor((p.SourceType.HEAD,), 60.0).to_json()
    with pytest.raises(p.ProtocolError):
        p.TrackingStream.from_json({**golden, "sources": [1, 1]})
    with pytest.raises(p.ProtocolError):
        p.TrackingStream.from_json({**golden, "rate_hz": bad_rate_hz})


def test_image_orientation_from_json_rejects_bad_values():
    golden = {"rotation_degrees": 0, "mirrored": False}
    _assert_rejects(
        p.ImageOrientation.from_json,
        golden,
        {
            "rotation_degrees": 90,
            "mirrored": True,
        },
    )


def test_camera_stream_from_json_rejects_bad_values():
    golden = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="camera_left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=640,
            height=480,
            fps=30.0,
            jpeg_quality=90,
        )
    ).to_json()
    _assert_rejects(
        p.CameraStream.from_json,
        golden,
        {
            "stream_id": 0,
            "name": "",
            "width": 0,
            "height": -1,
            "fps": True,
            "jpeg_quality": 0,
        },
    )


def test_camera_selection_round_trip_and_rejection():
    config = p.CaptureConfiguration(
        tracking=p.TrackingConfiguration(
            sources=(p.SourceType.HEAD,), rate_hz=60.0, headset=p.HeadsetConfiguration()
        ),
        plugins=(
            p.PluginConfiguration(
                plugin=p.PluginType.MANUS,
                enabled=True,
                required=False,
                datasets=(p.ManusDataset.HUMAN,),
            ),
        ),
        cameras=(p.CameraSelection(name="camera_left", type=p.CameraType.OAK_D_COLOR),),
    )
    golden = config.to_json()
    assert golden["cameras"] == [{"name": "camera_left", "type": 1}]
    assert p.CaptureConfiguration.from_json(golden) == config

    _assert_rejects(p.HeadsetConfiguration.from_json, {"device_index": 0}, {"device_index": 1})
    _assert_rejects(
        p.TrackingConfiguration.from_json,
        {"sources": [1], "rate_hz": 60.0, "headset": {"device_index": 0}},
        {"sources": "x", "rate_hz": -1.0},
    )
    for bad_rate_hz in (float("nan"), float("inf"), float("-inf"), 0.0, 121.0):
        with pytest.raises(p.ProtocolError):
            p.TrackingConfiguration.from_json(
                {"sources": [1], "rate_hz": bad_rate_hz, "headset": {"device_index": 0}}
            )
    with pytest.raises(p.ProtocolError, match="duplicate"):
        p.TrackingConfiguration.from_json({"sources": [1, 1], "rate_hz": 60.0, "headset": {"device_index": 0}})
    _assert_rejects(
        p.PluginConfiguration.from_json,
        {"plugin": 1, "enabled": True, "required": False, "datasets": [1]},
        {"datasets": []},
    )
    _assert_rejects(
        p.CaptureConfiguration.from_json,
        golden,
        {"plugins": "x", "cameras": "x"},
    )
    with pytest.raises(p.ProtocolError, match="nonempty string"):
        p._parse_camera_selection({"name": "", "type": 1})


def test_provider_status_from_json_rejects_bad_values():
    golden = p.ProviderStatus(
        id="provider-1",
        type=p.ProviderType.OPENXR_RUNTIME,
        plugin=None,
        name="OpenXR",
        status=p.ProviderState.AVAILABLE,
        reason=p.StatusReason.NONE,
        error="",
    ).to_json()
    _assert_rejects(
        p.ProviderStatus.from_json,
        golden,
        {
            "id": "",
            "name": 1,
            "error": 1,
        },
    )


def test_device_status_from_json_rejects_bad_values():
    golden = p.DeviceStatus(
        id="device-1",
        path="/hand/left",
        type="hand",
        description="Left hand",
        provider_id="provider-1",
        status=p.DeviceState.CONNECTED,
        reason=p.StatusReason.NONE,
        error="",
    ).to_json()
    _assert_rejects(p.DeviceStatus.from_json, golden, {"id": "", "path": 1})


def test_camera_status_from_json_rejects_bad_values():
    golden = p.CameraStatus(
        name="camera_left",
        stream_id=1,
        status=p.DeviceState.CONNECTED,
        reason=p.StatusReason.NONE,
        error="",
    ).to_json()
    _assert_rejects(
        p.CameraStatus.from_json,
        golden,
        {
            "name": "",
            "stream_id": 0,
            "error": 1,
        },
    )


def test_status_snapshot_from_json_rejects_bad_values():
    golden = p.StatusSnapshot(updated_at_ns=0).to_json()
    _assert_rejects(
        p.StatusSnapshot.from_json,
        golden,
        {
            "updated_at_ns": -1,
            "providers": "x",
            "devices": "x",
            "cameras": "x",
        },
    )


# ---------------------------------------------------------------------------
# parse_server_message dispatch and the remaining event-class rejections.
# ---------------------------------------------------------------------------


def _golden_session_init() -> dict:
    begin = p.parse_begin(_valid_begin_dict())
    streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
    return p.session_init_event(
        session_id="session-1", mode=p.CaptureMode.MOCK, begin=begin, streams=streams
    ).to_json()


def test_session_init_from_json_rejects_bad_values():
    _assert_rejects(
        p.SessionInit.from_json,
        _golden_session_init(),
        {
            "type": 1,
            "session_id": "",
            "streams": "not-a-list",
        },
    )


def test_session_init_from_json_rejects_duplicate_stream_ids():
    golden = _golden_session_init()
    golden = dict(golden, streams=[*golden["streams"], golden["streams"][0]])
    with pytest.raises(p.ProtocolError, match="duplicate stream_id"):
        p.SessionInit.from_json(golden)


def test_session_init_from_json_rejects_a_non_dict_stream_entry():
    """A malformed streams[] entry (not a dict at all) must raise a clean
    ProtocolError from the shared *_.from_json validation it falls through
    to -- not AttributeError from calling .get on something that isn't a
    dict in the first place."""
    golden = _golden_session_init()
    golden = dict(golden, streams=["not-a-stream-object"])
    with pytest.raises(p.ProtocolError):
        p.SessionInit.from_json(golden)


def test_session_starting_from_json_rejects_empty_message():
    golden = p.SessionStarting(
        session_id="session-1", stage=p.StartupStage.INITIALIZING, message="hello", status=None
    ).to_json()
    _assert_rejects(p.SessionStarting.from_json, golden, {"message": ""})


def test_session_stopped_from_json_rejects_bad_values():
    golden = p.SessionStopped(
        session_id="session-1",
        reason=p.StopReason.REQUESTED,
        error_message=None,
        status=None,
    ).to_json()
    _assert_rejects(p.SessionStopped.from_json, golden, {"error_message": "", "streams": "x"})


def test_request_rejected_from_json_rejects_bad_values():
    golden = p.RequestRejected(
        session_id="session-1",
        request_type=p.MessageType.BEGIN,
        reason=p.RejectionReason.INVALID_CONFIG,
        error_message="bad config",
    ).to_json()
    mutated = dict(golden, request_type=int(p.MessageType.SESSION_INIT))
    with pytest.raises(p.ProtocolError, match="BEGIN or STOP"):
        p.RequestRejected.from_json(mutated)
    _assert_rejects(p.RequestRejected.from_json, golden, {"error_message": ""})


def test_parse_server_message_dispatches_every_event_type():
    for event in (
        p.SessionInit.from_json(_golden_session_init()),
        p.SessionStarting(session_id="s", stage=p.StartupStage.INITIALIZING, message="hi", status=None),
        p.SessionBegun(session_id="s", status=p.StatusSnapshot(updated_at_ns=0)),
        p.SessionStatus(session_id="s", status=p.StatusSnapshot(updated_at_ns=0)),
        p.SessionStopped(session_id="s", reason=p.StopReason.REQUESTED, error_message=None, status=None),
        p.RequestRejected(
            session_id="s",
            request_type=p.MessageType.BEGIN,
            reason=p.RejectionReason.BUSY,
            error_message="busy",
        ),
    ):
        assert p.parse_server_message(event.to_json()) == event


def test_parse_server_message_rejects_non_dict_and_wrong_direction_type():
    with pytest.raises(p.ProtocolError, match="must be an object"):
        p.parse_server_message("not-a-dict")
    with pytest.raises(p.ProtocolError, match="unexpected message type"):
        p.parse_server_message(_valid_begin_dict())


def test_parse_intrinsics_rejects_wrong_shape_and_non_numeric_values():
    with pytest.raises(p.ProtocolError, match=r"\[3,3\]"):
        p._parse_intrinsics([[1, 2], [3, 4]], "intrinsics")
    with pytest.raises(p.ProtocolError, match="only numbers"):
        p._parse_intrinsics([["x", 0, 0], [0, 1, 0], [0, 0, 1]], "intrinsics")
    assert p._parse_intrinsics(None, "intrinsics") is None


def test_stream_counters_from_json_round_trips_and_rejects_bad_values():
    golden = p.StreamCounters(stream_id=0, produced=10, sent=8, dropped=2).to_json()
    assert p.StreamCounters.from_json(golden) == p.StreamCounters(stream_id=0, produced=10, sent=8, dropped=2)
    _assert_rejects(
        p.StreamCounters.from_json,
        golden,
        {
            "stream_id": 9,
            "produced": -1,
            "sent": -1,
            "dropped": -1,
        },
    )
