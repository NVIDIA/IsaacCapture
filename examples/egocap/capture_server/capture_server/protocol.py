"""Wire protocol for the generic IsaacCapture (IC) + OAK-D CaptureServer.

Implements the 32-byte GTC1 binary header, the ``ic_tracking_v1`` tracking
payload codec, the request models, the server event dataclasses and their
JSON encode/decode (``to_json``/``from_json``), and strict wire parsing for
Begin/Stop. This module performs no numeric tracking-payload decoding for
delivery to applications -- ``capture_server.client``'s ``CaptureFrame``
hands the raw payload through unchanged: no automatic pose/joint/button/
presence decoding.

Every closed protocol choice is a Python ``IntEnum`` sent as a JSON or
binary integer on the wire -- never an enum name or other string.

This module is acquisition-free: it only packs/unpacks bytes and JSON
dicts, never opens a socket or a device (``capture_server.client`` owns the
WebSocket transport and calls into this module to encode/decode messages).
``ProtocolError`` is raised for a received wire violation: parsing
Begin/Stop (the two families CaptureServer itself receives) and parsing any
server-to-client JSON/binary message or status record (what a client
receives). ``ValueError`` is raised for local construction mistakes in any
of this module's dataclasses -- e.g. building a ``BeginRequest`` or a
``SessionStopped`` with invalid field values before ever touching JSON.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, TypedDict

import numpy as np

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """Raised for a received envelope, descriptor, header or payload that
    violates the protocol. Distinct from ``ValueError``, which this
    module's dataclasses raise for a local construction mistake (building
    a request/event with invalid field values before it ever touches the
    wire). Defined early (before the header/tracking-payload codecs below)
    so ``HeaderError``/``TrackingPayloadError`` can subclass it -- every
    malformed-header or malformed-payload condition IS a protocol
    violation, and callers that catch ``ProtocolError`` (e.g.
    ``CaptureClient``'s receive loop) must catch those too."""


# ---------------------------------------------------------------------------
# Wire IntEnums. Every closed protocol choice is sent on the wire as an
# integer, never an enum name or other string. Values are assigned
# explicitly and never reordered or reused.
# ---------------------------------------------------------------------------


class MessageType(IntEnum):
    BEGIN = 1
    STOP = 2
    SESSION_STARTING = 3
    SESSION_BEGUN = 4
    SESSION_STATUS = 5
    SESSION_STOPPED = 6
    CAPTURE_FRAME = 7
    CAPTURE_IMAGE = 8
    SESSION_INIT = 9
    REQUEST_REJECTED = 10


class SourceType(IntEnum):
    HEAD = 1
    CONTROLLERS = 2
    HANDS = 3


class PluginType(IntEnum):
    MANUS = 1


class ManusDataset(IntEnum):
    HUMAN = 1


class CaptureMode(IntEnum):
    MOCK = 0
    LIVE = 1


class StartupStage(IntEnum):
    INITIALIZING = 1
    WAITING_FOR_HEADSET = 2
    CONNECTING = 3
    STARTING_SOURCES = 4


class StopReason(IntEnum):
    REQUESTED = 1
    OWNER_DISCONNECTED = 4
    SERVER_SHUTDOWN = 6
    IC_ERROR = 7
    IC_EXITED = 8
    CAMERA_ERROR = 9


class RejectionReason(IntEnum):
    INVALID_CONFIG = 1
    BUSY = 2
    UNKNOWN_SESSION = 3


class StreamKind(IntEnum):
    TRACKING = 1
    IMAGE = 2


class BinaryKind(IntEnum):
    CAPTURE_FRAME = 1
    CAPTURE_IMAGE = 2


class TrackingSchema(IntEnum):
    IC_TRACKING_V1 = 1


class WireDType(IntEnum):
    UINT8 = 1
    FLOAT32_LE = 2
    FLOAT64_LE = 3


class CameraType(IntEnum):
    OAK_D_COLOR = 1


class ImageCodec(IntEnum):
    JPEG = 1


class ClockDomain(IntEnum):
    CAPTURE_HOST_MONOTONIC = 1


class TimeUnit(IntEnum):
    NANOSECONDS = 1


class SampleTimeSemantics(IntEnum):
    HOST_POLL = 1


class PoseRepresentation(IntEnum):
    POSITION_QUATERNION = 1


class QuaternionOrder(IntEnum):
    XYZW = 1


class PositionUnit(IntEnum):
    METERS = 1


class Axes(IntEnum):
    RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD = 1


class HandJointOrder(IntEnum):
    OPENXR_26 = 1


class ProviderType(IntEnum):
    PLUGIN = 1
    OPENXR_RUNTIME = 2


class ProviderState(IntEnum):
    INITIALIZING = 1
    AVAILABLE = 2
    FAILED = 3
    STOPPED = 4


class DeviceState(IntEnum):
    INITIALIZING = 1
    UNKNOWN = 2
    CONNECTED = 3
    DISCONNECTED = 4
    DEGRADED = 5
    FAILED = 6
    UNAVAILABLE = 7
    DISABLED = 8


class StatusReason(IntEnum):
    NONE = 0
    PROVIDER_INITIALIZING = 1
    PROVIDER_FAILED = 2
    PROVIDER_STOPPED = 3
    NO_REPORT = 4
    STALE_REPORT = 5
    MALFORMED_REPORT = 6
    UNSUPPORTED_SCHEMA_VERSION = 7
    NO_HARDWARE_SIGNAL = 8
    CALIBRATION_FAILED = 9
    NO_CURRENT_DATA = 10
    DEVICE_ERROR = 11
    DISABLED_BY_CONFIGURATION = 12
    PROCESS_EXITED = 13
    PROCESS_SIGNALED = 14
    PROCESS_OBSERVATION_ERROR = 15
    STARTUP_FAILED = 16
    RUNTIME_UPDATE_FAILED = 17
    OPENXR_SESSION_READY = 18
    EXTERNAL_OPENXR_HEALTH_UNKNOWN = 19
    OPENXR_FORM_FACTOR_UNAVAILABLE = 20
    OPENXR_SESSION_LOST = 21
    OPENXR_INSTANCE_LOST = 22
    OPENXR_POLL_ERROR = 23


# ---------------------------------------------------------------------------
# Binary header: GTC1, 32 bytes, no implicit padding.
# ---------------------------------------------------------------------------

_HEADER_STRUCT = struct.Struct("<4sBBHQQQ")
HEADER_SIZE = _HEADER_STRUCT.size
assert HEADER_SIZE == 32

_MAGIC = b"GTC1"


@dataclass(frozen=True)
class FrameMetadata:
    """The decoded Python values from one binary message's 32-byte header.
    ``stream_id`` identifies which SessionInit stream descriptor this
    message belongs to; the session ID is never repeated per-frame -- it
    comes from the owner connection's active SessionBegun association."""

    kind: BinaryKind
    stream_id: int
    sequence: int
    capture_time_ns: int
    receive_time_ns: int


class HeaderError(ProtocolError):
    """Raised for a binary message whose header is malformed."""


def encode_header(header: FrameMetadata, payload: bytes) -> bytes:
    packed = _HEADER_STRUCT.pack(
        _MAGIC,
        PROTOCOL_VERSION,
        int(header.kind),
        header.stream_id,
        header.sequence,
        header.capture_time_ns,
        header.receive_time_ns,
    )
    return packed + payload


def decode_header(message: bytes) -> tuple[FrameMetadata, bytes]:
    """Split one binary WebSocket message into its header and payload."""
    if len(message) < HEADER_SIZE:
        raise HeaderError(f"message too short for a header: {len(message)} bytes")
    magic, version, kind, stream_id, sequence, capture_time_ns, receive_time_ns = _HEADER_STRUCT.unpack(
        message[:HEADER_SIZE]
    )
    if magic != _MAGIC:
        raise HeaderError(f"bad magic {magic!r}, expected {_MAGIC!r}")
    if version != PROTOCOL_VERSION:
        raise HeaderError(f"unsupported version {version}, expected {PROTOCOL_VERSION}")
    try:
        kind_enum = BinaryKind(kind)
    except ValueError as exc:
        raise HeaderError(f"unknown binary kind {kind}") from exc
    header = FrameMetadata(
        kind=kind_enum,
        stream_id=stream_id,
        sequence=sequence,
        capture_time_ns=capture_time_ns,
        receive_time_ns=receive_time_ns,
    )
    return header, message[HEADER_SIZE:]


# ---------------------------------------------------------------------------
# ic_tracking_v1: the combined tracking payload, stream ID 0.
# ---------------------------------------------------------------------------

_WIRE_DTYPE_BY_NUMPY: dict[str, WireDType] = {
    "u1": WireDType.UINT8,
    "f4": WireDType.FLOAT32_LE,
    "f8": WireDType.FLOAT64_LE,
}


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str  # numpy dtype string: "u1", "f4", "f8"
    shape: tuple[int, ...]
    offset: int

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def wire_dtype(self) -> WireDType:
        return _WIRE_DTYPE_BY_NUMPY[self.dtype]

    @property
    def nbytes(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count * self.numpy_dtype.itemsize


def _hand_fields(prefix: str, start_offset: int) -> list[Field]:
    offset = start_offset
    fields = [Field(f"{prefix}.present", "u1", (), offset)]
    offset += fields[-1].nbytes
    fields.append(Field(f"{prefix}.joint_positions", "f4", (26, 3), offset))
    offset += fields[-1].nbytes
    fields.append(Field(f"{prefix}.joint_orientations", "f4", (26, 4), offset))
    offset += fields[-1].nbytes
    fields.append(Field(f"{prefix}.joint_radii", "f4", (26,), offset))
    offset += fields[-1].nbytes
    fields.append(Field(f"{prefix}.joint_valid", "u1", (26,), offset))
    return fields


def _controller_fields(prefix: str, start_offset: int) -> list[Field]:
    offset = start_offset
    names_dtypes_shapes = [
        ("present", "u1", ()),
        ("grip_position", "f4", (3,)),
        ("grip_orientation", "f4", (4,)),
        ("grip_is_valid", "u1", ()),
        ("aim_position", "f4", (3,)),
        ("aim_orientation", "f4", (4,)),
        ("aim_is_valid", "u1", ()),
        ("primary_click", "f8", ()),
        ("secondary_click", "f8", ()),
        ("thumbstick_x", "f8", ()),
        ("thumbstick_y", "f8", ()),
        ("thumbstick_click", "f8", ()),
        ("menu_click", "f8", ()),
        ("squeeze_value", "f8", ()),
        ("trigger_value", "f8", ()),
    ]
    fields = []
    for name, dtype, shape in names_dtypes_shapes:
        f = Field(f"{prefix}.{name}", dtype, shape, offset)
        fields.append(f)
        offset += f.nbytes
    return fields


def _build_ic_tracking_v1_fields() -> tuple[Field, ...]:
    fields: list[Field] = []
    offset = 0

    head_spec = [
        ("head.present", "u1", ()),
        ("head.position", "f4", (3,)),
        ("head.orientation", "f4", (4,)),
        ("head.is_valid", "u1", ()),
        ("head.is_tracked", "u1", ()),
    ]
    for name, dtype, shape in head_spec:
        f = Field(name, dtype, shape, offset)
        fields.append(f)
        offset += f.nbytes

    left_controller = _controller_fields("controller_left", offset)
    fields.extend(left_controller)
    offset = left_controller[-1].offset + left_controller[-1].nbytes

    right_controller = _controller_fields("controller_right", offset)
    fields.extend(right_controller)
    offset = right_controller[-1].offset + right_controller[-1].nbytes

    left_hand = _hand_fields("hand_left", offset)
    fields.extend(left_hand)
    offset = left_hand[-1].offset + left_hand[-1].nbytes

    right_hand = _hand_fields("hand_right", offset)
    fields.extend(right_hand)
    offset = right_hand[-1].offset + right_hand[-1].nbytes

    return tuple(fields)


IC_TRACKING_V1_FIELDS: tuple[Field, ...] = _build_ic_tracking_v1_fields()
IC_TRACKING_V1_PAYLOAD_SIZE = IC_TRACKING_V1_FIELDS[-1].offset + IC_TRACKING_V1_FIELDS[-1].nbytes
assert IC_TRACKING_V1_PAYLOAD_SIZE == 1995, (
    f"ic_tracking_v1 payload size drifted: {IC_TRACKING_V1_PAYLOAD_SIZE}, "
    "expected the fixed 1995-byte tracking payload"
)

# 26-joint OpenXR hand order: palm, wrist, four thumb joints, then five each
# for index/middle/ring/little. Never reordered, never reduced to 25 (that
# is a client-side conversion, not something CaptureServer performs).
HAND_JOINT_NAMES: tuple[str, ...] = (
    "PALM",
    "WRIST",
    "THUMB_METACARPAL",
    "THUMB_PROXIMAL",
    "THUMB_DISTAL",
    "THUMB_TIP",
    "INDEX_METACARPAL",
    "INDEX_PROXIMAL",
    "INDEX_INTERMEDIATE",
    "INDEX_DISTAL",
    "INDEX_TIP",
    "MIDDLE_METACARPAL",
    "MIDDLE_PROXIMAL",
    "MIDDLE_INTERMEDIATE",
    "MIDDLE_DISTAL",
    "MIDDLE_TIP",
    "RING_METACARPAL",
    "RING_PROXIMAL",
    "RING_INTERMEDIATE",
    "RING_DISTAL",
    "RING_TIP",
    "LITTLE_METACARPAL",
    "LITTLE_PROXIMAL",
    "LITTLE_INTERMEDIATE",
    "LITTLE_DISTAL",
    "LITTLE_TIP",
)
assert len(HAND_JOINT_NAMES) == 26


class TrackingPayloadError(ProtocolError):
    """Raised for a tracking payload that doesn't match ic_tracking_v1."""


_FIELDS_BY_NAME: dict[str, Field] = {f.name: f for f in IC_TRACKING_V1_FIELDS}


def encode_tracking_frame(values: dict[str, Any]) -> bytes:
    """Pack *values* (field name -> array-like/scalar) into one
    ic_tracking_v1 payload. Every declared field must be present; this
    function does not fill in defaults -- callers (the mock source) decide
    what an absent group's zeroed/identity placeholder values are."""
    missing = [f.name for f in IC_TRACKING_V1_FIELDS if f.name not in values]
    if missing:
        raise TrackingPayloadError(f"missing required field(s): {missing}")
    buf = bytearray(IC_TRACKING_V1_PAYLOAD_SIZE)
    for field_ in IC_TRACKING_V1_FIELDS:
        arr = np.asarray(values[field_.name], dtype=field_.numpy_dtype).reshape(field_.shape)
        buf[field_.offset : field_.offset + field_.nbytes] = arr.tobytes()
    return bytes(buf)


def decode_tracking_frame(payload: bytes) -> dict[str, np.ndarray]:
    """Unpack one ic_tracking_v1 payload into field name -> numpy array."""
    if len(payload) != IC_TRACKING_V1_PAYLOAD_SIZE:
        raise TrackingPayloadError(f"payload is {len(payload)} bytes, expected {IC_TRACKING_V1_PAYLOAD_SIZE}")
    result: dict[str, np.ndarray] = {}
    for field_ in IC_TRACKING_V1_FIELDS:
        raw = payload[field_.offset : field_.offset + field_.nbytes]
        arr = np.frombuffer(raw, dtype=field_.numpy_dtype).reshape(field_.shape)
        result[field_.name] = arr.copy()
    return result


def field(name: str) -> Field:
    try:
        return _FIELDS_BY_NAME[name]
    except KeyError:
        raise TrackingPayloadError(f"no such ic_tracking_v1 field: {name!r}") from None


def _intrinsics_to_json(
    intrinsics: tuple[tuple[float, float, float], ...] | None,
) -> list[list[float]] | None:
    return [list(row) for row in intrinsics] if intrinsics is not None else None


def _parse_intrinsics(data: Any, context: str) -> tuple[tuple[float, float, float], ...] | None:
    if data is None:
        return None
    if (
        not isinstance(data, list)
        or len(data) != 3
        or not all(isinstance(row, list) and len(row) == 3 for row in data)
    ):
        raise ProtocolError(f"{context} must be null or a [3,3] numeric array")
    try:
        return tuple(tuple(float(v) for v in row) for row in data)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{context} must contain only numbers") from exc


@dataclass(frozen=True)
class FieldDescriptor:
    """One field of a TrackingLayout."""

    name: str
    dtype: WireDType
    shape: tuple[int, ...]
    offset: int

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": int(self.dtype),
            "shape": list(self.shape),
            "offset": self.offset,
        }

    @classmethod
    def from_json(cls, data: Any) -> FieldDescriptor:
        _require_keys(data, {"name", "dtype", "shape", "offset"}, "a field descriptor")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError("field.name must be a nonempty string")
        dtype = _require_int_enum(data.get("dtype"), WireDType, "field.dtype")
        shape_raw = data.get("shape")
        if not isinstance(shape_raw, list) or not all(type(d) is int and d > 0 for d in shape_raw):
            raise ProtocolError("field.shape must be a list of positive integers")
        offset = data.get("offset")
        if type(offset) is not int or offset < 0:
            raise ProtocolError("field.offset must be a nonnegative integer")
        return cls(name=name, dtype=dtype, shape=tuple(shape_raw), offset=offset)


@dataclass(frozen=True)
class TrackingLayout:
    """The complete ic_tracking_v1 byte-level decoding contract. Only used
    to parse the wire ``layout`` object into ``TrackingStream.payload_size``/
    ``fields`` -- ``TrackingStream.to_json()`` builds the wire ``layout``
    dict directly, so this type has no ``to_json()`` of its own."""

    payload_size: int
    fields: tuple[FieldDescriptor, ...]

    @classmethod
    def from_json(cls, data: Any) -> TrackingLayout:
        _require_keys(data, {"payload_size", "fields"}, "layout")
        payload_size = data.get("payload_size")
        if type(payload_size) is not int or payload_size <= 0:
            raise ProtocolError("layout.payload_size must be a positive integer")
        fields_raw = data.get("fields")
        if not isinstance(fields_raw, list):
            raise ProtocolError("layout.fields must be a list")
        fields = tuple(FieldDescriptor.from_json(f) for f in fields_raw)
        return cls(payload_size=payload_size, fields=fields)


def ic_tracking_v1_layout() -> TrackingLayout:
    return TrackingLayout(
        payload_size=IC_TRACKING_V1_PAYLOAD_SIZE,
        fields=tuple(
            FieldDescriptor(name=f.name, dtype=f.wire_dtype, shape=f.shape, offset=f.offset)
            for f in IC_TRACKING_V1_FIELDS
        ),
    )


@dataclass(frozen=True)
class TrackingStream:
    """The stream_id=0 descriptor of SessionInit's ``streams`` array,
    re-exported as ``FrameDescription`` from ``capture_server.client``: applications
    decode ``CaptureFrame.payload`` themselves using ``payload_size``/
    ``fields`` here, so this Python type carries the same byte-level
    information as the wire ``layout`` object it serializes to/from."""

    stream_id: int
    kind: StreamKind
    name: str
    schema: TrackingSchema
    sources: tuple[SourceType, ...]
    rate_hz: float
    pose_representation: PoseRepresentation
    quaternion_order: QuaternionOrder
    position_unit: PositionUnit
    axes: Axes
    reference_space: str
    hand_joint_order: HandJointOrder
    payload_size: int
    fields: tuple[FieldDescriptor, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "kind": int(self.kind),
            "name": self.name,
            "schema": int(self.schema),
            "sources": [int(s) for s in self.sources],
            "rate_hz": self.rate_hz,
            "pose_representation": int(self.pose_representation),
            "quaternion_order": int(self.quaternion_order),
            "position_unit": int(self.position_unit),
            "axes": int(self.axes),
            "reference_space": self.reference_space,
            "hand_joint_order": int(self.hand_joint_order),
            "layout": {
                "payload_size": self.payload_size,
                "fields": [f.to_json() for f in self.fields],
            },
        }

    @classmethod
    def from_json(cls, data: Any) -> TrackingStream:
        _require_keys(
            data,
            {
                "stream_id",
                "kind",
                "name",
                "schema",
                "sources",
                "rate_hz",
                "pose_representation",
                "quaternion_order",
                "position_unit",
                "axes",
                "reference_space",
                "hand_joint_order",
                "layout",
            },
            "a tracking stream descriptor",
        )
        stream_id = data.get("stream_id")
        if stream_id != 0:
            raise ProtocolError("tracking stream descriptor stream_id must be 0")
        kind = _require_int_enum(data.get("kind"), StreamKind, "stream.kind")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError("stream.name must be a nonempty string")
        schema = _require_int_enum(data.get("schema"), TrackingSchema, "stream.schema")
        sources_raw = data.get("sources")
        if not isinstance(sources_raw, list):
            raise ProtocolError("stream.sources must be a list")
        sources = tuple(_require_int_enum(s, SourceType, "stream.sources[]") for s in sources_raw)
        _require_unique_sources(sources, "stream.sources")
        rate_hz = _require_rate_hz(data.get("rate_hz"), "stream.rate_hz")
        reference_space = data.get("reference_space")
        if not isinstance(reference_space, str) or not reference_space:
            raise ProtocolError("stream.reference_space must be a nonempty string")
        layout = TrackingLayout.from_json(data.get("layout"))
        return cls(
            stream_id=stream_id,
            kind=kind,
            name=name,
            schema=schema,
            sources=sources,
            rate_hz=rate_hz,
            pose_representation=_require_int_enum(
                data.get("pose_representation"), PoseRepresentation, "stream.pose_representation"
            ),
            quaternion_order=_require_int_enum(
                data.get("quaternion_order"), QuaternionOrder, "stream.quaternion_order"
            ),
            position_unit=_require_int_enum(data.get("position_unit"), PositionUnit, "stream.position_unit"),
            axes=_require_int_enum(data.get("axes"), Axes, "stream.axes"),
            reference_space=reference_space,
            hand_joint_order=_require_int_enum(
                data.get("hand_joint_order"), HandJointOrder, "stream.hand_joint_order"
            ),
            payload_size=layout.payload_size,
            fields=layout.fields,
        )


def tracking_stream_descriptor(
    sources: tuple[SourceType, ...],
    rate_hz: float,
    *,
    reference_space: str = "session_floor",
) -> TrackingStream:
    """The stream_id=0 entry of SessionInit's ``streams`` array."""
    layout = ic_tracking_v1_layout()
    return TrackingStream(
        stream_id=0,
        kind=StreamKind.TRACKING,
        name="tracking",
        schema=TrackingSchema.IC_TRACKING_V1,
        sources=sources,
        rate_hz=rate_hz,
        pose_representation=PoseRepresentation.POSITION_QUATERNION,
        quaternion_order=QuaternionOrder.XYZW,
        position_unit=PositionUnit.METERS,
        axes=Axes.RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD,
        reference_space=reference_space,
        hand_joint_order=HandJointOrder.OPENXR_26,
        payload_size=layout.payload_size,
        fields=layout.fields,
    )


# ---------------------------------------------------------------------------
# Camera stream descriptor (stream IDs 1-3).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageOrientation:
    rotation_degrees: int = 0
    mirrored: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"rotation_degrees": self.rotation_degrees, "mirrored": self.mirrored}

    @classmethod
    def from_json(cls, data: Any) -> ImageOrientation:
        _require_keys(data, {"rotation_degrees", "mirrored"}, "orientation")
        rotation_degrees = data.get("rotation_degrees")
        if rotation_degrees != 0:
            raise ProtocolError("orientation.rotation_degrees must be 0")
        mirrored = _require_strict_bool(data.get("mirrored"), "orientation.mirrored")
        if mirrored:
            raise ProtocolError("orientation.mirrored must be false")
        return cls(rotation_degrees=rotation_degrees, mirrored=mirrored)


@dataclass(frozen=True)
class CameraStreamConfig:
    """The server-resolved settings for one accepted camera request,
    returned in SessionInit and never negotiated through Begin."""

    stream_id: int
    name: str
    type: CameraType
    required: bool
    width: int
    height: int
    fps: float
    jpeg_quality: int
    intrinsics: tuple[tuple[float, float, float], ...] | None = None
    rotation_degrees: int = 0
    mirrored: bool = False


@dataclass(frozen=True)
class CameraStream:
    """One camera descriptor (stream_id 1-3) of SessionInit's ``streams``
    array."""

    stream_id: int
    kind: StreamKind
    name: str
    type: CameraType
    required: bool
    width: int
    height: int
    fps: float
    codec: ImageCodec
    jpeg_quality: int
    orientation: ImageOrientation
    intrinsics: tuple[tuple[float, float, float], ...] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "kind": int(self.kind),
            "name": self.name,
            "type": int(self.type),
            "required": self.required,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "codec": int(self.codec),
            "jpeg_quality": self.jpeg_quality,
            "orientation": self.orientation.to_json(),
            "intrinsics": _intrinsics_to_json(self.intrinsics),
        }

    @classmethod
    def from_json(cls, data: Any) -> CameraStream:
        _require_keys(
            data,
            {
                "stream_id",
                "kind",
                "name",
                "type",
                "required",
                "width",
                "height",
                "fps",
                "codec",
                "jpeg_quality",
                "orientation",
                "intrinsics",
            },
            "a camera stream descriptor",
        )
        stream_id = data.get("stream_id")
        if type(stream_id) is not int or not (1 <= stream_id <= 3):
            raise ProtocolError("camera stream descriptor stream_id must be an integer 1-3")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError("camera stream.name must be a nonempty string")
        width, height = data.get("width"), data.get("height")
        if type(width) is not int or width <= 0:
            raise ProtocolError("camera stream.width must be a positive integer")
        if type(height) is not int or height <= 0:
            raise ProtocolError("camera stream.height must be a positive integer")
        fps = data.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
            raise ProtocolError("camera stream.fps must be a positive number")
        jpeg_quality = data.get("jpeg_quality")
        if type(jpeg_quality) is not int or not (1 <= jpeg_quality <= 100):
            raise ProtocolError("camera stream.jpeg_quality must be an integer 1-100")
        return cls(
            stream_id=stream_id,
            kind=_require_int_enum(data.get("kind"), StreamKind, "camera stream.kind"),
            name=name,
            type=_require_int_enum(data.get("type"), CameraType, "camera stream.type"),
            required=_require_strict_bool(data.get("required"), "camera stream.required"),
            width=width,
            height=height,
            fps=float(fps),
            codec=_require_int_enum(data.get("codec"), ImageCodec, "camera stream.codec"),
            jpeg_quality=jpeg_quality,
            orientation=ImageOrientation.from_json(data.get("orientation")),
            intrinsics=_parse_intrinsics(data.get("intrinsics"), "camera stream.intrinsics"),
        )


def camera_stream_descriptor(camera: CameraStreamConfig) -> CameraStream:
    return CameraStream(
        stream_id=camera.stream_id,
        kind=StreamKind.IMAGE,
        name=camera.name,
        type=camera.type,
        required=camera.required,
        width=camera.width,
        height=camera.height,
        fps=camera.fps,
        codec=ImageCodec.JPEG,
        jpeg_quality=camera.jpeg_quality,
        orientation=ImageOrientation(rotation_degrees=camera.rotation_degrees, mirrored=camera.mirrored),
        intrinsics=camera.intrinsics,
    )


# ---------------------------------------------------------------------------
# Shared JSON validation helpers, used only by the wire-parsing functions
# below (parse_begin/parse_stop) -- the only two families CaptureServer
# itself receives.
# ---------------------------------------------------------------------------

_MAX_SESSION_ID_LEN = 128
_MAX_RATE_HZ = 120.0
_MAX_CAMERAS = 3


def _require_keys(data: Any, allowed: set[str], context: str) -> dict:
    if not isinstance(data, dict):
        raise ProtocolError(f"{context} must be a JSON object")
    unknown = set(data) - allowed
    if unknown:
        raise ProtocolError(f"{context} has unknown key(s): {sorted(unknown)}")
    return data


def _require_strict_bool(value: Any, context: str) -> bool:
    # bool is a subclass of int in Python; isinstance(True, int) is True,
    # so check the exact type rather than isinstance to reject an integer
    # where a boolean is required.
    if type(value) is not bool:
        raise ProtocolError(f"{context} must be a boolean, got {type(value).__name__}")
    return value


def _require_rate_hz(value: Any, context: str) -> float:
    # `0 < value <= _MAX_RATE_HZ` already rejects NaN (every comparison
    # with NaN is False) and +/-inf (inf is never <= a finite bound), so
    # one range check covers "a number, finite, positive, in range" --
    # no separate math.isfinite/isnan call is needed.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{context} must be a number")
    value = float(value)
    if not (0 < value <= _MAX_RATE_HZ):
        raise ProtocolError(f"{context} must be finite, positive and at most {_MAX_RATE_HZ}")
    return value


def _require_unique_sources(sources: tuple[SourceType, ...], context: str) -> None:
    if len(sources) != len(set(sources)):
        raise ProtocolError(f"{context} has duplicate entries: {[int(s) for s in sources]}")


def _require_int_enum(value: Any, enum_cls: type[IntEnum], context: str) -> Any:
    # type(x) is int, not isinstance: a JSON boolean decodes to Python
    # bool, a subclass of int, and must not silently become a valid enum
    # member (e.g. SourceType(True) would otherwise resolve to HEAD).
    if type(value) is not int:
        raise ProtocolError(f"{context} must be an integer, got {type(value).__name__}")
    try:
        return enum_cls(value)
    except ValueError:
        raise ProtocolError(f"{context} must be one of {[m.value for m in enum_cls]}, got {value}") from None


def _require_version(data: dict) -> None:
    version = data.get("version")
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise ProtocolError(f"expected version {PROTOCOL_VERSION}")


def _require_message_type(data: dict, expected: MessageType) -> None:
    actual = data.get("type")
    if type(actual) is not int or actual != int(expected):
        raise ProtocolError(f"expected type {int(expected)} ({expected.name}), got {actual!r}")


def _require_session_id(data: dict) -> str:
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id or len(session_id) > _MAX_SESSION_ID_LEN:
        raise ProtocolError(f"session_id must be a nonempty string of at most {_MAX_SESSION_ID_LEN} characters")
    return session_id


def require_envelope(data: dict, expected_type: MessageType) -> str:
    """Envelope-level validation shared by every message kind that needs
    it: message type, protocol version, and the canonical session_id rule
    (nonempty string, bounded length) -- the exact split between a
    malformed envelope (closes 1002) and a merely unsupported
    configuration (RequestRejected). Returns the validated
    session_id. Used both by ``parse_begin``/``parse_stop`` (which go on
    to validate message-specific content) and directly by
    ``CaptureServer``'s own pre-dispatch envelope check, so there is
    exactly one definition of a valid envelope, never two independently
    maintained ones that can quietly drift apart."""
    _require_message_type(data, expected_type)
    _require_version(data)
    return _require_session_id(data)


# ---------------------------------------------------------------------------
# Request models. Each raises ValueError from __post_init__ for a local
# construction mistake.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadsetConfig:
    device_index: int = dataclass_field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        # type(x) is int, not isinstance: a caller-supplied bool must not
        # silently pass as 0/1 (False != 0 is False, so a plain != check
        # would wrongly accept False).
        if type(self.device_index) is not int or self.device_index != 0:
            raise ValueError("HeadsetConfig.device_index must be 0 (the only supported headset index)")


@dataclass(frozen=True)
class TrackingConfig:
    sources: tuple[SourceType, ...]
    rate_hz: float = dataclass_field(default=60.0, kw_only=True)
    headset: HeadsetConfig = dataclass_field(default_factory=HeadsetConfig, kw_only=True)

    def __post_init__(self) -> None:
        if not all(type(s) is SourceType for s in self.sources):
            raise ValueError(
                "TrackingConfig.sources must contain only SourceType members, got "
                f"{[type(s).__name__ for s in self.sources]}"
            )
        if len(self.sources) != len(set(self.sources)):
            raise ValueError(f"TrackingConfig.sources has duplicate entries: {[int(s) for s in self.sources]}")
        if isinstance(self.rate_hz, bool) or not isinstance(self.rate_hz, (int, float)):
            raise ValueError("TrackingConfig.rate_hz must be a number")
        if not (0 < self.rate_hz <= _MAX_RATE_HZ):
            raise ValueError(f"TrackingConfig.rate_hz must be in (0, {_MAX_RATE_HZ}]")


@dataclass(frozen=True)
class PluginRequest:
    plugin: PluginType
    enabled: bool = dataclass_field(default=True, kw_only=True)
    required: bool = dataclass_field(default=False, kw_only=True)
    datasets: tuple[ManusDataset, ...] = dataclass_field(default=(ManusDataset.HUMAN,), kw_only=True)

    def __post_init__(self) -> None:
        if type(self.plugin) is not PluginType:
            raise ValueError(f"PluginRequest.plugin must be a PluginType, got {self.plugin!r}")
        if type(self.enabled) is not bool:
            raise ValueError("PluginRequest.enabled must be a boolean")
        if type(self.required) is not bool:
            raise ValueError("PluginRequest.required must be a boolean")
        if not self.datasets:
            raise ValueError("PluginRequest.datasets must be nonempty")
        if not all(type(d) is ManusDataset for d in self.datasets):
            raise ValueError(
                "PluginRequest.datasets must contain only ManusDataset members, got "
                f"{[type(d).__name__ for d in self.datasets]}"
            )
        if len(self.datasets) != len(set(self.datasets)):
            raise ValueError(f"PluginRequest.datasets has duplicate entries: {[int(d) for d in self.datasets]}")


@dataclass(frozen=True)
class CameraRequest:
    name: str
    type: CameraType

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CameraRequest.name must be a nonempty string")
        if type(self.type) is not CameraType:
            raise ValueError(f"CameraRequest.type must be a CameraType, got {self.type!r}")


@dataclass(frozen=True)
class BeginRequest:
    tracking: TrackingConfig
    plugins: tuple[PluginRequest, ...] = dataclass_field(default=(), kw_only=True)
    cameras: tuple[CameraRequest, ...] = dataclass_field(default=(), kw_only=True)
    session_id: str | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.session_id is not None and (not self.session_id or len(self.session_id) > _MAX_SESSION_ID_LEN):
            raise ValueError(
                "BeginRequest.session_id must be None or a nonempty string of at most "
                f"{_MAX_SESSION_ID_LEN} characters"
            )
        kinds = [p.plugin for p in self.plugins]
        if len(kinds) != len(set(kinds)):
            raise ValueError(f"BeginRequest.plugins has duplicate plugin kind(s): {[int(k) for k in kinds]}")
        if len(self.cameras) > _MAX_CAMERAS:
            raise ValueError(f"BeginRequest.cameras supports at most {_MAX_CAMERAS} entries")
        names = [c.name for c in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError(f"BeginRequest.cameras has duplicate name(s): {names}")


class CameraSelection(TypedDict):
    """A station-configured logical camera selection. Contains only name
    and type."""

    name: str
    type: CameraType


def _camera_selection_to_json(selection: CameraSelection) -> dict[str, Any]:
    return {"name": selection["name"], "type": int(selection["type"])}


def _parse_camera_selection(data: Any) -> CameraSelection:
    _require_keys(data, {"name", "type"}, "a camera selection")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ProtocolError("camera selection name must be a nonempty string")
    camera_type = _require_int_enum(data.get("type"), CameraType, "camera selection type")
    # CameraSelection is a TypedDict, a plain dict at runtime; wrap it so a
    # received CaptureConfiguration snapshot can't be mutated in place by a
    # caller holding onto it, matching the documented immutable-snapshot
    # contract. CameraSelection({...}) still accepts a plain dict (`start`'s
    # own input shape), since MappingProxyType only wraps the return value.
    return MappingProxyType(CameraSelection(name=name, type=camera_type))  # type: ignore[return-value]


@dataclass(frozen=True)
class HeadsetConfiguration:
    """The resolved headset selection echoed back in
    ``CaptureConfiguration.tracking.headset``."""

    device_index: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"device_index": self.device_index}

    @classmethod
    def from_json(cls, data: Any) -> HeadsetConfiguration:
        _require_keys(data, {"device_index"}, "headset configuration")
        device_index = data.get("device_index")
        if type(device_index) is not int or device_index != 0:
            raise ProtocolError("headset configuration device_index must be integer 0")
        return cls(device_index=device_index)


@dataclass(frozen=True)
class TrackingConfiguration:
    """The resolved tracking selection echoed back in
    ``CaptureConfiguration.tracking``."""

    sources: tuple[SourceType, ...]
    rate_hz: float
    headset: HeadsetConfiguration

    def to_json(self) -> dict[str, Any]:
        return {
            "sources": [int(s) for s in self.sources],
            "rate_hz": self.rate_hz,
            "headset": self.headset.to_json(),
        }

    @classmethod
    def from_json(cls, data: Any) -> TrackingConfiguration:
        _require_keys(data, {"sources", "rate_hz", "headset"}, "tracking configuration")
        sources_raw = data.get("sources")
        if not isinstance(sources_raw, list):
            raise ProtocolError("tracking configuration sources must be a list")
        sources = tuple(_require_int_enum(s, SourceType, "tracking configuration sources[]") for s in sources_raw)
        _require_unique_sources(sources, "tracking configuration sources")
        rate_hz = _require_rate_hz(data.get("rate_hz"), "tracking configuration rate_hz")
        return cls(
            sources=sources,
            rate_hz=rate_hz,
            headset=HeadsetConfiguration.from_json(data.get("headset")),
        )


@dataclass(frozen=True)
class PluginConfiguration:
    """The resolved plugin selection echoed back in
    ``CaptureConfiguration.plugins``."""

    plugin: PluginType
    enabled: bool
    required: bool
    datasets: tuple[ManusDataset, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "plugin": int(self.plugin),
            "enabled": self.enabled,
            "required": self.required,
            "datasets": [int(d) for d in self.datasets],
        }

    @classmethod
    def from_json(cls, data: Any) -> PluginConfiguration:
        _require_keys(data, {"plugin", "enabled", "required", "datasets"}, "plugin configuration")
        datasets_raw = data.get("datasets")
        if not isinstance(datasets_raw, list) or not datasets_raw:
            raise ProtocolError("plugin configuration datasets must be a nonempty list")
        return cls(
            plugin=_require_int_enum(data.get("plugin"), PluginType, "plugin configuration plugin"),
            enabled=_require_strict_bool(data.get("enabled"), "plugin configuration enabled"),
            required=_require_strict_bool(data.get("required"), "plugin configuration required"),
            datasets=tuple(
                _require_int_enum(d, ManusDataset, "plugin configuration datasets[]") for d in datasets_raw
            ),
        )


@dataclass(frozen=True)
class CaptureConfiguration:
    """``SessionInit.effective_config``: the accepted Begin selections with
    all defaults resolved."""

    tracking: TrackingConfiguration
    plugins: tuple[PluginConfiguration, ...]
    cameras: tuple[CameraSelection, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "tracking": self.tracking.to_json(),
            "plugins": [p.to_json() for p in self.plugins],
            "cameras": [_camera_selection_to_json(c) for c in self.cameras],
        }

    @classmethod
    def from_json(cls, data: Any) -> CaptureConfiguration:
        _require_keys(data, {"tracking", "plugins", "cameras"}, "effective_config")
        plugins_raw = data.get("plugins")
        cameras_raw = data.get("cameras")
        if not isinstance(plugins_raw, list):
            raise ProtocolError("effective_config.plugins must be a list")
        if not isinstance(cameras_raw, list):
            raise ProtocolError("effective_config.cameras must be a list")
        return cls(
            tracking=TrackingConfiguration.from_json(data.get("tracking")),
            plugins=tuple(PluginConfiguration.from_json(p) for p in plugins_raw),
            cameras=tuple(_parse_camera_selection(c) for c in cameras_raw),
        )


def effective_config(begin: BeginRequest) -> CaptureConfiguration:
    """The ``effective_config`` object SessionInit echoes back: the resolved
    Begin configuration, including defaults."""
    return CaptureConfiguration(
        tracking=TrackingConfiguration(
            sources=begin.tracking.sources,
            rate_hz=begin.tracking.rate_hz,
            headset=HeadsetConfiguration(device_index=begin.tracking.headset.device_index),
        ),
        plugins=tuple(
            PluginConfiguration(plugin=p.plugin, enabled=p.enabled, required=p.required, datasets=p.datasets)
            for p in begin.plugins
        ),
        cameras=tuple(CameraSelection(name=c.name, type=c.type) for c in begin.cameras),
    )


@dataclass(frozen=True)
class CaptureClock:
    """Explains how to interpret binary-header timestamps."""

    domain: ClockDomain
    unit: TimeUnit
    sample_time_semantics: SampleTimeSemantics

    def to_json(self) -> dict[str, Any]:
        return {
            "domain": int(self.domain),
            "unit": int(self.unit),
            "sample_time_semantics": int(self.sample_time_semantics),
        }

    @classmethod
    def from_json(cls, data: Any) -> CaptureClock:
        _require_keys(data, {"domain", "unit", "sample_time_semantics"}, "clock")
        return cls(
            domain=_require_int_enum(data.get("domain"), ClockDomain, "clock.domain"),
            unit=_require_int_enum(data.get("unit"), TimeUnit, "clock.unit"),
            sample_time_semantics=_require_int_enum(
                data.get("sample_time_semantics"), SampleTimeSemantics, "clock.sample_time_semantics"
            ),
        )


def _capture_clock() -> CaptureClock:
    return CaptureClock(
        domain=ClockDomain.CAPTURE_HOST_MONOTONIC,
        unit=TimeUnit.NANOSECONDS,
        sample_time_semantics=SampleTimeSemantics.HOST_POLL,
    )


# ---------------------------------------------------------------------------
# Begin/Stop wire parsing (client -> server; the only two families
# CaptureServer itself receives).
# ---------------------------------------------------------------------------


def parse_begin(data: dict) -> BeginRequest:
    _require_keys(data, {"type", "version", "session_id", "tracking", "plugins", "cameras"}, "Begin")
    session_id = require_envelope(data, MessageType.BEGIN)

    tracking = _parse_tracking_request(data.get("tracking"))
    plugins = _parse_plugin_requests(data.get("plugins", []))
    cameras = _parse_camera_requests(data.get("cameras", []))

    return BeginRequest(session_id=session_id, tracking=tracking, plugins=plugins, cameras=cameras)


def _parse_tracking_request(data: Any) -> TrackingConfig:
    _require_keys(data, {"sources", "rate_hz", "headset"}, "Begin.tracking")

    sources_raw = data.get("sources")
    if not isinstance(sources_raw, list):
        raise ProtocolError("tracking.sources must be a list")
    sources = tuple(_require_int_enum(s, SourceType, "tracking.sources[]") for s in sources_raw)
    if len(sources) != len(set(sources)):
        raise ProtocolError(f"tracking.sources has duplicate entries: {[int(s) for s in sources]}")

    rate_hz = data.get("rate_hz", 60)
    if type(rate_hz) is bool or not isinstance(rate_hz, (int, float)):
        raise ProtocolError("tracking.rate_hz must be a number")
    rate_hz = float(rate_hz)
    if not (0 < rate_hz <= _MAX_RATE_HZ):
        raise ProtocolError(f"tracking.rate_hz must be in (0, {_MAX_RATE_HZ}]")

    headset_raw = data.get("headset")
    if headset_raw is None:
        headset = HeadsetConfig()
    else:
        _require_keys(headset_raw, {"device_index"}, "tracking.headset")
        device_index = headset_raw.get("device_index", 0)
        if type(device_index) is not int or device_index != 0:
            raise ProtocolError("tracking.headset.device_index must be integer 0")
        headset = HeadsetConfig(device_index=device_index)

    return TrackingConfig(sources=sources, rate_hz=rate_hz, headset=headset)


def _parse_plugin_requests(data: Any) -> tuple[PluginRequest, ...]:
    if not isinstance(data, list):
        raise ProtocolError("plugins must be a list")
    requests = tuple(_parse_plugin_request(p) for p in data)
    kinds = [r.plugin for r in requests]
    if len(kinds) != len(set(kinds)):
        raise ProtocolError(f"plugins has duplicate plugin kind(s): {[int(k) for k in kinds]}")
    return requests


def _parse_plugin_request(data: Any) -> PluginRequest:
    _require_keys(data, {"plugin", "enabled", "required", "datasets"}, "a plugin request")
    plugin = _require_int_enum(data.get("plugin"), PluginType, "plugin.plugin")
    enabled = _require_strict_bool(data.get("enabled", True), "plugin.enabled")
    required = _require_strict_bool(data.get("required", False), "plugin.required")

    datasets_raw = data.get("datasets", [int(ManusDataset.HUMAN)])
    if not isinstance(datasets_raw, list) or not datasets_raw:
        raise ProtocolError("plugin.datasets must be a nonempty list")
    datasets = tuple(_require_int_enum(d, ManusDataset, "plugin.datasets[]") for d in datasets_raw)
    if len(datasets) != len(set(datasets)):
        raise ProtocolError(f"plugin.datasets has duplicate entries: {[int(d) for d in datasets]}")

    return PluginRequest(plugin=plugin, enabled=enabled, required=required, datasets=datasets)


def _parse_camera_requests(data: Any) -> tuple[CameraRequest, ...]:
    if not isinstance(data, list):
        raise ProtocolError("cameras must be a list")
    if len(data) > _MAX_CAMERAS:
        raise ProtocolError(f"at most {_MAX_CAMERAS} camera requests are supported")
    cameras = tuple(_parse_camera_request(c) for c in data)
    names = [c.name for c in cameras]
    if len(names) != len(set(names)):
        raise ProtocolError(f"duplicate camera name(s) in cameras: {names}")
    return cameras


def _parse_camera_request(data: Any) -> CameraRequest:
    _require_keys(data, {"name", "type"}, "a camera request")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ProtocolError("camera.name must be a nonempty string")
    camera_type = _require_int_enum(data.get("type"), CameraType, "camera.type")
    return CameraRequest(name=name, type=camera_type)


@dataclass(frozen=True)
class StopRequest:
    session_id: str


def parse_stop(data: dict) -> StopRequest:
    _require_keys(data, {"type", "version", "session_id"}, "Stop")
    session_id = require_envelope(data, MessageType.STOP)
    return StopRequest(session_id=session_id)


# ---------------------------------------------------------------------------
# Status value objects. SessionStarting/SessionBegun/SessionStatus/
# SessionStopped all reuse this one schema.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderStatus:
    id: str
    type: ProviderType
    plugin: PluginType | None
    name: str
    status: ProviderState
    reason: StatusReason
    error: str
    pid: int | None = None
    exit_code: int | None = None
    term_signal: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": int(self.type),
            "plugin": int(self.plugin) if self.plugin is not None else None,
            "name": self.name,
            "status": int(self.status),
            "reason": int(self.reason),
            "error": self.error,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "term_signal": self.term_signal,
        }

    @classmethod
    def from_json(cls, data: Any) -> ProviderStatus:
        _require_keys(
            data,
            {"id", "type", "plugin", "name", "status", "reason", "error", "pid", "exit_code", "term_signal"},
            "a provider status record",
        )
        provider_id = data.get("id")
        if not isinstance(provider_id, str) or not provider_id:
            raise ProtocolError("provider.id must be a nonempty string")
        plugin_raw = data.get("plugin")
        plugin = None if plugin_raw is None else _require_int_enum(plugin_raw, PluginType, "provider.plugin")
        name = data.get("name")
        if not isinstance(name, str):
            raise ProtocolError("provider.name must be a string")
        error = data.get("error")
        if not isinstance(error, str):
            raise ProtocolError("provider.error must be a string")
        return cls(
            id=provider_id,
            type=_require_int_enum(data.get("type"), ProviderType, "provider.type"),
            plugin=plugin,
            name=name,
            status=_require_int_enum(data.get("status"), ProviderState, "provider.status"),
            reason=_require_int_enum(data.get("reason"), StatusReason, "provider.reason"),
            error=error,
            pid=data.get("pid"),
            exit_code=data.get("exit_code"),
            term_signal=data.get("term_signal"),
        )


@dataclass(frozen=True)
class DeviceStatus:
    id: str
    path: str
    type: str
    description: str
    provider_id: str
    status: DeviceState
    reason: StatusReason
    error: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "type": self.type,
            "description": self.description,
            "provider_id": self.provider_id,
            "status": int(self.status),
            "reason": int(self.reason),
            "error": self.error,
        }

    @classmethod
    def from_json(cls, data: Any) -> DeviceStatus:
        _require_keys(
            data,
            {"id", "path", "type", "description", "provider_id", "status", "reason", "error"},
            "a device status record",
        )
        device_id = data.get("id")
        if not isinstance(device_id, str) or not device_id:
            raise ProtocolError("device.id must be a nonempty string")
        for key in ("path", "type", "description", "provider_id", "error"):
            if not isinstance(data.get(key), str):
                raise ProtocolError(f"device.{key} must be a string")
        return cls(
            id=device_id,
            path=data["path"],
            type=data["type"],
            description=data["description"],
            provider_id=data["provider_id"],
            status=_require_int_enum(data.get("status"), DeviceState, "device.status"),
            reason=_require_int_enum(data.get("reason"), StatusReason, "device.reason"),
            error=data["error"],
        )


@dataclass(frozen=True)
class CameraStatus:
    name: str
    stream_id: int
    status: DeviceState
    reason: StatusReason
    error: str
    intrinsics: tuple[tuple[float, float, float], ...] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stream_id": self.stream_id,
            "status": int(self.status),
            "reason": int(self.reason),
            "error": self.error,
            "intrinsics": ([list(row) for row in self.intrinsics] if self.intrinsics is not None else None),
        }

    @classmethod
    def from_json(cls, data: Any) -> CameraStatus:
        _require_keys(
            data,
            {"name", "stream_id", "status", "reason", "error", "intrinsics"},
            "a camera status record",
        )
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError("camera status name must be a nonempty string")
        stream_id = data.get("stream_id")
        if type(stream_id) is not int or not (1 <= stream_id <= 3):
            raise ProtocolError("camera status stream_id must be an integer 1-3")
        error = data.get("error")
        if not isinstance(error, str):
            raise ProtocolError("camera status error must be a string")
        return cls(
            name=name,
            stream_id=stream_id,
            status=_require_int_enum(data.get("status"), DeviceState, "camera status.status"),
            reason=_require_int_enum(data.get("reason"), StatusReason, "camera status.reason"),
            error=error,
            intrinsics=_parse_intrinsics(data.get("intrinsics"), "camera status.intrinsics"),
        )


@dataclass(frozen=True)
class StatusSnapshot:
    updated_at_ns: int
    providers: tuple[ProviderStatus, ...] = ()
    devices: tuple[DeviceStatus, ...] = ()
    cameras: tuple[CameraStatus, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "updated_at_ns": self.updated_at_ns,
            "providers": [p.to_json() for p in self.providers],
            "devices": [d.to_json() for d in self.devices],
            "cameras": [c.to_json() for c in self.cameras],
        }

    @classmethod
    def from_json(cls, data: Any) -> StatusSnapshot:
        _require_keys(data, {"updated_at_ns", "providers", "devices", "cameras"}, "a status snapshot")
        updated_at_ns = data.get("updated_at_ns")
        if type(updated_at_ns) is not int or updated_at_ns < 0:
            raise ProtocolError("status.updated_at_ns must be a nonnegative integer")
        providers_raw, devices_raw, cameras_raw = (data.get("providers"), data.get("devices"), data.get("cameras"))
        if not isinstance(providers_raw, list):
            raise ProtocolError("status.providers must be a list")
        if not isinstance(devices_raw, list):
            raise ProtocolError("status.devices must be a list")
        if not isinstance(cameras_raw, list):
            raise ProtocolError("status.cameras must be a list")
        return cls(
            updated_at_ns=updated_at_ns,
            providers=tuple(ProviderStatus.from_json(p) for p in providers_raw),
            devices=tuple(DeviceStatus.from_json(d) for d in devices_raw),
            cameras=tuple(CameraStatus.from_json(c) for c in cameras_raw),
        )


def _parse_optional_status(data: Any) -> StatusSnapshot | None:
    return None if data is None else StatusSnapshot.from_json(data)


# ---------------------------------------------------------------------------
# Server event classes. CaptureServer constructs and serializes these;
# capture_server.client parses received JSON back into them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionInit:
    session_id: str
    mode: CaptureMode
    effective_config: CaptureConfiguration
    clock: CaptureClock
    streams: tuple[TrackingStream | CameraStream, ...]
    type: MessageType = dataclass_field(default=MessageType.SESSION_INIT, init=False)
    version: int = dataclass_field(default=PROTOCOL_VERSION, init=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": int(self.type),
            "version": self.version,
            "session_id": self.session_id,
            "mode": int(self.mode),
            "effective_config": self.effective_config.to_json(),
            "clock": self.clock.to_json(),
            "streams": [s.to_json() for s in self.streams],
        }

    @classmethod
    def from_json(cls, data: Any) -> SessionInit:
        _require_keys(
            data,
            {"type", "version", "session_id", "mode", "effective_config", "clock", "streams"},
            "SessionInit",
        )
        _require_message_type(data, MessageType.SESSION_INIT)
        _require_version(data)
        session_id = _require_session_id(data)
        mode = _require_int_enum(data.get("mode"), CaptureMode, "SessionInit.mode")
        streams_raw = data.get("streams")
        if not isinstance(streams_raw, list):
            raise ProtocolError("SessionInit.streams must be a list")
        streams = tuple(
            TrackingStream.from_json(s)
            # isinstance guard first: a non-dict entry (e.g. a string or
            # list, however malformed) must still reach *_.from_json's own
            # _require_keys check and raise a clean ProtocolError there,
            # never AttributeError from calling .get on something that
            # isn't a dict.
            if isinstance(s, dict) and s.get("kind") == int(StreamKind.TRACKING)
            else CameraStream.from_json(s)
            for s in streams_raw
        )
        stream_ids = [s.stream_id for s in streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise ProtocolError(f"SessionInit.streams has duplicate stream_id(s): {stream_ids}")
        return cls(
            session_id=session_id,
            mode=mode,
            effective_config=CaptureConfiguration.from_json(data.get("effective_config")),
            clock=CaptureClock.from_json(data.get("clock")),
            streams=streams,
        )


def session_init_event(
    *,
    session_id: str,
    mode: CaptureMode,
    begin: BeginRequest,
    streams: list[TrackingStream | CameraStream],
) -> SessionInit:
    return SessionInit(
        session_id=session_id,
        mode=mode,
        effective_config=effective_config(begin),
        clock=_capture_clock(),
        streams=tuple(streams),
    )


@dataclass(frozen=True)
class SessionStarting:
    session_id: str
    stage: StartupStage
    message: str
    status: StatusSnapshot | None
    type: MessageType = dataclass_field(default=MessageType.SESSION_STARTING, init=False)
    version: int = dataclass_field(default=PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("SessionStarting.message must be a nonempty string")

    def to_json(self) -> dict[str, Any]:
        return {
            "type": int(self.type),
            "version": self.version,
            "session_id": self.session_id,
            "stage": int(self.stage),
            "message": self.message,
            "status": self.status.to_json() if self.status is not None else None,
        }

    @classmethod
    def from_json(cls, data: Any) -> SessionStarting:
        _require_keys(
            data,
            {"type", "version", "session_id", "stage", "message", "status"},
            "SessionStarting",
        )
        _require_message_type(data, MessageType.SESSION_STARTING)
        _require_version(data)
        session_id = _require_session_id(data)
        message = data.get("message")
        if not isinstance(message, str) or not message:
            raise ProtocolError("SessionStarting.message must be a nonempty string")
        return cls(
            session_id=session_id,
            stage=_require_int_enum(data.get("stage"), StartupStage, "SessionStarting.stage"),
            message=message,
            status=_parse_optional_status(data.get("status")),
        )


@dataclass(frozen=True)
class SessionBegun:
    session_id: str
    status: StatusSnapshot
    type: MessageType = dataclass_field(default=MessageType.SESSION_BEGUN, init=False)
    version: int = dataclass_field(default=PROTOCOL_VERSION, init=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": int(self.type),
            "version": self.version,
            "session_id": self.session_id,
            "status": self.status.to_json(),
        }

    @classmethod
    def from_json(cls, data: Any) -> SessionBegun:
        _require_keys(data, {"type", "version", "session_id", "status"}, "SessionBegun")
        _require_message_type(data, MessageType.SESSION_BEGUN)
        _require_version(data)
        session_id = _require_session_id(data)
        return cls(session_id=session_id, status=StatusSnapshot.from_json(data.get("status")))


@dataclass(frozen=True)
class SessionStatus:
    session_id: str
    status: StatusSnapshot
    type: MessageType = dataclass_field(default=MessageType.SESSION_STATUS, init=False)
    version: int = dataclass_field(default=PROTOCOL_VERSION, init=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": int(self.type),
            "version": self.version,
            "session_id": self.session_id,
            "status": self.status.to_json(),
        }

    @classmethod
    def from_json(cls, data: Any) -> SessionStatus:
        _require_keys(data, {"type", "version", "session_id", "status"}, "SessionStatus")
        _require_message_type(data, MessageType.SESSION_STATUS)
        _require_version(data)
        session_id = _require_session_id(data)
        return cls(session_id=session_id, status=StatusSnapshot.from_json(data.get("status")))


@dataclass(frozen=True)
class StreamCounters:
    """Final per-stream delivery accounting. ``produced`` must equal
    ``sent + dropped``."""

    stream_id: int
    produced: int
    sent: int
    dropped: int

    def __post_init__(self) -> None:
        if self.produced != self.sent + self.dropped:
            raise ValueError(
                f"stream {self.stream_id} counters violate produced=sent+dropped: "
                f"{self.produced} != {self.sent}+{self.dropped}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "produced": self.produced,
            "sent": self.sent,
            "dropped": self.dropped,
        }

    @classmethod
    def from_json(cls, data: Any) -> StreamCounters:
        _require_keys(data, {"stream_id", "produced", "sent", "dropped"}, "stream counters")
        stream_id = data.get("stream_id")
        if type(stream_id) is not int or not (0 <= stream_id <= 3):
            raise ProtocolError("stream counters stream_id must be an integer 0-3")
        values = {}
        for key in ("produced", "sent", "dropped"):
            value = data.get(key)
            if type(value) is not int or value < 0:
                raise ProtocolError(f"stream counters {key} must be a nonnegative integer")
            values[key] = value
        return cls(stream_id=stream_id, **values)


@dataclass(frozen=True)
class SessionStopped:
    session_id: str
    reason: StopReason
    error_message: str | None
    status: StatusSnapshot | None
    streams: tuple[StreamCounters, ...] = ()
    type: MessageType = dataclass_field(default=MessageType.SESSION_STOPPED, init=False)
    version: int = dataclass_field(default=PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        if self.error_message is not None and not self.error_message:
            raise ValueError("SessionStopped.error_message must be None or a nonempty string")

    def to_json(self) -> dict[str, Any]:
        return {
            "type": int(self.type),
            "version": self.version,
            "session_id": self.session_id,
            "reason": int(self.reason),
            "error_message": self.error_message,
            "status": self.status.to_json() if self.status is not None else None,
            "streams": [s.to_json() for s in self.streams],
        }

    @classmethod
    def from_json(cls, data: Any) -> SessionStopped:
        _require_keys(
            data,
            {"type", "version", "session_id", "reason", "error_message", "status", "streams"},
            "SessionStopped",
        )
        _require_message_type(data, MessageType.SESSION_STOPPED)
        _require_version(data)
        session_id = _require_session_id(data)
        error_message = data.get("error_message")
        if error_message is not None and (not isinstance(error_message, str) or not error_message):
            raise ProtocolError("SessionStopped.error_message must be null or a nonempty string")
        streams_raw = data.get("streams")
        if not isinstance(streams_raw, list):
            raise ProtocolError("SessionStopped.streams must be a list")
        return cls(
            session_id=session_id,
            reason=_require_int_enum(data.get("reason"), StopReason, "SessionStopped.reason"),
            error_message=error_message,
            status=_parse_optional_status(data.get("status")),
            streams=tuple(StreamCounters.from_json(s) for s in streams_raw),
        )


@dataclass(frozen=True)
class RequestRejected:
    session_id: str
    request_type: MessageType
    reason: RejectionReason
    error_message: str
    type: MessageType = dataclass_field(default=MessageType.REQUEST_REJECTED, init=False)
    version: int = dataclass_field(default=PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        if self.request_type not in (MessageType.BEGIN, MessageType.STOP):
            raise ValueError("RequestRejected.request_type must be BEGIN or STOP")
        if not self.error_message:
            raise ValueError("RequestRejected.error_message must be a nonempty string")

    def to_json(self) -> dict[str, Any]:
        return {
            "type": int(self.type),
            "version": self.version,
            "session_id": self.session_id,
            "request_type": int(self.request_type),
            "reason": int(self.reason),
            "error_message": self.error_message,
        }

    @classmethod
    def from_json(cls, data: Any) -> RequestRejected:
        _require_keys(
            data,
            {"type", "version", "session_id", "request_type", "reason", "error_message"},
            "RequestRejected",
        )
        _require_message_type(data, MessageType.REQUEST_REJECTED)
        _require_version(data)
        session_id = _require_session_id(data)
        request_type = _require_int_enum(data.get("request_type"), MessageType, "RequestRejected.request_type")
        if request_type not in (MessageType.BEGIN, MessageType.STOP):
            raise ProtocolError("RequestRejected.request_type must be BEGIN or STOP")
        error_message = data.get("error_message")
        if not isinstance(error_message, str) or not error_message:
            raise ProtocolError("RequestRejected.error_message must be a nonempty string")
        return cls(
            session_id=session_id,
            request_type=request_type,
            reason=_require_int_enum(data.get("reason"), RejectionReason, "RequestRejected.reason"),
            error_message=error_message,
        )


# A server-to-client JSON message, dispatched by its envelope's "type".
ServerJsonMessage = SessionInit | SessionStarting | SessionBegun | SessionStatus | SessionStopped | RequestRejected

_JSON_PARSERS_BY_TYPE: dict[MessageType, Any] = {
    MessageType.SESSION_INIT: SessionInit.from_json,
    MessageType.SESSION_STARTING: SessionStarting.from_json,
    MessageType.SESSION_BEGUN: SessionBegun.from_json,
    MessageType.SESSION_STATUS: SessionStatus.from_json,
    MessageType.SESSION_STOPPED: SessionStopped.from_json,
    MessageType.REQUEST_REJECTED: RequestRejected.from_json,
}


def parse_server_message(data: Any) -> ServerJsonMessage:
    """Parse one received JSON text message into its typed event class,
    dispatching on the envelope's ``type``. Raises ProtocolError for an
    unknown or wrong-direction type (e.g. Begin/Stop, which only
    CaptureServer itself receives)."""
    if not isinstance(data, dict):
        raise ProtocolError("a received JSON message must be an object")
    message_type = _require_int_enum(data.get("type"), MessageType, "message.type")
    parser = _JSON_PARSERS_BY_TYPE.get(message_type)
    if parser is None:
        raise ProtocolError(f"unexpected message type for a client to receive: {message_type!r}")
    return parser(data)
