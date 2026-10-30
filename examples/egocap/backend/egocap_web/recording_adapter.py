"""Converts the native CaptureServer tracking/camera callbacks into
``RecordingService``-ready samples, and owns the bounded ingress
queue/consumer that converts and submits them -- see
``recording/recording_service/README.md`` for the recording service's own
contract.

Data contract (verified against actual source -- see each function's
docstring for citations): quaternions are XYZW
(``capture_server.protocol.QuaternionOrder.XYZW``); hand joints are
OpenXR's 26-joint layout (``HandJointOrder.OPENXR_26``: PALM=0, WRIST=1,
4 thumb joints, then 5 joints each for index/middle/ring/little); head and
controller validity are explicit wire bits
(``head.is_valid``/``{side}.aim_is_valid``), never inferred from pose
content; controller pose is AIM, never GRIP; face buttons are
``{side}.primary_click``/``secondary_click`` thresholded ``> 0.5``; hand
conversion reproduces the wrist-relative Manus skeleton convention from the
OpenXR hand-joint layout. No native device/SDK module is imported
at runtime -- the math is a narrow, pure-function port.

This module never infers validity from pose VALUES (an identity or
all-zero transform that is explicitly marked valid is real data, not a
sentinel) -- every validity decision here traces back to an actual wire
bit, threaded through explicitly as a boolean mask, never reconstructed
after the fact from what a converted pose happens to look like.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from capture_server import protocol as capture_protocol
from capture_server.client import CameraImage, CaptureFrame, FrameDescription
from recording_service import (
    FieldSpec,
    NumericSample,
    NumericSourceSpec,
    PixelFormat,
    SubmissionStatus,
    VideoSample,
)

Side = Literal["left", "right"]

PICO_CONTROLLER_SOURCE = "pico_controller"
MANUS_GLOVE_SOURCE = "manus_glove"

# The source schemas the conversion functions below produce, defined beside
# them so the schema and the code that fills it cannot drift apart. Fixed
# regardless of which cameras are configured: head/controller/hand tracking
# is always recordable once a capture attempt is accepted.
PICO_CONTROLLER_SPEC = NumericSourceSpec(
    PICO_CONTROLLER_SOURCE,
    (
        FieldSpec("head_pose", "float64", (4, 4)),
        FieldSpec("left_pose", "float64", (4, 4)),
        FieldSpec("right_pose", "float64", (4, 4)),
        FieldSpec("left_trigger", "float64"),
        FieldSpec("right_trigger", "float64"),
        FieldSpec("a_button", "bool"),
        FieldSpec("b_button", "bool"),
        FieldSpec("x_button", "bool"),
        FieldSpec("y_button", "bool"),
    ),
)
MANUS_GLOVE_SPEC = NumericSourceSpec(
    MANUS_GLOVE_SOURCE,
    (
        FieldSpec("left_hand_pose", "float64", (25, 4, 4)),
        FieldSpec("right_hand_pose", "float64", (25, 4, 4)),
    ),
)


class RecordingAdapterError(ValueError):
    """Raised when the actually-offered tracking descriptor doesn't match
    the fixed contract this adapter was written against -- a real
    incompatibility, not a business rejection; the caller should disable
    the profile rather than silently convert mismatched data."""


def validate_tracking_descriptor(descriptor: FrameDescription) -> None:
    """Confirms the native descriptor matches every assumption this
    module's conversion math depends on, once per capture attempt.
    ``descriptor.fields`` (wire ``FieldDescriptor`` instances, once
    reconstructed off the network) is compared against
    ``ic_tracking_v1_layout().fields`` -- the SAME ``FieldDescriptor``
    conversion of the authoritative ``IC_TRACKING_V1_FIELDS`` tuple that
    CaptureServer's own protocol module already defines for this exact
    purpose, not ``IC_TRACKING_V1_FIELDS`` directly (a tuple of the
    internal ``Field`` type, a different dataclass that can never compare
    equal to a wire ``FieldDescriptor``). ``FieldDescriptor`` is a frozen
    dataclass, so this single equality check still proves every field's
    name, dtype, shape AND offset match exactly."""
    p = capture_protocol
    problems: list[str] = []
    if descriptor.schema != p.TrackingSchema.IC_TRACKING_V1:
        problems.append(f"schema {descriptor.schema!r} != IC_TRACKING_V1")
    if descriptor.fields != p.ic_tracking_v1_layout().fields:
        problems.append("fields do not match the fixed ic_tracking_v1 layout this decoder assumes")
    if descriptor.payload_size != p.IC_TRACKING_V1_PAYLOAD_SIZE:
        problems.append(f"payload_size {descriptor.payload_size} != {p.IC_TRACKING_V1_PAYLOAD_SIZE}")
    if descriptor.quaternion_order != p.QuaternionOrder.XYZW:
        problems.append(f"quaternion_order {descriptor.quaternion_order!r} != XYZW")
    if descriptor.hand_joint_order != p.HandJointOrder.OPENXR_26:
        problems.append(f"hand_joint_order {descriptor.hand_joint_order!r} != OPENXR_26")
    if descriptor.position_unit != p.PositionUnit.METERS:
        problems.append(f"position_unit {descriptor.position_unit!r} != METERS")
    if descriptor.pose_representation != p.PoseRepresentation.POSITION_QUATERNION:
        problems.append(f"pose_representation {descriptor.pose_representation!r} != POSITION_QUATERNION")
    if descriptor.axes != p.Axes.RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD:
        problems.append(f"axes {descriptor.axes!r} != RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD")
    if problems:
        raise RecordingAdapterError(
            "tracking descriptor incompatible with recording conversion: " + "; ".join(problems)
        )


# --- pose/button/trigger conversion -----------------------------------------


def pose7d_to_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Builds a float64 4x4 homogeneous transform from a position (3,) and
    an XYZW quaternion (4,) -- rotation in the top-left 3x3 block,
    translation as the last column. Caller must
    only call this for an already-validated (valid-bit-gated) pose --
    this never checks validity itself."""
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    t[:3, 3] = np.asarray(position, dtype=np.float64)
    return t


def convert_head_pose(tracking: dict[str, np.ndarray]) -> np.ndarray | None:
    """4x4 head pose, or ``None`` if ``head.is_valid`` is false this tick --
    the explicit validity bit is checked before reading position/orientation,
    never inferred from the resulting pose."""
    if not bool(tracking["head.is_valid"]):
        return None
    return pose7d_to_matrix(tracking["head.position"], tracking["head.orientation"])


def convert_controller_aim_pose(tracking: dict[str, np.ndarray], side: Side) -> np.ndarray | None:
    """4x4 controller AIM pose, or ``None`` if ``{side}.aim_is_valid`` is
    false this tick -- always AIM, never GRIP; the validity bit is checked
    first."""
    prefix = f"controller_{side}"
    if not bool(tracking[f"{prefix}.aim_is_valid"]):
        return None
    return pose7d_to_matrix(tracking[f"{prefix}.aim_position"], tracking[f"{prefix}.aim_orientation"])


def button_pressed(value: float) -> bool:
    """A strict ``> 0.5`` threshold on the wire's raw float click value."""
    return float(value) > 0.5


def build_pico_controller_fields(tracking: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    """One ``pico_controller`` row, or ``None`` if head or either
    controller is invalid this tick -- whole-source gating on the combined
    ``head_valid and left_valid and right_valid``; an omitted
    tick is a real, countable gap, never a fabricated/backfilled row."""
    head_pose = convert_head_pose(tracking)
    left_pose = convert_controller_aim_pose(tracking, "left")
    right_pose = convert_controller_aim_pose(tracking, "right")
    if head_pose is None or left_pose is None or right_pose is None:
        return None
    return {
        "head_pose": head_pose,
        "left_pose": left_pose,
        "right_pose": right_pose,
        "left_trigger": np.asarray(tracking["controller_left.trigger_value"], dtype=np.float64),
        "right_trigger": np.asarray(tracking["controller_right.trigger_value"], dtype=np.float64),
        "a_button": np.asarray(button_pressed(tracking["controller_right.primary_click"]), dtype=np.bool_),
        "b_button": np.asarray(button_pressed(tracking["controller_right.secondary_click"]), dtype=np.bool_),
        "x_button": np.asarray(button_pressed(tracking["controller_left.primary_click"]), dtype=np.bool_),
        "y_button": np.asarray(button_pressed(tracking["controller_left.secondary_click"]), dtype=np.bool_),
    }


# --- hand/glove conversion ---------------------------------------------------

#: The thumb-tip's slot in the 25-entry fingertip array -- the one joint
#: that gets its own extra Z rotation below.
_THUMB_TIP_SLOT = 4

#: OXR_TO_CASADI_R = Rz(90deg) . Ry(180deg): the fixed rotation from OpenXR's
#: hand-joint frame convention to the skeleton frame this module's outputs use.
_OXR_TO_CASADI_R = (
    Rotation.from_rotvec(np.pi / 2.0 * np.array([0.0, 0.0, 1.0]))
    * Rotation.from_rotvec(np.pi * np.array([0.0, 1.0, 0.0]))
).as_matrix()
_OXR_TO_CASADI_R_INV = np.linalg.inv(_OXR_TO_CASADI_R)

#: The neutral-tilt correction this conversion applies, in degrees, about the local
#: X axis after alignment -- a plain float constant, hardcoded rather than
#: imported per this module's no-native-runtime-import rule. Update this value by
#: hand if the correction this conversion needs ever changes.
_MANUS_NEUTRAL_TILT_CORRECTION_DEG = 0.0


def _joint_matrices_26(positions: np.ndarray, orientations: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(26,3) position + (26,4) XYZW quaternion + (26,) validity -> (26,4,4)
    float64 homogeneous transforms. Only VALID joints are ever passed
    through ``Rotation.from_quat`` -- an invalid joint's quaternion is
    unconstrained wire data (may be zero/non-normalized) and must never be
    converted at all, let alone raise. Invalid slots are filled with
    identity as a geometric placeholder only; validity itself is tracked
    separately by the caller and must never be re-derived from this
    array's content."""
    valid = np.asarray(valid, dtype=bool)
    out = np.tile(np.eye(4, dtype=np.float64), (26, 1, 1))
    if valid.any():
        quats = np.asarray(orientations, dtype=np.float64)[valid]
        out[valid, :3, :3] = Rotation.from_quat(quats).as_matrix()
        out[valid, :3, 3] = np.asarray(positions, dtype=np.float64)[valid]
    return out


def oxr_fingertips_to_manus_skeleton(
    matrices_25: np.ndarray, valid_25: np.ndarray, hand: Side
) -> np.ndarray | None:
    """Wrist-relative Manus-skeleton conversion of a (25,4,4) fingertip
    array. ``valid_25`` is the
    explicit, already-known per-joint validity mask -- never inferred from
    whether a transform happens to look like zero/identity (a REAL,
    explicitly-valid identity or zero pose must be preserved as data, and
    an explicitly INVALID non-identity pose must never be treated as
    real). Returns ``None`` only when the wrist itself (``valid_25[0]``)
    is invalid -- without a real wrist there is no root frame to compute
    anything relative to; every other invalid joint is zeroed in the
    output rather than fabricated or held over from a previous tick."""
    pose = np.asarray(matrices_25, dtype=np.float64)
    valid = np.asarray(valid_25, dtype=bool)
    if pose.shape != (25, 4, 4) or valid.shape != (25,):
        return None
    if not np.all(np.isfinite(pose)):
        return None
    if not valid[0]:
        return None

    root_pos = pose[0, :3, 3].copy()
    root_rot = pose[0, :3, :3].copy()
    align = _OXR_TO_CASADI_R

    relative_pos = (pose[:, :3, 3] - root_pos) @ root_rot
    relative_rot = root_rot.T @ pose[:, :3, :3]
    rotated_pos = relative_pos @ align.T
    rotated_rot = align @ relative_rot @ _OXR_TO_CASADI_R_INV

    rot_angle = 45.0 if hand == "left" else -45.0
    z_rot_mat = Rotation.from_rotvec(rot_angle / 180.0 * np.pi * np.array([0, 0, 1])).as_matrix()
    rotated_rot[_THUMB_TIP_SLOT] = rotated_rot[_THUMB_TIP_SLOT] @ z_rot_mat

    out = np.zeros((25, 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    out[:, :3, :3] = rotated_rot
    out[:, :3, 3] = rotated_pos
    # Explicit invalid rows are zeroed using the REAL validity mask, never
    # a post-hoc "this looks like identity/zero" guess.
    out[~valid] = 0.0

    correction_sign = 1.0 if hand == "left" else -1.0
    t_correction = np.eye(4)
    t_correction[:3, :3] = Rotation.from_rotvec(
        correction_sign * np.radians(_MANUS_NEUTRAL_TILT_CORRECTION_DEG) * np.array([1, 0, 0])
    ).as_matrix()
    return np.matmul(t_correction[None], out)


def convert_hand_pose(tracking: dict[str, np.ndarray], side: Side) -> np.ndarray | None:
    """One hand's (25,4,4) Manus-skeleton pose from this tick's native
    ``hand_{side}.*`` fields, or ``None`` if the wrist is untracked this
    tick. Native OXR joints 1-25 (WRIST + 24 finger joints) map to
    fingertip slots 0-24 by a plain contiguous offset (every key maps to
    ``key - 1`` with no reordering) -- joint 0 (PALM) is dropped by a
    direct slice, not a permutation table."""
    prefix = f"hand_{side}"
    valid_26 = np.asarray(tracking[f"{prefix}.joint_valid"], dtype=bool)
    matrices_26 = _joint_matrices_26(
        tracking[f"{prefix}.joint_positions"], tracking[f"{prefix}.joint_orientations"], valid_26
    )
    return oxr_fingertips_to_manus_skeleton(matrices_26[1:], valid_26[1:], side)


def build_manus_glove_fields(tracking: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    """One ``manus_glove`` row, or ``None`` if either hand's wrist is
    invalid this tick. Whole-row validity policy, matching
    ``pico_controller``'s combined gating above: a tick with one
    untracked hand is omitted ENTIRELY for this source, counted as a real
    gap, never backfilled with a stale or fabricated pose for the missing
    hand (there is no per-hand stale-hold state anywhere in this
    module)."""
    left = convert_hand_pose(tracking, "left")
    right = convert_hand_pose(tracking, "right")
    if left is None or right is None:
        return None
    return {"left_hand_pose": left, "right_hand_pose": right}


# --- camera conversion --------------------------------------------------------


def convert_camera_image(
    image: CameraImage, source_name: str, intrinsics: np.ndarray | None = None
) -> VideoSample:
    """Decodes ``image`` (already-validated RGB per ``CameraImage.decode()``)
    into a ``VideoSample`` -- no local color conversion; the service
    converts RGB8 to BGR at write time (``recording_service.py:686-688``)."""
    return VideoSample(
        source=source_name,
        image=image.decode(),
        pixel_format=PixelFormat.RGB8,
        capture_time=image.capture_time_ns / 1e9,
        receive_time=image.receive_time_ns / 1e9,
        intrinsics=intrinsics,
    )


def convert_capture_frame(frame: CaptureFrame) -> list[NumericSample]:
    """Decodes one raw tracking payload and converts it into zero, one or
    two ``NumericSample``s (``pico_controller``/``manus_glove``), each
    omitted entirely (not fabricated) when its own whole-row validity
    gate fails this tick. Caller is expected to have already validated
    the descriptor once per attempt (``validate_tracking_descriptor``) --
    this does not re-validate it per frame."""
    tracking = capture_protocol.decode_tracking_frame(frame.payload)
    capture_time = frame.capture_time_ns / 1e9
    receive_time = frame.receive_time_ns / 1e9
    samples: list[NumericSample] = []
    pico_fields = build_pico_controller_fields(tracking)
    if pico_fields is not None:
        samples.append(NumericSample(PICO_CONTROLLER_SOURCE, capture_time, receive_time, pico_fields))
    manus_fields = build_manus_glove_fields(tracking)
    if manus_fields is not None:
        samples.append(NumericSample(MANUS_GLOVE_SOURCE, capture_time, receive_time, manus_fields))
    return samples


# --- bounded ingress queue/consumer -------------------------------------------

RawItem = CaptureFrame | CameraImage


def _raw_item_bytes(raw_item: RawItem) -> int:
    """The memory one queued raw item pins: the compressed JPEG or the raw tracking blob."""
    return len(raw_item.jpeg) if isinstance(raw_item, CameraImage) else len(raw_item.payload)


def _convert_raw_item(raw_item: RawItem) -> list[NumericSample | VideoSample]:
    """The default, real conversion dispatch -- CPU-bound (JPEG decode,
    pose math), so ``RecordingIngress`` always runs this inside its
    bounded consumer (off the synchronous capture-callback path that
    calls ``offer()``), via a worker executor, never before the queue."""
    if isinstance(raw_item, CaptureFrame):
        return list(convert_capture_frame(raw_item))
    if isinstance(raw_item, CameraImage):
        return [convert_camera_image(raw_item, raw_item.name)]
    raise TypeError(f"unsupported raw item type: {type(raw_item).__name__}")


class RecordingIngress:
    """One bounded queue plus one owned consumer task, scoped to exactly
    ONE take: ``offer()`` queues a raw capture callback value (a
    ``CaptureFrame``/``CameraImage``) without blocking or doing any
    conversion itself; the consumer decodes/converts off the event loop
    thread (via an executor) and submits the result to the real service,
    retrying a FULL result within a bounded deadline (the same sample
    object every retry, never dropped/replaced silently). Any real
    failure anywhere in this pipeline (queue overflow, a FULL-retry
    timeout, a conversion/submission exception) is retained as this
    ingress's own owned ``failure`` fact, inspectable via the ``failure``
    property once ``stop()`` reports settlement -- there is no optional/
    best-effort failure channel.

    Scoped to one take deliberately: binding ``submit`` to one specific
    episode id at construction time (e.g.
    ``functools.partial(service.submit_sample, episode_id)``) is what
    pins every sample this ingress ever handles to the correct episode --
    a NEW take gets a NEW ``RecordingIngress`` instance, never a reused
    one, so delayed conversion work from an old take can never reach a
    replacement episode's gate."""

    def __init__(
        self,
        *,
        submit: Callable[[NumericSample | VideoSample], object],
        queue_capacity: int,
        queue_bytes: int | None = None,
        convert: Callable[[RawItem], list[NumericSample | VideoSample]] = _convert_raw_item,
        full_retry_deadline_seconds: float = 2.0,
        full_retry_backoff_seconds: float = 0.01,
        in_flight_timeout_seconds: float = 30.0,
        on_failure: Callable[[str], None] | None = None,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError(f"queue_capacity must be positive, got {queue_capacity}")
        if queue_bytes is not None and queue_bytes <= 0:
            raise ValueError(f"queue_bytes must be positive, got {queue_bytes}")
        self._queue: asyncio.Queue[tuple[RawItem, int]] = asyncio.Queue(maxsize=queue_capacity)
        # Optional second bound, on the bytes the queued items pin. Camera frames vary by orders of magnitude
        # from tracking blobs, so a count alone either wastes memory or overflows on a short stall.
        self._queue_bytes = queue_bytes
        self._queued_bytes = 0
        self._submit = submit
        self._convert = convert
        self._full_retry_deadline_seconds = full_retry_deadline_seconds
        self._full_retry_backoff_seconds = full_retry_backoff_seconds
        # How long stop() waits, on top of its own queue-join deadline, for
        # one specific genuinely-stuck in-flight executor conversion to
        # finish before reporting itself as still unsettled (see stop()'s
        # own docstring) -- configurable so tests can exercise the
        # still-stuck path without a real 30s wait.
        self._in_flight_timeout_seconds = in_flight_timeout_seconds
        # Called synchronously, on the event loop thread, the first time
        # this ingress actually fails -- so a caller can settle a take
        # immediately on a mid-recording failure instead of waiting
        # indefinitely for an operator to eventually press Stop. Must
        # never block or raise.
        self._on_failure = on_failure
        self._active = True
        self._failure: str | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._stopped = False

    @property
    def failure(self) -> str | None:
        """The first real failure this ingress ever hit, or ``None``."""
        return self._failure

    def _fail(self, message: str) -> None:
        # First diagnostic wins -- a cascade of follow-on errors after the
        # real cause must never overwrite it.
        first = self._failure is None
        if first:
            self._failure = message
        # A real failure closes admission immediately: continuing to
        # accept samples into a take already known to be broken would
        # just grow a queue nothing can make trustworthy again.
        self._active = False
        if first and self._on_failure is not None:
            self._on_failure(message)

    def offer(self, raw_item: RawItem) -> bool:
        """Non-blocking; called directly from a synchronous capture
        callback -- queues the RAW item only, no conversion here. Returns
        whether it was actually queued."""
        if not self._active:
            return False
        size = _raw_item_bytes(raw_item) if self._queue_bytes is not None else 0
        if self._queue_bytes is not None and self._queued_bytes + size > self._queue_bytes:
            self._fail("ingress queue overflow (byte budget)")
            return False
        try:
            self._queue.put_nowait((raw_item, size))
        except asyncio.QueueFull:
            self._fail("ingress queue overflow")
            return False
        self._queued_bytes += size
        return True

    async def start(self) -> None:
        if self._consumer_task is None:
            self._consumer_task = asyncio.create_task(self._run())

    def close(self) -> None:
        """Stops admitting further items -- idempotent. Already-queued/
        in-flight items are unaffected; see ``stop()``."""
        self._active = False

    async def stop(self) -> bool:
        """The ONE retirement call -- closes admission, then waits once,
        under a single overall budget, for the whole queue to genuinely
        finish, then cancels/joins the now-idle consumer task. Returns
        ``True`` only once this ingress is ACTUALLY confirmed settled --
        ``False`` if real work is still unresolved when the budget
        elapses, in which case the consumer task and queue are left
        exactly as they are (nothing cancelled/cleared) so a later
        ``stop()`` call can retry/rejoin the same still-unresolved work.
        Never marks itself settled on a mere timeout."""
        self.close()
        if self._stopped:
            return True
        if self._consumer_task is None:
            self._stopped = True
            return True
        budget = self._full_retry_deadline_seconds + self._in_flight_timeout_seconds
        try:
            await asyncio.wait_for(self._queue.join(), timeout=budget)
        except TimeoutError:
            self._fail(
                f"ingress did not settle before the stop deadline ({self._queue.qsize()} item(s) still queued)"
            )
            return False
        self._consumer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._consumer_task
        self._consumer_task = None
        self._stopped = True
        return True

    async def _run(self) -> None:
        while True:
            raw_item, size = await self._queue.get()
            self._queued_bytes -= size
            try:
                loop = asyncio.get_event_loop()
                samples = await loop.run_in_executor(None, self._convert, raw_item)
                for sample in samples:
                    await self._submit_with_full_retry(sample)
            except Exception as exc:  # noqa: BLE001 -- must never kill this loop; see _fail's own doc
                self._fail(str(exc))
            finally:
                self._queue.task_done()

    async def _submit_with_full_retry(self, sample: NumericSample | VideoSample) -> None:
        """Retries the exact same sample on FULL (never dropping/replacing
        it) until either it's accepted or a bounded deadline elapses, at
        which point this ingress records itself as failed -- a dropped
        sample is never silent, see ``_fail``."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._full_retry_deadline_seconds
        while True:
            result = self._submit(sample)
            if result.status is not SubmissionStatus.FULL:
                return
            if loop.time() >= deadline:
                self._fail(f"{sample.source}: FULL-retry deadline exceeded, sample dropped")
                return
            await asyncio.sleep(self._full_retry_backoff_seconds)
