# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for recording_adapter.py: the pure native-tracking-callback ->
RecordingService-sample conversion math (head/controller/hand/button), and
the bounded ingress queue/consumer -- see the module's own docstring for
the pinned data contract each test is checking against.

Explicit-validity regression focus: every conversion path must trace
validity back to a real wire bit, never infer it from pose content (an
identity/zero transform that is actually marked VALID is real data; an
INVALID joint's quaternion must never even be converted, let alone
fabricate a usable pose).
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from capture_server import protocol as capture_protocol
from capture_server.client import CaptureFrame
from egocap_web.recording_adapter import (
    RecordingAdapterError,
    RecordingIngress,
    build_manus_glove_fields,
    build_pico_controller_fields,
    button_pressed,
    convert_camera_image,
    convert_controller_aim_pose,
    convert_hand_pose,
    convert_head_pose,
    oxr_fingertips_to_manus_skeleton,
    pose7d_to_matrix,
    validate_tracking_descriptor,
)
from recording_service import NumericSample, SubmissionResult, SubmissionStatus, VideoSample

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _controller_defaults(prefix: str) -> dict[str, object]:
    return {
        f"{prefix}.present": 1,
        f"{prefix}.grip_position": (0.0, 0.0, 0.0),
        f"{prefix}.grip_orientation": IDENTITY_QUAT,
        f"{prefix}.grip_is_valid": 1,
        f"{prefix}.aim_position": (0.0, 0.0, 0.0),
        f"{prefix}.aim_orientation": IDENTITY_QUAT,
        f"{prefix}.aim_is_valid": 1,
        f"{prefix}.primary_click": 0.0,
        f"{prefix}.secondary_click": 0.0,
        f"{prefix}.thumbstick_x": 0.0,
        f"{prefix}.thumbstick_y": 0.0,
        f"{prefix}.thumbstick_click": 0.0,
        f"{prefix}.menu_click": 0.0,
        f"{prefix}.squeeze_value": 0.0,
        f"{prefix}.trigger_value": 0.0,
    }


def _hand_defaults(prefix: str) -> dict[str, object]:
    # Distinct, nonzero per-joint positions -- real recorded data never
    # happens to land exactly on identity, and this module must not treat
    # identity as a special "missing" sentinel at all (validity is the
    # ONLY thing that decides that).
    positions = (np.arange(26 * 3, dtype=np.float32).reshape(26, 3) + 1.0) * 0.01
    return {
        f"{prefix}.present": 1,
        f"{prefix}.joint_positions": positions,
        f"{prefix}.joint_orientations": np.tile(np.array(IDENTITY_QUAT, dtype=np.float32), (26, 1)),
        f"{prefix}.joint_radii": np.zeros(26, dtype=np.float32),
        f"{prefix}.joint_valid": np.ones(26, dtype=np.uint8),
    }


def default_tracking_values() -> dict[str, object]:
    """One full, structurally-valid ic_tracking_v1 value dict -- every
    field present with everything tracked/valid, so a test only needs to
    override the handful of fields it cares about."""
    values: dict[str, object] = {
        "head.present": 1,
        "head.position": (0.0, 0.0, 0.0),
        "head.orientation": IDENTITY_QUAT,
        "head.is_valid": 1,
        "head.is_tracked": 1,
    }
    values.update(_controller_defaults("controller_left"))
    values.update(_controller_defaults("controller_right"))
    values.update(_hand_defaults("hand_left"))
    values.update(_hand_defaults("hand_right"))
    return values


def tracking_dict(**overrides: object) -> dict[str, np.ndarray]:
    """Encodes+decodes through the real wire codec (not hand-assembled) so
    these tests exercise the exact same field dtypes/shapes the real
    decode path produces."""
    values = default_tracking_values()
    values.update(overrides)
    payload = capture_protocol.encode_tracking_frame(values)
    return capture_protocol.decode_tracking_frame(payload)


# --- pose7d_to_matrix / head / controller -----------------------------------


def test_pose7d_to_matrix_identity():
    m = pose7d_to_matrix((1.0, 2.0, 3.0), IDENTITY_QUAT)
    assert np.allclose(m[:3, :3], np.eye(3))
    assert np.allclose(m[:3, 3], [1.0, 2.0, 3.0])
    assert m[3, 3] == 1.0


def test_head_pose_valid():
    t = tracking_dict(**{"head.position": (1.0, 0.0, 0.0), "head.is_valid": 1})
    pose = convert_head_pose(t)
    assert pose is not None
    assert np.allclose(pose[:3, 3], [1.0, 0.0, 0.0], atol=1e-6)


def test_head_pose_invalid_is_none_not_identity():
    t = tracking_dict(**{"head.is_valid": 0})
    assert convert_head_pose(t) is None


def test_controller_uses_aim_not_grip():
    t = tracking_dict(
        **{
            "controller_left.aim_position": (5.0, 0.0, 0.0),
            "controller_left.grip_position": (99.0, 0.0, 0.0),
        }
    )
    pose = convert_controller_aim_pose(t, "left")
    assert pose is not None
    assert np.allclose(pose[:3, 3], [5.0, 0.0, 0.0], atol=1e-6)


def test_controller_invalid_aim_is_none():
    t = tracking_dict(**{"controller_right.aim_is_valid": 0})
    assert convert_controller_aim_pose(t, "right") is None


# --- buttons/triggers --------------------------------------------------------


@pytest.mark.parametrize("value,expected", [(0.0, False), (0.5, False), (0.50001, True), (1.0, True)])
def test_button_pressed_strict_threshold(value, expected):
    assert button_pressed(value) is expected


def test_pico_controller_fields_button_mapping_and_triggers():
    t = tracking_dict(
        **{
            "controller_right.primary_click": 1.0,  # -> A
            "controller_right.secondary_click": 0.0,  # -> B
            "controller_left.primary_click": 1.0,  # -> X
            "controller_left.secondary_click": 0.0,  # -> Y
            "controller_left.trigger_value": 0.25,
            "controller_right.trigger_value": 0.75,
        }
    )
    fields = build_pico_controller_fields(t)
    assert fields is not None
    assert bool(fields["a_button"]) is True
    assert bool(fields["b_button"]) is False
    assert bool(fields["x_button"]) is True
    assert bool(fields["y_button"]) is False
    assert float(fields["left_trigger"]) == pytest.approx(0.25)
    assert float(fields["right_trigger"]) == pytest.approx(0.75)
    assert fields["head_pose"].dtype == np.float64
    assert fields["left_pose"].shape == (4, 4)


def test_pico_controller_fields_none_when_any_of_head_or_controllers_invalid():
    assert build_pico_controller_fields(tracking_dict(**{"head.is_valid": 0})) is None
    assert build_pico_controller_fields(tracking_dict(**{"controller_left.aim_is_valid": 0})) is None
    assert build_pico_controller_fields(tracking_dict(**{"controller_right.aim_is_valid": 0})) is None


# --- hand/glove conversion: explicit validity, never inferred ----------------


def test_convert_hand_pose_all_valid_produces_25x4x4_float64():
    t = tracking_dict()  # every joint valid
    pose = convert_hand_pose(t, "left")
    assert pose is not None
    assert pose.shape == (25, 4, 4)
    assert pose.dtype == np.float64


def test_valid_identity_wrist_is_not_treated_as_missing():
    # A wrist that is EXPLICITLY valid, even though its pose happens to be
    # exactly identity, must be preserved as real data -- never
    # reclassified as "missing" just because it looks like a sentinel.
    valid = np.ones(26, dtype=np.uint8)
    positions = np.zeros((26, 3), dtype=np.float32)  # identity positions
    t = tracking_dict(
        **{
            "hand_left.joint_positions": positions,
            "hand_left.joint_orientations": np.tile(np.array(IDENTITY_QUAT, dtype=np.float32), (26, 1)),
            "hand_left.joint_valid": valid,
        }
    )
    pose = convert_hand_pose(t, "left")
    assert pose is not None
    # The wrist row itself is real (non-zeroed) data, not blanked out.
    assert not np.allclose(pose[0], 0.0)


def test_invalid_wrist_is_none_not_fabricated_or_preserved():
    # An explicitly INVALID wrist must never be filled with identity and
    # treated as a usable root -- regression for the bug where an invalid
    # wrist got identity-filled then its "missing" flag was cleared,
    # fabricating a root transform from invalid data.
    valid = np.ones(26, dtype=np.uint8)
    valid[1] = 0  # native WRIST index
    t = tracking_dict(**{"hand_left.joint_valid": valid})
    assert convert_hand_pose(t, "left") is None


def test_invalid_joint_quaternion_never_raises_even_if_zero():
    # An invalid joint's quaternion is unconstrained wire data and must
    # never even be passed through Rotation.from_quat, let alone raise.
    valid = np.ones(26, dtype=np.uint8)
    valid[6] = 0  # one INDEX joint
    orientations = np.tile(np.array(IDENTITY_QUAT, dtype=np.float32), (26, 1))
    orientations[6] = (0.0, 0.0, 0.0, 0.0)  # a genuinely invalid, non-normalized quaternion
    t = tracking_dict(**{"hand_left.joint_valid": valid, "hand_left.joint_orientations": orientations})
    pose = convert_hand_pose(t, "left")  # must not raise
    assert pose is not None
    # Fingertip slot 5 (native oxr index 6 -> slot 6-1=5) is zeroed, not
    # fabricated from the garbage quaternion.
    assert np.allclose(pose[5], 0.0)


def test_thumb_tip_rotation_sign_flips_by_hand():
    matrices_25 = np.tile(np.eye(4, dtype=np.float64), (25, 1, 1))
    matrices_25[:, :3, 3] = (np.arange(25 * 3, dtype=np.float64).reshape(25, 3) + 1.0) * 0.01
    valid_25 = np.ones(25, dtype=bool)
    left = oxr_fingertips_to_manus_skeleton(matrices_25, valid_25, "left")
    right = oxr_fingertips_to_manus_skeleton(matrices_25, valid_25, "right")
    assert left is not None and right is not None
    assert not np.allclose(left[4, :3, :3], right[4, :3, :3])
    # A non-thumb-tip joint is untouched by the twist, so left/right agree
    # there (neutral-tilt correction is 0deg today).
    assert np.allclose(left[5, :3, :3], right[5, :3, :3])


def test_build_manus_glove_fields_whole_row_omitted_when_one_hand_untracked():
    # Whole-row validity policy: if either hand's wrist is invalid this
    # tick, the ENTIRE manus_glove row is omitted -- never a stale-held or
    # fabricated pose for the missing hand.
    valid = np.ones(26, dtype=np.uint8)
    valid[1] = 0  # right wrist untracked
    t = tracking_dict(**{"hand_right.joint_valid": valid})
    assert build_manus_glove_fields(t) is None


def test_build_manus_glove_fields_present_when_both_hands_valid():
    t = tracking_dict()
    fields = build_manus_glove_fields(t)
    assert fields is not None
    assert fields["left_hand_pose"].shape == (25, 4, 4)
    assert fields["right_hand_pose"].shape == (25, 4, 4)


# --- descriptor validation ----------------------------------------------------


def _descriptor(**overrides) -> capture_protocol.TrackingStream:
    p = capture_protocol
    kwargs = dict(
        stream_id=0,
        kind=p.StreamKind.TRACKING,
        name="ic_tracking_v1",
        schema=p.TrackingSchema.IC_TRACKING_V1,
        sources=(),
        rate_hz=90.0,
        pose_representation=p.PoseRepresentation.POSITION_QUATERNION,
        quaternion_order=p.QuaternionOrder.XYZW,
        position_unit=p.PositionUnit.METERS,
        axes=p.Axes.RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD,
        reference_space="",
        hand_joint_order=p.HandJointOrder.OPENXR_26,
        payload_size=p.IC_TRACKING_V1_PAYLOAD_SIZE,
        # The real wire type (FieldDescriptor, via ic_tracking_v1_layout()),
        # not the internal Field type (IC_TRACKING_V1_FIELDS) -- a real
        # TrackingStream, reconstructed off the network, never carries the
        # latter; see validate_tracking_descriptor's own docstring.
        fields=p.ic_tracking_v1_layout().fields,
    )
    kwargs.update(overrides)
    return p.TrackingStream(**kwargs)


def test_validate_tracking_descriptor_accepts_the_real_contract():
    validate_tracking_descriptor(_descriptor())  # must not raise


def test_validate_tracking_descriptor_rejects_wrong_payload_size():
    with pytest.raises(RecordingAdapterError):
        validate_tracking_descriptor(_descriptor(payload_size=1))


def test_validate_tracking_descriptor_rejects_mismatched_fields():
    bad_fields = tuple(capture_protocol.ic_tracking_v1_layout().fields[:-1])  # drop the last field
    with pytest.raises(RecordingAdapterError):
        validate_tracking_descriptor(_descriptor(fields=bad_fields))


# --- camera conversion ---------------------------------------------------------


class _FakeCameraImage:
    def __init__(self, rgb: np.ndarray, capture_time_ns: int, receive_time_ns: int, name: str = "cam"):
        self._rgb = rgb
        self.capture_time_ns = capture_time_ns
        self.receive_time_ns = receive_time_ns
        self.name = name

    def decode(self) -> np.ndarray:
        return self._rgb


def test_convert_camera_image_is_rgb8_with_seconds_timestamps():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    image = _FakeCameraImage(rgb, capture_time_ns=2_000_000_000, receive_time_ns=3_000_000_000)
    sample = convert_camera_image(image, "head_human_camera")  # type: ignore[arg-type]
    assert isinstance(sample, VideoSample)
    assert sample.pixel_format.value == "rgb8"
    assert sample.capture_time == pytest.approx(2.0)
    assert sample.receive_time == pytest.approx(3.0)
    assert sample.image is rgb


# --- bounded ingress -----------------------------------------------------------


def _numeric_sample(n: int = 0) -> NumericSample:
    return NumericSample("pico_controller", float(n), float(n), {})


def _identity_convert(raw_item):
    return [raw_item]


async def _started_ingress(
    submit,
    *,
    queue_capacity=4,
    full_retry_deadline_seconds=0.2,
    in_flight_timeout_seconds=0.2,
    convert=_identity_convert,
):
    ingress = RecordingIngress(
        queue_capacity=queue_capacity,
        submit=submit,
        convert=convert,
        full_retry_deadline_seconds=full_retry_deadline_seconds,
        in_flight_timeout_seconds=in_flight_timeout_seconds,
    )
    await ingress.start()
    return ingress


def test_queue_capacity_must_be_positive():
    with pytest.raises(ValueError):
        RecordingIngress(queue_capacity=0, submit=lambda s: None)


def test_offer_rejected_after_close():
    async def run():
        ingress = await _started_ingress(lambda s: SubmissionResult(SubmissionStatus.ACCEPTED, s.source, 0))
        ingress.close()
        assert ingress.offer(_numeric_sample()) is False
        await ingress.stop()

    asyncio.run(run())


def test_offer_accepted_and_submitted_once_active():
    submitted = []

    def submit(sample):
        submitted.append(sample)
        return SubmissionResult(SubmissionStatus.ACCEPTED, sample.source, 0)

    async def run():
        ingress = await _started_ingress(submit)
        assert ingress.offer(_numeric_sample()) is True
        settled = await ingress.stop()
        assert settled is True
        assert len(submitted) == 1
        assert ingress.failure is None

    asyncio.run(run())


def test_full_result_is_retried_with_the_same_sample_until_accepted():
    attempts = []

    def submit(sample):
        attempts.append(sample)
        if len(attempts) < 3:
            return SubmissionResult(SubmissionStatus.FULL, sample.source, None)
        return SubmissionResult(SubmissionStatus.ACCEPTED, sample.source, 0)

    async def run():
        ingress = await _started_ingress(submit, full_retry_deadline_seconds=5.0)
        target = _numeric_sample()
        ingress.offer(target)
        settled = await ingress.stop()
        assert settled is True
        assert len(attempts) == 3
        assert all(a is target for a in attempts)  # the exact same sample object every retry, never replaced

    asyncio.run(run())


def test_full_retry_deadline_elapses_is_a_real_recorded_failure_but_still_settles_cleanly():
    # A FULL-retry exhaustion is a fast, definitive outcome recorded via
    # _fail() -- the item still completes (task_done() fires), so the
    # queue genuinely drains and stop() settles normally; it's the
    # ingress's own `.failure`, not an unsettled/hung consumer, that
    # reports this was not a clean success.
    def always_full(sample):
        return SubmissionResult(SubmissionStatus.FULL, sample.source, None)

    async def run():
        ingress = await _started_ingress(always_full, full_retry_deadline_seconds=0.05)
        ingress.offer(_numeric_sample())
        settled = await ingress.stop()
        assert settled is True
        assert ingress.failure is not None

    asyncio.run(run())


def test_submit_exception_does_not_kill_the_consumer_and_stop_reports_it():
    def explode(sample):
        raise RuntimeError("boom")

    async def run():
        ingress = await _started_ingress(explode)
        ingress.offer(_numeric_sample(1))
        # The consumer loop stays alive (task_done() was still called for
        # the failed item) -- a second offer after a failure is rejected
        # because the ingress closed admission on first failure, not
        # because the loop died.
        settled = await ingress.stop()
        assert settled is True  # the queue itself did drain -- just with a recorded failure
        assert ingress.failure is not None
        assert ingress.offer(_numeric_sample(2)) is False

    asyncio.run(run())


def test_queue_overflow_closes_admission_and_is_a_real_failure():
    def never_finishes(sample):
        return SubmissionResult(SubmissionStatus.FULL, sample.source, None)

    async def run():
        ingress = await _started_ingress(never_finishes, queue_capacity=1, full_retry_deadline_seconds=10.0)
        assert ingress.offer(_numeric_sample(1)) is True  # taken by the consumer's in-flight retry loop
        await asyncio.sleep(0)  # let the consumer actually pick up item 1
        assert ingress.offer(_numeric_sample(2)) is True  # fills the 1-slot queue
        assert ingress.offer(_numeric_sample(3)) is False  # queue genuinely full -- real failure
        assert ingress.failure is not None
        await ingress.stop()

    asyncio.run(run())


def test_stop_reports_unsettled_while_converter_is_genuinely_blocked_then_settles_once_it_finishes():
    # A real genuinely-blocked background conversion thread (not a
    # simulated timeout) -- stop() must report False (unsettled) while it
    # is still actually running, never silently proceed as if cancelling
    # the asyncio wrapper released it. Once the real work finishes, a
    # LATER stop() call retries the SAME still-retained ingress and
    # genuinely settles.
    release = threading.Event()
    started = threading.Event()

    def slow_convert(raw_item):
        started.set()
        release.wait(timeout=5.0)
        return [_numeric_sample()]

    def submit(sample):
        return SubmissionResult(SubmissionStatus.ACCEPTED, sample.source, 0)

    async def run():
        ingress = RecordingIngress(
            queue_capacity=4,
            submit=submit,
            convert=slow_convert,
            full_retry_deadline_seconds=0.05,
            in_flight_timeout_seconds=0.05,
        )
        await ingress.start()
        ingress.offer(_numeric_sample())
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, started.wait, 5.0)  # the real thread has genuinely begun

        settled = await ingress.stop()
        assert settled is False
        assert ingress.failure is not None  # the timeout is a real recorded failure, not silent

        release.set()  # let the real background work actually finish
        settled_again = await ingress.stop()
        assert settled_again is True

    asyncio.run(run())


def test_conversion_runs_in_consumer_not_in_offer():
    # offer() must never invoke `convert` itself -- conversion only ever
    # happens inside the bounded consumer.
    converted = []

    def convert(raw_item):
        converted.append(raw_item)
        return [_numeric_sample()]

    def submit(sample):
        return SubmissionResult(SubmissionStatus.ACCEPTED, sample.source, 0)

    async def run():
        ingress = await _started_ingress(submit, convert=convert)
        ingress.offer("raw-item")
        assert converted == []  # not yet converted synchronously by offer()
        settled = await ingress.stop()
        assert settled is True
        assert converted == ["raw-item"]

    asyncio.run(run())


def test_queue_bytes_bounds_the_queue_by_size_not_count():
    def never_finishes(sample):
        return SubmissionResult(SubmissionStatus.FULL, sample.source, None)

    async def run():
        ingress = RecordingIngress(
            queue_capacity=100,
            queue_bytes=1000,
            submit=never_finishes,
            convert=lambda raw: [_numeric_sample(1)],
            # Short, not the real default: stop() below waits out this
            # deadline once per already-accepted item whose submission
            # never resolves (never_finishes always replies FULL) -- a
            # realistic ~10s deadline would make this one test take
            # ~30s (three accepted frames) for no benefit to what it
            # actually asserts (the byte-budget overflow itself).
            full_retry_deadline_seconds=0.05,
        )
        await ingress.start()
        frame = lambda n: CaptureFrame(  # noqa: E731
            session_id="s",
            description=None,
            stream_id=0,
            sequence=n,
            capture_time_ns=0,
            receive_time_ns=0,
            payload=b"x" * 400,
        )
        assert ingress.offer(frame(1)) is True  # taken by the consumer, no longer counted as queued
        await asyncio.sleep(0)
        assert ingress.offer(frame(2)) is True
        assert ingress.offer(frame(3)) is True  # 800 queued bytes, well under the 100-item cap
        assert ingress.offer(frame(4)) is False  # 1200 > 1000: the byte budget, not the item count
        assert "byte budget" in ingress.failure
        await ingress.stop()

    asyncio.run(run())


def test_queue_bytes_must_be_positive():
    with pytest.raises(ValueError):
        RecordingIngress(queue_capacity=1, queue_bytes=0, submit=lambda s: None)
