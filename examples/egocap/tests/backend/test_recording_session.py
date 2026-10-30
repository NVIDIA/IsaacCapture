"""Tests for recording_session.py's RecordingSessionCoordinator, against
the REAL RecordingService (synthetic source profiles -- no
EGOCAP_REAL_EPISODE fixture needed, matching test_configuration.py's
own no-fixture pattern), not a fake/mock service: operation admission
(mutual exclusion, registered before the first await, survives caller
cancellation), readiness gating, set_task's phase guard, stale-target
rejection that leaves the real active take untouched, and a genuine
capture-callback -> ingress -> RecordingService -> stop -> independent
Zarr read-back round trip.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import threading
from pathlib import Path

import numpy as np
import pytest
import zarr

from capture_server import protocol as capture_protocol
from capture_server.client import CaptureFrame
from egocap_web import recording_session as coordinator_module
from egocap_web.recording_adapter import MANUS_GLOVE_SPEC, PICO_CONTROLLER_SPEC, RecordingIngress
from egocap_web.recording_session import CoordinatorBusyError, RecordingSessionCoordinator, SessionStatus
from recording_service import (
    Disposition,
    FieldSpec,
    NumericSourceSpec,
    RecordingPhase,
    ServiceConfig,
    StateError,
    SubmissionResult,
    SubmissionStatus,
)

NUMERIC = NumericSourceSpec("s", (FieldSpec("v", "float64"),))

ALWAYS_READY = lambda: (True, None)  # noqa: E731

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def make_config_factory(tmp_path: Path, *, numeric_sources=(NUMERIC,), **overrides):
    def factory() -> ServiceConfig:
        return ServiceConfig(
            output_root=tmp_path / "out", numeric_sources=numeric_sources, video_sources=(), **overrides
        )

    return factory


def make_coordinator(tmp_path: Path, *, readiness_check=ALWAYS_READY, **overrides) -> RecordingSessionCoordinator:
    return RecordingSessionCoordinator(
        make_config_factory(tmp_path, **overrides), queue_capacity=8, readiness_check=readiness_check
    )


def _tracking_descriptor() -> capture_protocol.TrackingStream:
    p = capture_protocol
    return p.TrackingStream(
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
        fields=p.ic_tracking_v1_layout().fields,
    )


def _controller_values(prefix: str) -> dict[str, object]:
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


def _hand_values(prefix: str) -> dict[str, object]:
    positions = (np.arange(26 * 3, dtype=np.float32).reshape(26, 3) + 1.0) * 0.01
    return {
        f"{prefix}.present": 1,
        f"{prefix}.joint_positions": positions,
        f"{prefix}.joint_orientations": np.tile(np.array(IDENTITY_QUAT, dtype=np.float32), (26, 1)),
        f"{prefix}.joint_radii": np.zeros(26, dtype=np.float32),
        f"{prefix}.joint_valid": np.ones(26, dtype=np.uint8),
    }


def _fake_tracking_frame(*, head_x: float = 0.0, capture_time_ns: int = 1_000_000_000) -> CaptureFrame:
    """A real CaptureFrame carrying a real, fully-valid ic_tracking_v1
    payload (encoded through the actual wire codec), so tests exercise
    recording_session's real on_capture_frame -> decode -> convert ->
    submit path end to end."""
    values: dict[str, object] = {
        "head.present": 1,
        "head.position": (head_x, 0.0, 0.0),
        "head.orientation": IDENTITY_QUAT,
        "head.is_valid": 1,
        "head.is_tracked": 1,
    }
    values.update(_controller_values("controller_left"))
    values.update(_controller_values("controller_right"))
    values.update(_hand_values("hand_left"))
    values.update(_hand_values("hand_right"))
    payload = capture_protocol.encode_tracking_frame(values)
    return CaptureFrame(
        session_id="s1",
        description=_tracking_descriptor(),
        stream_id=0,
        sequence=1,
        capture_time_ns=capture_time_ns,
        receive_time_ns=capture_time_ns,
        payload=payload,
    )


async def test_open_happy_path(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    assert coordinator.status is SessionStatus.OPEN
    snap = coordinator.snapshot()
    assert snap.service is not None
    assert snap.service.phase is RecordingPhase.IDLE
    await coordinator.close()


async def test_open_twice_is_rejected(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    with pytest.raises(CoordinatorBusyError):
        await coordinator.open()
    await coordinator.close()


async def test_set_task_commits_outside_recording_and_is_guarded_while_recording(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("fold the towel")
    assert coordinator.snapshot().context.task_text == "fold the towel"

    await coordinator.start_episode()
    with pytest.raises(CoordinatorBusyError):
        coordinator.set_task("something else")
    assert coordinator.snapshot().context.task_text == "fold the towel"
    await coordinator.close()


async def test_start_episode_activates_recording(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    await coordinator.start_episode()
    assert coordinator.snapshot().service.phase is RecordingPhase.RECORDING
    await coordinator.close()


async def test_start_episode_rejected_when_sources_not_ready(tmp_path: Path):
    coordinator = make_coordinator(tmp_path, readiness_check=lambda: (False, "camera not selected"))
    await coordinator.open()
    with pytest.raises(CoordinatorBusyError, match="camera not selected"):
        await coordinator.start_episode()
    assert coordinator.snapshot().service.phase is RecordingPhase.IDLE
    await coordinator.close()


async def test_two_concurrent_start_episode_calls_the_second_is_rejected_not_queued(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    results = await asyncio.gather(
        coordinator.start_episode(), coordinator.start_episode(), return_exceptions=True
    )
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], CoordinatorBusyError)
    assert coordinator.snapshot().service.phase is RecordingPhase.RECORDING
    await coordinator.close()


async def test_operation_survives_caller_cancellation(tmp_path: Path):
    """A caller's own coroutine being cancelled (e.g. a browser disconnect
    cancelling the WebSocket handler task) must not abandon the real
    operation -- it keeps running and the coordinator still correctly
    clears its in-flight-operation bookkeeping once it actually finishes."""
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")

    task = asyncio.create_task(coordinator.start_episode())
    await asyncio.sleep(0)  # let it begin and register the operation
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The real start_episode operation is still running/just finished in
    # its own owned task -- wait for it to actually settle.
    for _ in range(100):
        if coordinator.snapshot().active_operation is None:
            break
        await asyncio.sleep(0.01)
    assert coordinator.snapshot().active_operation is None
    assert coordinator.snapshot().service.phase is RecordingPhase.RECORDING
    await coordinator.close()


async def test_stop_episode_closes_the_gate_drains_and_returns_the_real_service_result(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    result = await coordinator.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT
    assert coordinator.snapshot().service.phase is RecordingPhase.IDLE
    await coordinator.close()


async def test_discard_episode_returns_discarded(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    result = await coordinator.discard_episode(episode.episode_id)
    assert result.disposition is Disposition.DISCARDED
    await coordinator.close()


async def test_stale_target_stop_is_rejected_and_the_real_active_take_keeps_ingesting(tmp_path: Path):
    """Regression: validating the exact active episode identity must
    happen BEFORE any gate/ingress mutation -- a stale/wrong stop target
    must have literally zero effect on the real currently-active take."""
    coordinator = make_coordinator(tmp_path, numeric_sources=(PICO_CONTROLLER_SPEC, MANUS_GLOVE_SPEC))
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()

    coordinator.on_capture_frame(_fake_tracking_frame(head_x=1.0, capture_time_ns=1_000_000_000))

    with pytest.raises(CoordinatorBusyError, match="no matching active episode"):
        await coordinator.stop_episode("not-the-real-episode-id")

    # The real active take must still be genuinely receiving submitted
    # samples after the rejected stale-target request.
    coordinator.on_capture_frame(_fake_tracking_frame(head_x=2.0, capture_time_ns=2_000_000_000))

    result = await coordinator.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT

    group = zarr.open_group(str(result.final_path), mode="r")
    head_x_values = group["pico_controller"]["head_pose"][:, 0, 3]
    assert list(head_x_values) == [1.0, 2.0]
    await coordinator.close()


async def test_real_capture_callback_to_zarr_round_trip(tmp_path: Path):
    """The real end-to-end path: a synthetic but REAL ic_tracking_v1
    payload, decoded and converted by recording_adapter, queued and
    submitted by a per-take RecordingIngress, written by the real
    RecordingService, then independently read back from the actual Zarr
    output and checked against what was actually submitted."""
    coordinator = make_coordinator(tmp_path, numeric_sources=(PICO_CONTROLLER_SPEC, MANUS_GLOVE_SPEC))
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()

    coordinator.on_capture_frame(_fake_tracking_frame(head_x=3.5, capture_time_ns=5_000_000_000))

    result = await coordinator.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT
    assert result.counts

    group = zarr.open_group(str(result.final_path), mode="r")
    pico = group["pico_controller"]
    assert pico["head_pose"].shape[0] == 1
    assert pico["head_pose"][0, 0, 3] == pytest.approx(3.5)
    assert pico["capture_time"][0] == pytest.approx(5.0)
    manus = group["manus_glove"]
    assert manus["left_hand_pose"].shape == (1, 25, 4, 4)
    await coordinator.close()


async def test_ingress_failure_during_recording_cannot_result_in_a_false_kept(tmp_path: Path):
    """The single most important regression in this round: a real
    pipeline-level failure (here, a tracking payload the real decoder
    genuinely rejects) that happens WHILE RECORDING must settle the take
    as RECOVERY_REQUIRED -- RETAINED, never a normal successful stop, and
    never silently DELETED either -- that would be discard_episode, which
    is an explicit operator action, not something a pipeline failure may
    trigger on its own. A
    subsequent operator stop against the now-settled episode must be
    rejected, never silently re-succeed as KEPT."""
    coordinator = make_coordinator(tmp_path, numeric_sources=(PICO_CONTROLLER_SPEC, MANUS_GLOVE_SPEC))
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()

    # A real CaptureFrame whose payload the actual decode_tracking_frame
    # genuinely rejects (wrong length) -- a real pre-service data-loss
    # cause, not a simulated/injected failure.
    bad_frame = dataclasses.replace(_fake_tracking_frame(), payload=b"too short to decode")
    coordinator.on_capture_frame(bad_frame)

    # The failure settles itself in the background -- wait for it, AND for
    # the settlement operation's own slot to actually free up (the
    # service's phase can flip to RECOVERY_REQUIRED slightly before
    # _run_operation's owning task fully unwinds and clears
    # active_operation; waiting on both avoids racing a stale "stopping
    # already in progress" against the very next call below).
    for _ in range(300):
        snap = coordinator.snapshot()
        if snap.service.phase is RecordingPhase.RECOVERY_REQUIRED and snap.active_operation is None:
            break
        await asyncio.sleep(0.01)
    snap = coordinator.snapshot()
    assert snap.service.phase is RecordingPhase.RECOVERY_REQUIRED
    assert snap.service.last_result is not None
    assert snap.service.last_result.disposition is Disposition.RECOVERY_REQUIRED
    # Never KEPT, but never silently deleted either -- the data is still
    # genuinely on disk, retained for recovery/discard_pending.
    assert snap.service.last_result.final_path is None
    assert snap.service.last_result.pending_path is not None
    assert snap.service.last_result.pending_path.exists()
    pending = coordinator.pending_episodes()
    assert any(p.episode_id == episode.episode_id for p in pending)

    # The episode needs recovery now -- an operator's own stop against it
    # must be rejected (a real StateError from the service itself, since
    # recover_episode/discard_pending are the only valid next actions),
    # never silently re-succeed as KEPT.
    with pytest.raises(StateError):
        await coordinator.stop_episode(episode.episode_id)
    await coordinator.close()


async def _swap_in_a_genuinely_blocked_ingress(
    coordinator: RecordingSessionCoordinator,
) -> tuple[threading.Event, threading.Event]:
    """Retires whatever ingress the coordinator is really using right now
    and replaces it with a REAL ``RecordingIngress`` whose consumer is
    genuinely blocked in a background thread -- the same real
    threading.Event technique test_recording_adapter.py's own blocked-
    converter test uses. The coordinator's own construction doesn't
    expose ``convert`` for injection, so this is wired in directly;
    everything downstream (discard/start/new_batch) still exercises the
    coordinator's real public API against this real, really-stuck ingress."""
    release = threading.Event()
    started = threading.Event()

    def slow_convert(raw_item):
        started.set()
        release.wait(timeout=5.0)
        return []

    old = coordinator._current_ingress
    if old is not None:
        await old.stop()
    blocked = RecordingIngress(
        queue_capacity=4,
        submit=lambda sample: SubmissionResult(SubmissionStatus.ACCEPTED, sample.source, 0),
        convert=slow_convert,
        full_retry_deadline_seconds=0.05,
        in_flight_timeout_seconds=0.05,
    )
    await blocked.start()
    blocked.offer("raw-item")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, started.wait, 5.0)
    coordinator._current_ingress = blocked
    return release, started


async def test_discard_refuses_while_the_old_ingress_is_still_unsettled(tmp_path: Path):
    """A still-genuinely-running consumer must never be deleted out from
    under -- Discard must refuse (not silently proceed) until the old
    ingress is ACTUALLY confirmed retired, then a retry succeeds."""
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    release, _ = await _swap_in_a_genuinely_blocked_ingress(coordinator)

    with pytest.raises(CoordinatorBusyError):
        await coordinator.discard_episode(episode.episode_id)

    release.set()  # let the real background work actually finish
    result = await coordinator.discard_episode(episode.episode_id)
    assert result.disposition is Disposition.DISCARDED
    await coordinator.close()


async def test_start_episode_refuses_while_a_previous_ingress_is_still_unsettled(tmp_path: Path):
    """A previous take's still-genuinely-running consumer must never be
    replaced by a brand new ingress -- start_episode must refuse until
    the old one is confirmed retired (a Stop/Discard attempt against it
    having already failed with CoordinatorBusyError, exactly as above)."""
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    release, _ = await _swap_in_a_genuinely_blocked_ingress(coordinator)
    with pytest.raises(CoordinatorBusyError):
        await coordinator.discard_episode(episode.episode_id)

    with pytest.raises(CoordinatorBusyError):
        await coordinator.start_episode()

    release.set()
    await coordinator.discard_episode(episode.episode_id)
    await coordinator.close()


async def test_new_batch_refuses_while_the_old_ingress_is_still_unsettled(tmp_path: Path):
    """Same guard for New batch -- storage must never be released/rotated
    out from under a consumer that's still genuinely running."""
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    release, _ = await _swap_in_a_genuinely_blocked_ingress(coordinator)

    with pytest.raises(CoordinatorBusyError):
        await coordinator.new_batch()

    release.set()
    await coordinator.discard_episode(episode.episode_id)
    await coordinator.close()


async def test_capture_lost_mid_recording_retains_the_take_never_kept(tmp_path: Path):
    """A terminal capture-loss signal arriving WHILE a take is active (the
    real path: CaptureStatusMonitor's own on_stopped, forwarded here via
    on_capture_lost) must settle the take as RECOVERY_REQUIRED -- never
    let a subsequent/normal Stop admit it as a plain successful save."""
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()

    coordinator.on_capture_lost("mock capture stopped")

    for _ in range(300):
        snap = coordinator.snapshot()
        if snap.service.phase is RecordingPhase.RECOVERY_REQUIRED and snap.active_operation is None:
            break
        await asyncio.sleep(0.01)
    snap = coordinator.snapshot()
    assert snap.service.phase is RecordingPhase.RECOVERY_REQUIRED
    assert snap.service.last_result is not None
    assert snap.service.last_result.disposition is Disposition.RECOVERY_REQUIRED
    with pytest.raises(StateError):
        await coordinator.stop_episode(episode.episode_id)
    await coordinator.close()


async def test_capture_lost_after_a_take_was_already_cleanly_stopped_is_a_noop(tmp_path: Path):
    """The cutoff boundary rule: a capture-loss signal arriving AFTER a
    take has already been deliberately stopped for a normal save must
    never retroactively invalidate it."""
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    result = await coordinator.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT

    coordinator.on_capture_lost("late/stale signal")
    await asyncio.sleep(0.05)

    snap = coordinator.snapshot()
    assert snap.service.last_result is not None
    assert snap.service.last_result.disposition is Disposition.KEPT
    await coordinator.close()


async def test_required_source_going_stale_mid_recording_is_caught_via_the_capture_tick(tmp_path: Path):
    """A required source going stale mid-take (here: the SAME injected
    readiness check start_episode itself uses, flipped to not-ready after
    activation) must be caught by the next accepted capture callback --
    no new timer/poll loop, just reusing the existing frame/image tick
    (see RecordingSessionCoordinator._check_still_ready_while_recording)."""
    ready_flag = {"ready": True}

    def readiness() -> tuple[bool, str | None]:
        return (True, None) if ready_flag["ready"] else (False, "camera stalled")

    coordinator = make_coordinator(tmp_path, readiness_check=readiness)
    await coordinator.open()
    coordinator.set_task("t")
    await coordinator.start_episode()

    ready_flag["ready"] = False
    coordinator.on_capture_frame(_fake_tracking_frame())

    for _ in range(300):
        snap = coordinator.snapshot()
        if snap.service.phase is RecordingPhase.RECOVERY_REQUIRED and snap.active_operation is None:
            break
        await asyncio.sleep(0.01)
    snap = coordinator.snapshot()
    assert snap.service.phase is RecordingPhase.RECOVERY_REQUIRED
    assert snap.service.last_result is not None
    assert snap.service.last_result.disposition is Disposition.RECOVERY_REQUIRED
    await coordinator.close()


async def test_repeated_takes_retire_their_own_consumer_tasks(tmp_path: Path):
    """Each take's per-episode RecordingIngress consumer must actually
    exit on stop -- draining alone leaves it waiting on queue.get()
    forever. Across several start/stop cycles, the live task count must
    not grow (no orphaned consumers accumulating)."""
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    await asyncio.sleep(0)  # let any just-finished setup tasks settle
    baseline = len(asyncio.all_tasks())

    for _ in range(4):
        episode = await coordinator.start_episode()
        await coordinator.stop_episode(episode.episode_id)

    await asyncio.sleep(0)
    assert len(asyncio.all_tasks()) <= baseline
    await coordinator.close()


async def test_on_capture_frame_after_stop_is_a_safe_noop_not_a_leak_into_a_future_take(tmp_path: Path):
    coordinator = make_coordinator(tmp_path, numeric_sources=(PICO_CONTROLLER_SPEC, MANUS_GLOVE_SPEC))
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    await coordinator.stop_episode(episode.episode_id)

    # No active take's ingress exists anymore -- this must not raise, and
    # must not resurrect/attach to the just-finished (or any future) take.
    coordinator.on_capture_frame(_fake_tracking_frame())

    coordinator.set_task("next")
    second = await coordinator.start_episode()
    result = await coordinator.stop_episode(second.episode_id)
    # The second (real, empty) take's own counts are untouched by the
    # dropped post-stop callback from the first take.
    assert result.disposition is Disposition.KEPT
    await coordinator.close()


async def test_new_batch_rejected_while_recording(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    await coordinator.start_episode()
    with pytest.raises(CoordinatorBusyError):
        await coordinator.new_batch()
    await coordinator.close()


async def test_new_batch_while_idle_releases_storage_without_reopening_and_preserves_task_text(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("fold the towel")
    before = coordinator.snapshot().service.session

    await coordinator.new_batch()

    # Released, not eagerly reopened -- opening storage before anything
    # has committed to recording with it would freeze a schema too early
    # (see new_batch's own docstring).
    after = coordinator.snapshot()
    assert after.service is None
    assert coordinator.status is SessionStatus.CLOSED
    assert after.context.task_text == "fold the towel"

    # The NEXT real storage open (here, via start_episode) is what
    # actually opens fresh storage, with a genuinely new session id.
    episode = await coordinator.start_episode()
    reopened = coordinator.snapshot().service.session
    assert reopened.session_id != before.session_id
    await coordinator.discard_episode(episode.episode_id)
    await coordinator.close()


async def test_new_batch_before_any_recording_only_rotates_local_facts(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    coordinator.set_task("fold the towel")
    assert coordinator.status is SessionStatus.CLOSED

    await coordinator.new_batch()

    snap = coordinator.snapshot()
    assert snap.service is None
    assert coordinator.status is SessionStatus.CLOSED
    assert snap.context.task_text == "fold the towel"
    await coordinator.close()


async def test_close_with_no_open_session_is_a_trivial_success(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    result = await coordinator.close()
    assert result.released


async def test_close_settles_and_marks_the_session_closed(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    result = await coordinator.close()
    assert result.released
    assert coordinator.status is SessionStatus.CLOSED


async def test_close_joins_an_in_flight_operation_rather_than_rejecting_it(tmp_path: Path):
    """Shutdown must be able to settle real in-progress work, not just
    fail to close while anything is running."""
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")

    start_task = asyncio.create_task(coordinator.start_episode())
    await asyncio.sleep(0)  # let start_episode begin and register

    result = await coordinator.close()
    assert result.released
    assert coordinator.status is SessionStatus.CLOSED
    with contextlib.suppress(Exception, asyncio.CancelledError):
        await start_task


async def test_a_new_batch_clears_an_interrupted_take_and_the_next_record_works(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    await coordinator.open()
    coordinator.set_task("t")
    first = await coordinator.start_episode()
    coordinator.on_capture_lost("mock capture stopped")
    for _ in range(300):
        snap = coordinator.snapshot()
        if snap.service.phase is RecordingPhase.RECOVERY_REQUIRED and snap.active_operation is None:
            break
        await asyncio.sleep(0.01)
    assert coordinator.snapshot().service.phase is RecordingPhase.RECOVERY_REQUIRED
    with pytest.raises(StateError):
        await coordinator.start_episode()  # an interrupted take blocks recording until it is cleared

    first_pending = tmp_path / "out" / first.session_id / first.pending_path
    assert first_pending.is_dir()  # really on disk before the reset, not just claimed by the snapshot
    await coordinator.new_batch()  # the interrupted take stays on disk as pending, not deleted
    assert coordinator.snapshot().service is None
    assert first_pending.is_dir()  # still there after the reset -- new_batch never deletes retained data
    second = await coordinator.start_episode()
    assert coordinator.snapshot().service.phase is RecordingPhase.RECORDING
    result = await coordinator.stop_episode(second.episode_id)
    assert result.disposition is Disposition.KEPT
    await coordinator.close()


async def test_a_failed_open_is_released_so_the_next_record_retries(tmp_path: Path, monkeypatch):
    coordinator = make_coordinator(tmp_path)
    coordinator.set_task("t")
    real_open = coordinator_module._RecordingService.open_session
    calls = {"n": 0}

    async def flaky_open(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk unavailable")
        return await real_open(self)

    monkeypatch.setattr(coordinator_module._RecordingService, "open_session", flaky_open)
    with pytest.raises(OSError):
        await coordinator.start_episode()
    assert coordinator.snapshot().service is None and coordinator.status is SessionStatus.CLOSED
    episode = await coordinator.start_episode()
    assert (await coordinator.stop_episode(episode.episode_id)).disposition is Disposition.KEPT
    await coordinator.close()


async def test_a_stop_after_the_source_went_stale_is_retained_not_kept(tmp_path: Path):
    state = {"ready": True}
    coordinator = make_coordinator(
        tmp_path, readiness_check=lambda: (state["ready"], None if state["ready"] else "camera stale")
    )
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    state["ready"] = False  # stalled without any callback arriving to notice it
    result = await coordinator.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    await coordinator.close()


async def test_a_final_close_keeps_admission_blocked_but_an_ordinary_close_reopens_it(tmp_path: Path):
    coordinator = make_coordinator(tmp_path)
    coordinator.set_task("t")
    await coordinator.close()
    coordinator.set_task("again")  # an ordinary close leaves the coordinator usable
    await coordinator.close(final=True)
    with pytest.raises(CoordinatorBusyError):
        coordinator.set_task("too late")
    with pytest.raises(CoordinatorBusyError):
        await coordinator.start_episode()
