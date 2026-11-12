# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Known physical events seen by a camera and by two host-polled sources, read back from the SAVED take.

Truth events happen at known times on one reference timeline (the PC's monotonic clock). They reach a recording
three ways, each through the production path:

* a camera on its own clock (offset, drift), through ``LiveKitRecordingFeed`` with the real ``ClockEstimator``; its
  frame stamp is a fixed, stated delay after the event, and its delivery delay jitters. Two camera bases are
  covered: ``SENSOR_CAPTURE_MONOTONIC_US`` (the stamp is the capture instant: exposure and readout after the event)
  and ``PUBLISHER_MONOTONIC_US`` (the stamp is when the frame was ready to encode: the capture delay plus further
  pipeline delay). Both are mapped identically and neither is corrected toward the event; the saved time is the
  stamped time on the PC clock, i.e. the event time plus that basis's stated delay, within the declared
  uncertainty;
* a glove-like and a controller-like source, as ``CaptureFrame`` tracking blobs through the real conversion, saved
  under the HOST POLL time (``capture_time_ns``). That is not a device measurement time and nothing here maps it:
  an event shows up in the first poll at or after it, so its saved time is the poll time, up to one poll period
  late, and a source with no device latency model is assumed to have none.

Everything goes through ``RecordingIngress`` into a real ``RecordingService`` take; the assertions read the kept
episode's ``capture_time`` arrays. Identity travels in the payloads (video: the value of the frame's pixels as
submitted; glove: a joint distance; controller: a trigger value), so a time is matched to the event it belongs to.

This proves the plumbing and the uncertainty accounting on synthetic clocks. It does not prove physical sensor
alignment: the camera's pipeline delay and the poll-source latency are stated constants here, not measured.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import numpy as np
import pytest
import zarr

from capture_server import protocol as p
from capture_server.client import CaptureFrame
from egocap_web.livekit_receiver import ReceivedFrame
from egocap_web.livekit_recording import LiveKitRecordingFeed
from egocap_web.recording_adapter import (
    MANUS_GLOVE_SOURCE,
    MANUS_GLOVE_SPEC,
    PICO_CONTROLLER_SOURCE,
    PICO_CONTROLLER_SPEC,
    RecordingIngress,
)
from recording_service import QcConfig, RecordingService, ServiceConfig, VideoSourceSpec

from .test_recording_adapter import default_tracking_values
from .test_synthetic_timeline import (
    QUANTIZATION_S,
    Device,
    DeviceClock,
    S,
    SimReceiver,
)

BASE_S = 5000.0  # the PC's monotonic time of the first event
EVENT_PERIOD_S = 0.25
EVENTS = 24
#: The camera stamps the capture instant this long after the physical event (exposure and readout).
CAM_PIPELINE_S = 0.012
#: Frame-ready stamps add encode-queue delay on top of capture; the recording applies no correction for it.
FRAME_READY_PIPELINE_S = 0.041
CAMERA_BASES = {
    p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US: CAM_PIPELINE_S,
    p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US: FRAME_READY_PIPELINE_S,
}
#: Host polls of the two tracking sources; phases differ so they never coincide.
GLOVE_POLL_S, GLOVE_PHASE_S = 0.010, 0.003
CONTROLLER_POLL_S, CONTROLLER_PHASE_S = 0.014, 0.005
#: A feed budget this tight is well above what the link below can produce, so no take fails on it by accident.
BUDGET_NS = 8_000_000
CAM = "cam"
GLOVE_EVENT_BASE_M, GLOVE_EVENT_STEP_M = 0.05, 0.001


def lan_link(rng: random.Random) -> tuple[int, int]:
    """2 ms each way plus up to 3 ms of independent queueing: uncertainty at most 5 ms."""
    return 2_000_000 + int(rng.uniform(0, 3e6)), 2_000_000 + int(rng.uniform(0, 3e6))


def event_time_s(k: int) -> float:
    return BASE_S + k * EVENT_PERIOD_S


def latest_event_at(time_s: float) -> int:
    """The newest event that has happened by ``time_s`` (what a host poll at that time sees)."""
    return min(EVENTS - 1, int((time_s - BASE_S) // EVENT_PERIOD_S))


def _poll_grid(period_s: float, phase_s: float, gaps: tuple[tuple[float, float], ...]) -> list[float]:
    """Host poll times from the first event to a little after the last, minus any stall windows."""
    end_s = event_time_s(EVENTS - 1) + 0.3
    times = [BASE_S + phase_s + i * period_s for i in range(round((end_s - BASE_S) / period_s))]
    return [t for t in times if not any(lo <= t < hi for lo, hi in gaps)]


def _tracking_frame(time_s: float, *, glove_event: int | None, controller_event: int | None) -> CaptureFrame:
    """One tracking blob polled at ``time_s``. Only one source is valid in it, so each source has its own polls."""
    values = default_tracking_values()
    if glove_event is None:
        for side in ("left", "right"):
            values[f"hand_{side}.joint_valid"] = np.zeros(26, dtype=np.uint8)
    else:
        for side in ("left", "right"):
            positions = np.array(values[f"hand_{side}.joint_positions"], dtype=np.float32)
            positions[10] = positions[1] + np.array(
                [GLOVE_EVENT_BASE_M + glove_event * GLOVE_EVENT_STEP_M, 0, 0], dtype=np.float32
            )
            values[f"hand_{side}.joint_positions"] = positions
    if controller_event is None:
        values["head.is_valid"] = 0
    else:
        values["controller_left.trigger_value"] = controller_event / 64.0  # exact in float32
    ns = round(time_s * S)
    return CaptureFrame(
        session_id="s",
        description=None,  # type: ignore[arg-type]
        stream_id=1,
        sequence=0,
        capture_time_ns=ns,
        receive_time_ns=ns + 1_000_000,
        payload=p.encode_tracking_frame(values),
    )


@dataclass
class Saved:
    """The kept episode's per-source ``capture_time`` arrays, with what each row says it is."""

    camera_t: np.ndarray
    camera_k: list[int]  # event identity of each saved camera row, from the pixels as submitted
    camera_unc_s: list[float]  # the mapping uncertainty the feed declared for each saved camera row
    glove_t: np.ndarray
    glove_k: np.ndarray
    controller_t: np.ndarray
    controller_k: np.ndarray
    feed_failure: str | None
    raw_cam_stamp_s: dict[int, float]  # what the camera wrote, before any mapping
    stamp_delay_s: float = CAM_PIPELINE_S  # the stated delay of the stamped instant after the event


async def run_take(
    tmp_path,
    *,
    cam_clock: DeviceClock,
    basis: p.LiveKitTimestampBasis = p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US,
    cam_drop: frozenset[int] = frozenset(),
    glove_gaps: tuple[tuple[float, float], ...] = (),
    controller_gaps: tuple[tuple[float, float], ...] = (),
    seed: int = 11,
) -> Saved:
    rng = random.Random(seed)
    end_s = event_time_s(EVENTS) + 3.0
    device = Device(cam_clock, lan_link, random.Random(rng.random()), BASE_S - 3.0, end_s)

    frames, raw_stamp_s = [], {}
    for k in range(EVENTS):
        if k in cam_drop:
            continue
        capture_pc_ns = round((event_time_s(k) + CAMERA_BASES[basis]) * S)
        # Delivery jitter stays under the event period so frames arrive in capture order.
        delay_s = rng.uniform(0.02, 0.2)
        stamp_us = round(cam_clock.read_ns(capture_pc_ns) / 1000)
        raw_stamp_s[k] = stamp_us / 1e6
        frames.append(
            ReceivedFrame(
                stream_id=16,
                image=np.full((48, 64, 3), k, dtype=np.uint8),
                receive_time_ns=capture_pc_ns + round(delay_s * S),
                transport_timestamp_us=0,
                user_timestamp_us=stamp_us,
                frame_id=k,
                user_data=None,
            )
        )
    receiver = SimReceiver(frames, device)
    stream = p.LiveKitCameraStream(
        stream_id=16,
        name=CAM,
        url="wss://lk",
        room="r",
        publisher="o",
        track="t",
        timestamp_basis=basis,
    )

    config = ServiceConfig(
        output_root=tmp_path / "out",
        numeric_sources=(PICO_CONTROLLER_SPEC, MANUS_GLOVE_SPEC),
        video_sources=(VideoSourceSpec(CAM, 64, 48, 4.0, intrinsics=False),),
        qc=QcConfig(required_sources=(PICO_CONTROLLER_SOURCE, MANUS_GLOVE_SOURCE, CAM)),
        queue_capacity=4096,
    )
    service = RecordingService(config)
    try:
        await service.open_session()
        episode = await service.start_episode("shared-truth")
        submitted_k: list[int] = []

        def submit(sample):
            if sample.source == CAM:
                submitted_k.append(int(sample.image[0, 0, 0]))
            return service.submit_sample(episode.episode_id, sample)

        ingress = RecordingIngress(submit=submit, queue_capacity=4096, full_retry_deadline_seconds=5.0)
        await ingress.start()
        feed = LiveKitRecordingFeed(receiver, [stream], ingress, {16: CAM}, max_uncertainty_ns=BUDGET_NS)
        feed.start()
        for _ in range(5000):
            await asyncio.sleep(0)
            if feed.failure:
                break
        await feed.stop()

        for t in _poll_grid(GLOVE_POLL_S, GLOVE_PHASE_S, glove_gaps):
            assert ingress.offer(_tracking_frame(t, glove_event=latest_event_at(t), controller_event=None))
        for t in _poll_grid(CONTROLLER_POLL_S, CONTROLLER_PHASE_S, controller_gaps):
            assert ingress.offer(_tracking_frame(t, glove_event=None, controller_event=latest_event_at(t)))
        assert await ingress.stop()
        assert ingress.failure is None, ingress.failure
        result = await service.stop_episode(episode.episode_id)
    finally:
        closed = await service.close()
    assert closed.released

    group = zarr.open_group(str(result.final_path), mode="r")
    glove_k, controller_k = [], []
    for row in group[MANUS_GLOVE_SOURCE]["left_hand_pose"][:]:
        # Wrist-relative distance of the tagged joint (slot 9): rotation-invariant, so the conversion keeps it.
        glove_k.append(round((np.linalg.norm(row[9, :3, 3]) - GLOVE_EVENT_BASE_M) / GLOVE_EVENT_STEP_M))
    for trigger in group[PICO_CONTROLLER_SOURCE]["left_trigger"][:]:
        controller_k.append(round(float(trigger) * 64))
    camera_t = group[CAM]["capture_time"][:]
    assert len(camera_t) == len(submitted_k)
    return Saved(
        camera_t=camera_t,
        camera_k=submitted_k,
        camera_unc_s=[m.uncertainty_ns / S for m in receiver.clock_sync.used[: len(submitted_k)] if m is not None],
        glove_t=group[MANUS_GLOVE_SOURCE]["capture_time"][:],
        glove_k=np.array(glove_k),
        controller_t=group[PICO_CONTROLLER_SOURCE]["capture_time"][:],
        controller_k=np.array(controller_k),
        feed_failure=feed.failure,
        raw_cam_stamp_s=raw_stamp_s,
        stamp_delay_s=CAMERA_BASES[basis],
    )


def first_poll_showing(times: np.ndarray, events: np.ndarray) -> dict[int, float]:
    """Saved time of the first row that shows each event: when the host first saw it."""
    first: dict[int, float] = {}
    for t, k in zip(times, events):
        first.setdefault(int(k), float(t))
    return first


def default_cam_clock() -> DeviceClock:
    return DeviceClock(3_200_000_000_000, 35.0)  # boots 3200 s ahead of the PC, 35 ppm fast


def assert_saved_artifacts_agree_with_truth(saved: Saved, *, observed_by_camera: set[int]) -> None:
    glove_first = first_poll_showing(saved.glove_t, saved.glove_k)
    controller_first = first_poll_showing(saved.controller_t, saved.controller_k)
    cam_by_k = dict(zip(saved.camera_k, saved.camera_t))
    unc_by_k = dict(zip(saved.camera_k, saved.camera_unc_s))
    assert set(cam_by_k) == observed_by_camera

    # Host-polled sources: the saved time IS the poll time, on the poll grid and never the event time.
    grids = {
        "glove": (saved.glove_t, GLOVE_POLL_S, GLOVE_PHASE_S),
        "controller": (saved.controller_t, CONTROLLER_POLL_S, CONTROLLER_PHASE_S),
    }
    for name, (times, period_s, phase_s) in grids.items():
        on_grid = (times - BASE_S - phase_s) / period_s
        assert np.allclose(on_grid, np.round(on_grid), atol=1e-6), name
        assert np.all(np.diff(times) > 0), name
    for k, poll_s in glove_first.items():
        assert 0 <= poll_s - event_time_s(k)  # never before the event
    for k, poll_s in controller_first.items():
        assert 0 <= poll_s - event_time_s(k)

    paired = 0
    for k, cam_s in cam_by_k.items():
        truth_s, unc_s = event_time_s(k), unc_by_k[k]
        # Camera against truth: the stated pipeline delay, within the declared uncertainty of its mapping.
        assert abs(cam_s - (truth_s + saved.stamp_delay_s)) <= unc_s + QUANTIZATION_S, (k, cam_s - truth_s)
        for firsts, period_s in ((glove_first, GLOVE_POLL_S), (controller_first, CONTROLLER_POLL_S)):
            if k not in firsts:
                continue  # a stall swallowed every poll that would have first shown this event
            poll_s = firsts[k]
            poll_delay_s = poll_s - truth_s
            # Relative time between two sources: known pipeline minus known poll delay, within the camera's
            # declared uncertainty. The poll-time source adds none: its time is host time.
            assert abs((cam_s - poll_s) - (saved.stamp_delay_s - poll_delay_s)) <= unc_s + QUANTIZATION_S, k
            # The same claim in the bound a reader can state without knowing the delays exactly.
            assert (
                abs(cam_s - poll_s) <= unc_s + saved.stamp_delay_s + max(poll_delay_s, period_s) + QUANTIZATION_S
            )
            paired += 1
    assert paired


@pytest.mark.parametrize("basis", list(CAMERA_BASES), ids=lambda b: b.name)
async def test_camera_events_line_up_with_host_polled_sources_in_the_saved_timestamps(tmp_path, basis):
    saved = await run_take(tmp_path, cam_clock=default_cam_clock(), basis=basis)
    assert saved.feed_failure == "cam: the LiveKit track was lost"
    assert len(saved.camera_t) == EVENTS
    assert set(saved.glove_k) == set(saved.controller_k) == set(range(EVENTS))
    assert_saved_artifacts_agree_with_truth(saved, observed_by_camera=set(range(EVENTS)))

    # The mapping did the work: the camera wrote a clock thousands of seconds from ours, and the saved time is not
    # that, nor the stamp with the boot offset subtracted (35 ppm of drift since boot is ~0.17 s by now).
    k = EVENTS - 1
    assert abs(saved.raw_cam_stamp_s[k] - saved.camera_t[-1]) > 1000
    naive_s = saved.raw_cam_stamp_s[k] - 3200.0
    assert abs(naive_s - saved.camera_t[-1]) > 0.1


async def test_dropped_camera_frames_and_poll_stalls_leave_the_remaining_saved_times_and_identities_correct(
    tmp_path,
):
    cam_drop = frozenset({3, 4, 17})
    glove_gap = (event_time_s(9) - 0.01, event_time_s(10) + 0.12)  # the host stalls over events 9 and 10
    saved = await run_take(tmp_path, cam_clock=default_cam_clock(), cam_drop=cam_drop, glove_gaps=(glove_gap,))
    assert saved.feed_failure == "cam: the LiveKit track was lost"
    assert sorted(saved.camera_k) == sorted(set(range(EVENTS)) - cam_drop)
    # Event 9 happened inside the stall and was superseded by event 10 before the first poll after it, so the
    # glove never shows 9 on its own row: it is absent, not shifted onto another event's time.
    assert 9 not in set(saved.glove_k)
    assert set(saved.glove_k) == set(range(EVENTS)) - {9}
    assert set(saved.controller_k) == set(range(EVENTS))
    assert not np.any((saved.glove_t >= glove_gap[0]) & (saved.glove_t < glove_gap[1]))
    assert_saved_artifacts_agree_with_truth(saved, observed_by_camera=set(range(EVENTS)) - cam_drop)
    # The first poll to see event 10 comes after the stall, much later than the usual poll delay.
    first = first_poll_showing(saved.glove_t, saved.glove_k)
    assert first[10] - event_time_s(10) > GLOVE_POLL_S


async def test_a_publisher_clock_jump_ends_the_camera_and_no_frame_after_it_is_recorded_under_a_new_epoch(
    tmp_path,
):
    cam_clock = default_cam_clock()
    step_s = event_time_s(8) + 0.05  # between events; the next periodic exchange comes about a second later
    segment = cam_clock.segments[0]
    step_ns = 40_000_000
    cam_clock.add_segment(step_s, segment[1] + step_ns, segment[2])  # the publisher's clock jumps 40 ms
    saved = await run_take(tmp_path, cam_clock=cam_clock)
    assert saved.feed_failure is not None and "clock mapping jumped" in saved.feed_failure
    assert 0 < len(saved.camera_t) < EVENTS
    # Tracking is host time and is unaffected by the camera's failure: it still covers every event.
    assert set(saved.glove_k) == set(saved.controller_k) == set(range(EVENTS))

    # Saved camera rows are a prefix of the events: nothing after the failure is recorded, under the old
    # mapping or a new one.
    assert saved.camera_k == list(range(len(saved.camera_k)))
    cam_by_k = dict(zip(saved.camera_k, saved.camera_t))
    clean = {k for k in cam_by_k if event_time_s(k) + CAM_PIPELINE_S < step_s}
    assert clean
    assert_saved_artifacts_agree_with_truth_for(saved, clean)

    # Known gap, pinned so it cannot change unseen: a frame captured after the jump but before the next
    # exchange reveals it is stamped under the old mapping, so it carries the whole jump as error.
    unseen = [k for k in cam_by_k if k not in clean]
    assert unseen, "the jump should have been detected only after at least one frame was recorded"
    for k in unseen:
        error_s = cam_by_k[k] - (event_time_s(k) + CAM_PIPELINE_S)
        assert error_s == pytest.approx(step_ns / 1e9, abs=0.006), (k, error_s)


def assert_saved_artifacts_agree_with_truth_for(saved: Saved, ks: set[int]) -> None:
    """The camera-against-truth and camera-against-poll checks, for the events in ``ks`` only."""
    cam_by_k = dict(zip(saved.camera_k, saved.camera_t))
    unc_by_k = dict(zip(saved.camera_k, saved.camera_unc_s))
    glove_first = first_poll_showing(saved.glove_t, saved.glove_k)
    for k in ks:
        truth_s = event_time_s(k)
        assert abs(cam_by_k[k] - (truth_s + saved.stamp_delay_s)) <= unc_by_k[k] + QUANTIZATION_S, k
        poll_s = glove_first[k]
        assert (
            abs((cam_by_k[k] - poll_s) - (saved.stamp_delay_s - (poll_s - truth_s)))
            <= unc_by_k[k] + QUANTIZATION_S
        )
