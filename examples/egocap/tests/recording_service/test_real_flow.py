"""RecordingService end to end with a real recorded episode and the real Zarr/MP4 backend."""

import asyncio
import json

import cv2
import numpy as np
import zarr
from real import CAMERA, NUMERIC, make_config, needs_real_episode, reader, samples, submit_all

from recording_service import Disposition, Outcome, RecordingPhase, RecordingService, SubmissionStatus

ROWS = 60

pytestmark = needs_real_episode


async def test_full_take_is_written_checked_finalized_and_matches_the_input(tmp_path, own):
    """Real rows/frames in -> KEPT episode whose reopened arrays equal the input and whose MP4 decodes."""
    service = own(RecordingService(make_config(tmp_path / "out")))
    session = await service.open_session()
    episode = await service.start_episode("stack blocks", "operator-1")
    await submit_all(service, episode.episode_id, samples(ROWS))
    result = await service.stop_episode(episode.episode_id)

    assert result.disposition is Disposition.KEPT, result.issues
    assert result.quality.outcome is Outcome.PASSED, result.quality
    assert {c.source_name: (c.accepted, c.written) for c in result.counts} == {
        "pico_controller": (ROWS, ROWS),
        "manus_glove": (ROWS, ROWS),
        CAMERA: (ROWS, ROWS),
    }
    assert result.final_path == session.session_path / "episode_0" and result.pending_path is None
    assert not (session.session_path / "episode_0.pending").exists()

    src, out = zarr.open_group(str(reader().path), mode="r"), zarr.open_group(str(result.final_path), mode="r")
    for name, fields in NUMERIC.items():
        for field in fields or [n for n, _ in src[name].arrays()]:
            if field in out[name]:
                assert np.array_equal(out[name][field][:], src[name][field][:ROWS]), (name, field)
    assert np.array_equal(out[CAMERA]["capture_time"][:], src[CAMERA]["capture_time"][:ROWS])
    decoded = cv2.VideoCapture(str(result.final_path / f"{CAMERA}.mp4"))
    assert int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)) == ROWS
    manifest = json.loads((session.session_path / ".recording" / "episode_0.json").read_text())
    assert manifest["storage_phase"] == "FINALIZED" and manifest["task"] == "stack blocks"
    assert service.snapshot().phase is RecordingPhase.IDLE
    assert (await service.close()).released


async def test_full_queue_consumes_nothing_and_stop_drains_everything_accepted(tmp_path, own):
    """A FULL result leaves counts unchanged; every accepted sample is written by stop."""
    service = own(RecordingService(make_config(tmp_path / "out", queue_capacity=4)))
    await service.open_session()
    episode = await service.start_episode("t")
    items = samples(3)
    statuses = [service.submit_sample(episode.episode_id, item) for item in items[:6]]
    accepted = [s for s in statuses if s.status is SubmissionStatus.ACCEPTED]
    full = [s for s in statuses if s.status is SubmissionStatus.FULL]
    assert len(accepted) == 4 and len(full) == 2 and all(s.admission_index is None for s in full)
    assert sum(c.accepted for c in service.snapshot().counts) == 4
    result = await service.stop_episode(episode.episode_id)
    assert sum(c.written for c in result.counts) == 4 == sum(c.accepted for c in result.counts)
    await service.close()


async def test_repeated_stop_joins_one_stop_and_finalizes_once(tmp_path, own):
    """Concurrent and later stops return the same result; the episode is finalized and counted once."""
    service = own(RecordingService(make_config(tmp_path / "out")))
    session = await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(20))
    first, second = await asyncio.gather(
        service.stop_episode(episode.episode_id), service.stop_episode(episode.episode_id)
    )
    assert first is second and first.disposition is Disposition.KEPT
    assert await service.stop_episode(episode.episode_id) is first
    assert [p.name for p in session.session_path.glob("episode_*")] == ["episode_0"]
    await service.close()


async def test_caller_cancellation_does_not_cancel_the_stop(tmp_path, own):
    """Cancelling the awaiting caller leaves the service-owned stop running to completion."""
    service = own(RecordingService(make_config(tmp_path / "out")))
    session = await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(20))
    waiter = asyncio.create_task(service.stop_episode(episode.episode_id))
    await asyncio.sleep(0)
    waiter.cancel()
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT and (session.session_path / "episode_0").is_dir()
    await service.close()


async def test_second_episode_in_the_same_session_and_discard_never_touch_final_output(tmp_path, own):
    """Episode indices advance; discarding a take removes only its own pending data."""
    service = own(RecordingService(make_config(tmp_path / "out")))
    session = await service.open_session()
    first = await service.start_episode("one")
    await submit_all(service, first.episode_id, samples(10))
    kept = await service.stop_episode(first.episode_id)
    second = await service.start_episode("two")
    assert second.episode_index == 1
    await submit_all(service, second.episode_id, samples(10))
    gone = await service.discard_episode(second.episode_id)
    assert gone.disposition is Disposition.DISCARDED
    assert not (session.session_path / "episode_1.pending").exists()
    assert kept.final_path.is_dir() and service.snapshot().phase is RecordingPhase.IDLE
    third = await service.start_episode("three")
    assert third.episode_index == 2
    await service.close()


async def test_close_discards_an_unsaved_take_and_releases_the_session_for_another_owner(tmp_path, own):
    """Close discards the active take, reports release, and frees the session lock."""
    config = make_config(tmp_path / "out")
    service = own(RecordingService(config))
    session = await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(5))
    assert (await service.close()).released
    assert not (session.session_path / "episode_0.pending").exists()
    other = own(RecordingService(config))
    await other.open_session(session.session_id)
    assert (await other.close()).released
