# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Failure, restart and shutdown behavior of RecordingService against the real backend.

Crashes are real process deaths (SIGKILL / os._exit inside the finalize step). Storage failures and a
blocked storage call are injected by wrapping the real EpisodeStore methods, never by replacing the store.
"""

import asyncio
import json
import signal
import threading
from pathlib import Path

import numpy as np
import pytest
import zarr
from real import CAMERA, kept_episode, make_config, needs_real_episode, reader, run_child, samples, submit_all

import egocap.recording.episode_store as store_module
from egocap.recording.episode_store import EpisodeStore
from recording_service import (
    Disposition,
    NumericSample,
    RecordingPhase,
    RecordingService,
    StateError,
    SubmissionStatus,
    ValidationError,
    VideoSample,
)

pytestmark = needs_real_episode


async def test_sigkill_mid_write_leaves_pending_data_that_is_exposed_and_discardable(tmp_path, own):
    """A killed recorder's interrupted episode is found on restart, refused for recovery, and removable alone."""
    config = make_config(tmp_path / "out")
    session, kept = await kept_episode(config)
    child = run_child(tmp_path / "out", "writing", session.session_id)
    try:
        assert child.stdout.readline().strip() == session.session_id
        assert await asyncio.to_thread(child.stdout.readline) == "READY\n"
        child.send_signal(signal.SIGKILL)
    finally:
        child.wait(timeout=30)

    service = own(RecordingService(config))
    await service.open_session(session.session_id)
    (pending,) = service.pending_episodes()
    assert pending.storage_phase == "WRITING" and not pending.recoverable
    with pytest.raises(StateError):
        await service.start_episode("blocked by pending data")
    with pytest.raises(StateError):
        await service.recover_episode(pending.episode_id)
    result = await service.discard_pending(pending.episode_id)
    assert result.disposition is Disposition.DISCARDED and not pending.pending_path.exists()
    assert kept.final_path.is_dir() and not service.pending_episodes()
    assert (await service.start_episode("after discard")).episode_index == 2
    await service.close()


async def test_crash_after_check_before_rename_is_recovered_without_rewriting_data(tmp_path, own):
    """A process death between the CHECKED manifest and the rename resumes at the rename and keeps the data."""
    config = make_config(tmp_path / "out")
    child = run_child(tmp_path / "out", "after_checked")
    session_id = child.stdout.readline().strip()
    assert child.wait(timeout=60) == 9

    service = own(RecordingService(config))
    await service.open_session(session_id)
    (pending,) = service.pending_episodes()
    assert pending.storage_phase == "CHECKED" and pending.recoverable
    result = await service.recover_episode(pending.episode_id)
    assert result.disposition is Disposition.KEPT and result.final_path.is_dir()
    assert not pending.pending_path.exists() and not service.pending_episodes()
    src = zarr.open_group(str(reader().path), mode="r")["pico_controller"]["head_pose"][:15]
    assert np.array_equal(
        zarr.open_group(str(result.final_path), mode="r")["pico_controller"]["head_pose"][:], src
    )
    with pytest.raises(StateError):
        await service.recover_episode(pending.episode_id)
    await service.close()


async def test_write_failure_retains_the_take_and_blocks_a_new_one_until_resolved(tmp_path, monkeypatch, own):
    """A genuine storage error is reported, the data is kept, and only an explicit discard clears it."""
    real_append, calls = EpisodeStore.append_numeric, {"n": 0}

    def failing_third_write(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        return real_append(self, *args, **kwargs)

    monkeypatch.setattr(EpisodeStore, "append_numeric", failing_third_write)
    service = own(RecordingService(make_config(tmp_path / "out")))
    await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(10))
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    assert any("disk full" in issue for issue in result.issues) and result.pending_path.is_dir()
    assert service.snapshot().phase is RecordingPhase.RECOVERY_REQUIRED
    with pytest.raises(StateError):
        await service.start_episode("blocked")
    with pytest.raises(StateError):
        await service.recover_episode(episode.episode_id)
    assert (await service.discard_pending(episode.episode_id)).disposition is Disposition.DISCARDED
    assert service.snapshot().phase is RecordingPhase.IDLE
    assert (await service.start_episode("again")).episode_index == 1
    await service.close()


async def test_close_reports_unreleased_work_truthfully_and_keeps_the_lock_until_it_ends(
    tmp_path, monkeypatch, own
):
    """A storage call that outlives the close deadline is reported, keeps the session locked, then completes."""
    real_append, entered, gate = EpisodeStore.append_numeric, threading.Event(), threading.Event()

    def blocked(self, *args, **kwargs):
        entered.set()
        gate.wait(30)
        return real_append(self, *args, **kwargs)

    monkeypatch.setattr(EpisodeStore, "append_numeric", blocked)
    config = make_config(tmp_path / "out", close_deadline_seconds=0.5)
    service = own(RecordingService(config))
    session = await service.open_session()
    episode = await service.start_episode("t")
    service.submit_sample(episode.episode_id, samples(1)[0])
    assert await asyncio.to_thread(entered.wait, 10)

    try:
        closed = await service.close()
        assert not closed.released and closed.unfinished_episode_ids == (episode.episode_id,)
        other = own(RecordingService(config))
        with pytest.raises(StateError, match="another recorder"):
            await other.open_session(session.session_id)
    finally:
        gate.set()  # always release the blocked storage call, even if an assertion above fails
    assert (await service.close()).released
    await other.open_session(session.session_id)
    await other.close()


async def test_malformed_samples_are_rejected_before_anything_is_consumed(tmp_path, own):
    """Each kind of schema violation raises ValidationError and leaves every count at zero."""
    service = own(RecordingService(make_config(tmp_path / "out")))
    await service.open_session()
    episode = await service.start_episode("t")
    pico, camera = samples(1)[0], samples(1)[2]
    bad = [
        NumericSample("nope", 1.0, 1.0, pico.fields),
        NumericSample(pico.source, float("nan"), 1.0, pico.fields),
        NumericSample(pico.source, 1.0, 1.0, {k: v for k, v in list(pico.fields.items())[1:]}),
        NumericSample(pico.source, 1.0, 1.0, {**pico.fields, "head_pose": np.zeros(3)}),
        NumericSample(
            pico.source, 1.0, 1.0, {**pico.fields, "head_pose": pico.fields["head_pose"].astype(np.float32)}
        ),
        VideoSample(CAMERA, camera.image[:10], camera.pixel_format, 1.0, 1.0, camera.intrinsics),
        VideoSample(CAMERA, camera.image, camera.pixel_format, 1.0, 1.0, None),
        VideoSample(pico.source, camera.image, camera.pixel_format, 1.0, 1.0, camera.intrinsics),
    ]
    for sample in bad:
        with pytest.raises(ValidationError):
            service.submit_sample(episode.episode_id, sample)
    assert all(c.accepted == 0 for c in service.snapshot().counts)
    with pytest.raises(StateError):
        service.submit_sample("wrong-episode", pico)
    await service.stop_episode(episode.episode_id)
    with pytest.raises(StateError):
        service.submit_sample(episode.episode_id, pico)
    await service.close()


class _FailingRelease:
    """Wraps a real video writer; release really finishes the file, then reports an encoder failure."""

    def __init__(self, real_writer):
        self._real = real_writer

    def write(self, frame):
        self._real.write(frame)

    def release(self):
        self._real.release()
        raise RuntimeError("encoder exited abnormally")


async def test_encoder_release_failure_blocks_finalize_delete_and_clean_close(tmp_path, monkeypatch, own):
    """An unconfirmed encoder release is never treated as success: no final data, no deletion, no lock release."""
    real_create = store_module.create_video_writer
    monkeypatch.setattr(store_module, "create_video_writer", lambda *a, **k: _FailingRelease(real_create(*a, **k)))
    config = make_config(tmp_path / "out", close_deadline_seconds=1.0)
    service = own(RecordingService(config), may_stay_open=True)
    session = await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(10))
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and any("unconfirmed" in i for i in result.issues)
    assert not (session.session_path / "episode_0").exists() and result.pending_path.is_dir()
    again = await service.discard_pending(episode.episode_id)
    assert again.disposition is Disposition.RECOVERY_REQUIRED and result.pending_path.is_dir()
    closed = await service.close()
    assert not closed.released and closed.unfinished_episode_ids == (episode.episode_id,)
    with pytest.raises(StateError, match="another recorder"):
        await RecordingService(config).open_session(session.session_id)


def test_store_seal_failure_is_permanent_and_appends_are_rejected_after_sealing(tmp_path, monkeypatch):
    """A failed seal never later reports success; abort reports unconfirmed release; appends are refused."""
    real_create = store_module.create_video_writer
    monkeypatch.setattr(store_module, "create_video_writer", lambda *a, **k: _FailingRelease(real_create(*a, **k)))
    r = reader()
    store = EpisodeStore(
        tmp_path / "p", (r.numeric_spec("pico_controller", ("head_pose",)),), (r.video_spec(CAMERA),)
    )
    store.start_encoders()
    with pytest.raises(RuntimeError, match="video encoding failed"):
        store.seal()
    with pytest.raises(RuntimeError, match="video encoding failed"):
        store.seal()
    assert store.abort() is False
    with pytest.raises(RuntimeError, match="closed to new samples"):
        store.append_numeric("pico_controller", 1.0, 1.0, {"head_pose": np.zeros((4, 4))})


async def test_concurrent_start_does_not_disturb_the_accepted_start(tmp_path, own):
    """A second start while one is in progress is refused; the first caller gets its own episode, recording."""
    service = own(RecordingService(make_config(tmp_path / "out")))
    await service.open_session()
    first, second = await asyncio.gather(
        service.start_episode("a"), service.start_episode("b"), return_exceptions=True
    )
    assert isinstance(second, StateError) and first.episode_index == 0
    assert service.submit_sample(first.episode_id, samples(1)[0]).status is SubmissionStatus.ACCEPTED
    assert (await service.close()).released


async def test_unreadable_manifest_blocks_new_recording_until_resolved(tmp_path, own):
    """An unsupported/corrupt manifest keeps the session closed to new takes instead of being skipped."""
    config = make_config(tmp_path / "out")
    session, _ = await kept_episode(config)
    (session.session_path / ".recording" / "episode_9.json").write_text("{not json")
    service = own(RecordingService(config))
    await service.open_session(session.session_id)
    with pytest.raises(StateError, match="unreadable"):
        await service.start_episode("blocked")
    await service.close()


def _tree_digest(root: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of every file under root, to prove recovery did not rewrite data."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


async def test_crash_after_rename_before_finalized_checkpoint_recovers_without_rewriting(tmp_path, own):
    """A death after the pending->final rename leaves a CHECKED manifest and no pending directory; recovery only
    verifies the owned final directory and checkpoints, leaving every data file untouched."""
    config = make_config(tmp_path / "out")
    child = run_child(tmp_path / "out", "after_rename")
    session_id = child.stdout.readline().strip()
    assert child.wait(timeout=60) == 9
    final = tmp_path / "out" / session_id / "episode_0"
    assert final.is_dir() and not (tmp_path / "out" / session_id / "episode_0.pending").exists()
    before = _tree_digest(final)

    service = own(RecordingService(config))
    await service.open_session(session_id)
    (pending,) = service.pending_episodes()
    assert pending.storage_phase == "CHECKED" and pending.recoverable
    result = await service.recover_episode(pending.episode_id)
    assert result.disposition is Disposition.KEPT and result.final_path == final
    assert _tree_digest(final) == before and not service.pending_episodes()
    manifest = json.loads((tmp_path / "out" / session_id / ".recording" / "episode_0.json").read_text())
    assert manifest["storage_phase"] == "FINALIZED"
    assert (await service.start_episode("next")).episode_index == 1
    await service.close()


async def test_final_directory_owned_by_another_episode_is_never_accepted_on_recovery(tmp_path, own):
    """After the same crash window, a final directory stamped with a different identity is refused, not adopted."""
    config = make_config(tmp_path / "out")
    child = run_child(tmp_path / "out", "after_rename")
    session_id = child.stdout.readline().strip()
    assert child.wait(timeout=60) == 9
    final = tmp_path / "out" / session_id / "episode_0"
    zarr.open_group(str(final), mode="r+").attrs["recording_episode_id"] = "someone-else"

    service = own(RecordingService(config))
    await service.open_session(session_id)
    (pending,) = service.pending_episodes()
    result = await service.recover_episode(pending.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and any("not owned" in i for i in result.issues)
    await service.close()


async def test_partially_started_multi_camera_take_stays_owned_when_cleanup_is_unconfirmed(
    tmp_path, monkeypatch, own
):
    """A later camera failing to open, with the first encoder's release unconfirmed, keeps the session locked."""
    real_create, calls = store_module.create_video_writer, {"n": 0}

    def first_ok_second_fails(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("second camera failed")
        return _FailingRelease(real_create(*args, **kwargs))

    monkeypatch.setattr(store_module, "create_video_writer", first_ok_second_fails)
    base = make_config(tmp_path / "out", close_deadline_seconds=1.0)
    second_camera = reader().video_spec("wrist_human_camera_left")
    config = type(base)(**{**base.__dict__, "video_sources": (*base.video_sources, second_camera)})
    service = own(RecordingService(config), may_stay_open=True)
    session = await service.open_session()
    with pytest.raises(RuntimeError, match="second camera failed"):
        await service.start_episode("t")
    assert service.snapshot().phase is RecordingPhase.RECOVERY_REQUIRED
    (pending,) = service.pending_episodes()
    assert any("unconfirmed" in i for i in pending.issues)
    again = await service.discard_pending(pending.episode_id)
    assert again.disposition is Disposition.RECOVERY_REQUIRED and pending.pending_path.is_dir()
    closed = await service.close()
    assert not closed.released and closed.unfinished_episode_ids == (pending.episode_id,)
    with pytest.raises(StateError, match="another recorder"):
        await RecordingService(config).open_session(session.session_id)
