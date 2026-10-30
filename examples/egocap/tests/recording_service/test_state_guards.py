"""State guards and error paths of RecordingService, against the real backend and a real recorded episode.

Failures are injected by wrapping real components (or corrupting real staged output), never by replacing the
store, writer or checker on the success path.
"""

import asyncio
import copy
import hashlib
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import zarr
from real import (
    CAMERA,
    kept_episode,
    make_config,
    needs_real_episode,
    run_child,
    samples,
    submit_all,
)

import recording_service.recording_service as impl
from egocap.recording.episode_layout import SessionLock
from egocap.recording.episode_store import EpisodeStore
from recording_service import (
    CalibrationAsset,
    ConfigurationError,
    Disposition,
    FieldSpec,
    NumericSample,
    NumericSourceSpec,
    Outcome,
    PixelFormat,
    QcConfig,
    RecordingPhase,
    RecordingService,
    ServiceConfig,
    StateError,
    ValidationError,
    VideoSample,
    VideoSourceSpec,
)

pytestmark = needs_real_episode


async def _recording(own, tmp_path, **overrides):
    """An open service with one active episode holding a few real samples."""
    service = own(RecordingService(make_config(tmp_path / "out", **overrides)))
    session = await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(5))
    return service, session, episode


async def test_conflicting_commands_are_refused_and_repeats_join(tmp_path, own):
    """Discard while a stop runs is refused; two discards share one result; samples are refused once stopping."""
    service, _, episode = await _recording(own, tmp_path)
    stop = asyncio.create_task(service.stop_episode(episode.episode_id))
    await asyncio.sleep(0)
    with pytest.raises(StateError, match="stop in progress"):
        await service.discard_episode(episode.episode_id)
    with pytest.raises(StateError, match="not admitting"):
        service.submit_sample(episode.episode_id, samples(1)[0])
    assert (await stop).disposition is Disposition.KEPT
    await service.close()

    service, _, episode = await _recording(own, tmp_path / "second")
    first, second = await asyncio.gather(
        service.discard_episode(episode.episode_id), service.discard_episode(episode.episode_id)
    )
    assert first is second and first.disposition is Disposition.DISCARDED
    starting = asyncio.create_task(service.start_episode("again"))
    await asyncio.sleep(0)
    closed = await service.close()
    assert closed.released
    assert (await starting).episode_index == 1  # the accepted start finished; close then discarded that take
    assert not list((tmp_path / "second" / "out").glob("*/episode_1*"))


async def test_stop_while_discarding_and_unknown_episode_are_refused(tmp_path, own):
    """Stop during a discard is refused; commands for an unknown episode id are refused."""
    service, _, episode = await _recording(own, tmp_path)
    discard = asyncio.create_task(service.discard_episode(episode.episode_id))
    await asyncio.sleep(0)
    with pytest.raises(StateError, match="discard in progress"):
        await service.stop_episode(episode.episode_id)
    await discard
    with pytest.raises(StateError, match="no active episode"):
        await service.discard_episode("not-an-episode")
    await service.close()


async def test_session_lifecycle_guards(tmp_path, own):
    """Start before open, double open, unknown session id and open-after-close are refused."""
    config = make_config(tmp_path / "out")
    service = own(RecordingService(config))
    with pytest.raises(StateError, match="no open session"):
        await service.start_episode("t")
    session = await service.open_session()
    with pytest.raises(StateError, match="already open"):
        await service.open_session()
    assert (await service.close()).released
    with pytest.raises(StateError, match="closed"):
        await service.open_session()
    with pytest.raises(ConfigurationError, match="no session"):
        await RecordingService(config).open_session("does-not-exist")
    with pytest.raises(ConfigurationError, match="no session"):
        await RecordingService(config).open_session("../" + session.session_id)


async def test_configuration_with_nonpositive_video_geometry_is_refused(tmp_path):
    """Zero width/height/fps is a configuration error."""
    base = make_config(tmp_path / "out")
    bad = VideoSourceSpec("cam", 0, 10, 30.0)
    with pytest.raises(ConfigurationError, match="positive"):
        await RecordingService(ServiceConfig(tmp_path / "o", base.numeric_sources, (bad,))).open_session()


async def test_sample_kind_and_intrinsics_mismatches_are_rejected(tmp_path, own):
    """Wrong sample kind for a source, a non-PixelFormat, and undeclared intrinsics all raise ValidationError."""
    numeric = NumericSourceSpec("n", (FieldSpec("v", "float64"),))
    plain = VideoSourceSpec("plain", 16, 12, 30.0, intrinsics=False)
    service = own(RecordingService(ServiceConfig(tmp_path / "out", (numeric,), (plain,))))
    await service.open_session()
    episode = await service.start_episode("t")
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    bad = [
        NumericSample("plain", 1.0, 1.0, {"v": np.float64(1)}),
        VideoSample("n", image, PixelFormat.RGB8, 1.0, 1.0, None),
        VideoSample("plain", image, "rgb8", 1.0, 1.0, None),
        VideoSample("plain", image, PixelFormat.RGB8, 1.0, 1.0, np.eye(3)),
        "not a sample",
    ]
    for sample in bad:
        with pytest.raises(ValidationError):
            service.submit_sample(episode.episode_id, sample)
    good = VideoSample("plain", image, PixelFormat.BGR8, 1.0, 1.0, None)
    assert service.submit_sample(episode.episode_id, good).admission_index == 0
    assert (await service.stop_episode(episode.episode_id)).disposition is Disposition.KEPT
    await service.close()


async def test_pending_episode_from_a_different_configuration_is_not_recoverable(tmp_path, own):
    """A sealed/checked pending episode from another configuration is exposed but refused for recovery."""
    config = make_config(tmp_path / "out")
    child = run_child(tmp_path / "out", "after_checked")
    session_id = child.stdout.readline().strip()
    assert child.wait(timeout=60) == 9
    other = make_config(tmp_path / "out", qc=QcConfig(required_sources=("pico_controller",)))
    service = own(RecordingService(other))
    await service.open_session(session_id)
    (pending,) = service.pending_episodes()
    assert not pending.recoverable and any("different configuration" in i for i in pending.issues)
    with pytest.raises(StateError, match="different configuration"):
        await service.recover_episode(pending.episode_id)
    await service.close()
    assert config  # the original configuration object stays valid for the child's session


async def test_pending_episode_whose_manifest_vanished_cannot_be_recovered(tmp_path, own):
    """If the manifest disappears after the scan, recovery refuses instead of guessing."""
    child = run_child(tmp_path / "out", "after_checked")
    session_id = child.stdout.readline().strip()
    assert child.wait(timeout=60) == 9
    service = own(RecordingService(make_config(tmp_path / "out")))
    await service.open_session(session_id)
    (pending,) = service.pending_episodes()
    next((tmp_path / "out" / session_id / ".recording").glob("episode_*.json")).unlink()
    with pytest.raises(StateError, match="no readable manifest"):
        await service.recover_episode(pending.episode_id)
    await service.close()


async def test_unreadable_staged_dataset_makes_the_check_error_and_retains_the_take(tmp_path, monkeypatch, own):
    """If the staged dataset cannot be opened by the real checker, the take is retained, never finalized."""
    real_check = impl.check_episode

    def break_root_then_check(pending_dir, *args):
        (pending_dir / "zarr.json").write_text("{not json")
        return real_check(pending_dir, *args)

    monkeypatch.setattr(impl, "check_episode", break_root_then_check)
    service, session, episode = await _recording(own, tmp_path)
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    assert any("quality check errored" in i for i in result.issues)
    assert not (session.session_path / "episode_0").exists()
    assert service.snapshot().phase is RecordingPhase.RECOVERY_REQUIRED
    await service.close()


async def test_start_failures_after_and_before_the_manifest_exists(tmp_path, monkeypatch, own):
    """A store failure retains the started take; a manifest-write failure leaves the service idle and clean."""
    real_store = impl.EpisodeStore

    def failing_store(*args, **kwargs):
        raise OSError("cannot open store")

    monkeypatch.setattr(impl, "EpisodeStore", failing_store)
    service = own(RecordingService(make_config(tmp_path / "a")))
    await service.open_session()
    with pytest.raises(OSError, match="cannot open store"):
        await service.start_episode("t")
    assert service.snapshot().phase is RecordingPhase.RECOVERY_REQUIRED and service.pending_episodes()
    (pending,) = service.pending_episodes()
    assert (await service.discard_pending(pending.episode_id)).disposition is Disposition.DISCARDED
    monkeypatch.setattr(impl, "EpisodeStore", real_store)
    assert (await service.start_episode("works now")).episode_index == 1
    await service.close()

    monkeypatch.setattr(impl, "write_manifest", lambda *a, **k: (_ for _ in ()).throw(OSError("no manifest")))
    service = own(RecordingService(make_config(tmp_path / "b")))
    await service.open_session()
    with pytest.raises(OSError, match="no manifest"):
        await service.start_episode("t")
    assert service.snapshot().phase is RecordingPhase.IDLE and not service.pending_episodes()
    await service.close()


async def test_discard_failure_retains_the_take(tmp_path, monkeypatch, own):
    """If removing the pending data fails, the take stays recoverable with the diagnostic."""
    monkeypatch.setattr(impl, "remove_owned_episode", lambda *a, **k: (_ for _ in ()).throw(OSError("rm failed")))
    service, session, episode = await _recording(own, tmp_path)
    result = await service.discard_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and any("rm failed" in i for i in result.issues)
    assert (session.session_path / "episode_0.pending").is_dir()
    with pytest.raises(StateError, match="needs recovery"):
        await service.stop_episode(episode.episode_id)
    monkeypatch.undo()
    assert (await service.discard_pending(episode.episode_id)).disposition is Disposition.DISCARDED
    await service.close()


async def test_a_kept_episode_does_not_block_the_next_take_in_the_same_session(tmp_path, own):
    """After a kept episode the session can be reopened and records the next take with the next index."""
    config = make_config(tmp_path / "out")
    session, kept = await kept_episode(config)
    assert kept.disposition is Disposition.KEPT and kept.final_path.parent == session.session_path
    assert CAMERA in {c.source_name for c in kept.counts}
    service = own(RecordingService(config))
    await service.open_session(session.session_id)
    assert not service.pending_episodes()
    assert (await service.start_episode("next")).episode_index == 1


async def test_pending_commands_after_close_are_refused_and_change_nothing(tmp_path, own):
    """After close releases the session, recover/discard/start/stop on the stale service touch nothing."""
    child = run_child(tmp_path / "out", "after_checked")
    session_id = child.stdout.readline().strip()
    assert child.wait(timeout=60) == 9
    config = make_config(tmp_path / "out")
    stale = own(RecordingService(config))
    await stale.open_session(session_id)
    (pending,) = stale.pending_episodes()
    session_path = tmp_path / "out" / session_id
    before = sorted((p.relative_to(session_path), p.stat().st_mtime_ns) for p in session_path.rglob("*"))
    assert (await stale.close()).released

    for call in (
        stale.recover_episode(pending.episode_id),
        stale.discard_pending(pending.episode_id),
        stale.start_episode("t"),
        stale.stop_episode(pending.episode_id),
        stale.discard_episode(pending.episode_id),
    ):
        with pytest.raises(StateError):
            await call
    assert before == sorted((p.relative_to(session_path), p.stat().st_mtime_ns) for p in session_path.rglob("*"))

    fresh = own(
        RecordingService(config)
    )  # the session is owned by a new service now; the stale one still changes nothing
    await fresh.open_session(session_id)
    with pytest.raises(StateError):
        await stale.recover_episode(pending.episode_id)
    assert fresh.pending_episodes()[0].storage_phase == "CHECKED"
    await fresh.close()


async def test_new_commands_during_close_are_refused_while_accepted_work_finishes(tmp_path, monkeypatch, own):
    """While close waits on a blocked storage call, new commands are refused; close still completes."""
    real_append, entered, gate = EpisodeStore.append_numeric, threading.Event(), threading.Event()

    def blocked(self, *args, **kwargs):
        entered.set()
        gate.wait(30)
        return real_append(self, *args, **kwargs)

    monkeypatch.setattr(EpisodeStore, "append_numeric", blocked)
    service = own(RecordingService(make_config(tmp_path / "out", close_deadline_seconds=20.0)))
    await service.open_session()
    episode = await service.start_episode("t")
    service.submit_sample(episode.episode_id, samples(1)[0])
    assert await asyncio.to_thread(entered.wait, 10)
    closing = asyncio.create_task(service.close())
    try:
        await asyncio.sleep(0.2)
        for call in (service.recover_episode("x"), service.discard_pending("x"), service.start_episode("t")):
            with pytest.raises(StateError):
                await call
        with pytest.raises(StateError, match="in progress"):
            await service.stop_episode(episode.episode_id)
    finally:
        gate.set()  # always release the blocked storage call, even if an assertion above fails
    assert (await closing).released


async def test_calibration_assets_and_latency_are_persisted_from_explicit_paths_only(tmp_path, own):
    """Supplied assets are copied and hashed, latency is stored, a missing optional asset is reported."""
    intrinsics = tmp_path / "src_intrinsics.json"
    intrinsics.write_text('{"fx": 600.0}')
    optional = tmp_path / "does_not_exist.json"
    base = make_config(tmp_path / "out")
    config = type(base)(
        **{
            **base.__dict__,
            "calibration_assets": (
                CalibrationAsset("head_camera_intrinsics.json", intrinsics),
                CalibrationAsset("wrist_offsets.json", optional, required=False),
            ),
            "latency": {CAMERA: 0.0875, "pico_controller": 0.05},
        }
    )
    service = own(RecordingService(config))
    session = await service.open_session()
    assert (session.session_path / "head_camera_intrinsics.json").read_text() == '{"fx": 600.0}'
    assert not (session.session_path / "wrist_offsets.json").exists()
    assert any("wrist_offsets.json" in issue for issue in service.snapshot().issues)
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(6))
    result = await service.stop_episode(episode.episode_id)
    attrs = dict(zarr.open_group(str(result.final_path), mode="r").attrs)
    assert attrs["latency"] == {CAMERA: 0.0875, "pico_controller": 0.05}
    assert set(attrs["calibration_assets"]) == {"head_camera_intrinsics.json"}
    await service.close()


@pytest.mark.parametrize(
    "assets,message",
    [
        (lambda p: (CalibrationAsset("a.json", p / "missing.json"),), "required calibration asset"),
        (lambda p: (CalibrationAsset("../escape.json", p / "x", required=False),), "safe file name"),
        (lambda p: (CalibrationAsset("episode_9", p / "x", required=False),), "collides"),
        (
            lambda p: (CalibrationAsset("n.json", p / "x", False), CalibrationAsset("n.json", p / "x", False)),
            "duplicate",
        ),
    ],
)
async def test_bad_calibration_configuration_is_refused_before_any_effect(tmp_path, assets, message):
    """Missing required, unsafe, colliding or duplicate calibration assets raise ConfigurationError at open."""
    base = make_config(tmp_path / "out")
    config = type(base)(**{**base.__dict__, "calibration_assets": assets(tmp_path)})
    with pytest.raises(ConfigurationError, match=message):
        await RecordingService(config).open_session()
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*"))


async def _crashed_with_asset(tmp_path, monkeypatch, content='{"fx": 600.0}'):
    """A real crashed recorder (after the check) whose session preserved a required calibration asset."""
    asset = tmp_path / "intrinsics_source.json"
    asset.write_text(content)
    monkeypatch.setenv("CRASH_CALIBRATION", str(asset))
    child = run_child(tmp_path / "out", "after_checked")
    session_id = child.stdout.readline().strip()
    assert child.wait(timeout=60) == 9
    base = make_config(tmp_path / "out")
    config = type(base)(
        **{**base.__dict__, "calibration_assets": (CalibrationAsset("head_camera_intrinsics.json", asset),)}
    )
    return asset, session_id, config


async def test_recovery_uses_the_preserved_session_copy_not_the_original_input(tmp_path, monkeypatch, own):
    """Deleting or changing the original calibration file does not block reopening or recovering the session."""
    asset, session_id, config = await _crashed_with_asset(tmp_path, monkeypatch)
    session_copy = tmp_path / "out" / session_id / "head_camera_intrinsics.json"
    preserved = session_copy.read_bytes()
    asset.unlink()  # the workstation's original input is gone
    service = own(RecordingService(config))
    await service.open_session(session_id)
    (pending,) = service.pending_episodes()
    assert pending.recoverable
    result = await service.recover_episode(pending.episode_id)
    assert result.disposition is Disposition.KEPT and session_copy.read_bytes() == preserved
    attrs = dict(zarr.open_group(str(result.final_path), mode="r").attrs)
    assert attrs["calibration_assets"] == {"head_camera_intrinsics.json": hashlib.sha256(preserved).hexdigest()}
    await service.close()

    asset.write_text("completely different content")  # a changed original is ignored too
    again = own(RecordingService(config))
    await again.open_session(session_id)
    assert session_copy.read_bytes() == preserved
    await again.close()


async def test_missing_or_corrupt_session_copy_of_a_required_asset_is_refused(tmp_path, monkeypatch, own):
    """The session's own copy is verified on reopen: corrupt or missing copies refuse the session."""
    _, session_id, config = await _crashed_with_asset(tmp_path, monkeypatch)
    session_copy = tmp_path / "out" / session_id / "head_camera_intrinsics.json"
    original = session_copy.read_bytes()
    session_copy.write_bytes(b"tampered")
    with pytest.raises(ConfigurationError, match="session copy"):
        await RecordingService(config).open_session(session_id)
    session_copy.unlink()
    with pytest.raises(ConfigurationError, match="session copy"):
        await RecordingService(config).open_session(session_id)
    session_copy.write_bytes(original)
    service = own(RecordingService(config))
    await service.open_session(session_id)
    await service.close()


async def _crashed_checked(tmp_path):
    """A real crashed recorder with a CHECKED pending episode; returns (session id, session, pending, config)."""
    child = run_child(tmp_path / "out", "after_checked")
    session_id = child.stdout.readline().strip()
    assert child.wait(timeout=60) == 9
    session = tmp_path / "out" / session_id
    return session_id, session, session / "episode_0.pending", make_config(tmp_path / "out")


def _digest(root):
    """(size, mtime_ns) of every file under root, plus the root attributes file text."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


async def test_noncanonical_manifest_paths_are_unreadable_and_touch_nothing(tmp_path, own):
    """A manifest pointing at an unrelated directory is unreadable and blocks recording; nothing changes."""
    session_id, session, pending_dir, config = await _crashed_checked(tmp_path)
    outside = tmp_path / "unrelated"
    zarr.open_group(str(outside), mode="w", zarr_format=3).attrs["keep"] = "me"
    before = _digest(outside)
    manifest_file = session / ".recording" / "episode_0.json"
    data = json.loads(manifest_file.read_text())
    data["pending_path"] = "../../unrelated"
    manifest_file.write_text(json.dumps(data))

    service = own(RecordingService(config))
    await service.open_session(session_id)
    assert not service.pending_episodes() and any("non-canonical" in i for i in service.snapshot().issues)
    with pytest.raises(StateError, match="unreadable"):
        await service.start_episode("blocked")
    assert _digest(outside) == before and dict(zarr.open_group(str(outside), mode="r").attrs) == {"keep": "me"}
    await service.close()


async def test_symlinked_or_wrongly_owned_pending_output_is_refused_and_untouched(tmp_path, own):
    """Recovery refuses a symlinked pending directory and one stamped for another episode, writing nothing."""
    session_id, session, pending_dir, config = await _crashed_checked(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(pending_dir, elsewhere)
    shutil.rmtree(pending_dir)
    pending_dir.symlink_to(elsewhere, target_is_directory=True)
    before = _digest(elsewhere)

    service = own(RecordingService(config))
    await service.open_session(session_id)
    (pending,) = service.pending_episodes()
    result = await service.recover_episode(pending.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and any("symlink" in i for i in result.issues)
    assert _digest(elsewhere) == before
    await service.close()

    pending_dir.unlink()
    shutil.move(str(elsewhere), str(pending_dir))
    zarr.open_group(str(pending_dir), mode="r+").attrs["recording_episode_id"] = "someone-else"
    before = _digest(pending_dir)
    attrs_before = dict(zarr.open_group(str(pending_dir), mode="r").attrs)
    again = own(RecordingService(config))
    await again.open_session(session_id)
    (pending,) = again.pending_episodes()
    result = await again.recover_episode(pending.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and any("not owned" in i for i in result.issues)
    assert _digest(pending_dir) == before
    assert dict(zarr.open_group(str(pending_dir), mode="r").attrs) == attrs_before
    discard = await again.discard_pending(pending.episode_id)
    assert discard.disposition is Disposition.RECOVERY_REQUIRED and pending_dir.is_dir()
    await again.close()


async def test_a_store_that_silently_loses_a_row_is_caught_by_the_count_check(tmp_path, monkeypatch, own):
    """If the real store drops a row without raising, stop reports a count mismatch and never finalizes."""
    real_append, seen = EpisodeStore.append_numeric, {"n": 0}

    def drops_the_second_row(self, *args, **kwargs):
        seen["n"] += 1
        if seen["n"] != 2:
            real_append(self, *args, **kwargs)

    monkeypatch.setattr(EpisodeStore, "append_numeric", drops_the_second_row)
    service, session, episode = await _recording(own, tmp_path)
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and any(
        "count mismatch" in i for i in result.issues
    )
    assert not (session.session_path / "episode_0").exists()
    await service.close()


async def test_unconvertible_intrinsics_are_rejected_without_consuming_an_index(tmp_path, own):
    """A real frame with malformed intrinsics is rejected, counts stay put, and valid data still finishes."""
    service, _, episode = await _recording(own, tmp_path)
    before = {c.source_name: c.accepted for c in service.snapshot().counts}
    frame = [s for s in samples(6) if s.source == CAMERA][5]  # the real next frame after the five submitted
    for bad in (np.array([["a", "b", "c"]] * 3), np.full((3, 3), np.nan), np.eye(2)):
        with pytest.raises(ValidationError):
            service.submit_sample(
                episode.episode_id,
                VideoSample(CAMERA, frame.image, frame.pixel_format, frame.capture_time, frame.receive_time, bad),
            )
    assert {c.source_name: c.accepted for c in service.snapshot().counts} == before
    good = VideoSample(
        CAMERA, frame.image, frame.pixel_format, frame.capture_time, frame.receive_time, frame.intrinsics
    )
    assert service.submit_sample(episode.episode_id, good).admission_index == before[CAMERA]
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT
    assert {c.source_name: (c.accepted, c.written) for c in result.counts}[CAMERA] == (before[CAMERA] + 1,) * 2


async def test_required_asset_that_vanishes_after_the_precheck_is_refused_not_downgraded(
    tmp_path, monkeypatch, own
):
    """If a required input disappears between validation and copy, opening fails and leaves no session behind."""
    asset = tmp_path / "intrinsics.json"
    asset.write_text("{}")
    base = make_config(tmp_path / "out")
    config = type(base)(**{**base.__dict__, "calibration_assets": (CalibrationAsset("intrinsics.json", asset),)})
    real_is_file, seen = Path.is_file, {"n": 0}

    def vanishing(self):
        if self == asset:
            seen["n"] += 1
            return seen["n"] == 1  # present at the up-front check, gone when it is copied
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", vanishing)
    with pytest.raises(ConfigurationError, match="not available"):
        await RecordingService(config).open_session()
    monkeypatch.undo()
    assert not list((tmp_path / "out").glob("*"))
    service = own(RecordingService(config))  # nothing was left locked or half-created
    await service.open_session()
    await service.close()


async def test_failed_new_session_setup_is_rolled_back_while_the_session_lock_is_still_held(
    tmp_path, monkeypatch, own
):
    """The directory a failed open created is removed before its lock is released, so nobody can take it first."""
    asset = tmp_path / "intrinsics.json"
    asset.write_text("{}")
    base = make_config(tmp_path / "out")
    config = type(base)(**{**base.__dict__, "calibration_assets": (CalibrationAsset("intrinsics.json", asset),)})
    real_rmtree, observed = shutil.rmtree, {}

    def lock_still_held_during_rollback(path, *args, **kwargs):
        with pytest.raises(BlockingIOError):
            SessionLock(Path(path))  # another owner cannot acquire the session mid-rollback
        observed["held"] = True
        real_rmtree(path, *args, **kwargs)

    def failing_copy(*_args, **_kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr(shutil, "copy2", failing_copy)
    monkeypatch.setattr(shutil, "rmtree", lock_still_held_during_rollback)
    with pytest.raises(OSError, match="copy failed"):
        await RecordingService(config).open_session()
    monkeypatch.undo()
    assert observed == {"held": True} and not list((tmp_path / "out").glob("*"))
    service = own(RecordingService(make_config(tmp_path / "out")))
    await service.open_session()


async def _fail_terminal_write_then_recover(tmp_path, monkeypatch, own, phase: str, corrupt: bool):
    """Stop a real take while its terminal manifest write fails; return (service, session, pending)."""
    real_write, real_check, state = impl.write_manifest, impl.check_episode, {"fail": True}

    def failing_terminal_write(session_path, manifest):
        if manifest.storage_phase == phase and state["fail"]:
            raise OSError("manifest write failed")
        return real_write(session_path, manifest)

    def corrupt_then_check(pending_dir, *args):
        shutil.rmtree(pending_dir / "pico_controller" / "head_pose")  # a copy of real output, damaged
        return real_check(pending_dir, *args)

    monkeypatch.setattr(impl, "write_manifest", failing_terminal_write)
    if corrupt:
        monkeypatch.setattr(impl, "check_episode", corrupt_then_check)
    service, session, episode = await _recording(own, tmp_path)
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and any(
        "manifest write failed" in i for i in result.issues
    )
    (pending,) = service.pending_episodes()
    on_disk = json.loads((session.session_path / ".recording" / "episode_0.json").read_text())
    assert pending.storage_phase == on_disk["storage_phase"] == "CHECKED" and pending.recoverable
    state["fail"] = False
    return service, session, pending


async def test_failed_finalized_checkpoint_stays_recoverable_and_same_service_retry_completes(
    tmp_path, monkeypatch, own
):
    """If the FINALIZED write fails after the rename, memory stays at CHECKED and a retry finishes it once."""
    service, session, pending = await _fail_terminal_write_then_recover(
        tmp_path, monkeypatch, own, "FINALIZED", False
    )
    final = session.session_path / "episode_0"
    before = _digest(final)  # the rename already happened
    result = await service.recover_episode(pending.episode_id)
    assert result.disposition is Disposition.KEPT and result.final_path == final
    assert _digest(final) == before, "recovery must not rewrite the already-final data"
    on_disk = json.loads((session.session_path / ".recording" / "episode_0.json").read_text())
    assert on_disk["storage_phase"] == "FINALIZED" and not service.pending_episodes()
    reopened = own(RecordingService(make_config(tmp_path / "out")))
    await service.close()
    await reopened.open_session(session.session_id)
    assert not reopened.pending_episodes()


async def test_failed_rejected_checkpoint_stays_recoverable_and_same_service_retry_completes(
    tmp_path, monkeypatch, own
):
    """The analogous REJECTED write failure also leaves a CHECKED, recoverable episode that retries to REJECTED."""
    service, session, pending = await _fail_terminal_write_then_recover(
        tmp_path, monkeypatch, own, "REJECTED", True
    )
    result = await service.recover_episode(pending.episode_id)
    assert result.disposition is Disposition.REJECTED and result.quality.outcome is Outcome.FAILED
    on_disk = json.loads((session.session_path / ".recording" / "episode_0.json").read_text())
    assert on_disk["storage_phase"] == "REJECTED" and not service.pending_episodes()
    assert not (session.session_path / "episode_0").exists()


async def test_service_keeps_its_own_configuration_snapshot(tmp_path, own):
    """Mutating the caller's config (latency, encoder options) afterwards changes nothing the service does."""
    base = make_config(tmp_path / "out")
    video = type(base.video_sources[0])(**{**base.video_sources[0].__dict__, "encoder_options": {"crf": 18}})
    config = type(base)(**{**base.__dict__, "video_sources": (video,), "latency": {CAMERA: 0.0875}})
    digest_before = impl._digest(copy.deepcopy(config))
    service = own(RecordingService(config))
    await service.open_session()
    first = await service.start_episode("t")
    config.latency[CAMERA] = 9.9  # the caller reuses and edits its own objects
    video.encoder_options["crf"] = 40
    await submit_all(service, first.episode_id, samples(6))
    first_result = await service.stop_episode(first.episode_id)
    second = await service.start_episode("t2")
    await submit_all(service, second.episode_id, samples(6))
    second_result = await service.stop_episode(second.episode_id)
    for result in (first_result, second_result):
        attrs = dict(zarr.open_group(str(result.final_path), mode="r").attrs)
        assert attrs["latency"] == {CAMERA: 0.0875}
        assert attrs["video_encoder_options"][CAMERA]["crf"] == 18
        assert attrs["config_digest"] == digest_before


async def test_unsupported_manifest_phase_blocks_recording_and_leaves_data_untouched(tmp_path, own):
    """An unsupported storage_phase is reported and refuses a new take; episode and manifest stay as they were."""
    session_id, session, pending_dir, config = await _crashed_checked(tmp_path)
    manifest_file = session / ".recording" / "episode_0.json"
    data = json.loads(manifest_file.read_text())
    data["storage_phase"] = "CHECKEDD"
    manifest_file.write_text(json.dumps(data))
    before, manifest_bytes = _digest(pending_dir), manifest_file.read_bytes()

    service = own(RecordingService(config))
    await service.open_session(session_id)
    assert not service.pending_episodes() and any(
        "unsupported storage_phase" in i for i in service.snapshot().issues
    )
    with pytest.raises(StateError, match="unreadable"):
        await service.start_episode("blocked")
    assert _digest(pending_dir) == before and manifest_file.read_bytes() == manifest_bytes


async def test_failed_sealed_checkpoint_fails_clearly_and_the_take_can_only_be_discarded(
    tmp_path, monkeypatch, own
):
    """If the SEALED manifest cannot be saved, the take fails clearly, is not recoverable, and can be discarded."""
    real_write = impl.write_manifest

    def failing_sealed_write(session_path, manifest):
        if manifest.storage_phase == "SEALED":
            raise OSError("manifest write failed")
        return real_write(session_path, manifest)

    monkeypatch.setattr(impl, "write_manifest", failing_sealed_write)
    service, session, episode = await _recording(own, tmp_path)
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    assert any("SEALED manifest could not be saved" in i and "discard it" in i for i in result.issues)
    (pending,) = service.pending_episodes()
    assert pending.storage_phase == "WRITING" and not pending.recoverable
    with pytest.raises(StateError, match="cannot be recovered; discard it"):
        await service.recover_episode(pending.episode_id)
    assert (await service.discard_pending(pending.episode_id)).disposition is Disposition.DISCARDED
    assert not (session.session_path / "episode_0.pending").exists()
