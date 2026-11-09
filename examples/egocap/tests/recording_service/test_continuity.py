# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-source continuity evidence of a take: noted per accepted row, stored in the manifest, read by alignment."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

import recording_service.recording_service as impl
from egocap.recording.aligned_writer import AlignedWriteError, write_aligned_episode
from egocap.recording.episode_layout import manifest_path, read_manifest
from recording_service import (
    Disposition,
    FieldSpec,
    NumericSample,
    NumericSourceSpec,
    PixelFormat,
    QcConfig,
    RecordingService,
    ServiceConfig,
    SubmissionStatus,
    VideoSample,
    VideoSourceSpec,
)

MAPPING = "clock_mapping"


def _config(tmp_path: Path, **overrides) -> ServiceConfig:
    return ServiceConfig(
        output_root=tmp_path / "out",
        numeric_sources=(
            NumericSourceSpec("pose", (FieldSpec("v", "float64"),)),
            NumericSourceSpec("idle", (FieldSpec("v", "float64"),)),
        ),
        qc=QcConfig(required_sources=("pose",)),
        **overrides,
    )


def _pose(i: int, continuity):
    return NumericSample("pose", float(i), float(i), {"v": float(i)}, continuity)


async def _record(tmp_path: Path, own, continuities, **overrides):
    """Record one take of ``pose`` rows carrying the given continuity values; returns (session, result)."""
    service = own(RecordingService(_config(tmp_path, **overrides)))
    session = await service.open_session()
    episode = await service.start_episode("t")
    for i, continuity in enumerate(continuities):
        assert service.submit_sample(episode.episode_id, _pose(i, continuity)).status is SubmissionStatus.ACCEPTED
    result = await service.stop_episode(episode.episode_id)
    return session, result


def _stored(session, index: int = 0):
    return read_manifest(manifest_path(session.session_path, index))


async def _evidence(tmp_path: Path, own, continuities):
    session, _ = await _record(tmp_path, own, continuities)
    return _stored(session).continuity["pose"]


async def test_a_constant_identity_is_known_with_no_breaks(tmp_path, own):
    entry = await _evidence(tmp_path, own, [(MAPPING, 1)] * 4)
    assert entry == {"scope": MAPPING, "identity": 1, "breaks": [], "known": True}


async def test_a_change_of_identity_is_a_break_at_the_accepted_row(tmp_path, own):
    entry = await _evidence(tmp_path, own, [(MAPPING, 1)] * 3 + [(MAPPING, 2)] * 2)
    assert entry["breaks"] == [3] and entry["known"] is True and entry["identity"] == 1


async def test_two_changes_are_two_breaks_and_a_return_to_an_old_identity_is_one_too(tmp_path, own):
    ids = [1, 1, 2, 2, 1, 1]
    entry = await _evidence(tmp_path, own, [(MAPPING, i) for i in ids])
    assert entry["breaks"] == [2, 4] and entry["known"] is True


async def test_str_and_int_identities_are_stored_as_given(tmp_path, own):
    entry = await _evidence(tmp_path, own, [("capture_session", "a"), ("capture_session", "b")])
    assert entry == {"scope": "capture_session", "identity": "a", "breaks": [1], "known": True}


async def test_a_row_without_evidence_makes_the_source_unknown(tmp_path, own):
    entry = await _evidence(tmp_path, own, [(MAPPING, 1), None, (MAPPING, 1)])
    assert entry["known"] is False and entry["breaks"] == []


async def test_a_first_row_without_evidence_makes_the_source_unknown(tmp_path, own):
    entry = await _evidence(tmp_path, own, [None, (MAPPING, 1)])
    assert entry["known"] is False


async def test_a_change_of_scope_makes_the_source_unknown_not_a_break(tmp_path, own):
    entry = await _evidence(tmp_path, own, [(MAPPING, 1), ("capture_session", 2)])
    assert entry["known"] is False and entry["breaks"] == [] and entry["scope"] == MAPPING


async def test_a_source_that_accepted_nothing_is_recorded_unknown(tmp_path, own):
    session, _ = await _record(tmp_path, own, [(MAPPING, 1)] * 2)
    assert _stored(session).continuity["idle"] == {"scope": None, "identity": None, "breaks": [], "known": False}


async def test_the_private_last_identity_never_reaches_the_manifest(tmp_path, own):
    session, _ = await _record(tmp_path, own, [(MAPPING, 1), (MAPPING, 2)])
    raw = json.loads(manifest_path(session.session_path, 0).read_text())
    assert all("last" not in entry for entry in raw["continuity"].values())


async def test_a_full_result_notes_nothing_and_the_retried_sample_is_noted_once(tmp_path, own):
    service = own(RecordingService(_config(tmp_path, queue_capacity=2)))
    await service.open_session()
    episode = await service.start_episode("t")
    eid = episode.episode_id
    for i in range(2):
        assert service.submit_sample(eid, _pose(i, (MAPPING, 1))).status is SubmissionStatus.ACCEPTED
    changed = _pose(2, (MAPPING, 2))
    for _ in range(3):  # a refused offer, repeated, leaves no trace
        assert service.submit_sample(eid, changed).status is SubmissionStatus.FULL
    assert service._take.continuity_record()["pose"]["breaks"] == []

    while (result := service.submit_sample(eid, changed)).status is SubmissionStatus.FULL:
        await asyncio.sleep(0)
    assert result.admission_index == 2
    assert service._take.continuity_record()["pose"]["breaks"] == [2]
    stopped = await service.stop_episode(eid)
    assert {c.source_name: c.accepted for c in stopped.counts}["pose"] == 3


async def test_stored_breaks_are_indices_of_accepted_rows(tmp_path, own):
    session, result = await _record(tmp_path, own, [(MAPPING, i // 2) for i in range(8)])
    entry = _stored(session).continuity["pose"]
    accepted = {c.source_name: c.accepted for c in result.counts}["pose"]
    assert entry["breaks"] == [2, 4, 6] and all(b < accepted for b in entry["breaks"])


async def test_evidence_survives_a_retained_take_and_its_recovery(tmp_path, own, monkeypatch):
    """A take stopped before the final rename keeps its sealed evidence; recovery finishes without changing it."""
    real_finalize = impl.finalize_episode_dir

    def refuse(*_a, **_k):
        raise OSError("interrupted")

    monkeypatch.setattr(impl, "finalize_episode_dir", refuse)
    first = own(RecordingService(_config(tmp_path)), may_stay_open=True)
    session = await first.open_session()
    episode = await first.start_episode("t")
    for i, identity in enumerate([1, 1, 2]):
        first.submit_sample(episode.episode_id, _pose(i, (MAPPING, identity)))
    result = await first.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    await first.close()  # releases the session lock, as a restart would
    before = _stored(session).continuity
    assert before["pose"]["breaks"] == [2]

    monkeypatch.setattr(impl, "finalize_episode_dir", real_finalize)
    service = own(RecordingService(_config(tmp_path)))
    await service.open_session(session.session_id)
    (pending,) = service.pending_episodes()
    recovered = await service.recover_episode(pending.episode_id)
    assert recovered.disposition is Disposition.KEPT
    assert _stored(session).continuity == before


async def test_a_manifest_that_recorded_no_evidence_stays_none(tmp_path, own):
    session, _ = await _record(tmp_path, own, [(MAPPING, 1)])
    file = manifest_path(session.session_path, 0)
    data = json.loads(file.read_text())
    del data["continuity"]
    data["schema_version"] = 3
    file.write_text(json.dumps(data))
    assert read_manifest(file).continuity is None


# --- a recorded episode, aligned only when its stored evidence allows it ------------------------


def _alignable_config(tmp_path: Path) -> ServiceConfig:
    return ServiceConfig(
        output_root=tmp_path / "out",
        numeric_sources=(
            NumericSourceSpec("pico_controller", (FieldSpec("left_trigger", "float64"),), "PicoTrackingFrame"),
        ),
        video_sources=(VideoSourceSpec("cam", 64, 48, 30.0),),
        qc=QcConfig(required_sources=("pico_controller", "cam")),
    )


async def _record_alignable(tmp_path: Path, own, camera_identities):
    """A tiny episode: camera frames (one identity per frame) bracketed by tracking rows, all clock_mapping."""
    service = own(RecordingService(_alignable_config(tmp_path)))
    session = await service.open_session()
    episode = await service.start_episode("t")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    items = []
    for i, identity in enumerate(camera_identities):
        t = 1.0 + i / 30
        items.append(VideoSample("cam", frame, PixelFormat.RGB8, t, t, np.eye(3), (MAPPING, identity)))
    for i in range(-2, 4 * len(camera_identities)):
        t = 1.0 + i / 60
        items.append(NumericSample("pico_controller", t, t, {"left_trigger": t}, (MAPPING, 1)))
    for item in sorted(items, key=lambda x: x.capture_time):
        while service.submit_sample(episode.episode_id, item).status is SubmissionStatus.FULL:
            await asyncio.sleep(0)
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT
    return session


def _align(session, tmp_path: Path):
    manifest = _stored(session)
    # The service writes each episode as its own store; the aligner reads episodes as groups of one session root.
    zarr.open_group(str(session.session_path), mode="a")
    qualified = dict.fromkeys(("cam", "pico_controller"), True)
    return write_aligned_episode(
        session.session_path,
        tmp_path / "aligned",
        0,
        "cam",
        qualified=qualified,
        continuity=manifest.continuity,
    )


async def test_an_episode_with_constant_evidence_aligns_from_its_stored_manifest(tmp_path, own):
    session = await _record_alignable(tmp_path, own, [1] * 5)
    manifest = _stored(session)
    assert all(e["known"] and not e["breaks"] for e in manifest.continuity.values())
    aligned = _align(session, tmp_path)
    assert aligned.valid_indices.any()
    assert zarr.open_group(str(tmp_path / "aligned"), mode="r")["episode_0"]["cam"] is not None


async def test_an_episode_with_a_recorded_break_is_refused_and_writes_nothing(tmp_path, own):
    session = await _record_alignable(tmp_path, own, [1, 1, 1, 2, 2])
    assert _stored(session).continuity["cam"]["breaks"] == [3]
    with pytest.raises(AlignedWriteError, match=r"cam: has a restart inside the episode \(rows \[3\]\)"):
        _align(session, tmp_path)
    assert not (tmp_path / "aligned").exists()


# --- malformed evidence is refused before any bookkeeping --------------------------------------------------


def _video(i: int, continuity):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return VideoSample("cam", frame, PixelFormat.RGB8, float(i), float(i), np.eye(3), continuity)


MALFORMED = [
    [MAPPING, 1],
    (MAPPING, 1, 2),
    (MAPPING,),
    (3, 1),
    (MAPPING, True),
    (MAPPING, 1.5),
    (MAPPING, None),
    "clock_mapping",
]


@pytest.mark.parametrize("bad", MALFORMED, ids=repr)
async def test_malformed_continuity_is_refused_and_changes_nothing(tmp_path, own, bad):
    service = own(RecordingService(_config(tmp_path)))
    session = await service.open_session()
    episode = await service.start_episode("t")
    eid = episode.episode_id
    assert service.submit_sample(eid, _pose(0, (MAPPING, 1))).admission_index == 0
    take = service._take
    accepted, queued = dict(take.accepted), list(take.queue)
    evidence = json.dumps(take.continuity_record(), sort_keys=True)

    with pytest.raises(impl.ValidationError, match="continuity"):
        service.submit_sample(eid, _pose(1, bad))

    assert take.accepted == accepted and list(take.queue) == queued
    assert json.dumps(take.continuity_record(), sort_keys=True) == evidence
    # The next valid sample takes the index the rejected one would have taken.
    assert service.submit_sample(eid, _pose(1, (MAPPING, 1))).admission_index == 1
    await service.stop_episode(eid)
    assert _stored(session).continuity["pose"] == {"scope": MAPPING, "identity": 1, "breaks": [], "known": True}


async def test_malformed_continuity_on_the_first_row_leaves_the_source_unaccepted(tmp_path, own):
    service = own(RecordingService(_config(tmp_path)))
    await service.open_session()
    episode = await service.start_episode("t")
    with pytest.raises(impl.ValidationError, match="continuity"):
        service.submit_sample(episode.episode_id, _pose(0, (MAPPING, 2.0)))
    assert service._take.accepted["pose"] == 0 and not service._take.queue
    assert service._take.continuity_record()["pose"]["known"] is False
    assert service.submit_sample(episode.episode_id, _pose(0, (MAPPING, 2))).admission_index == 0


def test_copy_keeps_the_continuity_of_a_numeric_sample():
    copied = RecordingService._copy(_pose(0, (MAPPING, "epoch-1")))
    assert isinstance(copied, NumericSample) and copied.continuity == (MAPPING, "epoch-1")


def test_copy_keeps_the_continuity_of_a_video_sample():
    copied = RecordingService._copy(_video(0, (MAPPING, 5)))
    assert isinstance(copied, VideoSample) and copied.continuity == (MAPPING, 5)


async def test_the_queued_copy_carries_the_submitted_continuity(tmp_path, own):
    service = own(RecordingService(_config(tmp_path)))
    await service.open_session()
    episode = await service.start_episode("t")
    service.submit_sample(episode.episode_id, _pose(0, (MAPPING, 9)))
    # No await since the submission, so the worker has not drained the queue yet.
    (queued,) = service._take.queue
    assert queued.continuity == (MAPPING, 9)
