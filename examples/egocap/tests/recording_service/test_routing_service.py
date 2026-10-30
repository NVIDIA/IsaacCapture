"""Local routing, output numbering and group totals through the real service, on a real recorded episode.

Samples are the stored rows and frames of an actual episode. Defects are explicit constructions: a camera is made
too slow by stretching its recorded timestamps (a copy of the real frames), and a pose is held still by repeating
its first value.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from real import CAMERA, make_config, needs_real_episode, samples, submit_all

import recording_service.recording_service as impl
from egocap.recording.episode_layout import read_manifest
from recording_service import (
    CaptureQcConfig,
    ConfigurationError,
    Disposition,
    QcConfig,
    QcDecision,
    RecordingService,
    RoutingConfig,
    VideoSample,
)

pytestmark = needs_real_episode
ROWS = 20
POSE = ("pico_controller", "head_pose")
ROUTING = RoutingConfig(studio_by_task=(("stack blocks", "Kitchen"), ("set the table", "Dining Room")))


def _duration() -> float:
    times = [s.capture_time for s in samples(ROWS) if s.source == POSE[0]]
    return times[-1] - times[0]


def _config(tmp_path, routing=ROUTING, capture=True):
    qc = QcConfig(
        required_sources=("pico_controller", CAMERA),
        capture=CaptureQcConfig(
            camera_expected_fps=((CAMERA, 1.0),), pose_streams=(POSE,), freeze_seconds=_duration() / 2
        )
        if capture
        else None,
    )
    return make_config(tmp_path / "out", qc=qc, routing=routing)


def _slow_camera(items):
    """Copies of the real frames with their timestamps stretched 1000x: a camera far below its expected rate."""
    return [
        dataclasses.replace(s, capture_time=s.capture_time * 1000, receive_time=s.receive_time * 1000)
        if isinstance(s, VideoSample)
        else s
        for s in items
    ]


def _frozen(items):
    first = next(s.fields[POSE[1]] for s in items if s.source == POSE[0])
    return [
        dataclasses.replace(s, fields={**s.fields, POSE[1]: first}) if s.source == POSE[0] else s for s in items
    ]


async def _take(service, task, items=None):
    episode = await service.start_episode(task, "op")
    await submit_all(service, episode.episode_id, items or samples(ROWS))
    return await service.stop_episode(episode.episode_id)


def _attrs(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text())["attributes"]


async def _service(config, own):
    service = own(RecordingService(config))
    session = await service.open_session()
    return service, session


async def test_a_kept_take_is_routed_to_its_studio_group_with_native_totals(tmp_path, own):
    """The take ends at <root>/Kitchen/<timestamp>/episode_0 and its group holds the dataset-level totals."""
    service, session = await _service(_config(tmp_path), own)
    result = await _take(service, "stack blocks")
    assert result.disposition is Disposition.KEPT and result.output_index == 0
    final = result.final_path
    assert final.name == "episode_0" and final.parent.parent.name == "Kitchen"
    assert final.parent.parent.parent == tmp_path / "out"
    assert not any(
        p.name.startswith("episode_") for p in session.session_path.iterdir()
    )  # nothing left in the session
    attrs = _attrs(final)
    assert attrs["studio"] == "Kitchen" and attrs["output_index"] == 0 and attrs["capture_qc_decision"] == "PASSED"
    assert attrs["freeze_check_passed"] is True and attrs["freeze_check_issues"] == []
    group = _attrs(final.parent)
    assert group["num_episodes"] == 1 and group["total_data_time_hours"] > 0
    assert group["episode_lengths"]["0"] == attrs["episode_length"] and group["discarded_episode_indices"] == []
    assert group["task_description"] == "stack blocks" and group["studio"] == "Kitchen"
    manifest = read_manifest(next((session.session_path / ".recording").glob("episode_*.json")))
    assert manifest.storage_phase == "FINALIZED" and manifest.finalization == "KEEP"
    assert manifest.studio == "Kitchen" and manifest.assignment == {
        "group": manifest.assignment["group"],
        "slot": 0,
    }
    assert result.final_path == tmp_path / "out" / manifest.assignment["group"] / "episode_0"


async def test_takes_of_one_task_share_a_group_and_number_their_kept_output_consecutively(tmp_path, own):
    service, _ = await _service(_config(tmp_path), own)
    first = await _take(service, "stack blocks")
    second = await _take(service, "stack blocks")
    assert second.final_path.parent == first.final_path.parent
    assert (first.output_index, second.output_index) == (0, 1) and (first.episode_index, second.episode_index) == (
        0,
        1,
    )
    assert _attrs(first.final_path.parent)["num_episodes"] == 2


async def test_a_rejected_attempt_takes_no_output_slot_but_a_flagged_bad_one_does(tmp_path, own):
    """Camera rejection removes the attempt without using a slot; a kept bad episode is real output and counts."""
    service, _ = await _service(_config(tmp_path), own)
    kept = await _take(service, "stack blocks")
    rejected = await _take(service, "stack blocks", _slow_camera(samples(ROWS)))
    bad = await _take(service, "stack blocks", _frozen(samples(ROWS)))
    after = await _take(service, "stack blocks")
    assert rejected.disposition is Disposition.REJECTED and rejected.output_index is None
    assert rejected.qc_decision is QcDecision.REJECTED_CAMERA and rejected.final_path is None
    assert bad.disposition is Disposition.KEPT and bad.qc_decision is QcDecision.FLAGGED_BAD
    assert [r.output_index for r in (kept, bad, after)] == [0, 1, 2]
    assert [r.episode_index for r in (kept, rejected, bad, after)] == [0, 1, 2, 3]  # attempt identity never skips
    group = _attrs(kept.final_path.parent)
    assert group["num_episodes"] == 3 and group["discarded_episode_indices"] == [1]
    assert "no motion for" in group["discarded_reasons"]["1"]
    assert _attrs(bad.final_path)["freeze_check_passed"] is False and _attrs(bad.final_path)["bad_episode"] is True


async def test_tasks_route_to_their_studio_or_the_unmatched_bucket(tmp_path, own):
    service, _ = await _service(_config(tmp_path), own)
    kitchen = await _take(service, "stack blocks")
    dining = await _take(service, "set the table")
    other = await _take(service, "an unlisted task")
    studios = [r.final_path.parent.parent.name for r in (kitchen, dining, other)]
    assert studios == ["Kitchen", "Dining_Room", "_unmatched"]
    assert len({r.final_path.parent for r in (kitchen, dining, other)}) == 3
    assert [r.output_index for r in (kitchen, dining, other)] == [0, 0, 0]  # each group numbers its own output


async def test_the_session_studio_override_routes_every_take(tmp_path, own):
    routing = RoutingConfig(session_studio="Living Room", studio_by_task=ROUTING.studio_by_task)
    service, _ = await _service(_config(tmp_path, routing), own)
    result = await _take(service, "stack blocks")
    assert result.final_path.parent.parent.name == "Living_Room"


async def test_the_destination_is_frozen_when_the_take_starts(tmp_path, own):
    """The studio is in the manifest from the start, before any data, so nothing later can reroute the take."""
    service, session = await _service(_config(tmp_path), own)
    episode = await service.start_episode("stack blocks", "op")
    manifest = read_manifest(next((session.session_path / ".recording").glob("episode_*.json")))
    assert manifest.studio == "Kitchen" and manifest.assignment is None and manifest.finalization is None
    await service.discard_episode(episode.episode_id)


async def test_a_failure_after_the_rename_recovers_without_double_counting(tmp_path, monkeypatch, own):
    """Renamed into its group but the FINALIZED checkpoint was lost: recovery finds it and totals stay right."""
    real = impl.write_manifest

    def failing_at_finalized(path, manifest):
        if manifest.storage_phase == "FINALIZED":
            raise OSError("cannot save the FINALIZED checkpoint")
        return real(path, manifest)

    monkeypatch.setattr(impl, "write_manifest", failing_at_finalized)
    service, _ = await _service(_config(tmp_path), own)
    result = await _take(service, "stack blocks")
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    (pending,) = service.pending_episodes()
    assert not pending.pending_path.exists()  # already renamed into its group
    monkeypatch.undo()
    recovered = await service.recover_episode(pending.episode_id)
    assert recovered.disposition is Disposition.KEPT and recovered.output_index == 0
    assert _attrs(recovered.final_path.parent)["num_episodes"] == 1


async def test_a_failed_totals_refresh_never_strands_a_published_take_and_the_next_take_repairs_it(
    tmp_path, monkeypatch, own
):
    """The take is FINALIZED before its group's totals are refreshed, so a refresh failure is only a warning."""
    real = impl.refresh_group_totals

    def failing(*args):
        raise OSError("totals unavailable")

    monkeypatch.setattr(impl, "refresh_group_totals", failing)
    service, _ = await _service(_config(tmp_path), own)
    first = await _take(service, "stack blocks")
    assert first.disposition is Disposition.KEPT and first.output_index == 0
    assert any("group totals were not refreshed" in i for i in first.issues)
    assert not service.pending_episodes()  # nothing is stuck
    monkeypatch.setattr(impl, "refresh_group_totals", real)
    second = await _take(service, "stack blocks")
    assert second.disposition is Disposition.KEPT and not second.issues
    assert _attrs(second.final_path.parent)["num_episodes"] == 2  # recomputed from both finalized episodes


async def test_an_existing_foreign_target_is_a_collision_and_the_take_is_retained_intact(
    tmp_path, monkeypatch, own
):
    """A directory already at the assigned slot is never overwritten or merged: the take stays pending, intact."""
    real_assign = impl.assign_output

    def colliding(*args):
        group, index = real_assign(*args)
        (args[0].parent / group / f"episode_{index}").mkdir()
        (args[0].parent / group / f"episode_{index}" / "stranger.txt").write_text("not ours")
        return group, index

    monkeypatch.setattr(impl, "assign_output", colliding)
    service, session = await _service(_config(tmp_path), own)
    result = await _take(service, "stack blocks")
    assert result.disposition is Disposition.RECOVERY_REQUIRED and "refusing to overwrite" in " ".join(
        result.issues
    )
    (pending,) = service.pending_episodes()
    assert pending.pending_path.is_dir()
    stranger = next((tmp_path / "out").glob("Kitchen/*/episode_0/stranger.txt"))
    assert stranger.read_text() == "not ours"


async def test_a_configuration_with_routing_is_a_different_configuration_from_one_without(tmp_path):
    """Whether routing is on is part of the digest; the studio mapping is not (each take freezes its studio)."""
    assert impl._digest(_config(tmp_path)) != impl._digest(_config(tmp_path, routing=None))
    other_mapping = RoutingConfig(studio_by_task=(("stack blocks", "Pantry"),), session_studio="Living Room")
    assert impl._digest(_config(tmp_path)) == impl._digest(_config(tmp_path, routing=other_mapping))


@pytest.mark.parametrize(
    "routing", [RoutingConfig(unmatched_studio=""), RoutingConfig(studio_by_task=(("", "Kitchen"),))]
)
async def test_an_unusable_routing_configuration_is_refused_before_any_effect(tmp_path, routing):
    with pytest.raises(ConfigurationError, match="routing"):
        await RecordingService(_config(tmp_path, routing)).open_session()
    assert not (tmp_path / "out").exists()


async def test_discarding_a_take_whose_slot_was_assigned_releases_that_slot(tmp_path, monkeypatch, own):
    """A take that got slot 0 but never published is discarded: the next kept take gets slot 0, not 1."""

    def failing(*args):
        raise OSError("metadata write failed")

    monkeypatch.setattr(impl, "write_episode_attributes", failing)
    service, session = await _service(_config(tmp_path), own)
    stuck = await _take(service, "stack blocks")
    assert stuck.disposition is Disposition.RECOVERY_REQUIRED
    (pending,) = service.pending_episodes()
    assert (
        read_manifest(next((session.session_path / ".recording").glob("episode_*.json"))).assignment["slot"] == 0
    )
    monkeypatch.undo()
    await service.discard_pending(pending.episode_id)
    kept = await _take(service, "stack blocks")
    assert kept.disposition is Disposition.KEPT and kept.output_index == 0 and kept.episode_index == 1
    assert kept.final_path.name == "episode_0"


async def test_the_result_of_a_finalized_take_does_no_filesystem_validation(tmp_path, monkeypatch, own):
    """If the group stamp turns unreadable right after finalization, the KEPT result is still returned."""
    real = impl.refresh_group_totals

    def refresh_then_corrupt(group):
        attributes = real(group)
        (group / "zarr.json").write_text("garbage")  # a transient fault after the data was published
        return attributes

    monkeypatch.setattr(impl, "refresh_group_totals", refresh_then_corrupt)
    service, _ = await _service(_config(tmp_path), own)
    result = await _take(service, "stack blocks")
    assert result.disposition is Disposition.KEPT and result.output_index == 0
    assert result.final_path.is_dir() and result.final_path.name == "episode_0"
