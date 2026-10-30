# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture-quality decisions through the real service, on a real recorded episode.

Samples are the stored rows and frames of an actual episode. The camera gate is driven by the configured
expected rate (far below / far above what the episode really achieved). The frozen-pose case submits copies of
the real rows with one pose field held at its first value, which is an explicitly constructed defect, not a
naturally frozen recording.
"""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import zarr
from real import CAMERA, make_config, needs_real_episode, samples, submit_all

import recording_service.recording_service as impl
from egocap.recording.episode_layout import MANIFEST_SCHEMA_VERSION, read_manifest
from recording_service import (
    CaptureQcConfig,
    CheckResult,
    ConfigurationError,
    Disposition,
    Outcome,
    QcConfig,
    QcDecision,
    QualityReport,
    RecordingService,
)

pytestmark = needs_real_episode
ROWS = 20
POSE = ("pico_controller", "head_pose")


def _duration() -> float:
    """Seconds the real pose rows span, so a freeze limit can be set relative to the data."""
    times = [s.capture_time for s in samples(ROWS) if s.source == POSE[0]]
    return times[-1] - times[0]


def _config(tmp_path, expected_fps: float = 1.0):
    """A service configuration with capture QC: the camera at ``expected_fps`` and a head-pose freeze check."""
    capture = CaptureQcConfig(
        camera_expected_fps=((CAMERA, expected_fps),), pose_streams=(POSE,), freeze_seconds=_duration() / 2
    )
    return make_config(
        tmp_path / "out", qc=QcConfig(required_sources=("pico_controller", CAMERA), capture=capture)
    )


def _frozen(items):
    """Copies of the real samples with the head pose held at its first value."""
    first = next(s.fields[POSE[1]] for s in items if s.source == POSE[0])
    return [
        dataclasses.replace(s, fields={**s.fields, POSE[1]: first}) if s.source == POSE[0] else s for s in items
    ]


async def _record(config, own, items=None):
    service = own(RecordingService(config))
    await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, items or samples(ROWS))
    return service, await service.stop_episode(episode.episode_id)


async def test_passing_capture_qc_keeps_the_episode_and_records_real_results(tmp_path, own):
    """Integrity and capture checks pass: KEPT, decision PASSED, per-check results in the result and metadata."""
    service, result = await _record(_config(tmp_path), own)
    assert result.disposition is Disposition.KEPT and result.qc_decision is QcDecision.PASSED
    assert result.capture_quality.outcome is Outcome.PASSED
    attrs = zarr.open_group(str(result.final_path), mode="r").attrs
    assert attrs["domain_checks"] == "EXECUTED" and attrs["capture_qc_decision"] == "PASSED"
    assert attrs["bad_episode"] is False and attrs["capture_qc_issues"] == []
    drops = attrs["frame_drops"][CAMERA]  # the native per-camera drop evidence
    assert set(drops) == {"window_max", "total", "window_s", "expected_fps"} and drops["expected_fps"] == 1.0
    assert attrs["freeze_check_passed"] is True and attrs["freeze_check_issues"] == []  # the native verdict
    assert {r["name"] for r in attrs["capture_qc_results"]} >= {
        f"camera_min_fps:{CAMERA}",
        "sustained_freeze:pico_controller/head_pose",
    }
    manifest = read_manifest(next((result.final_path.parent / ".recording").glob("episode_*.json")))
    assert manifest.qc_decision == "PASSED" and manifest.capture_quality is not None
    # The exact policy and algorithm version are frozen with the take, in its manifest and its metadata.
    assert manifest.capture_policy["algorithm_version"] == impl.ALGORITHM_VERSION
    assert manifest.capture_policy["camera_expected_fps"] == [[CAMERA, 1.0]]
    assert attrs["capture_qc_policy"] == manifest.capture_policy


async def test_capture_qc_not_configured_is_recorded_as_skipped_not_passed(tmp_path, own):
    """Without capture QC the decision is SKIPPED and the metadata says nothing was measured."""
    _, result = await _record(make_config(tmp_path / "out"), own)
    assert result.disposition is Disposition.KEPT and result.qc_decision is QcDecision.SKIPPED
    assert result.capture_quality is None
    attrs = zarr.open_group(str(result.final_path), mode="r").attrs
    assert attrs["domain_checks"] == "SKIPPED" and attrs["capture_qc_decision"] == "SKIPPED"


async def test_a_camera_below_its_minimum_rate_rejects_the_take_and_removes_only_its_data(tmp_path, own):
    """An unreachable expected rate fails the camera gate: REJECTED, measured rate reported, pending data gone."""
    config = _config(tmp_path, expected_fps=1e6)
    service, result = await _record(config, own)
    assert result.disposition is Disposition.REJECTED and result.qc_decision is QcDecision.REJECTED_CAMERA
    assert result.pending_path is None and result.final_path is None  # the data was removed: no retained location
    assert any("fps average is below" in i for i in result.capture_quality.issues)
    session = next((tmp_path / "out").iterdir())
    assert not any(p.name.startswith("episode_") for p in session.iterdir())  # no pending, no final directory
    assert not service.pending_episodes()
    assert service.snapshot().last_result.qc_decision is QcDecision.REJECTED_CAMERA


async def test_a_frozen_pose_keeps_the_episode_flagged_bad(tmp_path, own):
    """A pose held still for over half the take: the data is kept and marked bad, with the issue in metadata."""
    _, result = await _record(_config(tmp_path), own, _frozen(samples(ROWS)))
    assert result.disposition is Disposition.KEPT and result.qc_decision is QcDecision.FLAGGED_BAD
    attrs = zarr.open_group(str(result.final_path), mode="r").attrs
    assert attrs["bad_episode"] is True and any("no motion for" in i for i in attrs["capture_qc_issues"])
    assert attrs["freeze_check_passed"] is False  # what native readers (the EOD quarantine filter) look for
    assert any("no motion for" in i for i in attrs["freeze_check_issues"])
    assert attrs["integrity_check_passed"] is True  # bad capture quality never masquerades as an integrity failure


async def test_camera_rejection_wins_when_the_pose_is_also_frozen(tmp_path, own):
    """Both failures measured: the camera rejection takes precedence over the bad-tracking flag."""
    _, result = await _record(_config(tmp_path, expected_fps=1e6), own, _frozen(samples(ROWS)))
    assert result.disposition is Disposition.REJECTED and result.qc_decision is QcDecision.REJECTED_CAMERA


async def test_a_failed_rejection_cleanup_stays_visible_and_recovers(tmp_path, monkeypatch, own):
    """If removing the rejected take's data fails it is retained and listed; recovery finishes the rejection."""
    real_remove = impl.remove_owned_episode

    def failing(*args):
        raise OSError("disk went away")

    monkeypatch.setattr(impl, "remove_owned_episode", failing)
    service, result = await _record(_config(tmp_path, expected_fps=1e6), own)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and "disk went away" in " ".join(result.issues)
    (pending,) = service.pending_episodes()
    assert pending.pending_path.exists()
    monkeypatch.setattr(impl, "remove_owned_episode", real_remove)
    recovered = await service.recover_episode(pending.episode_id)
    assert recovered.disposition is Disposition.REJECTED and recovered.qc_decision is QcDecision.REJECTED_CAMERA
    assert not pending.pending_path.exists() and not service.pending_episodes()


async def test_recovery_after_the_data_was_removed_but_the_rejection_was_not_recorded(tmp_path, monkeypatch, own):
    """A crash between removing the data and recording REJECTED leaves only the record: recovery completes it."""
    real_remove = impl.remove_owned_episode

    def removes_then_fails(session_path, manifest):
        real_remove(session_path, manifest)
        raise OSError("lost the checkpoint")

    monkeypatch.setattr(impl, "remove_owned_episode", removes_then_fails)
    service, result = await _record(_config(tmp_path, expected_fps=1e6), own)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    (pending,) = service.pending_episodes()
    assert not pending.pending_path.exists()  # the data really is gone; only the manifest remains
    recovered = await service.recover_episode(pending.episode_id)
    assert recovered.disposition is Disposition.REJECTED and not service.pending_episodes()


async def test_a_version_one_manifest_is_read_as_capture_qc_not_run(tmp_path, own):
    """Manifests written before capture QC lack its fields: they load, and mean SKIPPED rather than passed."""
    service, result = await _record(make_config(tmp_path / "out"), own)
    manifest_file = next((result.final_path.parent / ".recording").glob("episode_*.json"))
    data = json.loads(manifest_file.read_text())
    data["schema_version"] = 1
    del data["qc_decision"], data["capture_quality"]
    manifest_file.write_text(json.dumps(data))
    manifest = read_manifest(manifest_file)
    assert manifest.qc_decision is None and manifest.capture_quality is None


@pytest.mark.parametrize(
    "capture",
    [
        CaptureQcConfig(camera_expected_fps=(("no_such_camera", 30.0),)),
        CaptureQcConfig(pose_streams=(("pico_controller", "no_such_field"),)),
        CaptureQcConfig(),
    ],
)
async def test_an_unusable_capture_qc_configuration_is_refused_before_any_effect(tmp_path, capture):
    """Unsupported or missing source mappings fail configuration; the output root is never created."""
    config = make_config(tmp_path / "out", qc=QcConfig(required_sources=("pico_controller",), capture=capture))
    with pytest.raises(ConfigurationError, match="capture QC"):
        await RecordingService(config).open_session()
    assert not (tmp_path / "out").exists()


async def test_a_pose_stream_that_is_not_a_required_source_is_refused_because_an_empty_one_cannot_freeze(tmp_path):
    capture = CaptureQcConfig(pose_streams=(("pico_controller", "head_pose"),))
    config = make_config(tmp_path / "out", qc=QcConfig(required_sources=(CAMERA,), capture=capture))
    with pytest.raises(ConfigurationError, match="pose stream source 'pico_controller' must be a required source"):
        await RecordingService(config).open_session()
    assert not (tmp_path / "out").exists()


async def test_an_invalid_source_declaration_is_a_configuration_error_even_with_capture_qc_on(tmp_path):
    """The source declarations are validated before the capture checks read their dtypes."""
    config = make_config(
        tmp_path / "out",
        qc=QcConfig(
            required_sources=("pico_controller",),
            capture=CaptureQcConfig(pose_streams=(("pico_controller", "head_pose"),)),
        ),
    )
    (spec,) = [n for n in config.numeric_sources if n.name == "pico_controller"]
    broken = dataclasses.replace(
        spec,
        fields=tuple(
            dataclasses.replace(f, dtype="not-a-dtype") if f.name == "head_pose" else f for f in spec.fields
        ),
    )
    numeric = tuple(broken if n.name == "pico_controller" else n for n in config.numeric_sources)
    with pytest.raises(ConfigurationError):
        await RecordingService(dataclasses.replace(config, numeric_sources=numeric)).open_session()
    assert not (tmp_path / "out").exists()


async def test_missing_data_of_a_checked_take_that_was_not_rejected_is_retained_not_rejected(
    tmp_path, monkeypatch, own
):
    """Only a camera-rejected take may finish without its data; any other take with data gone stays unresolved."""

    def failing(*args):
        raise OSError("cannot publish")

    monkeypatch.setattr(impl, "finalize_episode_dir", failing)
    service, result = await _record(_config(tmp_path), own)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    (pending,) = service.pending_episodes()
    shutil.rmtree(pending.pending_path)  # the data is lost behind the service's back
    again = await service.recover_episode(pending.episode_id)
    assert again.disposition is Disposition.RECOVERY_REQUIRED and "neither" in " ".join(again.issues)
    assert service.pending_episodes()  # still visible: never silently treated as rejected or kept


async def test_a_partial_removal_keeps_the_ownership_stamp_so_it_can_be_retried(tmp_path, monkeypatch, own):
    """If deleting the rejected take's data stops midway, what is left is still provably its own and retryable."""
    import egocap.recording.episode_layout as layout

    real_rmtree = layout.shutil.rmtree
    calls = []

    def interrupted(path, *args, **kwargs):
        if not Path(path).parent.name.endswith(".pending"):  # only the removal of the take's own contents fails
            return real_rmtree(path, *args, **kwargs)
        calls.append(path)
        raise OSError("interrupted midway")

    monkeypatch.setattr(layout.shutil, "rmtree", interrupted)
    service, result = await _record(_config(tmp_path, expected_fps=1e6), own)
    assert calls and result.disposition is Disposition.RECOVERY_REQUIRED
    (pending,) = service.pending_episodes()
    assert (pending.pending_path / "zarr.json").is_file()  # the stamp outlives the rest of the contents
    monkeypatch.setattr(layout.shutil, "rmtree", real_rmtree)
    recovered = await service.recover_episode(pending.episode_id)
    assert recovered.disposition is Disposition.REJECTED and not pending.pending_path.exists()


def _legacy_digest(config) -> str:
    """The configuration digest exactly as computed before capture QC existed (no ``capture`` key)."""
    canonical = {
        "numeric": [dataclasses.asdict(s) for s in config.numeric_sources],
        "video": [dataclasses.asdict(s) for s in config.video_sources],
        "qc": {"required_sources": list(config.qc.required_sources)},
        "latency": config.latency,
        "calibration": [a.name for a in config.calibration_assets],
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


async def test_a_take_sealed_by_the_previous_version_is_still_recoverable_and_upgraded_on_checkpoint(
    tmp_path, monkeypatch, own
):
    """A version-1 SEALED manifest with the legacy digest recovers when the configuration is unchanged."""
    config = make_config(tmp_path / "out")
    assert impl._digest(config) == _legacy_digest(config)  # capture QC off: the digest is unchanged

    def failing(*args):
        raise OSError("checker unavailable")

    monkeypatch.setattr(impl, "check_episode", failing)
    first, result = await _record(config, own)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    session = result.pending_path.parent
    await first.close()
    monkeypatch.undo()
    manifest_file = next((session / ".recording").glob("episode_*.json"))
    data = json.loads(manifest_file.read_text())
    data["schema_version"] = 1
    for key in ("qc_decision", "capture_quality", "capture_policy"):
        del data[key]
    manifest_file.write_text(json.dumps(data))

    service = own(RecordingService(config))
    await service.open_session(data["session_id"])
    (pending,) = service.pending_episodes()
    assert pending.recoverable and not pending.issues
    recovered = await service.recover_episode(pending.episode_id)
    assert recovered.disposition is Disposition.KEPT and recovered.qc_decision is QcDecision.SKIPPED
    assert (
        json.loads(manifest_file.read_text())["schema_version"] == MANIFEST_SCHEMA_VERSION
    )  # the touched manifest is now genuinely current


async def test_removal_interrupted_after_the_stamp_was_deleted_leaves_only_an_empty_directory_and_recovers(
    tmp_path, monkeypatch, own
):
    """The last window of the removal (stamp gone, directory not yet) is safe to finish."""
    real_rmdir = Path.rmdir

    def failing_rmdir(self):
        if self.name.endswith(".pending"):
            raise OSError("interrupted before the directory was removed")
        return real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", failing_rmdir)
    service, result = await _record(_config(tmp_path, expected_fps=1e6), own)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    (pending,) = service.pending_episodes()
    assert pending.pending_path.is_dir() and not any(pending.pending_path.iterdir())
    monkeypatch.setattr(Path, "rmdir", real_rmdir)
    recovered = await service.recover_episode(pending.episode_id)
    assert recovered.disposition is Disposition.REJECTED and not pending.pending_path.exists()


async def test_the_capture_policy_is_frozen_when_the_take_is_created_and_a_changed_algorithm_blocks_resuming(
    tmp_path, monkeypatch, own
):
    """The policy is in the manifest from the start; resuming under another algorithm is refused."""
    config = _config(tmp_path)

    def failing(*args):
        raise OSError("checker unavailable")

    service = own(RecordingService(config))
    session = await service.open_session()
    episode = await service.start_episode("t")
    started = read_manifest(next((session.session_path / ".recording").glob("episode_*.json")))
    assert started.storage_phase == "WRITING"
    assert started.capture_policy["algorithm_version"] == impl.ALGORITHM_VERSION
    monkeypatch.setattr(impl, "check_episode", failing)
    await submit_all(service, episode.episode_id, samples(ROWS))
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    await service.close()
    monkeypatch.undo()
    monkeypatch.setattr(impl, "ALGORITHM_VERSION", impl.ALGORITHM_VERSION + 1)  # a different algorithm
    again = own(RecordingService(config))
    await again.open_session(session.session_id)
    (pending,) = again.pending_episodes()
    assert not pending.recoverable and "different configuration" in " ".join(pending.issues)


async def test_older_manifests_decode_onto_the_current_fields_in_memory(tmp_path, own):
    """A record from before routing has no destination or finalization intent; it decodes to the current shape."""
    _, result = await _record(make_config(tmp_path / "out"), own)
    manifest_file = next((result.final_path.parent / ".recording").glob("episode_*.json"))
    data = json.loads(manifest_file.read_text())
    for key in ("studio", "assignment", "finalization"):
        del data[key]
    data["schema_version"] = 2  # before routing: no destination at all
    manifest_file.write_text(json.dumps(data))
    manifest = read_manifest(manifest_file)
    assert manifest.studio is None and manifest.assignment is None
    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION


async def test_the_native_freeze_verdict_is_written_only_when_a_freeze_check_established_it(tmp_path, own):
    """Camera-only capture QC never ran a freeze check, so it must not claim one passed."""
    capture = CaptureQcConfig(camera_expected_fps=((CAMERA, 1.0),))
    config = make_config(
        tmp_path / "out", qc=QcConfig(required_sources=("pico_controller", CAMERA), capture=capture)
    )
    _, result = await _record(config, own)
    attrs = zarr.open_group(str(result.final_path), mode="r").attrs
    assert result.qc_decision is QcDecision.PASSED and attrs["capture_qc_decision"] == "PASSED"
    assert "freeze_check_passed" not in attrs and "freeze_check_issues" not in attrs


async def test_a_freeze_check_that_could_not_run_retains_the_take_instead_of_labelling_it_passed(
    tmp_path, monkeypatch, own
):
    """ERROR is not a verdict: the take stays SEALED and unresolved, with no final output and no freeze verdict."""
    errored = QualityReport(
        Outcome.ERROR,
        ("sustained_freeze:pico_controller/head_pose could not run: boom",),
        (),
        (CheckResult("sustained_freeze:pico_controller/head_pose", Outcome.ERROR, ("could not run: boom",)),),
    )
    monkeypatch.setattr(impl, "check_capture_quality", lambda *args: errored)
    service, result = await _record(_config(tmp_path), own)
    assert result.disposition is Disposition.RECOVERY_REQUIRED and result.final_path is None
    assert any("could not run: boom" in i for i in result.issues)
    (pending,) = service.pending_episodes()
    assert pending.storage_phase == "SEALED" and pending.pending_path.is_dir()
    manifest = read_manifest(next((pending.pending_path.parent / ".recording").glob("episode_*.json")))
    assert manifest.qc_decision is None and manifest.finalization is None  # no decision was recorded


async def test_the_native_freeze_issues_list_only_the_freeze_findings(tmp_path, own):
    """A flagged episode reports freeze_check_passed False with the freeze issues, not camera or other findings."""
    _, result = await _record(_config(tmp_path), own, _frozen(samples(ROWS)))
    attrs = zarr.open_group(str(result.final_path), mode="r").attrs
    assert attrs["freeze_check_passed"] is False
    assert attrs["freeze_check_issues"] and all("no motion for" in i for i in attrs["freeze_check_issues"])


async def _make_legacy(session_path):
    """Make the record look like one written before finalization intents existed (schema 2)."""
    manifest_file = next((session_path / ".recording").glob("episode_*.json"))
    data = json.loads(manifest_file.read_text())
    del data["finalization"]
    data["schema_version"] = 2
    manifest_file.write_text(json.dumps(data))


async def test_a_checked_record_without_an_intent_derives_the_same_intent_from_its_results(
    tmp_path, monkeypatch, own
):
    """A kept take and a camera-rejected take, checked before intents were recorded, both recover correctly."""
    config = _config(tmp_path, expected_fps=1.0)
    monkeypatch.setattr(impl, "finalize_episode_dir", lambda *a: (_ for _ in ()).throw(OSError("cannot publish")))
    service, result = await _record(config, own)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    session = result.pending_path.parent
    await service.close()
    monkeypatch.undo()
    await _make_legacy(session)
    again = own(RecordingService(config))
    await again.open_session(session.name)
    (pending,) = again.pending_episodes()
    kept = await again.recover_episode(pending.episode_id)
    assert kept.disposition is Disposition.KEPT and kept.qc_decision is QcDecision.PASSED

    rejecting = _config(tmp_path / "second", expected_fps=1e6)
    real_remove = impl.remove_owned_episode

    def removes_then_fails(session_path, manifest):
        real_remove(session_path, manifest)
        raise OSError("lost the checkpoint")

    monkeypatch.setattr(impl, "remove_owned_episode", removes_then_fails)
    service, result = await _record(rejecting, own)
    session = result.pending_path.parent
    await service.close()
    monkeypatch.undo()
    await _make_legacy(session)  # the data is already gone: only the derived intent says it was a removal
    again = own(RecordingService(rejecting))
    await again.open_session(session.name)
    (pending,) = again.pending_episodes()
    removed = await again.recover_episode(pending.episode_id)
    assert removed.disposition is Disposition.REJECTED and removed.pending_path is None


async def test_a_capture_check_that_cannot_run_retains_the_take_and_a_recovery_reruns_it(
    tmp_path, monkeypatch, own
):
    """An unmeasured gate says nothing about the take: it stays SEALED, never kept as unflagged data."""
    real = impl.check_capture_quality
    unmeasured = ("could not be measured",)

    def erroring(pending_dir, config):
        report = real(pending_dir, config)
        results = tuple(
            dataclasses.replace(r, outcome=Outcome.ERROR, issues=unmeasured) for r in report.check_results
        )
        return dataclasses.replace(report, outcome=Outcome.ERROR, issues=unmeasured, check_results=results)

    monkeypatch.setattr(impl, "check_capture_quality", erroring)
    service, result = await _record(_config(tmp_path), own)
    assert result.disposition is Disposition.RECOVERY_REQUIRED
    assert any("capture quality check errored" in i for i in result.issues)
    (pending,) = service.pending_episodes()
    assert pending.storage_phase == "SEALED" and pending.pending_path.is_dir()
    monkeypatch.undo()  # the measurement works again
    recovered = await service.recover_episode(pending.episode_id)
    assert recovered.disposition is Disposition.KEPT and recovered.qc_decision is QcDecision.PASSED


async def test_a_default_camera_rate_is_the_sources_own_fps_and_is_frozen_with_the_take(tmp_path, own):
    """The policy stored with a take holds the concrete rate the default resolved to, never None."""
    capture = CaptureQcConfig(camera_expected_fps=((CAMERA, None),), pose_streams=(POSE,))
    config = make_config(
        tmp_path / "out", qc=QcConfig(required_sources=("pico_controller", CAMERA), capture=capture)
    )
    (video,) = config.video_sources
    service, _ = await _record(config, own)
    manifest = read_manifest(next((tmp_path / "out").glob("*/.recording/episode_*.json")))
    assert manifest.capture_policy["camera_expected_fps"] == [[CAMERA, video.fps]]
    assert service._config.qc.capture.camera_expected_fps == ((CAMERA, video.fps),)
