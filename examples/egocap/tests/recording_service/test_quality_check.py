# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The quality checker reads real written output; corrupted COPIES of real output must be flagged.

The clean episode is produced by the real service from a real recorded episode. Corruptions are applied
to copies of that real output (never to the source fixture), which are explicitly not naturally defective
recordings.
"""

import shutil

import numpy as np
import pytest_asyncio
import zarr
from real import CAMERA, make_config, needs_real_episode, samples, submit_all

import recording_service.recording_service as impl
from egocap.recording.dataset_check import check_episode
from recording_service import Disposition, Outcome, QcConfig, RecordingService

pytestmark = needs_real_episode
ROWS = 20


@pytest_asyncio.fixture
async def written(tmp_path, own):
    """(config, final episode path, per-source accepted counts) for a clean real take."""
    config = make_config(tmp_path / "out")
    service = own(RecordingService(config))
    await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(ROWS))
    result = await service.stop_episode(episode.episode_id)
    await service.close()
    assert result.disposition is Disposition.KEPT
    return config, result.final_path, {c.source_name: c.accepted for c in result.counts}


def _check(config, path, counts, required=("pico_controller", CAMERA)):
    """Run the checker on ``path`` with the service's own configuration."""
    return check_episode(path, config.numeric_sources, config.video_sources, required, counts)


def _failed(report):
    """Names of the checks that did not pass."""
    return {r.name for r in report.check_results if r.outcome is not Outcome.PASSED}


def test_clean_real_output_passes_every_check(written):
    """Every mandatory check reports PASSED on a real, uncorrupted take."""
    config, path, counts = written
    report = _check(config, path, counts)
    assert report.outcome is Outcome.PASSED and not report.issues
    assert {r.name for r in report.check_results} == {
        "schema_readable",
        "required_sources_nonempty",
        "accepted_equals_written",
        "timestamps_valid",
        "numeric_values_finite",
        "video_readable_and_aligned",
    }


def test_missing_array_and_missing_group_fail_the_schema_check(written, tmp_path):
    """Deleting a declared array (or a whole source) from a copy fails schema_readable."""
    config, path, counts = written
    copy = tmp_path / "copy_array"
    shutil.copytree(path, copy)
    shutil.rmtree(copy / "pico_controller" / "head_pose")
    assert "schema_readable" in _failed(_check(config, copy, counts))
    copy2 = tmp_path / "copy_group"
    shutil.copytree(path, copy2)
    shutil.rmtree(copy2 / "manus_glove")
    assert "schema_readable" in _failed(_check(config, copy2, counts))


def test_timestamp_reversal_and_nonfinite_values_are_flagged(written, tmp_path):
    """A backwards capture_time and a NaN pose value in a copy fail their specific checks."""
    config, path, counts = written
    copy = tmp_path / "copy_ts"
    shutil.copytree(path, copy)
    group = zarr.open_group(str(copy), mode="r+")
    stamps = group["pico_controller"]["capture_time"]
    stamps[5] = stamps[3] - 1.0
    pose = group["manus_glove"]["left_hand_pose"]
    values = pose[:]
    values[2, 0, 0, 0] = np.nan
    pose[:] = values
    report = _check(config, copy, counts)
    assert {"timestamps_valid", "numeric_values_finite"} <= _failed(report)
    assert report.outcome is Outcome.FAILED


def test_truncated_array_breaks_counts_and_is_flagged(written, tmp_path):
    """Truncating one array of a source (mismatched lengths) fails accepted_equals_written."""
    config, path, counts = written
    copy = tmp_path / "copy_trunc"
    shutil.copytree(path, copy)
    zarr.open_group(str(copy), mode="r+")["pico_controller"]["head_pose"].resize((ROWS - 5, 4, 4))
    assert "accepted_equals_written" in _failed(_check(config, copy, counts))


def test_missing_or_truncated_video_is_flagged(written, tmp_path):
    """A removed MP4 and a half-written MP4 both fail video_readable_and_aligned."""
    config, path, counts = written
    gone = tmp_path / "copy_gone"
    shutil.copytree(path, gone)
    (gone / f"{CAMERA}.mp4").unlink()
    assert "video_readable_and_aligned" in _failed(_check(config, gone, counts))
    cut = tmp_path / "copy_cut"
    shutil.copytree(path, cut)
    video = cut / f"{CAMERA}.mp4"
    video.write_bytes(video.read_bytes()[: video.stat().st_size // 3])
    assert "video_readable_and_aligned" in _failed(_check(config, cut, counts))


def test_required_source_with_no_rows_fails_and_unreadable_dataset_is_an_error(written, tmp_path):
    """Required-source coverage is enforced; a directory that is not a dataset yields ERROR, never PASSED."""
    config, path, counts = written
    assert "required_sources_nonempty" in _failed(_check(config, path, counts, required=("absent_source",)))
    empty = tmp_path / "not_a_dataset"
    empty.mkdir()
    assert _check(config, empty, counts).outcome is Outcome.ERROR


async def test_take_missing_a_required_source_is_rejected_and_its_data_retained(tmp_path, own):
    """A take with no rows for a required source is REJECTED with a FAILED verdict; its data is kept."""
    base = make_config(tmp_path / "out")
    config = type(base)(
        **{**base.__dict__, "qc": QcConfig(required_sources=("pico_controller", "manus_glove"))},
    )
    service = own(RecordingService(config))
    await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, [s for s in samples(10) if s.source != "manus_glove"])
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.REJECTED and result.final_path is None
    assert result.quality.outcome is Outcome.FAILED
    assert any("manus_glove" in issue for issue in result.quality.issues)
    assert result.pending_path.is_dir()
    await service.close()


async def test_corrupted_staged_output_is_rejected_by_the_service_not_just_the_checker(tmp_path, monkeypatch, own):
    """Output corrupted between seal and check (real checker, corrupted copy of real output) is never finalized."""
    real_check = impl.check_episode

    def corrupt_then_check(pending_dir, *args):
        shutil.rmtree(pending_dir / "pico_controller" / "head_pose")
        return real_check(pending_dir, *args)

    monkeypatch.setattr(impl, "check_episode", corrupt_then_check)
    service = own(RecordingService(make_config(tmp_path / "out")))
    session = await service.open_session()
    episode = await service.start_episode("t")
    await submit_all(service, episode.episode_id, samples(10))
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.REJECTED
    assert "schema_readable" in {r.name for r in result.quality.check_results if r.outcome is not Outcome.PASSED}
    assert not (session.session_path / "episode_0").exists()
    await service.close()


async def test_kept_episode_carries_dataset_metadata_and_identity(tmp_path, own):
    """A kept episode's root metadata has the task, operator, length, duration and owning episode id."""
    service = own(RecordingService(make_config(tmp_path / "out")))
    await service.open_session()
    episode = await service.start_episode("stack two blocks", "alice")
    await submit_all(service, episode.episode_id, samples(12))
    result = await service.stop_episode(episode.episode_id)
    attrs = dict(zarr.open_group(str(result.final_path), mode="r").attrs)
    assert attrs["task_description"] == attrs["instruction_language"] == "stack two blocks"
    assert attrs["operator_name"] == "alice" and attrs["episode_length"] == 12
    assert attrs["episode_duration"] == result.duration_seconds > 0
    assert attrs["recording_episode_id"] == episode.episode_id
    assert attrs["integrity_check_passed"] is True and attrs["domain_checks"] == "SKIPPED"
    assert "freeze_check_passed" not in attrs  # no domain verdict is claimed for a check that did not run
    assert attrs["video_encoder_options"] and {s["name"] for s in attrs["sources"]} >= {"pico_controller", CAMERA}
    await service.close()


async def test_caller_may_reuse_its_buffers_after_submission(tmp_path, own):
    """Mutating a submitted array right after submit_sample does not change what is written."""
    service = own(RecordingService(make_config(tmp_path / "out")))
    await service.open_session()
    episode = await service.start_episode("t")
    items = samples(3)
    originals = [np.array(items[0].fields["head_pose"], copy=True), np.array(items[2].image, copy=True)]
    service.submit_sample(episode.episode_id, items[0])
    service.submit_sample(episode.episode_id, items[2])
    items[0].fields["head_pose"][...] = 0.0
    items[2].image[...] = 0
    result = await service.stop_episode(episode.episode_id)
    out = zarr.open_group(str(result.final_path), mode="r")
    assert np.array_equal(out["pico_controller"]["head_pose"][0], originals[0])
    await service.close()
