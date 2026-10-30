"""Dataset checker and reader on a small episode written by the real EpisodeStore (real Zarr + real MP4).

The input pixels/values are generated, but every file is produced by the production backend and read back
from disk; nothing is substituted for the writer, encoder or checker. Real recorded data is exercised in
tests/recording_service.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr

from egocap.recording.dataset_check import (
    CheckResult,
    Outcome,
    check_episode,
    report_from_dict,
    report_to_dict,
)
from egocap.recording.dataset_reader import EpisodeReader
from egocap.recording.episode_layout import SessionLock, read_identity, write_episode_attributes
from egocap.recording.episode_store import EpisodeStore, FieldSpec, NumericSourceSpec, VideoSourceSpec

ROWS = 6
NUMERIC = NumericSourceSpec("tracker", (FieldSpec("pose", "float64", (4, 4)), FieldSpec("flag", "bool")))
CAMERA = VideoSourceSpec("cam", 64, 48, 30.0)
BARE_CAMERA = VideoSourceSpec("bare", 64, 48, 30.0, intrinsics=False)


def _episode(path: Path) -> dict[str, int]:
    """Write ROWS rows/frames of generated values through the production store."""
    rng = np.random.default_rng(0)
    store = EpisodeStore(path, (NUMERIC,), (CAMERA, BARE_CAMERA))
    store.start_encoders()
    for i in range(ROWS):
        store.append_numeric(
            "tracker", float(i), float(i) + 0.01, {"pose": np.eye(4) * i, "flag": np.bool_(i % 2)}
        )
        for name, intrinsics in (("cam", np.eye(3)), ("bare", None)):
            frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            store.append_video(name, frame, float(i), float(i) + 0.02, intrinsics)
    return store.seal()


def _check(path, counts, required=("tracker", "cam")):
    return check_episode(path, (NUMERIC,), (CAMERA, BARE_CAMERA), required, counts)


@pytest.fixture
def episode(tmp_path):
    """(path, counts) of a clean episode."""
    path = tmp_path / "ep"
    return path, _episode(path)


def test_clean_episode_passes_and_report_round_trips(episode):
    """A clean episode passes every check; the report survives serialization for the manifest."""
    path, counts = episode
    report = _check(path, counts)
    assert report.outcome is Outcome.PASSED and len(report.check_results) == 6
    assert report_from_dict(report_to_dict(report)) == report
    assert CheckResult("x", Outcome.SKIPPED).issues == ()


def test_checker_flags_each_corruption_of_a_copy(episode, tmp_path):
    """Missing array/group, bad timestamps, NaN, short arrays, missing/truncated/mismatched video are flagged."""
    path, counts = episode

    def failed(copy):
        return {r.name for r in _check(copy, counts).check_results if r.outcome is not Outcome.PASSED}

    def corrupted(name):
        copy = tmp_path / name
        shutil.copytree(path, copy)
        return copy

    copy = corrupted("array")
    shutil.rmtree(copy / "tracker" / "pose")
    assert "schema_readable" in failed(copy)

    copy = corrupted("group")
    shutil.rmtree(copy / "tracker")
    assert {"schema_readable", "required_sources_nonempty"} <= failed(copy)

    copy = corrupted("dtype")
    group = zarr.open_group(str(copy), mode="r+")
    del group["tracker"]["flag"]
    group["tracker"].create_array("flag", shape=(ROWS,), dtype="int8", fill_value=0)
    assert "schema_readable" in failed(copy)

    copy = corrupted("times")
    group = zarr.open_group(str(copy), mode="r+")
    group["tracker"]["capture_time"][3] = -5.0
    group["cam"]["receive_time"][0] = np.nan
    group["cam"]["capture_time"][2] = np.inf
    pose = group["tracker"]["pose"]
    values = pose[:]
    values[1, 0, 0] = np.nan
    pose[:] = values
    assert {"timestamps_valid", "numeric_values_finite"} <= failed(copy)

    copy = corrupted("short")
    zarr.open_group(str(copy), mode="r+")["tracker"]["pose"].resize((ROWS - 2, 4, 4))
    assert "accepted_equals_written" in failed(copy)

    copy = corrupted("novideo")
    (copy / "cam.mp4").unlink()
    assert "video_readable_and_aligned" in failed(copy)

    copy = corrupted("cutvideo")
    video = copy / "cam.mp4"
    video.write_bytes(video.read_bytes()[:200])
    assert "video_readable_and_aligned" in failed(copy)

    copy = corrupted("garbage")
    (copy / "cam.mp4").write_bytes(b"not a video")
    assert "video_readable_and_aligned" in failed(copy)

    copy = corrupted("fewer_frames")
    zarr.open_group(str(copy), mode="r+")["cam"]["capture_time"].resize((ROWS - 1,))
    assert "video_readable_and_aligned" in failed(copy)


def test_unreadable_dataset_is_an_error_and_missing_required_source_fails(episode, tmp_path):
    """A non-dataset directory yields ERROR; a required source with no rows fails its check."""
    path, counts = episode
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _check(empty, counts).outcome is Outcome.ERROR
    assert _check(path, counts, required=("ghost",)).outcome is Outcome.FAILED


def test_reader_returns_stored_rows_frames_and_schema(episode):
    """The reader yields the stored rows and decoded RGB frames with timestamps, and derives specs."""
    path, _ = episode
    reader = EpisodeReader(path)
    assert reader.source_names() == ["bare", "cam", "tracker"]
    assert reader.is_video("cam") and not reader.is_video("tracker")
    assert reader.row_count("tracker") == ROWS
    spec = reader.numeric_spec("tracker")
    assert {f.name: (f.dtype, f.shape) for f in spec.fields} == {"flag": ("bool", ()), "pose": ("float64", (4, 4))}
    assert [f.name for f in reader.numeric_spec("tracker", ("pose",)).fields] == ["pose"]
    video_spec = reader.video_spec("cam")
    assert (video_spec.width, video_spec.height, video_spec.intrinsics) == (64, 48, True)
    assert reader.video_spec("bare").intrinsics is False

    rows = list(reader.numeric_rows("tracker", 1, 4))
    assert [r.capture_time for r in rows] == [1.0, 2.0, 3.0] and np.array_equal(rows[0].fields["pose"], np.eye(4))
    assert len(list(reader.numeric_rows("tracker"))) == ROWS
    frames = list(reader.video_frames("cam", 2, 5))
    assert len(frames) == 3 and frames[0].image_rgb.shape == (48, 64, 3) and frames[0].intrinsics.shape == (3, 3)
    assert next(reader.video_frames("bare", 0, 1)).intrinsics is None
    assert len(list(reader.video_frames("cam"))) == ROWS


def test_reader_reports_an_undecodable_frame(episode):
    """Asking for more frames than the MP4 holds raises instead of fabricating frames."""
    path, _ = episode
    zarr.open_group(str(path), mode="r+")["cam"]["capture_time"].resize((ROWS + 3,))
    with pytest.raises(RuntimeError, match="could not decode"):
        list(EpisodeReader(path).video_frames("cam", ROWS, ROWS + 2))


def test_attributes_identity_and_session_lock(tmp_path):
    """Attributes merge durably, identity reads back (or is None), and the session lock is exclusive."""
    path = tmp_path / "e"
    zarr.open_group(str(path), mode="w", zarr_format=3)
    write_episode_attributes(path, {"recording_episode_id": "e1", "k": 1})
    assert read_identity(path) == "e1" and read_identity(tmp_path / "missing") is None
    session = tmp_path / "session"
    session.mkdir()
    lock = SessionLock(session)
    with pytest.raises(BlockingIOError):
        SessionLock(session)
    lock.release()
    lock.release()
    SessionLock(session).release()
