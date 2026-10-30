"""EpisodeStore and episode layout, exercised with a real recorded episode.

The round-trip tests read actual Zarr arrays and decode an actual MP4 from a previously recorded
episode named by ``EGOCAP_REAL_EPISODE`` and write through the real Zarr/ffmpeg backend.
Without that environment variable they are SKIPPED (reported, never passed). The validation and
layout tests use no fixture.
"""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import zarr

from egocap.recording.episode_layout import (
    FINALIZATION_KEEP,
    FINALIZATION_REMOVE,
    FINALIZATION_RETAIN,
    IDENTITY_ATTR,
    MANIFEST_SCHEMA_VERSION,
    EpisodeManifest,
    episode_dir_names,
    finalize_episode_dir,
    locate_episode,
    manifest_path,
    next_episode_index,
    read_manifest,
    remove_owned_episode,
    scan_manifests,
    write_manifest,
)
from egocap.recording.episode_store import (
    EpisodeStore,
    FieldSpec,
    NumericSourceSpec,
    VideoSourceSpec,
    validate_source_specs,
)

_FIXTURE = os.environ.get("EGOCAP_REAL_EPISODE")
needs_real_episode = pytest.mark.skipif(
    not _FIXTURE, reason="EGOCAP_REAL_EPISODE not set (real-data test SKIPPED)"
)
ROWS = 40
PICO_FIELDS = ("head_pose", "a_button", "left_trigger")


def _manifest(index: int = 0, phase: str = "WRITING") -> EpisodeManifest:
    pending, final = episode_dir_names(index)
    return EpisodeManifest(
        session_id="s",
        episode_id=f"e{index}",
        episode_index=index,
        config_digest="d",
        task="t",
        operator="o",
        sources=[],
        pending_path=pending,
        final_path=final,
        storage_phase=phase,
    )


@needs_real_episode
def test_real_numeric_and_video_round_trip(tmp_path: Path) -> None:
    """Real rows and frames written through the store read back identical (numeric) and complete (video)."""
    src = Path(_FIXTURE)
    group = zarr.open_group(str(src), mode="r")
    pico, cam = group["pico_controller"], group["head_human_camera"]
    capture = cv2.VideoCapture(str(src / "head_human_camera.mp4"))
    ok, first = capture.read()
    assert ok
    height, width = first.shape[:2]
    fields = tuple(FieldSpec(k, str(pico[k].dtype), tuple(pico[k].shape[1:])) for k in PICO_FIELDS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    store = EpisodeStore(
        tmp_path / "episode_0.pending",
        (NumericSourceSpec("pico_controller", fields),),
        (VideoSourceSpec("head_human_camera", width, height, 30.0),),
    )
    store.start_encoders()
    for i in range(ROWS):
        row = {k: pico[k][i] for k in PICO_FIELDS}
        store.append_numeric(
            "pico_controller", float(pico["capture_time"][i]), float(pico["receive_time"][i]), row
        )
        ok, frame = capture.read()
        assert ok
        store.append_video(
            "head_human_camera",
            frame,
            float(cam["capture_time"][i]),
            float(cam["receive_time"][i]),
            cam["intrinsics"][i],
        )
    assert store.seal() == {"pico_controller": ROWS, "head_human_camera": ROWS}

    out = zarr.open_group(str(tmp_path / "episode_0.pending"), mode="r")
    for name in PICO_FIELDS + ("capture_time", "receive_time"):
        assert out["pico_controller"][name].dtype == pico[name].dtype
        assert np.array_equal(out["pico_controller"][name][:], pico[name][:ROWS]), name
    for name in ("capture_time", "receive_time", "intrinsics"):
        assert np.array_equal(out["head_human_camera"][name][:], cam[name][:ROWS]), name
    decoded = cv2.VideoCapture(str(tmp_path / "episode_0.pending" / "head_human_camera.mp4"))
    assert int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)) == ROWS
    assert (int(decoded.get(cv2.CAP_PROP_FRAME_WIDTH)), int(decoded.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (
        width,
        height,
    )


@needs_real_episode
def test_real_numeric_rows_cross_the_flush_boundary(tmp_path: Path) -> None:
    """More rows than the in-memory flush size still read back in order with none lost."""
    pico = zarr.open_group(_FIXTURE, mode="r")["pico_controller"]
    total = pico["capture_time"].shape[0]
    assert total > 256, "fixture must exceed the flush size to exercise chunked appends"
    spec = NumericSourceSpec("pico_controller", (FieldSpec("head_pose", "float64", (4, 4)),))
    store = EpisodeStore(tmp_path / "p", (spec,), ())
    store.start_encoders()
    for i in range(total):
        store.append_numeric(
            "pico_controller", float(pico["capture_time"][i]), 0.0, {"head_pose": pico["head_pose"][i]}
        )
    assert store.seal() == {"pico_controller": total}
    out = zarr.open_group(str(tmp_path / "p"), mode="r")["pico_controller"]
    assert np.array_equal(out["head_pose"][:], pico["head_pose"][:])
    assert np.array_equal(out["capture_time"][:], pico["capture_time"][:])


def test_rows_with_wrong_shape_dtype_or_fields_are_rejected_and_not_counted(tmp_path: Path) -> None:
    """A bad row raises before anything is stored, so the count only reflects valid rows."""
    spec = NumericSourceSpec("s", (FieldSpec("v", "float64", (2,)),))
    store = EpisodeStore(tmp_path / "p", (spec,), ())
    store.start_encoders()
    good = {"v": np.zeros(2)}
    store.append_numeric("s", 1.0, 1.0, good)
    with pytest.raises(ValueError, match="shape"):
        store.append_numeric("s", 2.0, 2.0, {"v": np.zeros(3)})
    with pytest.raises(ValueError, match="dtype"):
        store.append_numeric("s", 2.0, 2.0, {"v": np.zeros(2, dtype=np.float32)})
    with pytest.raises(ValueError, match="fields"):
        store.append_numeric("s", 2.0, 2.0, {"w": np.zeros(2)})
    assert store.counts() == {"s": 1}
    assert store.seal() == {"s": 1}


def test_duplicate_source_names_and_existing_pending_dir_are_refused(tmp_path: Path) -> None:
    """Sources must be uniquely named, and an existing pending directory is never overwritten."""
    spec = NumericSourceSpec("s", (FieldSpec("v", "float64"),))
    with pytest.raises(ValueError, match="duplicate"):
        EpisodeStore(tmp_path / "a", (spec, spec), ())
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        EpisodeStore(tmp_path / "taken", (spec,), ())


def test_manifest_round_trip_atomic_and_unsupported_version_reported(tmp_path: Path) -> None:
    """A checkpoint reads back equal with no temp file left; an unknown schema version is reported, not guessed."""
    manifest = _manifest(3)
    write_manifest(tmp_path, manifest)
    assert read_manifest(manifest_path(tmp_path, 3)) == manifest
    assert not list((tmp_path / ".recording").glob("*.tmp"))
    manifest.storage_phase = "SEALED"
    write_manifest(tmp_path, manifest)
    assert read_manifest(manifest_path(tmp_path, 3)).storage_phase == "SEALED"

    bad = json.loads(manifest_path(tmp_path, 3).read_text())
    bad["schema_version"] = 99
    manifest_path(tmp_path, 4).write_text(json.dumps(bad))
    found, problems = scan_manifests(tmp_path)
    assert [m.episode_index for m in found] == [3]
    assert len(problems) == 1 and "schema_version" in problems[0]


def test_a_damaged_older_record_is_listed_as_unreadable_not_raised(tmp_path: Path) -> None:
    """A checked older record whose results are missing or damaged must not fail the whole scan."""
    write_manifest(tmp_path, _manifest(1))
    good = json.loads(manifest_path(tmp_path, 1).read_text())
    older = {**good, "schema_version": 2, "storage_phase": "CHECKED"}
    del older["finalization"]  # older records have no finalization intent: it is derived from the outcome
    no_outcome = {**older, "quality": {"issues": []}}
    not_a_report = {**older, "quality": "oops"}
    manifest_path(tmp_path, 2).write_text(json.dumps(no_outcome))
    manifest_path(tmp_path, 3).write_text(json.dumps(not_a_report))
    found, problems = scan_manifests(tmp_path)
    assert [m.episode_index for m in found] == [1]
    assert len(problems) == 2 and all("missing or damaged" in p for p in problems)


def test_next_episode_index_skips_pending_final_and_manifest_only_episodes(tmp_path: Path) -> None:
    """Indices are never reused, whether the episode survives as a directory or only as a manifest."""
    assert next_episode_index(tmp_path) == 0
    (tmp_path / "episode_0.pending").mkdir()
    (tmp_path / "episode_1").mkdir()
    write_manifest(tmp_path, _manifest(5))
    assert next_episode_index(tmp_path) == 6


def test_finalize_is_retry_safe_and_discard_never_deletes_final_output(tmp_path: Path) -> None:
    """A repeated finalize after the rename is a no-op; discard refuses finalized data but removes pending data."""
    manifest = _manifest(0)
    zarr.open_group(str(tmp_path / manifest.pending_path), mode="w", zarr_format=3).attrs[IDENTITY_ATTR] = (
        manifest.episode_id
    )
    (tmp_path / manifest.pending_path / "x").write_text("payload")
    final = finalize_episode_dir(tmp_path, manifest)
    assert (final / "x").read_text() == "payload"
    assert finalize_episode_dir(tmp_path, manifest) == final
    manifest.storage_phase = "FINALIZED"
    with pytest.raises(PermissionError):
        remove_owned_episode(tmp_path, manifest)
    assert (final / "x").exists()

    other = _manifest(1)
    (tmp_path / other.pending_path).mkdir()
    remove_owned_episode(tmp_path, other)
    assert not (tmp_path / other.pending_path).exists()
    assert final.exists()


def test_manifest_paths_are_validated_before_any_rename_or_delete(tmp_path: Path) -> None:
    """Traversal, absolute, non-canonical and symlinked manifest paths are refused; nothing outside is touched."""
    outside = tmp_path / "outside"
    outside.mkdir()
    session = tmp_path / "session"
    session.mkdir()
    for bad_pending in ("../outside", str(outside), "episode_0.pending/../..", "other.pending"):
        manifest = _manifest(0)
        manifest.pending_path = bad_pending
        with pytest.raises(ValueError):
            finalize_episode_dir(session, manifest)
        with pytest.raises(ValueError):
            remove_owned_episode(session, manifest)
    (session / "episode_1.pending").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        remove_owned_episode(session, _manifest(1))
    assert outside.is_dir()


def test_repeat_finalize_requires_the_final_directory_to_be_owned_by_the_same_episode(tmp_path: Path) -> None:
    """An existing final directory is accepted only when stamped with this episode's identity."""
    manifest = _manifest(0)
    final = tmp_path / manifest.final_path
    group = zarr.open_group(str(final), mode="w", zarr_format=3)
    with pytest.raises(FileExistsError, match="not owned"):
        finalize_episode_dir(tmp_path, manifest)
    group.attrs[IDENTITY_ATTR] = "someone-else"
    with pytest.raises(FileExistsError, match="not owned"):
        finalize_episode_dir(tmp_path, manifest)
    group.attrs[IDENTITY_ATTR] = manifest.episode_id
    assert finalize_episode_dir(tmp_path, manifest) == final


def _stamped(path: Path, identity: str | None) -> None:
    """Create a Zarr directory, stamped with an owning episode id (or left unstamped)."""
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    if identity is not None:
        group.attrs[IDENTITY_ATTR] = identity


def test_locate_episode_decides_ownership_before_anything_is_touched(tmp_path: Path) -> None:
    """Pending/final location requires canonical names, matching identity and no ambiguity."""
    manifest = _manifest(0, "CHECKED")
    with pytest.raises(FileNotFoundError):
        locate_episode(tmp_path, manifest)

    _stamped(tmp_path / manifest.pending_path, manifest.episode_id)
    assert locate_episode(tmp_path, manifest) == "pending"
    _stamped(tmp_path / manifest.final_path, manifest.episode_id)
    with pytest.raises(FileExistsError, match="both"):
        locate_episode(tmp_path, manifest)
    shutil.rmtree(tmp_path / manifest.pending_path)
    assert locate_episode(tmp_path, manifest) == "final"
    _stamped(tmp_path / manifest.final_path, "someone-else")
    with pytest.raises(FileExistsError, match="not owned"):
        locate_episode(tmp_path, manifest)


def test_unstamped_pending_is_accepted_only_while_writing_and_foreign_pending_is_never_deleted(
    tmp_path: Path,
) -> None:
    """An unstamped pending directory can exist only before the stamp is written; it must not be adopted later."""
    _stamped(tmp_path / "episode_0.pending", None)
    assert locate_episode(tmp_path, _manifest(0, "WRITING")) == "pending"
    with pytest.raises(PermissionError, match="not owned"):
        locate_episode(tmp_path, _manifest(0, "CHECKED"))
    with pytest.raises(PermissionError, match="not owned"):
        remove_owned_episode(tmp_path, _manifest(0, "CHECKED"))
    assert (tmp_path / "episode_0.pending").is_dir()
    remove_owned_episode(tmp_path, _manifest(0, "WRITING"))  # the interrupted-start case may be discarded
    assert not (tmp_path / "episode_0.pending").exists()


def test_read_manifest_rejects_noncanonical_paths_and_indices(tmp_path: Path) -> None:
    """A manifest whose paths or index are not canonical is unreadable, so nothing can be redirected by it."""
    write_manifest(tmp_path, _manifest(2))
    path = manifest_path(tmp_path, 2)
    good = json.loads(path.read_text())
    patches = (
        {"pending_path": "../x"},
        {"final_path": "/abs"},
        {"episode_index": -1},
        {"episode_index": "2"},
        {"storage_phase": "SEALEDD"},
    )
    for patch in patches:
        path.write_text(json.dumps({**good, **patch}))
        with pytest.raises(ValueError):
            read_manifest(path)
    path.write_text(json.dumps(good))
    assert read_manifest(path).episode_index == 2


@pytest.mark.parametrize(
    "numeric,video,message",
    [
        (NumericSourceSpec("s", (FieldSpec("v", "float64"), FieldSpec("v", "int8"))), None, "duplicate field"),
        (NumericSourceSpec("s", (FieldSpec("", "float64"),)), None, "safe array name"),
        (NumericSourceSpec("s", (FieldSpec("a/b", "float64"),)), None, "safe array name"),
        (NumericSourceSpec("s", (FieldSpec(".v", "float64"),)), None, "safe array name"),
        (NumericSourceSpec("s", (FieldSpec("receive_time", "float64"),)), None, "timestamp"),
        (NumericSourceSpec("s", (FieldSpec("v", "not-a-dtype"),)), None, "invalid dtype"),
        (NumericSourceSpec("s", (FieldSpec("v", "U4"),)), None, "bool, integer or float"),
        (NumericSourceSpec("s", (FieldSpec("v", "float64", (2, -1)),)), None, "invalid shape"),
        (NumericSourceSpec("s", (FieldSpec("v", "float64", (True,)),)), None, "invalid shape"),
        (None, VideoSourceSpec("c", 8, 8, 30.0, "ffmpeg_h264", True, {"crf": 99}), "crf"),
        (None, VideoSourceSpec("c", 8, 8, 30.0, "opencv_mp4v", True, {"crf": 18}), "only supported"),
    ],
)
def test_validate_source_specs_refuses_schemas_the_store_cannot_write(numeric, video, message) -> None:
    """Bad field names, dtypes, shapes and encoder options are named before any episode is started."""
    with pytest.raises(ValueError, match=message):
        validate_source_specs((numeric,) if numeric else (), (video,) if video else ())


def test_validate_source_specs_accepts_the_real_schema_shapes() -> None:
    """Typical pose, flag and camera specs, with and without encoder options, are accepted."""
    numeric = NumericSourceSpec("t", (FieldSpec("pose", "float64", (4, 4)), FieldSpec("flag", "bool")))
    validate_source_specs(
        (numeric,),
        (VideoSourceSpec("c", 8, 8, 30.0), VideoSourceSpec("d", 8, 8, 30.0, "ffmpeg_h264", False, {"crf": 18})),
    )


def test_removal_deletes_the_ownership_stamp_last_so_an_interrupted_removal_can_be_retried(
    tmp_path: Path, monkeypatch
) -> None:
    """Contents (files and directories) go first; if that stops midway the stamp remains and a retry finishes."""
    manifest = _manifest(0)
    pending = tmp_path / manifest.pending_path
    zarr.open_group(str(pending), mode="w", zarr_format=3).attrs[IDENTITY_ATTR] = manifest.episode_id
    (pending / "video.mp4").write_text("frames")
    (pending / "data").mkdir()
    (pending / "data" / "chunk").write_text("rows")

    def interrupted(path, *args, **kwargs):
        raise OSError("stopped midway")

    monkeypatch.setattr("egocap.recording.episode_layout.shutil.rmtree", interrupted)
    with pytest.raises(OSError, match="stopped midway"):
        remove_owned_episode(tmp_path, manifest)
    assert (pending / "zarr.json").is_file()  # still provably this episode's, so the removal is retryable
    monkeypatch.undo()
    remove_owned_episode(tmp_path, manifest)
    assert not pending.exists()


def test_an_empty_unstamped_pending_directory_is_accepted_only_for_a_removal_in_progress(tmp_path: Path) -> None:
    """Empty residue of an interrupted removal is finished only when the manifest durably intends to remove it."""
    manifest = _manifest(0)
    manifest.storage_phase = "CHECKED"
    pending = tmp_path / manifest.pending_path
    pending.mkdir()
    for intent in (None, FINALIZATION_KEEP, FINALIZATION_RETAIN):  # an emptied directory of another take
        manifest.finalization = intent
        with pytest.raises(PermissionError):
            locate_episode(tmp_path, manifest)
    manifest.finalization = FINALIZATION_REMOVE
    assert locate_episode(tmp_path, manifest) == "pending"
    remove_owned_episode(tmp_path, manifest)
    assert not pending.exists()
    pending.mkdir()
    (pending / "stranger.txt").write_text("not ours")
    with pytest.raises(PermissionError):  # a non-empty unstamped directory is never accepted
        locate_episode(tmp_path, manifest)
    with pytest.raises(PermissionError):
        remove_owned_episode(tmp_path, manifest)
    assert (pending / "stranger.txt").exists()
    manifest.storage_phase = "SEALED"  # nor is an emptied one before the take was checked
    (pending / "stranger.txt").unlink()
    with pytest.raises(PermissionError):
        locate_episode(tmp_path, manifest)


def test_an_older_record_without_a_destination_is_decoded_as_unrouted(tmp_path: Path) -> None:
    """Records from before routing carry no studio or assignment; they read back at the current version."""
    manifest = _manifest(0)
    write_manifest(tmp_path, manifest)
    file = manifest_path(tmp_path, 0)
    data = json.loads(file.read_text())
    for current in ("studio", "assignment", "finalization"):
        del data[current]
    data["schema_version"] = 2
    file.write_text(json.dumps(data))
    decoded = read_manifest(file)
    assert decoded.studio is None and decoded.assignment is None
    assert decoded.schema_version == MANIFEST_SCHEMA_VERSION


def _checked_record(tmp_path: Path, **changes) -> Path:
    """A written manifest for a checked take, adjusted to look like an older or damaged record."""
    manifest = _manifest(0)
    manifest.storage_phase = "CHECKED"
    manifest.quality = {"outcome": "PASSED", "issues": [], "warnings": [], "check_results": []}
    manifest.finalization = FINALIZATION_KEEP
    write_manifest(tmp_path, manifest)
    file = manifest_path(tmp_path, 0)
    data = json.loads(file.read_text())
    data.update(changes)
    data = {k: v for k, v in data.items() if v != "__drop__"}
    file.write_text(json.dumps(data))
    return file


def test_an_older_checked_record_gets_the_intent_its_results_imply(tmp_path: Path) -> None:
    """Normalized once when read, so storage and recovery only ever see the current intent."""
    legacy = {"schema_version": 2, "finalization": "__drop__"}
    assert read_manifest(_checked_record(tmp_path, **legacy)).finalization == FINALIZATION_KEEP
    failed = {"outcome": "FAILED", "issues": ["x"], "warnings": [], "check_results": []}
    assert read_manifest(_checked_record(tmp_path, quality=failed, **legacy)).finalization == FINALIZATION_RETAIN
    rejected = read_manifest(_checked_record(tmp_path, qc_decision="REJECTED_CAMERA", **legacy))
    assert rejected.finalization == FINALIZATION_REMOVE


def test_an_interrupted_older_camera_rejection_cleanup_is_still_recoverable(tmp_path: Path) -> None:
    """Stamp deleted, directory not yet removed, in a record from before intents: the empty residue is finished."""
    file = _checked_record(tmp_path, schema_version=2, finalization="__drop__", qc_decision="REJECTED_CAMERA")
    manifest = read_manifest(file)
    pending = tmp_path / manifest.pending_path
    pending.mkdir()
    assert locate_episode(tmp_path, manifest) == "pending"
    remove_owned_episode(tmp_path, manifest)
    assert not pending.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"finalization": "DELETE"},  # unknown value
        {"finalization": "__drop__"},  # a current-version checked record must carry its intent
        {"route": {"studio": "K", "task": "t"}},  # a field the current record does not have is refused
    ],
)
def test_current_records_are_validated_strictly(tmp_path: Path, changes) -> None:
    with pytest.raises((ValueError, TypeError)):
        read_manifest(_checked_record(tmp_path, **changes))
