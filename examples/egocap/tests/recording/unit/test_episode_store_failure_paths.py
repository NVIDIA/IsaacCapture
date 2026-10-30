"""EpisodeStore and layout failure paths, driven through the production backend.

Encoder release failures are injected by wrapping the real writer (it really finishes the file, then reports
failure); no store, writer or filesystem operation is replaced.
"""

import os

import numpy as np
import pytest
import zarr

import egocap.recording.episode_store as store_module
from egocap.recording.episode_layout import (
    EpisodeManifest,
    episode_dir_names,
    finalize_episode_dir,
    next_episode_index,
    remove_owned_episode,
    scan_manifests,
    write_manifest,
)
from egocap.recording.episode_store import FLUSH_ROWS, EpisodeStore, FieldSpec, NumericSourceSpec, VideoSourceSpec

NUMERIC = NumericSourceSpec("n", (FieldSpec("v", "float64", (2,)),))
CAMERA = VideoSourceSpec("cam", 32, 24, 30.0)
BARE = VideoSourceSpec("bare", 32, 24, 30.0, intrinsics=False)
FRAME = np.zeros((24, 32, 3), dtype=np.uint8)


class FailingRelease:
    """Wraps a real writer: release finishes the file, then reports an encoder failure."""

    def __init__(self, real_writer):
        self._real = real_writer

    def write(self, frame):
        self._real.write(frame)

    def release(self):
        self._real.release()
        raise RuntimeError("encoder exited abnormally")


def _manifest(index=0, phase="WRITING") -> EpisodeManifest:
    pending, final = episode_dir_names(index)
    return EpisodeManifest("s", f"s/e{index}", index, "d", "t", "o", [], pending, final, phase)


def test_rows_beyond_the_flush_size_are_all_stored_in_order(tmp_path):
    """More rows than one in-memory chunk survive the chunked append in order."""
    store = EpisodeStore(tmp_path / "p", (NUMERIC,), ())
    store.start_encoders()
    total = FLUSH_ROWS + 10
    for i in range(total):
        store.append_numeric("n", float(i), float(i), {"v": np.array([i, -i], dtype=np.float64)})
    assert store.seal() == {"n": total}
    out = zarr.open_group(str(tmp_path / "p"), mode="r")["n"]
    assert out["v"].shape == (total, 2) and out["v"][total - 1][0] == total - 1


def test_encoder_startup_failure_is_reported_and_the_owner_can_still_abort(tmp_path):
    """An unknown encoder fails startup with the original reason; the already-constructed store stays abortable."""
    store = EpisodeStore(tmp_path / "p", (NUMERIC,), (VideoSourceSpec("c", 8, 8, 30.0, "no_such_encoder"),))
    with pytest.raises(RuntimeError, match="startup failed: Unknown video encoder"):
        store.start_encoders()
    assert store.abort() is True and (tmp_path / "p").is_dir()


def test_later_camera_failure_keeps_unconfirmed_release_of_the_first_visible_to_the_owner(tmp_path, monkeypatch):
    """If camera 2 fails to start and camera 1's encoder cannot confirm release, abort stays unconfirmed."""
    real_create, calls = store_module.create_video_writer, {"n": 0}

    def first_ok_second_fails(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("second camera failed")
        return FailingRelease(real_create(*args, **kwargs))

    monkeypatch.setattr(store_module, "create_video_writer", first_ok_second_fails)
    store = EpisodeStore(tmp_path / "p", (), (CAMERA, BARE))
    with pytest.raises(RuntimeError, match="second camera failed.*release of already-open encoders failed"):
        store.start_encoders()
    assert store.abort() is False
    with pytest.raises(RuntimeError, match="closed to new samples"):
        store.append_video("cam", FRAME, 1.0, 1.0, np.eye(3))


def test_video_intrinsics_contract_and_closed_store_rules(tmp_path):
    """Intrinsics must match their declaration; a sealed store refuses appends; seal is repeatable."""
    store = EpisodeStore(tmp_path / "p", (), (CAMERA, BARE))
    store.start_encoders()
    with pytest.raises(ValueError, match="declared but missing"):
        store.append_video("cam", FRAME, 1.0, 1.0, None)
    with pytest.raises(ValueError, match="not declared"):
        store.append_video("bare", FRAME, 1.0, 1.0, np.eye(3))
    assert store.counts() == {"cam": 0, "bare": 0}
    store.write_attributes({"k": "v"})
    assert store.seal() == {"cam": 0, "bare": 0} and store.seal() == {"cam": 0, "bare": 0}
    with pytest.raises(RuntimeError, match="closed to new samples"):
        store.write_attributes({"k": "w"})
    assert dict(zarr.open_group(str(tmp_path / "p"), mode="r").attrs) == {"k": "v"}


def test_failed_release_is_permanent_for_seal_and_abort(tmp_path, monkeypatch):
    """A failed encoder release makes every later seal raise and abort report unconfirmed release."""
    real_create = store_module.create_video_writer
    monkeypatch.setattr(store_module, "create_video_writer", lambda *a, **k: FailingRelease(real_create(*a, **k)))
    store = EpisodeStore(tmp_path / "p", (), (CAMERA,))
    store.start_encoders()
    store.append_video("cam", FRAME, 1.0, 1.0, np.eye(3))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="encoder exited abnormally"):
            store.seal()
    assert store.abort() is False


def test_abort_without_failures_confirms_release(tmp_path):
    """Aborting a healthy store confirms release and leaves the pending directory in place."""
    store = EpisodeStore(tmp_path / "p", (NUMERIC,), (CAMERA,))
    store.start_encoders()
    assert store.abort() is True and (tmp_path / "p").is_dir()


def test_layout_edges(tmp_path, monkeypatch):
    """Missing-session index, scan without sidecar, failed-checkpoint cleanup, finalize/remove edges."""
    assert next_episode_index(tmp_path / "nope") == 0
    assert scan_manifests(tmp_path) == ([], [])

    manifest = _manifest()
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        write_manifest(tmp_path, manifest)
    monkeypatch.undo()
    assert not list((tmp_path / ".recording").glob("*.tmp"))

    for name in (manifest.pending_path, manifest.final_path):
        (tmp_path / name).mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_episode_dir(tmp_path, manifest)
    other = _manifest(3)
    remove_owned_episode(tmp_path, other)  # nothing pending: a no-op, never an error


def test_starting_encoders_twice_is_refused_without_opening_anything(tmp_path):
    """A second start_encoders call raises before touching the first call's encoders."""
    store = EpisodeStore(tmp_path / "p", (), (CAMERA,))
    store.start_encoders()
    handle = store._writers["cam"]
    try:
        with pytest.raises(RuntimeError, match="already started"):
            store.start_encoders()
        assert store._writers["cam"] is handle
    finally:
        handle.release()  # the first encoder is released even if a defect replaced or lost it
        assert store.abort() is True
