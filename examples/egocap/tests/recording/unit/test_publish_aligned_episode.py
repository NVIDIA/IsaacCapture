# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr

from egocap.recording import aligned_writer
from egocap.recording.aligned_writer import (
    AlignedWriteError,
    publish_aligned_episode,
    validate_aligned_episode,
    write_aligned_episode,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "aligned_writer"
META = json.loads((FIXTURES / "scenarios.json").read_text())
INDEX = META["index_source"]
QUALIFIED = dict.fromkeys(("head_camera", "wrist_camera", "pico_controller", "glove"), True)
CAMERAS = ("head_camera", "wrist_camera")


def clean_continuity(sources=tuple(QUALIFIED)):
    """Continuity evidence that vouches for every source: the clock mapping, no restart, known."""
    return {name: {"scope": "clock_mapping", "identity": 0, "breaks": [], "known": True} for name in sources}


def video_bytes(source, episode):
    return (f"dummy video {source} episode {episode} ".encode()) * 16


def build_raw(root, scenario, drop_sources=()):
    """The raw session a scenario starts from, in the layout the recording service writes."""
    root.mkdir(parents=True)
    group = zarr.open_group(str(root), mode="w")
    for key, value in scenario["raw_attrs"].items():
        group.attrs[key] = value
    with np.load(FIXTURES / "raw_inputs.npz") as inputs:
        arrays = {k: inputs[k] for k in inputs.files}
    for episode in (0, 1):
        episode_group = group.require_group(f"episode_{episode}")
        for key, value in META["episode_attrs"].items():
            episode_group.attrs[key] = value
        for key, array in arrays.items():
            tag, source, name = key.split("__")
            if tag != f"e{episode}" or source in drop_sources:
                continue
            source_group = episode_group.require_group(source)
            for attr, value in META["source_attrs"].get(source, {}).items():
                source_group.attrs[attr] = value
            source_group[name] = array
        for camera in CAMERAS:
            (root / f"episode_{episode}" / f"{camera}.mp4").write_bytes(video_bytes(camera, episode))
    shutil.copy2(FIXTURES / "config.yaml", root / "config.yaml")
    for name in scenario["sidecars"]:
        shutil.copy2(FIXTURES / name, root / name)


def snapshot(root):
    """Everything a reader of an aligned session sees: files, zarr nodes and their contents."""
    entries = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        # Chunk files and node metadata belong to the zarr nodes, which are compared through zarr below.
        if path.is_file() and path.name != "zarr.json" and "c" not in rel.parts[:-1]:
            entries[rel.as_posix()] = ("file", path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    root_json = json.loads((root / "zarr.json").read_text())
    entries["zarr.json"] = ("root_json", root_json["attributes"])

    def walk(group, prefix):
        for name in sorted(group.group_keys()):
            entries[f"{prefix}{name}"] = ("group", dict(group[name].attrs))
            walk(group[name], f"{prefix}{name}/")
        for name in sorted(group.array_keys()):
            array = group[name]
            entries[f"{prefix}{name}"] = (
                "array",
                str(array.dtype),
                tuple(array.shape),
                dict(array.attrs),
                array[:],
            )

    walk(zarr.open_group(str(root), mode="r"), "")
    return entries


def publish(raw, aligned, episode=0, downsample=1, qualified=QUALIFIED, continuity="clean"):
    return publish_aligned_episode(
        raw,
        aligned,
        episode,
        INDEX,
        downsample=downsample,
        qualified=qualified,
        continuity=clean_continuity() if continuity == "clean" else continuity,
    )


def write_control(raw, target, episode=0, downsample=1):
    write_aligned_episode(
        raw, target, episode, INDEX, downsample=downsample, qualified=QUALIFIED, continuity=clean_continuity()
    )


def listing(root):
    """Every path under ``root`` (hidden ones included) with its size, to compare a tree before and after."""
    if not root.exists():
        return None
    return {p.relative_to(root).as_posix(): (p.stat().st_size if p.is_file() else None) for p in root.rglob("*")}


def raw_state(root):
    """Recursive listing plus a content hash of every file, for proving the raw session was not touched."""
    state = {}
    for p in sorted(root.rglob("*")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        state[p.relative_to(root).as_posix()] = digest
    return state


def assert_no_leftovers(aligned):
    assert not (aligned.parent / f".{aligned.name}.staging").exists()


def assert_snapshots_equal(got, want):
    assert set(got) == set(want)
    for path, expected in want.items():
        actual = got[path]
        assert actual[0] == expected[0], path
        if expected[0] != "array":
            assert actual[1:] == expected[1:], path
        else:
            assert actual[1:4] == expected[1:4], path
            assert np.array_equal(actual[4], expected[4]), path


@pytest.fixture
def raw(tmp_path):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    return tmp_path / "raw"


@pytest.mark.parametrize("episode", [0, 1])
def test_published_tree_equals_what_write_aligned_episode_produces(tmp_path, raw, episode):
    write_control(raw, tmp_path / "ref", episode, downsample=2)
    before = raw_state(raw)
    publish(raw, tmp_path / "aligned", episode, downsample=2)

    assert_snapshots_equal(snapshot(tmp_path / "aligned"), snapshot(tmp_path / "ref"))
    assert_no_leftovers(tmp_path / "aligned")
    assert validate_aligned_episode(tmp_path / "aligned", episode, INDEX) == []
    assert raw_state(raw) == before


def test_publish_returns_the_alignment(tmp_path, raw):
    aligned = publish(raw, tmp_path / "aligned")
    again = write_aligned_episode(
        raw, tmp_path / "ref", 0, INDEX, qualified=QUALIFIED, continuity=clean_continuity()
    )
    assert np.array_equal(aligned.valid_indices, again.valid_indices)
    assert np.array_equal(aligned.valid_timestamps, again.valid_timestamps)


def test_published_videos_are_hard_links_to_the_raw_ones(tmp_path, raw):
    publish(raw, tmp_path / "aligned")
    for camera in CAMERAS:
        raw_video = raw / "episode_0" / f"{camera}.mp4"
        aligned_video = tmp_path / "aligned" / "episode_0" / f"{camera}.mp4"
        assert os.stat(raw_video).st_ino == os.stat(aligned_video).st_ino
        assert aligned_video.read_bytes() == video_bytes(camera, 0)


def test_published_videos_are_copies_when_linking_is_not_possible(tmp_path, raw, monkeypatch):
    real_link = os.link

    def no_link(source, target, *args, **kwargs):
        if str(source).endswith(".mp4"):
            raise OSError("cross-device link")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(aligned_writer.os, "link", no_link)
    publish(raw, tmp_path / "aligned")
    for camera in CAMERAS:
        raw_video = raw / "episode_0" / f"{camera}.mp4"
        aligned_video = tmp_path / "aligned" / "episode_0" / f"{camera}.mp4"
        assert os.stat(raw_video).st_ino != os.stat(aligned_video).st_ino
        assert aligned_video.read_bytes() == video_bytes(camera, 0)


def test_an_existing_episode_is_refused_and_left_untouched(tmp_path, raw):
    aligned = tmp_path / "aligned"
    publish(raw, aligned, 0)
    before = snapshot(aligned)
    files = listing(aligned)

    with pytest.raises(AlignedWriteError, match="already exists"):
        publish(raw, aligned, 0, downsample=2)

    assert listing(aligned) == files
    assert_snapshots_equal(snapshot(aligned), before)
    assert_no_leftovers(aligned)


def test_a_different_episode_publishes_beside_the_first(tmp_path, raw):
    aligned = tmp_path / "aligned"
    publish(raw, aligned, 0, downsample=2)
    publish(raw, aligned, 1, downsample=2)

    write_control(raw, tmp_path / "ref", 0, downsample=2)
    write_control(raw, tmp_path / "ref", 1, downsample=2)
    assert_snapshots_equal(snapshot(aligned), snapshot(tmp_path / "ref"))
    assert_no_leftovers(aligned)


def with_entry(source, **changes):
    evidence = clean_continuity()
    evidence[source] = {**evidence[source], **changes}
    return evidence


REFUSALS = {
    "unqualified": {"qualified": {**QUALIFIED, "glove": False}},
    "unqualified_index": {"qualified": {**QUALIFIED, INDEX: False}},
    "continuity_none": {"continuity": None},
    "break_in_evidence": {"continuity": with_entry("glove", breaks=[3])},
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
@pytest.mark.parametrize("existing", [False, True])
def test_a_refused_publication_leaves_the_aligned_session_unchanged(tmp_path, raw, case, existing):
    aligned = tmp_path / "aligned"
    if existing:
        publish(raw, aligned, 1)
    before, raw_before = listing(aligned), raw_state(raw)
    snap = snapshot(aligned) if existing else None

    with pytest.raises(AlignedWriteError):
        publish(raw, aligned, 0, **REFUSALS[case])

    assert listing(aligned) == before
    if existing:
        assert_snapshots_equal(snapshot(aligned), snap)
    else:
        assert not aligned.exists()
    assert not (tmp_path / ".aligned.staging").exists()
    assert raw_state(raw) == raw_before


def test_an_aligned_path_aliasing_the_raw_session_is_refused(tmp_path, raw):
    (tmp_path / "alias").symlink_to(raw, target_is_directory=True)
    before = raw_state(raw)
    for target in (raw, tmp_path / "alias", raw / "inside"):
        with pytest.raises(AlignedWriteError, match="separate"):
            publish(raw, target)
    assert raw_state(raw) == before


def test_a_staging_path_that_is_the_raw_session_is_refused_and_not_deleted(tmp_path):
    raw_named_like_staging = tmp_path / ".aligned.staging"
    build_raw(raw_named_like_staging, META["scenarios"]["basic_ds1"])
    before = raw_state(raw_named_like_staging)
    with pytest.raises(AlignedWriteError, match="separate"):
        publish(raw_named_like_staging, tmp_path / "aligned")
    assert raw_state(raw_named_like_staging) == before
    assert not (tmp_path / "aligned").exists()


def test_a_failed_rename_removes_the_staging_tree(tmp_path, raw, monkeypatch):
    aligned = tmp_path / "aligned"
    publish(raw, aligned, 1)
    before = listing(aligned)

    def failing_rename(source, target):
        raise OSError("simulated failure")

    monkeypatch.setattr(aligned_writer.os, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated failure"):
        publish(raw, aligned, 0)
    monkeypatch.undo()

    assert listing(aligned) == before
    assert_no_leftovers(aligned)
    publish(raw, aligned, 0)
    assert validate_aligned_episode(aligned, 0, INDEX) == []


def test_a_leftover_staging_tree_is_discarded(tmp_path, raw):
    aligned = tmp_path / "aligned"
    staging = tmp_path / ".aligned.staging"
    (staging / "episode_0").mkdir(parents=True)
    (staging / "episode_0" / "junk.bin").write_bytes(b"half written")

    publish(raw, aligned, 0)

    assert not staging.exists()
    assert not (aligned / "episode_0" / "junk.bin").exists()
    assert validate_aligned_episode(aligned, 0, INDEX) == []


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_a_failure_after_staging_started_removes_the_staging_tree_and_propagates(
    tmp_path, raw, monkeypatch, error
):
    aligned = tmp_path / "aligned"
    publish(raw, aligned, 1)
    before = listing(aligned)
    seen = {}

    def fail_in_validation(session, episode, index_source):
        seen["staged"] = session.exists()
        raise error("boom")

    monkeypatch.setattr(aligned_writer, "validate_aligned_episode", fail_in_validation)
    with pytest.raises(error, match="boom"):
        publish(raw, aligned, 0)

    # Staging existed when the failure hit, and nothing of it is left or reached the aligned session.
    assert seen["staged"]
    assert not (tmp_path / ".aligned.staging").exists()
    assert listing(aligned) == before


def test_a_refused_run_creates_nothing_inside_the_aligned_session(tmp_path, raw):
    aligned = tmp_path / "aligned"
    publish(raw, aligned, 1)
    before = listing(aligned)
    with pytest.raises(AlignedWriteError):
        publish(raw, aligned, 0, continuity=None)
    assert listing(aligned) == before
    # A never-created session stays absent.
    with pytest.raises(AlignedWriteError):
        publish(raw, tmp_path / "fresh", 0, continuity=None)
    assert not (tmp_path / "fresh").exists()
    assert not (tmp_path / ".fresh.staging").exists()
