# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr

from egocap.recording import aligned_writer
from egocap.recording.aligned_writer import AlignedWriteError, validate_aligned_episode, write_aligned_episode

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


def write_steps(raw, aligned, steps):
    for episode, downsample in steps:
        write_aligned_episode(
            raw, aligned, episode, INDEX, downsample=downsample, qualified=QUALIFIED, continuity=clean_continuity()
        )


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


def golden(name):
    manifest = json.loads((FIXTURES / f"golden_{name}.json").read_text())["entries"]
    with np.load(FIXTURES / f"golden_{name}.npz") as store:
        values = {k: store[k] for k in store.files}
    entries = {}
    for e in manifest:
        if e["kind"] == "file":
            entries[e["path"]] = ("file", e["size"], e["sha256"])
        elif e["kind"] == "root_json":
            entries[e["path"]] = ("root_json", e["attributes"])
        elif e["kind"] == "group":
            entries[e["path"]] = ("group", e["attributes"])
        else:
            entries[e["path"]] = (
                "array",
                e["dtype"],
                tuple(e["shape"]),
                e["attributes"],
                values[e["npz_key"]],
            )
    return entries


@pytest.mark.parametrize("name", sorted(META["scenarios"]))
def test_written_tree_equals_the_expected_tree(tmp_path, name):
    scenario = META["scenarios"][name]
    build_raw(tmp_path / "raw", scenario)
    write_steps(tmp_path / "raw", tmp_path / "aligned", scenario["steps"])

    ours, expected = snapshot(tmp_path / "aligned"), golden(name)
    # Sidecars are carried over byte for byte, whatever header a fixture file has gained.
    for sidecar in ("config.yaml", *scenario["sidecars"]):
        raw_bytes = (tmp_path / "raw" / sidecar).read_bytes()
        expected[sidecar] = ("file", len(raw_bytes), hashlib.sha256(raw_bytes).hexdigest())

    assert set(ours) == set(expected)
    for path, want in expected.items():
        got = ours[path]
        assert got[0] == want[0], path
        if want[0] != "array":
            assert got[1:] == want[1:], path
            continue
        assert got[1:4] == want[1:4], path
        if want[1] in ("bool", "int64"):
            assert np.array_equal(got[4], want[4]), path
        else:
            assert np.allclose(got[4], want[4], rtol=0, atol=1e-12), path


def test_videos_are_hard_linked_to_the_raw_ones(tmp_path):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    write_steps(tmp_path / "raw", tmp_path / "aligned", [(0, 1)])
    for camera in CAMERAS:
        raw = tmp_path / "raw" / "episode_0" / f"{camera}.mp4"
        aligned = tmp_path / "aligned" / "episode_0" / f"{camera}.mp4"
        assert os.stat(raw).st_ino == os.stat(aligned).st_ino
        assert aligned.read_bytes() == video_bytes(camera, 0)


def test_videos_are_copied_when_linking_is_not_possible(tmp_path, monkeypatch):
    real_link = os.link

    def no_link(source, target, *args, **kwargs):
        if str(source).endswith(".mp4"):
            raise OSError("cross-device link")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(aligned_writer.os, "link", no_link)
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    write_steps(tmp_path / "raw", tmp_path / "aligned", [(0, 1)])
    for camera in CAMERAS:
        raw = tmp_path / "raw" / "episode_0" / f"{camera}.mp4"
        aligned = tmp_path / "aligned" / "episode_0" / f"{camera}.mp4"
        assert os.stat(raw).st_ino != os.stat(aligned).st_ino
        assert aligned.read_bytes() == video_bytes(camera, 0)


@pytest.mark.parametrize(
    "qualified",
    [
        {**QUALIFIED, "glove": False},
        {**QUALIFIED, INDEX: False},
        {name: ok for name, ok in QUALIFIED.items() if name != "glove"},
    ],
    ids=["unqualified", "unqualified_index", "not_listed"],
)
def test_an_unqualified_or_unlisted_source_refuses_and_writes_nothing(tmp_path, qualified):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    with pytest.raises(AlignedWriteError, match="qualified"):
        write_aligned_episode(
            tmp_path / "raw", tmp_path / "aligned", 0, INDEX, qualified=qualified, continuity=clean_continuity()
        )
    assert not (tmp_path / "aligned").exists()


def test_a_declared_source_absent_from_the_episode_is_skipped(tmp_path):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"], drop_sources=("glove",))
    # Not recorded, so its qualification is not asked for either.
    qualified = {name: ok for name, ok in QUALIFIED.items() if name != "glove"}
    write_aligned_episode(
        tmp_path / "raw", tmp_path / "aligned", 0, INDEX, qualified=qualified, continuity=clean_continuity()
    )
    episode = zarr.open_group(str(tmp_path / "aligned"), mode="r")["episode_0"]
    assert sorted(episode.group_keys()) == ["head_camera", "pico_controller", "wrist_camera"]


def test_realigning_replaces_the_episode_whole(tmp_path):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    write_steps(tmp_path / "raw", tmp_path / "aligned", [(0, 1)])
    stale = zarr.open_group(str(tmp_path / "aligned"), mode="a")["episode_0"]["glove"]
    stale["stale_array"] = np.arange(3)

    write_steps(tmp_path / "raw", tmp_path / "aligned", [(0, 2)])

    episode = zarr.open_group(str(tmp_path / "aligned"), mode="r")["episode_0"]
    assert "stale_array" not in episode["glove"]
    assert episode["valid_timestamps"].shape == (18,)
    assert (tmp_path / "aligned" / "episode_0" / "head_camera.mp4").read_bytes() == video_bytes("head_camera", 0)


def test_an_existing_target_video_is_replaced_by_the_link(tmp_path):
    source, target = tmp_path / "raw.mp4", tmp_path / "aligned.mp4"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    aligned_writer._link_or_copy(source, target)
    assert target.read_bytes() == b"new"
    assert os.stat(source).st_ino == os.stat(target).st_ino


def test_an_existing_aligned_sidecar_is_not_overwritten(tmp_path):
    build_raw(tmp_path / "raw", META["scenarios"]["attrs_sidecars"])
    (tmp_path / "aligned").mkdir()
    (tmp_path / "aligned" / "head_camera_intrinsics.json").write_text("kept: true\n")
    write_steps(tmp_path / "raw", tmp_path / "aligned", [(0, 1)])
    assert (tmp_path / "aligned" / "head_camera_intrinsics.json").read_text() == "kept: true\n"
    assert any((tmp_path / "aligned" / name).is_file() for name in aligned_writer.ALIGNED_SESSION_SIDECARS)


def test_an_edited_aligned_config_is_replaced_by_the_raw_one(tmp_path):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    write_steps(tmp_path / "raw", tmp_path / "aligned", [(0, 1)])
    (tmp_path / "aligned" / "config.yaml").write_text("edited: true\n")
    write_steps(tmp_path / "raw", tmp_path / "aligned", [(0, 1)])
    assert (tmp_path / "aligned" / "config.yaml").read_bytes() == (tmp_path / "raw" / "config.yaml").read_bytes()


def raw_listing(root):
    """Every path under the raw session with its size and content hash, to compare before and after."""
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        if path.is_file()
        else None
        for path in sorted(root.rglob("*"))
    }


def test_the_aligned_root_is_written_from_a_raw_root_without_zarr_json_and_the_raw_session_gains_no_files(
    tmp_path,
):
    raw = tmp_path / "raw"
    build_raw(raw, META["scenarios"]["basic_ds1"])
    (raw / "zarr.json").unlink()
    before = raw_listing(raw)
    aligned_writer._write_root(raw, tmp_path / "aligned", INDEX)
    assert raw_listing(raw) == before
    assert not (raw / "zarr.json").exists()
    assert dict(zarr.open_group(str(tmp_path / "aligned"), mode="r").attrs) == {"index_source": INDEX}


def test_a_raw_session_without_a_root_zarr_json_aligns_and_is_left_untouched(tmp_path):
    raw = tmp_path / "raw"
    build_raw(raw, META["scenarios"]["basic_ds1"])
    (raw / "zarr.json").unlink()
    before = raw_listing(raw)
    write_steps(raw, tmp_path / "aligned", [(0, 1)])
    assert raw_listing(raw) == before
    assert not (raw / "zarr.json").exists()
    assert zarr.open_group(str(tmp_path / "aligned"), mode="r").attrs["index_source"] == INDEX


@pytest.fixture
def raw_session(tmp_path):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    return tmp_path / "raw"


def test_the_aligned_session_must_be_separate_from_the_raw_session(tmp_path, raw_session):
    (tmp_path / "alias").symlink_to(raw_session, target_is_directory=True)
    (tmp_path / "outer").mkdir()
    (tmp_path / "outer" / "inner_alias").symlink_to(raw_session, target_is_directory=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "up").symlink_to(tmp_path, target_is_directory=True)
    targets = {
        "same_path": raw_session,
        "inside_raw": raw_session / "aligned",
        "inside_episode": raw_session / "episode_0" / "aligned",
        "containing_raw": tmp_path,
        "symlink_alias": tmp_path / "alias",
        "inside_through_symlink": tmp_path / "alias" / "aligned",
        "containing_through_symlink": outside / "up",
    }
    before = raw_listing(raw_session)
    for label, target in targets.items():
        with pytest.raises(AlignedWriteError, match="must be separate"):
            write_aligned_episode(
                raw_session, target, 0, INDEX, qualified=QUALIFIED, continuity=clean_continuity()
            )
        # Nothing is created in, or removed from, the raw session, and its content is byte for byte the same.
        assert raw_listing(raw_session) == before, label
    assert not (raw_session / "aligned").exists()
    assert (raw_session / "episode_0" / "head_camera.mp4").read_bytes() == video_bytes("head_camera", 0)


@pytest.fixture
def aligned_session(tmp_path):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    write_steps(tmp_path / "raw", tmp_path / "aligned", [(0, 1)])
    return tmp_path / "aligned"


def problems(session, episode=0):
    return validate_aligned_episode(session, episode, INDEX)


def test_a_written_episode_validates(aligned_session):
    assert problems(aligned_session) == []


def test_a_missing_episode_group_is_reported(aligned_session):
    assert problems(aligned_session, 7) == ["episode_7 group missing from the aligned session"]


@pytest.mark.parametrize("name", ["valid_indices", "valid_timestamps"])
def test_a_missing_episode_array_is_reported(aligned_session, name):
    del zarr.open_group(str(aligned_session), mode="a")["episode_0"][name]
    assert problems(aligned_session) == [f"episode_0/{name} missing"]


def test_a_missing_index_source_group_is_reported(aligned_session):
    del zarr.open_group(str(aligned_session), mode="a")["episode_0"][INDEX]
    assert problems(aligned_session) == [f"episode_0/{INDEX} group missing"]


def test_a_missing_aligned_frame_indices_is_reported(aligned_session):
    del zarr.open_group(str(aligned_session), mode="a")["episode_0"][INDEX]["aligned_frame_indices"]
    assert problems(aligned_session) == [f"episode_0/{INDEX}/aligned_frame_indices missing"]


def test_an_empty_array_is_reported(aligned_session):
    zarr.open_group(str(aligned_session), mode="a")["episode_0"]["glove"]["empty"] = np.zeros(0)
    assert problems(aligned_session) == ["episode_0/glove/empty is empty"]


def corrupt_chunk(session):
    chunk = session / "episode_0" / "valid_timestamps" / "c" / "0"
    chunk.write_bytes(b"not a zstd frame")


def test_an_unreadable_array_is_reported(aligned_session):
    corrupt_chunk(aligned_session)
    (found,) = problems(aligned_session)
    assert found.startswith("episode_0/valid_timestamps unreadable")


def test_a_large_array_is_not_read_back(aligned_session, monkeypatch):
    corrupt_chunk(aligned_session)
    monkeypatch.setattr(aligned_writer, "_CHECK_MAX_BYTES", 0)
    assert problems(aligned_session) == []


def test_a_write_that_does_not_validate_raises(tmp_path, monkeypatch):
    build_raw(tmp_path / "raw", META["scenarios"]["basic_ds1"])
    real = aligned_writer.align_episode

    def empty_timestamps(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), valid_timestamps=np.zeros(0))

    monkeypatch.setattr(aligned_writer, "align_episode", empty_timestamps)
    with pytest.raises(AlignedWriteError, match="valid_timestamps is empty"):
        write_aligned_episode(
            tmp_path / "raw", tmp_path / "aligned", 0, INDEX, qualified=QUALIFIED, continuity=clean_continuity()
        )


def with_entry(name, **changes):
    """Clean evidence with one source's entry changed."""
    evidence = clean_continuity()
    evidence[name] = {**evidence[name], **changes}
    return evidence


@pytest.mark.parametrize(
    "continuity",
    [
        None,
        {k: v for k, v in clean_continuity().items() if k != "glove"},
        with_entry("glove", known=False),
        with_entry("glove", scope="capture_session"),
        with_entry("glove", breaks=[12]),
        with_entry(INDEX, breaks=[3]),
    ],
    ids=["none", "source_missing", "unknown", "capture_session_scope", "breaks", "index_breaks"],
)
def test_missing_or_broken_continuity_refuses_and_writes_nothing(tmp_path, continuity):
    raw = tmp_path / "raw"
    build_raw(raw, META["scenarios"]["basic_ds1"])
    before = raw_listing(raw)
    with pytest.raises(AlignedWriteError):
        write_aligned_episode(raw, tmp_path / "aligned", 0, INDEX, qualified=QUALIFIED, continuity=continuity)
    assert not (tmp_path / "aligned").exists()
    assert raw_listing(raw) == before


@pytest.mark.parametrize("source", ["glove", INDEX])
def test_a_break_beyond_the_episodes_rows_is_refused_and_writes_nothing(tmp_path, source):
    raw = tmp_path / "raw"
    build_raw(raw, META["scenarios"]["basic_ds1"])
    before = raw_listing(raw)
    with pytest.raises(AlignedWriteError, match=rf"{source}: continuity evidence is incomplete or malformed"):
        write_aligned_episode(
            raw, tmp_path / "aligned", 0, INDEX, qualified=QUALIFIED, continuity=with_entry(source, breaks=[10**6])
        )
    assert not (tmp_path / "aligned").exists()
    assert raw_listing(raw) == before


def test_partial_evidence_without_identity_or_breaks_is_refused(tmp_path):
    raw = tmp_path / "raw"
    build_raw(raw, META["scenarios"]["basic_ds1"])
    partial = {name: {"scope": "clock_mapping", "known": True} for name in QUALIFIED}
    with pytest.raises(AlignedWriteError, match="incomplete or malformed"):
        write_aligned_episode(raw, tmp_path / "aligned", 0, INDEX, qualified=QUALIFIED, continuity=partial)
    assert not (tmp_path / "aligned").exists()
