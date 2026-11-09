# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from egocap.recording.aligned_writer import (
    AlignedWriteError,
    align_kept_episode,
    check_kept_episode,
    publish_aligned_episode,
)
from egocap.recording.episode_layout import (
    FINALIZATION_KEEP,
    IDENTITY_ATTR,
    ROUTE_ATTR,
    EpisodeManifest,
    episode_dir_names,
    manifest_path,
    route_id,
    write_manifest,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "aligned_writer"
META = json.loads((FIXTURES / "scenarios.json").read_text())
INDEX = META["index_source"]
SOURCES = ("head_camera", "wrist_camera", "pico_controller", "glove")
CAMERAS = ("head_camera", "wrist_camera")
TOOLS = Path(__file__).resolve().parents[3] / "recording" / "tools"


def clean_continuity():
    return {name: {"scope": "clock_mapping", "identity": 0, "breaks": [], "known": True} for name in SOURCES}


def good_timing():
    return {"event_time": {name: {"meaning": "capture", "qualified": True} for name in SOURCES}}


def build_raw(root, timing="default"):
    """A raw session in the recording service's layout, with the episode's ``timing`` attribute set."""
    scenario = META["scenarios"]["basic_ds1"]
    root.mkdir(parents=True)
    group = zarr.open_group(str(root), mode="w")
    for key, value in scenario["raw_attrs"].items():
        group.attrs[key] = value
    with np.load(FIXTURES / "raw_inputs.npz") as inputs:
        arrays = {k: inputs[k] for k in inputs.files}
    for episode in (0, 1):
        episode_group = group.require_group(f"episode_{episode}")
        episode_group.attrs[IDENTITY_ATTR] = f"e{episode}"
        for key, value in META["episode_attrs"].items():
            episode_group.attrs[key] = value
        if timing == "default":
            episode_group.attrs["timing"] = good_timing()
        elif timing is not None:
            episode_group.attrs["timing"] = timing
        for key, array in arrays.items():
            tag, source, name = key.split("__")
            if tag != f"e{episode}":
                continue
            source_group = episode_group.require_group(source)
            for attr, value in META["source_attrs"].get(source, {}).items():
                source_group.attrs[attr] = value
            source_group[name] = array
        for camera in CAMERAS:
            (root / f"episode_{episode}" / f"{camera}.mp4").write_bytes(f"video {camera} {episode} ".encode() * 16)
    shutil.copy2(FIXTURES / "config.yaml", root / "config.yaml")
    for name in scenario["sidecars"]:
        shutil.copy2(FIXTURES / name, root / name)


def written_counts(root, index=0):
    """The row counts the recorder would have sealed: what is really on disk for the episode."""
    episode = zarr.open_group(str(root / f"episode_{index}"), mode="r")
    rows = {name: int(episode[name]["capture_time"].shape[0]) for name in episode.group_keys()}
    return {name: {"accepted": n, "written": n} for name, n in rows.items()}


def make_manifest(root, index=0, **overrides):
    """A finalized, kept manifest of the unrouted episode ``index`` whose counts match the rows on disk."""
    pending, final = episode_dir_names(index)
    fields = {
        "session_id": "s",
        "episode_id": f"e{index}",
        "episode_index": index,
        "config_digest": "d",
        "task": "t",
        "operator": "o",
        "sources": [],
        "pending_path": pending,
        "final_path": final,
        "continuity": clean_continuity(),
        "storage_phase": "FINALIZED",
        "finalization": FINALIZATION_KEEP,
        "counts": written_counts(root, index),
    }
    fields.update(overrides)
    return EpisodeManifest(**fields)


def store_manifest(session, index=0, root=None):
    write_manifest(session, make_manifest(root or session, index))


def snapshot(root):
    """Files by hash plus every zarr array's contents."""
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    group = zarr.open_group(str(root), mode="r")
    for episode in sorted(group.group_keys()):
        for source in sorted(group[episode].group_keys()):
            for name in sorted(group[episode][source].array_keys()):
                entries[f"{episode}/{source}/{name}"] = group[episode][source][name][:].tobytes()
    return entries


def tree_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


@pytest.fixture
def raw(tmp_path):
    build_raw(tmp_path / "raw")
    return tmp_path / "raw"


def test_success_publishes_beside_the_raw_session_and_equals_publish(tmp_path, raw):
    got = align_kept_episode(raw, 0, INDEX, make_manifest(raw))
    publish_aligned_episode(
        raw,
        tmp_path / "ref",
        0,
        INDEX,
        qualified=dict.fromkeys(SOURCES, True),
        continuity=clean_continuity(),
    )
    aligned = tmp_path / "raw_aligned"
    assert aligned.is_dir()
    assert snapshot(aligned) == snapshot(tmp_path / "ref")
    assert got.valid_indices.any()


def test_explicit_aligned_session_is_used(tmp_path, raw):
    target = tmp_path / "elsewhere"
    align_kept_episode(raw, 0, INDEX, make_manifest(raw), aligned_session=target)
    assert (target / "episode_0").is_dir()
    assert not (tmp_path / "raw_aligned").exists()


def test_downsample_is_passed_through(tmp_path, raw):
    full = align_kept_episode(raw, 0, INDEX, make_manifest(raw), aligned_session=tmp_path / "a")
    half = align_kept_episode(raw, 0, INDEX, make_manifest(raw), aligned_session=tmp_path / "b", downsample=2)
    assert 0 < int(half.valid_indices.sum()) < int(full.valid_indices.sum())


def bad_timings():
    event = good_timing()["event_time"]
    return {
        "missing": (None, SOURCES),
        "no_event_time": ({}, SOURCES),
        "source_absent": ({"event_time": {k: v for k, v in event.items() if k != "glove"}}, ("glove",)),
        "false": ({"event_time": {**event, "glove": {"qualified": False}}}, ("glove",)),
        "string_true": ({"event_time": {**event, "glove": {"qualified": "true"}}}, ("glove",)),
    }


@pytest.mark.parametrize("case", sorted(bad_timings()))
def test_unqualified_timing_is_refused_and_nothing_is_written(tmp_path, case):
    timing, unqualified = bad_timings()[case]
    build_raw(tmp_path / "raw", timing=timing)
    before = snapshot(tmp_path / "raw")
    with pytest.raises(AlignedWriteError, match="qualified") as info:
        align_kept_episode(tmp_path / "raw", 0, INDEX, make_manifest(tmp_path / "raw"))
    for name in unqualified:
        assert name in str(info.value)
    assert not (tmp_path / "raw_aligned").exists()
    assert snapshot(tmp_path / "raw") == before


def test_a_source_with_recorded_rows_missing_from_the_episode_is_refused_and_nothing_is_written(tmp_path, raw):
    manifest = make_manifest(raw)
    assert manifest.counts["glove"]["written"] > 0
    shutil.rmtree(raw / "episode_0" / "glove")
    before = snapshot(raw)
    files = tree_files(raw)
    with pytest.raises(AlignedWriteError, match="recorded rows are missing"):
        align_kept_episode(raw, 0, INDEX, manifest)
    assert not (tmp_path / "raw_aligned").exists()
    assert tree_files(raw) == files and snapshot(raw) == before


def test_missing_continuity_is_refused(tmp_path, raw):
    with pytest.raises(AlignedWriteError):
        align_kept_episode(raw, 0, INDEX, make_manifest(raw, continuity=None))
    assert not (tmp_path / "raw_aligned").exists()


def test_a_continuity_break_is_refused(tmp_path, raw):
    continuity = clean_continuity()
    continuity["glove"]["breaks"] = [1]
    with pytest.raises(AlignedWriteError):
        align_kept_episode(raw, 0, INDEX, make_manifest(raw, continuity=continuity))
    assert not (tmp_path / "raw_aligned").exists()


def refused(tmp_path, raw, manifest, index=0, match=None):
    """Align is refused before any output and leaves the raw session untouched."""
    before = snapshot(raw)
    with pytest.raises(AlignedWriteError, match=match):
        align_kept_episode(raw, index, INDEX, manifest)
    assert not (tmp_path / "raw_aligned").exists()
    assert snapshot(raw) == before


def test_another_episodes_manifest_with_the_same_index_is_refused(tmp_path, raw):
    other = make_manifest(raw, episode_id="someone-else")
    refused(tmp_path, raw, other, match="episode id differs")


def test_an_episode_without_an_identity_stamp_is_refused(tmp_path, raw):
    zarr.open_group(str(raw / "episode_0"), mode="r+").attrs.pop(IDENTITY_ATTR)
    refused(tmp_path, raw, make_manifest(raw), match="episode id differs")


@pytest.mark.parametrize(
    ("phase", "finalization"),
    [
        ("WRITING", FINALIZATION_KEEP),
        ("SEALED", FINALIZATION_KEEP),
        ("CHECKED", FINALIZATION_KEEP),
        ("FINALIZED", "REMOVE"),
        ("FINALIZED", "RETAIN"),
        ("FINALIZED", None),
    ],
)
def test_a_manifest_that_is_not_a_finalized_kept_episode_is_refused(tmp_path, raw, phase, finalization):
    manifest = make_manifest(raw, storage_phase=phase, finalization=finalization)
    refused(tmp_path, raw, manifest, match="not a finalized kept episode")


@pytest.mark.parametrize("delta", [1, -1])
def test_a_written_count_that_differs_from_the_disk_is_refused(tmp_path, raw, delta):
    manifest = make_manifest(raw)
    manifest.counts["glove"]["written"] += delta
    refused(tmp_path, raw, manifest, match="glove: the rows on disk differ")


def test_a_source_that_wrote_no_rows_may_be_absent_from_the_episode(raw):
    manifest = make_manifest(raw)
    manifest.counts["not_recorded"] = {"accepted": 3, "written": 0}
    assert check_kept_episode(raw, 0, manifest) == []


def test_an_unrouted_index_mismatch_is_refused(tmp_path, raw):
    manifest = make_manifest(raw, episode_index=1)
    refused(tmp_path, raw, manifest, match="episode index is not the manifest's")


def test_an_unreadable_episode_is_a_problem_not_an_exception(raw):
    problems = check_kept_episode(raw, 7, make_manifest(raw))
    assert any("cannot be read" in p for p in problems)
    assert any("episode id differs" in p for p in problems)


def test_a_missing_manifest_attribute_file_reads_as_no_stamp(tmp_path):
    # Malformed route metadata (not JSON) is the same as no stamp.
    build_raw(tmp_path / "raw")
    manifest = make_manifest(tmp_path / "raw", assignment={"group": "Studio/20260101", "slot": 0}, session_id="s")
    (tmp_path / "raw" / "zarr.json").write_text("{not json")
    problems = check_kept_episode(tmp_path / "raw", 0, manifest)
    assert "the episode is not at the routed location the manifest records" in problems


GROUP = "Studio/20260101-000000"


def build_routed(tmp_path, slot=3, stamp="own"):
    """A routed layout: the group directory is the raw session, its episode sits under ``slot``.

    The manifest is indexed by the recording index (5), which differs from the slot.
    """
    build_raw(tmp_path / "raw")
    group_dir = tmp_path / "raw"
    if slot != 0:
        (group_dir / "episode_0").rename(group_dir / f"episode_{slot}")
    attrs = zarr.open_group(str(group_dir), mode="r+").attrs
    if stamp == "own":
        attrs[ROUTE_ATTR] = route_id("s", GROUP)
    elif stamp == "other":
        attrs[ROUTE_ATTR] = route_id("another-session", GROUP)
    manifest = make_manifest(
        group_dir,
        slot,
        episode_index=5,
        pending_path="episode_5.pending",
        final_path="episode_5",
        assignment={"group": GROUP, "slot": slot},
    )
    manifest.episode_id = "e0"
    return group_dir, manifest


def test_a_routed_episode_aligns_at_its_slot(tmp_path):
    group_dir, manifest = build_routed(tmp_path)
    got = align_kept_episode(group_dir, 3, INDEX, manifest)
    assert (tmp_path / "raw_aligned" / "episode_3").is_dir()
    assert got.valid_indices.any()


def test_a_routed_episode_at_another_slot_is_refused(tmp_path):
    group_dir, manifest = build_routed(tmp_path)
    manifest.assignment["slot"] = 4
    refused(tmp_path, group_dir, manifest, index=3, match="not at the routed location")


@pytest.mark.parametrize("stamp", ["missing", "other"])
def test_a_routed_episode_without_this_sessions_route_stamp_is_refused(tmp_path, stamp):
    group_dir, manifest = build_routed(tmp_path, stamp=stamp)
    refused(tmp_path, group_dir, manifest, index=3, match="not at the routed location")


def test_a_routed_episode_named_by_its_recording_index_is_refused(tmp_path):
    group_dir, manifest = build_routed(tmp_path)
    problems = check_kept_episode(group_dir, 5, manifest)
    assert "the episode is not at the routed location the manifest records" in problems


# The CLI lives in tools/, which is not a package.
sys.path.insert(0, str(TOOLS))
import align_episode  # noqa: E402


def run_cli(*args):
    return align_episode.main([str(a) for a in args])


def test_cli_success_prints_the_kept_count(tmp_path, raw, capsys):
    store_manifest(raw)
    assert run_cli(raw, "--episode", 0, "--index-source", INDEX) == 0
    out = capsys.readouterr().out
    assert out.startswith("aligned episode_0: ") and " of " in out
    assert (tmp_path / "raw_aligned" / "episode_0").is_dir()


def test_cli_missing_manifest_exits_1(raw, capsys):
    assert run_cli(raw, "--episode", 0, "--index-source", INDEX) == 1
    assert "cannot read the episode's manifest" in capsys.readouterr().err


def test_cli_garbage_manifest_exits_1(raw, capsys):
    path = manifest_path(raw, 0)
    path.parent.mkdir()
    path.write_text("{not json")
    assert run_cli(raw, "--episode", 0, "--index-source", INDEX) == 1
    assert "cannot read the episode's manifest" in capsys.readouterr().err


def test_cli_refused_episode_exits_1(tmp_path, capsys):
    build_raw(tmp_path / "raw", timing=None)
    store_manifest(tmp_path / "raw")
    assert run_cli(tmp_path / "raw", "--episode", 0, "--index-source", INDEX) == 1
    assert "not aligned:" in capsys.readouterr().err
    assert not (tmp_path / "raw_aligned").exists()


def test_cli_manifest_session_option(tmp_path, raw, capsys):
    other = tmp_path / "other"
    other.mkdir()
    store_manifest(other, root=raw)
    assert run_cli(raw, "--episode", 0, "--index-source", INDEX, "--manifest-session", other) == 0
    assert "aligned episode_0" in capsys.readouterr().out


def test_cli_aligned_session_and_downsample_are_honoured(tmp_path, raw, capsys):
    store_manifest(raw)
    run_cli(raw, "--episode", 0, "--index-source", INDEX, "--aligned-session", tmp_path / "x")
    full = capsys.readouterr().out
    assert (
        run_cli(
            raw, "--episode", 0, "--index-source", INDEX, "--aligned-session", tmp_path / "y", "--downsample", 2
        )
        == 0
    )
    half = capsys.readouterr().out

    def kept(line):
        return int(line.split(": ")[1].split(" of ")[0])

    assert (tmp_path / "x" / "episode_0").is_dir() and (tmp_path / "y" / "episode_0").is_dir()
    assert not (tmp_path / "raw_aligned").exists()
    assert kept(half) < kept(full)


def test_cli_missing_episode_is_an_argument_error(raw):
    with pytest.raises(SystemExit) as info:
        run_cli(raw, "--index-source", INDEX)
    assert info.value.code == 2


def test_cli_routed_episode_with_a_separate_manifest_index(tmp_path, capsys):
    group_dir, manifest = build_routed(tmp_path)
    sessions = tmp_path / "recorded"
    sessions.mkdir()
    write_manifest(sessions, manifest)
    args = (group_dir, "--episode", 3, "--index-source", INDEX, "--manifest-session", sessions)
    assert run_cli(*args, "--manifest-episode", 5) == 0
    assert "aligned episode_3" in capsys.readouterr().out
    assert (tmp_path / "raw_aligned" / "episode_3").is_dir()


def test_cli_routed_episode_without_the_manifest_episode_cannot_find_it(tmp_path, capsys):
    group_dir, manifest = build_routed(tmp_path)
    sessions = tmp_path / "recorded"
    sessions.mkdir()
    write_manifest(sessions, manifest)
    args = (group_dir, "--episode", 3, "--index-source", INDEX, "--manifest-session", sessions)
    assert run_cli(*args) == 1
    assert "cannot read the episode's manifest" in capsys.readouterr().err


def test_cli_refuses_a_manifest_that_does_not_bind(tmp_path, raw, capsys):
    write_manifest(raw, make_manifest(raw, episode_id="someone-else"))
    assert run_cli(raw, "--episode", 0, "--index-source", INDEX) == 1
    assert "episode id differs" in capsys.readouterr().err
    assert not (tmp_path / "raw_aligned").exists()
