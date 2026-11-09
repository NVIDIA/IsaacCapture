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

from egocap.recording.aligned_writer import AlignedWriteError, align_kept_episode, publish_aligned_episode
from egocap.recording.episode_layout import EpisodeManifest, episode_dir_names, manifest_path, write_manifest

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


def store_manifest(session, index=0):
    pending, final = episode_dir_names(index)
    write_manifest(
        session,
        EpisodeManifest(
            session_id="s",
            episode_id=f"e{index}",
            episode_index=index,
            config_digest="d",
            task="t",
            operator="o",
            sources=[],
            pending_path=pending,
            final_path=final,
            continuity=clean_continuity(),
        ),
    )


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
    got = align_kept_episode(raw, 0, INDEX, clean_continuity())
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
    align_kept_episode(raw, 0, INDEX, clean_continuity(), aligned_session=target)
    assert (target / "episode_0").is_dir()
    assert not (tmp_path / "raw_aligned").exists()


def test_downsample_is_passed_through(tmp_path, raw):
    full = align_kept_episode(raw, 0, INDEX, clean_continuity(), aligned_session=tmp_path / "a")
    half = align_kept_episode(raw, 0, INDEX, clean_continuity(), aligned_session=tmp_path / "b", downsample=2)
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
        align_kept_episode(tmp_path / "raw", 0, INDEX, clean_continuity())
    for name in unqualified:
        assert name in str(info.value)
    assert not (tmp_path / "raw_aligned").exists()
    assert snapshot(tmp_path / "raw") == before


def test_missing_continuity_is_refused(tmp_path, raw):
    with pytest.raises(AlignedWriteError):
        align_kept_episode(raw, 0, INDEX, None)
    assert not (tmp_path / "raw_aligned").exists()


def test_a_continuity_break_is_refused(tmp_path, raw):
    continuity = clean_continuity()
    continuity["glove"]["breaks"] = [1]
    with pytest.raises(AlignedWriteError):
        align_kept_episode(raw, 0, INDEX, continuity)
    assert not (tmp_path / "raw_aligned").exists()


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
    store_manifest(other)
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
