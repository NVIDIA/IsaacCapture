# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# The read pattern of a downstream consumer of an aligned session, re-implemented here from its description:
# open the aligned root; per episode read the boolean ``valid_indices`` and ``valid_timestamps``; per source
# read every ``<field>_interp`` array (one row per aligned timestamp) and, for each camera, the integer
# ``aligned_frame_indices`` (also one per aligned timestamp) that select frames of the ``<camera>.mp4`` stored
# beside the arrays; the root attribute ``index_source`` names the camera whose frames define the timeline.

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import zarr

from egocap.recording.aligned_writer import write_aligned_episode

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "aligned_writer"
META = json.loads((FIXTURES / "scenarios.json").read_text())
INDEX = META["index_source"]
CAMERAS = ("head_camera", "wrist_camera")
QUALIFIED = dict.fromkeys((*CAMERAS, "pico_controller", "glove"), True)
CONTINUITY = {name: {"scope": "clock_mapping", "identity": 0, "breaks": [], "known": True} for name in QUALIFIED}
RAW_ATTRS = {"task_description": "pick and place", "operator": {"id": 7, "tags": ["a", "b"]}}


def build_raw(root):
    """A raw session with non-empty root attributes and real videos, one frame per recorded camera sample."""
    group = zarr.open_group(str(root), mode="w")
    for key, value in RAW_ATTRS.items():
        group.attrs[key] = value
    with np.load(FIXTURES / "raw_inputs.npz") as inputs:
        arrays = {k: inputs[k] for k in inputs.files}
    for episode in (0, 1):
        episode_group = group.require_group(f"episode_{episode}")
        for key, value in META["episode_attrs"].items():
            episode_group.attrs[key] = value
        for key, array in arrays.items():
            tag, source, name = key.split("__")
            if tag == f"e{episode}":
                episode_group.require_group(source)[name] = array
        for camera in CAMERAS:
            frames = len(episode_group[camera]["capture_time"][:])
            writer = cv2.VideoWriter(
                str(root / f"episode_{episode}" / f"{camera}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48)
            )
            for _ in range(frames):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
    shutil.copy2(FIXTURES / "config.yaml", root / "config.yaml")


def video_frame_count(path):
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


@pytest.fixture(scope="module")
def aligned(tmp_path_factory):
    base = tmp_path_factory.mktemp("session")
    build_raw(base / "raw")
    for episode in (0, 1):
        write_aligned_episode(
            base / "raw", base / "aligned", episode, INDEX, qualified=QUALIFIED, continuity=CONTINUITY
        )
    return base / "aligned"


@pytest.mark.parametrize("episode", [0, 1])
def test_a_consumer_reads_the_aligned_episode_by_array_name(aligned, episode):
    root = zarr.open_group(str(aligned), mode="r")
    assert root.attrs["index_source"] == INDEX
    # The raw root's own attributes are carried over beside the index source.
    assert {k: root.attrs[k] for k in RAW_ATTRS} == RAW_ATTRS

    group = root[f"episode_{episode}"]
    valid_indices = np.asarray(group["valid_indices"])
    timestamps = np.asarray(group["valid_timestamps"])
    assert valid_indices.dtype == np.bool_
    assert timestamps.ndim == 1 and np.all(np.diff(timestamps) > 0)
    rows = timestamps.shape[0]

    interp_seen = 0
    for source in group.group_keys():
        for name in group[source].array_keys():
            if name.endswith("_interp"):
                interp_seen += 1
                assert np.asarray(group[source][name]).shape[0] == rows, (source, name)
                # A resampled field is read under its ``_interp`` name only.
                assert name[: -len("_interp")] not in group[source], (source, name)
    assert interp_seen

    for camera in CAMERAS:
        indices = np.asarray(group[camera]["aligned_frame_indices"])
        assert np.issubdtype(indices.dtype, np.integer) and indices.shape == (rows,)
        video = aligned / f"episode_{episode}" / f"{camera}.mp4"
        assert video.is_file()
        assert indices.min() >= 0 and indices.max() < video_frame_count(video)
    # The index camera plays every frame of the valid range once, in order.
    index_frames = np.asarray(group[INDEX]["aligned_frame_indices"])
    assert np.array_equal(np.diff(index_frames), np.ones(rows - 1, dtype=index_frames.dtype))
