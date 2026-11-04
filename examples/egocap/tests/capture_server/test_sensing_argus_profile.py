# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The SENSING Argus rig profile and its guide: structure only, since no camera or Orin is involved."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

DIRECTORY = Path(__file__).resolve().parents[2] / "capture_server" / "sensing_argus"
PROFILE = DIRECTORY / "shw5g-dual-mono-2560x1984-30.yaml"
PLACEHOLDER = "SET_FROM_LIST_MODES"


@pytest.fixture(scope="module")
def profile() -> dict:
    return yaml.safe_load(PROFILE.read_text())


def test_every_stream_is_a_single_camera_mono_hevc_stream(profile):
    assert profile["version"] == 1
    assert len(profile["streams"]) == 2
    for stream in profile["streams"]:
        assert stream["type"] == "mono" and len(stream["cameras"]) == 1
        assert stream["output"]["codec"] == "hevc"
        assert (stream["output"]["width"], stream["output"]["height"]) == (2560, 1984)
        # Native geometry on the first run: no remap, no rotation, no resize.
        assert "remap" not in stream["output"] and stream["output"]["rotation"] == 0


def test_stream_ids_are_unique_and_are_the_published_ipc_names(profile):
    ids = [s["id"] for s in profile["streams"]]
    assert len(set(ids)) == len(ids)
    assert [s["receiver"]["ipc"]["mono"] for s in profile["streams"]] == ids


def test_each_stream_has_its_own_camera_device(profile):
    devices = [s["cameras"][0]["device"] for s in profile["streams"]]
    assert len(set(devices)) == len(devices)


def test_no_mode_is_guessed(profile):
    # The sender refuses a placeholder, so nothing starts until --list-modes has been read on the rig.
    assert [s["cameras"][0]["mode"] for s in profile["streams"]] == [PLACEHOLDER] * 2


def test_the_receiver_feed_matches_the_camera_streamer_defaults(profile):
    assert profile["network"]["port"] == 5000
    assert profile["ipc"]["socket_path"] == "/tmp/isaac-camera-streamer.sock"


def test_the_guide_serves_every_stream_to_the_capture_server():
    guide = (DIRECTORY / "README.md").read_text()
    ids = [s["id"] for s in yaml.safe_load(PROFILE.read_text())["streams"]]
    assert "--gmsl-ipc /tmp/isaac-camera-streamer.sock" in guide
    for stream_id in ids:
        assert f"={stream_id}'" in guide
    assert PROFILE.name in guide
