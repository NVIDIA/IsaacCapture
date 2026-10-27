# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import yaml

from camera_receiver import _load_config, _parse_stream_config


PROFILES = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize(
    "path", sorted(PROFILES.glob("s56-*.yaml")), ids=lambda p: p.stem
)
def test_shared_profiles(path):
    source = yaml.safe_load(path.read_text())
    receiver = _load_config(path)
    assert receiver["receiver"]["port"] == source["network"]["port"]
    for index, (sent, received) in enumerate(
        zip(source["streams"], receiver["streams"], strict=True)
    ):
        stream = _parse_stream_config(received)
        assert (stream.index, stream.name) == (index, sent["id"])
        assert (stream.width, stream.height) == (
            sent["output"]["width"],
            sent["output"]["height"],
        )
        assert stream.stereo == (sent["type"] == "stereo")
        assert stream.ipc_names == (
            ("head.left", "head.right") if stream.stereo else ("head",)
        )
        assert len(sent["cameras"]) == (2 if stream.stereo else 1)


def test_multiple_streams_share_order_geometry_and_port(tmp_path):
    cfg = yaml.safe_load((PROFILES / "s56-stereo-1080p60.yaml").read_text())
    cfg["network"]["port"] = 54321
    cfg["streams"].append(
        dict(
            id="aux",
            type="mono",
            cameras=[dict(device=2)],
            output=dict(width=640, height=480, fps=30, codec="hevc"),
        )
    )
    path = tmp_path / "shared.yaml"
    path.write_text(yaml.safe_dump(cfg))
    received = _load_config(path)
    assert received["receiver"]["port"] == 54321
    aux = _parse_stream_config(received["streams"][1])
    assert (aux.index, aux.width, aux.height, aux.ipc_names) == (1, 640, 480, ("aux",))


@pytest.mark.parametrize("conflict", ["port", "width", "missing_dimensions"])
def test_shared_config_rejects_ambiguous_geometry_or_port(tmp_path, conflict):
    cfg = yaml.safe_load((PROFILES / "s56-stereo-1080p60.yaml").read_text())
    if conflict == "port":
        cfg["receiver"]["port"] = 6000
    elif conflict == "width":
        cfg["streams"][0]["receiver"]["width"] = 640
    else:
        del cfg["streams"][0]["output"]["width"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        _load_config(path)


def test_legacy_receiver_profile(tmp_path):
    cfg = dict(
        receiver=dict(port=6000),
        streams=[dict(index=0, name="old", type="mono", width=640, height=480)],
    )
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert _load_config(path) == cfg
