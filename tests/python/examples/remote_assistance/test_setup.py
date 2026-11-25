# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check shipped presets and isolated credential bundles without cameras."""

import importlib.util
import json
import os
import base64

import pytest
from repo_paths import repo_root
from isaac_media_transport.stereo import camera_specs
from isaac_transport_livekit.network_setup import _generate_bundle
from remote_assistance.__main__ import load_config, select_camera
from remote_assistance.gst_camera import parameters
from remote_assistance.gst_sources import camera_source

EXAMPLE = repo_root() / "examples/remote_assistance"


def test_sensing_preset_can_construct_both_camera_sources():
    config = load_config(EXAMPLE / "camera-s56-stereo.yaml")
    specs = camera_specs(config["livekit"], select_camera(config))
    assert [s["sensor_id"] for s in specs] == [1, 0]
    for spec in specs:
        codec, width, height, fps, bitrate = parameters(config["livekit"], spec)
        assert (codec, width, height, fps, bitrate) == (
            "h264",
            2560,
            1984,
            30,
            15000000,
        )
        source = camera_source(spec)
        assert source.controls["exposuretimerange"] == "9000000 9000000"
        assert "nvarguscamerasrc" in source.pipeline(width, height, fps)


def test_unknown_camera_is_rejected_before_gstreamer_import():
    with pytest.raises(ValueError, match="Unsupported GStreamer camera"):
        camera_source({"type": "unknown"})


def test_bundle_credentials_are_scoped_and_replacement_preserves_previous(tmp_path):
    bundle = tmp_path / "bundle"
    args = (
        bundle,
        "192.0.2.1",
        EXAMPLE / "session.yaml",
        EXAMPLE / "camera-synthetic.yaml",
    )
    _generate_bundle(*args)
    original = (bundle / "client/credentials.env").read_bytes()
    for role in ("host", "client"):
        credentials = bundle / role / "credentials.env"
        assert credentials.stat().st_mode & 0o777 == 0o600
        assert (bundle / role).stat().st_mode & 0o777 == 0o700
        values = dict(
            line.removeprefix("export ").split("=", 1)
            for line in credentials.read_text().splitlines()
        )
        token = values["RA_CAMERA_TOKEN"]
        claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "==="))
        assert claims["video"]["canPublish"] == (role == "client")
        assert claims["video"]["canSubscribe"] == (role == "host")
        assert "ca.key" not in os.listdir(bundle / "client")
    _generate_bundle(*args)
    backups = list(tmp_path.glob("bundle.backup.*"))
    assert len(backups) == 1
    assert (backups[0] / "client/credentials.env").read_bytes() == original
    assert (bundle / "client/credentials.env").read_bytes() != original


def test_external_plugin_is_discovered_without_importing_its_sdk(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "installer", EXAMPLE / "install_transport_plugins.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    external = tmp_path / "external"
    external.mkdir()
    (external / "pyproject.toml").write_text(
        '[project.entry-points."isaac_media_bridge.transports.v1"]\nsample="missing_sdk:plugin"\n'
    )
    catalog = installer.projects(repo_root(), [external])
    assert catalog == {
        "livekit": repo_root() / "src/plugins/transport_livekit",
        "sample": external,
    }
