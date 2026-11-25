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
    assert [s["sensor_id"] for s in specs] == [0, 1]
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


def test_ipc_display_preserves_eye_order_and_rejects_wrong_dimensions(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/camera_viz/python"))
    from isaaccapture_examples.camera_viz.sources.media_ipc import MediaIpcSource
    from isaac_media_transport.transport import VideoFrame, ReceivedFrame, ReceivedPair

    source = MediaIpcSource(
        {}, {"name": "cam", "width": 8, "height": 4, "stereo": True}
    )
    left, right = object(), object()
    pair = ReceivedPair(
        ReceivedFrame(VideoFrame(8, 4, left, "cuda"), 10, None),
        ReceivedFrame(VideoFrame(8, 4, right, "cuda"), 11, None),
        12,
    )
    monkeypatch.setattr(source._receiver, "latest", lambda: pair)
    frame = source.latest()
    assert frame.image is left and frame.image_right is right
    assert frame.timestamp_ns == 12
    pair.right.frame = VideoFrame(9, 4, right, "cuda")
    with pytest.raises(ValueError, match="dimensions"):
        source.latest()
    monkeypatch.setattr(source._receiver, "latest", lambda: pair.left)
    with pytest.raises(ValueError, match="stereo layout"):
        source.latest()
    monkeypatch.setattr(source._receiver, "latest", lambda: None)
    assert source.latest() is None


def test_bundle_uses_one_shared_latency_policy(monkeypatch, tmp_path):
    import sys
    import yaml
    from remote_assistance import network_setup

    output = tmp_path / "bundle"
    policy = tmp_path / "latency.yaml"
    policy.write_text(
        'enabled: true\nfile_enabled: false\nterminal_enabled: true\nclock:\n  client: "${CLIENT_USER}@${CLIENT_IP}"\n'
    )
    monkeypatch.setenv("CLIENT_USER", "operator")
    monkeypatch.setenv("CLIENT_IP", "192.0.2.2")

    class Provisioner:
        def generate(self, output, *_args, **_kwargs):
            for role in ("host", "client"):
                site = output / role
                site.mkdir(parents=True)
                for name in ("camera.yaml", "session.yaml"):
                    (site / name).write_text("livekit: {}\n")
            return output

    class Transport:
        def provisioner(self):
            return Provisioner()

    monkeypatch.setattr(network_setup, "get_transport", lambda _: Transport())
    monkeypatch.setattr(network_setup, "available_transports", lambda: ["livekit"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "setup",
            "--host-ip",
            "192.0.2.1",
            "--output",
            str(output),
            "--camera-config",
            str(EXAMPLE / "camera-s56-stereo.yaml"),
            "--session-config",
            str(EXAMPLE / "session.yaml"),
            "--latency-config",
            str(policy),
        ],
    )
    network_setup.main()
    host = yaml.safe_load((output / "host/latency.yaml").read_text())
    assert host == yaml.safe_load((output / "client/latency.yaml").read_text())
    assert host["clock"]["client"] == "operator@192.0.2.2"
    assert host["run_id"]
    assert load_config(output / "host/camera.yaml")["livekit"]["latency"] == host
