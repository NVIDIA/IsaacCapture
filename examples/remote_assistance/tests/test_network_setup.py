# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment credentials stay site-specific and use the existing connection API."""

from pathlib import Path
import shlex

import jwt
import pytest
import yaml

from remote_assistance.connection import token_for
from remote_assistance.network_setup import generate

ROOT = Path(__file__).parents[1]


def setup(output, address="192.168.1.20", **kwargs):
    return generate(
        output,
        address,
        ROOT / "session.yaml",
        ROOT.parent / "camera_viz/configs/livekit-webcam.yaml",
        **kwargs,
    )


def test_site_bundles_and_scoped_tokens(tmp_path, monkeypatch):
    output = setup(tmp_path / "network")
    server = yaml.safe_load((output / "edge/livekit.yaml").read_text())
    key, secret = next(iter(server["keys"].items()))
    assert len(secret) >= 32
    assert not server["development"]
    assert server["rtc"]["node_ip"] == "192.168.1.20"
    assert "interfaces" not in server["rtc"]
    assert not (output / "robot/livekit.yaml").exists()
    subjects = set()
    for role in ("edge", "robot"):
        env = output / role / "credentials.env"
        assert env.stat().st_mode & 0o777 == 0o600
        for line in env.read_text().splitlines():
            if line.startswith("export SSL_CERT_FILE="):
                continue
            assignment = shlex.split(line)[1]
            name, value = assignment.split("=", 1)
            monkeypatch.setenv(name, value)
            claims = jwt.decode(value, secret, algorithms=["HS256"], issuer=key)
            subjects.add(claims["sub"])
            grants = claims["video"]
            assert grants["roomJoin"]
            assert not grants.get("roomAdmin", False)
            if name == "RA_SESSION_TOKEN":
                assert grants["canPublishData"] and not grants["canPublish"]
                assert claims["sub"] == role + "-session"
            elif role == "robot":
                assert grants["canPublish"] and not grants["canSubscribe"]
            else:
                assert grants["canSubscribe"] and not grants["canPublish"]
        cfg = yaml.safe_load((output / role / "session.yaml").read_text())
        assert cfg["livekit"]["url"] == "wss://192.168.1.20:8443"
        assert not cfg["livekit"]["development"]
        assert "127.0.0.1" in cfg[role]["subscribe"]
        assert token_for(cfg["livekit"], role + "-session")
        assert secret not in env.read_text()
    assert subjects == {
        "edge-session",
        "robot-session",
        "robot-camera",
        "edge-video-cam",
        "edge-probe",
    }
    with pytest.raises(FileExistsError):
        setup(output, replace=False)


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "0.0.0.0", "224.0.0.1", "169.254.1.1", "::1", "edge.example"],
)
def test_invalid_advertised_address(tmp_path, address):
    with pytest.raises(ValueError):
        setup(tmp_path / "network", address)
    assert not (tmp_path / "network").exists()


def test_default_replacement_preserves_old_bundle_and_rotates_credentials(tmp_path):
    output = setup(tmp_path / "network")
    old = (output / "edge/credentials.env").read_bytes()
    setup(output)
    backups = list(tmp_path.glob("network.backup.*"))
    assert len(backups) == 1
    assert (backups[0] / "edge/credentials.env").read_bytes() == old
    assert (output / "edge/credentials.env").read_bytes() != old
    assert output.stat().st_mode & 0o777 == 0o700


def test_failed_replacement_leaves_existing_bundle_intact(tmp_path, monkeypatch):
    output = setup(tmp_path / "network")
    old = (output / "edge/credentials.env").read_bytes()

    def fail(*args):
        raise OSError("certificate generation failed")

    monkeypatch.setattr("remote_assistance.network_setup.make_certificates", fail)
    with pytest.raises(OSError, match="certificate generation failed"):
        setup(output)
    assert (output / "edge/credentials.env").read_bytes() == old
    assert not list(tmp_path.glob("network.backup.*"))
    assert not list(tmp_path.glob(".network.staging-*"))


def test_replace_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "network"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        setup(output)
    assert output.is_symlink()


@pytest.mark.parametrize("flags", [[], ["--replace"]])
def test_cli_replaces_existing_bundle_by_default(tmp_path, monkeypatch, flags):
    from remote_assistance.network_setup import main

    output = setup(tmp_path / "network")
    old = (output / "edge/credentials.env").read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        ["network_setup", "--edge-ip", "192.168.1.20", "--output", str(output), *flags],
    )
    main()
    backups = list(tmp_path.glob("network.backup.*"))
    assert len(backups) == 1
    assert (backups[0] / "edge/credentials.env").read_bytes() == old
    assert (output / "edge/credentials.env").read_bytes() != old


def test_oak_profiles_preserve_site_settings(tmp_path):
    from remote_assistance.camera_profiles import OAK_MODES
    from remote_assistance.oak_camera import parameters

    output = setup(tmp_path / "network")
    for role in ("edge", "robot"):
        original = yaml.safe_load((output / role / "camera.yaml").read_text())
        for label, dimensions in OAK_MODES.items():
            path = output / role / f"camera-oak-{label}.yaml"
            cfg = yaml.safe_load(path.read_text())
            assert path.stat().st_mode & 0o777 == 0o600
            assert cfg["display"] == original["display"]
            for key in ("url", "room", "token_env", "publisher", "viewer_identity"):
                assert cfg["livekit"].get(key) == original["livekit"].get(key)
            assert cfg["cameras"][0]["name"] == original["cameras"][0]["name"]
            assert parameters(cfg["livekit"], cfg["cameras"][0])[1:4] == dimensions
            assert cfg["livekit"]["video_decoder"] == "auto"


def test_existing_bundle_profiles_do_not_rotate_credentials(tmp_path, monkeypatch):
    from remote_assistance.camera_profiles import main

    output = setup(tmp_path / "network")
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    for path in (
        *output.glob("*/camera-oak-*.yaml"),
        *output.glob("*/camera-s56-*.yaml"),
    ):
        path.unlink()
    monkeypatch.setattr("sys.argv", ["camera_profiles", "--bundle", str(output)])
    main()
    for path, content in before.items():
        assert path.read_bytes() == content


def test_s56_profiles_preserve_auth_and_select_argus(tmp_path):
    from remote_assistance.camera_profiles import S56_MODES
    from remote_assistance.gst_camera import parameters

    output = setup(tmp_path / "network")
    for role in ("edge", "robot"):
        original = yaml.safe_load((output / role / "camera.yaml").read_text())
        for label, (width, height, fps, mode) in S56_MODES.items():
            path = output / role / f"camera-s56-{label}.yaml"
            cfg = yaml.safe_load(path.read_text())
            assert path.stat().st_mode & 0o777 == 0o600
            assert cfg["display"] == original["display"]
            for key in ("url", "room", "token_env", "publisher", "viewer_identity"):
                assert cfg["livekit"].get(key) == original["livekit"].get(key)
            spec = cfg["cameras"][0]
            assert spec["name"] == original["cameras"][0]["name"]
            assert spec["sensor_mode"] == mode and spec["flip_method"] == 2
            assert parameters(cfg["livekit"], spec)[1:4] == (width, height, fps)
            assert cfg["livekit"]["video_encoder"] == "gstreamer"
