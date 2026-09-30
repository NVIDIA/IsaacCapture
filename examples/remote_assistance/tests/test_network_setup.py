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


def setup(output, address="192.168.1.20"):
    return generate(
        output,
        address,
        ROOT / "session.yaml",
        ROOT.parent / "camera_viz/configs/livekit-webcam.yaml",
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
        setup(output)


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "0.0.0.0", "224.0.0.1", "169.254.1.1", "::1", "edge.example"],
)
def test_invalid_advertised_address(tmp_path, address):
    with pytest.raises(ValueError):
        setup(tmp_path / "network", address)
    assert not (tmp_path / "network").exists()
