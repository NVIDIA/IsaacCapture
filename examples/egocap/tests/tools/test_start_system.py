# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import signal
import sys
from pathlib import Path

import pytest
import yaml

# tools/ is a plain scripts directory, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import start_system as ss  # noqa: E402


def _decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def test_publisher_token_is_a_verifiable_hs256_jwt():
    token = ss.publisher_token("k", "s3cret", "room-a", "bridge")
    header, body, signature = token.split(".")
    expected = hmac.new(b"s3cret", f"{header}.{body}".encode(), hashlib.sha256).digest()
    assert _decode(signature) == expected
    claims = json.loads(_decode(body))
    assert claims["iss"] == "k"
    assert claims["sub"] == "bridge"
    assert claims["video"]["room"] == "room-a"
    assert claims["video"]["canPublish"] is True


def test_generated_configs_parse_and_carry_parameters():
    streams = yaml.safe_load(ss.streams_yaml("ws://h:1", "r", "p", "trk", 640, 480, 30))
    stream = streams["livekit"]["streams"][0]
    assert (stream["track"], stream["width"], stream["height"]) == ("trk", 640, 480)

    receiver = yaml.safe_load(ss.receiver_yaml("h", 5555, "/tmp/s.sock", 640, 480))
    assert receiver["receiver"]["port"] == 5555
    assert (receiver["streams"][0]["width"], receiver["streams"][0]["height"]) == (640, 480)

    sender = yaml.safe_load(ss.sender_yaml(5555, 30, 8))
    assert sender["network"]["port"] == 5555


def test_control_token_found_or_absent(tmp_path: Path):
    log = tmp_path / "backend.log"
    log.write_text(f"noise\n{ss.CONTROL_TOKEN_MARKER} abc\n")
    assert ss.control_token(log) == "abc"
    log.write_text("noise only\n")
    assert ss.control_token(log) is None


def test_main_requires_deploy_script_with_orin(monkeypatch, capsys):
    monkeypatch.delenv("EGOCAP_DEPLOY_SCRIPT", raising=False)
    # main() installs a SIGTERM handler; put the original back afterwards.
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    assert ss.main(["--orin", "x@y"]) == 1
    assert "--deploy-script" in capsys.readouterr().err


def test_stack_stop_all_terminates_child(tmp_path: Path):
    stack = ss.Stack(tmp_path)
    child = stack.start("sleeper", ["sleep", "30"])
    assert child.process.poll() is None
    stack.stop_all()
    assert child.process.poll() is not None


def test_wait_until_raises_when_child_exits_early(tmp_path: Path):
    stack = ss.Stack(tmp_path)
    child = stack.start("quick", ["true"])
    try:
        with pytest.raises(ss.StartupError):
            stack.wait_until(child, lambda: False, "ready", 5)
    finally:
        stack.stop_all()
