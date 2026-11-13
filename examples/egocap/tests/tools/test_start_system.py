# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import signal
import sys
import time
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
    streams = yaml.safe_load(ss.streams_yaml("ws://h:1", "r", "p", "trk", 640, 480, 30, "b"))
    stream = streams["livekit"]["streams"][0]
    assert (stream["track"], stream["width"], stream["height"]) == ("trk", 640, 480)

    receiver = yaml.safe_load(ss.receiver_yaml("h", 5555, "/tmp/s.sock", 640, 480, "head"))
    assert receiver["receiver"]["port"] == 5555
    assert (receiver["streams"][0]["width"], receiver["streams"][0]["height"]) == (640, 480)

    assert streams["livekit"]["timestamp_basis"] == "b"
    assert receiver["streams"][0]["name"] == "head"
    assert receiver["streams"][0]["ipc"]["mono"] == "head"


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


def test_main_rejects_a_hostname_orin(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    argv = ["--orin", "user@some-host-name", "--deploy-script", "x", "--bridge", "b", "--livekit-server", "l"]
    assert ss.main(argv) == 1
    assert "IPv4 address" in capsys.readouterr().err


def test_main_refuses_to_start_without_a_built_frontend(monkeypatch, capsys, tmp_path: Path):
    def fail_start(*_, **__):
        raise AssertionError("a child process was started")

    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(workdir))
    monkeypatch.setattr(ss, "FRONTEND_INDEX", tmp_path / "dist" / "index.html")
    monkeypatch.setattr(ss.Stack, "start", fail_start)
    argv = ["--orin", "user@10.0.0.5", "--deploy-script", "x", "--bridge", "b", "--livekit-server", "l"]
    assert ss.main(argv) == 1
    assert "frontend is not built" in capsys.readouterr().err


def test_default_profile_gives_the_head_camera_settings():
    config = ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG)
    assert ss.sender_settings(config) == ss.SenderSettings(
        port=5000, clock_port=5611, fps=30, bitrate_mbps=4, stream_id="head", full_range=True
    )


def test_argus_backend_is_not_full_range():
    config = {**ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG), "capture_backend": "argus"}
    assert ss.sender_settings(config).full_range is False


def test_overrides_replace_only_their_values():
    config = ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG, 60, 9, 6000, 5100)
    assert ss.sender_settings(config) == ss.SenderSettings(
        port=5100, clock_port=6000, fps=60, bitrate_mbps=9, stream_id="head", full_range=True
    )
    partial = ss.sender_settings(ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG, fps=15))
    assert (partial.fps, partial.bitrate_mbps, partial.clock_port, partial.port) == (15, 4, 5611, 5000)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["network"].pop("timing_extension"),
        lambda c: c["network"].pop("clock_port"),
        lambda c: c["streams"][0]["output"].update(fps=0),
        lambda c: c["streams"][0]["output"].pop("fps"),
    ],
)
def test_sender_settings_rejects_an_incomplete_profile(mutate):
    config = ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG)
    mutate(config)
    with pytest.raises(ss.StartupError):
        ss.sender_settings(config)


def test_load_sender_config_rejects_missing_file_and_two_streams(tmp_path: Path):
    with pytest.raises(ss.StartupError):
        ss.load_sender_config(tmp_path / "absent.yaml")
    two = tmp_path / "two.yaml"
    two.write_text("streams:\n  - {id: a}\n  - {id: b}\n")
    with pytest.raises(ss.StartupError):
        ss.load_sender_config(two)


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


POLITE_CHILD = """
import signal, sys, time
name, shared, ready = sys.argv[1:4]
def log(what):
    with open(shared, "a") as f:
        f.write(f"{name} {what} {time.monotonic()}\\n")
def on_int(*_):
    log("int")
    time.sleep(0.3)
    log("exit")
    sys.exit(0)
signal.signal(signal.SIGINT, on_int)
open(ready, "w").close()
time.sleep(60)
"""

STUBBORN_CHILD = """
import signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
open(sys.argv[1], "w").close()
time.sleep(60)
"""


def _await_files(paths: list[Path]) -> None:
    deadline = time.monotonic() + 10
    while not all(p.exists() for p in paths):
        assert time.monotonic() < deadline, "child never became ready"
        time.sleep(0.02)


def _polite(stack: ss.Stack, tmp_path: Path, name: str, **kwargs) -> Path:
    ready = tmp_path / f"{name}.ready"
    stack.start(name, [sys.executable, "-c", POLITE_CHILD, name, str(tmp_path / "events"), str(ready)], **kwargs)
    return ready


def _stubborn(stack: ss.Stack, tmp_path: Path, name: str, **kwargs) -> Path:
    ready = tmp_path / f"{name}.ready"
    stack.start(name, [sys.executable, "-c", STUBBORN_CHILD, str(ready)], **kwargs)
    return ready


@pytest.fixture
def stack(tmp_path: Path):
    stack = ss.Stack(tmp_path)
    yield stack
    # Whatever a test left running must not outlive it.
    for child in stack.children:
        try:
            os.killpg(child.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The test already stopped this child's group; reaping it below is all that is left to do.
            pass
        child.process.wait(timeout=5)


def test_stop_all_signals_in_reverse_order_and_waits_for_each_exit(stack, tmp_path: Path):
    ready = [_polite(stack, tmp_path, n) for n in "abc"]
    _await_files(ready)
    stack.stop_all()
    events = [line.split() for line in (tmp_path / "events").read_text().splitlines()]
    assert [(n, what) for n, what, _ in events] == [
        ("c", "int"),
        ("c", "exit"),
        ("b", "int"),
        ("b", "exit"),
        ("a", "int"),
        ("a", "exit"),
    ]
    stamps = [float(t) for _, _, t in events]
    assert stamps == sorted(stamps)
    assert all(c.process.poll() == 0 for c in stack.children)


def test_child_that_ignores_sigint_is_killed_after_its_grace(stack, tmp_path: Path):
    stubborn = _stubborn(stack, tmp_path, "stubborn", grace_s=0.5)
    polite = _polite(stack, tmp_path, "polite")
    _await_files([stubborn, polite])
    started = time.monotonic()
    stack.stop_all()
    assert time.monotonic() - started < 5
    assert all(c.process.poll() is not None for c in stack.children)
    assert stack.children[0].process.returncode == -signal.SIGKILL
    assert stack.children[1].process.returncode == 0


def test_child_whose_group_is_already_gone_is_treated_as_stopped(stack, tmp_path: Path, monkeypatch):
    ready = [_polite(stack, tmp_path, n) for n in "ab"]
    _await_files(ready)
    real_killpg = os.killpg
    calls: list[int] = []

    def flaky_killpg(pgid: int, sig: int) -> None:
        calls.append(pgid)
        if len(calls) == 1:
            raise ProcessLookupError
        real_killpg(pgid, sig)

    monkeypatch.setattr(ss.os, "killpg", flaky_killpg)
    stack.stop_all()
    assert calls[0] == stack.children[1].process.pid
    # b's group "vanished" at the first call, so it is skipped and only a is signalled.
    assert stack.children[0].process.poll() == 0


def test_grace_is_recorded_per_child(stack, tmp_path: Path):
    ready = [_polite(stack, tmp_path, "a", grace_s=7), _polite(stack, tmp_path, "b")]
    _await_files(ready)
    assert stack.children[0].grace_s == 7
    assert stack.children[1].grace_s == ss.DEFAULT_GRACE_S
    stack.stop_all()


def test_second_ctrl_c_kills_the_child_and_the_rest_at_once(stack, tmp_path: Path, monkeypatch):
    ready = [
        _stubborn(stack, tmp_path, "a", grace_s=30),
        _stubborn(stack, tmp_path, "b", grace_s=30),
        _stubborn(stack, tmp_path, "c", grace_s=30),
    ]
    _await_files(ready)
    last = stack.children[2].process
    real_wait = last.wait
    calls = []

    def interrupted_wait(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(last, "wait", interrupted_wait)
    started = time.monotonic()
    stack.stop_all()
    assert time.monotonic() - started < 5
    assert [c.process.returncode for c in stack.children] == [-signal.SIGKILL] * 3
