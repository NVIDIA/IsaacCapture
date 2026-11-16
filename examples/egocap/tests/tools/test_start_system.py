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
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

# tools/ is a plain scripts directory, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import start_system as ss  # noqa: E402

EXAMPLE_STATION = ss.DEFAULT_STATION_CONFIG
EXAMPLE_CAMERAS = [
    source["name"]
    for source in next(iter(yaml.safe_load(EXAMPLE_STATION.read_text())["stations"].values()))["video_sources"]
]
CAMERA_ARGS = [arg for name in EXAMPLE_CAMERAS for arg in ("--capture-camera", f"{name}=dev0")]


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
    streams = yaml.safe_load(ss.streams_yaml("ws://h:1", "r", "p", "trk", 640, 480, 30, "b", "head"))
    stream = streams["livekit"]["streams"][0]
    assert (stream["track"], stream["width"], stream["height"]) == ("trk", 640, 480)

    receiver = yaml.safe_load(ss.receiver_yaml("h", 5555, "/tmp/s.sock", 640, 480, "head"))
    assert receiver["receiver"]["port"] == 5555
    assert (receiver["streams"][0]["width"], receiver["streams"][0]["height"]) == (640, 480)

    assert streams["livekit"]["timestamp_basis"] == "b"
    assert stream["name"] == "head"
    assert receiver["streams"][0]["name"] == "head"
    assert receiver["streams"][0]["ipc"]["mono"] == "head"


def test_control_token_found_or_absent(tmp_path: Path):
    log = tmp_path / "backend.log"
    log.write_text(f"noise\n{ss.CONTROL_TOKEN_MARKER} abc\n")
    assert ss.control_token(log) == "abc"
    log.write_text("noise only\n")
    assert ss.control_token(log) is None


def test_main_requires_deploy_script_with_orin(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.delenv("EGOCAP_DEPLOY_SCRIPT", raising=False)
    # signal.signal is stubbed so main()'s SIGTERM handler is never installed in the test process.
    installed: list[tuple] = []
    monkeypatch.setattr(signal, "signal", lambda *args: installed.append(args))
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    assert ss.main(["--orin", "x@y"]) == 1
    assert "--deploy-script" in capsys.readouterr().err


def test_main_rejects_a_hostname_orin(monkeypatch, capsys, tmp_path: Path):
    installed: list[tuple] = []
    monkeypatch.setattr(signal, "signal", lambda *args: installed.append(args))
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    argv = [
        "--orin",
        "user@some-host-name",
        "--deploy-script",
        "x",
        "--bridge",
        "b",
        "--livekit-server",
        "l",
        *CAMERA_ARGS,
    ]
    assert ss.main(argv) == 1
    assert "IPv4 address" in capsys.readouterr().err


def test_main_refuses_to_start_without_a_built_frontend(monkeypatch, capsys, tmp_path: Path):
    def fail_start(*_, **__):
        raise AssertionError("a child process was started")

    workdir = tmp_path / "work"
    workdir.mkdir()
    installed: list[tuple] = []
    monkeypatch.setattr(signal, "signal", lambda *args: installed.append(args))
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(workdir))
    monkeypatch.setattr(ss, "FRONTEND_INDEX", tmp_path / "dist" / "index.html")
    monkeypatch.setattr(ss.Stack, "start", fail_start)
    argv = [
        "--orin",
        "user@192.0.2.10",
        "--deploy-script",
        "x",
        "--bridge",
        "b",
        "--livekit-server",
        "l",
        *CAMERA_ARGS,
    ]
    assert ss.main(argv) == 1
    assert "frontend is not built" in capsys.readouterr().err


def test_default_profile_gives_the_head_camera_settings():
    config = ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG)
    assert ss.sender_settings(config) == ss.SenderSettings(
        port=5000, clock_port=5611, fps=30, bitrate_mbps=4, stream_id="head", full_range=True, sipl=True
    )


def test_argus_backend_is_not_full_range():
    config = {**ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG), "capture_backend": "argus"}
    settings = ss.sender_settings(config)
    assert (settings.full_range, settings.sipl) == (False, False)


def test_overrides_replace_only_their_values():
    config = ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG, 60, 9, 6000, 5100)
    assert ss.sender_settings(config) == ss.SenderSettings(
        port=5100, clock_port=6000, fps=60, bitrate_mbps=9, stream_id="head", full_range=True, sipl=True
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
        lambda c: c.update(capture_backend="auto"),
        lambda c: c.pop("capture_backend"),
        lambda c: c["streams"][0].pop("id"),
        lambda c: c["streams"][0].update(id=""),
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


def _profile_with(tmp_path: Path, mutate) -> Path:
    config = yaml.safe_load(ss.DEFAULT_SENDER_CONFIG.read_text())
    mutate(config)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("section", ["network", "output"])
def test_load_sender_config_treats_a_null_section_as_empty_and_applies_overrides(tmp_path: Path, section: str):
    def null_it(config):
        if section == "network":
            config["network"] = None
        else:
            config["streams"][0]["output"] = None

    config = ss.load_sender_config(_profile_with(tmp_path, null_it), 30, 12, 5700, 5701)
    output = config["streams"][0]["output"]
    assert output["fps"] == 30 and output["bitrate_mbps"] == 12
    assert config["network"]["clock_port"] == 5700 and config["network"]["port"] == 5701
    if section == "network":
        assert config["network"] == {"clock_port": 5700, "port": 5701}
    else:
        assert output == {"fps": 30, "bitrate_mbps": 12}


@pytest.mark.parametrize(
    ("mutate", "key"),
    [
        (lambda c: c.update(network=5), "network"),
        (lambda c: c["streams"][0].update(output=[1]), "output"),
        (lambda c: c.update(streams=[3]), "streams[0]"),
    ],
)
def test_load_sender_config_rejects_a_wrongly_typed_section(tmp_path: Path, mutate, key: str):
    with pytest.raises(ss.StartupError, match=key.replace("[", r"\[").replace("]", r"\]")):
        ss.load_sender_config(_profile_with(tmp_path, mutate))


def test_load_sender_config_rejects_malformed_yaml(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [")
    with pytest.raises(ss.StartupError):
        ss.load_sender_config(bad)


def test_start_of_a_missing_executable_is_a_startup_error_with_no_child(tmp_path: Path):
    stack = ss.Stack(tmp_path)
    with pytest.raises(ss.StartupError, match="cannot start ghost"):
        stack.start("ghost", [str(tmp_path / "no-such-tool")])
    assert stack.children == []


def test_check_station_cameras_with_the_example_station():
    ss.check_station_cameras(EXAMPLE_STATION, [f"{name}=dev0" for name in EXAMPLE_CAMERAS])
    with pytest.raises(ss.StartupError, match=EXAMPLE_CAMERAS[0]):
        ss.check_station_cameras(EXAMPLE_STATION, ["other=dev1"])


def test_check_station_cameras_accepts_a_station_without_video_sources(tmp_path: Path):
    station = tmp_path / "station.yaml"
    station.write_text("stations:\n  one:\n    meta: {}\n")
    ss.check_station_cameras(station, [])


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
        real_killpg(pgid, sig)
        if len(calls) == 1:
            raise ProcessLookupError

    monkeypatch.setattr(ss.os, "killpg", flaky_killpg)
    stack.stop_all()
    assert calls[0] == stack.children[1].process.pid
    # b's first signal reports its group as gone; it is not an error, and both children still end up stopped.
    assert [c.process.poll() for c in stack.children] == [0, 0]


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


def test_mock_sender_command_follows_the_profile():
    settings = ss.sender_settings(ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG))
    command = ss.mock_sender_command("python", Path("mock.py"), settings, 2560, 1984)
    assert command == ["python", "mock.py", "--port", "5000", "--stream", "head:2560x1984@30"]


def test_bridge_timing_arguments_for_mock_and_real_sender():
    sipl = ss.sender_settings(ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG))
    assert ss.bridge_timing_arguments(sipl, True, None) == ["--timestamp-basis", "monotonic_us"]
    real = " ".join(ss.bridge_timing_arguments(sipl, False, "192.0.2.10"))
    assert "--timestamp-basis capture_monotonic_us" in real
    assert "--sender-clock 192.0.2.10:5611" in real
    assert "--source-full-range" in real
    argus_config = {**ss.load_sender_config(ss.DEFAULT_SENDER_CONFIG), "capture_backend": "argus"}
    argus = ss.sender_settings(argus_config)
    assert "--source-full-range" not in ss.bridge_timing_arguments(argus, False, "192.0.2.10")


def test_main_refuses_orin_together_with_mock(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.delenv("EGOCAP_DEPLOY_SCRIPT", raising=False)
    installed: list[tuple] = []
    monkeypatch.setattr(signal, "signal", lambda *args: installed.append(args))
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    assert ss.main(["--orin", "user@192.0.2.10", "--mock"]) == 1
    assert "do not pass --orin" in capsys.readouterr().err


def test_main_refuses_a_missing_mock_sender(monkeypatch, capsys, tmp_path: Path):
    def fail_start(*_, **__):
        raise AssertionError("a child process was started")

    installed: list[tuple] = []
    monkeypatch.setattr(signal, "signal", lambda *args: installed.append(args))
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    monkeypatch.setattr(ss.Stack, "start", fail_start)
    monkeypatch.delenv("EGOCAP_MOCK_SENDER", raising=False)
    argv = ["--mock", "--bridge", "b", "--livekit-server", "l", "--mock-sender", str(tmp_path / "missing.py")]
    assert ss.main(argv) == 1
    assert "--mock-sender" in capsys.readouterr().err
    assert ss.main(["--mock", "--bridge", "b", "--livekit-server", "l"]) == 1
    assert "--mock-sender" in capsys.readouterr().err


def test_main_refuses_a_station_that_requests_cameras_not_given(monkeypatch, capsys, tmp_path: Path):
    installed: list[tuple] = []
    monkeypatch.setattr(signal, "signal", lambda *args: installed.append(args))
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    assert ss.main(["--capture-camera", "other=dev1"]) == 1
    assert EXAMPLE_CAMERAS[0] in capsys.readouterr().err


@pytest.fixture
def recorded_main(monkeypatch, tmp_path: Path):
    # Every child, wait, deploy and sleep is stubbed; the wait loop ends by raising KeyboardInterrupt.
    workdir = tmp_path / "work"
    workdir.mkdir()
    frontend = tmp_path / "index.html"
    frontend.write_text("")
    children: list[SimpleNamespace] = []
    deploys: list[list[str]] = []

    def fake_start(self, name, command, env=None, cwd=None, grace_s=ss.DEFAULT_GRACE_S, leader_only=False):
        log = self.workdir / f"{name}.log"
        log.write_text("")
        children.append(SimpleNamespace(name=name, command=list(command), env=env, leader_only=leader_only))
        managed = ss.Managed(name, SimpleNamespace(poll=lambda: None, pid=0), log, grace_s, leader_only)
        self.children.append(managed)
        return managed

    def interrupt(_seconds):
        raise KeyboardInterrupt

    installed: list[tuple] = []
    monkeypatch.setattr(signal, "signal", lambda *args: installed.append(args))
    monkeypatch.setattr(ss.tempfile, "mkdtemp", lambda **_: str(workdir))
    monkeypatch.setattr(ss, "FRONTEND_INDEX", frontend)
    monkeypatch.setattr(ss.Stack, "start", fake_start)
    monkeypatch.setattr(ss.Stack, "wait_until", lambda *_, **__: None)
    # The fake children are not real processes, so there is nothing to signal.
    monkeypatch.setattr(ss.Stack, "stop_all", lambda self: None)
    monkeypatch.setattr(ss.subprocess, "run", lambda command, **_: deploys.append(list(command)))
    monkeypatch.setattr(ss.time, "sleep", interrupt)
    return SimpleNamespace(children=children, deploys=deploys, workdir=workdir, installed=installed)


def _command(recorded, name: str) -> str:
    return " ".join(next(c.command for c in recorded.children if c.name == name))


def test_main_starts_the_orin_chain_in_order(recorded_main):
    argv = ["--orin", "user@192.0.2.10", "--deploy-script", "x", "--bridge", "b", "--livekit-server", "l"]
    assert ss.main([*argv, *CAMERA_ARGS]) == 0
    names = [c.name for c in recorded_main.children]
    assert names == ["sender", "livekit", "capture-server", "backend", "receiver", "bridge"]
    capture = _command(recorded_main, "capture-server")
    assert "--camera" in capture
    assert all(f"{name}=dev0" in capture for name in EXAMPLE_CAMERAS)
    assert "--livekit-streams" in capture
    assert "--mock" not in capture
    bridge = _command(recorded_main, "bridge")
    assert "--timestamp-basis capture_monotonic_us" in bridge
    assert "--sender-clock 192.0.2.10:5611" in bridge
    assert "--source-full-range" in bridge
    assert "--config" in _command(recorded_main, "receiver")
    assert len(recorded_main.deploys) == 1
    assert "--with-sipl" in recorded_main.deploys[0]


def test_main_starts_the_mock_chain_in_order(recorded_main, tmp_path: Path):
    mock_sender = tmp_path / "mock_sender.py"
    mock_sender.write_text("")
    argv = ["--mock", "--bridge", "b", "--livekit-server", "l", "--mock-sender", str(mock_sender)]
    assert ss.main(argv) == 0
    names = [c.name for c in recorded_main.children]
    assert names == ["mock-sender", "livekit", "capture-server", "backend", "receiver", "bridge"]
    assert "--mock" in recorded_main.children[2].command
    bridge = _command(recorded_main, "bridge")
    assert "--timestamp-basis monotonic_us" in bridge
    assert "--sender-clock" not in bridge
    assert recorded_main.deploys == []


ORIN_ARGV = ["--orin", "user@192.0.2.10", "--deploy-script", "x", "--bridge", "b", "--livekit-server", "l"]


def test_main_returns_zero_when_the_sender_finishes_its_run(recorded_main, monkeypatch, capsys):
    started = ss.Stack.start

    def start(self, name, *args, **kwargs):
        managed = started(self, name, *args, **kwargs)
        if name == "sender":
            managed.process = SimpleNamespace(poll=lambda: 0, pid=0, returncode=0)
        return managed

    monkeypatch.setattr(ss.Stack, "start", start)
    assert ss.main([*ORIN_ARGV, *CAMERA_ARGS]) == 0
    captured = capsys.readouterr()
    assert "finished" in captured.out
    assert "error" not in captured.err


@pytest.mark.parametrize(("name", "code"), [("sender", 3), ("receiver", 0)])
def test_main_fails_when_the_sender_errors_or_another_child_exits(recorded_main, monkeypatch, capsys, name, code):
    started = ss.Stack.start

    def start(self, n, *args, **kwargs):
        managed = started(self, n, *args, **kwargs)
        if n == name:
            managed.process = SimpleNamespace(poll=lambda: code, pid=0, returncode=code)
        return managed

    monkeypatch.setattr(ss.Stack, "start", start)
    assert ss.main([*ORIN_ARGV, *CAMERA_ARGS]) == 1
    assert f"{name} exited with {code}" in capsys.readouterr().err


def test_main_installs_the_stop_handlers_and_starts_the_backend_leader_only(recorded_main, tmp_path: Path):
    mock_sender = tmp_path / "mock_sender.py"
    mock_sender.write_text("")
    argv = ["--mock", "--bridge", "b", "--livekit-server", "l", "--mock-sender", str(mock_sender)]
    assert ss.main(argv) == 0
    handlers = dict(recorded_main.installed)
    assert handlers[signal.SIGTERM] is ss._raise_interrupt
    assert handlers[signal.SIGHUP] is ss._raise_interrupt
    leader_only = {c.name: c.leader_only for c in recorded_main.children}
    assert leader_only["backend"] is True


GRANDCHILD = """
import signal, sys, time
pid_file, int_file = sys.argv[1:3]
def on_int(*_):
    open(int_file, "w").write("grandchild-int")
    sys.exit(0)
signal.signal(signal.SIGINT, on_int)
open(pid_file, "w").write(str(__import__("os").getpid()))
time.sleep(60)
"""

PARENT_OF_GRANDCHILD = """
import signal, subprocess, sys, time
from pathlib import Path
grandchild, pid_file, int_file, ready = sys.argv[1:5]
subprocess.Popen([sys.executable, "-c", grandchild, pid_file, int_file])
while not Path(pid_file).exists():
    time.sleep(0.02)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
open(ready, "w").close()
time.sleep(60)
"""


def _with_grandchild(stack: ss.Stack, tmp_path: Path, **kwargs) -> tuple[Path, Path, Path]:
    pid_file, int_file, ready = tmp_path / "gc.pid", tmp_path / "gc.int", tmp_path / "parent.ready"
    command = [sys.executable, "-c", PARENT_OF_GRANDCHILD, GRANDCHILD, str(pid_file), str(int_file), str(ready)]
    stack.start("parent", command, **kwargs)
    _await_files([ready])
    return pid_file, int_file, ready


def _assert_pid_gone(pid: int) -> None:
    # A killed orphan stays visible until init reaps it, so allow a moment.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} still exists")


def test_leader_only_child_keeps_its_helper_from_the_polite_signal_and_the_helper_is_killed(stack, tmp_path: Path):
    pid_file, int_file, _ = _with_grandchild(stack, tmp_path, leader_only=True)
    stack.stop_all()
    assert stack.children[0].process.poll() is not None
    assert not int_file.exists()
    _assert_pid_gone(int(pid_file.read_text()))


def test_helper_left_in_the_group_of_an_exited_child_is_killed(stack, tmp_path: Path):
    pid_file, int_file, _ = _with_grandchild(stack, tmp_path)
    stack.stop_all()
    assert stack.children[0].process.poll() is not None
    _assert_pid_gone(int(pid_file.read_text()))


def test_sighup_during_stop_kills_everything_at_once_and_restores_the_handler(stack, tmp_path: Path):
    ready = [_stubborn(stack, tmp_path, "a", grace_s=30), _stubborn(stack, tmp_path, "b", grace_s=30)]
    _await_files(ready)
    before = signal.getsignal(signal.SIGHUP)
    timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGHUP))
    timer.start()
    started = time.monotonic()
    try:
        stack.stop_all()
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5
    assert [c.process.returncode for c in stack.children] == [-signal.SIGKILL] * 2
    assert signal.getsignal(signal.SIGHUP) is before
