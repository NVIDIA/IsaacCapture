# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server without CloudXR: its start, its CloudXR check and --no-headset.

In process, stand-ins replace CloudXRLauncher.launch_context(), the probe of the CloudXR runtime
and the reading of its environment file; whatever a test says is never called fails the test
when called. As a program, the server runs with --no-headset, with an install folder whose EULA
is not accepted, and with no OpenXR runtime; a RemoteSession is the robot app and a RobotState
reader reads what it publishes. No CloudXR is started and no headset is used.
"""

import contextlib
import json
import logging
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import isaaccapture
from isaaccapture import oxr
from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.cloudxr.service import __main__ as service_cli
from isaaccapture.deviceio_session import RemoteSession
from isaaccapture.deviceio_trackers import HeadTracker
from isaaccapture.robot_state import RobotState, parse_robot_description, records
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server import _cloudxr

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

WAIT_MS = 10_000
SERVER_LOGGER = "isaaccapture.teleop_session_server"
BANNER = re.compile(r"^Teleop Session Server listening on (tcp://\S+)$")
NO_ENTRY = {"state": "", "headset_state": "", "reason": ""}
NOT_AVAILABLE = "CloudXR is not available: "
EULA_NOT_ACCEPTED = "CloudXR is not started: the NVIDIA CloudXR EULA is not accepted"
STOPPED = "The CloudXR runtime serving {run_dir} has stopped"
NO_ENVIRONMENT_FILE = (
    "A CloudXR runtime is serving {run_dir}, but its environment file is missing or "
    "unreadable, so OpenXR cannot be pointed at it"
)
ATTACHED = "Attached to the CloudXR runtime serving {run_dir}"
NO_HEADSET = "this Teleop Session Server runs without a headset (--no-headset)"
NO_HEADSET_LOG = "--no-headset: no headset is served; CloudXR and OpenXR are not used"
OPEN_FAILED = "The Teleop Session Server could not open an OpenXR session: "
KEEPS_SERVING = (
    "Robot apps are served without headset data until this is fixed; the Teleop Session "
    "Server keeps checking and needs no restart"
)
START_COMMAND = "python -m isaaccapture.cloudxr.service start"
START_HINT = f". Start CloudXR with: {START_COMMAND}"
EULA_QUESTION = "Accept NVIDIA CloudXR EULA? [y/N]: "
# The words in the file names of the OpenXR loader and the CloudXR runtime's libraries.
HEADSET_LIBRARIES = ("openxr", "cloudxr", "nvstream")
INSTALL_DIR = "/cloudxr-install"
RUN_DIR = f"{INSTALL_DIR}/run"
# What CloudXR writes to its environment file.
ENVIRONMENT = {
    "XR_RUNTIME_JSON": f"{INSTALL_DIR}/openxr_cloudxr.json",
    "NV_CXR_RUNTIME_DIR": RUN_DIR,
}
DESCRIPTION = parse_robot_description(
    """
robot = "test_arm"

[series.arm]
shape = "joints"
joints = [
  { name = "joint1",  kind = "revolute", min = -2.0, max = 2.0 },
  { name = "gripper", kind = "gripper",  min = 0.0,  max = 1.0 },
]
"""
)


def _wait_for(predicate, timeout_s=10.0):
    """The first truthy value of predicate(), or its last value after timeout_s."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = predicate()
        if value or time.monotonic() > deadline:
            return value
        time.sleep(0.005)


def _args(*argv):
    return server_main._build_parser().parse_args(
        ["--cloudxr-install-dir", INSTALL_DIR, *argv]
    )


# ============================================================================
# Stand-ins
# ============================================================================


class _Never:
    """Stands in for what a test says is never called: a call is recorded and raises."""

    def __init__(self):
        self.calls = []

    def stand_in(self, name):
        def called(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"{name} was called")

        return called


@pytest.fixture
def never():
    calls = _Never()
    yield calls
    # Also seen when the caller caught what the stand-in raised.
    assert calls.calls == []


class _Launcher:
    """What CloudXRLauncher.launch_context() gives: a context manager with a health check."""

    def __init__(self):
        self.calls = []
        # Raised by health_check() while set.
        self.error = None

    def __enter__(self):
        self.calls.append("enter")
        return self

    def __exit__(self, *exc):
        self.calls.append("exit")

    def health_check(self):
        self.calls.append("check")
        if self.error is not None:
            raise self.error


class _Runtime:
    """The CloudXR runtime as _cloudxr sees it: whether it is live, and its environment file."""

    def __init__(self, monkeypatch):
        self.live = False
        self.environment = dict(ENVIRONMENT)
        self.probed = []
        self.read = []
        monkeypatch.setattr(_cloudxr, "is_runtime_live", self._is_live)
        monkeypatch.setattr(_cloudxr, "read_exported_env", self._read)

    def _is_live(self, run_dir):
        self.probed.append(run_dir)
        return self.live

    def _read(self, path):
        self.read.append(path)
        return dict(self.environment)


@pytest.fixture
def runtime(monkeypatch):
    return _Runtime(monkeypatch)


@pytest.fixture
def environment(monkeypatch):
    """Every key the tests' CloudXR writes is restored; gives the (key, value) pairs set since."""
    for key in ENVIRONMENT:
        monkeypatch.setenv(key, "before")
    sets = []
    set_item = type(os.environ).__setitem__

    def recording(self, key, value):
        sets.append((key, value))
        set_item(self, key, value)

    monkeypatch.setattr(type(os.environ), "__setitem__", recording)
    return sets


def _launch_context(monkeypatch, result):
    """Replace CloudXRLauncher.launch_context(): it returns ``result``, or raises it."""
    calls = []

    def launch_context(args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(CloudXRLauncher, "launch_context", staticmethod(launch_context))
    return calls


class _Stdin:
    def __init__(self, terminal):
        self._terminal = terminal

    def isatty(self):
        if isinstance(self._terminal, Exception):
            raise self._terminal
        return self._terminal


# Standard input of a server no person can answer: a service, nohup, an agent.
NO_TERMINAL = {
    "not a terminal": _Stdin(False),
    "closed": _Stdin(ValueError("I/O operation on closed file")),
    "none": None,
}


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger=SERVER_LOGGER)
    return caplog


def _unavailable(cloudxr):
    """The reason the health check raises."""
    with pytest.raises(RuntimeError) as raised:
        cloudxr.health_check()
    assert type(raised.value) is RuntimeError
    return str(raised.value)


# ============================================================================
# The start
# ============================================================================


def test_start_keeps_a_launcher_that_works(monkeypatch, never):
    launcher = _Launcher()
    args = _args()
    calls = _launch_context(monkeypatch, launcher)
    # A CloudXR that was started or attached to is not probed or read a second time.
    monkeypatch.setattr(_cloudxr, "is_runtime_live", never.stand_in("is_runtime_live"))
    monkeypatch.setattr(
        _cloudxr, "read_exported_env", never.stand_in("read_exported_env")
    )

    cloudxr = _cloudxr.start(args)

    assert calls == [(args, {})]
    with cloudxr as entered:
        assert entered is cloudxr
        cloudxr.health_check()
        cloudxr.health_check()
        # The launcher's own failure is the reason.
        launcher.error = RuntimeError(STOPPED.format(run_dir=RUN_DIR))
        with pytest.raises(RuntimeError) as raised:
            cloudxr.health_check()
        assert raised.value is launcher.error
    assert launcher.calls == ["enter", "check", "check", "check", "exit"]
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError])
def test_start_without_cloudxr_gives_one_that_is_not_available(
    monkeypatch, runtime, capsys, error_type
):
    calls = _launch_context(monkeypatch, error_type("no CloudXR SDK"))

    cloudxr = _cloudxr.start(_args())

    assert len(calls) == 1
    with cloudxr as entered:
        assert entered is cloudxr
        for _ in range(3):
            assert _unavailable(cloudxr) == NOT_AVAILABLE + "no CloudXR SDK"
    assert runtime.probed == [RUN_DIR] * 3
    assert runtime.read == []
    assert len(calls) == 1
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("stdin", list(NO_TERMINAL))
def test_start_with_the_eula_refused_and_no_terminal_goes_on(
    monkeypatch, runtime, capsys, stdin
):
    monkeypatch.setattr(sys, "stdin", NO_TERMINAL[stdin])
    # What the launcher's EULA check does when the answer is not "yes".
    calls = _launch_context(monkeypatch, SystemExit(1))

    cloudxr = _cloudxr.start(_args())

    assert len(calls) == 1
    with cloudxr:
        assert _unavailable(cloudxr) == EULA_NOT_ACCEPTED
    assert runtime.probed == [RUN_DIR]
    # The launcher's question has no line end; what the server prints next starts a line.
    assert capsys.readouterr().out == "\n"


def test_start_keeps_the_exit_of_a_person_at_a_terminal(monkeypatch, runtime):
    monkeypatch.setattr(sys, "stdin", _Stdin(True))
    _launch_context(monkeypatch, SystemExit(1))

    with pytest.raises(SystemExit) as raised:
        _cloudxr.start(_args())

    assert raised.value.code == 1
    assert runtime.probed == []


@pytest.mark.parametrize("terminal", [True, False])
def test_start_keeps_the_exit_of_a_key_press_at_the_launchers_notice(
    monkeypatch, runtime, terminal
):
    monkeypatch.setattr(sys, "stdin", _Stdin(terminal))
    aborted = "Aborted: start the CloudXR service with this configuration"
    _launch_context(monkeypatch, SystemExit(aborted))

    with pytest.raises(SystemExit) as raised:
        _cloudxr.start(_args())

    assert raised.value.code == aborted
    assert runtime.probed == []


def test_no_launch_cloudxr_runtime_is_never_probed(monkeypatch, never):
    """The real launch_context(): it builds no launcher for this option."""
    monkeypatch.setattr(CloudXRLauncher, "__init__", never.stand_in("CloudXRLauncher"))
    monkeypatch.setattr(_cloudxr, "is_runtime_live", never.stand_in("is_runtime_live"))
    monkeypatch.setattr(
        _cloudxr, "read_exported_env", never.stand_in("read_exported_env")
    )

    cloudxr = _cloudxr.start(_args("--no-launch-cloudxr-runtime"))

    with cloudxr as entered:
        assert entered is cloudxr
        for _ in range(3):
            cloudxr.health_check()

    # Without a run folder nothing is checked, whatever was handed over as the launcher.
    launcher = _Launcher()
    launcher.error = RuntimeError("asked")
    _cloudxr.CloudXR(launcher, None).health_check()
    assert launcher.calls == []


def _started_folder(command):
    """The run folder of the CloudXR that ``command`` starts, by the service's own parser."""
    words = shlex.split(command)
    assert words[:3] == ["python", "-m", "isaaccapture.cloudxr.service"]
    started = service_cli._build_parser().parse_args(words[3:])
    assert started.command == "start"
    return os.path.join(os.path.abspath(started.cloudxr_install_dir), "run")


@pytest.mark.parametrize("launched", ["a launcher", "no CloudXR"])
def test_the_start_command_starts_cloudxr_in_the_folder_the_server_looks_at(
    monkeypatch, runtime, tmp_path, launched
):
    _launch_context(
        monkeypatch,
        _Launcher() if launched == "a launcher" else RuntimeError("no CloudXR SDK"),
    )
    parser = server_main._build_parser()
    default_run = os.path.join(os.path.expanduser("~/.cloudxr"), "run")

    # The service's default folder is the server's default folder.
    for argv in ([], ["--cloudxr-install-dir", "~/.cloudxr/"]):
        command = _cloudxr.start(parser.parse_args(argv)).start_command
        assert command == START_COMMAND
        assert _started_folder(command) == default_run

    # A folder with a space is one word of the command.
    for folder in (INSTALL_DIR, str(tmp_path / "cloud xr")):
        cloudxr = _cloudxr.start(parser.parse_args(["--cloudxr-install-dir", folder]))
        command = cloudxr.start_command
        assert command.startswith(START_COMMAND + " --cloudxr-install-dir ")
        assert _started_folder(command) == os.path.join(folder, "run")
        if launched == "no CloudXR":
            # The folder this CloudXR is looked for in.
            assert _unavailable(cloudxr) == NOT_AVAILABLE + "no CloudXR SDK"
            assert runtime.probed[-1] == os.path.join(folder, "run")
    assert (
        _cloudxr.start(_args()).start_command
        == f"{START_COMMAND} --cloudxr-install-dir {INSTALL_DIR}"
    )

    # A folder given relative to the server's working folder is named in full.
    monkeypatch.chdir(tmp_path)
    relative = _cloudxr.start(parser.parse_args(["--cloudxr-install-dir", "cxr"]))
    assert _started_folder(relative.start_command) == os.path.join(
        os.getcwd(), "cxr", "run"
    )

    # With --no-launch-cloudxr-runtime no folder is looked at.
    assert _cloudxr.CloudXR(None, None).start_command == START_COMMAND


# ============================================================================
# The health check of a CloudXR that was not there at the start
# ============================================================================


def test_health_check_attaches_once_cloudxr_runs_and_sets_only_what_differs(
    monkeypatch, runtime, environment, logs
):
    # One key already has the value of CloudXR's environment file.
    monkeypatch.setenv("NV_CXR_RUNTIME_DIR", RUN_DIR)
    monkeypatch.setenv("ISAACCAPTURE_TEST_UNRELATED", "kept")
    environment.clear()
    cloudxr = _cloudxr.CloudXR(None, RUN_DIR, NOT_AVAILABLE + "nothing to start")

    for _ in range(2):
        assert _unavailable(cloudxr) == NOT_AVAILABLE + "nothing to start"
    assert runtime.read == []
    assert environment == []

    # A person starts CloudXR.
    runtime.live = True
    cloudxr.health_check()
    assert runtime.read == [Path(RUN_DIR) / "cloudxr.env"]
    assert environment == [("XR_RUNTIME_JSON", ENVIRONMENT["XR_RUNTIME_JSON"])]
    assert {key: os.environ[key] for key in ENVIRONMENT} == ENVIRONMENT
    assert os.environ["ISAACCAPTURE_TEST_UNRELATED"] == "kept"
    attached = [
        (record.name, record.levelno, record.getMessage()) for record in logs.records
    ]
    assert attached == [
        (SERVER_LOGGER + ".cloudxr", logging.INFO, ATTACHED.format(run_dir=RUN_DIR))
    ]

    # While it runs, the file is not read again and nothing is set.
    for _ in range(3):
        cloudxr.health_check()
    assert len(runtime.read) == 1
    assert len(environment) == 1
    assert len(logs.records) == 1


def test_health_check_reads_the_environment_again_each_time_cloudxr_returns(
    runtime, environment, logs
):
    cloudxr = _cloudxr.CloudXR(None, RUN_DIR, NOT_AVAILABLE + "nothing to start")
    assert _unavailable(cloudxr) == NOT_AVAILABLE + "nothing to start"
    runtime.live = True
    cloudxr.health_check()
    assert len(runtime.read) == 1

    # It stops: the reason is that of a CloudXR that was attached to.
    runtime.live = False
    for _ in range(2):
        assert _unavailable(cloudxr) == STOPPED.format(run_dir=RUN_DIR)
    assert len(runtime.read) == 1

    # It is started again with another runtime file.
    runtime.environment["XR_RUNTIME_JSON"] = f"{INSTALL_DIR}/other.json"
    runtime.live = True
    environment.clear()
    cloudxr.health_check()
    cloudxr.health_check()
    assert len(runtime.read) == 2
    assert environment == [("XR_RUNTIME_JSON", f"{INSTALL_DIR}/other.json")]
    assert os.environ["XR_RUNTIME_JSON"] == f"{INSTALL_DIR}/other.json"
    assert logs.messages == [ATTACHED.format(run_dir=RUN_DIR)] * 2


@pytest.mark.parametrize(
    "error", [RuntimeError("stopped"), OSError("probe failed")], ids=repr
)
def test_health_check_attaches_again_after_any_failed_check_of_the_launcher(
    runtime, environment, error
):
    launcher = _Launcher()
    cloudxr = _cloudxr.CloudXR(launcher, RUN_DIR)
    cloudxr.health_check()
    # The launcher set the environment at the start.
    assert runtime.read == []

    launcher.error = error
    with pytest.raises(type(error)) as raised:
        cloudxr.health_check()
    assert raised.value is error
    assert runtime.read == []

    # The CloudXR that serves from here on may be another one.
    launcher.error = None
    cloudxr.health_check()
    cloudxr.health_check()
    assert runtime.read == [Path(RUN_DIR) / "cloudxr.env"]
    assert {key: os.environ[key] for key in ENVIRONMENT} == ENVIRONMENT
    # The launcher's probe is the one used.
    assert runtime.probed == []


def test_health_check_names_a_missing_environment_file(runtime, environment, logs):
    cloudxr = _cloudxr.CloudXR(None, RUN_DIR, NOT_AVAILABLE + "nothing to start")
    runtime.live = True
    runtime.environment = {}
    environment.clear()

    for _ in range(2):
        assert _unavailable(cloudxr) == NO_ENVIRONMENT_FILE.format(run_dir=RUN_DIR)
    # Looked for at every check, and nothing was set or announced.
    assert len(runtime.read) == 2
    assert environment == []
    assert logs.messages == []

    runtime.environment = dict(ENVIRONMENT)
    cloudxr.health_check()
    assert {key: os.environ[key] for key in ENVIRONMENT} == ENVIRONMENT
    assert logs.messages == [ATTACHED.format(run_dir=RUN_DIR)]


def test_health_check_never_builds_a_launcher(monkeypatch, runtime, environment, never):
    monkeypatch.setattr(CloudXRLauncher, "__init__", never.stand_in("CloudXRLauncher"))
    monkeypatch.setattr(
        CloudXRLauncher,
        "launch_context",
        staticmethod(never.stand_in("CloudXRLauncher.launch_context")),
    )
    cloudxr = _cloudxr.CloudXR(None, RUN_DIR, EULA_NOT_ACCEPTED)

    with cloudxr:
        # Not there, started, stopped, started again, without its environment file, with it.
        for live, file in (
            (False, ENVIRONMENT),
            (True, ENVIRONMENT),
            (False, ENVIRONMENT),
            (True, ENVIRONMENT),
            (False, ENVIRONMENT),
            (True, {}),
            (True, ENVIRONMENT),
        ):
            runtime.live = live
            runtime.environment = dict(file)
            with contextlib.suppress(RuntimeError):
                cloudxr.health_check()
    assert len(runtime.read) == 4


# ============================================================================
# main()
# ============================================================================


@pytest.fixture
def recorded_server(monkeypatch):
    """Every _Server that main() built, as a stand-in that records what it was built with."""
    built = []

    class RecordedServer:
        def __init__(self, listening, launcher, stop, **options):
            self.listening = listening
            self.launcher = launcher
            self.options = options
            built.append(self)

        def run(self):
            for item in self.listening:
                item.listener.close()
            return 0

    monkeypatch.setattr(server_main, "_Server", RecordedServer)
    # main() sends the OpenXR runtime's raw output to a file; there is none here.
    monkeypatch.setattr(server_main, "capture_native_output", contextlib.nullcontext)
    return built


def test_main_hands_the_server_no_cloudxr_with_no_headset(
    monkeypatch, recorded_server, never, logs, capsys, tmp_path
):
    monkeypatch.setattr(CloudXRLauncher, "__init__", never.stand_in("CloudXRLauncher"))
    monkeypatch.setattr(
        CloudXRLauncher,
        "launch_context",
        staticmethod(never.stand_in("CloudXRLauncher.launch_context")),
    )
    monkeypatch.setattr(_cloudxr, "is_runtime_live", never.stand_in("is_runtime_live"))
    monkeypatch.setattr(
        _cloudxr, "read_exported_env", never.stand_in("read_exported_env")
    )
    monkeypatch.setattr(oxr, "OpenXRSession", never.stand_in("oxr.OpenXRSession"))
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    socket_path = str(tmp_path / "rs.sock")

    # The CloudXR options are accepted and not used.
    code = server_main.main(
        [
            "--no-headset",
            "--listen",
            "tcp://127.0.0.1:0",
            "--robot-state-socket",
            socket_path,
            "--cloudxr-install-dir",
            INSTALL_DIR,
            "--accept-eula",
        ]
    )

    assert code == 0
    [server] = recorded_server
    assert server.launcher is None
    assert server.options == {
        "placeholder": False,
        "robot_state_socket": socket_path,
        "headset": False,
    }
    [line] = capsys.readouterr().out.splitlines()
    assert BANNER.match(line)
    said = [
        (record.name, record.levelno)
        for record in logs.records
        if record.getMessage() == NO_HEADSET_LOG
    ]
    assert said == [(SERVER_LOGGER, logging.INFO)]
    assert not [r for r in logs.records if r.levelno >= logging.WARNING]
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


def test_main_hands_the_server_the_cloudxr_it_started(
    monkeypatch, recorded_server, logs, capsys
):
    launcher = _Launcher()
    calls = _launch_context(monkeypatch, launcher)

    assert server_main.main(["--listen", "tcp://127.0.0.1:0"]) == 0

    [server] = recorded_server
    assert isinstance(server.launcher, _cloudxr.CloudXR)
    assert server.options == {
        "placeholder": False,
        "robot_state_socket": None,
        "headset": True,
    }
    # Started once, and entered for as long as the server ran.
    assert len(calls) == 1
    assert launcher.calls == ["enter", "exit"]
    assert NO_HEADSET_LOG not in logs.messages
    [line] = capsys.readouterr().out.splitlines()
    assert BANNER.match(line)


def test_no_headset_with_placeholder_openxr_is_a_command_line_error(
    monkeypatch, recorded_server, never, capsys
):
    monkeypatch.setattr(
        CloudXRLauncher,
        "launch_context",
        staticmethod(never.stand_in("CloudXRLauncher.launch_context")),
    )

    with pytest.raises(SystemExit) as raised:
        server_main.main(["--no-headset", "--placeholder-openxr"])

    assert raised.value.code == 2
    output = capsys.readouterr()
    assert output.err.rstrip().endswith(
        "error: --no-headset cannot be combined with --placeholder-openxr"
    )
    # Before anything is listened on or started.
    assert output.out == ""
    assert recorded_server == []


# ============================================================================
# The server as a program
# ============================================================================


class _Program:
    """A Teleop Session Server subprocess on a loopback port the system picks."""

    def __init__(self, folder, *args, env=None):
        self.socket_path = str(folder / "rs.sock")
        self.log_path = folder / "server.log"
        environment = dict(os.environ)
        # Leads its own logging, so its console output lands in log_path.
        environment.pop("ISAACCAPTURE_LOG_SOCKET", None)
        environment["ISAACCAPTURE_LOG_DIR"] = str(folder / "logs")
        for key, value in (env or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        command = [
            sys.executable,
            "-m",
            "isaaccapture.teleop_session_server",
            "--listen",
            "tcp://127.0.0.1:0",
            "--robot-state-socket",
            self.socket_path,
            *args,
        ]
        with open(self.log_path, "wb") as log:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=log,
                env=environment,
                start_new_session=True,
                text=True,
            )
        # Standard output up to the banner.
        self.before_banner = ""
        self._lines = queue.Queue()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self.endpoint = self._read_banner()

    def _read_stdout(self):
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _read_banner(self):
        deadline = time.monotonic() + 90.0
        while True:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                line = None
            if line is None:
                self.kill()
                pytest.fail(
                    f"the server printed no listening banner:\n{self.before_banner}\n"
                    f"{self.log()}"
                )
            match = BANNER.match(line.rstrip("\n"))
            if match:
                return match.group(1)
            self.before_banner += line

    def log(self):
        return self.log_path.read_text(errors="replace")

    def logged(self, level):
        """The log lines of ``level``, e.g. "WARNING"."""
        return [line for line in self.log().splitlines() if f"[{level}" in line]

    def mapped_files(self):
        """The file names of everything mapped into the server's memory."""
        names = set()
        for line in Path(f"/proc/{self.proc.pid}/maps").read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) == 6 and fields[5].startswith("/"):
                names.add(os.path.basename(fields[5]))
        return names

    def stop(self):
        """Send SIGTERM and wait for the server to exit; its exit code."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        try:
            return self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.kill()
            pytest.fail(f"the server did not stop:\n{self.log()}")

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


@pytest.fixture
def program(tmp_path_factory):
    programs = []

    def start(*args, env=None):
        # A short folder name: a Unix socket path holds about a hundred bytes.
        server = _Program(tmp_path_factory.mktemp("nc"), *args, env=env)
        programs.append(server)
        return server

    yield start
    for server in programs:
        server.kill()


def _hello(session):
    return json.dumps(
        {
            "protocol": 1,
            "isaaccapture_version": isaaccapture.__version__,
            "app_name": "NoCloudXRTest",
            "tick_rate_hz": 90.0,
            "trackers": [
                {"name": "0", "type": "HeadTracker", "params": {}, "vendor": None}
            ],
            "topics": session.subscribed_topics(),
            "robot_state": {"version": 1},
        }
    )


def _publish_arm(session, joint1):
    now_ns = time.monotonic_ns()
    session.publish_robot_record(
        "arm",
        records.encode_reading(DESCRIPTION, "arm", [joint1, 1.0], time_ns=now_ns),
        now_ns,
    )


def _reads(reader, joint1):
    return _wait_for(
        lambda: (
            (arm := reader.joints("arm")) is not None
            and arm["joint1"].position == joint1
        )
    )


def _serves_robot_state(server, names_reason):
    """A robot app with a head tracker and robot state enters ``server``, and a reader reads
    what it publishes.

    The robot app gets no READY and no SAMPLES. Returns the STATUS message, which
    ``names_reason(message)`` accepts.
    """
    with RemoteSession.create([(HeadTracker(), "0")]) as session:
        session.set_robot_state(records.declarations(DESCRIPTION))
        session.connect(server.endpoint, _hello(session), WAIT_MS)
        assert json.loads(session.hello_ack())["robot_state"] == {"version": 1}
        session.start_robot_state()
        now_ns = time.monotonic_ns()
        session.publish_robot_record(
            records.DESCRIPTION_TOPIC,
            records.encode_description(DESCRIPTION, now_ns),
            now_ns,
        )
        _publish_arm(session, 0.5)
        with RobotState.connect(server.socket_path, robot="test_arm") as reader:
            assert _wait_for(lambda: reader.connected), server.log()
            assert reader.server_reachable
            assert reader.description() == DESCRIPTION
            assert _reads(reader, 0.5), server.log()

            assert _wait_for(session.latest_status), server.log()
            assert _wait_for(
                lambda: names_reason(json.loads(session.latest_status())["message"])
            ), (session.latest_status(), server.log())
            status = json.loads(session.latest_status())
            reason = status.pop("message")
            assert status == {"stage": "connecting_openxr", "headset": NO_ENTRY}
            # The trackers report no data, and the session goes on.
            assert not session.wait_ready(300)
            session.update()
            assert not session.is_fresh
            assert session.tick_age_ns is None

            _publish_arm(session, 0.25)
            assert _reads(reader, 0.25), server.log()
            assert reader.connected
    return reason


def _warning(reason, start_hint=False):
    return f"{reason}. {KEEPS_SERVING}{START_HINT if start_hint else ''}"


def test_the_server_as_a_program_serves_robot_state_with_no_headset(program):
    server = program("--no-headset")

    assert server.before_banner == ""
    _serves_robot_state(server, NO_HEADSET.__eq__)

    mapped = server.mapped_files()
    # The file names are read: the server's own library is among them.
    assert any(name.startswith("_deviceio_session") for name in mapped), mapped
    assert [
        name
        for name in mapped
        if any(word in name.lower() for word in HEADSET_LIBRARIES)
    ] == []

    assert server.stop() == 0
    log = server.log()
    assert log.count(NO_HEADSET_LOG) == 1, log
    assert server.logged("WARNING") == [], log
    assert server.logged("ERROR") == [], log
    assert "Received SIGTERM; stopping the Teleop Session Server" in log
    assert not os.path.exists(server.socket_path)


def test_the_server_as_a_program_serves_when_the_eula_is_not_accepted(
    program, tmp_path
):
    """The real launcher stops at its EULA check, before it starts anything."""
    install_dir = tmp_path / "cloudxr"
    install_dir.mkdir()
    server = program("--cloudxr-install-dir", str(install_dir))

    # The launcher's question, then the banner on a line of its own.
    assert server.before_banner.strip() == EULA_QUESTION.strip()
    _serves_robot_state(server, EULA_NOT_ACCEPTED.__eq__)

    assert server.stop() == 0
    log = server.log()
    assert "EULA not accepted. Exiting." in log
    [warning] = server.logged("WARNING")
    # The command names the folder this server looks at, not the service's default.
    assert warning.endswith(
        _warning(EULA_NOT_ACCEPTED, start_hint=True)
        + f" --cloudxr-install-dir {install_dir}"
    ), log
    assert log.index("EULA not accepted. Exiting.") < log.index(warning)
    assert server.logged("ERROR") == [], log
    # Nothing was started or written for CloudXR.
    assert list(install_dir.iterdir()) == []
    assert not os.path.exists(server.socket_path)


def test_the_server_as_a_program_serves_with_no_openxr_runtime(program):
    """--no-launch-cloudxr-runtime on a machine whose environment names no OpenXR runtime:
    the real oxr.OpenXRSession fails before it loads one."""
    server = program(
        "--no-launch-cloudxr-runtime",
        env={"XR_RUNTIME_JSON": None, "NV_CXR_RUNTIME_DIR": None},
    )

    assert server.before_banner == ""
    reason = _serves_robot_state(server, lambda text: text.startswith(OPEN_FAILED))

    assert server.stop() == 0
    # Said once, without the start hint: CloudXR is not what this server waits for.
    [warning] = server.logged("WARNING")
    assert warning.endswith(_warning(reason.rstrip("."))), server.log()
    assert not os.path.exists(server.socket_path)
