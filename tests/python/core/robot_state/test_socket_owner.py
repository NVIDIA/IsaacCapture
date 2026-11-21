# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One program serves robot state at a socket path: the lock file, a fork, a service that died.

Every socket is a Unix socket in a temporary directory of the test; the other programs are
processes the test starts and ends.
"""

import json
import logging
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import repo_paths
from test_reader import start_robot

from isaaccapture.robot_state import (
    RobotState,
    RobotStateLocalService,
    RobotStateStore,
    _local,
    parse_robot_description,
)

pytestmark = pytest.mark.skipif(
    not _local._HAS_UNIX_SOCKETS, reason="the reader socket is a Unix domain socket"
)

ROUNDS = 20
ROBOT = """
robot = "{name}"

[series.arm]
shape = "joints"
joints = [{{ name = "joint1", kind = "revolute", min = -1.0, max = 1.0 }}]
"""


def robot_named(name):
    return parse_robot_description(ROBOT.format(name=name))


def store_of(name):
    """A store as a robot app named `name` leaves it."""
    store = RobotStateStore()
    start_robot(store, robot_named(name))
    return store


def refusal(path):
    return f"another program serves robot state at {path}"


def refused(service):
    """Start `service` at a path another serves; the error, which says so by its type."""
    with pytest.raises(OSError) as raised:
        service.start()
    assert type(raised.value) is _local.ReaderSocketTaken
    assert service.serving is False
    return raised.value


def fails(service):
    """Start `service` where it cannot serve for another reason; the error."""
    with pytest.raises((OSError, RuntimeError)) as raised:
        service.start()
    assert not isinstance(raised.value, _local.ReaderSocketTaken)
    assert service.serving is False
    return raised.value


def robot_at(path):
    """The robot a new reader at `path` reads; None when nothing answers."""
    with RobotState.connect(path) as reader:
        description = reader.description()
        return None if description is None else description.robot


def wait_for(condition, timeout_s=20.0):
    """The first true value of `condition()`; fails when none comes in time."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = condition()
        if value:
            return value
        assert time.monotonic() < deadline, "the awaited state did not come"
        time.sleep(0.01)


def serving_threads():
    return [
        thread for thread in threading.enumerate() if thread.name == _local.THREAD_NAME
    ]


@pytest.fixture
def socket_path(tmp_path_factory):
    # Shorter than `tmp_path`, which holds the test's name: a Unix socket path is limited.
    return str(tmp_path_factory.mktemp("rs") / "rs.sock")


@pytest.fixture
def serve():
    """Make services that are closed when the test ends; `start=False` leaves starting to the test."""
    services = []

    def make(store, path, *, start=True):
        service = RobotStateLocalService(store, path)
        services.append(service)
        if start:
            service.start()
        return service

    yield make
    for service in services:
        service.close()


# ---------------------------------------------------------------------------------------------
# An owner in a program of its own
# ---------------------------------------------------------------------------------------------

_PROGRAM = (
    "import sys; sys.path[:0] = sys.argv[1:3]; "
    "import test_socket_owner; test_socket_owner.own_commands()"
)

# A program that owns `path` as a service does, without the package: killed, it leaves the
# socket file and the lock file behind.
_BARE_OWNER = """
import fcntl, os, socket, sys, time
path = sys.argv[1]
lock = os.open(path + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(path)
listener.listen(1)
print("up", flush=True)
time.sleep(600)
"""


def _checks_in_child(service, path):
    """What a forked child sees of its parent's service; the child's exit code."""
    if service.serving:
        return 2
    # Closing the child's copy takes nothing from the parent.
    service.close()
    if not (os.path.exists(path) and os.path.exists(path + _local.LOCK_SUFFIX)):
        return 3
    # The path is still the parent's.
    try:
        RobotStateLocalService(RobotStateStore(), path).start()
    except _local.ReaderSocketTaken as error:
        return 0 if str(error) == refusal(path) else 4
    return 5


def own_commands():
    """An owner's side, run in a program of its own: one command per line of its input."""
    service = None
    path = None
    for line in sys.stdin:
        command = json.loads(line)
        do = command["do"]
        answer = "done"
        if do == "start":
            path = command["path"]
            service = RobotStateLocalService(store_of(command["robot"]), path)
            # Two programs told the same moment of this machine's clock start together.
            while time.monotonic_ns() < command.get("at_ns", 0):
                pass
            try:
                service.start()
            except _local.ReaderSocketTaken as error:
                answer = f"refused {error}"
            except OSError as error:
                answer = f"failed {error}"
            else:
                answer = "serving"
        elif do == "close":
            service.close()
        elif do == "serving":
            answer = str(service is not None and service.serving)
        elif do == "fork_and_sleep":
            pid = os.fork()
            if pid == 0:
                time.sleep(command["seconds"])
                os._exit(0)
            answer = str(pid)
        elif do == "fork_and_check":
            pid = os.fork()
            if pid == 0:
                code = 1
                try:
                    code = _checks_in_child(service, path)
                finally:
                    os._exit(code)
            _, status = os.waitpid(pid, 0)
            answer = str(os.waitstatus_to_exitcode(status))
        print(json.dumps({"do": do, "answer": answer}), flush=True)


class OwnerProgram:
    """A process that runs `own_commands()`."""

    def __init__(self):
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _PROGRAM,
                str(Path(__file__).resolve().parent),
                str(Path(repo_paths.__file__).resolve().parent),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def send(self, do, **command):
        self._process.stdin.write(json.dumps({"do": do, **command}) + "\n")
        self._process.stdin.flush()

    def answer(self, do):
        while True:
            ready, _, _ = select.select([self._process.stdout], [], [], 60.0)
            assert ready, f"the owner program did not answer '{do}'"
            line = self._process.stdout.readline()
            assert line, f"the owner program ended (exit code {self._process.poll()})"
            if line.startswith("{"):
                reply = json.loads(line)
                if reply["do"] == do:
                    return reply["answer"]

    def do(self, do, **command):
        self.send(do, **command)
        return self.answer(do)

    def kill(self):
        self._process.kill()
        self._process.wait(30.0)
        self._process.stdin.close()
        self._process.stdout.close()


@pytest.fixture
def owner_program():
    programs = []

    def start():
        program = OwnerProgram()
        programs.append(program)
        return program

    yield start
    for program in programs:
        program.kill()


def leave_a_dead_owner_s_files(path):
    """Run a program that owns `path` and kill it: its socket file and lock file stay."""
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", _BARE_OWNER, path],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "up"
    finally:
        process.kill()
        process.wait(30.0)
        process.stdout.close()
    assert stat.S_ISSOCK(os.lstat(path).st_mode)
    assert os.path.exists(path + _local.LOCK_SUFFIX)


# ---------------------------------------------------------------------------------------------
# The first program keeps the path
# ---------------------------------------------------------------------------------------------


def test_a_second_service_in_the_same_program_is_refused(socket_path, serve):
    first = serve(store_of("first"), socket_path)
    with RobotState.connect(socket_path) as reader:
        assert reader.description().robot == "first"

        second = serve(store_of("second"), socket_path, start=False)
        for _ in range(2):
            assert str(refused(second)) == refusal(socket_path)
        second.close()

        # The first one and its reader are untouched.
        assert first.serving is True
        assert stat.S_ISSOCK(os.lstat(socket_path).st_mode)
        assert reader.refresh() is True
        assert reader.description().robot == "first"
    assert robot_at(socket_path) == "first"
    assert len(serving_threads()) == 1


def test_a_service_in_another_program_is_refused(socket_path, serve, owner_program):
    serve(store_of("first"), socket_path)
    with RobotState.connect(socket_path) as reader:
        assert reader.description().robot == "first"
        other = owner_program()
        assert other.do("start", path=socket_path, robot="second") == (
            f"refused {refusal(socket_path)}"
        )
        assert other.do("serving") == "False"
        assert reader.refresh() is True
        assert reader.description().robot == "first"
    assert robot_at(socket_path) == "first"


def test_the_first_program_is_refused_while_another_serves(
    socket_path, serve, owner_program
):
    other = owner_program()
    assert other.do("start", path=socket_path, robot="other") == "serving"
    mine = serve(store_of("mine"), socket_path, start=False)
    assert str(refused(mine)) == refusal(socket_path)
    assert robot_at(socket_path) == "other"

    # The other program closes its service: the same service object starts.
    other.do("close")
    assert robot_at(socket_path) is None
    mine.start()
    assert mine.serving is True
    assert robot_at(socket_path) == "mine"


def test_a_program_that_serves_without_the_lock_keeps_its_path(socket_path, serve):
    """A program whose package has no lock file answers the look at its socket."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(4)
    try:
        service = serve(RobotStateStore(), socket_path, start=False)
        assert str(refused(service)) == refusal(socket_path)
        assert stat.S_ISSOCK(os.lstat(socket_path).st_mode)
    finally:
        listener.close()
    # It ended and left its socket file: the refused start had freed the lock again.
    service.start()
    assert service.serving is True


def test_a_program_without_the_lock_that_takes_no_reader_keeps_its_path(
    socket_path, serve
):
    """Its socket does not answer the look in time: it is alive all the same."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(0)
    waiting = []
    try:
        # Fill its queue of readers it has not accepted.
        while True:
            reader = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            waiting.append(reader)
            reader.settimeout(0.2)
            try:
                reader.connect(socket_path)
            except OSError:
                break
            assert len(waiting) < 64
        service = serve(RobotStateStore(), socket_path, start=False)
        error = refused(service)
        assert str(error).startswith(refusal(socket_path) + " (")
        assert stat.S_ISSOCK(os.lstat(socket_path).st_mode)
    finally:
        for reader in waiting:
            reader.close()
        listener.close()
    service.start()
    assert service.serving is True


def test_the_refusal_is_an_oserror_of_its_own_type():
    assert issubclass(_local.ReaderSocketTaken, OSError)
    assert _local.ReaderSocketTaken is not OSError
    # Private to the module, like the service's other helpers.
    assert "ReaderSocketTaken" not in _local.__all__


# ---------------------------------------------------------------------------------------------
# Two programs that start together
# ---------------------------------------------------------------------------------------------


def test_of_two_programs_that_start_at_once_exactly_one_serves(
    socket_path, owner_program
):
    programs = {"left": owner_program(), "right": owner_program()}
    for program in programs.values():
        assert program.do("serving") == "False"  # both have finished importing
    winners = []
    for _ in range(ROUNDS):
        leave_a_dead_owner_s_files(socket_path)
        at_ns = time.monotonic_ns() + 30_000_000
        for name, program in programs.items():
            program.send("start", path=socket_path, robot=name, at_ns=at_ns)
        answers = {name: program.answer("start") for name, program in programs.items()}

        serving = [name for name, answer in answers.items() if answer == "serving"]
        assert len(serving) == 1, answers
        (winner,) = serving
        (loser,) = set(programs) - {winner}
        assert answers[loser] == f"refused {refusal(socket_path)}"
        assert programs[loser].do("serving") == "False"
        # The socket file is the winner's: a reader reads its robot, again and again.
        with RobotState.connect(socket_path) as reader:
            for _ in range(3):
                assert reader.refresh() is True
                assert reader.description().robot == winner
        assert robot_at(socket_path) == winner
        winners.append(winner)
        programs[winner].do("close")
        assert not os.path.exists(socket_path)
    assert len(winners) == ROUNDS


def test_of_two_programs_that_start_at_once_in_a_new_directory_one_is_refused(
    tmp_path_factory, owner_program
):
    """The socket's directory is made by whichever comes first; the other finds it private."""
    base = tmp_path_factory.mktemp("rs")
    programs = {"left": owner_program(), "right": owner_program()}
    for program in programs.values():
        assert program.do("serving") == "False"  # both have finished importing
    for index in range(ROUNDS):
        path = str(base / f"d{index}" / "rs.sock")
        at_ns = time.monotonic_ns() + 30_000_000
        for name, program in programs.items():
            program.send("start", path=path, robot=name, at_ns=at_ns)
        answers = {name: program.answer("start") for name, program in programs.items()}

        serving = [name for name, answer in answers.items() if answer == "serving"]
        assert len(serving) == 1, answers
        (winner,) = serving
        (loser,) = set(programs) - {winner}
        assert answers[loser] == f"refused {refusal(path)}"
        assert stat.S_IMODE(os.stat(Path(path).parent).st_mode) == 0o700
        assert robot_at(path) == winner
        programs[winner].do("close")


def test_the_socket_s_directory_is_private_from_its_creation(
    tmp_path_factory, serve, monkeypatch
):
    path = tmp_path_factory.mktemp("rs") / "new" / "rs.sock"
    made = []
    make = os.mkdir

    def watched(directory, mode=0o777, **options):
        make(directory, mode, **options)
        made.append((str(directory), stat.S_IMODE(os.stat(directory).st_mode)))

    monkeypatch.setattr(os, "mkdir", watched)
    previous = os.umask(0o022)
    try:
        service = serve(RobotStateStore(), path)
    finally:
        os.umask(previous)
    assert service.serving is True
    assert made == [(str(path.parent), 0o700)]


# ---------------------------------------------------------------------------------------------
# The lock file, an owner that is killed, an owner that forked
# ---------------------------------------------------------------------------------------------


def test_the_lock_file_stays_and_the_path_is_served_again(socket_path, serve):
    lock_path = socket_path + _local.LOCK_SUFFIX
    assert lock_path == socket_path + ".lock"
    directory = Path(socket_path).parent

    first = serve(store_of("first"), socket_path)
    assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600
    assert stat.S_ISREG(os.lstat(lock_path).st_mode)
    assert sorted(entry.name for entry in directory.iterdir()) == [
        "rs.sock",
        "rs.sock.lock",
    ]
    lock_file = os.stat(lock_path).st_ino

    first.close()
    assert [entry.name for entry in directory.iterdir()] == ["rs.sock.lock"]
    assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600

    second = serve(store_of("second"), socket_path)
    assert second.serving is True
    assert robot_at(socket_path) == "second"
    # The same file is locked again; none is made beside it.
    assert os.stat(lock_path).st_ino == lock_file
    second.close()
    second.close()
    assert [entry.name for entry in directory.iterdir()] == ["rs.sock.lock"]


def test_the_lock_file_is_not_inherited_by_a_program_this_one_runs(socket_path, serve):
    service = serve(store_of("first"), socket_path)
    assert os.get_inheritable(service._lock_fd) is False
    assert os.get_inheritable(service._listener.fileno()) is False


def test_a_killed_owner_s_path_is_served_by_the_next(socket_path, serve, owner_program):
    owner = owner_program()
    assert owner.do("start", path=socket_path, robot="killed") == "serving"
    assert robot_at(socket_path) == "killed"
    owner.kill()
    # Its socket file and its lock file are still there; nobody holds either.
    assert stat.S_ISSOCK(os.lstat(socket_path).st_mode)
    assert os.path.exists(socket_path + _local.LOCK_SUFFIX)
    assert robot_at(socket_path) is None

    serve(store_of("next"), socket_path)
    assert robot_at(socket_path) == "next"


not_root = pytest.mark.skipif(
    hasattr(os, "getuid") and os.getuid() == 0,
    reason="root connects to a socket file whatever its mode",
)


@not_root
def test_a_dead_owner_s_socket_file_closed_to_this_user_refuses_nobody(
    socket_path, serve
):
    """Whether a program serves a socket file this user may not connect to cannot be told."""
    leave_a_dead_owner_s_files(socket_path)
    os.chmod(socket_path, 0)
    left = os.lstat(socket_path).st_ino

    service = serve(store_of("next"), socket_path, start=False)
    for _ in range(2):
        error = fails(service)
        assert type(error) is PermissionError
        assert str(error) == (
            f"{socket_path} cannot be connected to ([Errno 13] Permission denied)"
        )
    # The file is left alone, and nothing of the failed starts stays.
    assert os.lstat(socket_path).st_ino == left
    assert serving_threads() == []

    # It can be connected to again and nobody answers: the same service replaces it.
    os.chmod(socket_path, 0o600)
    service.start()
    assert service.serving is True
    assert robot_at(socket_path) == "next"


needs_fork = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="this platform has no fork()"
)


@needs_fork
def test_a_child_forked_by_a_killed_owner_does_not_keep_the_path(
    socket_path, serve, owner_program
):
    owner = owner_program()
    assert owner.do("start", path=socket_path, robot="parent") == "serving"
    child = int(owner.do("fork_and_sleep", seconds=120))
    try:
        assert robot_at(socket_path) == "parent"
        owner.kill()
        # The child lives on; it holds neither the listening socket nor the lock.
        os.kill(child, 0)
        started = time.monotonic()
        service = serve(store_of("next"), socket_path)
        assert robot_at(socket_path) == "next"
        assert time.monotonic() - started < 1.0
        assert service.serving is True
        os.kill(child, 0)
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


@needs_fork
def test_a_forked_child_serves_nothing_and_takes_nothing_from_its_parent(
    socket_path, serve, owner_program
):
    owner = owner_program()
    assert owner.do("start", path=socket_path, robot="parent") == "serving"
    with RobotState.connect(socket_path) as reader:
        assert reader.description().robot == "parent"
        # 0: in the child the service does not serve, its close() leaves the socket file and
        # the lock file, and a new service there is refused.
        assert owner.do("fork_and_check") == "0"

        # The parent still owns the path, serves its kept reader and takes new ones.
        assert owner.do("serving") == "True"
        assert reader.refresh() is True
        assert reader.description().robot == "parent"
        assert robot_at(socket_path) == "parent"
        late = serve(RobotStateStore(), socket_path, start=False)
        assert str(refused(late)) == refusal(socket_path)

    owner.do("close")
    assert not os.path.exists(socket_path)
    late.start()
    assert late.serving is True


def test_the_fork_hook_is_registered_once_for_a_program(
    tmp_path_factory, serve, monkeypatch
):
    if not hasattr(os, "register_at_fork"):
        pytest.skip("this platform has no fork hooks")
    registered = []
    monkeypatch.setattr(_local, "_fork_hook_registered", False)
    monkeypatch.setattr(
        _local.os, "register_at_fork", lambda **hooks: registered.append(hooks)
    )
    base = tmp_path_factory.mktemp("rs")
    services = [
        serve(RobotStateStore(), str(base / f"{name}.sock")) for name in ("a", "b", "c")
    ]
    assert registered == [{"after_in_child": _local._after_fork_in_child}]
    for service in services:
        assert service in _local._started


# ---------------------------------------------------------------------------------------------
# A service that died, and `serving`
# ---------------------------------------------------------------------------------------------


class Exploding:
    """A store that cannot be read."""

    def export_snapshot(self, known_setup_generation):
        raise RuntimeError("the store exploded")


def test_a_service_whose_thread_ended_frees_the_path(socket_path, serve, caplog):
    caplog.set_level(logging.INFO, logger="isaaccapture.robot_state")
    service = serve(Exploding(), socket_path)
    (thread,) = serving_threads()
    assert service.serving is True

    # The first request ends the thread.
    assert robot_at(socket_path) is None
    wait_for(lambda: not service.serving)
    thread.join(10.0)
    assert not thread.is_alive()
    assert not os.path.exists(socket_path)
    assert os.path.exists(socket_path + _local.LOCK_SUFFIX)
    errors = [entry for entry in caplog.records if entry.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].getMessage() == (
        f"Serving robot state at {socket_path} stopped; reader programs get no answer"
    )

    # Another service starts at the path, and closing the dead one takes nothing from it.
    successor = serve(store_of("successor"), socket_path)
    assert robot_at(socket_path) == "successor"
    service.close()
    service.close()
    assert service.serving is False
    assert successor.serving is True
    assert robot_at(socket_path) == "successor"


def test_a_service_whose_thread_ended_can_be_started_again(socket_path, serve):
    service = serve(Exploding(), socket_path)
    assert robot_at(socket_path) is None
    wait_for(lambda: not service.serving)
    wait_for(lambda: not os.path.exists(socket_path))

    service._store = store_of("again")
    service.start()
    assert service.serving is True
    assert robot_at(socket_path) == "again"
    service.close()
    assert service.serving is False
    assert not os.path.exists(socket_path)
    assert serving_threads() == []


def test_serving_turns_false_only_once_the_path_is_free(
    socket_path, serve, monkeypatch
):
    seen = []
    free_lock = _local._free_lock

    def watched(fd):
        free_lock(fd)
        # The last step of giving the path up: what a program that polls `serving` would find.
        seen.append((service.serving, os.path.exists(socket_path)))

    monkeypatch.setattr(_local, "_free_lock", watched)

    # A service whose thread ended.
    service = serve(Exploding(), socket_path)
    assert robot_at(socket_path) is None
    wait_for(lambda: not service.serving)
    assert seen == [(True, False)]

    # A service that is closed.
    service = serve(store_of("closed"), socket_path)
    service.close()
    assert seen == [(True, False)] * 2
    assert service.serving is False


def test_serving_is_true_only_between_start_and_close(socket_path, serve):
    service = serve(store_of("first"), socket_path, start=False)
    assert service.serving is False
    service.start()
    assert service.serving is True
    service.start()  # a second start changes nothing
    assert service.serving is True
    assert len(serving_threads()) == 1
    service.close()
    assert service.serving is False
    service.close()
    assert service.serving is False
    with pytest.raises(RuntimeError, match="RobotStateLocalService is closed"):
        service.start()
    assert service.serving is False

    never_started = serve(RobotStateStore(), socket_path, start=False)
    never_started.close()
    assert never_started.serving is False


def test_a_start_that_fails_serves_nothing_and_frees_the_lock(
    socket_path, serve, monkeypatch
):
    Path(socket_path).write_text("notes")
    service = serve(RobotStateStore(), socket_path, start=False)
    assert str(fails(service)) == f"{socket_path} exists and is not a socket"
    assert serving_threads() == []
    Path(socket_path).unlink()

    class NoThread(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    with monkeypatch.context() as patched:
        patched.setattr(_local.threading, "Thread", NoThread)
        error = fails(service)
        assert type(error) is RuntimeError
        assert str(error) == "can't start new thread"
    # Nothing of the failed start is left but the lock file, and that is not held.
    assert [entry.name for entry in Path(socket_path).parent.iterdir()] == [
        "rs.sock.lock"
    ]
    other = serve(store_of("other"), socket_path)
    assert other.serving is True
    assert robot_at(socket_path) == "other"


def test_a_path_that_cannot_be_served_makes_no_lock_file(tmp_path_factory):
    base = tmp_path_factory.mktemp("rs")
    too_long = RobotStateLocalService(
        RobotStateStore(), str(base / ("d" * 100) / "rs.sock")
    )
    assert "over the 103 a Unix domain socket allows" in str(fails(too_long))

    directory = base / "open"
    directory.mkdir()
    directory.chmod(0o755)
    open_to_others = RobotStateLocalService(RobotStateStore(), directory / "rs.sock")
    assert "is open to group or others" in str(fails(open_to_others))
    assert [entry.name for entry in base.iterdir()] == ["open"]
    assert list(directory.iterdir()) == []


def test_where_there_are_no_unix_sockets_the_start_fails_for_that(
    socket_path, serve, monkeypatch
):
    monkeypatch.setattr(_local, "_HAS_UNIX_SOCKETS", False)
    service = serve(RobotStateStore(), socket_path, start=False)
    assert str(fails(service)) == "this platform has no Unix domain sockets"
    assert list(Path(socket_path).parent.iterdir()) == []


_WITHOUT_FCNTL = """
import sys
sys.modules["fcntl"] = None
try:
    import fcntl
except ImportError:
    pass
else:
    raise SystemExit("fcntl is still importable")
from isaaccapture.robot_state import RobotState, _local, default_socket_path
assert default_socket_path()
with RobotState.connect(sys.argv[1], robot="first") as reader:
    assert reader.server_reachable
    print(reader.description().robot)
"""


def test_the_module_imports_and_reads_where_fcntl_is_missing(socket_path, serve):
    serve(store_of("first"), socket_path)
    done = subprocess.run(
        [sys.executable, "-c", _WITHOUT_FCNTL, socket_path],
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "first"
