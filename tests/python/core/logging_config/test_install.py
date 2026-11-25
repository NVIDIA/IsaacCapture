# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No route may crash or silence a process that merely imports IsaacTeleop.

``install()`` runs from ``import isaaccapture``, so whatever it raises is raised
by the host application. Every case here breaks one facility and asks for two
things back: the import succeeded, and the records still went somewhere.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest
from conftest import clean_env, read_all, run_python, session_logs

not_root = pytest.mark.skipif(
    os.getuid() == 0, reason="root is not refused by directory permissions"
)

_PROBE = """
import json, logging, os, sys
import isaaccapture  # noqa: F401 -- the import under test
log = logging.getLogger("isaaccapture.test.install")
log.debug("probe debug")
log.info("probe info")
log.warning("probe warning")
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "socket": os.environ.get("ISAACCAPTURE_LOG_SOCKET"),
            "capture": os.environ.get("ISAACCAPTURE_NATIVE_CAPTURE_FILE"),
        },
        handle,
    )
"""


def probe(tmp_path, log_dir, **overrides):
    report_path = tmp_path / "report.json"
    result = run_python(_PROBE, clean_env(log_dir, **overrides), str(report_path))
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {}
    )
    return result, report


def usable_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@not_root
def test_a_log_directory_it_cannot_create_costs_the_files_and_nothing_else(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        result, report = probe(tmp_path, locked / "logs")
    finally:
        locked.chmod(0o700)

    assert result.returncode == 0
    # Reported through the handlers already attached, which is why install()
    # opens the capture sink after the console handler exists.
    assert "File logging disabled" in result.stderr
    assert "Native output capture disabled" in result.stderr
    assert "probe warning" in result.stderr
    assert report["capture"] is None


def test_a_log_directory_that_is_a_file_costs_the_files_and_nothing_else(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("operator error\n")

    result, _ = probe(tmp_path, blocker)
    assert result.returncode == 0
    assert "File logging disabled" in result.stderr
    assert "probe warning" in result.stderr


@pytest.mark.parametrize("kind", ["leftover", "never-existed"])
def test_an_address_no_one_answers_is_not_believed(tmp_path, kind):
    # A forwarding process gets no console handler and no file handler, so a
    # process that trusted a dead address would emit nothing, anywhere.
    dead = tmp_path / "dead.sock"
    if kind == "leftover":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(dead))
        sock.close()

    log_dir = usable_log_dir(tmp_path)
    result, report = probe(tmp_path, log_dir, ISAACCAPTURE_LOG_SOCKET=str(dead))

    assert result.returncode == 0
    assert report["socket"] != str(dead)
    assert "probe warning" in result.stderr
    assert "probe warning" in read_all(session_logs(log_dir))


def test_a_socket_path_that_will_not_fit_costs_forwarding_and_nothing_else(tmp_path):
    runtime = tmp_path.joinpath(*("d" * 40 for _ in range(3)))
    runtime.mkdir(parents=True)

    log_dir = usable_log_dir(tmp_path)
    result, report = probe(tmp_path, log_dir, XDG_RUNTIME_DIR=str(runtime))

    assert result.returncode == 0
    assert "Log forwarding disabled" in result.stderr
    assert "a Unix domain socket allows" in result.stderr
    assert report["socket"] is None
    # Each process keeping its own file is the documented fallback, not silence.
    assert "probe warning" in read_all(session_logs(log_dir))


@not_root
def test_a_runtime_directory_it_cannot_create_costs_forwarding_and_nothing_else(
    tmp_path,
):
    locked = tmp_path / "runtime-locked"
    locked.mkdir()
    locked.chmod(0o500)
    log_dir = usable_log_dir(tmp_path)
    try:
        # Missing XDG roots are ignored; use an existing root that cannot create children.
        result, report = probe(tmp_path, log_dir, XDG_RUNTIME_DIR=str(locked))
    finally:
        locked.chmod(0o700)

    assert result.returncode == 0
    assert "Log forwarding disabled: cannot use the runtime directory" in result.stderr
    assert report["socket"] is None
    assert "probe warning" in read_all(session_logs(log_dir))


def test_a_junk_log_level_falls_back_to_info(tmp_path):
    log_dir = usable_log_dir(tmp_path)
    result, _ = probe(tmp_path, log_dir, ISAACCAPTURE_LOG_LEVEL="verbose-please")

    assert result.returncode == 0
    assert "probe info" in result.stderr
    assert "probe debug" not in result.stderr
    # The file always captures everything, whatever the console is set to.
    assert "probe debug" in read_all(session_logs(log_dir))


def test_a_host_that_started_with_fd_1_closed_keeps_it_closed(tmp_path):
    # The kernel hands out the lowest free number, so every descriptor this
    # package opens would otherwise land on fd 1 and change the host's state.
    log_dir = usable_log_dir(tmp_path)
    report_path = tmp_path / "report.json"
    code = """
import json, logging, os, sys
import isaaccapture  # noqa: F401
from isaaccapture.logging_config import _native_api


def fd_open(fd):
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


after_import = fd_open(1)
with _native_api.capture_native_output():
    os.write(2, b"vendor noise\\n")
after_scope = fd_open(1)
logging.getLogger("isaaccapture.test.install").warning("probe warning")
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {"after_import": after_import, "after_scope": after_scope, "stdout": sys.stdout is None},
        handle,
    )
"""
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'exec "$0" -c "$1" "$2" 1>&-',
            sys.executable,
            code,
            str(report_path),
        ],
        env=clean_env(log_dir),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stdout"] is True, "the interpreter should have no stdout to follow"
    assert report["after_import"] is False
    # A closed descriptor is left closed rather than pinned to /dev/null.
    assert report["after_scope"] is False
    assert "probe warning" in read_all(session_logs(log_dir))
