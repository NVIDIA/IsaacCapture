# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import signal

import pytest

from capture_server.__main__ import main
from capture_server.server import _build_arg_parser


def test_camera_option_is_rejected_with_mock(capsys):
    """--camera configures real OAK-D devices; mock cameras keep their fixed settings."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--mock", "--camera", "front=ABC"])
    assert excinfo.value.code == 2
    assert "only applies to real" in capsys.readouterr().err


def test_mock_only_options_are_rejected_with_real_acquisition(capsys):
    """--seed/--scenario/--fault/--connect-delay belong to the mock; real setup must not accept them."""
    mock_only = (["--seed", "3"], ["--scenario", "reach"], ["--fault", "startup_error"], ["--connect-delay", "1"])
    for flags in mock_only:
        with pytest.raises(SystemExit) as excinfo:
            main(flags)
        assert excinfo.value.code == 2
        assert "only apply with --mock" in capsys.readouterr().err


def test_arg_parser_defaults():
    args = _build_arg_parser().parse_args(["--mock"])
    assert args.host == "127.0.0.1"
    assert args.port == 8082
    assert args.seed == 0
    assert args.scenario == "demo"
    assert args.connect_delay == 0.0
    assert args.fault == []


def test_arg_parser_round_trips_all_flags():
    args = _build_arg_parser().parse_args(
        [
            "--mock",
            "--host",
            "0.0.0.0",
            "--port",
            "18999",
            "--seed",
            "1",
            "--scenario",
            "reach",
            "--connect-delay",
            "0.5",
            "--fault",
            "startup_error",
            "--fault",
            "sample_freeze",
        ]
    )
    assert (args.host, args.port, args.seed, args.scenario, args.connect_delay) == (
        "0.0.0.0",
        18999,
        1,
        "reach",
        0.5,
    )
    assert args.fault == ["startup_error", "sample_freeze"]


def test_unknown_mock_scenario_is_rejected_at_setup(capsys):
    """The scenario is checked against the mock's scenarios only when the mock is selected."""
    assert main(["--mock", "--scenario", "not-a-scenario"]) == 2
    assert "unknown mock scenario" in capsys.readouterr().err


def test_main_with_mock_flag_starts_and_stops_on_sigterm(unused_tcp_port):
    """Exercises the REAL `capture_server.__main__.main` entry point and
    REAL process signal handling -- a subprocess, not a test-only
    substitute for `_run_cli`: `main` calls `asyncio.run(_run_cli(...))`
    internally, so it can't be awaited from inside this already-running
    event loop; a real child process is the only way to exercise its
    actual SIGINT/SIGTERM handling end to end."""
    import os
    import select
    import subprocess
    import sys
    import time

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "capture_server",
            "--mock",
            "--host",
            "127.0.0.1",
            "--port",
            str(unused_tcp_port),
        ],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        banner_seen = False
        buf = ""
        while time.monotonic() < deadline:
            # select() only proves the fd has SOME bytes available right
            # now, not a full line -- readline() can still make further
            # underlying read() calls looking for the trailing newline and
            # block on a partial (non-newline-terminated) write from the
            # subprocess. A raw bounded os.read() per select()-confirmed
            # wakeup, accumulated and substring-matched ourselves, has no
            # such hidden second blocking call.
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([proc.stderr], [], [], max(remaining, 0))
            if not ready:
                break
            chunk = os.read(proc.stderr.fileno(), 4096)
            if not chunk:
                break  # EOF: the process closed stderr without ever printing the banner
            buf += chunk.decode(errors="replace")
            if "listening on" in buf:
                banner_seen = True
                break
        assert banner_seen, "server never printed its startup banner within the bounded window"

        proc.send_signal(signal.SIGTERM)
        try:
            exit_code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("process did not exit within the bounded shutdown window")
        assert exit_code == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
