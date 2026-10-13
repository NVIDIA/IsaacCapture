# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`conftest.FakeAdb` / :func:`conftest.mock_adb` themselves -
the shared fake-adb test double, not any test that consumes it. If this fake answers a
command wrong, every future oob_teleop_adb test built on it is unreliable regardless of
how correct that test's own assertions look."""

from __future__ import annotations

import subprocess

import pytest

from conftest import FakeAdb, mock_adb


def test_get_state_device_succeeds() -> None:
    fake = FakeAdb(device_state="device")
    result = fake(["adb", "get-state"])
    assert result.returncode == 0
    assert result.stdout == "device"


@pytest.mark.parametrize("state", ["unauthorized", "offline", ""])
def test_get_state_non_device_fails(state: str) -> None:
    fake = FakeAdb(device_state=state)
    result = fake(["adb", "get-state"])
    assert result.returncode == 1
    assert result.stdout == state


def test_reconnect_always_succeeds() -> None:
    fake = FakeAdb(device_state="offline")
    result = fake(["adb", "reconnect"])
    assert result.returncode == 0


def test_devtools_socket_scan_returns_matching_line() -> None:
    fake = FakeAdb(devtools_socket="weblayer_devtools_remote_1234")
    result = fake(["adb", "shell", "cat", "/proc/net/unix"])
    assert result.returncode == 0
    assert "@weblayer_devtools_remote_1234" in result.stdout
    # One token per line, matching the real /proc/net/unix format - a naive substring
    # match elsewhere in the line would be a false positive the real
    # _DEVTOOLS_SOCKET_RE.fullmatch(token) scan wouldn't accept.
    assert result.stdout.split()[-1] == "@weblayer_devtools_remote_1234"


def test_devtools_socket_scan_empty_socket_has_no_matching_token() -> None:
    """devtools_socket="" simulates a browser that never exposed a DevTools socket -
    the output line must contain no token _DEVTOOLS_SOCKET_RE would match, not just an
    empty string, so _discover_devtools_socket()'s real regex scan genuinely finds
    nothing rather than happening to pass on an empty-string edge case."""
    fake = FakeAdb(devtools_socket="")
    result = fake(["adb", "shell", "cat", "/proc/net/unix"])
    assert result.returncode == 0
    assert "_devtools_remote" not in result.stdout


def test_getprop_returns_empty_unknown_vendor() -> None:
    fake = FakeAdb()
    result = fake(["adb", "shell", "getprop", "ro.product.manufacturer"])
    assert result.returncode == 0
    assert result.stdout == ""


def test_forward_uses_configured_rc() -> None:
    fake = FakeAdb(forward_rc=1)
    result = fake(
        ["adb", "forward", "tcp:48889", "localabstract:chrome_devtools_remote"]
    )
    assert result.returncode == 1


def test_forward_remove_always_succeeds_regardless_of_forward_rc() -> None:
    """--remove is teardown, not the forward-establishment call forward_rc models - it
    must succeed even when forward_rc simulates the *establishing* forward failing,
    since a caller cleaning up after a failed launch still needs --remove to work."""
    fake = FakeAdb(forward_rc=1)
    result = fake(["adb", "forward", "--remove", "tcp:48889"])
    assert result.returncode == 0


def test_am_start_uses_configured_rc() -> None:
    fake = FakeAdb(am_start_rc=99)
    result = fake(
        [
            "adb",
            "shell",
            "am start -a android.intent.action.VIEW -d https://headset.local/",
        ]
    )
    assert result.returncode == 99


def test_unrecognized_command_raises() -> None:
    fake = FakeAdb()
    with pytest.raises(AssertionError, match="unscripted adb call"):
        fake(["adb", "reverse", "--list"])


def test_calls_are_recorded_in_order() -> None:
    fake = FakeAdb()
    fake(["adb", "get-state"])
    fake(["adb", "shell", "getprop", "ro.product.brand"])
    assert fake.calls == [
        ["adb", "get-state"],
        ["adb", "shell", "getprop", "ro.product.brand"],
    ]


def test_mock_adb_patches_subprocess_run_and_restores_it() -> None:
    original = subprocess.run
    with mock_adb(device_state="device") as fake:
        assert subprocess.run is not original
        result = subprocess.run(["adb", "get-state"], check=False)
        assert result.returncode == 0
        assert fake.calls == [["adb", "get-state"]]
    assert subprocess.run is original


def test_mock_adb_accepts_subprocess_options() -> None:
    """subprocess.run() callers pass capture_output/text/timeout/check; the fake must
    accept them rather than raise a TypeError on an unexpected kwarg."""
    with mock_adb():
        result = subprocess.run(
            ["adb", "get-state"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0


def test_check_true_returns_on_success_and_raises_on_failure() -> None:
    """check=True is honored, as subprocess.run would: success returns, a non-zero rc
    raises CalledProcessError carrying the command and return code."""
    fake = FakeAdb(forward_rc=1)
    ok = fake(["adb", "forward", "--remove", "tcp:48889"], check=True)
    assert ok.returncode == 0
    cmd = ["adb", "forward", "tcp:48889", "localabstract:chrome_devtools_remote"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        fake(cmd, check=True)
    assert excinfo.value.returncode == 1
    assert excinfo.value.cmd == cmd


@pytest.mark.parametrize(
    "args",
    [
        ["adb", "forward"],
        ["adb", "forward", "--remove"],
        ["adb", "forward", "tcp:48889"],
        ["adb", "forward", "--remove", "48889"],
        ["adb", "shell", "echo am start"],
        ["adb", "shell", "am start"],
        ["adb", "shell", "am start -a android.intent.action.VIEW"],
        ["adb", "shell", "am start -p -a android.intent.action.VIEW -d https://h/"],
        ["adb", "shell", "dumpsys"],
    ],
)
def test_malformed_commands_raise(args: list[str]) -> None:
    """A command that merely resembles a supported one must not silently succeed."""
    with pytest.raises(AssertionError, match="unscripted adb call"):
        FakeAdb()(args)


@pytest.mark.parametrize(
    "shell_cmd",
    [
        "am start -a android.intent.action.VIEW -d https://headset.local/",
        "am start -p com.oculus.browser -a android.intent.action.VIEW -d "
        "'https://headset.local/?a=1&b=2'",
    ],
)
def test_intent_launch_shapes_are_accepted(shell_cmd: str) -> None:
    assert FakeAdb(am_start_rc=0)(["adb", "shell", shell_cmd]).returncode == 0


def test_dumpsys_power_reports_configured_wakefulness() -> None:
    assert (
        "mWakefulness=Awake" in FakeAdb()(["adb", "shell", "dumpsys", "power"]).stdout
    )
    asleep = FakeAdb(wakefulness="Asleep")
    assert "mWakefulness=Asleep" in asleep(["adb", "shell", "dumpsys", "power"]).stdout


def test_pinned_serial_matches_and_is_recorded_as_issued() -> None:
    fake = FakeAdb()
    assert fake(["adb", "-s", "HEADSET", "get-state"]).returncode == 0
    assert fake(["adb", "-s", "HEADSET", "shell", "dumpsys", "power"]).returncode == 0
    assert fake.calls[0] == ["adb", "-s", "HEADSET", "get-state"]


@pytest.mark.parametrize("serial", [None, "HEADSET"])
def test_real_open_url_on_headset_runs_against_the_fake(serial: str | None) -> None:
    """The real launch helper (device-state, wakefulness, vendor probe, am start) must
    complete against the fake, with and without a pinned serial, so the fake cannot
    drift behind the helper it exists to drive."""
    from cloudxr_py_test_ns.oob_teleop_adb import (
        SELECTED_ADB_SERIAL,
        open_url_on_headset,
    )

    token = SELECTED_ADB_SERIAL.set(serial)
    try:
        with mock_adb() as fake:
            rc, diag = open_url_on_headset("https://headset.local/?a=1&b=2")
    finally:
        SELECTED_ADB_SERIAL.reset(token)
    assert (rc, diag) == (0, "")
    assert any("am start" in c[-1] for c in fake.calls)
    if serial:
        assert all(c[1:3] == ["-s", serial] for c in fake.calls)
