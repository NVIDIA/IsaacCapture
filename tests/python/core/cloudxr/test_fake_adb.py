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


def test_mock_adb_forwards_kwargs() -> None:
    """subprocess.run() callers pass capture_output/text/timeout/check kwargs - the fake
    must accept and ignore them rather than raising a TypeError on an unexpected kwarg."""
    with mock_adb():
        result = subprocess.run(
            ["adb", "get-state"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0
