# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`oob_teleop_adb` (hints, device validation, bookmark automation with mocked subprocess)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cloudxr_py_test_ns.oob_teleop_adb import (
    OobAdbError,
    adb_automation_failure_hint,
    adb_device_state,
    assert_adb_device_online,
    assert_exactly_one_adb_device,
    coturn_binary_path,
    oob_adb_automation_message,
    require_adb_on_path,
    require_coturn_available,
    run_adb_headset_bookmark,
)


@pytest.fixture(autouse=True)
def _clear_adb_device_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make device-selection tests independent of the developer's shell env."""
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)


@pytest.mark.parametrize(
    "diag,needle",
    [
        ("device unauthorized", "unauthorized"),
        ("no devices/emulators found", "No adb device"),
        ("device not found", "No adb device"),
        ("more than one device or emulator", "Multiple adb devices"),
        ("device offline", "offline"),
    ],
)
def test_adb_automation_failure_hint(diag: str, needle: str) -> None:
    hint = adb_automation_failure_hint(diag)
    assert needle.lower() in hint.lower()


def test_adb_automation_failure_hint_unknown() -> None:
    assert adb_automation_failure_hint("unknown error") == ""


def test_oob_adb_automation_message() -> None:
    msg = oob_adb_automation_message(1, "device offline", "Device offline hint.")
    assert "exit code 1" in msg
    assert "device offline" in msg
    assert "Device offline hint." in msg
    assert "open the teleop URL on the headset" in msg


def test_oob_adb_automation_message_empty_detail() -> None:
    msg = oob_adb_automation_message(2, "", "")
    assert "no output from adb" in msg


@patch("cloudxr_py_test_ns.oob_teleop_adb.shutil.which", return_value="/usr/bin/adb")
def test_require_adb_on_path_found(mock_which: MagicMock) -> None:
    require_adb_on_path()


@patch("cloudxr_py_test_ns.oob_teleop_adb.shutil.which", return_value=None)
def test_require_adb_on_path_missing(mock_which: MagicMock) -> None:
    with pytest.raises(OobAdbError, match="not found on PATH"):
        require_adb_on_path()


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_assert_exactly_one_adb_device_zero_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="List of devices attached\n\n",
        stderr="",
    )
    with pytest.raises(OobAdbError, match="No adb device"):
        assert_exactly_one_adb_device()


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_assert_exactly_one_adb_device_one(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="List of devices attached\nABC123\tdevice\n\n",
        stderr="",
    )
    assert_exactly_one_adb_device()


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_assert_exactly_one_adb_device_two_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=("List of devices attached\nABC123\tdevice\nDEF456\tdevice\n\n"),
        stderr="",
    )
    with pytest.raises(OobAdbError, match="Too many"):
        assert_exactly_one_adb_device()


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_assert_exactly_one_adb_device_pin_via_android_serial(
    mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two devices, but ANDROID_SERIAL pins one — accept and proceed.
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="List of devices attached\nABC123\tdevice\nDEF456\tdevice\n\n",
        stderr="",
    )
    monkeypatch.setenv("ANDROID_SERIAL", "DEF456")
    assert_exactly_one_adb_device()


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_assert_exactly_one_adb_device_pin_unknown_serial_raises(
    mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Operator pinned a serial that is not actually connected — error
    # surfaces the ones that are, so they can fix the typo.
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="List of devices attached\nABC123\tdevice\nDEF456\tdevice\n\n",
        stderr="",
    )
    monkeypatch.setenv("ANDROID_SERIAL", "GHI789")
    with pytest.raises(OobAdbError, match="not currently in `device` state"):
        assert_exactly_one_adb_device()


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_assert_exactly_one_too_many_hints_at_android_serial(
    mock_run: MagicMock,
) -> None:
    # The "too many devices" error must mention ANDROID_SERIAL so the
    # operator knows the disambiguation knob exists.
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="List of devices attached\nABC123\tdevice\nDEF456\tdevice\n\n",
        stderr="",
    )
    with pytest.raises(OobAdbError, match="ANDROID_SERIAL"):
        assert_exactly_one_adb_device()


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_assert_exactly_one_ignores_unauthorized(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=("List of devices attached\nABC123\tdevice\nDEF456\tunauthorized\n\n"),
        stderr="",
    )
    assert_exactly_one_adb_device()


@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value="device")
@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
@patch(
    "cloudxr_py_test_ns.oob_teleop_adb.resolve_lan_host_for_oob",
    return_value="10.0.0.1",
)
def test_run_adb_headset_bookmark_success(
    mock_lan: MagicMock, mock_run: MagicMock, _mock_state: MagicMock
) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    rc, diag = run_adb_headset_bookmark(resolved_port=48322)
    assert rc == 0
    assert diag == ""
    args = mock_run.call_args[0][0]
    assert args[0] == "adb"
    assert args[1] == "shell"
    assert "am start" in args[2]


@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value="device")
@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
@patch(
    "cloudxr_py_test_ns.oob_teleop_adb.resolve_lan_host_for_oob",
    return_value="10.0.0.1",
)
def test_run_adb_headset_bookmark_failure(
    mock_lan: MagicMock, mock_run: MagicMock, _mock_state: MagicMock
) -> None:
    mock_run.return_value = MagicMock(
        returncode=1, stdout="", stderr="no devices/emulators found"
    )
    rc, diag = run_adb_headset_bookmark(resolved_port=48322)
    assert rc == 1
    assert "no devices" in diag


# coturn binary lookup -------------------------------------------------------


@patch("cloudxr_py_test_ns.oob_teleop_adb.os.path.exists", return_value=False)
@patch("cloudxr_py_test_ns.oob_teleop_adb.shutil.which")
def test_coturn_binary_path_prefers_turnserver(
    mock_which: MagicMock, _mock_exists: MagicMock
) -> None:
    mock_which.side_effect = lambda name: (
        "/usr/local/bin/turnserver" if name == "turnserver" else None
    )
    assert coturn_binary_path() == "/usr/local/bin/turnserver"


@patch("cloudxr_py_test_ns.oob_teleop_adb.os.path.exists", return_value=False)
@patch("cloudxr_py_test_ns.oob_teleop_adb.shutil.which")
def test_coturn_binary_path_accepts_coturn_name(
    mock_which: MagicMock, _mock_exists: MagicMock
) -> None:
    mock_which.side_effect = lambda name: (
        "/opt/coturn/bin/coturn" if name == "coturn" else None
    )
    assert coturn_binary_path() == "/opt/coturn/bin/coturn"


@patch("cloudxr_py_test_ns.oob_teleop_adb.shutil.which", return_value=None)
def test_coturn_binary_path_falls_back_to_usr_bin(mock_which: MagicMock) -> None:
    with patch(
        "cloudxr_py_test_ns.oob_teleop_adb.os.path.exists",
        side_effect=lambda p: p == "/usr/bin/coturn",
    ):
        assert coturn_binary_path() == "/usr/bin/coturn"


@patch("cloudxr_py_test_ns.oob_teleop_adb.shutil.which", return_value=None)
@patch("cloudxr_py_test_ns.oob_teleop_adb.os.path.exists", return_value=False)
def test_require_coturn_available_missing_mentions_both_names(
    _mock_exists: MagicMock, _mock_which: MagicMock
) -> None:
    with pytest.raises(OobAdbError) as excinfo:
        require_coturn_available()
    msg = str(excinfo.value)
    assert "turnserver" in msg and "coturn" in msg


# Device-state guard ---------------------------------------------------------


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_adb_device_state_device(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="device\n", stderr="")
    assert adb_device_state() == "device"


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_adb_device_state_unauthorized_via_stderr(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=1, stdout="", stderr="error: device unauthorized\n"
    )
    assert "unauthorized" in adb_device_state()


@patch(
    "cloudxr_py_test_ns.oob_teleop_adb.subprocess.run",
    side_effect=FileNotFoundError(),
)
def test_adb_device_state_no_adb(mock_run: MagicMock) -> None:
    assert adb_device_state() == ""


@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value="device")
def test_assert_adb_device_online_ok(mock_state: MagicMock) -> None:
    assert_adb_device_online()


@pytest.mark.parametrize(
    "state,needle",
    [
        ("unauthorized", "unauthorized"),
        ("error: device unauthorized", "unauthorized"),
        ("", "not responding"),
        ("recovery", "expected `device`"),
    ],
)
def test_assert_adb_device_online_messages(state: str, needle: str) -> None:
    with patch(
        "cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value=state
    ):
        with pytest.raises(OobAdbError) as excinfo:
            assert_adb_device_online()
        assert needle.lower() in str(excinfo.value).lower()


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state")
@patch("cloudxr_py_test_ns.oob_teleop_adb.time.sleep")
def test_assert_adb_device_online_recovers_offline_via_reconnect(
    _mock_sleep: MagicMock, mock_state: MagicMock, mock_run: MagicMock
) -> None:
    mock_state.side_effect = ["offline", "device"]
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    assert_adb_device_online()  # should not raise
    cmd = mock_run.call_args[0][0]
    assert cmd == ["adb", "reconnect"]


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state")
@patch("cloudxr_py_test_ns.oob_teleop_adb.time.sleep")
def test_assert_adb_device_online_offline_persists(
    _mock_sleep: MagicMock, mock_state: MagicMock, mock_run: MagicMock
) -> None:
    mock_state.side_effect = ["offline", "offline"]
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    with pytest.raises(OobAdbError, match="offline"):
        assert_adb_device_online()


@patch("cloudxr_py_test_ns.oob_teleop_adb.time.sleep")
@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value="offline")
@patch(
    "cloudxr_py_test_ns.oob_teleop_adb.resolve_lan_host_for_oob",
    return_value="10.0.0.1",
)
def test_run_adb_headset_bookmark_offline_returns_clean_diag(
    _mock_lan: MagicMock,
    _mock_state: MagicMock,
    _mock_run: MagicMock,
    _mock_sleep: MagicMock,
) -> None:
    rc, diag = run_adb_headset_bookmark(resolved_port=48322)
    assert rc != 0
    assert "offline" in diag.lower()


# Reverse-setup wraps subprocess errors as OobAdbError --------------------------


def test_build_teleop_url_usb_local_uses_resolved_proxy_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """USB-local routing follows the port where the WSS proxy is listening."""
    from cloudxr_py_test_ns.oob_teleop_adb import build_teleop_url

    monkeypatch.delenv("TELEOP_WEB_CLIENT_BASE", raising=False)
    monkeypatch.setenv("PROXY_PORT", "48322")
    url = build_teleop_url(resolved_port=49322, usb_local=True)
    assert "https://localhost:49322/client" in url
    assert "8080" not in url
    assert "serverIP=127.0.0.1" in url
    assert "port=49322" in url


def test_build_teleop_url_host_client_uses_resolved_proxy_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cloudxr_py_test_ns.oob_teleop_adb import build_teleop_url

    monkeypatch.delenv("TELEOP_WEB_CLIENT_BASE", raising=False)
    monkeypatch.setenv("PROXY_PORT", "48322")
    monkeypatch.setenv("TELEOP_PROXY_HOST", "proxy.example.test")
    monkeypatch.setattr(
        "cloudxr_py_test_ns.oob_teleop_env.guess_lan_ipv4",
        lambda: "10.0.0.2",
    )
    url = build_teleop_url(resolved_port=49322, host_client=True)
    assert "https://proxy.example.test:49322/client" in url
    assert "serverIP=proxy.example.test" in url
    assert "10.0.0.2" not in url
    assert "port=49322" in url


def test_build_teleop_url_forwards_reliability_config_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TELEOP_CLIENT_* reconnect/warm-up env vars reach the bookmark URL end to end.

    Without this, a real headset launch leaves reconnectEnabled/streamAttachTimeoutMs/
    warmupBeginTimeoutMs/warmupEndTimeoutMs at their client-side defaults no matter what
    the operator sets in the environment - client_ui_fields_from_env() is a generic dict
    merge (build_teleop_url -> build_headset_bookmark_url), so this exercises the whole
    chain rather than just the two boundary functions in test_oob_teleop_env.py.
    """
    from cloudxr_py_test_ns.oob_teleop_adb import build_teleop_url

    monkeypatch.delenv("TELEOP_WEB_CLIENT_BASE", raising=False)
    monkeypatch.setenv("PROXY_PORT", "48322")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_ENABLED", "true")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_DELAY_MS", "2500")
    monkeypatch.setenv("TELEOP_CLIENT_STREAM_ATTACH_TIMEOUT_MS", "90000")
    monkeypatch.setenv("TELEOP_CLIENT_WARMUP_BEGIN_TIMEOUT_MS", "8000")
    monkeypatch.setenv("TELEOP_CLIENT_WARMUP_END_TIMEOUT_MS", "20000")
    url = build_teleop_url(resolved_port=49322, usb_local=True)
    assert "reconnectEnabled=true" in url
    assert "reconnectMaxAttempts=5" in url
    assert "reconnectDelayMs=2500" in url
    assert "streamAttachTimeoutMs=90000" in url
    assert "warmupBeginTimeoutMs=8000" in url
    assert "warmupEndTimeoutMs=20000" in url


@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value="device")
@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_setup_adb_reverse_ports_uses_resolved_proxy_port(
    mock_run: MagicMock, _mock_state: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """adb reverse uses the listener port, not a conflicting environment value."""
    from cloudxr_py_test_ns.oob_teleop_adb import setup_adb_reverse_ports
    from cloudxr_py_test_ns.oob_teleop_env import USB_BACKEND_DEFAULT_PORT

    monkeypatch.setenv("PROXY_PORT", "48322")
    monkeypatch.delenv("USB_BACKEND_PORT", raising=False)
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    setup_adb_reverse_ports(49322)
    ports = [
        call.args[0][2].removeprefix("tcp:")
        for call in mock_run.call_args_list
        if call.args and call.args[0][:2] == ["adb", "reverse"]
    ]
    assert ports == ["49322", str(USB_BACKEND_DEFAULT_PORT)]
    assert "8080" not in ports


@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_teardown_adb_reverse_ports_uses_resolved_proxy_port(
    mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloudxr_py_test_ns.oob_teleop_adb import teardown_adb_reverse_ports
    from cloudxr_py_test_ns.oob_teleop_env import USB_BACKEND_DEFAULT_PORT

    monkeypatch.setenv("PROXY_PORT", "48322")
    monkeypatch.delenv("USB_BACKEND_PORT", raising=False)
    teardown_adb_reverse_ports(49322)
    assert [call.args[0] for call in mock_run.call_args_list] == [
        ["adb", "reverse", "--remove", "tcp:49322"],
        ["adb", "reverse", "--remove", f"tcp:{USB_BACKEND_DEFAULT_PORT}"],
    ]


@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value="device")
@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
def test_setup_adb_reverse_ports_wraps_called_process_error(
    mock_run: MagicMock, _mock_state: MagicMock
) -> None:
    from cloudxr_py_test_ns.oob_teleop_adb import (
        setup_adb_reverse_ports,
    )
    import subprocess as sp

    mock_run.side_effect = sp.CalledProcessError(
        returncode=1, cmd=["adb"], stderr="error: device offline"
    )
    with pytest.raises(OobAdbError) as excinfo:
        setup_adb_reverse_ports()
    msg = str(excinfo.value)
    assert "adb reverse" in msg
    assert "device offline" in msg


@patch("cloudxr_py_test_ns.oob_teleop_adb.time.sleep")
@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value="offline")
def test_setup_adb_reverse_ports_offline_short_circuits(
    _mock_state: MagicMock, _mock_run: MagicMock, _mock_sleep: MagicMock
) -> None:
    from cloudxr_py_test_ns.oob_teleop_adb import (
        setup_adb_reverse_ports,
    )

    with pytest.raises(OobAdbError, match="offline"):
        setup_adb_reverse_ports()


@patch("cloudxr_py_test_ns.oob_teleop_adb.time.sleep")
@patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run")
@patch("cloudxr_py_test_ns.oob_teleop_adb.adb_device_state", return_value="offline")
def test_setup_adb_reverse_turn_offline_short_circuits(
    _mock_state: MagicMock, _mock_run: MagicMock, _mock_sleep: MagicMock
) -> None:
    from cloudxr_py_test_ns.oob_teleop_adb import (
        setup_adb_reverse_turn,
    )

    with pytest.raises(OobAdbError, match="offline"):
        setup_adb_reverse_turn(3478)


# WiFi-drop monitor (H6) -----------------------------------------------------


import asyncio  # noqa: E402

from cloudxr_py_test_ns.oob_teleop_adb import monitor_headset_wifi  # noqa: E402


async def test_monitor_headset_wifi_warns_on_drop(capsys) -> None:
    # Sequence: had ifaces → still ifaces → drops → still dropped.
    seq = [
        [("wlan0", "10.0.0.1")],
        [("wlan0", "10.0.0.1")],
        [],
        [],
    ]
    with patch(
        "cloudxr_py_test_ns.oob_teleop_adb.headset_non_loopback_interfaces",
        side_effect=lambda: seq.pop(0) if seq else [],
    ):
        task = asyncio.create_task(monitor_headset_wifi(poll_seconds=0.001))
        # Poll for the warning rather than racing a fixed sleep budget. On
        # Windows, asyncio.sleep resolution (~15ms timer tick) plus to_thread
        # dispatch makes the two loop iterations needed to detect the drop
        # blow past a 50ms budget.
        out = ""
        for _ in range(200):  # up to ~2s
            await asyncio.sleep(0.01)
            out += capsys.readouterr().err
            if "Headset Wi-Fi dropped" in out:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    out += capsys.readouterr().err
    assert "Headset Wi-Fi dropped" in out
    # Reason should be spelled out so operators don't think USB-local removed the WiFi requirement.
    assert "required even in USB-local mode" in out


async def test_monitor_headset_wifi_silent_when_steady(capsys) -> None:
    with patch(
        "cloudxr_py_test_ns.oob_teleop_adb.headset_non_loopback_interfaces",
        return_value=[("wlan0", "10.0.0.1")],
    ):
        task = asyncio.create_task(monitor_headset_wifi(poll_seconds=0.001))
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert capsys.readouterr().err == ""


# Coturn watchdog (H7) -------------------------------------------------------


from cloudxr_py_test_ns.oob_teleop_adb import watch_coturn  # noqa: E402


from cloudxr_py_test_ns.oob_teleop_adb import _teleop_error_hint  # noqa: E402


@pytest.mark.parametrize(
    "banner,needle",
    [
        ("CloudXR session stopped (0xC0F2220F)", "ICE candidates"),
        ("No local connection candidates", "ICE candidates"),
        ("wss connection close 1006", "WSS dropped"),
        ("something unrelated", ""),
    ],
)
def test_teleop_error_hint(banner: str, needle: str) -> None:
    hint = _teleop_error_hint(banner)
    if needle:
        assert needle.lower() in hint.lower()
    else:
        assert hint == ""


async def test_watch_coturn_restarts_once_then_gives_up(capsys) -> None:
    dead_proc = MagicMock()
    dead_proc.poll.return_value = 1
    dead_proc.returncode = 1
    proc_box = [dead_proc]
    new_proc = MagicMock(pid=4242)
    new_proc.poll.return_value = 1  # also dead, triggers give-up
    new_proc.returncode = 1

    with (
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb.start_coturn", return_value=new_proc
        ) as mock_start,
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb._tail_file", return_value="<log tail>"
        ),
    ):
        task = asyncio.create_task(
            watch_coturn(
                proc_box,
                turn_port=3478,
                user="u",
                credential="c",
                poll_seconds=0.001,
            )
        )
        await asyncio.wait_for(task, timeout=1.0)
    assert mock_start.call_count == 1
    assert proc_box[0] is new_proc
    assert "died again" in capsys.readouterr().err


# ============================================================================
# CDP: _discover_devtools_socket / _close_stale_teleop_tabs
#
# _discover_devtools_socket mocks only `_run_adb` (no real device). _close_stale_teleop_tabs
# drives the real function against a genuine local HTTP server standing in for Chromium's CDP
# `/json` endpoint - only the two `adb forward` calls either side of it are mocked, since there's
# no real device - so the tab-matching/close-request logic itself is exercised for real rather
# than asserted via mocked call arguments.
# ============================================================================

import http.server  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from typing import ClassVar  # noqa: E402

from cloudxr_py_test_ns.oob_teleop_adb import (  # noqa: E402
    _CDP_LOCAL_PORT,
    _close_stale_teleop_tabs,
    _discover_devtools_socket,
)


@patch("cloudxr_py_test_ns.oob_teleop_adb._run_adb")
def test_discover_devtools_socket_prefers_weblayer_over_generic(
    mock_run_adb: MagicMock,
) -> None:
    mock_run_adb.return_value = (
        "0: 00000003 00000000 00010000 0001 01 12345 @chrome_devtools_remote\n"
        "1: 00000003 00000000 00010000 0001 01 12346 @weblayer_devtools_remote_777\n"
    )
    assert _discover_devtools_socket() == "weblayer_devtools_remote_777"


@patch("cloudxr_py_test_ns.oob_teleop_adb._run_adb")
def test_discover_devtools_socket_falls_back_to_first_candidate(
    mock_run_adb: MagicMock,
) -> None:
    mock_run_adb.return_value = (
        "0: 00000003 00000000 00010000 0001 01 12345 @some.custom_devtools_remote\n"
    )
    assert _discover_devtools_socket() == "some.custom_devtools_remote"


@patch("cloudxr_py_test_ns.oob_teleop_adb._run_adb")
def test_discover_devtools_socket_no_candidates_returns_none(
    mock_run_adb: MagicMock,
) -> None:
    mock_run_adb.return_value = (
        "0: 00000003 00000000 00010000 0001 01 12345 @unrelated_socket\n"
    )
    assert _discover_devtools_socket() is None


@patch("cloudxr_py_test_ns.oob_teleop_adb._run_adb")
def test_discover_devtools_socket_adb_unreachable_returns_none(
    mock_run_adb: MagicMock,
) -> None:
    mock_run_adb.return_value = None
    assert _discover_devtools_socket() is None


class _FakeCdpHandler(http.server.BaseHTTPRequestHandler):
    """Stands in for Chromium's CDP HTTP endpoint: GET /json (tab list) and GET /json/close/<id>."""

    tabs: ClassVar[list[dict]] = []
    closed_ids: ClassVar[list[str]] = []

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's own naming)
        if self.path == "/json":
            body = json.dumps(self.tabs).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/json/close/"):
            self.closed_ids.append(self.path.removeprefix("/json/close/"))
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(
        self, *args
    ) -> None:  # silence BaseHTTPRequestHandler's request logging
        pass


@contextmanager
def _fake_cdp_server(tabs: list[dict]):
    """Serves *tabs* on `_CDP_LOCAL_PORT` for the duration of the block; yields the handler class
    so the test can inspect `.closed_ids` after."""
    _FakeCdpHandler.tabs = tabs
    _FakeCdpHandler.closed_ids = []
    # Must bind the real _CDP_LOCAL_PORT, not an OS-assigned one: _close_stale_teleop_tabs()
    # hardcodes that constant internally rather than taking a port parameter.
    server = http.server.HTTPServer(("localhost", _CDP_LOCAL_PORT), _FakeCdpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _FakeCdpHandler
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_remove")
@patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_cdp")
@patch("cloudxr_py_test_ns.oob_teleop_adb._discover_devtools_socket")
def test_close_stale_teleop_tabs_closes_only_oob_tabs(
    mock_discover: MagicMock,
    mock_forward: MagicMock,
    mock_remove: MagicMock,
) -> None:
    mock_discover.return_value = "chrome_devtools_remote"
    tabs = [
        {"id": "stale-1", "url": "https://headset.local/?oobEnable=1"},
        {"id": "unrelated", "url": "https://headset.local/some-other-page"},
        {"id": "stale-2", "url": "https://headset.local/?oobEnable=1&codec=av1"},
    ]
    with _fake_cdp_server(tabs) as handler:
        closed = _close_stale_teleop_tabs()

    assert closed == 2
    assert sorted(handler.closed_ids) == ["stale-1", "stale-2"]
    mock_forward.assert_called_once_with("chrome_devtools_remote", _CDP_LOCAL_PORT)
    mock_remove.assert_called_once_with(_CDP_LOCAL_PORT)


@patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_remove")
@patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_cdp")
@patch("cloudxr_py_test_ns.oob_teleop_adb._discover_devtools_socket")
def test_close_stale_teleop_tabs_no_matches_closes_nothing(
    mock_discover: MagicMock,
    mock_forward: MagicMock,
    mock_remove: MagicMock,
) -> None:
    mock_discover.return_value = "chrome_devtools_remote"
    tabs = [{"id": "unrelated", "url": "https://headset.local/some-other-page"}]
    with _fake_cdp_server(tabs) as handler:
        closed = _close_stale_teleop_tabs()

    assert closed == 0
    assert handler.closed_ids == []
    mock_remove.assert_called_once_with(_CDP_LOCAL_PORT)


@patch("cloudxr_py_test_ns.oob_teleop_adb._discover_devtools_socket")
def test_close_stale_teleop_tabs_no_devtools_socket_returns_zero(
    mock_discover: MagicMock,
) -> None:
    mock_discover.return_value = None
    assert _close_stale_teleop_tabs() == 0


@patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_remove")
@patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_cdp")
@patch("cloudxr_py_test_ns.oob_teleop_adb._discover_devtools_socket")
def test_close_stale_teleop_tabs_forward_failure_returns_zero_without_remove(
    mock_discover: MagicMock,
    mock_forward: MagicMock,
    mock_remove: MagicMock,
) -> None:
    """`adb forward` failing means no rule was ever set up - `_adb_forward_remove` (a `--remove`
    of that rule) is therefore not called, unlike the success-then-cleanup path above."""
    mock_discover.return_value = "chrome_devtools_remote"
    mock_forward.side_effect = OobAdbError("adb forward failed")
    assert _close_stale_teleop_tabs() == 0
    mock_remove.assert_not_called()


# ============================================================================
# CDP: _cdp_session_click_connect
#
# Drives the real function against a genuine local CDP WebSocket server (unlike
# _close_stale_teleop_tabs's HTTP /json above, this function speaks CDP entirely over one
# WebSocket) - _CdpScript below is a scripted responder standing in for a real Chromium
# DevTools target, so the cert-interstitial bypass and readiness-poll state machine run for
# real rather than being asserted via mocked call arguments. Unlike _CDP_LOCAL_PORT above, the
# WebSocket URL is a plain function argument, so each test gets its own OS-assigned port.
# ============================================================================

from contextlib import asynccontextmanager  # noqa: E402

from websockets.asyncio.server import serve as ws_serve  # noqa: E402

from cloudxr_py_test_ns.oob_teleop_adb import _cdp_session_click_connect  # noqa: E402


class _CdpScript:
    """Scripted responder for the CDP calls `_cdp_session_click_connect` makes.

    *readiness_states* is consumed one entry per readiness-poll call (repeating the last entry
    once exhausted). *interstitial_url* starting with `chrome-error:` forces the DOM
    click-through fallback instead of the primary `Page.navigate` bypass - the same branch
    condition the real function itself checks.
    """

    def __init__(
        self,
        *,
        interstitial: bool = False,
        interstitial_url: str = "https://headset.local/",
        readiness_states: list[dict] | None = None,
    ) -> None:
        self.interstitial = interstitial
        self.interstitial_url = interstitial_url
        self.readiness_states = readiness_states or [
            {"state": "ready", "text": "CONNECT", "disabled": False, "x": 10, "y": 20}
        ]
        self._readiness_index = 0
        self.calls: list[tuple[str, dict]] = []

    def respond(self, method: str, params: dict) -> dict:
        self.calls.append((method, params))
        expr = params.get("expression", "")

        if method in (
            "Security.setIgnoreCertificateErrors",
            "Page.bringToFront",
            "Page.navigate",
            "Input.dispatchMouseEvent",
        ):
            return {}
        if expr == "!!document.getElementById('details-button')":
            return {"result": {"value": self.interstitial}}
        if expr == "window.location.href":
            return {"result": {"value": self.interstitial_url}}
        if expr in (
            "document.getElementById('details-button')?.click()",
            "document.getElementById('proceed-link')?.click()",
            "document.getElementById('startButton')?.click()",
        ):
            return {}
        if "getBoundingClientRect" in expr:
            i = min(self._readiness_index, len(self.readiness_states) - 1)
            self._readiness_index += 1
            return {"result": {"value": self.readiness_states[i]}}
        if "errorMessageBox" in expr:
            # Reports the session as already active (btnText != 'CONNECT') so the post-click
            # outcome-monitor loop returns on its first iteration instead of polling for real.
            return {"result": {"value": {"btnText": "DISCONNECT", "errorText": None}}}
        raise AssertionError(f"unscripted CDP call: {method} {params}")


@asynccontextmanager
async def _fake_cdp_ws(script: _CdpScript):
    """Serves *script*'s responses on an OS-assigned port; yields the `ws://` URL to connect to."""

    async def handler(websocket):
        async for raw in websocket:
            msg = json.loads(raw)
            result = script.respond(msg["method"], msg.get("params", {}))
            await websocket.send(json.dumps({"id": msg["id"], "result": result}))

    async with ws_serve(handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://localhost:{port}"


async def test_cdp_session_click_connect_no_interstitial_reaches_ready_and_clicks() -> (
    None
):
    script = _CdpScript(
        interstitial=False,
        readiness_states=[
            {
                "state": "initializing",
                "text": "CONNECT (checking capabilities)",
                "disabled": True,
            },
            {"state": "ready", "text": "CONNECT", "disabled": False, "x": 10, "y": 20},
        ],
    )
    async with _fake_cdp_ws(script) as ws_url:
        await _cdp_session_click_connect(ws_url)
    methods = [m for m, _ in script.calls]
    assert "Input.dispatchMouseEvent" in methods
    assert "Page.navigate" not in methods


async def test_cdp_session_click_connect_failed_capability_check_raises() -> None:
    script = _CdpScript(
        interstitial=False,
        readiness_states=[
            {
                "state": "failed",
                "text": "CONNECT (capability check failed)",
                "disabled": True,
            }
        ],
    )
    async with _fake_cdp_ws(script) as ws_url:
        with pytest.raises(OobAdbError, match="startButton marked failed"):
            await _cdp_session_click_connect(ws_url)
    assert "Input.dispatchMouseEvent" not in [m for m, _ in script.calls]


async def test_cdp_session_click_connect_cert_interstitial_primary_bypass() -> None:
    script = _CdpScript(interstitial=True, interstitial_url="https://headset.local/")
    async with _fake_cdp_ws(script) as ws_url:
        await _cdp_session_click_connect(ws_url)
    methods = [m for m, _ in script.calls]
    fallback_clicks = [
        p.get("expression")
        for m, p in script.calls
        if m == "Runtime.evaluate" and "proceed-link" in p.get("expression", "")
    ]
    assert "Page.navigate" in methods
    assert fallback_clicks == []


async def test_cdp_session_click_connect_cert_interstitial_dom_fallback() -> None:
    script = _CdpScript(
        interstitial=True, interstitial_url="chrome-error://chromewebdata/"
    )
    async with _fake_cdp_ws(script) as ws_url:
        await _cdp_session_click_connect(ws_url)
    methods = [m for m, _ in script.calls]
    fallback_clicks = [
        p.get("expression")
        for m, p in script.calls
        if m == "Runtime.evaluate" and "proceed-link" in p.get("expression", "")
    ]
    assert "Page.navigate" not in methods
    assert fallback_clicks == ["document.getElementById('proceed-link')?.click()"]
