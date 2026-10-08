# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`oob_teleop_adb` (hints, device validation, bookmark automation with mocked subprocess)."""

from __future__ import annotations

import asyncio
import http.server
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from typing import ClassVar
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import serve as ws_serve

from cloudxr_py_test_ns import oob_teleop_adb as adb_module
from cloudxr_py_test_ns.oob_teleop_adb import (
    OobAdbError,
    _CDP_LOCAL_PORT,
    _cdp_session_click_connect,
    _close_stale_teleop_tabs,
    _discover_devtools_socket,
    adb_automation_failure_hint,
    adb_device_state,
    assert_adb_device_online,
    assert_exactly_one_adb_device,
    build_teleop_url,
    coturn_binary_path,
    oob_adb_automation_message,
    require_adb_on_path,
    require_coturn_available,
    run_adb_headset_bookmark,
    run_oob_connect,
)


def test_build_teleop_url_includes_default_reconnect_policy(monkeypatch) -> None:
    for name in (
        "TELEOP_CLIENT_RECONNECT_MAX_ATTEMPTS",
        "TELEOP_CLIENT_RECONNECT_DELAY_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    url = build_teleop_url(
        resolved_port=48322,
        usb_local=True,
        web_client_base="https://localhost:8080",
    )
    query = parse_qs(urlparse(url).query)
    assert "reconnectEnabled" not in query
    assert query["reconnectMaxAttempts"] == ["10"]
    assert query["reconnectDelayMs"] == ["3000"]


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


@pytest.mark.asyncio
async def test_local_client_reloads_without_http_cache_before_connect() -> None:
    """The updated local bundle is fetched even if this URL was cached before."""

    class FakeCDP:
        def __init__(self) -> None:
            self.methods: list[str] = []
            self.last: dict = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, raw: str) -> None:
            self.last = json.loads(raw)
            self.methods.append(self.last["method"])

        async def recv(self) -> str:
            result = {}
            if self.last["method"] == "Runtime.evaluate":
                expression = self.last["params"]["expression"]
                if "details-button" in expression:
                    value = False
                elif "document.readyState" in expression:
                    value = {"state": "ready", "x": 12, "y": 34}
                elif "const btnText" in expression:
                    value = {"btnText": "DISCONNECT", "errorText": None}
                else:
                    value = None
                result = {"result": {"value": value}}
            return json.dumps({"id": self.last["id"], "result": result})

    cdp = FakeCDP()
    dispatched = []
    milestones = []

    async def loaded():
        milestones.append("clientLoaded")

    def on_dispatched():
        dispatched.append(list(cdp.methods))

        async def publish():
            assert "startButton" in cdp.last["params"]["expression"]
            milestones.append("connectDispatched")

        return publish()

    with patch("websockets.asyncio.client.connect", return_value=cdp):
        await adb_module._cdp_session_click_connect(
            "ws://test",
            refresh_static_assets=True,
            on_client_loaded=loaded,
            on_dispatched=on_dispatched,
        )
    assert milestones == ["clientLoaded", "connectDispatched"]
    assert len(dispatched) == 1
    assert dispatched[0][-1] == "Input.dispatchMouseEvent"
    assert cdp.methods.index("Network.setCacheDisabled") < cdp.methods.index(
        "Page.reload"
    )
    assert cdp.methods.index("Page.reload") < cdp.methods.index(
        "Input.dispatchMouseEvent"
    )


@pytest.mark.asyncio
async def test_attach_existing_tab_clicks_without_navigation_or_tab_cleanup() -> None:
    async def immediate(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def monitor(*_args):
        await asyncio.Event().wait()

    with (
        patch.object(adb_module.asyncio, "to_thread", side_effect=immediate),
        patch.object(adb_module, "_discover_devtools_socket", return_value="socket"),
        patch.object(adb_module, "_adb_forward_cdp"),
        patch.object(
            adb_module,
            "_cdp_list_tabs",
            return_value=[
                {
                    "url": "https://localhost:8080/other",
                    "webSocketDebuggerUrl": "ws://other",
                },
                {
                    "url": "https://localhost:8080/?oobEnable=1",
                    "webSocketDebuggerUrl": "ws://teleop",
                },
            ],
        ),
        patch.object(adb_module, "_cdp_session_click_connect") as click,
        patch.object(adb_module, "_cdp_send_reset_panel_key") as reset_key,
        patch.object(adb_module, "_monitor_teleop_error_banner", side_effect=monitor),
        patch.object(adb_module, "_close_stale_teleop_tabs") as close_tabs,
        patch.object(adb_module, "run_adb_headset_bookmark") as launch,
    ):
        task = await adb_module.attach_existing_oob_tab(
            resolved_port=48322, click_connect=True
        )
        click.assert_awaited_once_with(
            "ws://teleop",
            refresh_static_assets=False,
            allow_navigation=False,
            clear_stale_error=True,
            on_dispatched=None,
            on_client_loaded=None,
        )
        # Unconditional: this reattach path is the one case the client's own
        # automatic first-XR-frame reset never covers (no new session starts).
        reset_key.assert_awaited_once_with("ws://teleop")
        close_tabs.assert_not_called()
        launch.assert_not_called()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_attach_existing_tab_without_exact_oob_page_cleans_forward() -> None:
    async def immediate(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with (
        patch.object(adb_module.asyncio, "to_thread", side_effect=immediate),
        patch.object(adb_module, "_discover_devtools_socket", return_value="socket"),
        patch.object(adb_module, "_adb_forward_cdp"),
        patch.object(
            adb_module,
            "_cdp_list_tabs",
            return_value=[
                {"url": "https://localhost:8080/", "webSocketDebuggerUrl": "ws://other"}
            ],
        ),
        patch.object(adb_module, "_adb_forward_remove") as cleanup,
        patch.object(adb_module, "_cdp_session_click_connect") as click,
    ):
        with pytest.raises(OobAdbError, match="no surviving teleop tab"):
            await adb_module.attach_existing_oob_tab(
                resolved_port=48322, click_connect=True
            )
    click.assert_not_awaited()
    cleanup.assert_called_once_with(9223)


@pytest.mark.asyncio
async def test_attach_existing_tab_monitor_runs_with_real_signature() -> None:
    """Regression test: attach_existing_oob_tab() must call the real (not mocked)
    _monitor_teleop_error_banner() with a signature it actually accepts, or the monitor
    task fails immediately with TypeError the instant it starts running - which a mocked
    monitor can't catch, since the mock accepts any arguments."""
    script = _MonitorScript(banners=[""])
    async with _fake_cdp_ws(script) as ws_url:

        async def immediate(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with (
            patch.object(adb_module.asyncio, "to_thread", side_effect=immediate),
            patch.object(
                adb_module, "_discover_devtools_socket", return_value="socket"
            ),
            patch.object(adb_module, "_adb_forward_cdp"),
            # The monitor's finally block tears down its forward on exit (cancellation here) -
            # unmocked, that's a real subprocess call to the system adb binary, which CI runners
            # without one installed don't have.
            patch.object(adb_module, "_adb_forward_remove"),
            patch.object(
                adb_module,
                "_cdp_list_tabs",
                return_value=[
                    {
                        "url": "https://localhost:8080/?oobEnable=1",
                        "webSocketDebuggerUrl": ws_url,
                    }
                ],
            ),
        ):
            task = await adb_module.attach_existing_oob_tab(
                resolved_port=48322, click_connect=False
            )
            # Give the task at least one real iteration on the real monitor loop
            # before cancelling, so a TypeError at startup would actually surface.
            await asyncio.sleep(1.5)
            assert not task.done(), f"monitor task ended early: {task.exception()}"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


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

    client_ui_fields_from_env() is a generic dict merge (build_teleop_url ->
    build_headset_bookmark_url), so this exercises the whole launch-path chain rather than
    just the two boundary functions in test_oob_teleop_env.py.
    """
    from cloudxr_py_test_ns.oob_teleop_adb import build_teleop_url

    monkeypatch.delenv("TELEOP_WEB_CLIENT_BASE", raising=False)
    monkeypatch.setenv("PROXY_PORT", "48322")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("TELEOP_CLIENT_RECONNECT_DELAY_MS", "2500")
    monkeypatch.setenv("TELEOP_CLIENT_STREAM_ATTACH_TIMEOUT_MS", "90000")
    monkeypatch.setenv("TELEOP_CLIENT_WARMUP_BEGIN_TIMEOUT_MS", "8000")
    monkeypatch.setenv("TELEOP_CLIENT_WARMUP_END_TIMEOUT_MS", "20000")
    url = build_teleop_url(resolved_port=49322, usb_local=True)
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


from cloudxr_py_test_ns.oob_teleop_adb import (  # noqa: E402
    HeadsetNetworkProbe,
    HeadsetNetworkState,
    monitor_headset_wifi,
    probe_headset_network,
    SELECTED_ADB_SERIAL,
)


@pytest.mark.parametrize(
    ("returncode", "stdout", "state"),
    [
        (
            0,
            "20: wlan0 inet 10.0.0.1/24 scope global wlan0",
            HeadsetNetworkState.NETWORK_PRESENT,
        ),
        (0, "1: lo inet 127.0.0.1/8 scope host lo", HeadsetNetworkState.NO_NETWORK),
        (0, "", HeadsetNetworkState.NO_NETWORK),
        (1, "", HeadsetNetworkState.ADB_UNAVAILABLE),
    ],
)
def test_network_probe_preserves_adb_outcome(returncode, stdout, state):
    proc = subprocess.CompletedProcess(
        [], returncode, stdout, "device offline" if returncode else ""
    )
    with patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run", return_value=proc):
        result = probe_headset_network(serial="headset-1")
    assert result.state is state
    assert bool(result.interfaces) is (state is HeadsetNetworkState.NETWORK_PRESENT)


def test_network_probe_uses_selected_serial():
    proc = subprocess.CompletedProcess([], 0, "", "")
    with patch(
        "cloudxr_py_test_ns.oob_teleop_adb.subprocess.run", return_value=proc
    ) as run:
        token = SELECTED_ADB_SERIAL.set("pinned")
        try:
            probe_headset_network()
        finally:
            SELECTED_ADB_SERIAL.reset(token)
    assert run.call_args.args[0][:3] == ["adb", "-s", "pinned"]


@pytest.mark.parametrize(
    "error", [FileNotFoundError("adb"), subprocess.TimeoutExpired("adb", 5)]
)
def test_network_probe_transport_exceptions(error):
    with patch("cloudxr_py_test_ns.oob_teleop_adb.subprocess.run", side_effect=error):
        assert probe_headset_network().state is HeadsetNetworkState.ADB_UNAVAILABLE


async def test_wifi_monitor_ignores_adb_disconnect(caplog):
    present = HeadsetNetworkProbe(
        HeadsetNetworkState.NETWORK_PRESENT, (("wlan0", "10.0.0.1"),)
    )
    unavailable = HeadsetNetworkProbe(HeadsetNetworkState.ADB_UNAVAILABLE)
    sequence = [present, unavailable, present]

    async def immediate(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with (
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb.probe_headset_network",
            side_effect=lambda: sequence.pop(0) if sequence else present,
        ),
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb.asyncio.to_thread", side_effect=immediate
        ),
    ):
        task = asyncio.create_task(monitor_headset_wifi(poll_seconds=0.001))
        await asyncio.sleep(0.02)
        task.cancel()
        await task
    assert not any("Wi-Fi dropped" in record.message for record in caplog.records)


async def test_monitor_headset_wifi_warns_on_drop(caplog) -> None:
    # Sequence: had ifaces → still ifaces → drops → still dropped.
    present = HeadsetNetworkProbe(
        HeadsetNetworkState.NETWORK_PRESENT, (("wlan0", "10.0.0.1"),)
    )
    absent = HeadsetNetworkProbe(HeadsetNetworkState.NO_NETWORK)
    seq = [present, present, absent, absent]

    async def immediate(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with (
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb.probe_headset_network",
            side_effect=lambda: seq.pop(0) if seq else absent,
        ),
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb.asyncio.to_thread", side_effect=immediate
        ),
    ):
        task = asyncio.create_task(monitor_headset_wifi(poll_seconds=0.001))
        # Poll for the warning rather than racing a fixed sleep budget. On
        # Windows, asyncio.sleep resolution (~15ms timer tick) plus to_thread
        # dispatch makes the two loop iterations needed to detect the drop
        # blow past a 50ms budget.
        for _ in range(200):  # up to ~2s
            await asyncio.sleep(0.01)
            if any(
                "Headset Wi-Fi dropped" in record.message for record in caplog.records
            ):
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    messages = [record.message for record in caplog.records]
    assert any("Headset Wi-Fi dropped" in message for message in messages)
    assert any("required even in USB-local mode" in message for message in messages)


async def test_monitor_headset_wifi_silent_when_steady(caplog) -> None:
    async def immediate(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with (
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(
                HeadsetNetworkState.NETWORK_PRESENT, (("wlan0", "10.0.0.1"),)
            ),
        ),
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb.asyncio.to_thread", side_effect=immediate
        ),
    ):
        task = asyncio.create_task(monitor_headset_wifi(poll_seconds=0.001))
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert not any("Wi-Fi dropped" in record.message for record in caplog.records)


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


async def test_watch_coturn_restarts_once_then_gives_up(caplog) -> None:
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
    assert any("died again" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_error_banner_monitor_removes_forward_off_event_loop() -> None:
    """A dropped CDP connection removes its forward without blocking WSS."""
    loop_thread = threading.get_ident()
    cleanup_calls: list[tuple[int, int]] = []

    def remove_forward(port: int) -> None:
        cleanup_calls.append((port, threading.get_ident()))

    async def off_loop(fn, *args):
        # Own the worker so pytest-asyncio does not inherit a pending default
        # executor thread from the module's other ADB tests.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return await asyncio.get_running_loop().run_in_executor(
                pool, lambda: fn(*args)
            )

    with (
        patch("websockets.asyncio.client.connect", side_effect=OSError("CDP closed")),
        patch.object(adb_module, "_adb_forward_remove", side_effect=remove_forward),
        patch.object(adb_module.asyncio, "to_thread", side_effect=off_loop),
    ):
        await adb_module._monitor_teleop_error_banner("ws://test", 9222)

    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][0] == 9222
    assert cleanup_calls[0][1] != loop_thread


# ============================================================================
# CDP: _discover_devtools_socket / _close_stale_teleop_tabs
#
# _discover_devtools_socket mocks only `_run_adb` (no real device). _close_stale_teleop_tabs
# drives the real function against a genuine local HTTP server standing in for Chromium's CDP
# `/json` endpoint - only the two `adb forward` calls either side of it are mocked, since there's
# no real device - so the tab-matching/close-request logic itself is exercised for real rather
# than asserted via mocked call arguments.
# ============================================================================


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
    """Serves *tabs* on an OS-assigned port for the duration of the block; yields the handler
    class so the test can inspect `.closed_ids` after.

    _close_stale_teleop_tabs() reads the module-level _CDP_LOCAL_PORT constant rather than
    taking a port parameter, so this patches that constant to the assigned port for the
    block's lifetime instead of binding the real (production) port - avoids colliding with an
    actual adb forward or other listener already using it on the test host.
    """
    _FakeCdpHandler.tabs = tabs
    _FakeCdpHandler.closed_ids = []
    server = http.server.HTTPServer(("localhost", 0), _FakeCdpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.object(adb_module, "_CDP_LOCAL_PORT", server.server_port):
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
        # Read the port while the fixture's patch is still active - it's restored to the
        # real _CDP_LOCAL_PORT once this block exits.
        patched_port = adb_module._CDP_LOCAL_PORT

    assert closed == 2
    assert sorted(handler.closed_ids) == ["stale-1", "stale-2"]
    mock_forward.assert_called_once_with("chrome_devtools_remote", patched_port)
    mock_remove.assert_called_once_with(patched_port)


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
        patched_port = adb_module._CDP_LOCAL_PORT

    assert closed == 0
    assert handler.closed_ids == []
    mock_remove.assert_called_once_with(patched_port)


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
        self._reached_ready = False
        self.calls: list[tuple[str, dict]] = []

    def respond(self, method: str, params: dict) -> dict:
        self.calls.append((method, params))
        expr = params.get("expression", "")

        if method == "Input.dispatchMouseEvent":
            # A real browser can't be clicked before the button is actionable - reject here so
            # a caller that dispatches too early fails the test instead of silently "succeeding".
            if not self._reached_ready:
                raise AssertionError(
                    "Input.dispatchMouseEvent before CONNECT reached its ready state"
                )
            return {}
        if method in (
            "Security.setIgnoreCertificateErrors",
            "Page.bringToFront",
            "Page.navigate",
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
            state = self.readiness_states[i]
            if state.get("state") == "ready":
                self._reached_ready = True
            return {"result": {"value": state}}
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
    assert "Page.navigate" not in methods
    # Ordered mousePressed then mouseReleased at the readiness poll's (x, y), followed by the
    # DOM startButton.click() safety net - not just "a dispatchMouseEvent happened somewhere",
    # which would also pass for an incomplete click (e.g. missing mouseReleased).
    mouse_events = [p for m, p in script.calls if m == "Input.dispatchMouseEvent"]
    assert [e["type"] for e in mouse_events] == ["mousePressed", "mouseReleased"]
    for event in mouse_events:
        assert event["x"] == 10
        assert event["y"] == 20
        assert event["button"] == "left"
        assert event["clickCount"] == 1
    dispatch_index = methods.index("Input.dispatchMouseEvent")
    click_index = next(
        i
        for i, (m, p) in enumerate(script.calls)
        if m == "Runtime.evaluate"
        and p.get("expression") == "document.getElementById('startButton')?.click()"
    )
    assert click_index > dispatch_index


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


# ============================================================================
# _find_and_click_teleop_tab: the tab-find + click-CONNECT helper run_oob_connect() uses
# for both a genuinely fresh launch and OobLifecycle's own full-bootstrap recovery
# (_automate(), which just calls run_oob_connect() again). Combines the two fake servers
# above: _fake_cdp_server (HTTP /json tab list, bound to the real _CDP_LOCAL_PORT) and
# _fake_cdp_ws (WebSocket CDP, OS-assigned port, referenced by the fake tab's
# webSocketDebuggerUrl).
# ============================================================================

import time  # noqa: E402

from cloudxr_py_test_ns.oob_teleop_adb import _find_and_click_teleop_tab  # noqa: E402


async def test_find_and_click_teleop_tab_finds_already_navigated_tab() -> None:
    """A tab whose URL already matches at snapshot time (Case C - no visible diff between
    polls) is still found and clicked, without ever needing to re-fire am start."""
    script = _CdpScript(interstitial=False)
    async with _fake_cdp_ws(script) as ws_url:
        tabs = [
            {
                "id": "teleop-1",
                "url": "https://headset.local/?oobEnable=1",
                "webSocketDebuggerUrl": ws_url,
            }
        ]
        with (
            _fake_cdp_server(tabs),
            patch(
                "cloudxr_py_test_ns.oob_teleop_adb.run_adb_headset_bookmark"
            ) as mock_bookmark,
        ):
            found_url = await _find_and_click_teleop_tab(
                resolved_port=48322,
                deadline=time.monotonic() + 5,
                timeout=5,
            )
    assert found_url == ws_url
    assert "Input.dispatchMouseEvent" in [m for m, _ in script.calls]
    mock_bookmark.assert_not_called()  # tab was already there - no re-fire needed


async def test_find_and_click_teleop_tab_no_tab_found_raises() -> None:
    """No matching tab within the deadline raises OobAdbError rather than hanging."""
    with (
        _fake_cdp_server([]),
        patch(
            "cloudxr_py_test_ns.oob_teleop_adb.run_adb_headset_bookmark",
            return_value=(0, ""),
        ),
    ):
        with pytest.raises(OobAdbError, match="tab for the teleop page not found"):
            await _find_and_click_teleop_tab(
                resolved_port=48322,
                deadline=time.monotonic() + 1.5,
                timeout=1.5,
            )


async def test_run_oob_connect_calls_find_tab_with_real_signature() -> None:
    """Regression test: run_oob_connect() must call the real (not mocked)
    _find_and_click_teleop_tab() with a signature it actually accepts. A prior revision
    passed on_client_loaded to a _find_and_click_teleop_tab() that didn't declare it,
    which a mocked helper can't catch since the mock accepts any arguments."""
    script = _CdpScript(interstitial=False)
    async with _fake_cdp_ws(script) as ws_url:
        tabs = [
            {
                "id": "teleop-1",
                "url": "https://headset.local/?oobEnable=1",
                "webSocketDebuggerUrl": ws_url,
            }
        ]
        with (
            _fake_cdp_server(tabs),
            patch.object(adb_module, "_close_stale_teleop_tabs", return_value=0),
            patch.object(adb_module, "run_adb_headset_bookmark", return_value=(0, "")),
            patch.object(
                adb_module, "_discover_devtools_socket", return_value="socket"
            ),
            patch.object(adb_module, "_adb_forward_cdp"),
            patch.object(adb_module, "_adb_forward_remove"),
            patch.object(adb_module, "_monitor_teleop_error_banner"),
        ):
            monitor_task = await run_oob_connect(
                resolved_port=48322,
                timeout=5,
                on_client_loaded=lambda: None,
            )
    assert monitor_task is not None
    monitor_task.cancel()


# ============================================================================
# _monitor_teleop_error_banner: error-banner tracking, reported via on_terminal_error.
# The monitor is diagnostic-only (see its own doc comment) - it never closes, navigates,
# or relaunches the tab itself, so these tests exercise reporting and continued watching,
# not any repair decision. Recovery itself is lifecycle-owned; see test_oob_teleop_lifecycle.py.
# ============================================================================

from cloudxr_py_test_ns.oob_teleop_adb import _monitor_teleop_error_banner  # noqa: E402


class _MonitorScript:
    """Scripted responder for _monitor_teleop_error_banner's own CDP calls: certificate-error
    suppression (ignored) and the errorMessageBox poll, which returns one entry of *banners*
    per call (repeating the last entry once exhausted)."""

    def __init__(self, banners: list[str]) -> None:
        self.banners = banners
        self._index = 0

    def respond(self, method: str, params: dict) -> dict:
        if method == "Security.setIgnoreCertificateErrors":
            return {}
        expr = params.get("expression", "")
        if "errorMessageBox" in expr and "classList.contains('show')" in expr:
            i = min(self._index, len(self.banners) - 1)
            self._index += 1
            return {"result": {"value": self.banners[i]}}
        raise AssertionError(f"unscripted CDP call: {method} {params}")


async def test_monitor_reports_terminal_error_without_touching_the_tab(capsys) -> None:
    """A terminal banner is reported via on_terminal_error and logged - the monitor itself
    never calls any of the relaunch-capable helpers (closing/navigating/re-clicking is
    exclusively the caller's decision now)."""
    script = _MonitorScript(banners=["", "Stream did not attach within 500ms"])
    reported: list[str] = []
    async with _fake_cdp_ws(script) as ws_url:
        with (
            patch(
                "cloudxr_py_test_ns.oob_teleop_adb._close_stale_teleop_tabs"
            ) as close_tabs,
            patch(
                "cloudxr_py_test_ns.oob_teleop_adb.run_adb_headset_bookmark"
            ) as bookmark,
            patch(
                "cloudxr_py_test_ns.oob_teleop_adb._find_and_click_teleop_tab"
            ) as find_tab,
            patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_remove"),
        ):
            task = asyncio.create_task(
                _monitor_teleop_error_banner(
                    ws_url,
                    _CDP_LOCAL_PORT,
                    on_terminal_error=reported.append,
                )
            )
            await asyncio.sleep(2.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert reported == ["Stream did not attach within 500ms"]
    close_tabs.assert_not_called()
    bookmark.assert_not_called()
    find_tab.assert_not_called()
    out = capsys.readouterr().err
    assert "Stream did not attach within 500ms" in out


async def test_monitor_dedupes_identical_banner_then_reports_a_change() -> None:
    """The same banner text repeated reports once; a genuinely different message reports
    again - matching the existing last_banner dedup, now observed via the callback instead
    of a relaunch attempt count."""
    script = _MonitorScript(
        banners=[
            "Stream did not attach within 500ms",
            "Stream did not attach within 500ms",
            "Stream did not attach within 500ms",
            "A different terminal error",
        ]
    )
    reported: list[str] = []
    async with _fake_cdp_ws(script) as ws_url:
        with patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_remove"):
            task = asyncio.create_task(
                _monitor_teleop_error_banner(
                    ws_url,
                    _CDP_LOCAL_PORT,
                    on_terminal_error=reported.append,
                )
            )
            await asyncio.sleep(4.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert reported == [
        "Stream did not attach within 500ms",
        "A different terminal error",
    ]


async def test_monitor_keeps_watching_the_same_tab_after_reporting() -> None:
    """Reporting a terminal error doesn't end the monitor - it keeps polling the same
    ws_url for as long as it stays reachable; only a dropped connection (e.g. the caller
    closing the tab as part of its own recovery) ends it."""
    script = _MonitorScript(banners=["Stream did not attach within 500ms", "", ""])
    reported: list[str] = []
    async with _fake_cdp_ws(script) as ws_url:
        with patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_remove"):
            task = asyncio.create_task(
                _monitor_teleop_error_banner(
                    ws_url,
                    _CDP_LOCAL_PORT,
                    on_terminal_error=reported.append,
                )
            )
            await asyncio.sleep(3.5)
            assert not task.done(), f"monitor ended early: {task.exception()}"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert reported == ["Stream did not attach within 500ms"]
    assert script._index >= 3  # kept polling past the one report


async def test_monitor_without_callback_still_logs_and_keeps_watching() -> None:
    """on_terminal_error is optional - a caller that doesn't pass one still gets logging
    and continued watching, just no callback invocation."""
    script = _MonitorScript(banners=["Stream did not attach within 500ms", ""])
    async with _fake_cdp_ws(script) as ws_url:
        with patch("cloudxr_py_test_ns.oob_teleop_adb._adb_forward_remove"):
            task = asyncio.create_task(
                _monitor_teleop_error_banner(ws_url, _CDP_LOCAL_PORT)
            )
            await asyncio.sleep(2.5)
            assert not task.done(), f"monitor ended early: {task.exception()}"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


# ============================================================================
# run_oob_connect(): real-browser integration test (see
# /oob-real-browser-integration-test-plan.md at the repo root). A real
# Chromium runs the real webxr_client (real IWER-emulated WebXR, real CDP),
# served from a real static production build (never a dev-server — see
# static_webxr_build()'s docstring for why HMR is structurally unsafe for
# this), behind a real wss.py proxy (real OOBControlHub). Only adb itself is
# faked (no physical Android device), via RealBrowserAdb: its `am start` and
# `adb forward` drive the real Chromium instead of returning canned output.
#
# ALWAYS the MockCloudXR build (`build:app-mock`), never the real
# `@nvidia/cloudxr` SDK: the real SDK genuinely tries to stream against
# whatever's on `backend_port`, which is unpredictable and uncontrollable in
# a test (see the plan's §0/§3 history — this was tried and abandoned).
# MockCloudXR gives the same real App.tsx/CloudXRComponent UI (same
# #startButton/#errorMessageBox DOM) with a fully deterministic, externally
# controllable session underneath (window.__mockCloudXRFail(), see
# cdp_evaluate()), and never opens a socket of its own — no
# minimal_mock_runtime() stand-in needed for CloudXR signaling at all.
#
# This replaces the fully-synthetic mock_adb()/fake-CDP-server
# run_oob_connect() tests from PR #1148 — those proved run_oob_connect()'s
# own orchestration logic against scripted responses; this proves the same
# code path against the real client it actually launches.
#
# Slow (real browser + a real webpack production build + real wss.py) and
# requires a system Chrome (PLAYWRIGHT_CHROME_PATH override; defaults to
# /usr/bin/google-chrome) and Node/npm on PATH — run manually for now; not
# yet wired into CI.
# ============================================================================

import contextlib  # noqa: E402
from pathlib import Path  # noqa: E402

from conftest import (  # noqa: E402
    _free_tcp_port,
    cdp_evaluate,
    real_browser_adb,
    real_chrome,
    real_wss_proxy,
    static_webxr_build,
)


async def test_run_oob_connect_real_browser_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real run_oob_connect(), driven against a real Chromium running the real
    webxr client (MockCloudXR underneath — see module docstring): device online ->
    stale-tab scan -> am start opens a real tab -> real devtools socket already
    reachable (Chrome's own --remote-debugging-port equals _CDP_LOCAL_PORT, so adb
    forward is a real no-op) -> real CDP tab discovery finds the real tab -> real
    _cdp_session_click_connect polls the real client's readiness state machine
    (real IWER capability checks) and clicks the real CONNECT button.
    """
    client_port = _free_tcp_port()
    monkeypatch.setenv("TELEOP_WEB_CLIENT_BASE", f"http://localhost:{client_port}")
    monkeypatch.delenv("CONTROL_TOKEN", raising=False)

    wss_port = _free_tcp_port()
    backend_port = _free_tcp_port()
    user_data_dir = tmp_path / "chrome-profile"
    user_data_dir.mkdir()
    install_dir = tmp_path / "cxr-install"

    with (
        static_webxr_build(
            client_port,
            npm_build_script="build:app-mock",
            build_dir_name="build-app-mock",
        ),
        real_chrome(_CDP_LOCAL_PORT, user_data_dir=user_data_dir) as chrome,
        real_browser_adb(chrome) as fake_adb,
    ):
        async with real_wss_proxy(
            install_dir, proxy_port=wss_port, backend_port=backend_port
        ):
            task = await run_oob_connect(
                resolved_port=wss_port, timeout=30.0, usb_local=True
            )

        # A real task, not None: the click genuinely succeeded and
        # _monitor_teleop_error_banner is genuinely running against the real
        # client's real tab, not swallowed as "click phase ok, monitor
        # failed to spawn" (see run_oob_connect()'s own return-value contract).
        assert isinstance(task, asyncio.Task)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert any(c[:2] == ["adb", "get-state"] for c in fake_adb.calls)
    assert any("am start" in " ".join(c) for c in fake_adb.calls)


async def test_run_oob_connect_real_mock_cxr_crash_surfaces_error_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real run_oob_connect() driven against the real App.tsx/CloudXRComponent
    UI with MockCloudXR swapped in for @nvidia/cloudxr (``npm run
    build:app-mock``, ``tests/mock/cloudxr-mock-alias.ts``, static build, no
    dev-server — see :func:`static_webxr_build`): identical
    ``#startButton``/``#errorMessageBox`` DOM to production, but
    ``window.__mockCloudXRFail()`` lets this test force a deterministic
    mid-session failure instead of guessing at how a real backend failure
    would surface over the wire.

    MockCloudXR never opens a socket of its own (zero ``WebSocket`` usage in
    ``tests/mock/MockCloudXR.ts``), so nothing needs to stand in for a CloudXR
    runtime here; ``wss.py`` + ``OOBControlHub`` still run for real so the URL
    ``run_oob_connect()`` builds is a genuine ``wss://.../oob/v1/ws`` endpoint,
    matching what a real launch produces.

    Proves the crash-trigger mechanism itself — a real, real-DOM error banner
    appears on demand — ahead of reusing it to test
    ``_find_and_click_teleop_tab``'s actual recovery once this test infra is
    merged onto ``gmorgan/oob-error-relaunch`` (that branch's monitor is
    repair-capable; this branch's is not, so this test only proves the crash
    signal is real and detectable, not that anything recovers from it).
    """
    client_port = _free_tcp_port()
    monkeypatch.setenv("TELEOP_WEB_CLIENT_BASE", f"http://localhost:{client_port}")
    monkeypatch.delenv("CONTROL_TOKEN", raising=False)

    wss_port = _free_tcp_port()
    backend_port = _free_tcp_port()
    user_data_dir = tmp_path / "chrome-profile"
    user_data_dir.mkdir()
    install_dir = tmp_path / "cxr-install"

    with (
        static_webxr_build(
            client_port,
            npm_build_script="build:app-mock",
            build_dir_name="build-app-mock",
        ),
        real_chrome(_CDP_LOCAL_PORT, user_data_dir=user_data_dir) as chrome,
        real_browser_adb(chrome),
    ):
        async with real_wss_proxy(
            install_dir, proxy_port=wss_port, backend_port=backend_port
        ):
            task = await run_oob_connect(
                resolved_port=wss_port, timeout=30.0, usb_local=True
            )
            assert isinstance(task, asyncio.Task)

            teleop_tabs = [
                t for t in chrome.list_tabs() if "oobEnable=" in (t.get("url") or "")
            ]
            assert len(teleop_tabs) == 1
            ws_url = teleop_tabs[0]["webSocketDebuggerUrl"]

            # code in the 0xc0f22300-0xc0f223ff (server-disconnect) range: a
            # code in the 0xc0f22300-0xc0f223ff (server-disconnect) range: a
            # code-less failure is "recoverable" per
            # helpers/streamingErrorClassification.ts's isRecoverable(), and
            # CloudXRComponent.tsx's onStreamStopped auto-reconnects on those
            # without ever showing #errorMessageBox at all.
            #
            # Retried inside the poll loop, not fired once beforehand:
            # triggerFailure() is a no-op unless MockCloudXR's session is
            # already Connecting/Connected (see MockCloudXR.ts), and
            # run_oob_connect() returns as soon as the button text changes -
            # which happens on WebXR session entry, near-instantly, likely
            # before CloudXR.createSession()'s own async chain
            # (applyTargetFrameRate/requestReferenceSpace) has finished. A
            # call before the session exists silently does nothing, so retry
            # until the banner actually appears rather than firing once and
            # hoping the timing lines up. Idempotent once it does land: a
            # session already in Error state fails the same Connecting/
            # Connected guard, so later retries no-op harmlessly.
            # 5s is ample margin above the ~1s this actually takes once the
            # session exists (observed) - MockCloudXR's own transitions are
            # near-instant, unlike the relaunch test below which waits on
            # real am start + tab discovery.
            banner = ""
            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline:
                await cdp_evaluate(
                    ws_url,
                    "window.__mockCloudXRFail('simulated crash', 0xc0f22300)",
                )
                r = await cdp_evaluate(
                    ws_url,
                    """(function() {
                        const box = document.getElementById('errorMessageBox');
                        if (!box || !box.classList.contains('show')) return '';
                        if (box.classList.contains('success') ||
                            box.classList.contains('info')) return '';
                        return document.getElementById('errorMessageText')
                                   ?.textContent?.trim() || '';
                    })()""",
                )
                banner = (r.get("result") or {}).get("value") or ""
                if banner:
                    break
                await asyncio.sleep(0.5)

            assert "simulated crash" in banner

            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def test_run_oob_connect_real_mock_cxr_crash_triggers_real_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real client-side terminal error should trigger a real auto-relaunch:
    a second real ``am start``, a second real tab, clicked back to CONNECT-ready.

    Anticipatory, like ``test_build_teleop_url_forwards_reliability_config_from_env``
    above: gmorgan/oob-error-relaunch (#1146) adds this behavior
    (``_find_and_click_teleop_tab``, a repair-capable ``_monitor_teleop_error_banner``)
    and hasn't merged here yet, so this currently fails - not on an ImportError of a
    private function (deliberately not imported here at all, so this stays correct
    across internal refactors), but on the black-box assertions below: today's
    monitor only logs the banner this test triggers, it never relaunches anything, so
    no second tab/``am start`` ever appears and the wait below times out. Exercising
    exactly what #1146 promises - a genuine crash recovered by genuine ADB+CDP
    automation, not a scripted stand-in for either - is exactly what the real-browser
    infrastructure in this module exists to prove, unlike
    ``test_build_teleop_url_forwards_reliability_config_from_env``'s plain
    env-var/string check.
    """
    client_port = _free_tcp_port()
    monkeypatch.setenv("TELEOP_WEB_CLIENT_BASE", f"http://localhost:{client_port}")
    monkeypatch.delenv("CONTROL_TOKEN", raising=False)

    wss_port = _free_tcp_port()
    backend_port = _free_tcp_port()
    user_data_dir = tmp_path / "chrome-profile"
    user_data_dir.mkdir()
    install_dir = tmp_path / "cxr-install"

    with (
        static_webxr_build(
            client_port,
            npm_build_script="build:app-mock",
            build_dir_name="build-app-mock",
        ),
        real_chrome(_CDP_LOCAL_PORT, user_data_dir=user_data_dir) as chrome,
        real_browser_adb(chrome) as fake_adb,
    ):
        async with real_wss_proxy(
            install_dir, proxy_port=wss_port, backend_port=backend_port
        ):
            task = await run_oob_connect(
                resolved_port=wss_port, timeout=30.0, usb_local=True
            )
            assert isinstance(task, asyncio.Task)

            teleop_tabs_before = [
                t for t in chrome.list_tabs() if "oobEnable=" in (t.get("url") or "")
            ]
            assert len(teleop_tabs_before) == 1
            ws_url = teleop_tabs_before[0]["webSocketDebuggerUrl"]
            am_start_calls_before = sum(
                1 for c in fake_adb.calls if "am start" in " ".join(c)
            )

            # Same retry-until-it-lands approach as the crash-banner test above:
            # triggerFailure() no-ops until MockCloudXR's session actually exists.
            # 15s: once #1146 lands, a relaunch has no cold-launch devtools-socket
            # wait (browser's already running) - just am start + tab discovery +
            # click, which the initial launch above already does in ~5s.
            relaunched = False
            deadline = asyncio.get_running_loop().time() + 15.0
            while asyncio.get_running_loop().time() < deadline:
                await cdp_evaluate(
                    ws_url,
                    "window.__mockCloudXRFail('simulated crash', 0xc0f22300)",
                )
                am_start_calls_now = sum(
                    1 for c in fake_adb.calls if "am start" in " ".join(c)
                )
                teleop_tabs_now = [
                    t
                    for t in chrome.list_tabs()
                    if "oobEnable=" in (t.get("url") or "")
                ]
                if (
                    am_start_calls_now > am_start_calls_before
                    and len(teleop_tabs_now) >= 2
                ):
                    relaunched = True
                    break
                await asyncio.sleep(0.5)

            assert relaunched, (
                "expected a second real `am start` and a second real tab after "
                "a terminal client error - today's (pre-#1146) monitor only logs "
                "the banner, it never relaunches anything"
            )

            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
