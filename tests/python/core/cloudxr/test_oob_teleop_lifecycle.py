# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State and timing tests for OOB recovery without headset hardware."""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import patch

import pytest

from isaacteleop.cloudxr import oob_teleop_adb as adb
from isaacteleop.cloudxr.oob_teleop_adb import (
    AdbDevices,
    AdbReverseProbe,
    HeadsetNetworkProbe,
    HeadsetNetworkState,
)
from isaacteleop.cloudxr.oob_teleop_env import resolve_oob_recovery_config
from isaacteleop.cloudxr.oob_teleop_lifecycle import (
    DeviceReplacedError,
    OobLifecycle,
    RecoveryConfig,
)

ORIGINAL_ADB_RUN = adb._adb_run


class FakeHub:
    def __init__(self):
        self.statuses = []

    async def set_lifecycle_snapshot(self, status):
        self.statuses.append(dict(status))

    async def get_snapshot(self):
        return {"headsets": []}


@pytest.fixture(autouse=True)
def immediate_to_thread(monkeypatch):
    async def immediate(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", immediate)
    monkeypatch.setattr(
        adb,
        "_adb_run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )


@pytest.mark.parametrize(
    "name", ["TELEOP_OOB_RECOVERY_TIMEOUT_SEC", "TELEOP_OOB_RETRY_INTERVAL_SEC"]
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "bad"])
def test_recovery_config_rejects_nonpositive_or_nonfinite(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        resolve_oob_recovery_config()


def test_recovery_config_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("TELEOP_OOB_RECOVERY_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("TELEOP_OOB_RETRY_INTERVAL_SEC", raising=False)
    assert resolve_oob_recovery_config() == RecoveryConfig(60, 5)
    monkeypatch.setenv("TELEOP_OOB_RECOVERY_TIMEOUT_SEC", "30.5")
    monkeypatch.setenv("TELEOP_OOB_RETRY_INTERVAL_SEC", "2")
    assert resolve_oob_recovery_config() == RecoveryConfig(30.5, 2)


async def test_absent_headset_keeps_observing_after_episode_expires(monkeypatch):
    hub = FakeHub()
    now = [0.0]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        if len(sleeps) == 5:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(10, 5),
        clock=lambda: now[0],
        sleep=sleep,
    )
    with patch(
        "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
        return_value=AdbDevices(()),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert sleeps == [5] * 5
    assert len(hub.statuses) >= 5
    assert all(status["health"] != "fatal" for status in hub.statuses)


async def test_selected_serial_replacement_is_terminal(monkeypatch):
    hub = FakeHub()
    statuses = []
    fatals = []
    devices = iter(
        [
            AdbDevices((("original", "device"),)),
            AdbDevices((("replacement", "device"),)),
        ]
    )

    async def no_wait(_):
        return None

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        sleep=no_wait,
        on_status=statuses.append,
        on_fatal=fatals.append,
    )
    with patch(
        "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
        side_effect=lambda: next(devices),
    ):
        with pytest.raises(DeviceReplacedError, match="DEVICE_REPLACED"):
            await lifecycle.run()
    assert len(fatals) == 1
    assert statuses[-1]["health"] == "fatal"
    assert statuses[-1]["expectedSerial"] == "original"
    assert statuses[-1]["replacementSerial"] == "replacement"


async def test_explicit_serial_replacement_is_terminal_after_selection(monkeypatch):
    monkeypatch.setenv("ANDROID_SERIAL", "original")
    hub = FakeHub()
    devices = iter(
        [
            AdbDevices((("original", "device"),)),
            AdbDevices((("replacement", "device"),)),
        ]
    )

    async def no_wait(_):
        return None

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        sleep=no_wait,
    )
    with patch(
        "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
        side_effect=lambda: next(devices),
    ):
        with pytest.raises(DeviceReplacedError):
            await lifecycle.run()
    assert lifecycle.selected == "original"
    assert hub.statuses[-1]["health"] == "fatal"


async def test_extra_ready_device_does_not_change_pinned_target():
    hub = FakeHub()
    ready = AdbDevices((("original", "device"), ("extra", "device")))

    async def stop(_):
        adb._adb_run(["adb", "get-state"])
        raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        sleep=stop,
    )
    lifecycle.selected = "original"
    lifecycle._selected_once = True
    with (
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_adb.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "device", ""),
        ) as command,
        patch.object(adb, "_adb_run", side_effect=ORIGINAL_ADB_RUN),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert lifecycle.selected == "original"
    assert command.call_args.args[0][:3] == ["adb", "-s", "original"]
    assert not any(status["health"] == "fatal" for status in hub.statuses)


async def test_same_serial_replug_stays_pinned(monkeypatch):
    hub = FakeHub()
    now = [0.0]
    calls = [
        AdbDevices((("original", "device"),)),
        AdbDevices(()),
        AdbDevices((("original", "device"),)),
    ]
    observations = []

    def devices():
        result = calls.pop(0) if calls else AdbDevices((("original", "device"),))
        observations.append(result)
        return result

    async def sleep(seconds):
        now[0] += seconds
        if len(observations) >= 5:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        clock=lambda: now[0],
        sleep=sleep,
    )

    async def automate():
        lifecycle.browser_ready = True

    with (
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            side_effect=devices,
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch.object(lifecycle, "_automate", new=automate),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert lifecycle.selected == "original"
    assert not any(s["health"] == "fatal" for s in hub.statuses)


async def test_late_coturn_fault_opens_new_episode():
    hub = FakeHub()
    now = [100.0]

    async def stop_after_fault(_):
        raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(60, 5),
        clock=lambda: now[0],
        sleep=stop_after_fault,
    )
    lifecycle.selected = "original"
    lifecycle._selected_once = True
    lifecycle._ready_count = 2
    lifecycle._last_health = "active"
    lifecycle.snapshot = {"health": "active"}
    lifecycle.browser_ready = True
    lifecycle.episode_start = 0.0
    ready = AdbDevices((("original", "device"),))
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    output = "\n".join(f"original tcp:{p} tcp:{p}" for p in (8080, 48322, 49100, 3478))

    async def restarted():
        return True

    with (
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb._run_adb", return_value=output
        ),
        patch.object(lifecycle, "_ensure_coturn", new=restarted),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert lifecycle.episode_start == 100.0
    assert hub.statuses[-1]["state"] == "DEGRADED"
    assert "coturn restarted" in hub.statuses[-1]["reason"]


async def test_wifi_restoration_reopens_expired_episode():
    hub = FakeHub()
    now = [100.0]

    async def stop_after_attempt(_):
        raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(60, 5),
        clock=lambda: now[0],
        sleep=stop_after_attempt,
    )
    lifecycle.selected = "original"
    lifecycle._selected_once = True
    lifecycle._ready_count = 2
    lifecycle.episode_start = 0.0
    lifecycle.last_network_state = HeadsetNetworkState.NO_NETWORK
    ready = AdbDevices((("original", "device"),))
    lifecycle._last_observation = (ready.devices, ready.diagnostic)

    async def noop():
        pass

    with (
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch.object(lifecycle, "_prepare_device", new=noop),
        patch.object(lifecycle, "_automate", new=noop),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert lifecycle.episode_start == 100.0
    assert lifecycle.attempts == 1


async def test_usb_rebuild_verifies_all_four_rules_and_rolls_back_partial_failure():
    hub = FakeHub()
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        host_listener_probe=lambda _: True,
    )
    lifecycle.selected = "original"
    calls = []
    ports = (8080, 48322, 49100, 3478)
    listing = "\n".join(f"original tcp:{port} tcp:{port}" for port in ports)

    def run(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["reverse", "--list"]:
            return subprocess.CompletedProcess(args, 0, listing, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    async def coturn():
        return True

    with (
        patch("isaacteleop.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value=listing,
        ),
        patch.object(lifecycle, "_ensure_coturn", new=coturn),
    ):
        await lifecycle._rebuild_usb()
    assert [
        args[2] for args in calls if args[1] == "reverse" and args[2] != "--remove"
    ] == [f"tcp:{p}" for p in ports]
    assert hub.statuses[-1]["reverseRulesVerified"] is True

    calls.clear()

    def fail_third(args, **kwargs):
        calls.append(args)
        if args[1] == "reverse" and args[2] == "tcp:49100":
            return subprocess.CompletedProcess(args, 1, "", "device offline")
        return subprocess.CompletedProcess(args, 0, "", "")

    with (
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb._adb_run",
            side_effect=fail_third,
        ),
        patch.object(lifecycle, "_ensure_coturn", new=coturn),
    ):
        with pytest.raises(Exception, match="49100"):
            await lifecycle._rebuild_usb()
    assert len([args for args in calls if "--remove" in args]) == 10


async def test_cable_loss_and_same_serial_replug_rebuilds_every_rule_and_reconnects():
    hub = FakeHub()
    hub.get_snapshot = lambda: asyncio.sleep(
        0,
        result={
            "headsets": [
                {"clientId": "page", "streaming": False, "lastMetricsAt": None}
            ]
        },
    )
    observations = iter(
        [
            AdbDevices((("original", "device"),)),
            AdbDevices((("original", "device"),)),
            AdbDevices(()),
            AdbDevices((("original", "device"),)),
            AdbDevices((("original", "device"),)),
        ]
    )
    calls = []
    connects = []
    ticks = [0]

    async def sleep(_):
        ticks[0] += 1
        if ticks[0] == 5:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        sleep=sleep,
        host_listener_probe=lambda _: True,
    )

    async def noop():
        return False

    async def automate():
        connects.append(lifecycle.selected)
        lifecycle.browser_ready = True
        lifecycle.browser_client = "page"

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    with (
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            side_effect=lambda: next(observations),
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value="rules",
        ),
        patch("isaacteleop.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.probe_adb_reverse_rules",
            return_value=AdbReverseProbe(True, ()),
        ),
        patch.object(lifecycle, "_prepare_device", new=noop),
        patch.object(lifecycle, "_ensure_coturn", new=noop),
        patch.object(lifecycle, "_automate", new=automate),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    installs = [
        args[2]
        for args in calls
        if len(args) > 3 and args[1] == "reverse" and args[2].startswith("tcp:")
    ]
    assert installs == [f"tcp:{p}" for p in (8080, 48322, 49100, 3478)] * 2
    assert connects == ["original", "original"]
    assert calls[-5:] == [
        ["adb", "forward", "--remove", "tcp:9223"],
        *[
            ["adb", "reverse", "--remove", f"tcp:{p}"]
            for p in (8080, 48322, 49100, 3478)
        ],
    ]
    assert not any(status["health"] == "fatal" for status in hub.statuses)


async def test_prepare_wakes_each_attempt_but_clears_cache_once():
    lifecycle = OobLifecycle(
        hub=FakeHub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
    )
    with (
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"
        ) as awake,
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.clear_headset_browser_cache"
        ) as cache,
    ):
        await lifecycle._prepare_device()
        await lifecycle._prepare_device()
    assert awake.call_count == 2
    cache.assert_called_once()


async def test_host_listener_failure_rolls_back_without_starting_turn():
    calls = []
    lifecycle = OobLifecycle(
        hub=FakeHub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        host_listener_probe=lambda port: port != 49100,
    )
    lifecycle.selected = "original"

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    with (
        patch("isaacteleop.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run),
        patch.object(lifecycle, "_ensure_coturn") as coturn,
    ):
        with pytest.raises(Exception, match="Host listener on tcp:49100"):
            await lifecycle._rebuild_usb()
    coturn.assert_not_called()
    assert len([args for args in calls if "--remove" in args]) == 10


async def test_cancellation_mid_rebuild_rolls_back_owned_rules():
    calls = []
    lifecycle = OobLifecycle(
        hub=FakeHub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        host_listener_probe=lambda _: True,
    )
    lifecycle.selected = "original"

    def run(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["reverse", "tcp:49100"]:
            raise asyncio.CancelledError
        return subprocess.CompletedProcess(args, 0, "", "")

    async def coturn():
        return True

    with (
        patch("isaacteleop.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run),
        patch.object(lifecycle, "_ensure_coturn", new=coturn),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle._rebuild_usb()
    assert calls[-5:] == [
        ["adb", "forward", "--remove", "tcp:9223"],
        *[
            ["adb", "reverse", "--remove", f"tcp:{p}"]
            for p in (8080, 48322, 49100, 3478)
        ],
    ]


async def test_episode_timeout_bounds_preparation_and_cleans_owned_forward():
    hub = FakeHub()
    ready = AdbDevices((("original", "device"),))
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(0.1, 5),
        sleep=lambda _: asyncio.sleep(0),
    )
    lifecycle.selected = "original"
    lifecycle._selected_once = True
    lifecycle._ready_count = 2
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    calls = []

    async def slow_prepare():
        await asyncio.sleep(1)

    async def stop_after_timeout(_):
        raise asyncio.CancelledError

    lifecycle.sleep = stop_after_timeout

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    with (
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch("isaacteleop.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run),
        patch.object(lifecycle, "_prepare_device", new=slow_prepare),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert any(
        "Recovery attempt timed out" in status["reason"] for status in hub.statuses
    )
    assert ["adb", "forward", "--remove", "tcp:9223"] in calls
