# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State and timing tests for OOB recovery without headset hardware."""

from __future__ import annotations

import asyncio
import subprocess
import time
from unittest.mock import patch

import pytest

from isaaccapture.cloudxr import oob_teleop_adb as adb
from isaaccapture.cloudxr.oob_teleop_adb import (
    AdbDevices,
    AdbReverseProbe,
    HeadsetNetworkProbe,
    HeadsetNetworkState,
)
from isaaccapture.cloudxr.oob_teleop_env import resolve_oob_recovery_config
from isaaccapture.cloudxr.oob_teleop_lifecycle import (
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
        "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
        return_value=AdbDevices(()),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert sleeps == [5] * 5
    assert len(hub.statuses) >= 5
    assert all(status["health"] != "fatal" for status in hub.statuses)


async def test_replug_repair_observes_after_deadline_until_topology_changes():
    hub = FakeHub()
    now = [0.0]
    rebuild_times = []

    async def sleep(seconds):
        now[0] += seconds
        if now[0] >= 5.0:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(timeout_sec=2, interval_sec=1),
        clock=lambda: now[0],
        sleep=sleep,
    )
    lifecycle.selected = "original"
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True

    async def fail_rebuild():
        rebuild_times.append(now[0])
        raise adb.OobAdbError("stable repair failure")

    def reverse_listing(*_args, **_kwargs):
        return "topology-changed" if now[0] >= 4.0 else "stable-topology"

    ready = AdbDevices((("original", "device"),))
    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch("isaaccapture.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            side_effect=reverse_listing,
        ),
        patch.object(lifecycle, "_rebuild_usb", new=fail_rebuild),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert rebuild_times == [0.0, 1.0, 4.0]
    expired = [
        status
        for status in hub.statuses
        if status["reason"] == "Recovery episode expired; observing for a change"
    ]
    assert expired
    assert all(status["state"] == "VERIFYING_EXISTING_BROWSER" for status in expired)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("extra_count", [1, 2])
@pytest.mark.parametrize("selected_state", [None, "offline", "unauthorized"])
@pytest.mark.parametrize("return_with_extras", [False, True])
async def test_unrelated_devices_wait_and_original_recovers(
    monkeypatch, caplog, explicit, extra_count, selected_state, return_with_extras
):
    if explicit:
        monkeypatch.setenv("ANDROID_SERIAL", "original")
    else:
        monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    hub = FakeHub()
    fatals = []
    extras = tuple((f"extra-{i}", "device") for i in range(extra_count))
    absent = (
        extras if selected_state is None else (("original", selected_state), *extras)
    )
    returned = (
        (("original", "device"), *extras)
        if return_with_extras
        else (("original", "device"),)
    )
    observations = [
        AdbDevices((("original", "device"),)),
        AdbDevices(absent),
        AdbDevices(absent, "poll diagnostic changed"),
        AdbDevices(returned),
        AdbDevices(returned),
    ]
    commands = []
    recovered = []

    def enumerate_devices():
        return observations.pop(0) if observations else AdbDevices(returned)

    def command(args, **kwargs):
        commands.append(tuple(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    async def sleep(_):
        if len(commands) > 20:
            raise AssertionError("Unexpected command loop")
        if not observations:
            raise asyncio.CancelledError

    async def prepare():
        recovered.append(adb.SELECTED_ADB_SERIAL.get())
        await asyncio.to_thread(adb._adb_run, ["adb", "shell", "true"])

    async def automate():
        lifecycle.browser_ready = True

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        sleep=sleep,
        on_fatal=fatals.append,
    )
    with (
        patch.object(adb, "enumerate_adb_devices", side_effect=enumerate_devices),
        patch.object(
            adb,
            "probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch.object(adb, "_adb_run", side_effect=ORIGINAL_ADB_RUN),
        patch(
            "isaaccapture.cloudxr.oob_teleop_adb.subprocess.run", side_effect=command
        ),
        patch.object(lifecycle, "_prepare_device", new=prepare),
        patch.object(lifecycle, "_automate", new=automate),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert not fatals
    assert not any(status["health"] == "fatal" for status in hub.statuses)
    assert lifecycle.selected == "original"
    assert any(
        status["state"] == "WAITING_FOR_ADB"
        and status["ignoredSerials"] == [x[0] for x in extras]
        for status in hub.statuses
    )
    assert all(status["explicitSerial"] is explicit for status in hub.statuses)
    if selected_state:
        assert any(selected_state in status["reason"] for status in hub.statuses)
    assert recovered == ["original"]
    assert commands
    assert all(command[:3] == ("adb", "-s", "original") for command in commands)
    assert sum(
        "Ignored ADB device(s)" in record.message for record in caplog.records
    ) == (2 if return_with_extras else 1)


async def test_explicit_serial_waits_without_ever_adopting_other(monkeypatch):
    monkeypatch.setenv("ANDROID_SERIAL", "original")
    hub = FakeHub()
    observations = [AdbDevices((("other", "device"),))] * 3

    async def sleep(_):
        if not observations:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        sleep=sleep,
    )
    with patch.object(
        adb, "enumerate_adb_devices", side_effect=lambda: observations.pop(0)
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert lifecycle.selected == "original"
    assert all(status["selectedSerial"] == "original" for status in hub.statuses)
    assert hub.statuses[-1]["ignoredSerials"] == ["other"]


async def test_implicit_selection_waits_for_exactly_one_ready(monkeypatch):
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    hub = FakeHub()
    observations = [AdbDevices((("one", "device"), ("two", "device")))] * 2

    async def sleep(_):
        if not observations:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        sleep=sleep,
    )
    with patch.object(
        adb, "enumerate_adb_devices", side_effect=lambda: observations.pop(0)
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert lifecycle.selected is None
    assert hub.statuses[-1]["ignoredSerials"] == []
    assert "Multiple ready devices" in hub.statuses[-1]["reason"]


async def test_adb_enumeration_reorder_does_not_restart_recovery(monkeypatch, caplog):
    monkeypatch.setenv("ANDROID_SERIAL", "original")
    hub = FakeHub()
    observations = [
        AdbDevices((("original", "device"), ("z", "device"), ("a", "device"))),
        AdbDevices((("a", "device"), ("original", "device"), ("z", "device"))),
    ]
    prepared = []

    async def sleep(_):
        if not observations:
            raise asyncio.CancelledError

    async def prepare():
        prepared.append(lifecycle.selected)

    async def automate():
        lifecycle.browser_ready = True

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        sleep=sleep,
    )
    with (
        patch.object(
            adb, "enumerate_adb_devices", side_effect=lambda: observations.pop(0)
        ),
        patch.object(
            adb,
            "probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch.object(lifecycle, "_prepare_device", new=prepare),
        patch.object(lifecycle, "_automate", new=automate),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert prepared == ["original"]
    assert hub.statuses[-1]["ignoredSerials"] == ["a", "z"]
    assert (
        sum("Ignored ADB device(s)" in record.message for record in caplog.records) == 1
    )


async def test_ignored_serial_display_is_bounded_and_safe(monkeypatch):
    monkeypatch.setenv("ANDROID_SERIAL", "original\n" + "X" * 100)
    hub = FakeHub()
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
    )
    lifecycle.ignored_serials = tuple(f"extra\n{i}" + "Y" * 100 for i in range(12))
    await lifecycle._publish("degraded", "WAITING_FOR_ADB", "waiting")
    status = hub.statuses[-1]
    assert status["schemaVersion"] == 1
    assert len(status["selectedSerial"]) == 80
    assert len(status["ignoredSerials"]) == 8
    assert all(
        "\n" not in serial and len(serial) <= 80 for serial in status["ignoredSerials"]
    )
    assert "\n" not in status["selectedSerial"]


async def test_cleanup_commands_stay_on_selected_serial_when_it_is_absent(monkeypatch):
    monkeypatch.setenv("ANDROID_SERIAL", "original")
    hub = FakeHub()
    commands = []
    observations = [AdbDevices((("other", "device"),))]

    def command(args, **kwargs):
        commands.append(tuple(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    async def sleep(_):
        raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=False,
        turn_port=3478,
        config=RecoveryConfig(),
        sleep=sleep,
    )
    with (
        patch.object(
            adb, "enumerate_adb_devices", side_effect=lambda: observations.pop(0)
        ),
        patch.object(adb, "_adb_run", side_effect=ORIGINAL_ADB_RUN),
        patch(
            "isaaccapture.cloudxr.oob_teleop_adb.subprocess.run", side_effect=command
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert any("reverse" in command for command in commands)
    assert all(command[:3] == ("adb", "-s", "original") for command in commands)


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
    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_adb.subprocess.run",
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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            side_effect=devices,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
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

    rebuilds = []

    async def no_sleep(_):
        pytest.fail("transport-loss transition must not take the normal retry sleep")

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(60, 5),
        clock=lambda: now[0],
        sleep=no_sleep,
    )
    lifecycle.selected = "original"
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

    async def stop_at_rebuild():
        rebuilds.append(True)
        raise asyncio.CancelledError

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value=output,
        ),
        patch.object(lifecycle, "_ensure_coturn", new=restarted),
        patch.object(lifecycle, "_rebuild_usb", new=stop_at_rebuild),
        patch.object(lifecycle, "_automate") as automate,
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert lifecycle.episode_start == 100.0
    assert lifecycle._transport_lost is True
    assert lifecycle._restore_existing_browser is True
    assert rebuilds == [True]
    automate.assert_not_called()
    transition = next(
        status for status in hub.statuses if "coturn restarted" in status["reason"]
    )
    assert transition["adbReady"] is True
    assert transition["networkPresent"] is True


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
    lifecycle._ready_count = 2
    lifecycle.episode_start = 0.0
    lifecycle.last_network_state = HeadsetNetworkState.NO_NETWORK
    ready = AdbDevices((("original", "device"),))
    lifecycle._last_observation = (ready.devices, ready.diagnostic)

    async def noop():
        pass

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch.object(lifecycle, "_prepare_device", new=noop),
        patch.object(lifecycle, "_automate", new=noop),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert lifecycle.episode_start == 100.0
    assert lifecycle.attempts == 1


async def test_usb_network_loss_preserves_browser_and_uses_fast_retry():
    hub = FakeHub()
    ready = AdbDevices((("original", "device"),))
    sleeps = []

    async def stop_after_transition(seconds):
        sleeps.append(seconds)
        raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(60, 5),
        sleep=stop_after_transition,
    )
    lifecycle.selected = "original"
    lifecycle._ready_count = 2
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle.browser_ready = True
    lifecycle.browser_client = "surviving-page"

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NO_NETWORK),
        ),
        patch.object(lifecycle, "_automate") as automate,
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert sleeps == [1.0]
    assert lifecycle._transport_lost is True
    assert lifecycle._restore_existing_browser is True
    automate.assert_not_called()
    transition = hub.statuses[-1]
    assert transition["adbReady"] is True
    assert transition["networkPresent"] is False


async def test_returning_transport_skips_second_ready_debounce():
    hub = FakeHub()
    ready = AdbDevices((("original", "device"),))
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        sleep=lambda _seconds: pytest.fail(
            "returning transport must repair immediately"
        ),
    )
    lifecycle.selected = "original"
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    rebuilds = []

    async def stop_at_rebuild():
        rebuilds.append(True)
        raise asyncio.CancelledError

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value="",
        ),
        patch("isaaccapture.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"),
        patch.object(lifecycle, "_rebuild_usb", new=stop_at_rebuild),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert lifecycle._ready_count == 1
    assert rebuilds == [True]


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
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._adb_run",
            side_effect=fail_third,
        ),
        patch.object(lifecycle, "_ensure_coturn", new=coturn),
    ):
        with pytest.raises(Exception, match="49100"):
            await lifecycle._rebuild_usb()
    assert len([args for args in calls if "--remove" in args]) == 10


async def test_cable_loss_and_same_serial_replug_rebuilds_rules_without_relaunch():
    hub = FakeHub()
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(0, result=None)
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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            side_effect=lambda: next(observations),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value="rules",
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_adb_reverse_rules",
            return_value=AdbReverseProbe(True, ()),
        ),
        patch.object(lifecycle, "_prepare_device", new=noop),
        patch("isaaccapture.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"),
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
    assert connects == ["original"]
    assert calls[-5:] == [
        ["adb", "forward", "--remove", "tcp:9223"],
        *[
            ["adb", "reverse", "--remove", f"tcp:{p}"]
            for p in (8080, 48322, 49100, 3478)
        ],
    ]
    assert not any(status["health"] == "fatal" for status in hub.statuses)


@pytest.mark.parametrize("late_health_report", [False, True])
async def test_replug_connect_is_not_repeated_while_waiting_for_browser_or_stream(
    late_health_report,
):
    """A successful CDP click is not a retry trigger when OOB proof is slow."""
    hub = FakeHub()
    probe_calls = []

    async def snapshot():
        return {
            "headsets": [
                {
                    "clientId": client,
                    # The replug page models a baseline client: it reports
                    # streamStatus but never understands healthProbe.
                    "streaming": client == "replug-page" and not late_health_report,
                    "lastMetricsAt": None,
                }
                for client in ("initial-page", "replug-page")
            ]
        }

    async def probe(generation, after, **_kwargs):
        probe_calls.append(generation)
        if len(probe_calls) == 1:
            return {"clientId": "initial-page"}
        if late_health_report and len(probe_calls) >= 4:
            return {"clientId": "replug-page"}
        return None

    hub.get_snapshot = snapshot
    hub.probe_browser = probe
    ready = AdbDevices((("original", "device"),))
    observations = iter(
        [ready, ready, ready, AdbDevices(()), ready, ready, ready, ready, ready]
    )
    clicks = []
    ticks = [0]

    async def sleep(_):
        ticks[0] += 1
        if ticks[0] == 9:
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

    async def connect(**_kwargs):
        clicks.append(lifecycle.selected)
        return asyncio.create_task(asyncio.Event().wait())

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            side_effect=lambda: next(observations),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_adb_reverse_rules",
            return_value=AdbReverseProbe(True, ()),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.run_oob_connect",
            side_effect=connect,
        ),
        patch.object(lifecycle, "_prepare_device", new=noop),
        patch("isaaccapture.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"),
        patch.object(lifecycle, "_ensure_coturn", new=noop),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert clicks == ["original"]
    assert lifecycle.generation == 1
    assert len(probe_calls) >= 3
    assert hub.statuses[-1]["turnEndToEndHealthy"] is False
    if late_health_report:
        assert hub.statuses[-1]["health"] == "browser_ready"
    else:
        assert hub.statuses[-1]["state"] == "VERIFYING_EXISTING_BROWSER"
        assert hub.statuses[-1]["connectDispatched"] is False
        assert hub.statuses[-1]["health"] == "degraded"


async def test_registered_browser_disconnect_triggers_new_connect():
    hub = FakeHub()
    hub.probe_browser = lambda *_args: asyncio.sleep(0, result={"clientId": "new-page"})
    ready = AdbDevices((("original", "device"),))
    ticks = [0]
    clicks = []

    async def sleep(_):
        ticks[0] += 1
        if ticks[0] == 2:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(),
        sleep=sleep,
    )
    lifecycle.selected = "original"
    lifecycle._ready_count = 2
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle.browser_ready = True
    lifecycle.browser_client = "old-page"

    async def connect(**_kwargs):
        clicks.append(lifecycle.selected)
        return asyncio.create_task(asyncio.Event().wait())

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.run_oob_connect",
            side_effect=connect,
        ),
        patch.object(lifecycle, "_prepare_device", new=lambda: asyncio.sleep(0)),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert clicks == ["original"]
    assert lifecycle.browser_client == "new-page"
    assert hub.statuses[-1]["health"] == "browser_ready"


async def test_usb_control_disconnect_enters_preservation_before_automation():
    hub = FakeHub()
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
    )
    lifecycle.browser_ready = True
    lifecycle.browser_client = "lost-control-client"

    with patch.object(lifecycle, "_automate") as automate:
        with pytest.raises(adb.OobAdbError, match="control client disconnected") as exc:
            await lifecycle._observe_stream()
        await lifecycle._enter_transport_recovery(
            str(exc.value),
            "REBUILDING_USB",
            adb_ready=exc.value.adb_ready,
            network_present=exc.value.network_present,
        )

    assert lifecycle._transport_lost is True
    assert lifecycle._restore_existing_browser is True
    assert lifecycle.browser_ready is False
    automate.assert_not_called()
    assert hub.statuses[-1]["adbReady"] is True
    assert hub.statuses[-1]["networkPresent"] is True


async def test_usb_retrying_phase_repairs_transport_before_browser_fallback():
    hub = FakeHub()
    hub.get_snapshot = lambda: asyncio.sleep(
        0,
        result={
            "headsets": [
                {
                    "clientId": "surviving-page",
                    "streaming": False,
                    "lastMetricsAt": None,
                    "streamPhase": "retrying",
                    "terminalEventId": None,
                }
            ]
        },
    )
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
    )
    lifecycle.browser_ready = True
    lifecycle.browser_client = "surviving-page"

    with (
        patch.object(lifecycle, "_attach_existing_monitor", return_value=True),
        patch.object(lifecycle, "_same_tab_connect") as same_tab,
        patch.object(lifecycle, "_automate") as automate,
    ):
        with pytest.raises(adb.OobAdbError, match="retrying") as exc:
            await lifecycle._observe_stream()
        await lifecycle._enter_transport_recovery(
            str(exc.value),
            "REBUILDING_USB",
            adb_ready=exc.value.adb_ready,
            network_present=exc.value.network_present,
        )

    assert lifecycle._transport_lost is True
    assert lifecycle._restore_existing_browser is True
    same_tab.assert_not_called()
    automate.assert_not_called()


async def test_missing_reverse_rule_during_probe_wait_triggers_rebuild():
    hub = FakeHub()
    ready = AdbDevices((("original", "device"),))
    ticks = [0]
    rebuilds = []

    async def sleep(_):
        ticks[0] += 1
        if ticks[0] == 1:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        sleep=sleep,
    )
    lifecycle.selected = "original"
    lifecycle._ready_count = 2
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle.connect_dispatched = True
    lifecycle.browser_probe_after = 0.0

    async def noop():
        return False

    async def rebuild():
        rebuilds.append(True)

    async def automate():
        pytest.fail("missing reverse rules must preserve the existing browser")

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_adb_reverse_rules",
            return_value=AdbReverseProbe(True, (3478,)),
        ),
        patch("isaaccapture.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"),
        patch.object(lifecycle, "_ensure_coturn", new=noop),
        patch.object(lifecycle, "_prepare_device", new=noop),
        patch.object(lifecycle, "_rebuild_usb", new=rebuild),
        patch.object(lifecycle, "_automate", new=automate),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert rebuilds == [True]
    transition = next(
        status
        for status in hub.statuses
        if "USB reverse rules missing" in status["reason"]
    )
    assert transition["adbReady"] is True
    assert transition["networkPresent"] is True


async def test_quick_cable_flap_repairs_transport_without_browser_automation():
    hub = FakeHub()
    ready = AdbDevices((("original", "device"),))
    events = []
    now = time.time()
    hub.get_snapshot = lambda: asyncio.sleep(
        0,
        result={
            "headsets": [
                {
                    "clientId": "surviving-page",
                    "streaming": True,
                    "lastMetricsAt": (now + 1) * 1000,
                    "streamPhase": "streaming",
                    "terminalEventId": None,
                }
            ]
        },
    )
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(
        0,
        result={
            "clientId": "surviving-page",
            "streaming": True,
            "lastMetricsAt": (now + 1) * 1000,
            "streamPhase": "streaming",
            "terminalEventId": None,
        },
    )

    async def stop_after_recovery(_):
        events.append("sleep")
        raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        sleep=stop_after_recovery,
        host_listener_probe=lambda _port: True,
    )
    lifecycle.selected = "original"
    lifecycle._ready_count = 2
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle._last_health = "active"
    lifecycle.browser_ready = True
    lifecycle.browser_client = "surviving-page"
    lifecycle.last_stream_at = now
    required = (8080, 48322, 49100, 3478)
    complete = "\n".join(f"original tcp:{port} tcp:{port}" for port in required)
    probes = [AdbReverseProbe(True, required), AdbReverseProbe(True, ())]
    # AdbReverseProbe stores missing ports, so the first probe represents the
    # topology erased by the flap and the second verifies the rebuild.

    async def no_coturn_restart():
        return False

    async def command(_command, *, timeout):
        return subprocess.CompletedProcess([], 0, "", "")

    original_rebuild = lifecycle._rebuild_usb

    async def rebuild():
        events.append("rebuild")
        await original_rebuild()

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_adb_reverse_rules",
            side_effect=probes,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value=complete,
        ),
        patch("isaaccapture.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"),
        patch.object(lifecycle, "_ensure_coturn", new=no_coturn_restart),
        patch.object(lifecycle, "_run_adb_command", new=command),
        patch.object(lifecycle, "_rebuild_usb", new=rebuild),
        patch.object(lifecycle, "_attach_existing_monitor", return_value=True),
        patch.object(lifecycle, "_automate") as automate,
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.run_oob_connect"
        ) as run_oob_connect,
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._close_stale_teleop_tabs"
        ) as close_tabs,
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.open_url_on_headset"
        ) as navigate,
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert events == ["rebuild", "sleep"]
    assert lifecycle.generation == 0
    assert lifecycle.snapshot["health"] == "active"
    assert lifecycle.snapshot["state"] == "ACTIVE"
    assert lifecycle.browser_client == "surviving-page"
    automate.assert_not_called()
    run_oob_connect.assert_not_called()
    close_tabs.assert_not_called()
    navigate.assert_not_called()


async def test_reverse_probe_adb_loss_enters_same_preservation_recovery():
    hub = FakeHub()
    ready = AdbDevices((("original", "device"),))
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        sleep=lambda _seconds: pytest.fail("transport transition must not sleep"),
    )
    lifecycle.selected = "original"
    lifecycle._ready_count = 2
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle.browser_ready = True
    lifecycle.browser_client = "surviving-page"
    rebuilds = []

    async def no_coturn_restart():
        return False

    async def stop_at_rebuild():
        rebuilds.append(True)
        raise asyncio.CancelledError

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_adb_reverse_rules",
            return_value=AdbReverseProbe(False, ()),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value="",
        ),
        patch.object(lifecycle, "_ensure_coturn", new=no_coturn_restart),
        patch.object(lifecycle, "_rebuild_usb", new=stop_at_rebuild),
        patch.object(lifecycle, "_automate") as automate,
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert lifecycle._transport_lost is True
    assert lifecycle._restore_existing_browser is True
    assert rebuilds == [True]
    automate.assert_not_called()
    transition = next(
        status for status in hub.statuses if "ADB unavailable" in status["reason"]
    )
    assert transition["adbReady"] is False
    assert transition["networkPresent"] is False


async def test_coturn_substrate_failure_enters_immediate_preservation_recovery():
    hub = FakeHub()
    ready = AdbDevices((("original", "device"),))

    async def no_sleep(_):
        pytest.fail("TURN transport-loss transition must not take the generic sleep")

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        sleep=no_sleep,
    )
    lifecycle.selected = "original"
    lifecycle._ready_count = 2
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle._last_health = "active"
    lifecycle.browser_ready = True
    lifecycle.browser_client = "surviving-page"
    rebuilds = []

    async def failed_coturn():
        raise adb.OobAdbError("coturn is not listening")

    async def stop_at_rebuild():
        rebuilds.append(True)
        raise asyncio.CancelledError

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value="",
        ),
        patch.object(lifecycle, "_ensure_coturn", new=failed_coturn),
        patch.object(lifecycle, "_rebuild_usb", new=stop_at_rebuild),
        patch.object(lifecycle, "_automate") as automate,
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_adb_reverse_rules"
        ) as reverse_probe,
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert lifecycle._transport_lost is True
    assert lifecycle._restore_existing_browser is True
    assert rebuilds == [True]
    automate.assert_not_called()
    reverse_probe.assert_not_called()
    transition = next(
        status
        for status in hub.statuses
        if "TURN prerequisite unavailable" in status["reason"]
    )
    assert transition["adbReady"] is True
    assert transition["networkPresent"] is True


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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"
        ) as awake,
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.clear_headset_browser_cache"
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
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run
        ),
        patch.object(lifecycle, "_ensure_coturn") as coturn,
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.run_oob_connect"
        ) as full_bootstrap,
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.attach_existing_oob_tab"
        ) as existing_tab,
    ):
        with pytest.raises(Exception, match="Host listener on tcp:49100"):
            await lifecycle._rebuild_usb()
    coturn.assert_not_called()
    full_bootstrap.assert_not_called()
    existing_tab.assert_not_called()
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
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run
        ),
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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run
        ),
        patch.object(lifecycle, "_prepare_device", new=slow_prepare),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert any(
        "Recovery attempt timed out" in status["reason"] for status in hub.statuses
    )
    assert ["adb", "forward", "--remove", "tcp:9223"] in calls


async def test_timeout_after_connect_click_does_not_dispatch_again():
    hub = FakeHub()
    hub.probe_browser = lambda *_args: asyncio.sleep(0, result=None)
    ready = AdbDevices((("original", "device"),))
    ticks = 0
    clicks = []

    async def stop_after_verification(_):
        nonlocal ticks
        ticks += 1
        if ticks == 3:
            raise asyncio.CancelledError

    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=False,
        host_client=False,
        config=RecoveryConfig(0.02, 0.001),
        sleep=stop_after_verification,
    )
    lifecycle.selected = "original"
    lifecycle._ready_count = 2
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT

    async def connect(*, on_dispatched, **_kwargs):
        clicks.append(True)
        on_dispatched()
        await asyncio.sleep(0.1)

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch.object(lifecycle, "_prepare_device", new=lambda: asyncio.sleep(0)),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.run_oob_connect",
            side_effect=connect,
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert len(clicks) == 1
    assert lifecycle.connect_dispatched
    assert any(s["state"] == "VERIFYING_BROWSER" for s in hub.statuses)


async def test_replug_repairs_transport_and_resumes_existing_page_without_automation():
    hub = FakeHub()
    now = time.time()
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(
        0,
        result={
            "clientId": "surviving-page",
            "streaming": True,
            "lastMetricsAt": (now + 1) * 1000,
            "streamPhase": "streaming",
            "terminalEventId": None,
        },
    )
    calls = []
    monitor_attaches = []
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(interval_sec=0.01),
        host_listener_probe=lambda _port: True,
    )
    lifecycle.selected = "original"
    lifecycle.generation = 4
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True

    def run(args, **_kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    async def no_coturn_restart():
        lifecycle._turn_listener_ready = True
        return False

    async def monitor_attached():
        monitor_attaches.append(True)
        return True

    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._adb_run", side_effect=run
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_adb_reverse_rules",
            return_value=AdbReverseProbe(True, ()),
        ),
        patch.object(lifecycle, "_ensure_coturn", new=no_coturn_restart),
        patch.object(lifecycle, "_attach_existing_monitor", new=monitor_attached),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.run_oob_connect"
        ) as full_bootstrap,
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.attach_existing_oob_tab"
        ) as same_tab,
    ):
        await lifecycle._rebuild_usb()
        lifecycle._repair_started_at = now
        lifecycle._client_grace_deadline = lifecycle.clock() + 20
        await lifecycle._recover_existing_browser()

    reversed_ports = [
        args[2]
        for args in calls
        if len(args) >= 4 and args[1] == "reverse" and args[2].startswith("tcp:")
    ]
    assert reversed_ports == ["tcp:8080", "tcp:48322", "tcp:49100", "tcp:3478"]
    full_bootstrap.assert_not_called()
    same_tab.assert_not_called()
    assert monitor_attaches
    assert lifecycle.generation == 4
    assert lifecycle.snapshot["health"] == "active"


async def test_existing_page_retrying_then_streaming_within_grace_never_clicks():
    hub = FakeHub()
    now = time.time()
    reports = iter(
        [
            {
                "clientId": "surviving-page",
                "streaming": False,
                "lastMetricsAt": None,
                "streamPhase": "retrying",
                "terminalEventId": None,
            },
            {
                "clientId": "surviving-page",
                "streaming": True,
                "lastMetricsAt": (now + 1) * 1000,
                "streamPhase": "streaming",
                "terminalEventId": None,
            },
        ]
    )
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(0, result=next(reports))
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
    )
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = now
    lifecycle._client_grace_deadline = lifecycle.clock() + 20

    async def monitor_attached():
        return True

    with (
        patch.object(lifecycle, "_attach_existing_monitor", new=monitor_attached),
        patch.object(lifecycle, "_same_tab_connect") as same_tab,
        patch.object(lifecycle, "_automate") as full_bootstrap,
    ):
        await lifecycle._recover_existing_browser()
        assert lifecycle.snapshot["streamPhase"] == "retrying"
        await lifecycle._recover_existing_browser()
    same_tab.assert_not_called()
    full_bootstrap.assert_not_called()
    assert lifecycle.snapshot["health"] == "active"


async def test_fresh_stream_without_cdp_never_mutates_live_browser():
    hub = FakeHub()
    now = time.time()
    clock = [0.0]
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(
        0,
        result={
            "clientId": "surviving-page",
            "streaming": True,
            "lastMetricsAt": (now + 1) * 1000,
            "streamPhase": "streaming",
            "terminalEventId": None,
        },
    )
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        clock=lambda: clock[0],
    )
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = now
    lifecycle._client_grace_deadline = 20.0
    bootstraps = []

    async def bootstrap():
        bootstraps.append(True)

    with (
        patch.object(lifecycle, "_attach_existing_monitor", return_value=False),
        patch.object(lifecycle, "_automate", new=bootstrap),
    ):
        await lifecycle._recover_existing_browser()
        assert lifecycle.snapshot["health"] == "degraded"
        assert lifecycle.snapshot["streaming"] is True
        assert lifecycle.snapshot["turnEndToEndHealthy"] is True
        assert lifecycle._transport_lost is True
        clock[0] = 21.0
        await lifecycle._recover_existing_browser()
    assert bootstraps == []


async def test_fresh_stream_passively_reattaches_cdp_after_episode_expires():
    hub = FakeHub()
    wall_now = time.time()
    clock = [3.0]
    ready = AdbDevices((("original", "device"),))
    report = {
        "clientId": "surviving-page",
        "streaming": True,
        "lastMetricsAt": (wall_now + 1) * 1000,
        "streamPhase": "streaming",
        "terminalEventId": None,
    }

    async def get_snapshot():
        return {"headsets": [report]}

    async def probe_browser(*_args, **_kwargs):
        return report

    hub.get_snapshot = get_snapshot
    hub.probe_browser = probe_browser
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(timeout_sec=2, interval_sec=5),
        clock=lambda: clock[0],
    )
    lifecycle.selected = "original"
    lifecycle.generation = 4
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle.episode_start = 0.0
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = wall_now
    lifecycle._client_grace_deadline = 0.0
    lifecycle._fresh_stream_without_cdp = True
    attach_results = iter((False, True))
    attach_times = []

    async def attach_monitor():
        attach_times.append(clock[0])
        return next(attach_results)

    async def sleep(seconds):
        if lifecycle.snapshot.get("health") == "active":
            raise asyncio.CancelledError
        assert seconds <= 1.0
        clock[0] += seconds

    lifecycle.sleep = sleep
    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            return_value=ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value="stable-topology",
        ),
        patch.object(lifecycle, "_attach_existing_monitor", new=attach_monitor),
        patch.object(lifecycle, "_same_tab_connect") as same_tab,
        patch.object(lifecycle, "_automate") as full_bootstrap,
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert attach_times == [3.0, 4.0]
    degraded_streaming = [
        status for status in hub.statuses if status["health"] == "degraded"
    ]
    assert degraded_streaming
    assert all(status["streaming"] is True for status in degraded_streaming)
    assert all(status["clientMetricsFresh"] is True for status in degraded_streaming)
    assert all(status["turnEndToEndHealthy"] is True for status in degraded_streaming)
    assert lifecycle.snapshot["health"] == "active"
    assert lifecycle.snapshot["state"] == "ACTIVE"
    assert lifecycle.generation == 4
    same_tab.assert_not_called()
    full_bootstrap.assert_not_called()


async def test_second_cable_loss_invalidates_repair_before_browser_recovery():
    hub = FakeHub()
    clock = [0.0]
    ready = AdbDevices((("original", "device"),))
    absent = AdbDevices(())
    observations = [absent, absent, ready]
    report = {
        "clientId": "surviving-page",
        "streaming": True,
        "lastMetricsAt": (time.time() + 10) * 1000,
        "streamPhase": "streaming",
        "terminalEventId": None,
    }

    async def get_snapshot():
        return {"headsets": [report]}

    async def probe_browser(*_args, **_kwargs):
        return {**report, "lastMetricsAt": (time.time() + 1) * 1000}

    hub.get_snapshot = get_snapshot
    hub.probe_browser = probe_browser
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(interval_sec=5),
        clock=lambda: clock[0],
    )
    lifecycle.selected = "original"
    lifecycle.generation = 4
    lifecycle._last_observation = (ready.devices, ready.diagnostic)
    lifecycle.last_network_state = HeadsetNetworkState.NETWORK_PRESENT
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = time.time()
    lifecycle._client_grace_deadline = 40.0
    lifecycle._fresh_stream_without_cdp = True
    actions = []
    invalidations = []
    original_invalidate = lifecycle._invalidate_completed_repair

    def invalidate(signature):
        changed = original_invalidate(signature)
        invalidations.append(changed)
        return changed

    async def rebuild():
        actions.append("rebuild")
        lifecycle._transport_disruption_signature = None
        lifecycle._prerequisite_signature = None

    async def attach_monitor():
        actions.append("attach")
        return True

    async def sleep(seconds):
        if lifecycle.snapshot.get("health") == "active":
            raise asyncio.CancelledError
        assert seconds <= 1.0
        if len(invalidations) == 1:
            assert lifecycle._repair_started_at is None
            assert lifecycle._client_grace_deadline is None
            assert lifecycle._fresh_stream_without_cdp is False
        clock[0] += seconds

    lifecycle.sleep = sleep
    with (
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.enumerate_adb_devices",
            side_effect=lambda: observations.pop(0) if observations else ready,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.probe_headset_network",
            return_value=HeadsetNetworkProbe(HeadsetNetworkState.NETWORK_PRESENT),
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.adb._run_adb",
            return_value="restored-topology",
        ),
        patch("isaaccapture.cloudxr.oob_teleop_lifecycle.adb.assert_headset_awake"),
        patch.object(lifecycle, "_invalidate_completed_repair", new=invalidate),
        patch.object(lifecycle, "_rebuild_usb", new=rebuild),
        patch.object(lifecycle, "_attach_existing_monitor", new=attach_monitor),
        patch.object(lifecycle, "_same_tab_connect") as same_tab,
        patch.object(lifecycle, "_automate") as full_bootstrap,
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()

    assert invalidations == [True, False]
    assert actions == ["rebuild", "attach"]
    assert lifecycle.snapshot["health"] == "active"
    assert lifecycle.generation == 4
    same_tab.assert_not_called()
    full_bootstrap.assert_not_called()


@pytest.mark.parametrize("phase", ["idle", "retrying"])
async def test_nonstreaming_client_without_cdp_bootstraps_once_after_grace(phase):
    hub = FakeHub()
    clock = [0.0]
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(
        0,
        result={
            "clientId": "surviving-page",
            "streaming": False,
            "lastMetricsAt": None,
            "streamPhase": phase,
            "terminalEventId": None,
        },
    )
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(timeout_sec=60),
        clock=lambda: clock[0],
    )
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = time.time()
    lifecycle._client_grace_deadline = 20.0
    same_tab_attempts = []
    bootstraps = []

    async def no_tab(key):
        same_tab_attempts.append(key)
        return False

    async def bootstrap():
        bootstraps.append(clock[0])
        # Mirror the ownership transition at the start of the real _automate.
        lifecycle._transport_lost = False
        lifecycle._restore_existing_browser = False
        lifecycle.connect_dispatched = True

    with (
        patch.object(lifecycle, "_attach_existing_monitor", return_value=False),
        patch.object(lifecycle, "_same_tab_connect", new=no_tab),
        patch.object(lifecycle, "_automate", new=bootstrap),
    ):
        await lifecycle._recover_existing_browser()
        clock[0] = 19.9
        await lifecycle._recover_existing_browser()
        assert same_tab_attempts == []
        assert bootstraps == []

        clock[0] = 20.0
        await lifecycle._recover_existing_browser()
        assert same_tab_attempts == ["grace:0"]
        assert bootstraps == [20.0]

        # A repeated stale report cannot dispatch another fallback generation.
        clock[0] = 59.9
        await lifecycle._recover_existing_browser()
    assert same_tab_attempts == ["grace:0"]
    assert bootstraps == [20.0]


@pytest.mark.parametrize(
    ("timeout_sec", "interval_sec", "expected_deadline"),
    [(5.0, 5.0, 0.0), (5.0, 1.0, 4.0)],
)
async def test_short_episode_caps_client_grace_and_falls_back_once(
    timeout_sec, interval_sec, expected_deadline
):
    hub = FakeHub()
    clock = [0.0]
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(
        0,
        result={
            "clientId": "surviving-page",
            "streaming": False,
            "lastMetricsAt": None,
            "streamPhase": "retrying",
            "terminalEventId": None,
        },
    )
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(timeout_sec=timeout_sec, interval_sec=interval_sec),
        clock=lambda: clock[0],
    )
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = time.time()
    lifecycle._client_grace_deadline = lifecycle._bounded_client_grace_deadline()
    assert lifecycle.config.client_recovery_grace_sec == 40.0
    assert lifecycle._client_grace_deadline == expected_deadline
    bootstraps = []

    async def bootstrap():
        bootstraps.append(clock[0])
        lifecycle._transport_lost = False
        lifecycle._restore_existing_browser = False
        lifecycle.connect_dispatched = True

    with (
        patch.object(lifecycle, "_attach_existing_monitor", return_value=False),
        patch.object(lifecycle, "_same_tab_connect", return_value=False),
        patch.object(lifecycle, "_automate", new=bootstrap),
    ):
        if expected_deadline:
            clock[0] = expected_deadline - 0.1
            await lifecycle._recover_existing_browser()
            assert bootstraps == []
        clock[0] = expected_deadline
        await lifecycle._recover_existing_browser()
        clock[0] = timeout_sec
        await lifecycle._recover_existing_browser()

    assert bootstraps == [expected_deadline]


async def test_terminal_event_clicks_same_tab_once_and_distinct_event_can_retry():
    hub = FakeHub()
    event = ["terminal-1"]
    phase = ["terminal"]
    now = [0.0]
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(
        0,
        result={
            "clientId": "surviving-page",
            "streaming": False,
            "lastMetricsAt": None,
            "streamPhase": phase[0],
            "terminalEventId": event[0],
        },
    )
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        clock=lambda: now[0],
    )
    lifecycle.selected = "original"
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = time.time()
    lifecycle._client_grace_deadline = lifecycle.clock() + 20
    clicks = []

    async def attach(*, click_connect, on_dispatched=None):
        clicks.append(click_connect)
        if on_dispatched:
            on_dispatched()
        return asyncio.create_task(asyncio.Event().wait())

    with patch(
        "isaaccapture.cloudxr.oob_teleop_lifecycle.adb.attach_existing_oob_tab",
        side_effect=attach,
    ):
        await lifecycle._recover_existing_browser()
        await lifecycle._recover_existing_browser()
        phase[0] = "retrying"
        now[0] = 100.0
        await lifecycle._recover_existing_browser()
        event[0] = "terminal-2"
        phase[0] = "terminal"
        await lifecycle._recover_existing_browser()
    await lifecycle._stop_monitor()
    assert clicks[0] is False
    assert [click for click in clicks if click] == [True, True]
    assert lifecycle.generation == 0


async def test_no_surviving_page_after_grace_bootstraps_once():
    hub = FakeHub()
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(0, result=None)
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        clock=lambda: 100.0,
    )
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = time.time()
    lifecycle._client_grace_deadline = 99.0
    bootstraps = []

    async def no_tab(_key):
        return False

    async def bootstrap():
        bootstraps.append(True)
        lifecycle._transport_lost = False

    with (
        patch.object(lifecycle, "_same_tab_connect", new=no_tab),
        patch.object(lifecycle, "_automate", new=bootstrap),
    ):
        await lifecycle._recover_existing_browser()
    assert bootstraps == [True]


async def test_terminal_without_attachable_tab_bootstraps_once_after_grace():
    hub = FakeHub()
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(
        0,
        result={
            "clientId": "surviving-page",
            "streaming": False,
            "lastMetricsAt": None,
            "streamPhase": "terminal",
            "terminalEventId": "terminal-1",
        },
    )
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        clock=lambda: 100.0,
    )
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = time.time()
    lifecycle._client_grace_deadline = 99.0
    bootstraps = []

    async def no_tab(_key):
        return False

    async def bootstrap():
        bootstraps.append(True)
        lifecycle._transport_lost = False

    with (
        patch.object(lifecycle, "_same_tab_connect", new=no_tab),
        patch.object(lifecycle, "_automate", new=bootstrap),
    ):
        await lifecycle._recover_existing_browser()
    assert bootstraps == [True]


async def test_later_terminal_without_cdp_enters_bounded_existing_browser_recovery():
    hub = FakeHub()
    hub.get_snapshot = lambda: asyncio.sleep(
        0,
        result={
            "headsets": [
                {
                    "clientId": "page",
                    "streaming": False,
                    "lastMetricsAt": None,
                    "streamPhase": "terminal",
                    "terminalEventId": "later-terminal",
                }
            ]
        },
    )
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
    )
    lifecycle.browser_ready = True
    lifecycle.browser_client = "page"

    with (
        patch.object(lifecycle, "_attach_existing_monitor", return_value=True),
        patch.object(lifecycle, "_same_tab_connect") as same_tab_connect,
    ):
        with pytest.raises(adb.OobAdbError, match="terminal event") as disruption:
            await lifecycle._observe_stream()
    same_tab_connect.assert_not_called()
    await lifecycle._enter_transport_recovery(
        str(disruption.value),
        "REBUILDING_USB",
        adb_ready=disruption.value.adb_ready,
        network_present=disruption.value.network_present,
    )
    assert lifecycle._transport_lost is True
    assert lifecycle._restore_existing_browser is True
    assert lifecycle._repair_started_at is None
    assert lifecycle._client_grace_deadline is None


async def test_stale_pre_repair_metrics_do_not_mark_existing_page_active():
    hub = FakeHub()
    now = time.time()
    hub.probe_browser = lambda *_args, **_kwargs: asyncio.sleep(
        0,
        result={
            "clientId": "surviving-page",
            "streaming": True,
            "lastMetricsAt": (now - 5) * 1000,
            "streamPhase": "streaming",
            "terminalEventId": None,
        },
    )
    lifecycle = OobLifecycle(
        hub=hub,
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
    )
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._repair_started_at = now
    lifecycle._client_grace_deadline = lifecycle.clock() + 20

    async def monitor_attached():
        return True

    with patch.object(lifecycle, "_attach_existing_monitor", new=monitor_attached):
        await lifecycle._recover_existing_browser()
    assert lifecycle.snapshot["health"] == "browser_ready"
    assert lifecycle.snapshot["clientMetricsFresh"] is False
