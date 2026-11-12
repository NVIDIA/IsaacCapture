# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presence-gated wake recovery without hardware or a running XR runtime."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from headset_presence_recovery import (
    PresenceObservation,
    PresenceRecovery,
    parse_pico_presence,
    run_device_adb,
)

SENSOR = "stk_stk3x1x_prox Proximity Sensor Wakeup"


def sensor_dump(events="1 (ts=9.9, wall=18:45:54.118) 0.00, 0.00, 0.00,"):
    count = len(events.splitlines())
    return f"""0x00000052) active-count = 1; sampling_period(ms) = {{200.0}}
0x00000052) {SENSOR} | sensortek | type: android.sensor.proximity(8)
{SENSOR}: last {count} events
\t {events}
Mode : NORMAL
Sensor Privacy: disabled
Connection Number: 5
\tOperating Mode: NORMAL
\tFrozen state: 0 com.pvr.sensormanager.ProximitySensorManager
\t {SENSOR} 0x00000052 | status: active | pending flush events 0
"""


@pytest.mark.parametrize(
    "change",
    [
        ("Sensor Privacy: disabled", "Sensor Privacy: enabled"),
        ("Mode : NORMAL", "Mode : REPLAY_DATA_INJECTION"),
        ("Operating Mode: NORMAL", "Operating Mode: DATA_INJECTION"),
        ("Frozen state: 0", "Frozen state: 1"),
        ("status: active", "status: inactive"),
        ("active-count = 1", "active-count = 0"),
        ("0x00000052 | status", "0x00000051 | status"),
        ("ProximitySensorManager", "OtherSensorManager"),
        ("0.00, 0.00, 0.00,", "NaN, 0.00, 0.00,"),
    ],
)
def test_unsupported_or_stale_dump_fails_closed(change):
    assert parse_pico_presence(sensor_dump().replace(*change)) == "unknown"


def test_latest_event_overrides_retained_near():
    near = "1 (ts=9.9, wall=18:45:54.118) 0.00, 0.00, 0.00,"
    far = "2 (ts=100, wall=18:45:55.118) 5.00, 0.00, 0.00,"
    assert parse_pico_presence(sensor_dump()) == "present"
    assert parse_pico_presence(sensor_dump(f"{near}\n\t {far}")) == "absent"
    assert (
        parse_pico_presence(sensor_dump(f"{near}\n\t 2 (ts=100, wall=x) garbage,"))
        == "unknown"
    )


@pytest.mark.parametrize(
    "presence,wake",
    [
        ("absent", "Asleep"),
        ("unknown", "Asleep"),
        ("present", "Awake"),
        ("present", "Dozing"),
    ],
)
async def test_only_present_and_asleep_can_wake(presence, wake):
    command = AsyncMock()
    guard = PresenceRecovery("target", command=command)
    guard.observe = AsyncMock(return_value=PresenceObservation(presence, wake))
    await guard.step()
    command.assert_not_called()


async def test_presence_is_rechecked_before_every_wake_and_removal_prevents_key():
    command = AsyncMock()
    guard = PresenceRecovery("target", command=command)
    guard.observe = AsyncMock(
        side_effect=[
            PresenceObservation("present", "Asleep"),
            PresenceObservation("absent", "Asleep"),
        ]
    )
    await guard.step()
    command.assert_not_called()


async def test_bounced_awake_does_not_reset_retry_budget_and_attempts_are_bounded():
    now = [0.0]
    command = AsyncMock()
    guard = PresenceRecovery("target", command=command, clock=lambda: now[0])
    asleep = PresenceObservation("present", "Asleep")
    guard.observe = AsyncMock(return_value=asleep)
    await guard.step()
    now[0] = 1
    guard.observe.return_value = PresenceObservation("present", "Awake")
    await guard.step()
    guard.observe.return_value = asleep
    await guard.step()
    assert command.await_count == 1
    for stamp in (5, 10, 15, 20):
        now[0] = stamp
        await guard.step()
    assert command.await_count == 3
    assert guard.snapshot["state"] == "recovery_failed"
    assert all(call.args[0] == "target" for call in command.await_args_list)
    guard.observe.return_value = PresenceObservation("absent", "Asleep")
    await guard.step()
    guard.observe.return_value = asleep
    now[0] = 25
    await guard.step()
    assert command.await_count == 4


async def test_unsupported_vendor_never_reads_sensor_or_sends_wake():
    command = AsyncMock(return_value="Meta\n")
    guard = PresenceRecovery("target", command=command)
    await guard.step()
    assert guard.snapshot["state"] == "unsupported"
    assert command.await_count == 1


async def test_adb_failures_back_off_and_do_not_inherit_presence():
    guard = PresenceRecovery(
        "target", command=AsyncMock(side_effect=OSError("offline"))
    )
    assert [await guard.step() for _ in range(5)] == [1, 2, 4, 8, 10]
    assert guard.snapshot["presence"] == "unknown"


async def test_cancel_kills_and_reaps_owned_adb(monkeypatch):
    started = asyncio.Event()
    process = AsyncMock()
    process.returncode = None
    process.kill = lambda: setattr(process, "returncode", -9)

    async def communicate():
        started.set()
        if process.returncode is None:
            await asyncio.Future()
        return b"", b""

    process.communicate.side_effect = communicate
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    task = asyncio.create_task(run_device_adb("target", ["shell", "dumpsys", "power"]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode == -9
    assert process.communicate.await_count == 2
    assert create.call_args.args[:3] == ("adb", "-s", "target")


def test_truncated_or_unindented_newest_event_cannot_reuse_near():
    near = "1 (ts=9.9, wall=18:45:54.118) 0.00, 0.00, 0.00,"
    assert (
        parse_pico_presence(sensor_dump(near).replace("last 1 events", "last 2 events"))
        == "unknown"
    )
    assert (
        parse_pico_presence(sensor_dump(near + "\n2 (ts=100, wall=x) 5.00,"))
        == "unknown"
    )


def test_lifecycle_context_disables_legacy_blind_wake(monkeypatch):
    from cloudxr_py_test_ns import oob_teleop_adb as adb

    monkeypatch.setattr(adb, "headset_wakefulness", lambda: "Asleep")
    monkeypatch.setattr(adb, "_adb_getprop", lambda _prop: "")
    calls = []
    monkeypatch.setattr(adb, "_adb_run", lambda *args, **kwargs: calls.append(args))
    token = adb.PRESENCE_RECOVERY_OWNS_WAKE.set(True)
    try:
        with pytest.raises(adb.HeadsetNotAwakeError):
            adb.assert_headset_awake(timeout=0, require_awake=True)
    finally:
        adb.PRESENCE_RECOVERY_OWNS_WAKE.reset(token)
    assert calls == []


async def test_stable_awake_resets_budget_but_unknown_does_not():
    now = [0.0]
    guard = PresenceRecovery("target", command=AsyncMock(), clock=lambda: now[0])
    guard.attempts = 3
    guard.observe = AsyncMock(return_value=PresenceObservation("unknown", "Asleep"))
    await guard.step()
    assert guard.attempts == 3
    guard.observe.return_value = PresenceObservation("present", "Awake")
    await guard.step()
    now[0] = 4
    await guard.step()
    assert guard.attempts == 3
    now[0] = 5
    await guard.step()
    assert guard.attempts == 0


@pytest.mark.parametrize(
    "manufacturer,brand,model,can_wake",
    [
        ("Meta", "oculus", "Quest 3", True),
        ("Oculus", "oculus", "Quest Pro", True),
        ("", "oculus", "Quest 2", True),
        ("PICO", "PICO", "PICO 4", False),
        ("PICO", "oculus", "PICO 4", False),
        ("Meta", "PICO", "PICO 4", False),
        ("Meta", "oculus", "", False),
        ("Samsung", "samsung", "Galaxy", False),
        ("", "", "", False),
    ],
)
def test_owned_legacy_wake_requires_fresh_quest_identity(
    monkeypatch, manufacturer, brand, model, can_wake
):
    from cloudxr_py_test_ns import oob_teleop_adb as adb

    properties = {
        "ro.product.manufacturer": manufacturer,
        "ro.product.brand": brand,
        "ro.product.model": model,
    }
    getprop = Mock(side_effect=properties.__getitem__)
    command = Mock()
    monkeypatch.setattr(adb, "_adb_getprop", getprop)
    monkeypatch.setattr(adb, "_adb_run", command)
    monkeypatch.setattr(adb, "headset_wakefulness", lambda: "Asleep")
    token = adb.PRESENCE_RECOVERY_OWNS_WAKE.set(True)
    try:
        with pytest.raises(adb.HeadsetNotAwakeError):
            adb.assert_headset_awake(timeout=0, require_awake=True)
        assert command.call_count == int(can_wake)
        if can_wake:
            assert command.call_args.args[0] == [
                "adb",
                "shell",
                "input",
                "keyevent",
                "KEYCODE_WAKEUP",
            ]
        properties.update(dict.fromkeys(properties, ""))
        with pytest.raises(adb.HeadsetNotAwakeError):
            adb.assert_headset_awake(timeout=0, require_awake=True)
        assert command.call_count == int(can_wake)
        assert getprop.call_count == 6
    finally:
        adb.PRESENCE_RECOVERY_OWNS_WAKE.reset(token)
