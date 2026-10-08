# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused topology, timing, report and startup ordering checks."""

from __future__ import annotations

import asyncio
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import pytest

from isaaccapture.cloudxr import oob_teleop_adb as adb
from isaaccapture.cloudxr import usb_health
from isaaccapture.cloudxr.oob_teleop_lifecycle import OobLifecycle, RecoveryConfig


def _field(directory: Path, name: str, value: str) -> None:
    (directory / name).write_text(value)


def test_pinned_adb_sysfs_topology_and_faster_root(tmp_path):
    controller = tmp_path / "0000:00:14.0"
    device = controller / "usb1" / "1-2" / "1-2.1"
    faster = controller / "usb2"
    device.mkdir(parents=True)
    faster.mkdir()
    root = tmp_path / "devices"
    root.mkdir()
    (root / "1-2.1").symlink_to(device, target_is_directory=True)
    (root / "usb1").symlink_to(controller / "usb1", target_is_directory=True)
    (root / "usb2").symlink_to(faster, target_is_directory=True)
    _field(device, "speed", "480")
    _field(device, "product", "PICO 4 Ultra")
    _field(faster, "speed", "20000")
    listing = "List of devices attached\nheadset device usb:1-2.1 model:A9210\nother device usb:3-1\n"
    status = usb_health.inspect_link("headset", listing, root)
    assert status["probeSucceeded"] is True
    assert status["negotiatedSpeedMbps"] == 480
    assert status["isBetterPortAvailable"] is True
    assert status["betterPortCandidates"][0]["speedMbps"] == 20000
    assert status["controller"] == "0000:00:14.0"
    assert "headset" not in json.dumps(status)


def test_ambiguous_or_missing_transport_has_unknown_result(tmp_path):
    status = usb_health.inspect_link("selected", "other device usb:1-2", tmp_path)
    assert status["probeSucceeded"] is False
    assert status["isBetterPortAvailable"] is None
    assert status["negotiatedSpeedMbps"] is None


@pytest.mark.asyncio
async def test_curl_measurement_counts_only_completed_responses(monkeypatch):
    calls = []

    async def shell(serial, script, timeout):
        calls.append((serial, script, timeout))
        if len(calls) == 1:
            return (
                0,
                "/system/bin/curl\n/system/bin/timeout\n1791403150633088373\n/system/bin/cat\n",
            )
        return (
            0,
            "B:500000:0.010000:0.040000\n"
            "B:500000:0.020000:0.060000\n"
            "curl: (28) partial final response was cut off\n"
            "T:1000000000000000000:1000000003000000000\n",
        )

    monkeypatch.setattr(usb_health, "_adb_shell", shell)
    status = await usb_health.measure_transfer("selected", 48322, 500000)
    assert status["completed"] is True
    assert status["effectiveMeasurementDurationMs"] == 3000
    assert status["completedBytes"] == 1000000
    assert status["hostToHeadsetMbps"] == pytest.approx(2.67, abs=0.01)
    assert status["hostToHeadsetResponseBodyMbps"] == pytest.approx(114.29)
    assert status["completedResponseBodyDurationMs"] == pytest.approx(70)
    assert status["completedFirstByteWaitDurationMs"] == pytest.approx(30)
    assert status["responseBodyTimingValid"] is True
    assert status["measurementRevision"] == 2
    assert status["method"] == "device_curl_static_asset_loop"
    assert calls[1][0] == "selected"
    assert "%{time_starttransfer}:%{time_total}" in calls[1][1]
    assert calls[1][2] == 5


@pytest.mark.asyncio
async def test_curl_phase_metric_unavailable_when_timings_are_invalid(monkeypatch):
    async def shell(_serial, _script, _timeout):
        if "command -v curl" in _script:
            return 0, "/system/bin/curl\n/system/bin/timeout\n1791403150633088373\n"
        return (
            0,
            "B:500000:0.060000:0.040000\nT:1000000000000000000:1000000003000000000\n",
        )

    monkeypatch.setattr(usb_health, "_adb_shell", shell)
    status = await usb_health.measure_transfer("selected", 48322, 500000)
    assert status["completed"] is True
    assert status["hostToHeadsetMbps"] == pytest.approx(1.33, abs=0.01)
    assert status["hostToHeadsetResponseBodyMbps"] is None
    assert status["responseBodyTimingValid"] is False


@pytest.mark.parametrize(
    "second_sample", ["B:123:bad:0.030000", "B:123:0.010000:0.030000"]
)
@pytest.mark.asyncio
async def test_curl_rejects_malformed_or_wrong_size_samples(monkeypatch, second_sample):
    async def shell(_serial, script, _timeout):
        if "command -v curl" in script:
            return 0, "/system/bin/curl\n/system/bin/timeout\n1791403150633088373\n"
        return (
            0,
            "B:500000:0.010000:0.040000\n"
            f"{second_sample}\nT:1000000000000000000:1000000003000000000\n",
        )

    monkeypatch.setattr(usb_health, "_adb_shell", shell)
    status = await usb_health.measure_transfer("selected", 48322, 500000)
    assert status["completed"] is False
    assert status["hostToHeadsetResponseBodyMbps"] is None
    assert status["outcome"] == "unsupported"


@pytest.mark.asyncio
async def test_missing_curl_uses_labeled_adb_fallback(monkeypatch):
    async def shell(_serial, _script, _timeout):
        return 0, "/system/bin/timeout\n1791403150633088373\n/system/bin/cat\n"

    async def fallback(_serial, result):
        result.update(
            completed=True,
            completedBytes=1024,
            effectiveMeasurementDurationMs=3000,
            hostToHeadsetMbps=0.003,
        )

    monkeypatch.setattr(usb_health, "_adb_shell", shell)
    monkeypatch.setattr(usb_health, "_measure_fallback", fallback)
    result = await usb_health.measure_transfer("selected", 48322, 500000)
    assert result["completed"] is True
    assert result["method"] == "adb_shell_sink"
    assert result["pathCoverage"] == "adb_transport_only"
    assert result["fallbackReason"]
    assert result["hostToHeadsetResponseBodyMbps"] is None
    assert result["headsetToHostMbps"] is None


def test_atomic_report_is_private_and_redacts_serial(tmp_path):
    report = usb_health.UsbHealthReport(tmp_path, "session-id", 0, "SECRET-SERIAL")
    report.update(topology={"probeSucceeded": False}, errors=["sysfs unavailable"])
    content = report.path.read_text()
    assert "SECRET-SERIAL" not in content
    assert json.loads(content)["schema"] == usb_health.SCHEMA
    assert report.path.stat().st_mode & 0o777 == 0o600
    assert report.directory.stat().st_mode & 0o777 == 0o700
    assert not list(report.directory.glob("*.tmp"))


def test_report_can_begin_before_adb_selection(tmp_path):
    report = usb_health.UsbHealthReport(tmp_path, "session-id", 0, None)
    report.event("WAITING_FOR_ADB")
    assert json.loads(report.path.read_text())["device"]["serialHash"] is None
    report.set_device("SECRET-SERIAL")
    payload = report.path.read_text()
    assert "SECRET-SERIAL" not in payload
    assert json.loads(payload)["device"]["serialHash"].startswith(
        "sha256-install-salted:"
    )


def test_concurrent_reports_share_only_a_complete_published_salt(tmp_path):
    barrier = Barrier(2)
    original_link = usb_health.os.link

    def race_publish(source, destination):
        barrier.wait(timeout=5)
        return original_link(source, destination)

    with (
        patch.object(usb_health.os, "link", side_effect=race_publish),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        reports = list(
            pool.map(
                lambda _: usb_health.UsbHealthReport(
                    tmp_path, "session", 0, "same-serial"
                ),
                range(2),
            )
        )
    directory = tmp_path / "usb-health"
    assert len((directory / ".serial-salt").read_bytes()) == 32
    assert not list(directory.glob(".serial-salt.*.tmp"))
    assert (
        reports[0].payload["device"]["serialHash"]
        == reports[1].payload["device"]["serialHash"]
    )


@pytest.mark.parametrize("invalid_salt", [b"", b"too-short"])
def test_incomplete_existing_serial_salt_is_rejected(tmp_path, invalid_salt):
    directory = tmp_path / "usb-health"
    directory.mkdir()
    (directory / ".serial-salt").write_bytes(invalid_salt)
    with pytest.raises(OSError, match="complete 32-byte"):
        usb_health.UsbHealthReport(tmp_path, "session", 0, "selected")


@pytest.mark.parametrize("wakefulness", ["Asleep", ""])
def test_strict_wake_confirmation_fails_closed(wakefulness):
    with (
        patch.object(adb, "headset_wakefulness", return_value=wakefulness),
        patch.object(adb, "_adb_run"),
    ):
        with pytest.raises(adb.OobAdbError, match="not confirmed"):
            adb.assert_headset_awake(timeout=0, require_awake=True)
        adb.assert_headset_awake(timeout=0)


def test_strict_wake_confirmation_accepts_observed_awake():
    with (
        patch.object(adb, "headset_wakefulness", return_value="Awake"),
        patch.object(adb, "_adb_run") as command,
    ):
        adb.assert_headset_awake(timeout=0, require_awake=True)
    command.assert_not_called()


@pytest.mark.asyncio
async def test_strict_wake_check_uses_lifecycle_pinned_adb_device():
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
    )
    lifecycle.selected = "selected-headset"
    token = adb.SELECTED_ADB_SERIAL.set(lifecycle.selected)
    try:
        with patch.object(
            adb.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "mWakefulness=Awake\n", ""),
        ) as command:
            await lifecycle._confirm_usb_awake()
    finally:
        adb.SELECTED_ADB_SERIAL.reset(token)
    assert lifecycle._usb_wake_confirmed is True
    assert command.call_args.args[0] == [
        "adb",
        "-s",
        "selected-headset",
        "shell",
        "dumpsys",
        "power",
    ]


class _Hub:
    async def set_lifecycle_snapshot(self, snapshot):
        pass

    async def get_snapshot(self):
        return {"headsets": []}


@pytest.mark.parametrize(
    ("usb_local", "policy", "speed", "expected"),
    [
        (
            True,
            "warn",
            480,
            ["prepare", "link", "rebuild", "awake", "transfer", "browser"],
        ),
        (True, "block", 480, ["prepare", "link", "rebuild", "awake", "transfer"]),
        (False, "warn", None, ["prepare", "browser"]),
    ],
)
@pytest.mark.asyncio
async def test_startup_checks_finish_before_browser(
    monkeypatch, usb_local, policy, speed, expected
):
    events = []
    selected = "selected"
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=usb_local,
        host_client=True,
        turn_port=3478 if usb_local else None,
        config=RecoveryConfig(),
        usb_health_policy=policy,
    )
    lifecycle.selected = selected
    devices = adb.AdbDevices(((selected, "device"),))
    lifecycle._last_observation = (devices.devices, devices.diagnostic)
    lifecycle._ready_count = 1

    async def step(name):
        events.append(name)

    async def link():
        events.append("link")
        lifecycle._usb_link = {"negotiatedSpeedMbps": speed}

    async def transfer():
        assert lifecycle._usb_wake_confirmed is True
        events.append("transfer")
        lifecycle._usb_transfer = {"completed": True}

    async def confirm_awake():
        events.append("awake")
        lifecycle._usb_wake_confirmed = True

    async def browser():
        events.append("browser")
        raise asyncio.CancelledError

    async def sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(lifecycle, "_prepare_device", lambda: step("prepare"))
    monkeypatch.setattr(lifecycle, "_check_usb_link", link)
    monkeypatch.setattr(lifecycle, "_rebuild_usb", lambda: step("rebuild"))
    monkeypatch.setattr(lifecycle, "_check_usb_transfer", transfer)
    monkeypatch.setattr(lifecycle, "_confirm_usb_awake", confirm_awake)
    monkeypatch.setattr(lifecycle, "_automate", browser)
    lifecycle.sleep = sleep
    with (
        patch.object(adb, "enumerate_adb_devices", return_value=devices),
        patch.object(
            adb,
            "probe_headset_network",
            return_value=adb.HeadsetNetworkProbe(
                adb.HeadsetNetworkState.NETWORK_PRESENT
            ),
        ),
        patch.object(adb, "_run_adb", return_value=""),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert events == expected


@pytest.mark.asyncio
async def test_missing_adb_writes_partial_usb_report(tmp_path, monkeypatch):
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        usb_health_logs_dir=tmp_path,
    )

    async def stop(_seconds):
        raise asyncio.CancelledError

    lifecycle.sleep = stop
    monkeypatch.setattr(adb, "enumerate_adb_devices", lambda: adb.AdbDevices(()))
    with pytest.raises(asyncio.CancelledError):
        await lifecycle.run()
    payload = json.loads(
        next((tmp_path / "usb-health").glob("usb-health-*.json")).read_text()
    )
    assert payload["decision"]["overallOutcome"] == "incomplete"
    assert payload["errors"][0]["code"] == "no_ready_device"
    assert payload["device"]["serialHash"] is None


@pytest.mark.asyncio
async def test_prepare_failure_writes_partial_usb_report(tmp_path, monkeypatch):
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        usb_health_logs_dir=tmp_path,
    )
    lifecycle.selected = "selected"
    devices = adb.AdbDevices((("selected", "device"),))
    lifecycle._last_observation = (devices.devices, devices.diagnostic)
    lifecycle._ready_count = 1

    async def fail_prepare():
        raise RuntimeError("private headset detail")

    async def stop(_seconds):
        raise asyncio.CancelledError

    lifecycle._prepare_device = fail_prepare
    lifecycle.sleep = stop
    with (
        patch.object(adb, "enumerate_adb_devices", return_value=devices),
        patch.object(
            adb,
            "probe_headset_network",
            return_value=adb.HeadsetNetworkProbe(
                adb.HeadsetNetworkState.NETWORK_PRESENT
            ),
        ),
        patch.object(adb, "_run_adb", return_value=""),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    payload_text = next((tmp_path / "usb-health").glob("usb-health-*.json")).read_text()
    payload = json.loads(payload_text)
    assert payload["decision"]["overallOutcome"] == "incomplete"
    assert any(
        item["state"] == "PREPARING_DEVICE" and item["code"] == "RuntimeError"
        for item in payload["errors"]
    )
    assert "private headset detail" not in payload_text
    assert '"selected"' not in payload_text


@pytest.mark.parametrize("failure", ["timeout", "symlink_loop"])
@pytest.mark.asyncio
async def test_link_probe_failure_preserves_report_and_warn_launch(
    tmp_path, monkeypatch, failure
):
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        usb_health_logs_dir=tmp_path,
    )
    lifecycle.selected = "selected"
    devices = adb.AdbDevices((("selected", "device"),))
    lifecycle._last_observation = (devices.devices, devices.diagnostic)
    lifecycle._ready_count = 1
    browser_reached = []

    async def no_op():
        pass

    async def awake():
        lifecycle._usb_wake_confirmed = True

    async def transfer():
        lifecycle._usb_transfer = {"completed": True, "outcome": "pass"}

    async def browser():
        browser_reached.append(True)
        raise asyncio.CancelledError

    async def stop(_seconds):
        raise asyncio.CancelledError

    def listing(args, **_kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 4)
        return subprocess.CompletedProcess(
            args, 0, "selected device usb:1-2.1 model:A9210\n", ""
        )

    def inspect(_serial, _listing):
        raise RuntimeError("sysfs symlink loop")

    lifecycle._prepare_device = no_op
    lifecycle._rebuild_usb = no_op
    lifecycle._confirm_usb_awake = awake
    lifecycle._check_usb_transfer = transfer
    lifecycle._automate = browser
    lifecycle.sleep = stop
    monkeypatch.setattr(adb, "_adb_run", listing)
    monkeypatch.setattr(usb_health, "inspect_link", inspect)
    with (
        patch.object(adb, "enumerate_adb_devices", return_value=devices),
        patch.object(
            adb,
            "probe_headset_network",
            return_value=adb.HeadsetNetworkProbe(
                adb.HeadsetNetworkState.NETWORK_PRESENT
            ),
        ),
        patch.object(adb, "_run_adb", return_value=""),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    payload = json.loads(
        next((tmp_path / "usb-health").glob("usb-health-*.json")).read_text()
    )
    assert browser_reached == [True]
    assert payload["linkProbe"]["probeSucceeded"] is False
    assert payload["topology"]["negotiatedSpeedMbps"] is None
    assert payload["topology"]["errors"]


@pytest.mark.asyncio
async def test_reverse_failure_retains_bounded_structured_error_history(
    tmp_path, monkeypatch
):
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        usb_health_logs_dir=tmp_path,
    )
    lifecycle.selected = "selected"
    devices = adb.AdbDevices((("selected", "device"),))
    lifecycle._last_observation = (devices.devices, devices.diagnostic)
    lifecycle._ready_count = 1
    lifecycle._ensure_usb_report()
    for index in range(20):
        lifecycle._record_usb_wait("WAITING_FOR_ADB", f"prior_{index}")

    async def no_op():
        pass

    async def fail_reverse():
        raise adb.OobAdbError("private reverse detail")

    async def stop(_seconds):
        raise asyncio.CancelledError

    lifecycle._prepare_device = no_op
    lifecycle._check_usb_link = no_op
    lifecycle._rebuild_usb = fail_reverse
    lifecycle.sleep = stop
    with (
        patch.object(adb, "enumerate_adb_devices", return_value=devices),
        patch.object(
            adb,
            "probe_headset_network",
            return_value=adb.HeadsetNetworkProbe(
                adb.HeadsetNetworkState.NETWORK_PRESENT
            ),
        ),
        patch.object(adb, "_run_adb", return_value=""),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    payload_text = next((tmp_path / "usb-health").glob("usb-health-*.json")).read_text()
    payload = json.loads(payload_text)
    assert len(payload["errors"]) == 20
    assert all(isinstance(item, dict) for item in payload["errors"])
    assert payload["errors"][0]["code"] == "prior_1"
    assert payload["errors"][-1]["state"] == "REBUILDING_USB"
    assert payload["errors"][-1]["code"] == "OobAdbError"
    assert payload["decision"]["overallOutcome"] == "incomplete"
    assert payload["decision"]["browserLaunchAllowed"] is False
    assert "private reverse detail" not in payload_text


@pytest.mark.asyncio
async def test_cancelled_transfer_persists_partial_json(tmp_path, monkeypatch):
    static = tmp_path / "static"
    static.mkdir()
    (static / "bundle.js").write_bytes(b"x" * 300000)
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        usb_health_logs_dir=tmp_path,
        web_client_static_dir=static,
    )
    lifecycle.selected = "selected"
    lifecycle._ensure_usb_report()
    lifecycle._usb_wake_confirmed = True

    async def cancel(_serial, _port, _size):
        raise asyncio.CancelledError

    monkeypatch.setattr(usb_health, "measure_transfer", cancel)
    with pytest.raises(asyncio.CancelledError):
        await lifecycle._check_usb_transfer()
    payload = json.loads(
        next((tmp_path / "usb-health").glob("usb-health-*.json")).read_text()
    )
    assert payload["transferTest"]["outcome"] == "cancelled"
    assert payload["transferTest"]["completed"] is False


@pytest.mark.asyncio
async def test_unconfirmed_wake_never_starts_transfer_or_browser(tmp_path, monkeypatch):
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        usb_health_logs_dir=tmp_path,
    )
    lifecycle.selected = "selected"
    devices = adb.AdbDevices((("selected", "device"),))
    lifecycle._last_observation = (devices.devices, devices.diagnostic)
    lifecycle._ready_count = 1
    events = []

    async def step(name):
        events.append(name)

    async def link():
        events.append("link")
        lifecycle._usb_link = {"negotiatedSpeedMbps": 5000}

    async def forbidden():
        pytest.fail("transfer/browser must not start before confirmed Awake")

    async def stop(_seconds):
        raise asyncio.CancelledError

    def wake(*, timeout, require_awake=False):
        assert require_awake is True
        events.append("wake_unconfirmed")
        raise adb.OobAdbError("Headset wakefulness was not confirmed as Awake")

    monkeypatch.setattr(lifecycle, "_prepare_device", lambda: step("prepare"))
    monkeypatch.setattr(lifecycle, "_check_usb_link", link)
    monkeypatch.setattr(lifecycle, "_rebuild_usb", lambda: step("rebuild"))
    monkeypatch.setattr(lifecycle, "_check_usb_transfer", forbidden)
    monkeypatch.setattr(lifecycle, "_automate", forbidden)
    lifecycle.sleep = stop
    with (
        patch.object(adb, "enumerate_adb_devices", return_value=devices),
        patch.object(
            adb,
            "probe_headset_network",
            return_value=adb.HeadsetNetworkProbe(
                adb.HeadsetNetworkState.NETWORK_PRESENT
            ),
        ),
        patch.object(adb, "_run_adb", return_value=""),
        patch.object(adb, "assert_headset_awake", side_effect=wake),
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    assert events == ["prepare", "link", "rebuild", "wake_unconfirmed"]
    payload = json.loads(
        next((tmp_path / "usb-health").glob("usb-health-*.json")).read_text()
    )
    assert payload["wakeCheck"]["confirmedAwake"] is False
    assert payload["transferTest"]["attempted"] is False
    assert payload["decision"]["overallOutcome"] == "incomplete"
    assert lifecycle.snapshot["usbWakeConfirmed"] is False


@pytest.mark.asyncio
async def test_transport_loss_invalidates_current_wake_but_preserves_history(tmp_path):
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        usb_health_logs_dir=tmp_path,
    )
    lifecycle.selected = "selected"
    lifecycle.browser_ready = True
    lifecycle._usb_wake_confirmed = True
    lifecycle._ensure_usb_report()
    lifecycle._persist_usb_health(
        wakeCheck={"confirmedAwake": True, "source": "adb-shell-dumpsys-power"},
        decision={"overallOutcome": "pass", "browserLaunchAllowed": True},
    )

    await lifecycle._enter_transport_recovery(
        "ADB disconnected", "WAITING_FOR_ADB", adb_ready=False, network_present=False
    )
    await lifecycle._remember_transport_loss()

    payload = json.loads(lifecycle._usb_health_report.path.read_text())
    assert lifecycle.snapshot["usbWakeConfirmed"] is None
    assert lifecycle._usb_wake_confirmed is None
    assert payload["wakeCheck"] == {
        "confirmedAwake": None,
        "outcome": "unknown_after_transport_loss",
    }
    assert payload["wakeHistory"] == [
        {"confirmedAwake": True, "source": "adb-shell-dumpsys-power"}
    ]
    assert payload["decision"]["overallOutcome"] == "incomplete"


@pytest.mark.parametrize(
    ("reconnected_speed", "decision"),
    [(480, "degraded"), (5000, "incomplete")],
)
@pytest.mark.asyncio
async def test_preserved_browser_reconnect_refreshes_link_without_active_transfer(
    tmp_path, monkeypatch, reconnected_speed, decision
):
    lifecycle = OobLifecycle(
        hub=_Hub(),
        resolved_port=48322,
        usb_local=True,
        host_client=True,
        turn_port=3478,
        config=RecoveryConfig(),
        usb_health_logs_dir=tmp_path,
    )
    lifecycle.selected = "selected"
    lifecycle._transport_lost = True
    lifecycle._restore_existing_browser = True
    lifecycle._usb_transfer = {
        "completed": True,
        "outcome": "pass",
        "method": "device_curl_static_asset_loop",
        "hostToHeadsetMbps": 300,
        "hostToHeadsetResponseBodyMbps": 900,
    }
    devices = adb.AdbDevices((("selected", "device"),))
    lifecycle._last_observation = (devices.devices, devices.diagnostic)
    lifecycle._ready_count = 1
    speeds = iter((5000, reconnected_speed))
    observed = []

    def inspect(_serial, _listing):
        speed = next(speeds)
        return {
            "probeSucceeded": True,
            "negotiatedSpeedMbps": speed,
            "isBetterPortAvailable": speed < 5000,
            "errors": [],
        }

    async def rebuild():
        observed.append(("rebuild", lifecycle._usb_link["negotiatedSpeedMbps"]))

    async def recover():
        raise asyncio.CancelledError

    async def forbidden_transfer():
        pytest.fail("preserved browser must not run active transfer")

    async def stop(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(usb_health, "inspect_link", inspect)
    monkeypatch.setattr(
        adb,
        "_adb_run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, "selected device usb:1-2.1 model:A9210\n", ""
        ),
    )
    lifecycle._rebuild_usb = rebuild
    lifecycle._recover_existing_browser = recover
    lifecycle._check_usb_transfer = forbidden_transfer
    lifecycle.sleep = stop
    await lifecycle._check_usb_link()
    with (
        patch.object(adb, "enumerate_adb_devices", return_value=devices),
        patch.object(
            adb,
            "probe_headset_network",
            return_value=adb.HeadsetNetworkProbe(
                adb.HeadsetNetworkState.NETWORK_PRESENT
            ),
        ),
        patch.object(adb, "_run_adb", return_value=""),
        patch.object(adb, "assert_headset_awake") as awake,
    ):
        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run()
    payload = json.loads(
        next((tmp_path / "usb-health").glob("usb-health-*.json")).read_text()
    )
    assert observed == [("rebuild", reconnected_speed)]
    assert [item["negotiatedSpeedMbps"] for item in payload["linkHistory"]] == [
        5000,
        reconnected_speed,
    ]
    assert payload["transferTest"]["staleAfterReconnect"] is True
    assert payload["transferTest"]["hostToHeadsetMbps"] is None
    assert payload["transferTest"]["outcome"] == "pending_retest"
    assert payload["transferHistory"][0]["hostToHeadsetMbps"] == 300
    assert payload["transferHistory"][0]["hostToHeadsetResponseBodyMbps"] == 900
    assert payload["decision"]["transferRetestDeferred"] is True
    assert payload["decision"]["overallOutcome"] == decision
    awake.assert_called_once_with(timeout=10.0)
    assert lifecycle.snapshot["usbLinkOutcome"] == (
        "pass" if reconnected_speed >= 5000 else "slow_link"
    )
    assert lifecycle.snapshot["usbTransferMbps"] is None
    assert lifecycle.snapshot["usbTransferResponseBodyMbps"] is None
    assert lifecycle.snapshot["usbTransferMethod"] is None
    assert lifecycle.snapshot["usbTransferOutcome"] == "stale"
