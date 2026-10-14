# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused reducer tests for provider monitoring."""

import threading
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from isaaccapture.teleop_session_manager import (
    DeviceState,
    ProviderState,
    StatusReason,
)
from isaaccapture.teleop_session_manager.status_monitor import (
    OPENXR_DEVICE_ID,
    OPENXR_PROVIDER_ID,
    PluginProviderSpec,
    StatusMonitor,
)


class FakeTracker:
    def __init__(self, report=None):
        self.report = report

    def get_device_status_snapshot(self, _session):
        if isinstance(self.report, BaseException):
            raise self.report
        return self.report


class FakeContext:
    def __init__(self, state="RUNNING", **values):
        defaults = {
            "state": state,
            "reason": "NONE",
            "pid": 123,
            "exit_code": None,
            "term_signal": None,
            "error_code": None,
            "error": "",
        }
        defaults.update(values)
        self.process = SimpleNamespace(**defaults)

    def get_process_snapshot(self):
        if isinstance(self.process, BaseException):
            raise self.process
        return self.process


class FakeOpenXRSession:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.poll_count = 0

    def get_provider_snapshot(self):
        self.poll_count += 1
        if isinstance(self.snapshot, BaseException):
            raise self.snapshot
        return self.snapshot


def entry(path, state, reason="NONE", error=""):
    return SimpleNamespace(path=path, state=state, reason=reason, error=error)


def openxr_snapshot(
    state="AVAILABLE",
    headset_state="CONNECTED",
    reason="NONE",
    result_code=None,
    error="",
):
    return SimpleNamespace(
        state=state,
        headset_state=headset_state,
        reason=reason,
        result_code=result_code,
        error=error,
    )


def report(now_ns, entries, *, version=1, report_time_ns=None):
    return SimpleNamespace(
        schema_version=version,
        report_time_ns=now_ns if report_time_ns is None else report_time_ns,
        devices=entries,
    )


def make_monitor(*, entries=None, context=None, now_ns=10_000_000_000):
    tracker = FakeTracker(
        report(
            now_ns,
            [
                entry("/left", "CONNECTED"),
                entry("/right", "DISCONNECTED", "NO_HARDWARE_SIGNAL"),
            ],
        )
        if entries is None
        else entries
    )
    spec = PluginProviderSpec(
        plugin_root_id="/provider",
        name="provider",
        devices=(
            SimpleNamespace(path="/left", type="hand", description="Left"),
            SimpleNamespace(path="/right", type="hand", description="Right"),
        ),
        tracker=tracker,
    )
    monitor = StatusMonitor()
    monitor.rebuild(
        (spec,),
        include_openxr=False,
        external_openxr=False,
        now_ns=now_ns - 1,
    )
    monitor.bind_plugin_context(spec.provider_id, context or FakeContext())
    monitor.mark_available(now_ns=now_ns - 1)
    return monitor, tracker


@pytest.mark.parametrize(
    ("schema_state", "schema_reason", "expected_state", "expected_reason"),
    [
        (
            "UNKNOWN",
            "NO_HARDWARE_SIGNAL",
            DeviceState.UNKNOWN,
            StatusReason.NO_HARDWARE_SIGNAL,
        ),
        (
            "UNKNOWN",
            "NO_CURRENT_DATA",
            DeviceState.UNKNOWN,
            StatusReason.NO_CURRENT_DATA,
        ),
        ("CONNECTED", "NONE", DeviceState.CONNECTED, StatusReason.NONE),
        (
            "DISCONNECTED",
            "NO_HARDWARE_SIGNAL",
            DeviceState.DISCONNECTED,
            StatusReason.NO_HARDWARE_SIGNAL,
        ),
        (
            "DEGRADED",
            "NO_CURRENT_DATA",
            DeviceState.DEGRADED,
            StatusReason.NO_CURRENT_DATA,
        ),
        (
            "DEGRADED",
            "CALIBRATION_FAILED",
            DeviceState.DEGRADED,
            StatusReason.CALIBRATION_FAILED,
        ),
        (
            "DEGRADED",
            "DEVICE_ERROR",
            DeviceState.DEGRADED,
            StatusReason.DEVICE_ERROR,
        ),
        ("FAILED", "DEVICE_ERROR", DeviceState.FAILED, StatusReason.DEVICE_ERROR),
        (
            "DISABLED",
            "DISABLED_BY_CONFIGURATION",
            DeviceState.DISABLED,
            StatusReason.DISABLED_BY_CONFIGURATION,
        ),
    ],
)
def test_all_schema_device_states_are_mapped(
    schema_state,
    schema_reason,
    expected_state,
    expected_reason,
):
    now_ns = 10_000_000_000
    monitor, tracker = make_monitor(now_ns=now_ns)
    tracker.report = report(
        now_ns,
        [
            entry("/left", schema_state, schema_reason),
            entry("/right", "CONNECTED"),
        ],
    )

    monitor.refresh(object(), now_ns=now_ns)

    device = monitor.get_device_status("/provider/left")
    assert device.status == expected_state
    assert device.reason == expected_reason


@pytest.mark.parametrize(
    ("process", "provider_state", "provider_reason"),
    [
        (FakeContext("STOPPED"), ProviderState.STOPPED, StatusReason.PROVIDER_STOPPED),
        (
            FakeContext("EXITED", exit_code=0, reason="CLEAN_EXIT"),
            ProviderState.FAILED,
            StatusReason.PROCESS_EXITED,
        ),
        (
            FakeContext("SIGNALED", term_signal=9, reason="SIGNAL"),
            ProviderState.FAILED,
            StatusReason.PROCESS_SIGNALED,
        ),
        (
            FakeContext("ERROR", error_code=5, reason="WAIT_ERROR"),
            ProviderState.FAILED,
            StatusReason.PROCESS_OBSERVATION_ERROR,
        ),
    ],
)
def test_provider_precedence_overrides_reports(
    process, provider_state, provider_reason
):
    monitor, _tracker = make_monitor(context=process)

    monitor.refresh(object(), now_ns=10_000_000_000)

    assert monitor.get_status().providers[0].status == provider_state
    assert monitor.get_status().providers[0].reason == provider_reason
    assert {device.status for device in monitor.get_status().devices} == {
        DeviceState.UNAVAILABLE
    }


@pytest.mark.parametrize(
    ("bad_report", "reason"),
    [
        (None, StatusReason.NO_REPORT),
        (
            report(10_000_000_000, [], version=2),
            StatusReason.UNSUPPORTED_SCHEMA_VERSION,
        ),
        (
            report(
                10_000_000_000,
                [],
                report_time_ns=6_999_999_999,
            ),
            StatusReason.STALE_REPORT,
        ),
        (
            report(
                10_000_000_000,
                [
                    entry("/left", "CONNECTED"),
                    entry("/unknown", "CONNECTED"),
                ],
            ),
            StatusReason.MALFORMED_REPORT,
        ),
        (
            report(
                10_000_000_000,
                [
                    entry("/left", "CONNECTED"),
                    entry("/left", "CONNECTED"),
                ],
            ),
            StatusReason.MALFORMED_REPORT,
        ),
        (
            report(
                10_000_000_000,
                [entry("/left", "CONNECTED")],
            ),
            StatusReason.MALFORMED_REPORT,
        ),
        (
            report(
                10_000_000_000,
                [
                    entry("/left", "CONNECTED", "CALIBRATION_FAILED"),
                    entry("/right", "CONNECTED"),
                ],
            ),
            StatusReason.MALFORMED_REPORT,
        ),
        (RuntimeError("transport failed"), StatusReason.MALFORMED_REPORT),
    ],
)
def test_invalid_reports_make_the_complete_provider_unknown(bad_report, reason):
    monitor, tracker = make_monitor(entries=bad_report)
    tracker.report = bad_report

    monitor.refresh(object(), now_ns=10_000_000_000)

    assert {device.status for device in monitor.get_status().devices} == {
        DeviceState.UNKNOWN
    }
    assert {device.reason for device in monitor.get_status().devices} == {reason}


def test_valid_report_recovers_after_malformed_report():
    now_ns = 10_000_000_000
    monitor, tracker = make_monitor(entries=None, now_ns=now_ns)
    tracker.report = report(now_ns, [entry("/left", "CONNECTED")])
    monitor.refresh(object(), now_ns=now_ns)
    assert monitor.get_status().devices[0].reason == StatusReason.MALFORMED_REPORT

    tracker.report = report(
        now_ns,
        [
            entry(
                "/left",
                "DEGRADED",
                "NO_CURRENT_DATA",
                error="waiting for skeleton",
            ),
            entry("/right", "CONNECTED"),
        ],
    )
    monitor.refresh(object(), now_ns=now_ns)

    assert [device.status for device in monitor.get_status().devices] == [
        DeviceState.DEGRADED,
        DeviceState.CONNECTED,
    ]
    assert [device.reason for device in monitor.get_status().devices] == [
        StatusReason.NO_CURRENT_DATA,
        StatusReason.NONE,
    ]
    assert monitor.get_status().devices[0].error == "waiting for skeleton"


@pytest.mark.parametrize(
    ("external", "device_state", "reason"),
    [
        (False, DeviceState.CONNECTED, StatusReason.OPENXR_SESSION_READY),
        (
            True,
            DeviceState.UNKNOWN,
            StatusReason.EXTERNAL_OPENXR_HEALTH_UNKNOWN,
        ),
    ],
)
def test_openxr_owned_and_external_states(external, device_state, reason):
    monitor = StatusMonitor()
    monitor.rebuild(
        (),
        include_openxr=True,
        external_openxr=external,
        now_ns=1,
    )
    monitor.mark_available(now_ns=2)

    assert (
        monitor.get_provider_status(OPENXR_PROVIDER_ID).status
        == ProviderState.AVAILABLE
    )
    assert monitor.get_device_status(OPENXR_DEVICE_ID).status == device_state
    assert monitor.get_device_status(OPENXR_DEVICE_ID).reason == reason


@pytest.mark.parametrize(
    (
        "native_snapshot",
        "provider_state",
        "device_state",
        "status_reason",
    ),
    [
        (
            openxr_snapshot(),
            ProviderState.AVAILABLE,
            DeviceState.CONNECTED,
            StatusReason.OPENXR_SESSION_READY,
        ),
        (
            openxr_snapshot(
                headset_state="DISCONNECTED",
                reason="TRACKING_UNAVAILABLE",
                error="head tracking unavailable",
            ),
            ProviderState.AVAILABLE,
            DeviceState.DISCONNECTED,
            StatusReason.NO_CURRENT_DATA,
        ),
        (
            openxr_snapshot(
                state="FAILED",
                reason="SESSION_LOST",
                error="session lost",
            ),
            ProviderState.FAILED,
            DeviceState.UNAVAILABLE,
            StatusReason.OPENXR_SESSION_LOST,
        ),
        (
            openxr_snapshot(
                state="FAILED",
                reason="INSTANCE_LOST",
                error="instance lost",
            ),
            ProviderState.FAILED,
            DeviceState.UNAVAILABLE,
            StatusReason.OPENXR_INSTANCE_LOST,
        ),
        (
            openxr_snapshot(
                state="FAILED",
                reason="POLL_ERROR",
                result_code=-1,
                error="poll failed",
            ),
            ProviderState.FAILED,
            DeviceState.UNAVAILABLE,
            StatusReason.OPENXR_POLL_ERROR,
        ),
    ],
)
def test_owned_openxr_snapshot_mapping(
    native_snapshot,
    provider_state,
    device_state,
    status_reason,
):
    monitor = StatusMonitor()
    monitor.rebuild(
        (),
        include_openxr=True,
        external_openxr=False,
        now_ns=1,
    )
    monitor.mark_available(now_ns=2)
    openxr_session = FakeOpenXRSession(native_snapshot)

    monitor.refresh(object(), openxr_session, now_ns=3)

    provider = monitor.get_provider_status(OPENXR_PROVIDER_ID)
    device = monitor.get_device_status(OPENXR_DEVICE_ID)
    assert provider.status == provider_state
    assert device.status == device_state
    assert device.reason == status_reason
    if provider_state == ProviderState.FAILED:
        assert provider.reason == status_reason


def test_failed_owned_openxr_state_is_terminal_without_more_polling():
    monitor = StatusMonitor()
    monitor.rebuild(
        (),
        include_openxr=True,
        external_openxr=False,
        now_ns=1,
    )
    monitor.mark_available(now_ns=2)
    openxr_session = FakeOpenXRSession(
        openxr_snapshot(state="FAILED", reason="SESSION_LOST")
    )
    monitor.refresh(object(), openxr_session, now_ns=3)

    openxr_session.snapshot = openxr_snapshot()
    monitor.refresh(object(), openxr_session, now_ns=4)

    assert openxr_session.poll_count == 1
    assert (
        monitor.get_provider_status(OPENXR_PROVIDER_ID).reason
        == StatusReason.OPENXR_SESSION_LOST
    )


def test_owned_openxr_headset_reconnects_while_provider_remains_available():
    monitor = StatusMonitor()
    monitor.rebuild(
        (),
        include_openxr=True,
        external_openxr=False,
        now_ns=1,
    )
    monitor.mark_available(now_ns=2)
    openxr_session = FakeOpenXRSession(
        openxr_snapshot(
            headset_state="DISCONNECTED",
            reason="TRACKING_UNAVAILABLE",
        )
    )
    monitor.refresh(object(), openxr_session, now_ns=3)

    openxr_session.snapshot = openxr_snapshot()
    monitor.refresh(object(), openxr_session, now_ns=4)

    assert (
        monitor.get_provider_status(OPENXR_PROVIDER_ID).status
        == ProviderState.AVAILABLE
    )
    assert monitor.get_device_status(OPENXR_DEVICE_ID).status == DeviceState.CONNECTED


def test_external_openxr_is_not_polled():
    monitor = StatusMonitor()
    monitor.rebuild(
        (),
        include_openxr=True,
        external_openxr=True,
        now_ns=1,
    )
    monitor.mark_available(now_ns=2)
    openxr_session = FakeOpenXRSession(RuntimeError("must not poll"))

    monitor.refresh(object(), openxr_session, now_ns=3)

    assert openxr_session.poll_count == 0
    assert (
        monitor.get_device_status(OPENXR_DEVICE_ID).reason
        == StatusReason.EXTERNAL_OPENXR_HEALTH_UNKNOWN
    )


def test_owned_openxr_poll_exception_is_cached_as_terminal_failure():
    monitor = StatusMonitor()
    monitor.rebuild(
        (),
        include_openxr=True,
        external_openxr=False,
        now_ns=1,
    )
    monitor.mark_available(now_ns=2)
    openxr_session = FakeOpenXRSession(RuntimeError("poll exploded"))

    monitor.refresh(object(), openxr_session, now_ns=3)

    assert (
        monitor.get_provider_status(OPENXR_PROVIDER_ID).reason
        == StatusReason.OPENXR_POLL_ERROR
    )
    assert monitor.get_device_status(OPENXR_DEVICE_ID).status == DeviceState.UNAVAILABLE


def test_old_snapshots_remain_immutable_and_unknown_ids_return_none():
    monitor, _tracker = make_monitor()
    old_snapshot = monitor.get_status()

    monitor.refresh(object(), now_ns=10_000_000_000)

    assert old_snapshot is not monitor.get_status()
    assert old_snapshot.devices[0].status == DeviceState.UNKNOWN
    with pytest.raises(FrozenInstanceError):
        old_snapshot.devices[0].error = "changed"
    assert monitor.get_provider_status("missing") is None
    assert monitor.get_device_status("missing") is None


def test_runtime_failure_wins_over_an_in_flight_process_refresh():
    refresh_entered = threading.Event()
    release_refresh = threading.Event()
    failure_started = threading.Event()
    failure_finished = threading.Event()

    class BlockingContext(FakeContext):
        def get_process_snapshot(self):
            refresh_entered.set()
            assert release_refresh.wait(timeout=1)
            return self.process

    monitor, _tracker = make_monitor(context=BlockingContext())
    provider_id = monitor.get_status().providers[0].id
    refresh_thread = threading.Thread(target=monitor.refresh_plugin_processes)
    refresh_thread.start()
    assert refresh_entered.wait(timeout=1)

    def mark_failed():
        failure_started.set()
        monitor.mark_runtime_failed("device update failed")
        failure_finished.set()

    failure_thread = threading.Thread(target=mark_failed)
    failure_thread.start()
    assert failure_started.wait(timeout=1)
    failure_finished.wait(timeout=0.1)
    release_refresh.set()

    refresh_thread.join(timeout=1)
    failure_thread.join(timeout=1)
    assert not refresh_thread.is_alive()
    assert not failure_thread.is_alive()
    assert monitor.get_provider_status(provider_id).status == ProviderState.FAILED
    assert (
        monitor.get_provider_status(provider_id).reason
        == StatusReason.RUNTIME_UPDATE_FAILED
    )


def test_teardown_preserves_existing_provider_failure():
    monitor, _tracker = make_monitor()
    provider_id = monitor.get_status().providers[0].id
    monitor.mark_runtime_failed("device update failed", now_ns=11)
    failed_provider = monitor.get_provider_status(provider_id)
    failed_device = monitor.get_device_status("/provider/left")

    monitor.mark_stopped(now_ns=12)

    assert monitor.get_provider_status(provider_id) == failed_provider
    assert monitor.get_device_status("/provider/left") == failed_device


def test_manifest_paths_and_ids_must_be_unique():
    descriptor = SimpleNamespace(path="/same", type="device", description="")
    specs = (
        PluginProviderSpec("/root", "one", (descriptor,), FakeTracker()),
        PluginProviderSpec("/root", "two", (descriptor,), FakeTracker()),
    )

    with pytest.raises(ValueError, match="provider IDs"):
        StatusMonitor().rebuild(
            specs,
            include_openxr=False,
            external_openxr=False,
        )


def test_plugin_status_collection_id_must_fit_openxr_identifier():
    descriptor = SimpleNamespace(path="/device", type="device", description="")
    spec = PluginProviderSpec(
        "a" * 242,
        "provider",
        (descriptor,),
        FakeTracker(),
    )

    with pytest.raises(ValueError, match="OpenXR limit"):
        StatusMonitor().rebuild(
            (spec,),
            include_openxr=False,
            external_openxr=False,
        )
