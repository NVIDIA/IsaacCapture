# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduction and caching for TeleopSession provider monitoring."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from .status import (
    DeviceState,
    DeviceStatus,
    ProviderState,
    ProviderStatus,
    ProviderType,
    StatusReason,
    StatusSnapshot,
)

SCHEMA_VERSION = 1
REPORT_STALE_AFTER_NS = 3_000_000_000
OPENXR_PROVIDER_ID = "openxr/runtime"
OPENXR_DEVICE_ID = "openxr/headset"
PLUGIN_STATUS_COLLECTION_SUFFIX = "/device_status"
OPENXR_IDENTIFIER_MAX_BYTES = 255

_PLUGIN_DEVICE_REASON_MAP = {
    "NONE": StatusReason.NONE,
    "NO_HARDWARE_SIGNAL": StatusReason.NO_HARDWARE_SIGNAL,
    "CALIBRATION_FAILED": StatusReason.CALIBRATION_FAILED,
    "NO_CURRENT_DATA": StatusReason.NO_CURRENT_DATA,
    "DEVICE_ERROR": StatusReason.DEVICE_ERROR,
    "DISABLED_BY_CONFIGURATION": StatusReason.DISABLED_BY_CONFIGURATION,
}

_VALID_PLUGIN_DEVICE_STATE_REASON_PAIRS = {
    ("UNKNOWN", "NO_HARDWARE_SIGNAL"),
    ("UNKNOWN", "NO_CURRENT_DATA"),
    ("CONNECTED", "NONE"),
    ("DISCONNECTED", "NO_HARDWARE_SIGNAL"),
    ("DEGRADED", "CALIBRATION_FAILED"),
    ("DEGRADED", "NO_CURRENT_DATA"),
    ("DEGRADED", "DEVICE_ERROR"),
    ("FAILED", "DEVICE_ERROR"),
    ("DISABLED", "DISABLED_BY_CONFIGURATION"),
}


@dataclass(frozen=True, slots=True)
class PluginProviderSpec:
    """Resolved plugin metadata and its internal status tracker."""

    plugin_root_id: str
    name: str
    devices: tuple[Any, ...]
    tracker: Any

    @property
    def provider_id(self) -> str:
        return f"plugin/{self.plugin_root_id}"


@dataclass(slots=True)
class _PluginRuntime:
    spec: PluginProviderSpec
    context: Any = None


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value).rsplit(".", 1)[-1]


def _unknown_devices(
    devices: tuple[DeviceStatus, ...],
    reason: StatusReason,
    error: str,
) -> tuple[DeviceStatus, ...]:
    return tuple(
        replace(
            device,
            status=DeviceState.UNKNOWN,
            reason=reason,
            error=error,
        )
        for device in devices
    )


def _provider_precedence(
    devices: tuple[DeviceStatus, ...],
    provider: ProviderStatus,
) -> tuple[DeviceStatus, ...] | None:
    if provider.status == ProviderState.INITIALIZING:
        return tuple(
            replace(
                device,
                status=DeviceState.INITIALIZING,
                reason=StatusReason.PROVIDER_INITIALIZING,
                error=provider.error,
            )
            for device in devices
        )
    if provider.status in (ProviderState.FAILED, ProviderState.STOPPED):
        reason = (
            StatusReason.PROVIDER_FAILED
            if provider.status == ProviderState.FAILED
            else StatusReason.PROVIDER_STOPPED
        )
        return tuple(
            replace(
                device,
                status=DeviceState.UNAVAILABLE,
                reason=reason,
                error=provider.error,
            )
            for device in devices
        )
    return None


def _reduce_report(
    devices: tuple[DeviceStatus, ...],
    report: Any,
    now_ns: int,
) -> tuple[DeviceStatus, ...]:
    if report is None:
        return _unknown_devices(devices, StatusReason.NO_REPORT, "")

    try:
        schema_version = report.schema_version
        if not isinstance(schema_version, int) or schema_version != SCHEMA_VERSION:
            return _unknown_devices(
                devices,
                StatusReason.UNSUPPORTED_SCHEMA_VERSION,
                f"unsupported plugin status schema version {schema_version!r}",
            )

        report_time_ns = report.report_time_ns
        if (
            not isinstance(report_time_ns, int)
            or report_time_ns < 0
            or report_time_ns > now_ns
        ):
            raise ValueError(f"invalid report_time_ns {report_time_ns!r}")
        age_ns = now_ns - report_time_ns
        if age_ns > REPORT_STALE_AFTER_NS:
            return _unknown_devices(
                devices,
                StatusReason.STALE_REPORT,
                f"plugin status report is stale by {age_ns} ns",
            )

        expected_paths = {device.path for device in devices}
        entries_by_path: dict[str, Any] = {}
        for entry in report.devices:
            path = entry.path
            if not isinstance(path, str) or not path:
                raise ValueError("plugin status report contains an empty device path")
            if path in entries_by_path:
                raise ValueError(
                    f"plugin status report contains duplicate device path {path!r}"
                )
            if path not in expected_paths:
                raise ValueError(
                    f"plugin status report contains unknown device path {path!r}"
                )
            entries_by_path[path] = entry

        missing_paths = expected_paths - entries_by_path.keys()
        if missing_paths:
            raise ValueError(
                "plugin status report is missing device paths "
                f"{sorted(missing_paths)!r}"
            )

        state_map = {
            "UNKNOWN": DeviceState.UNKNOWN,
            "CONNECTED": DeviceState.CONNECTED,
            "DISCONNECTED": DeviceState.DISCONNECTED,
            "DEGRADED": DeviceState.DEGRADED,
            "FAILED": DeviceState.FAILED,
            "DISABLED": DeviceState.DISABLED,
        }
        reduced = []
        for device in devices:
            entry = entries_by_path[device.path]
            state_name = _enum_name(entry.state)
            try:
                state = state_map[state_name]
            except KeyError as error:
                raise ValueError(
                    f"plugin status report contains invalid state {state_name!r}"
                ) from error

            reason_name = _enum_name(entry.reason)
            try:
                reason = _PLUGIN_DEVICE_REASON_MAP[reason_name]
            except KeyError as error:
                raise ValueError(
                    f"plugin status report contains invalid reason {reason_name!r}"
                ) from error

            if (
                state_name,
                reason_name,
            ) not in _VALID_PLUGIN_DEVICE_STATE_REASON_PAIRS:
                raise ValueError(
                    "plugin status report contains invalid state/reason pair "
                    f"{state_name}/{reason_name}"
                )

            entry_error = entry.error
            if not isinstance(entry_error, str):
                raise TypeError(
                    f"plugin status report error for {device.path!r} is not a string"
                )
            reduced.append(
                replace(
                    device,
                    status=state,
                    reason=reason,
                    error=entry_error,
                )
            )
        return tuple(reduced)
    except Exception as error:  # noqa: BLE001 - malformed bound objects are data
        return _unknown_devices(
            devices,
            StatusReason.MALFORMED_REPORT,
            str(error),
        )


def _reduce_process(
    provider: ProviderStatus,
    process: Any,
) -> ProviderStatus:
    state = _enum_name(process.state)
    pid = process.pid if isinstance(process.pid, int) and process.pid >= 0 else None
    if state == "RUNNING":
        return replace(
            provider,
            status=ProviderState.AVAILABLE,
            reason=StatusReason.NONE,
            error="",
            pid=pid,
            exit_code=None,
            term_signal=None,
        )
    if state == "STOPPED":
        return replace(
            provider,
            status=ProviderState.STOPPED,
            reason=StatusReason.PROVIDER_STOPPED,
            error=process.error,
            pid=pid,
            exit_code=process.exit_code,
            term_signal=process.term_signal,
        )

    process_reason = _enum_name(process.reason)
    details = process.error or process_reason
    if state == "EXITED":
        if process.exit_code is not None:
            details = f"process exited with code {process.exit_code}: {details}"
        reason = StatusReason.PROCESS_EXITED
    elif state == "SIGNALED":
        if process.term_signal is not None:
            details = f"process terminated by signal {process.term_signal}: {details}"
        reason = StatusReason.PROCESS_SIGNALED
    else:
        error_code = getattr(process, "error_code", None)
        if error_code is not None:
            details = f"process observation error {error_code}: {details}"
        reason = StatusReason.PROCESS_OBSERVATION_ERROR
    return replace(
        provider,
        status=ProviderState.FAILED,
        reason=reason,
        error=details,
        pid=pid,
        exit_code=process.exit_code,
        term_signal=process.term_signal,
    )


def _poll_process(provider: ProviderStatus, runtime: _PluginRuntime) -> ProviderStatus:
    try:
        return _reduce_process(provider, runtime.context.get_process_snapshot())
    except Exception as error:  # noqa: BLE001 - provider I/O is isolated
        return replace(
            provider,
            status=ProviderState.FAILED,
            reason=StatusReason.PROCESS_OBSERVATION_ERROR,
            error=str(error),
        )


def _reduce_openxr(
    provider: ProviderStatus,
    headset: DeviceStatus,
    snapshot: Any,
) -> tuple[ProviderStatus, DeviceStatus]:
    state = _enum_name(snapshot.state)
    headset_state = _enum_name(snapshot.headset_state)
    reason = _enum_name(snapshot.reason)
    error = snapshot.error
    if not isinstance(error, str):
        raise TypeError("OpenXR provider snapshot error is not a string")
    result_code = snapshot.result_code
    if result_code is not None and not isinstance(result_code, int):
        raise TypeError("OpenXR provider snapshot result_code is not an integer")

    reason_map = {
        "SESSION_LOST": StatusReason.OPENXR_SESSION_LOST,
        "INSTANCE_LOST": StatusReason.OPENXR_INSTANCE_LOST,
        "POLL_ERROR": StatusReason.OPENXR_POLL_ERROR,
    }
    if state == "FAILED":
        try:
            status_reason = reason_map[reason]
        except KeyError as exc:
            raise ValueError(
                f"failed OpenXR provider snapshot has invalid reason {reason!r}"
            ) from exc
        return (
            replace(
                provider,
                status=ProviderState.FAILED,
                reason=status_reason,
                error=error,
            ),
            replace(
                headset,
                status=DeviceState.UNAVAILABLE,
                reason=status_reason,
                error=error,
            ),
        )

    if state != "AVAILABLE":
        raise ValueError(f"OpenXR provider snapshot has invalid state {state!r}")
    if headset_state == "DISCONNECTED":
        if reason != "TRACKING_UNAVAILABLE":
            raise ValueError(
                f"disconnected OpenXR headset snapshot has invalid reason {reason!r}"
            )
        return (
            replace(
                provider,
                status=ProviderState.AVAILABLE,
                reason=StatusReason.NONE,
                error="",
            ),
            replace(
                headset,
                status=DeviceState.DISCONNECTED,
                reason=StatusReason.NO_CURRENT_DATA,
                error=error,
            ),
        )
    if headset_state != "CONNECTED" or reason != "NONE":
        raise ValueError("connected OpenXR headset snapshot must report AVAILABLE/NONE")
    return (
        replace(
            provider,
            status=ProviderState.AVAILABLE,
            reason=StatusReason.NONE,
            error="",
        ),
        replace(
            headset,
            status=DeviceState.CONNECTED,
            reason=StatusReason.OPENXR_SESSION_READY,
            error="",
        ),
    )


class StatusMonitor:
    """Owns status inventory and atomically cached immutable snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._update_lock = threading.Lock()
        self._snapshot = StatusSnapshot(0, (), ())
        self._providers_by_id: dict[str, ProviderStatus] = {}
        self._devices_by_id: dict[str, DeviceStatus] = {}
        self._plugin_runtimes: tuple[_PluginRuntime, ...] = ()
        self._external_openxr = False
        self._runtime_failed = False

    def get_status(self) -> StatusSnapshot:
        with self._lock:
            return self._snapshot

    def get_provider_status(self, provider_id: str) -> ProviderStatus | None:
        with self._lock:
            return self._providers_by_id.get(provider_id)

    def get_device_status(self, device_id: str) -> DeviceStatus | None:
        with self._lock:
            return self._devices_by_id.get(device_id)

    def rebuild(
        self,
        plugin_specs: tuple[PluginProviderSpec, ...],
        *,
        include_openxr: bool,
        external_openxr: bool,
        now_ns: int | None = None,
    ) -> None:
        """Validate and install a fresh INITIALIZING startup inventory."""
        providers: list[ProviderStatus] = []
        devices: list[DeviceStatus] = []
        provider_ids: set[str] = set()
        device_ids: set[str] = set()

        for spec in plugin_specs:
            if not spec.plugin_root_id:
                raise ValueError("plugin_root_id must not be empty")
            collection_id = f"{spec.plugin_root_id}{PLUGIN_STATUS_COLLECTION_SUFFIX}"
            if len(collection_id.encode("utf-8")) > OPENXR_IDENTIFIER_MAX_BYTES:
                raise ValueError(
                    "plugin status collection ID exceeds the OpenXR limit of "
                    f"{OPENXR_IDENTIFIER_MAX_BYTES} bytes: {collection_id!r}"
                )
            provider_id = spec.provider_id
            if not provider_id or provider_id in provider_ids:
                raise ValueError(
                    f"provider IDs must be nonempty and unique: {provider_id!r}"
                )
            provider_ids.add(provider_id)
            providers.append(
                ProviderStatus(
                    id=provider_id,
                    type=ProviderType.PLUGIN,
                    name=spec.name,
                    status=ProviderState.INITIALIZING,
                    reason=StatusReason.PROVIDER_INITIALIZING,
                    error="",
                )
            )
            manifest_paths: set[str] = set()
            for descriptor in spec.devices:
                path = descriptor.path
                if not isinstance(path, str) or not path:
                    raise ValueError("plugin manifest device paths must not be empty")
                if path in manifest_paths:
                    raise ValueError(
                        f"plugin manifest device paths must be unique: {path!r}"
                    )
                manifest_paths.add(path)
                device_id = f"{spec.plugin_root_id}{path}"
                if not device_id or device_id in device_ids:
                    raise ValueError(
                        f"device IDs must be nonempty and unique: {device_id!r}"
                    )
                device_ids.add(device_id)
                devices.append(
                    DeviceStatus(
                        id=device_id,
                        path=path,
                        type=descriptor.type,
                        description=descriptor.description,
                        provider_id=provider_id,
                        status=DeviceState.INITIALIZING,
                        reason=StatusReason.PROVIDER_INITIALIZING,
                        error="",
                    )
                )

        if include_openxr:
            if OPENXR_PROVIDER_ID in provider_ids or OPENXR_DEVICE_ID in device_ids:
                raise ValueError(
                    "OpenXR provider or device ID conflicts with plugin inventory"
                )
            providers.append(
                ProviderStatus(
                    id=OPENXR_PROVIDER_ID,
                    type=ProviderType.OPENXR_RUNTIME,
                    name="OpenXR Runtime",
                    status=ProviderState.INITIALIZING,
                    reason=StatusReason.PROVIDER_INITIALIZING,
                    error="",
                )
            )
            devices.append(
                DeviceStatus(
                    id=OPENXR_DEVICE_ID,
                    path="/headset",
                    type="headset",
                    description="OpenXR headset",
                    provider_id=OPENXR_PROVIDER_ID,
                    status=DeviceState.INITIALIZING,
                    reason=StatusReason.PROVIDER_INITIALIZING,
                    error="",
                )
            )

        self._plugin_runtimes = tuple(_PluginRuntime(spec) for spec in plugin_specs)
        self._external_openxr = external_openxr
        self._runtime_failed = False
        self._replace_snapshot(
            StatusSnapshot(
                time.monotonic_ns() if now_ns is None else now_ns,
                tuple(providers),
                tuple(devices),
            )
        )

    def bind_plugin_context(self, provider_id: str, context: Any) -> None:
        for runtime in self._plugin_runtimes:
            if runtime.spec.provider_id == provider_id:
                runtime.context = context
                return
        raise KeyError(f"unknown plugin provider {provider_id!r}")

    def mark_available(self, *, now_ns: int | None = None) -> None:
        snapshot = self.get_status()
        providers = []
        devices = []
        for provider in snapshot.providers:
            providers.append(
                replace(
                    provider,
                    status=ProviderState.AVAILABLE,
                    reason=StatusReason.NONE,
                    error="",
                )
            )
        for device in snapshot.devices:
            if device.provider_id == OPENXR_PROVIDER_ID:
                devices.append(
                    replace(
                        device,
                        status=(
                            DeviceState.UNKNOWN
                            if self._external_openxr
                            else DeviceState.CONNECTED
                        ),
                        reason=(
                            StatusReason.EXTERNAL_OPENXR_HEALTH_UNKNOWN
                            if self._external_openxr
                            else StatusReason.OPENXR_SESSION_READY
                        ),
                        error="",
                    )
                )
            else:
                devices.append(
                    replace(
                        device,
                        status=DeviceState.UNKNOWN,
                        reason=StatusReason.NO_REPORT,
                        error="",
                    )
                )
        self._replace_snapshot(
            StatusSnapshot(
                time.monotonic_ns() if now_ns is None else now_ns,
                tuple(providers),
                tuple(devices),
            )
        )

    def refresh(
        self,
        deviceio_session: Any,
        openxr_session: Any = None,
        *,
        now_ns: int | None = None,
    ) -> None:
        """Poll owned provider inputs and publish one reduced snapshot."""
        with self._update_lock:
            self._refresh(
                deviceio_session,
                openxr_session,
                now_ns=now_ns,
            )

    def _refresh(
        self,
        deviceio_session: Any,
        openxr_session: Any,
        *,
        now_ns: int | None,
    ) -> None:
        if self._runtime_failed:
            return
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        snapshot = self.get_status()
        providers_by_id = {provider.id: provider for provider in snapshot.providers}
        devices_by_provider: dict[str, tuple[DeviceStatus, ...]] = {
            provider.id: tuple(
                device
                for device in snapshot.devices
                if device.provider_id == provider.id
            )
            for provider in snapshot.providers
        }

        for runtime in self._plugin_runtimes:
            provider_id = runtime.spec.provider_id
            provider = _poll_process(providers_by_id[provider_id], runtime)
            providers_by_id[provider_id] = provider

            precedence_devices = _provider_precedence(
                devices_by_provider[provider_id], provider
            )
            if precedence_devices is not None:
                devices_by_provider[provider_id] = precedence_devices
                continue
            try:
                report = runtime.spec.tracker.get_device_status_snapshot(
                    deviceio_session
                )
            except Exception as error:  # noqa: BLE001 - tracker I/O is isolated
                devices_by_provider[provider_id] = _unknown_devices(
                    devices_by_provider[provider_id],
                    StatusReason.MALFORMED_REPORT,
                    str(error),
                )
            else:
                devices_by_provider[provider_id] = _reduce_report(
                    devices_by_provider[provider_id], report, now_ns
                )

        if OPENXR_PROVIDER_ID in providers_by_id and not self._external_openxr:
            provider = providers_by_id[OPENXR_PROVIDER_ID]
            headset = devices_by_provider[OPENXR_PROVIDER_ID][0]
            if provider.status != ProviderState.FAILED:
                try:
                    if openxr_session is None:
                        raise RuntimeError("owned OpenXR session is unavailable")
                    openxr_snapshot = openxr_session.get_provider_snapshot()
                    provider, headset = _reduce_openxr(
                        provider, headset, openxr_snapshot
                    )
                except Exception as error:  # noqa: BLE001 - provider I/O is isolated
                    provider = replace(
                        provider,
                        status=ProviderState.FAILED,
                        reason=StatusReason.OPENXR_POLL_ERROR,
                        error=str(error),
                    )
                    headset = replace(
                        headset,
                        status=DeviceState.UNAVAILABLE,
                        reason=StatusReason.OPENXR_POLL_ERROR,
                        error=str(error),
                    )
                providers_by_id[OPENXR_PROVIDER_ID] = provider
                devices_by_provider[OPENXR_PROVIDER_ID] = (headset,)

        providers = tuple(
            providers_by_id[provider.id] for provider in snapshot.providers
        )
        devices = tuple(
            device
            for provider in providers
            for device in devices_by_provider[provider.id]
        )
        self._replace_snapshot(StatusSnapshot(now_ns, providers, devices))

    def refresh_plugin_processes(self, *, now_ns: int | None = None) -> None:
        """Refresh process-owned states without polling device reports."""
        with self._update_lock:
            self._refresh_plugin_processes(now_ns=now_ns)

    def _refresh_plugin_processes(self, *, now_ns: int | None) -> None:
        if self._runtime_failed:
            return
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        snapshot = self.get_status()
        providers_by_id = {provider.id: provider for provider in snapshot.providers}
        devices_by_provider = {
            provider.id: tuple(
                device
                for device in snapshot.devices
                if device.provider_id == provider.id
            )
            for provider in snapshot.providers
        }

        for runtime in self._plugin_runtimes:
            provider_id = runtime.spec.provider_id
            provider = _poll_process(providers_by_id[provider_id], runtime)
            providers_by_id[provider_id] = provider
            precedence_devices = _provider_precedence(
                devices_by_provider[provider_id], provider
            )
            if precedence_devices is not None:
                devices_by_provider[provider_id] = precedence_devices

        providers = tuple(
            providers_by_id[provider.id] for provider in snapshot.providers
        )
        devices = tuple(
            device
            for provider in providers
            for device in devices_by_provider[provider.id]
        )
        self._replace_snapshot(StatusSnapshot(now_ns, providers, devices))

    def mark_startup_failed(
        self, error: BaseException | str, *, now_ns: int | None = None
    ) -> None:
        with self._update_lock:
            self._mark_terminal(
                ProviderState.FAILED,
                StatusReason.STARTUP_FAILED,
                str(error),
                now_ns=now_ns,
            )

    def mark_runtime_failed(
        self, error: BaseException | str, *, now_ns: int | None = None
    ) -> None:
        with self._update_lock:
            self._runtime_failed = True
            self._mark_terminal(
                ProviderState.FAILED,
                StatusReason.RUNTIME_UPDATE_FAILED,
                str(error),
                device_reason=StatusReason.RUNTIME_UPDATE_FAILED,
                now_ns=now_ns,
            )

    def mark_stopped(self, *, now_ns: int | None = None) -> None:
        with self._update_lock:
            self._mark_terminal(
                ProviderState.STOPPED,
                StatusReason.PROVIDER_STOPPED,
                "",
                preserve_failed=True,
                now_ns=now_ns,
            )

    def _mark_terminal(
        self,
        state: ProviderState,
        reason: StatusReason,
        error: str,
        *,
        device_reason: StatusReason | None = None,
        preserve_failed: bool = False,
        now_ns: int | None,
    ) -> None:
        snapshot = self.get_status()
        providers = tuple(
            (
                provider
                if preserve_failed and provider.status == ProviderState.FAILED
                else replace(provider, status=state, reason=reason, error=error)
            )
            for provider in snapshot.providers
        )
        preserved_provider_ids = {
            provider.id
            for provider in providers
            if preserve_failed and provider.status == ProviderState.FAILED
        }
        if device_reason is None:
            device_reason = (
                StatusReason.PROVIDER_FAILED
                if state == ProviderState.FAILED
                else StatusReason.PROVIDER_STOPPED
            )
        devices = tuple(
            (
                device
                if device.provider_id in preserved_provider_ids
                else replace(
                    device,
                    status=DeviceState.UNAVAILABLE,
                    reason=device_reason,
                    error=error,
                )
            )
            for device in snapshot.devices
        )
        self._replace_snapshot(
            StatusSnapshot(
                time.monotonic_ns() if now_ns is None else now_ns,
                providers,
                devices,
            )
        )

    def _replace_snapshot(self, snapshot: StatusSnapshot) -> None:
        providers_by_id = {provider.id: provider for provider in snapshot.providers}
        devices_by_id = {device.id: device for device in snapshot.devices}
        with self._lock:
            self._snapshot = snapshot
            self._providers_by_id = providers_by_id
            self._devices_by_id = devices_by_id
