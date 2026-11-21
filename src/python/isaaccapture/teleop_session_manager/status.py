# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable public status contracts for :class:`TeleopSession`."""

from dataclasses import dataclass
from enum import Enum


class ProviderType(str, Enum):
    """Kind of runtime component that owns devices."""

    PLUGIN = "plugin"
    OPENXR_RUNTIME = "openxr_runtime"


class ProviderState(str, Enum):
    """Current provider lifecycle state."""

    INITIALIZING = "initializing"
    AVAILABLE = "available"
    FAILED = "failed"
    STOPPED = "stopped"


class DeviceState(str, Enum):
    """Current normalized device state."""

    INITIALIZING = "initializing"
    UNKNOWN = "unknown"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class StatusReason(str, Enum):
    """Typed explanation for a provider or device state."""

    NONE = "none"
    PROVIDER_INITIALIZING = "provider_initializing"
    PROVIDER_FAILED = "provider_failed"
    PROVIDER_STOPPED = "provider_stopped"
    NO_REPORT = "no_report"
    STALE_REPORT = "stale_report"
    MALFORMED_REPORT = "malformed_report"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    NO_HARDWARE_SIGNAL = "no_hardware_signal"
    CALIBRATION_FAILED = "calibration_failed"
    NO_CURRENT_DATA = "no_current_data"
    DEVICE_ERROR = "device_error"
    DISABLED_BY_CONFIGURATION = "disabled_by_configuration"
    PROCESS_EXITED = "process_exited"
    PROCESS_SIGNALED = "process_signaled"
    PROCESS_OBSERVATION_ERROR = "process_observation_error"
    STARTUP_FAILED = "startup_failed"
    RUNTIME_UPDATE_FAILED = "runtime_update_failed"
    OPENXR_SESSION_READY = "openxr_session_ready"
    EXTERNAL_OPENXR_HEALTH_UNKNOWN = "external_openxr_health_unknown"
    OPENXR_SESSION_LOST = "openxr_session_lost"
    OPENXR_INSTANCE_LOST = "openxr_instance_lost"
    OPENXR_POLL_ERROR = "openxr_poll_error"


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Current state of one device provider."""

    id: str
    type: ProviderType
    name: str
    status: ProviderState
    reason: StatusReason
    error: str
    pid: int | None = None
    exit_code: int | None = None
    term_signal: int | None = None


@dataclass(frozen=True, slots=True)
class DeviceStatus:
    """Current state of one statically declared device."""

    id: str
    path: str
    type: str
    description: str
    provider_id: str
    status: DeviceState
    reason: StatusReason
    error: str


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Atomic point-in-time view of all providers and devices."""

    updated_at_ns: int
    providers: tuple[ProviderStatus, ...]
    devices: tuple[DeviceStatus, ...]
