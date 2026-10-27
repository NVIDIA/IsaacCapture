# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot-side pieces of ``SessionMode.REMOTE``: HELLO, STATUS and proxies for the server's plugins and headset.

The JSON shapes are the remote session protocol's; the Teleop Session Server reads the same ones.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any

PROTOCOL_VERSION = 1

# Stage named when the server never answered HELLO.
SERVER_DID_NOT_ANSWER = "server did not answer"
# How RemoteSession.start's RuntimeError starts when nothing accepted the connection
# (tcp_transport.cpp); TeleopSession retries it.
CONNECT_FAILURE_PREFIX = "cannot connect to "
# How a failed WebSocket upgrade of a ws:// endpoint starts (websocket_transport.cpp); a
# RemotePeerClosedError with this text is retried instead.
UPGRADE_FAILURE_PREFIX = "WebSocket upgrade to "
# Longest TCP connect attempt, so an unreachable host is retried instead of awaited.
CONNECT_ATTEMPT_S = 2.0
# Shortest one worth starting: a shorter one reports a refused connection as timed out.
CONNECT_ATTEMPT_MIN_S = 0.1
# Connect attempts that nothing accepted, or that the server closed unanswered, start at most
# this often.
CONNECT_RETRY_S = 0.5

# Hand-written trackers the server rebuilds from their type alone.
_PARAMETERLESS_TRACKERS = frozenset(
    {"HeadTracker", "HandTracker", "ControllerTracker", "FullBodyTracker"}
)
# Its payload times are on the server's clock; plugin health arrives through STATUS instead.
_PLUGIN_STATUS_TRACKER = "PluginDeviceStatusTracker"


def has_metadata(tracker: Any) -> bool:
    """Whether a tracker or tracker class has metadata, as ``TactileTracker`` does.

    Only generated trackers with metadata bind ``get_metadata``. A remote session does not carry
    them: the robot side reads a tracker's metadata once, before any sample arrives.
    """
    return hasattr(tracker, "get_metadata")


def reject_unsupported_inputs(sources: Iterable[Any], trackers: Iterable[Any]) -> None:
    """Raise ``ValueError`` for keyboard and message-channel inputs and trackers with metadata."""
    from isaaccapture.deviceio_trackers import KeyboardTracker, MessageChannelTracker
    from isaaccapture.retargeting_engine.deviceio_source_nodes import (
        KeyboardSource,
        MessageChannelSource,
    )

    for source in sources:
        if isinstance(source, KeyboardSource):
            raise ValueError(
                f"KeyboardSource {source.name!r} is not supported in SessionMode.REMOTE: "
                "its key events come from this machine, not from the Teleop Session Server"
            )
        if isinstance(source, MessageChannelSource):
            raise ValueError(
                f"MessageChannelSource {source.name!r} is not supported in "
                "SessionMode.REMOTE yet"
            )
        if has_metadata(source.get_tracker()):
            raise ValueError(
                f"{type(source).__name__} {source.name!r} is not supported in "
                "SessionMode.REMOTE yet: its tracker's metadata is not forwarded"
            )
    unsupported = (KeyboardTracker, MessageChannelTracker)
    for tracker in trackers:
        if isinstance(tracker, unsupported) or has_metadata(tracker):
            raise ValueError(
                f"{type(tracker).__name__} in TeleopSessionConfig.trackers is not "
                "supported in SessionMode.REMOTE"
            )


def collect_trackers(
    sources: Sequence[Any], sink_trackers: Iterable[Any], extra_trackers: Sequence[Any]
) -> list[tuple[Any, Any, str]]:
    """``(tracker, vendor, owner)`` for each tracker the server must read, deduplicated by identity.

    Source trackers come first, then ``extra_trackers`` (``TeleopSessionConfig.trackers``). A
    tracker only a sink uses, and plugin device-status trackers, are left out. ``vendor`` is the
    first owning source's ``get_vendor()``; ``owner`` names the tracker in error messages.
    """
    source_ids = {id(source.get_tracker()) for source in sources}
    skipped = {id(t) for t in sink_trackers if t is not None} - source_ids
    entries: dict[int, list[Any]] = {}
    candidates = [
        (source.get_tracker(), source.get_vendor(), f"source {source.name!r}")
        for source in sources
    ]
    candidates += [
        (tracker, None, f"TeleopSessionConfig.trackers[{index}]")
        for index, tracker in enumerate(extra_trackers)
    ]
    for tracker, vendor, owner in candidates:
        if tracker is None or id(tracker) in skipped:
            continue
        if _tracker_type(tracker, owner) == _PLUGIN_STATUS_TRACKER:
            continue
        entry = entries.setdefault(id(tracker), [tracker, vendor, owner])
        if entry[1] is None:
            entry[1] = vendor
    return [tuple(entry) for entry in entries.values()]


def describe_tracker(tracker: Any, name: str, vendor: Any, owner: str) -> dict:
    """The HELLO descriptor from which the server builds a matching tracker."""
    type_name = _tracker_type(tracker, owner)
    if type_name in _PARAMETERLESS_TRACKERS:
        params: dict[str, Any] = {}
    elif hasattr(tracker, "collection_id") and hasattr(tracker, "max_flatbuffer_size"):
        # A generated pull tracker (see deviceio_trackers/trackers.toml).
        params = {
            "collection_id": tracker.collection_id,
            "max_flatbuffer_size": int(tracker.max_flatbuffer_size),
        }
    else:
        raise ValueError(
            f"{type_name} ({owner}) is not supported over a remote session "
            "(SessionMode.REMOTE)"
        )
    return {
        "name": name,
        "type": type_name,
        "params": params,
        "vendor": None
        if vendor is None
        else {"id": vendor.id, "params": dict(vendor.params)},
    }


def plugin_requests(plugins: Iterable[Any]) -> list[dict]:
    """HELLO plugin requests for the enabled plugins; the server finds them by name."""
    return [
        {
            "plugin_name": plugin.plugin_name,
            "plugin_root_id": plugin.plugin_root_id,
            "plugin_args": [str(arg) for arg in plugin.plugin_args],
            "required": bool(plugin.required),
        }
        for plugin in plugins
        if plugin.enabled
    ]


def build_hello(
    *,
    app_name: str,
    tick_rate_hz: float,
    trackers: list[dict],
    topics: list[str],
    plugins: list[dict],
) -> str:
    """The HELLO JSON."""
    import isaaccapture

    return json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "isaaccapture_version": isaaccapture.__version__,
            "app_name": app_name,
            "tick_rate_hz": float(tick_rate_hz),
            "trackers": trackers,
            "topics": list(topics),
            "plugins": plugins,
        }
    )


def describe_stage(status: dict | None) -> str:
    """The stage of a STATUS, with its message, for a timeout error."""
    if not status:
        return "no status from the server yet"
    stage = str(status.get("stage") or "unknown stage")
    message = status.get("message")
    return f"{stage} ({message})" if message else stage


def not_ready_message(endpoint: str, stage: str, timeout_s: float | None) -> str:
    after = "" if timeout_s is None else f" after {timeout_s:g} s"
    return f"Teleop Session Server at {endpoint} was not ready{after}: {stage}"


def server_version(hello_ack: str) -> str:
    """The ``server_version`` of a HELLO_ACK, or ``"unknown"``."""
    version = (_parse_object(hello_ack) or {}).get("server_version")
    return version if isinstance(version, str) and version else "unknown"


class Deadline:
    """The end of ``timeout_s`` from now; ``None`` never ends."""

    def __init__(self, timeout_s: float | None) -> None:
        self.timeout_s = timeout_s
        self._end = None if timeout_s is None else time.monotonic() + timeout_s

    def remaining_s(self) -> float | None:
        return None if self._end is None else max(0.0, self._end - time.monotonic())

    def remaining_ms(self, cap_s: float | None = None) -> int:
        """Milliseconds left, at most ``cap_s``; -1 (no limit) when neither bounds it."""
        remaining = self.remaining_s()
        if cap_s is not None:
            remaining = cap_s if remaining is None else min(remaining, cap_s)
        return -1 if remaining is None else round(remaining * 1000)

    def expired(self) -> bool:
        return self._end is not None and time.monotonic() >= self._end


class RemoteStatus:
    """The server's newest STATUS (or READY), parsed again only when it changes. Thread-safe."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._text: str | None = None
        self._status: dict | None = None
        # Newest entry per (plugin_name, plugin_root_id) seen in any STATUS.
        self._plugins: dict[tuple[str, str], dict] = {}

    def latest(self) -> dict | None:
        text = self._session.latest_status()
        with self._lock:
            if text is not None and text != self._text:
                self._text = text
                self._status = _parse_object(text)
                for entry in (self._status or {}).get("plugins") or ():
                    if isinstance(entry, dict):
                        key = (
                            str(entry.get("plugin_name", "")),
                            str(entry.get("plugin_root_id", "")),
                        )
                        self._plugins[key] = entry
            return self._status

    def plugin_keys(self) -> list[tuple[str, str]]:
        self.latest()
        with self._lock:
            return list(self._plugins)

    def plugin(self, key: tuple[str, str]) -> dict | None:
        self.latest()
        with self._lock:
            return self._plugins.get(key)


class RemotePluginContext:
    """Stands in for a plugin the Teleop Session Server started; its state comes from STATUS."""

    def __init__(self, status: RemoteStatus, plugin_name: str, plugin_root_id: str):
        self._status = status
        self.plugin_name = plugin_name
        self.plugin_root_id = plugin_root_id

    def get_process_snapshot(self) -> SimpleNamespace:
        """The plugin's process state as the server last reported it."""
        entry = self._status.plugin((self.plugin_name, self.plugin_root_id))
        if entry is None:
            return SimpleNamespace(
                state="ERROR",
                reason="NONE",
                pid=None,
                exit_code=None,
                term_signal=None,
                error_code=None,
                error="not reported by the Teleop Session Server",
            )
        return SimpleNamespace(
            state=str(entry.get("state", "ERROR")),
            reason=str(entry.get("reason", "NONE")),
            pid=_optional_int(entry.get("pid")),
            exit_code=_optional_int(entry.get("exit_code")),
            term_signal=_optional_int(entry.get("term_signal")),
            error_code=_optional_int(entry.get("error_code")),
            error=str(entry.get("error") or ""),
        )

    def check_health(self) -> None:
        """Raise ``PluginCrashException`` unless the plugin is running on the server."""
        from isaaccapture.plugin_manager import PluginCrashException

        snapshot = self.get_process_snapshot()
        if snapshot.state != "RUNNING":
            detail = snapshot.error or f"process {snapshot.state} ({snapshot.reason})"
            raise PluginCrashException(
                f"Plugin {self.plugin_name!r} on the Teleop Session Server: {detail}"
            )


class NoDeviceReports:
    """Status tracker stand-in for a remote plugin: its devices are not reported to the robot."""

    def get_device_status_snapshot(self, _session: Any) -> None:
        return None


class RemoteOpenXRSession:
    """Stands in for the server's OpenXR session; its provider snapshot comes from STATUS."""

    def __init__(self, status: RemoteStatus) -> None:
        self._status = status

    def get_provider_snapshot(self) -> SimpleNamespace:
        """The newest STATUS headset entry as ``oxr.OpenXRSession.get_provider_snapshot()`` returns it.

        Raises ``ValueError`` when the entry is missing or malformed.
        """
        from isaaccapture import oxr

        entry = (self._status.latest() or {}).get("headset")
        try:
            state = oxr.OpenXRProviderState.__members__[entry["state"]]
            headset_state = oxr.OpenXRHeadsetState.__members__[entry["headset_state"]]
            reason = oxr.OpenXRProviderReason.__members__[entry["reason"]]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"STATUS from the Teleop Session Server has no valid headset entry: {entry!r}"
            ) from error
        healthy = (
            state == oxr.OpenXRProviderState.AVAILABLE
            and headset_state == oxr.OpenXRHeadsetState.CONNECTED
        )
        return SimpleNamespace(
            state=state,
            headset_state=headset_state,
            reason=reason,
            result_code=None,
            error=""
            if healthy
            else (
                f"Teleop Session Server reports OpenXR {state.name}, headset "
                f"{headset_state.name} ({reason.name})"
            ),
        )


def _tracker_type(tracker: Any, owner: str) -> str:
    get_name = getattr(tracker, "get_name", None)
    if not callable(get_name):
        raise ValueError(f"{owner} is not a DeviceIO tracker: {type(tracker).__name__}")
    return get_name()


def _parse_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
