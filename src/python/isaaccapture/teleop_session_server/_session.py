# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session process of the Teleop Session Server: serves one robot app on an accepted connection.

The main process starts one per connection, in its own process group. Startup follows
``TeleopSession``'s local order: OpenXR session, trackers, plugins, then READY once the headset has
stayed connected. The process ends with the connection.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
import uuid
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaaccapture
import isaaccapture.plugin_manager as pm
from isaaccapture import deviceio_trackers, oxr
from isaaccapture.deviceio_session import (
    DeviceIOSession,
    RemoteSessionHost,
    TrackerVendor,
    VendorConfig,
)
from isaaccapture.deviceio_trackers._generated_tracker_exports import (
    __all__ as _GENERATED_TRACKERS,
)
from isaaccapture.logging_config._native_api import capture_native_output
from isaaccapture.teleop_session_manager._remote import PROTOCOL_VERSION, has_metadata

from ._common import SHUTDOWN_MESSAGE, configure_logging

logger = logging.getLogger("isaaccapture.teleop_session_server.session")

HELLO_TIMEOUT_MS = 5000
HELLO_POLL_MS = 100
OPENXR_RETRY_S = 1.0
# How long the headset must stay connected before READY.
READY_AFTER_CONNECTED_S = 1.1
STATUS_PERIOD_S = 1.0
# The teardown's time besides stopping the plugins: the host's graceful close (up to 1 s), the
# OpenXR session and the process exit.
TEARDOWN_MARGIN_S = 3.0
MIN_TICK_RATE_HZ = 1.0
MAX_TICK_RATE_HZ = 240.0
DEFAULT_APP_NAME = "TeleopSessionServer"
PLACEHOLDER_HANDLES = (1, 1, 1, 1)
WAITING_FOR_HEADSET = "waiting for the headset to connect to CloudXR"

# The one xrGetSystem failure that means "no headset yet"; OpenXRSession's own wait retries only it.
_NO_HEADSET_ERROR = "Failed to get OpenXR system: XR_ERROR_FORM_FACTOR_UNAVAILABLE"
# Before the OpenXR session exists; the robot app reads the headset entry from READY on.
_NO_HEADSET = {"state": "", "headset_state": "", "reason": ""}
# --placeholder-openxr has no OpenXR runtime, so no headset is ever tracked: what an OpenXR
# session reports without head tracking, which the robot app shows as headset DISCONNECTED.
_UNTRACKED_HEADSET = {
    "state": "AVAILABLE",
    "headset_state": "DISCONNECTED",
    "reason": "TRACKING_UNAVAILABLE",
}

_PARAMETERLESS_TRACKERS = {
    cls.__name__: cls
    for cls in (
        deviceio_trackers.HeadTracker,
        deviceio_trackers.HandTracker,
        deviceio_trackers.ControllerTracker,
        deviceio_trackers.FullBodyTracker,
    )
}
_UNSUPPORTED_TRACKERS = {
    "KeyboardTracker": "it would read the Teleop Session Server's keyboard",
    "MessageChannelTracker": "message channels are not forwarded yet",
    "TensorPushTracker": "sink data is not forwarded yet",
    "HapticCommandPushTracker": "sink data is not forwarded yet",
    "HapticCommandReaderTracker": "it reads sink data, which is not forwarded yet",
    "PluginDeviceStatusTracker": "plugin health is reported in STATUS instead",
    **{
        name: "its metadata is not forwarded yet"
        for name in _GENERATED_TRACKERS
        if has_metadata(getattr(deviceio_trackers, name))
    },
}


def _generated_pull_trackers() -> dict[str, type]:
    """Generated trackers that read a collection (``collection_id``, ``max_flatbuffer_size``)."""
    classes = {}
    for name in _GENERATED_TRACKERS:
        cls = getattr(deviceio_trackers, name)
        if hasattr(cls, "push") or name in _UNSUPPORTED_TRACKERS:
            continue
        if all(
            isinstance(getattr(cls, attribute, None), property)
            for attribute in ("collection_id", "max_flatbuffer_size")
        ):
            classes[name] = cls
    return classes


_GENERATED_PULL_TRACKERS = _generated_pull_trackers()


class HelloError(ValueError):
    """The robot app's HELLO cannot be served; its text goes back to the robot app as ERROR."""


@dataclass
class PluginRequest:
    plugin_name: str
    plugin_root_id: str
    plugin_args: list[str]
    required: bool


@dataclass
class SessionRequest:
    """A valid HELLO, with its trackers built."""

    app_name: str
    isaaccapture_version: str
    tick_rate_hz: float
    trackers: list[tuple[Any, str]]
    vendors: list[tuple[Any, Any]]
    topics: list[str]
    # Only the plugins this server found, in request order.
    plugins: list[PluginRequest]


def parse_hello(
    text: str,
    *,
    placeholder: bool,
    plugin_manager: Any,
    plugin_dirs: Sequence[Path],
) -> SessionRequest:
    """Validate a HELLO and build its trackers; raise :class:`HelloError` with the reason."""
    try:
        hello = json.loads(text)
    except ValueError as error:
        raise HelloError(f"HELLO is not valid JSON: {error}") from None
    if not isinstance(hello, dict):
        raise HelloError("HELLO is not a JSON object")
    protocol = hello.get("protocol")
    if _is_bool(protocol) or protocol != PROTOCOL_VERSION:
        raise HelloError(
            f"Remote session protocol {protocol!r} is not supported: this Teleop Session Server "
            f"(Isaac Capture {isaaccapture.__version__}) speaks protocol {PROTOCOL_VERSION}"
        )

    app_name = hello.get("app_name", DEFAULT_APP_NAME)
    if not isinstance(app_name, str):
        raise HelloError(f"app_name must be a string, got {app_name!r}")
    version = hello.get("isaaccapture_version", "")
    rate = hello.get("tick_rate_hz", 90.0)
    if (
        _is_bool(rate)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise HelloError(f"tick_rate_hz must be a number above zero, got {rate!r}")

    descriptors = _list(hello, "trackers")
    trackers: list[tuple[Any, str]] = []
    vendors: list[tuple[Any, Any]] = []
    for index, descriptor in enumerate(descriptors):
        tracker, name, vendor = _build_tracker(index, descriptor, placeholder)
        if any(name == existing for _, existing in trackers):
            raise HelloError(f"Tracker name {name!r} is used twice")
        trackers.append((tracker, name))
        if vendor is not None:
            vendors.append((tracker, vendor))

    topics = _list(hello, "topics")
    if not all(isinstance(topic, str) for topic in topics):
        raise HelloError("topics must be a list of strings")

    return SessionRequest(
        app_name=app_name or DEFAULT_APP_NAME,
        isaaccapture_version=str(version),
        tick_rate_hz=min(max(float(rate), MIN_TICK_RATE_HZ), MAX_TICK_RATE_HZ),
        trackers=trackers,
        vendors=vendors,
        topics=list(topics),
        plugins=_plugin_requests(_list(hello, "plugins"), plugin_manager, plugin_dirs),
    )


def _build_tracker(index: int, descriptor: Any, placeholder: bool) -> tuple:
    """``(tracker, name, TrackerVendor | None)`` for one HELLO tracker descriptor."""
    if not isinstance(descriptor, dict):
        raise HelloError(f"trackers[{index}] is not a JSON object")
    name = descriptor.get("name")
    type_name = descriptor.get("type")
    params = descriptor.get("params") or {}
    if not isinstance(name, str) or not name:
        raise HelloError(f"trackers[{index}] has no name")
    if not isinstance(type_name, str) or not isinstance(params, dict):
        raise HelloError(f"Tracker {name!r} needs a type string and a params object")
    what = f"{type_name} (tracker {name!r})"

    if placeholder:
        # Every other tracker needs OpenXR (see --placeholder-openxr).
        if type_name != "KeyboardTracker":
            raise HelloError(
                "This Teleop Session Server runs with --placeholder-openxr and serves only "
                f"KeyboardTracker, not {what}"
            )
        cls, args = deviceio_trackers.KeyboardTracker, _no_params(what, params)
    elif type_name in _PARAMETERLESS_TRACKERS:
        cls, args = _PARAMETERLESS_TRACKERS[type_name], _no_params(what, params)
    elif type_name in _GENERATED_PULL_TRACKERS:
        cls, args = (
            _GENERATED_PULL_TRACKERS[type_name],
            _collection_params(what, params),
        )
    elif type_name in _UNSUPPORTED_TRACKERS:
        raise HelloError(
            f"{what} is not supported over a remote session: "
            f"{_UNSUPPORTED_TRACKERS[type_name]}"
        )
    else:
        raise HelloError(
            f"{what} is unknown to this Teleop Session Server "
            f"(Isaac Capture {isaaccapture.__version__})"
        )
    try:
        tracker = cls(*args)
    except (TypeError, ValueError, RuntimeError) as error:
        raise HelloError(f"{what} could not be created: {error}") from None
    return tracker, name, _vendor(what, descriptor.get("vendor"))


def _no_params(what: str, params: dict) -> tuple:
    if params:
        raise HelloError(f"{what} takes no params, got {sorted(params)!r}")
    return ()


def _collection_params(what: str, params: dict) -> tuple:
    unknown = sorted(set(params) - {"collection_id", "max_flatbuffer_size"})
    if unknown:
        raise HelloError(f"{what} has unknown params {unknown!r}")
    collection_id = params.get("collection_id")
    if not isinstance(collection_id, str) or not collection_id:
        raise HelloError(f"{what} needs a collection_id")
    if "max_flatbuffer_size" not in params:
        return (collection_id,)
    size = params["max_flatbuffer_size"]
    if _is_bool(size) or not isinstance(size, int) or size <= 0:
        raise HelloError(f"{what} max_flatbuffer_size must be a positive integer")
    return collection_id, size


def _vendor(what: str, vendor: Any) -> Any:
    if vendor is None:
        return None
    if not isinstance(vendor, dict) or not isinstance(vendor.get("id"), str):
        raise HelloError(f"{what} vendor needs an id string")
    params = vendor.get("params") or {}
    if not isinstance(params, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in params.items()
    ):
        raise HelloError(f"{what} vendor params must map strings to strings")
    return TrackerVendor(vendor["id"], params)


def _plugin_requests(
    entries: list, plugin_manager: Any, plugin_dirs: Sequence[Path]
) -> list[PluginRequest]:
    """The requested plugins this server has; a missing required one is a HelloError."""
    discovered = sorted(plugin_manager.get_plugin_names())
    searched = [str(path) for path in plugin_dirs]
    requests = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise HelloError(f"plugins[{index}] is not a JSON object")
        name = entry.get("plugin_name")
        root_id = entry.get("plugin_root_id", "")
        args = entry.get("plugin_args") or []
        required = entry.get("required", False)
        if not isinstance(name, str) or not name or not isinstance(root_id, str):
            raise HelloError(f"plugins[{index}] needs a plugin_name and plugin_root_id")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise HelloError(f"Plugin {name!r} plugin_args must be a list of strings")
        if not _is_bool(required):
            raise HelloError(f"Plugin {name!r} required must be true or false")
        if name not in discovered:
            if required:
                # Same leading words as TeleopSession's local error; CI greps for them.
                raise HelloError(
                    f"Required plugin {name!r} was not discovered on the Teleop Session "
                    f"Server in {searched!r}. Discovered plugins: {discovered!r}. Install "
                    "the plugin there, or start the server with --plugin-dir."
                )
            logger.info(
                "Optional plugin %r was not discovered in %s; not starting it",
                name,
                searched,
            )
            continue
        requests.append(PluginRequest(name, root_id, list(args), required))
    return requests


def status_json(
    stage: str,
    message: str = "",
    plugins: Sequence[dict] = (),
    headset: dict | None = None,
) -> str:
    """The STATUS / READY JSON."""
    return json.dumps(
        {
            "stage": stage,
            "message": message,
            "plugins": list(plugins),
            "headset": headset or _NO_HEADSET,
        }
    )


def plugin_status(request: PluginRequest, context: Any) -> dict:
    """One STATUS plugin entry from the plugin's process snapshot."""
    entry: dict[str, Any] = {
        "plugin_name": request.plugin_name,
        "plugin_root_id": request.plugin_root_id,
    }
    try:
        snapshot = context.get_process_snapshot()
    except Exception as error:  # noqa: BLE001 - reported to the robot app as a failure
        return entry | {
            "state": "ERROR",
            "reason": "NONE",
            "pid": None,
            "exit_code": None,
            "term_signal": None,
            "error_code": None,
            "error": str(error),
        }
    return entry | {
        "state": snapshot.state.name,
        "reason": snapshot.reason.name,
        "pid": snapshot.pid,
        "exit_code": snapshot.exit_code,
        "term_signal": snapshot.term_signal,
        "error_code": snapshot.error_code,
        "error": snapshot.error,
    }


def headset_status(snapshot: Any) -> dict:
    """The STATUS headset entry from an OpenXR provider snapshot."""
    return {
        "state": snapshot.state.name,
        "headset_state": snapshot.headset_state.name,
        "reason": snapshot.reason.name,
    }


class _SessionError(Exception):
    """Ends the session; its text goes to the robot app as ERROR."""


class _Shutdown(SystemExit):
    """SIGTERM or SIGINT: the main process is stopping, or has died (PR_SET_PDEATHSIG)."""


_STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def _on_stop_signal(signum: int, frame: Any) -> None:
    for sig in _STOP_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)
    raise _Shutdown(0)


class _ServingPipe:
    """Tells the main process, through a pipe, whether this session still serves its robot app.

    Once the HELLO is accepted, how long the teardown may take, in seconds as ASCII text (one
    write, atomic below PIPE_BUF); closing the pipe means the session is ending.
    """

    def __init__(self, fd: int | None = None) -> None:
        self._fd = fd

    def mark_serving(self, teardown_s: float) -> None:
        if self._fd is not None:
            try:
                os.write(self._fd, repr(float(teardown_s)).encode())
            except OSError:
                pass

    def close(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)


class _Session:
    """Serves the robot app at the other end of ``host``."""

    def __init__(
        self,
        host: Any,
        peer: str,
        placeholder: bool,
        plugin_dirs: Sequence[Path],
        serving_pipe: _ServingPipe | None = None,
    ) -> None:
        self._host = host
        self._peer = peer
        self._placeholder = placeholder
        self._plugin_dirs = list(plugin_dirs)
        self._serving_pipe = serving_pipe or _ServingPipe()
        self._plugin_manager: Any = None
        self._last_status: tuple | None = None
        self._last_status_time = -math.inf

    def serve(self) -> int:
        """Run the session until the robot app leaves; the process exit code."""
        logger.debug("Waiting for HELLO from %s", self._peer)
        try:
            # In slices, so SIGTERM is handled within the main process's grace period.
            deadline = time.monotonic() + HELLO_TIMEOUT_MS / 1000
            hello = None
            while hello is None and time.monotonic() < deadline:
                hello = self._host.wait_hello(HELLO_POLL_MS)
        except RuntimeError as error:
            logger.info("Robot app at %s left before HELLO: %s", self._peer, error)
            return 0
        except UnicodeDecodeError as error:
            # pybind11 decodes the HELLO payload as UTF-8.
            logger.warning("Refused the robot app at %s: %s", self._peer, error)
            self._host.send_error(f"HELLO is not valid UTF-8: {error}")
            return 1
        except Exception as error:
            return self._fail(error)
        if hello is None:
            logger.warning(
                "Robot app at %s sent no HELLO within %g s",
                self._peer,
                HELLO_TIMEOUT_MS / 1000,
            )
            return 1
        logger.debug("HELLO from %s: %s", self._peer, hello)
        try:
            self._plugin_manager = pm.PluginManager([str(p) for p in self._plugin_dirs])
            request = parse_hello(
                hello,
                placeholder=self._placeholder,
                plugin_manager=self._plugin_manager,
                plugin_dirs=self._plugin_dirs,
            )
            teardown_s = self._teardown_s(request.plugins)
        except HelloError as error:
            logger.warning("Refused the robot app at %s: %s", self._peer, error)
            self._host.send_error(str(error))
            return 1
        except Exception as error:
            return self._fail(error)
        if request.isaaccapture_version != isaaccapture.__version__:
            logger.warning(
                "Robot app %r runs Isaac Capture %s; this server runs %s",
                request.app_name,
                request.isaaccapture_version or "(unknown)",
                isaaccapture.__version__,
            )
        self._host.send_hello_ack(
            json.dumps(
                {
                    "session_id": uuid.uuid4().hex,
                    "server_version": isaaccapture.__version__,
                }
            )
        )
        self._serving_pipe.mark_serving(teardown_s)
        logger.info(
            "Serving robot app %r at %s: %d tracker(s), %d plugin(s), %g Hz",
            request.app_name,
            self._peer,
            len(request.trackers),
            len(request.plugins),
            request.tick_rate_hz,
        )
        with ExitStack() as stack:
            try:
                return self._run(stack, request)
            except _Shutdown:
                # Queued before the stack closes the host, which delivers it.
                self._host.send_error(SHUTDOWN_MESSAGE)
                raise
            except _SessionError as error:
                logger.error("Robot app at %s: %s", self._peer, error)
                self._host.send_error(str(error))
                return 1
            except Exception as error:
                return self._fail(error)
            finally:
                # Before the teardown, so a robot app that reconnects waits for it.
                self._serving_pipe.close()

    def _teardown_s(self, plugins: Sequence[PluginRequest]) -> float:
        """How long the teardown may take; the plugins stop one after another.

        Each plugin gets its plugin.yaml ``shutdown_timeout_seconds`` (default 2 s).
        """
        info = self._plugin_manager.get_plugin_info
        timeouts = (info(p.plugin_name).shutdown_timeout_seconds for p in plugins)
        return TEARDOWN_MARGIN_S + sum(timeouts)

    def _run(self, stack: ExitStack, request: SessionRequest) -> int:
        self._send_status("connecting_openxr", "opening an OpenXR session")
        vendor_config = VendorConfig(request.vendors)
        if self._placeholder:
            oxr_session = None
            handles = oxr.OpenXRSessionHandles(*PLACEHOLDER_HANDLES)
        else:
            oxr_session = self._open_openxr(stack, request, vendor_config)
            if oxr_session is None:
                return self._robot_left()
            handles = oxr_session.get_handles()

        # The trackers read through the OpenXR session, so the host closes before it.
        stack.callback(self._host.close)
        try:
            self._host.start_session(
                request.trackers, handles, vendor_config, request.topics
            )
        except Exception as error:
            raise _SessionError(
                f"The Teleop Session Server could not start the trackers: {error}"
            ) from error

        names = ", ".join(plugin.plugin_name for plugin in request.plugins)
        self._send_status("starting_plugins", f"starting {names}" if names else "")
        plugins = self._start_plugins(stack, request.plugins)
        return self._stream(request, oxr_session, plugins)

    def _open_openxr(
        self, stack: ExitStack, request: SessionRequest, vendor_config: Any
    ) -> Any:
        """The OpenXR session, retried while no headset is connected; None if the robot left."""
        try:
            extensions = DeviceIOSession.get_required_extensions(
                [tracker for tracker, _ in request.trackers], vendor_config
            )
        except Exception as error:
            raise _SessionError(
                f"The Teleop Session Server could not resolve the trackers: {error}"
            ) from error
        waiting = False
        while self._host.alive:
            try:
                session = oxr.OpenXRSession(
                    request.app_name, extensions, wait_for_system=False
                )
            except RuntimeError as error:
                if not str(error).startswith(_NO_HEADSET_ERROR):
                    raise _SessionError(
                        "The Teleop Session Server could not open an OpenXR session: "
                        f"{error}"
                    ) from error
                if not waiting:
                    waiting = True
                    logger.info("No headset is connected to CloudXR; waiting for one")
                    self._send_status("connecting_openxr", WAITING_FOR_HEADSET)
                time.sleep(OPENXR_RETRY_S)
                continue
            return stack.enter_context(session)
        return None

    def _start_plugins(
        self, stack: ExitStack, requests: Sequence[PluginRequest]
    ) -> list[tuple[PluginRequest, Any]]:
        started = []
        for request in requests:
            try:
                context = self._plugin_manager.start(
                    request.plugin_name, request.plugin_root_id, request.plugin_args
                )
            except Exception as error:
                # Keeps PluginManager's text, e.g. "Plugin process exited immediately: ...".
                raise _SessionError(
                    f"Plugin {request.plugin_name!r} could not start on the Teleop "
                    f"Session Server: {error}"
                ) from error
            stack.callback(_stop_plugin, request, context)
            started.append((request, context))
            logger.info(
                "Started plugin %r (root id %r)",
                request.plugin_name,
                request.plugin_root_id,
            )
        return started

    def _stream(
        self,
        request: SessionRequest,
        oxr_session: Any,
        plugins: Sequence[tuple[PluginRequest, Any]],
    ) -> int:
        """Tick at the robot app's tick rate until it leaves; READY once the headset is steady."""
        self._send_status(
            "waiting_for_headset",
            WAITING_FOR_HEADSET,
            [plugin_status(plugin, context) for plugin, context in plugins],
        )
        period = 1.0 / request.tick_rate_hz
        ready = False
        connected_since = None
        reported_failures: set[tuple[str, str]] = set()
        next_tick = time.monotonic()
        while self._host.alive:
            try:
                self._host.tick()
            except Exception as error:
                raise _SessionError(
                    f"The Teleop Session Server could not update the trackers: {error}"
                ) from error
            now = time.monotonic()
            if oxr_session is None:
                headset = _UNTRACKED_HEADSET
            else:
                snapshot = oxr_session.get_provider_snapshot()
                if snapshot.state == oxr.OpenXRProviderState.FAILED:
                    raise _SessionError(
                        "The OpenXR session on the Teleop Session Server failed "
                        f"({snapshot.reason.name}): {snapshot.error}"
                    )
                if snapshot.headset_state == oxr.OpenXRHeadsetState.CONNECTED:
                    connected_since = (
                        now if connected_since is None else connected_since
                    )
                else:
                    connected_since = None
                headset = headset_status(snapshot)
            entries = [plugin_status(plugin, context) for plugin, context in plugins]
            _log_plugin_failures(entries, reported_failures)

            if not ready and (
                oxr_session is None
                or (
                    connected_since is not None
                    and now - connected_since >= READY_AFTER_CONNECTED_S
                )
            ):
                ready = True
                self._send_status("ready", "", entries, headset, ready=True)
                logger.info("Ready; streaming to the robot app at %s", self._peer)
            elif ready:
                self._send_status("running", "", entries, headset, periodic=True)
            else:
                message = (
                    WAITING_FOR_HEADSET
                    if connected_since is None
                    else "headset connected"
                )
                self._send_status(
                    "waiting_for_headset", message, entries, headset, periodic=True
                )

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
        return self._robot_left()

    def _fail(self, error: Exception) -> int:
        """Log an unexpected failure and send it as ERROR: the robot app retries a bare close."""
        logger.error(
            "Session for the robot app at %s failed", self._peer, exc_info=error
        )
        self._host.send_error(_failure_text(error))
        return 1

    def _robot_left(self) -> int:
        logger.info(
            "Robot app at %s disconnected: %s", self._peer, self._host.close_reason
        )
        return 0

    def _send_status(
        self,
        stage: str,
        message: str = "",
        plugins: Sequence[dict] = (),
        headset: dict | None = None,
        *,
        ready: bool = False,
        periodic: bool = False,
    ) -> None:
        """Send on any change; with ``periodic``, also once every STATUS_PERIOD_S."""
        summary = (
            # READY is the first "running" status.
            "running" if stage == "ready" else stage,
            message,
            tuple((p["plugin_name"], p["plugin_root_id"], p["state"]) for p in plugins),
            tuple(sorted((headset or {}).items())),
        )
        now = time.monotonic()
        if (
            periodic
            and summary == self._last_status
            and now - self._last_status_time < STATUS_PERIOD_S
        ):
            return
        self._last_status = summary
        self._last_status_time = now
        text = status_json(stage, message, plugins, headset)
        if ready:
            self._host.send_ready(text)
        else:
            self._host.send_status(text)


def _log_plugin_failures(entries: Sequence[dict], reported: set) -> None:
    for entry in entries:
        key = (entry["plugin_name"], entry["plugin_root_id"])
        if entry["state"] != "RUNNING" and key not in reported:
            reported.add(key)
            logger.error(
                "Plugin %r (root id %r) is no longer running: %s",
                key[0],
                key[1],
                entry["error"] or f"{entry['state']} ({entry['reason']})",
            )


def _stop_plugin(request: PluginRequest, context: Any) -> None:
    try:
        context.stop()
    except Exception as error:  # noqa: BLE001 - its failure was already reported in STATUS
        logger.debug("Plugin %r had already stopped: %s", request.plugin_name, error)


def _failure_text(error: Exception) -> str:
    """The ERROR text for an unexpected failure."""
    return f"The Teleop Session Server failed: {type(error).__name__}: {error}"


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _list(hello: dict, key: str) -> list:
    value = hello.get(key, [])
    if not isinstance(value, list):
        raise HelloError(f"{key} must be a list")
    return value


def _exit_with_parent(parent_pid: int) -> None:
    """Get SIGTERM when the main process dies; exit now if it already has."""
    if sys.platform.startswith("linux"):
        import ctypes  # noqa: PLC0415

        ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    if os.getppid() != parent_pid:
        raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture.teleop_session_server._session",
        description="Session process of the Teleop Session Server; its main process starts it.",
    )
    parser.add_argument("--fd", type=int, required=True, help="accepted socket")
    parser.add_argument("--websocket", action="store_true", help="fd speaks WebSocket")
    parser.add_argument("--peer", default="(unknown)", help="robot app address")
    parser.add_argument(
        "--serving-fd", type=int, default=None, help="pipe to report serving on"
    )
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--plugin-dir", type=Path, action="append", default=[])
    parser.add_argument("--placeholder-openxr", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _exit_with_parent(args.parent_pid)
    for sig in _STOP_SIGNALS:
        signal.signal(sig, _on_stop_signal)
    # Plugins must not hold the robot app's connection or the serving pipe open.
    for fd in (args.fd, args.serving_fd):
        if fd is not None:
            os.set_inheritable(fd, False)
    serving_pipe = _ServingPipe(args.serving_fd)
    configure_logging(args.verbose)
    # This process is ours, so the OpenXR runtime's and plugins' raw output can go to the
    # capture file for its whole life.
    with capture_native_output():
        with RemoteSessionHost.adopt(args.fd, args.websocket) as host:
            session = _Session(
                host, args.peer, args.placeholder_openxr, args.plugin_dir, serving_pipe
            )
            try:
                return session.serve()
            except _Shutdown:
                host.send_error(SHUTDOWN_MESSAGE)
                logger.info("Session for the robot app at %s stopped", args.peer)
                return 0
            finally:
                serving_pipe.close()


if __name__ == "__main__":
    sys.exit(main())
