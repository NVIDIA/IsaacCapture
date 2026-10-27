# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One robot app's connection to the Teleop Session Server: its HELLO, then its trackers.

The server's loop steps a :class:`RobotConnection`. Its trackers read through the server's OpenXR
session, which outlives the connection; READY goes out once the headset has stayed connected.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

import isaaccapture
from isaaccapture import deviceio_trackers
from isaaccapture.deviceio_session import DeviceIOSession, TrackerVendor, VendorConfig
from isaaccapture.deviceio_trackers._generated_tracker_exports import (
    __all__ as _GENERATED_TRACKERS,
)
from isaaccapture.teleop_session_manager._remote import PROTOCOL_VERSION, has_metadata

logger = logging.getLogger("isaaccapture.teleop_session_server.session")

HELLO_TIMEOUT_S = 5.0
# Longest a HELLO wait holds up the server's loop; also how often a robot app waiting for the
# OpenXR session is checked.
WAIT_SLICE_S = 0.1
# How long the headset must stay connected before READY.
READY_AFTER_CONNECTED_S = 1.1
STATUS_PERIOD_S = 1.0
MIN_TICK_RATE_HZ = 1.0
MAX_TICK_RATE_HZ = 240.0
# Far above any real HELLO, which is a few KB with tens of trackers; a collection id fits in 255
# characters (XR_MAX_TENSOR_IDENTIFIER_SIZE). They bound the work one HELLO costs the server's loop.
MAX_HELLO_BYTES = 1 << 20
MAX_TRACKERS = 256
MAX_TOPICS = 1024
MAX_TEXT_CHARS = 1024
WAITING_FOR_HEADSET = "waiting for the headset to connect to CloudXR"
PLUGINS_ON_SERVER = (
    "plugins are started on the Teleop Session Server machine (by hand or a script), "
    "not requested by the robot app"
)

# Before the robot app's trackers run; it reads the headset entry from READY on.
_NO_HEADSET = {"state": "", "headset_state": "", "reason": ""}
# --placeholder-openxr has no OpenXR runtime, so no headset is ever tracked: what an OpenXR
# session reports without head tracking, which the robot app shows as headset DISCONNECTED.
UNTRACKED_HEADSET = {
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
# FullBodyTracker vendors served besides its default one.
_OTHER_BODY_VENDORS = ("body.noitom", "body.xsens")
_UNSUPPORTED_TRACKERS = {
    "KeyboardTracker": "it would read the Teleop Session Server's keyboard",
    "MessageChannelTracker": "message channels are not forwarded yet",
    "TensorPushTracker": "sink data is not forwarded yet",
    "HapticCommandPushTracker": "sink data is not forwarded yet",
    "HapticCommandReaderTracker": "it reads sink data, which is not forwarded yet",
    "PluginDeviceStatusTracker": "plugin device status is not forwarded yet",
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


def served_extensions() -> list[str]:
    """OpenXR extensions for every tracker kind and vendor a robot app may request.

    An OpenXR session's extensions are fixed when it is created, and the server's serves every
    robot app.
    """
    trackers = [cls() for cls in _PARAMETERLESS_TRACKERS.values()]
    vendored = [
        (deviceio_trackers.FullBodyTracker(), TrackerVendor(vendor))
        for vendor in _OTHER_BODY_VENDORS
    ]
    trackers += [tracker for tracker, _ in vendored]
    trackers += [cls("served_extensions") for cls in _GENERATED_PULL_TRACKERS.values()]
    extensions = DeviceIOSession.get_required_extensions(
        trackers, VendorConfig(vendored)
    )
    # CloudXR provides body tracking.
    return sorted({*extensions, "XR_BD_body_tracking"})


class HelloError(ValueError):
    """The robot app's HELLO cannot be served; its text goes back to the robot app as ERROR."""


@dataclass
class SessionRequest:
    """A valid HELLO, with its trackers built."""

    app_name: str
    isaaccapture_version: str
    tick_rate_hz: float
    trackers: list[tuple[Any, str]]
    vendors: list[tuple[Any, Any]]
    topics: list[str]


def parse_hello(
    text: str, *, placeholder: bool, extensions: Collection[str] | None = None
) -> SessionRequest:
    """Validate a HELLO and build its trackers; raise :class:`HelloError` with the reason.

    With ``extensions`` (the OpenXR session's), trackers that need any other are refused.
    """
    size = len(text.encode())
    if size > MAX_HELLO_BYTES:
        raise HelloError(
            f"HELLO is {size} bytes; at most {MAX_HELLO_BYTES} are accepted"
        )
    try:
        hello = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise HelloError(f"HELLO is not valid JSON: {error}") from None
    if not isinstance(hello, dict):
        raise HelloError("HELLO is not a JSON object")
    _check_text_lengths(hello)
    protocol = hello.get("protocol")
    if _is_bool(protocol) or protocol != PROTOCOL_VERSION:
        raise HelloError(
            f"Remote session protocol {protocol!r} is not supported: this Teleop Session Server "
            f"(Isaac Capture {isaaccapture.__version__}) speaks protocol {PROTOCOL_VERSION}"
        )
    plugins = _list(hello, "plugins")
    if plugins:
        names = [
            entry.get("plugin_name") if isinstance(entry, dict) else entry
            for entry in plugins
        ]
        raise HelloError(
            f"The robot app requested plugins {names!r}, but {PLUGINS_ON_SERVER}"
        )

    app_name = hello.get("app_name", "")
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

    descriptors = _list(hello, "trackers", MAX_TRACKERS)
    trackers: list[tuple[Any, str]] = []
    vendors: list[tuple[Any, Any]] = []
    names: set[str] = set()
    for index, descriptor in enumerate(descriptors):
        tracker, name, vendor = _build_tracker(index, descriptor, placeholder)
        if name in names:
            raise HelloError(f"Tracker name {name!r} is used twice")
        names.add(name)
        trackers.append((tracker, name))
        if vendor is not None:
            vendors.append((tracker, vendor))
    if extensions is not None and trackers:
        _check_extensions(trackers, vendors, extensions)

    topics = _list(hello, "topics", MAX_TOPICS)
    if not all(isinstance(topic, str) for topic in topics):
        raise HelloError("topics must be a list of strings")

    return SessionRequest(
        app_name=app_name,
        isaaccapture_version=str(version),
        tick_rate_hz=min(max(float(rate), MIN_TICK_RATE_HZ), MAX_TICK_RATE_HZ),
        trackers=trackers,
        vendors=vendors,
        topics=list(topics),
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


def _check_extensions(
    trackers: list[tuple[Any, str]],
    vendors: list[tuple[Any, Any]],
    extensions: Collection[str],
) -> None:
    try:
        required = DeviceIOSession.get_required_extensions(
            [tracker for tracker, _ in trackers], VendorConfig(vendors)
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise HelloError(
            f"The Teleop Session Server could not resolve the trackers: {error}"
        ) from None
    missing = sorted(set(required) - set(extensions))
    if missing:
        raise HelloError(
            f"The trackers need OpenXR extensions {missing!r} that the Teleop Session "
            "Server's OpenXR session does not have"
        )


def status_json(stage: str, message: str = "", headset: dict | None = None) -> str:
    """The STATUS / READY JSON."""
    return json.dumps(
        {"stage": stage, "message": message, "headset": headset or _NO_HEADSET}
    )


def headset_status(snapshot: Any) -> dict:
    """The STATUS headset entry from an OpenXR provider snapshot."""
    return {
        "state": snapshot.state.name,
        "headset_state": snapshot.headset_state.name,
        "reason": snapshot.reason.name,
    }


class RobotConnection:
    """Serves the robot app at the other end of ``host``, one :meth:`step` at a time.

    ``extensions`` are the OpenXR session's (None with ``placeholder``). ``ended`` turns true when
    the connection is over; the server then calls :meth:`close`.
    """

    def __init__(
        self,
        host: Any,
        peer: str,
        *,
        placeholder: bool,
        extensions: Collection[str] | None,
    ) -> None:
        self.host = host
        self.peer = peer
        self._placeholder = placeholder
        self._extensions = extensions
        self._hello_deadline = time.monotonic() + HELLO_TIMEOUT_S
        self._request: SessionRequest | None = None
        self._period = 0.0
        self._streaming = False
        self._ready = False
        self._connected_since: float | None = None
        self._last_status: tuple | None = None
        self._last_status_time = -math.inf
        # time.monotonic() at which step() next has work.
        self.next_step = time.monotonic()
        self.ended = False

    def step(self, handles: Any, headset: Callable[[], tuple[dict, bool]]) -> None:
        """Do what is due. ``handles`` are the OpenXR session's, None while it does not exist.

        ``headset()`` gives the STATUS headset entry and whether the headset is connected. What it
        raises, and any unexpected failure, propagates to the server.
        """
        if self._request is None:
            self._wait_for_hello()
        elif not self.host.alive:
            logger.info(
                "Robot app at %s disconnected: %s", self.peer, self.host.close_reason
            )
            self.ended = True
        elif not self._streaming:
            self._start(handles)
        else:
            self._tick(headset)

    def fail(self, error: Exception) -> None:
        """Log an unexpected failure and send it as ERROR: the robot app retries a bare close."""
        logger.error("Serving the robot app at %s failed", self.peer, exc_info=error)
        self._send_error(_failure_text(error))

    def close(self, error: str | None = None) -> None:
        """Send ``error`` if given, then drop the trackers and close the connection."""
        if error is not None:
            self.host.send_error(error)
        self.host.close()

    def _wait_for_hello(self) -> None:
        remaining_s = min(self._hello_deadline - time.monotonic(), WAIT_SLICE_S)
        try:
            hello = self.host.wait_hello(max(0, round(remaining_s * 1000)))
        except RuntimeError as error:
            logger.info("Robot app at %s left before HELLO: %s", self.peer, error)
            self.ended = True
            return
        except UnicodeDecodeError as error:
            # pybind11 decodes the HELLO payload as UTF-8.
            self._refuse(f"HELLO is not valid UTF-8: {error}")
            return
        if hello is None:
            if time.monotonic() >= self._hello_deadline:
                logger.warning(
                    "Robot app at %s sent no HELLO within %g s",
                    self.peer,
                    HELLO_TIMEOUT_S,
                )
                self.ended = True
            return
        logger.debug("HELLO from %s: %s", self.peer, hello)
        try:
            request = parse_hello(
                hello, placeholder=self._placeholder, extensions=self._extensions
            )
        except HelloError as error:
            self._refuse(str(error))
            return
        if request.isaaccapture_version != isaaccapture.__version__:
            logger.warning(
                "Robot app %r runs Isaac Capture %s; this server runs %s",
                request.app_name,
                request.isaaccapture_version or "(unknown)",
                isaaccapture.__version__,
            )
        self.host.send_hello_ack(
            json.dumps(
                {
                    "session_id": uuid.uuid4().hex,
                    "server_version": isaaccapture.__version__,
                }
            )
        )
        self._request = request
        logger.info(
            "Serving robot app %r at %s: %d tracker(s), %g Hz",
            request.app_name,
            self.peer,
            len(request.trackers),
            request.tick_rate_hz,
        )

    def _start(self, handles: Any) -> None:
        """Start the trackers once the OpenXR session exists."""
        if handles is None:
            self._send_status("connecting_openxr", WAITING_FOR_HEADSET)
            self.next_step = time.monotonic() + WAIT_SLICE_S
            return
        request = self._request
        try:
            self.host.start_session(
                request.trackers, handles, VendorConfig(request.vendors), request.topics
            )
        except Exception as error:
            self._drop(
                f"The Teleop Session Server could not start the trackers: {error}"
            )
            return
        self._streaming = True
        self._period = 1.0 / request.tick_rate_hz
        self.next_step = time.monotonic()

    def _tick(self, headset: Callable[[], tuple[dict, bool]]) -> None:
        """One tick at the robot app's tick rate; READY once the headset is steady."""
        try:
            self.host.tick()
        except Exception as error:
            self._drop(
                f"The Teleop Session Server could not update the trackers: {error}"
            )
            return
        entry, connected = headset()
        now = time.monotonic()
        if not connected:
            self._connected_since = None
        elif self._connected_since is None:
            self._connected_since = now

        if not self._ready and (
            self._placeholder
            or (
                self._connected_since is not None
                and now - self._connected_since >= READY_AFTER_CONNECTED_S
            )
        ):
            self._ready = True
            self._send_status("ready", "", entry, ready=True)
            logger.info("Ready; streaming to the robot app at %s", self.peer)
        elif self._ready:
            self._send_status("running", "", entry, periodic=True)
        else:
            message = (
                WAITING_FOR_HEADSET
                if self._connected_since is None
                else "headset connected"
            )
            self._send_status("waiting_for_headset", message, entry, periodic=True)

        self.next_step += self._period
        if self.next_step < now:
            self.next_step = now

    def _refuse(self, text: str) -> None:
        """End the connection over its HELLO."""
        logger.warning("Refused the robot app at %s: %s", self.peer, text)
        self._send_error(text)

    def _drop(self, text: str) -> None:
        """End the connection over a failure of its trackers."""
        logger.error("Robot app at %s: %s", self.peer, text)
        self._send_error(text)

    def _send_error(self, text: str) -> None:
        self.host.send_error(text)
        self.ended = True

    def _send_status(
        self,
        stage: str,
        message: str = "",
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
            tuple(sorted((headset or {}).items())),
        )
        now = time.monotonic()
        if summary == self._last_status and (
            not periodic or now - self._last_status_time < STATUS_PERIOD_S
        ):
            return
        self._last_status = summary
        self._last_status_time = now
        text = status_json(stage, message, headset)
        if ready:
            self.host.send_ready(text)
        else:
            self.host.send_status(text)


def _failure_text(error: Exception) -> str:
    """The ERROR text for an unexpected failure."""
    return f"The Teleop Session Server failed: {type(error).__name__}: {error}"


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _list(hello: dict, key: str, limit: int | None = None) -> list:
    value = hello.get(key, [])
    if not isinstance(value, list):
        raise HelloError(f"{key} must be a list")
    if limit is not None and len(value) > limit:
        raise HelloError(
            f"HELLO lists {len(value)} {key}; at most {limit} are accepted"
        )
    return value


def _check_text_lengths(hello: dict) -> None:
    """Refuse a HELLO with any key or string longer than MAX_TEXT_CHARS."""
    pending: list[Any] = [hello]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and len(value) > MAX_TEXT_CHARS:
            raise HelloError(
                f"HELLO has a text of {len(value)} characters ({value[:40]!r}...); "
                f"at most {MAX_TEXT_CHARS} are accepted"
            )
