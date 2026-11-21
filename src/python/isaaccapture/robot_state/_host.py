# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot state kept and served by the robot app itself, in a session without a Teleop Session
Server.

Private to this package: `TeleopSession` makes one host per entry in LIVE and REPLAY mode.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

from . import records
from ._local import (
    _CLOSE_JOIN_S,
    ReaderSocketTaken,
    RobotStateLocalService,
    default_socket_path,
)
from ._robot_state import RobotStateStore, monotonic_now_ns

if TYPE_CHECKING:
    from .description import RobotDescription

logger = logging.getLogger(__name__)

__all__ = ["RETRY_S", "RobotStateHost"]

# How often a host without a reader socket tries to open it.
RETRY_S = 2.0
RETRY_THREAD_NAME = "isaaccapture-robot-state-socket-retry"
# One robot per reader socket path: a robot app that starts as the second one is refused.
_TAKEN = (
    "Another program on this machine already serves a robot's state at {path}: a Teleop "
    "Session Server or another robot app. One robot per machine is served; stop that "
    "program, then start this robot app again."
)


class RobotStateHost:
    """Robot state hosted in the robot app: publish calls fill a store that readers on this
    machine read.

    `publish_robot_record()` may be called from any thread. A host does not start at a path
    another program serves. While the reader socket cannot be opened for another reason the
    host is not active and stores nothing; a thread of the host's own tries the socket again
    when a publish call asks, so a publish call never waits for the socket.
    """

    def __init__(self, socket_path: str | os.PathLike | None = None) -> None:
        self._path = (
            default_socket_path() if socket_path is None else os.fspath(socket_path)
        )
        self._store: RobotStateStore | None = None
        self._service: RobotStateLocalService | None = None
        self._closed = False
        # Serialises the attempts to open the socket; a put never takes it.
        self._attempt_lock = threading.Lock()
        self._last_attempt: float | None = None
        self._reported: str | None = None
        # Lives while the socket is not served; waits until a publish call asks.
        self._retry_thread: threading.Thread | None = None
        self._retry_asked = threading.Event()
        # Serialises the publish calls that ask; never waited for.
        self._ask_lock = threading.Lock()

    @property
    def robot_state_active(self) -> bool:
        """True while started, not closed, and the reader socket is served."""
        service = self._service
        return not self._closed and service is not None and service.serving

    def start(self, description: RobotDescription) -> None:
        """Fill a new store with the robot's series and description, then open the reader socket.

        RuntimeError when another program serves robot state at the path; the host is closed
        then. A socket that cannot be opened for another reason is logged, not raised; publish
        calls have it tried again.
        """
        if self._closed:
            raise RuntimeError("RobotStateHost is closed")
        if self._store is not None:
            raise RuntimeError("RobotStateHost is started")
        store = RobotStateStore()
        store.begin_session()
        for series in records.declarations(description):
            store.declare_series(
                series.name,
                series.shape,
                series.schema_name,
                series.schema_encoding,
                series.schema_data,
                series.message_encoding,
                stale_after_ns=series.stale_after_ns,
                in_order=series.in_order,
                rate_hz=series.rate_hz,
            )
        now_ns = time.monotonic_ns()
        _put(
            store,
            records.DESCRIPTION_TOPIC,
            records.encode_description(description, now_ns),
            now_ns,
        )
        # Filled before the socket opens: a reader's first answer holds the description.
        self._store = store
        self._last_attempt = time.monotonic()
        try:
            served = self._open(at_start=True)
        except ReaderSocketTaken as error:
            self.close()
            raise RuntimeError(_TAKEN.format(path=self._path)) from error
        if not served:
            # Started here, not by a publish call: starting a thread makes its caller wait.
            with self._ask_lock:
                self._start_retry_thread()

    def publish_robot_record(
        self, series: str, data: bytes, sample_time_ns: int
    ) -> bool:
        """Store one record for the readers; False when closed or while no reader socket is served.

        ValueError for a record the store does not take (above 1 MiB, undeclared series).
        """
        store = self._store
        if self._closed or store is None:
            return False
        service = self._service
        if service is None or not service.serving:
            self._ask_retry()
            return False
        _put(store, series, data, sample_time_ns)
        return True

    def close(self) -> None:
        """End the robot's session in the store and close the reader socket. Idempotent."""
        if self._closed:
            return
        # Set before the service is taken: an attempt under way closes what it opens after this.
        self._closed = True
        if self._store is not None:
            self._store.end_session()
        self._retry_asked.set()
        thread = self._retry_thread
        if thread is not None and thread is not threading.current_thread():
            # An attempt under way ends first: nothing of this host holds the path afterwards.
            thread.join(_CLOSE_JOIN_S)
        service, self._service = self._service, None
        if service is not None:
            service.close()

    def _ask_retry(self) -> None:
        """Have the retry thread try the socket, at most once per RETRY_S; waits for nothing."""
        last = self._last_attempt
        if last is not None and time.monotonic() - last < RETRY_S:
            return
        if not self._ask_lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            last = self._last_attempt
            if self._closed or (last is not None and now - last < RETRY_S):
                return
            self._last_attempt = now
            if self._start_retry_thread():
                self._retry_asked.set()
        finally:
            self._ask_lock.release()

    def _start_retry_thread(self) -> bool:
        """Start the retry thread unless it runs; False when there is none afterwards.

        `self._ask_lock` is held.
        """
        thread = self._retry_thread
        if thread is not None and thread.is_alive():
            return True
        if self._closed:
            return False
        thread = threading.Thread(
            target=self._retry, name=RETRY_THREAD_NAME, daemon=True
        )
        try:
            thread.start()
        except RuntimeError as error:
            self._report(error)
            return False
        self._retry_thread = thread
        return True

    def _retry(self) -> None:
        """The retry thread: one attempt each time a publish call asks, until the socket is
        served or the host closes."""
        while True:
            self._retry_asked.wait()
            self._retry_asked.clear()
            if self._closed:
                return
            if self._open():
                with self._ask_lock:
                    # A publish call that asked meanwhile is not left without a thread.
                    if not self._retry_asked.is_set():
                        self._retry_thread = None
                        return

    def _open(self, at_start: bool = False) -> bool:
        """One attempt to serve the reader socket; True when it is served.

        With `at_start`, ReaderSocketTaken is raised, not logged.
        """
        if not self._attempt_lock.acquire(blocking=False):
            return False
        try:
            service = self._service
            if service is not None and service.serving:
                return True
            if self._closed:
                return False
            if service is not None:
                # Its thread ended.
                self._service = None
                service.close()
            service = RobotStateLocalService(self._store, self._path)
            try:
                service.start()
            except (OSError, RuntimeError) as error:
                if at_start and isinstance(error, ReaderSocketTaken):
                    raise
                self._report(error)
                return False
            self._reported = None
            self._service = service
            if self._closed:
                # close() ran meanwhile and may not have seen this service.
                self._service = None
                service.close()
                return False
            return True
        finally:
            self._attempt_lock.release()

    def _report(self, error: Exception) -> None:
        """One warning per reason, however often the socket is tried."""
        if str(error) == self._reported:
            return
        self._reported = str(error)
        logger.warning(
            "Reader programs cannot get robot state from this robot app: %s. "
            "Publish calls return False until the socket can be opened; trying again "
            "every %g s",
            error,
            RETRY_S,
        )


def _put(store: RobotStateStore, series: str, data: bytes, sample_time_ns: int) -> None:
    """Put a record whose times are on Python's monotonic clock."""
    now_ns = time.monotonic_ns()
    store.put_record(
        series,
        data,
        sample_time_ns,
        publish_time_ns=now_ns,
        # The store ages records on its own clock; the offset between the two is measured.
        robot_minus_server_ns=now_ns - monotonic_now_ns(),
        clock_two_way=True,
    )
