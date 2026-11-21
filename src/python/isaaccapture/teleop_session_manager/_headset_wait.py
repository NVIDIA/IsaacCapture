# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIVE with ``wait_for_headset`` False: open the OpenXR session once a headset has reached the runtime.

One background thread constructs ``oxr.OpenXRSession``, hands it over through ``poll()`` and
closes it in ``close()``. The thread that calls ``TeleopSession.step()`` does everything else.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any

from isaaccapture import oxr

from . import _remote

logger = logging.getLogger(__name__)

# How oxr.OpenXRSession(wait_for_system=False) fails while no headset has reached the runtime (oxr_session.cpp).
NO_HEADSET_ERROR = "Failed to get OpenXR system: XR_ERROR_FORM_FACTOR_UNAVAILABLE"
# Pause before each attempt, as the Teleop Session Server's.
RETRY_S = 1.0
# Longest close() waits for the thread.
CLOSE_JOIN_S = 5.0
THREAD_NAME = "isaaccapture-headset-wait"
WAITING = "No headset has reached the OpenXR runtime yet; waiting for one"
OPEN_FAILED = (
    "The OpenXR session could not be opened while waiting for a headset: {error}"
)


class HeadsetWait:
    """Opens the OpenXR session on a background thread and keeps it until ``close()``.

    The thread creates the session and closes it; no capture scope is opened on it. The lock
    guards the session and the failure. Every method but ``get_provider_snapshot`` belongs to
    the thread that calls ``TeleopSession.step()``.
    """

    def __init__(self, app_name: str, extensions: list[str]) -> None:
        self._app_name = app_name
        self._extensions = list(extensions)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._session: Any | None = None
        self._failure: tuple[str, Exception] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
        thread.start()
        self._thread = thread

    def poll(self) -> Any | None:
        """The open session; None while no headset has reached the runtime.

        Raises a new RuntimeError on each call after an attempt failed for another reason
        or after ``fail()``; its cause is the failure.
        """
        with self._lock:
            if self._failure is not None:
                text, cause = self._failure
                raise RuntimeError(text) from cause
            return self._session

    def fail(self, text: str, cause: Exception) -> None:
        """Make every later ``poll()`` raise ``RuntimeError(text)`` from ``cause``."""
        with self._lock:
            self._failure = (text, cause)

    def get_provider_snapshot(self) -> SimpleNamespace:
        """What the status monitor polls until step() starts the trackers: no headset is connected."""
        return _remote.no_headset_snapshot(WAITING)

    def close(self) -> None:
        """Stop the thread, which closes a session it opened. Idempotent."""
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(CLOSE_JOIN_S)
        if thread.is_alive():
            logger.warning(
                "The thread that waits for a headset is still inside the OpenXR runtime "
                "after %g s; it ends, and closes a session it still opens, when the "
                "runtime answers",
                CLOSE_JOIN_S,
            )

    def _run(self) -> None:
        session = self._open()
        if session is None:
            return
        with self._lock:
            self._session = session
        if not self._stop.is_set():
            logger.info(
                "A headset has reached the OpenXR runtime; opened the OpenXR session"
            )
        self._stop.wait()
        with self._lock:
            self._session = None
        try:
            session.close()
        except Exception:  # noqa: BLE001 - nobody is left to raise to
            logger.exception("Failed to close the OpenXR session")

    def _open(self) -> Any | None:
        """One attempt every RETRY_S until one succeeds; None when stopped or failed."""
        while not self._stop.wait(RETRY_S):
            try:
                return oxr.OpenXRSession(
                    self._app_name, self._extensions, wait_for_system=False
                )
            except Exception as error:  # noqa: BLE001 - handed to step()
                if isinstance(error, RuntimeError) and str(error).startswith(
                    NO_HEADSET_ERROR
                ):
                    continue
                text = OPEN_FAILED.format(error=error)
                logger.error("%s", text)
                self.fail(text, error)
                return None
        return None
