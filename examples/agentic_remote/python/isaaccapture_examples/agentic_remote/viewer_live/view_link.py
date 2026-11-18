# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The viewer's word to the robot: VIEW_STATUS, how fresh the live view is.

The live view is the viewer's page with its live picture showing. ViewLink
attaches to the robot link service's local socket in the role "viewer", as the
live view program of view/ does, and sends what the ViewModel of view/model.py
says: "live" with the age of the picture the page loaded last has drawn and
acknowledged, else not live. VIEW_STATUS is the only link message it sends.

Without a robot link service the viewer serves everything as before: the
attach is tried again after a pause that grows as the camera's does.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..link_service.local import LocalCarrier, connect_local, default_socket_path
from ..protocol.carrier import receive_message, send
from ..protocol.local_messages import ROLE_VIEWER
from ..protocol.messages import ViewStatus
from ..view.__main__ import ATTACH_TIMEOUT_S, SECOND_VIEW, ViewSettings
from ..view.model import AGE_WHEN_NOT_LIVE_S, ViewModel

_log = logging.getLogger("isaaccapture.examples.agentic_remote.viewer_live")

PROGRAM = "viewer"
# The name the viewer attaches under.
NAME = "viewer"
# One VIEW_STATUS this often, also while the view is not live: view.yaml's.
PERIOD_S = ViewSettings().period_s
# A failed attach is tried again after a pause: the first of these after the
# viewer was attached, the next after each further failed try, then the last.
RETRY_PAUSES_S = (5.0, 10.0, 20.0, 40.0, 60.0)
NOT_LIVE = ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S)
_RECEIVE_S = 0.2
# How long close() waits for the threads.
_CLOSE_S = ATTACH_TIMEOUT_S + 5.0

Say = Callable[[str], None]
Attach = Callable[..., LocalCarrier]


def _say(text: str) -> None:
    print(text, flush=True)


class ViewLink:
    """Sends the robot VIEW_STATUS from a ViewModel while a service is there.

    attach(path, role, name, timeout_s) gives the carrier of the local API.
    Each change between attached and not gets one line from `say`; the tries
    of one stretch without a service get that one line, with the first reason.
    """

    def __init__(
        self,
        model: ViewModel,
        socket_path: str | None = None,
        *,
        attach: Attach = connect_local,
        say: Say = _say,
        period_s: float = PERIOD_S,
        retry_pauses_s: Sequence[float] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model = model
        self._socket_path = socket_path
        self._attach = attach
        self._say = say
        self._period_s = float(period_s)
        pauses = RETRY_PAUSES_S if retry_pauses_s is None else retry_pauses_s
        self._retry_pauses_s = tuple(pauses)
        if not self._retry_pauses_s:
            raise ValueError("retry_pauses_s names no pause")
        self._monotonic = monotonic
        self._end = threading.Event()
        self._lock = threading.Lock()
        self._attached = False
        self._why = "the robot link service is being asked"
        self._sent: ViewStatus | None = None
        self._told: bool | None = None  # what the last line said: attached or not
        self._thread: threading.Thread | None = None

    @property
    def socket_path(self) -> str:
        """Where the robot link service is looked for."""
        return self._socket_path or default_socket_path()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="viewer-live-link", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """End the thread; the robot is told that the view is over. Safe twice."""
        self._end.set()
        if self._thread is not None:
            self._thread.join(_CLOSE_S)
            if self._thread.is_alive():
                _log.warning("the live view's thread did not end")

    def state(self) -> dict[str, Any]:
        """What /api/live.json says of the live view: what the robot is told."""
        with self._lock:
            sent = self._sent if self._attached else None
            return {
                "attached": self._attached,
                "why": "" if self._attached else self._why,
                "live": None if sent is None else sent.live,
                "age_s": None if sent is None or not sent.live else sent.age_s,
            }

    def _run(self) -> None:
        failed = 0  # tries in a row, since the viewer was attached
        while not self._end.is_set():
            carrier, why = self._attach_once()
            if carrier is not None:
                failed = 0
                with self._lock:
                    self._attached, self._sent = True, None
                self._tell(True, "")
                try:
                    why = self._serve(carrier)
                finally:
                    carrier.close()
                    with self._lock:
                        self._attached = False
                    self._model.on_local(None, False)
                if self._end.is_set():
                    return
            pauses = self._retry_pauses_s
            pause_s = pauses[min(failed, len(pauses) - 1)]
            failed += 1
            self._tell(False, why)
            self._end.wait(pause_s)

    def _attach_once(self) -> tuple[LocalCarrier | None, str]:
        """The carrier of a new attachment, or None and why there is none."""
        try:
            carrier = self._attach(
                self._socket_path, ROLE_VIEWER, NAME, ATTACH_TIMEOUT_S
            )
        except OSError as problem:  # also LocalRefused and TimeoutError
            return None, f"cannot attach to the robot link service: {problem}"
        status = carrier.status
        if status is not None and status.roles.get(ROLE_VIEWER, 0) > 1:
            # Two would send the Thor two different VIEW_STATUS in turn.
            carrier.close()
            return None, SECOND_VIEW
        return carrier, ""

    def _serve(self, carrier: LocalCarrier) -> str:
        """VIEW_STATUS every period until the service is gone; why it ended.

        "" when told to end: the robot is then told that the view is over.
        """
        reader = threading.Thread(
            target=self._receive, args=(carrier,), name="viewer-live-receive"
        )
        reader.daemon = True
        reader.start()  # it ends with the carrier, which the caller closes
        while carrier.alive:
            try:
                self._model.on_local(carrier.status, carrier.alive)
                status = self._model.status_to_send(self._monotonic())
                send(carrier, status)
                self._model.note_sent(status)
                with self._lock:
                    self._sent = status
            except Exception:
                _log.exception("sending VIEW_STATUS failed")
            if self._end.wait(self._period_s):
                with contextlib.suppress(Exception):
                    send(carrier, NOT_LIVE)
                return ""
        return f"the robot link service is gone ({carrier.close_reason})"

    def _receive(self, carrier: LocalCarrier) -> None:
        """Hand the Thor's WELCOME, ROBOT_STATE and EVENT to the model."""
        while carrier.alive and not self._end.is_set():
            try:
                message = receive_message(carrier, _RECEIVE_S)
            except Exception:
                _log.exception("receiving a link message failed")
                self._end.wait(_RECEIVE_S)
                continue
            if message is not None:
                self._model.on_message(message, self._monotonic())

    def _tell(self, attached: bool, why: str) -> None:
        """Say a change between attached and not once; keep the newest reason."""
        with self._lock:
            self._why = why
            if self._end.is_set() or self._told is attached:
                return
            self._told = attached
        if attached:
            self._say(
                f"{PROGRAM}: live view: attached to the robot link service at "
                f"{self.socket_path}; the robot is told how fresh the page's live "
                "picture is"
            )
        else:
            low, high = self._retry_pauses_s[0], self._retry_pauses_s[-1]
            every = f"{low:g} s" if low == high else f"{low:g} to {high:g} s"
            self._say(
                f"{PROGRAM}: live view: not reported to the robot: {why}; trying "
                f"again every {every}. The page is served as before"
            )
