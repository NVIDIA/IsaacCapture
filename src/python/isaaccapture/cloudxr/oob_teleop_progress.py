# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse operator progress derived from OOB lifecycle snapshots."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class OobProgressEvent:
    level: int
    message: str


class OobProgressReporter:
    """Describe setup transitions for foreground and embedded service owners."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._adb_ready = False
        self._ever_adb_ready = False
        self._last_stage: tuple | None = None
        self._last_health: str | None = None
        self._last_wait: tuple | None = None
        self._last_delay_at: dict[str, float] = {}

    def observe(self, snapshot: dict) -> list[OobProgressEvent]:
        """Return leveled progress on meaningful state changes."""
        if snapshot.get("health") == "starting":
            return []

        state = snapshot.get("state")
        health = snapshot.get("health")
        reason = snapshot.get("reason") or "unknown reason"
        serial = snapshot.get("selectedSerial")
        ready = bool(snapshot.get("adbReady"))
        stage = (
            state,
            reason,
            snapshot.get("episodeStartedAt"),
            snapshot.get("generation"),
        )
        entered = stage != self._last_stage
        state_changed = self._last_stage is None or state != self._last_stage[0]
        now = self._clock()
        messages: list[OobProgressEvent] = []

        def emit(level: int, message: str) -> None:
            messages.append(OobProgressEvent(level, message))

        if ready and not self._adb_ready:
            verb = "reconnected" if self._ever_adb_ready else "detected"
            emit(
                logging.INFO,
                f"USB-attached HMD {verb} (serial {serial}); continuing OOB setup.",
            )
            self._ever_adb_ready = True
        self._adb_ready = ready

        if state == "WAITING_FOR_ADB" and not ready:
            diagnostic = snapshot.get("adbDiagnostic")
            lower_reason = reason.lower()
            # ADB authorization needs operator action even when raw diagnostics exist.
            if "unauthorized" in lower_reason:
                wait = ("unauthorized", serial)
                level = logging.WARNING
                message = (
                    "ADB authorization required: unlock the headset and accept "
                    "the USB debugging prompt."
                )
            elif "offline" in lower_reason:
                wait = ("offline", serial)
                level = logging.INFO
                message = f"Waiting for USB-attached HMD: {reason}"
            elif "multiple" in lower_reason:
                wait = ("multiple", serial)
                level = logging.WARNING
                message = f"Waiting for USB-attached HMD: {reason}"
            elif diagnostic:
                wait = ("adb_diagnostic", diagnostic)
                level = logging.WARNING
                message = f"ADB diagnostic: {diagnostic}"
            elif serial is None:
                wait = ("no_headset",)
                level = logging.INFO
                message = "No USB-attached HMD detected; connect a headset to continue."
            else:
                wait = ("selected_missing", serial)
                level = logging.INFO
                message = f"Waiting for USB-attached HMD: {reason}"
            if wait != self._last_wait:
                emit(level, message)
            self._last_wait = wait
        else:
            self._last_wait = None

        if entered:
            if state == "PREPARING_DEVICE":
                emit(
                    logging.DEBUG,
                    f"Waiting for headset transport: {reason}"
                    if snapshot.get("networkPresent") is False
                    else f"Preparing headset {serial} for OOB setup.",
                )
            elif state == "REBUILDING_USB":
                emit(logging.DEBUG, "Configuring USB reverse rules and TURN.")
            elif state == "AUTOMATING_BROWSER":
                emit(
                    logging.DEBUG,
                    "USB reverse rules and TURN verified; opening headset browser."
                    if reason == "TURN and reverse rules verified"
                    else "Opening headset browser and connecting OOB client.",
                )
            elif state == "VERIFYING_EXISTING_BROWSER":
                emit(
                    logging.DEBUG, "USB transport restored; checking existing browser."
                )
            elif state == "VERIFYING_BROWSER":
                emit(logging.DEBUG, "Waiting for headset browser health report.")
            elif state == "WAITING_FOR_AWAKE" and state_changed:
                emit(logging.INFO, f"Waiting for selected headset to wake: {reason}")
            elif state == "WAITING_FOR_NETWORK" and state_changed:
                emit(
                    logging.WARNING,
                    "Headset network unavailable; connect it to Wi-Fi to continue.",
                )
            elif state == "USB_HEALTH_BLOCKED" and state_changed:
                emit(
                    logging.WARNING,
                    f"USB health policy blocked browser launch: {reason}",
                )
        if state == "DEGRADED" or "Recovery episode expired" in reason:
            if now - self._last_delay_at.get(reason, float("-inf")) >= 30:
                emit(logging.WARNING, f"OOB recovery delayed: {reason}")
                self._last_delay_at[reason] = now

        if health == "browser_ready" and health != self._last_health:
            emit(
                logging.INFO,
                "OOB browser ready; waiting for fresh stream metrics."
                if snapshot.get("streaming")
                else "OOB browser ready; waiting for OpenXR stream.",
            )
        self._last_stage = stage
        self._last_health = health
        return messages
