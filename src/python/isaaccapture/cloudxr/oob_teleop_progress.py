# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse operator progress derived from OOB lifecycle snapshots."""

from __future__ import annotations

import time
from collections.abc import Callable


class OobProgressReporter:
    """Describe setup transitions for foreground and embedded service owners."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._adb_ready = False
        self._ever_adb_ready = False
        self._last_stage: tuple | None = None
        self._last_health: str | None = None
        self._last_wait: tuple | None = None
        self._last_wait_at = float("-inf")
        self._last_delay_at: dict[str, float] = {}

    def observe(self, snapshot: dict) -> list[str]:
        """Return new messages; repeated waits get at most one reminder per minute."""
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
        now = self._clock()
        messages: list[str] = []

        if ready and not self._adb_ready:
            verb = "reconnected" if self._ever_adb_ready else "detected"
            messages.append(
                f"USB-attached HMD {verb} (serial {serial}); continuing OOB setup."
            )
            self._ever_adb_ready = True
        self._adb_ready = ready

        if state == "WAITING_FOR_ADB" and not ready:
            wait = (serial, reason, snapshot.get("adbDiagnostic"))
            if wait != self._last_wait or now - self._last_wait_at >= 60:
                if snapshot.get("adbDiagnostic"):
                    messages.append(f"ADB diagnostic: {snapshot['adbDiagnostic']}")
                elif serial is None and not any(
                    word in reason.lower()
                    for word in ("unauthorized", "offline", "multiple")
                ):
                    messages.append(
                        "No USB-attached HMD detected; connect a headset to continue."
                    )
                else:
                    messages.append(f"Waiting for USB-attached HMD: {reason}")
                self._last_wait_at = now
            self._last_wait = wait
        else:
            self._last_wait = None

        if entered:
            if state == "PREPARING_DEVICE":
                messages.append(
                    f"Waiting for headset transport: {reason}"
                    if snapshot.get("networkPresent") is False
                    else f"Preparing headset {serial} for OOB setup."
                )
            elif state == "REBUILDING_USB":
                messages.append("Configuring USB reverse rules and TURN.")
            elif state == "AUTOMATING_BROWSER":
                messages.append(
                    "USB reverse rules and TURN verified; opening headset browser."
                    if reason == "TURN and reverse rules verified"
                    else "Opening headset browser and connecting OOB client."
                )
            elif state == "VERIFYING_EXISTING_BROWSER":
                messages.append("USB transport restored; checking existing browser.")
            elif state == "VERIFYING_BROWSER":
                messages.append("Waiting for headset browser health report.")
            elif state == "WAITING_FOR_AWAKE":
                messages.append(f"Waiting for selected headset to wake: {reason}")
            elif state == "USB_HEALTH_BLOCKED":
                messages.append(f"USB health policy blocked browser launch: {reason}")
            elif state == "DEGRADED" or "Recovery episode expired" in reason:
                if now - self._last_delay_at.get(reason, float("-inf")) >= 30:
                    messages.append(f"OOB recovery delayed: {reason}")
                    self._last_delay_at[reason] = now

        if health == "browser_ready" and health != self._last_health:
            messages.append(
                "OOB browser ready; waiting for fresh stream metrics."
                if snapshot.get("streaming")
                else "OOB browser ready; waiting for OpenXR stream."
            )
        self._last_stage = stage
        self._last_health = health
        return messages
