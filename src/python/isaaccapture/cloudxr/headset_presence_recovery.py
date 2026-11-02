# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-owned power recovery, gated by a live vendor presence observation."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)
_SENSOR = "stk_stk3x1x_prox Proximity Sensor Wakeup"


@dataclass(frozen=True)
class PresenceObservation:
    presence: str
    wakefulness: str
    supported: bool = True


def parse_pico_presence(sensors: str) -> str:
    """Read the latest on-change value only while its vendor listener is active."""
    # Android mProximityPositive is a phone screen-off signal, not PICO wear state.
    # On-change timestamps can be old while continuously worn; re-read the current
    # sensor dump for every decision instead of expiring or caching the NEAR value.
    if not re.search(r"^Mode\s*:\s*NORMAL\s*$", sensors, re.MULTILINE) or not re.search(
        r"^Sensor Privacy:\s*disabled\s*$", sensors, re.MULTILINE
    ):
        return "unknown"
    inventory = re.search(
        r"^(0x[0-9a-fA-F]+)\) "
        + re.escape(_SENSOR)
        + r"\s*\|[^\n]*android.sensor.proximity\(8\)",
        sensors,
        re.MULTILINE,
    )
    if not inventory or not re.search(
        r"^" + inventory[1] + r"\) active-count = [1-9][0-9]*;", sensors, re.MULTILINE
    ):
        return "unknown"
    active = False
    for connection in re.split(r"Connection Number:\s*\d+", sensors)[1:]:
        if (
            "com.pvr.sensormanager.ProximitySensorManager" in connection
            and "Operating Mode: NORMAL" in connection
            and re.search(r"Frozen state:\s*0\b", connection)
            and re.search(
                re.escape(_SENSOR) + r" " + inventory[1] + r" \| status:\s*active\b",
                connection,
            )
        ):
            active = True
            break
    if not active:
        return "unknown"
    block = re.search(
        r"^"
        + re.escape(_SENSOR)
        + r": last (\d+) events[ \t]*\n((?:[ \t]+[^\n]*\n?)*)",
        sensors,
        re.MULTILINE,
    )
    if not block:
        return "unknown"
    events = []
    for index, line in enumerate(block[2].splitlines(), start=1):
        event = re.fullmatch(
            r"\s*(\d+) \(ts=([^,]+),\s*wall=[^)]+\)\s+([^,]+),.*", line
        )
        if not event or int(event[1]) != index:
            return "unknown"
        try:
            stamp, value = float(event[2]), float(event[3])
        except ValueError:
            return "unknown"
        if not math.isfinite(stamp) or not math.isfinite(value) or stamp < 0:
            return "unknown"
        events.append((stamp, value))
    if (
        not events
        or len(events) != int(block[1])
        or len({stamp for stamp, _ in events}) != len(events)
    ):
        return "unknown"
    _, value = max(events)
    # Only the measured 0/5 encoding is supported; do not guess for new hardware.
    return {0.0: "present", 5.0: "absent"}.get(value, "unknown")


async def run_device_adb(serial: str, arguments: list[str]) -> str:
    """Bound and reap every device command, including cancellation during shutdown."""
    process = await asyncio.create_subprocess_exec(
        "adb",
        "-s",
        serial,
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise OSError("Selected headset unavailable; reconnect and check USB debugging")
    return stdout.decode(errors="replace")


class PresenceRecovery:
    """Wake only confirmed-present sleeping headsets without altering their XR session."""

    def __init__(
        self,
        serial: str,
        *,
        command: Callable[[str, list[str]], Awaitable[str]] = run_device_adb,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.serial = serial
        self.command = command
        self.clock = clock
        self.attempts = 0
        self.next_attempt = 0.0
        self.awake_since: float | None = None
        self.failures = 0
        self.snapshot = {
            "state": "monitoring",
            "presence": "unknown",
            "wakefulness": "unknown",
            "attempts": 0,
            "reason": "Checking headset presence",
        }

    async def observe(self) -> PresenceObservation:
        # Check identity on each probe: a reboot/reconnect cannot inherit capability.
        vendor = await self.command(
            self.serial, ["shell", "getprop", "ro.product.manufacturer"]
        )
        if vendor.strip().casefold() != "pico":
            return PresenceObservation("unknown", "unknown", supported=False)
        power = await self.command(self.serial, ["shell", "dumpsys", "power"])
        wake = re.search(r"\bmWakefulness=(\w+)", power)
        if not wake:
            raise OSError("Headset power state unavailable")
        sensors = await self.command(self.serial, ["shell", "dumpsys", "sensorservice"])
        return PresenceObservation(parse_pico_presence(sensors), wake[1])

    def _publish(
        self, state: str, reason: str, observation: PresenceObservation
    ) -> None:
        if self.snapshot["state"] != state:
            log.info("%s", reason)
        self.snapshot = {
            "state": state,
            "presence": observation.presence,
            "wakefulness": observation.wakefulness,
            "attempts": self.attempts,
            "reason": reason,
        }

    async def step(self) -> float:
        """Observe once and return the next poll delay; no background work escapes."""
        try:
            observation = await self.observe()
            self.failures = 0
            if not observation.supported:
                self.awake_since = None
                self._publish(
                    "unsupported",
                    "Automatic wake unavailable; wake the headset manually",
                    observation,
                )
                return 10.0
            if observation.presence != "present":
                self.awake_since = None
                # FAR ends this wear episode; unknown does not renew its retry budget.
                if observation.presence == "absent":
                    self.attempts = 0
                self._publish(
                    "paused",
                    "Headset removed; waiting for presence"
                    if observation.presence == "absent"
                    else "Presence unknown; put on the headset or wake it manually",
                    observation,
                )
                return 1.0
            if observation.wakefulness == "Awake":
                if self.awake_since is None:
                    self.awake_since = self.clock()
                # A transient Awake can bounce straight back to sleep on this PICO.
                if self.clock() - self.awake_since >= 5:
                    self.attempts = 0
                self._publish(
                    "awake", "Headset awake; waiting for live tracking", observation
                )
                return 1.0
            self.awake_since = None
            if observation.wakefulness != "Asleep":
                self._publish(
                    "paused", "Waiting for confirmed headset sleep or wake", observation
                )
                return 1.0
            if self.clock() < self.next_attempt:
                self._publish(
                    "waking", "Waking headset; waiting for confirmation", observation
                )
                return 1.0
            if self.attempts >= 3:
                self._publish(
                    "recovery_failed",
                    "Headset did not stay awake; press its power button",
                    observation,
                )
                return 1.0
            # Re-read presence immediately before each key event. Never reuse a NEAR
            # observation from the preceding poll or from an earlier wake attempt.
            observation = await self.observe()
            if observation.presence != "present" or observation.wakefulness != "Asleep":
                self._publish(
                    "paused", "Headset state changed; checking presence", observation
                )
                return 1.0
            self.attempts += 1
            self.next_attempt = self.clock() + 5
            await self.command(
                self.serial, ["shell", "input", "keyevent", "KEYCODE_WAKEUP"]
            )
            self._publish(
                "waking", "Waking headset; waiting for confirmation", observation
            )
            return 1.0
        except (OSError, TimeoutError) as exc:
            self.awake_since = None
            self.failures += 1
            self._publish(
                "unavailable", str(exc), PresenceObservation("unknown", "unknown")
            )
            return min(10.0, 2.0 ** min(self.failures - 1, 4))

    async def run(self) -> None:
        while True:
            delay = await self.step()
            await asyncio.sleep(delay)
