# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-lived ADB and OOB recovery owner for a single pinned headset."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass

from . import oob_teleop_adb as adb
from .oob_teleop_env import (
    USB_TURN_CREDENTIAL,
    USB_TURN_USER,
    redact_control_token,
    usb_backend_port,
    usb_ui_port,
)

log = logging.getLogger("oob-teleop-lifecycle")


@dataclass(frozen=True)
class RecoveryConfig:
    timeout_sec: float = 60.0
    interval_sec: float = 5.0


class DeviceReplacedError(RuntimeError):
    """A different ready serial appeared after the first headset was pinned."""

    def __init__(self, selected: str, replacement: str):
        self.selected = selected
        self.replacement = replacement
        super().__init__(
            f"DEVICE_REPLACED: selected {selected}, observed {replacement}"
        )


def _host_listener_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


class OobLifecycle:
    """Owns device setup, coturn repair, CDP monitoring, and browser verification."""

    def __init__(
        self,
        *,
        hub,
        resolved_port: int,
        usb_local: bool,
        host_client: bool,
        config: RecoveryConfig,
        turn_port: int | None = None,
        on_status: Callable[[dict], None] | None = None,
        on_fatal: Callable[[DeviceReplacedError], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = asyncio.sleep,
        metrics_stale_sec: float = 5.0,
        host_listener_probe: Callable[[int], bool] = _host_listener_ready,
    ) -> None:
        self.hub = hub
        self.resolved_port = resolved_port
        self.usb_local = usb_local
        self.host_client = host_client
        self.config = config
        self.turn_port = turn_port
        self.on_status = on_status
        self.on_fatal = on_fatal
        self.clock = clock
        self.sleep = sleep
        self.metrics_stale_sec = metrics_stale_sec
        self.host_listener_probe = host_listener_probe
        self.selected = os.environ.get("ANDROID_SERIAL", "").strip() or None
        self.explicit_serial = self.selected is not None
        self._selected_once = False
        self._cache_cleared = False
        self.generation = 0
        self.monitor: asyncio.Task | None = None
        self.coturn = None
        self.browser_client: str | None = None
        self.browser_ready = False
        self.connect_dispatched = False
        self.last_metrics_at: float | None = None
        self.last_stream_at: float | None = None
        self.connect_at: float | None = None
        self.browser_probe_after: float | None = None
        self.last_adb_at: float | None = None
        self.last_network_at: float | None = None
        self.last_rules_at: float | None = None
        self.last_browser_at: float | None = None
        self.last_turn_at: float | None = None
        self._turn_listener_ready = False
        self.last_network_state: adb.HeadsetNetworkState | None = None
        self.episode_start = self.clock()
        self.episode_wall_start = time.time()
        self.attempts = 0
        self._ready_count = 0
        self._last_observation: tuple | None = None
        self._last_reason = ""
        self._last_health: str | None = None
        self._prerequisite_signature: tuple | None = None
        self.snapshot: dict = {}

    def _restart_episode(self) -> None:
        self.episode_start = self.clock()
        self.episode_wall_start = time.time()
        self.attempts = 0

    async def _publish(self, health: str, state: str, reason: str, **flags) -> None:
        if health == "degraded" and self._last_health in {"browser_ready", "active"}:
            self._restart_episode()
        self._last_health = health
        reason = re.sub(r"https?://\S+", "<URL>", redact_control_token(reason))[:300]
        snapshot = {
            "schemaVersion": 1,
            "generation": self.generation,
            "health": health,
            "state": state,
            "reason": reason,
            "selectedSerial": self.selected,
            "explicitSerial": self.explicit_serial,
            "episodeStartedAt": self.episode_wall_start,
            "episodeDeadline": self.episode_wall_start + self.config.timeout_sec,
            "attemptCount": self.attempts,
            "updatedAt": time.time(),
            "lastAdbAt": self.last_adb_at,
            "lastNetworkAt": self.last_network_at,
            "lastRulesAt": self.last_rules_at,
            "lastTurnAt": self.last_turn_at,
            "lastBrowserAt": self.last_browser_at,
            "lastStreamAt": self.last_stream_at,
            "adbReady": False,
            "networkPresent": False,
            "reverseRulesVerified": False,
            "coturnProcessReady": bool(self.coturn and self.coturn.poll() is None),
            "coturnListenerReady": self._turn_listener_ready,
            "turnPrerequisitesReady": False,
            "turnEndToEndHealthy": False,
            "browserRegistered": self.browser_ready,
            "healthProbeAcknowledged": self.browser_ready,
            "browserReady": self.browser_ready,
            "connectDispatched": self.connect_dispatched,
            "streaming": False,
            "clientMetricsFresh": False,
            **flags,
        }
        self.snapshot = snapshot
        await self.hub.set_lifecycle_snapshot(snapshot)
        if self.on_status:
            self.on_status(dict(snapshot))
        if reason != self._last_reason:
            log.info("OOB %s: %s", state, reason)
            self._last_reason = reason

    async def _stop_monitor(self) -> None:
        had_monitor = self.monitor is not None
        if self.monitor is not None:
            self.monitor.cancel()
            try:
                await self.monitor
            except (asyncio.CancelledError, Exception):
                pass
            self.monitor = None
        if self.selected and had_monitor:
            try:
                await asyncio.to_thread(
                    adb._adb_run,
                    ["adb", "forward", "--remove", "tcp:9223"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                    check=False,
                )
            except Exception:
                log.debug("CDP forward cleanup failed", exc_info=True)

    async def _run_adb_command(self, command: list[str], *, timeout: float) -> object:
        """Finish an in-flight ADB mutation before cancellation cleanup runs."""
        task = asyncio.create_task(
            asyncio.to_thread(
                adb._adb_run,
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except Exception:
                pass
            raise

    async def _cleanup_owned_rules(self) -> None:
        """Remove only mappings for this lifecycle's selected serial."""
        if not self.selected:
            return
        commands = [["adb", "forward", "--remove", "tcp:9223"]]
        if self.usb_local and self.turn_port is not None:
            commands.extend(
                ["adb", "reverse", "--remove", f"tcp:{port}"]
                for port in (
                    usb_ui_port(),
                    self.resolved_port,
                    usb_backend_port(),
                    self.turn_port,
                )
            )
        for command in commands:
            try:
                await self._run_adb_command(command, timeout=1)
            except Exception:
                log.debug(
                    "Owned ADB mapping cleanup failed: %s", command, exc_info=True
                )

    async def _shielded_cleanup(self) -> None:
        task = asyncio.create_task(self._cleanup_owned_rules())
        await asyncio.shield(task)

    async def _ensure_coturn(self) -> bool:
        if not self.usb_local or self.turn_port is None:
            return False
        if self.coturn is not None and self.coturn.poll() is None:
            if await asyncio.to_thread(adb.verify_coturn_listening, self.turn_port):
                self._turn_listener_ready = True
                self.last_turn_at = time.time()
                return False
        self._turn_listener_ready = False
        if self.coturn is not None:
            await asyncio.to_thread(adb.stop_coturn, self.coturn)
        self.coturn = await asyncio.to_thread(
            adb.start_coturn, self.turn_port, USB_TURN_USER, USB_TURN_CREDENTIAL
        )
        if self.coturn is None or not await asyncio.to_thread(
            adb.verify_coturn_listening, self.turn_port
        ):
            raise adb.OobAdbError("coturn is not listening")
        self._turn_listener_ready = True
        self.last_turn_at = time.time()
        return True

    async def _rebuild_usb(self) -> None:
        assert self.turn_port is not None
        ports = [usb_ui_port(), self.resolved_port, usb_backend_port(), self.turn_port]
        await self._publish(
            "degraded",
            "REBUILDING_USB",
            "Rebuilding USB reverse rules",
            adbReady=True,
            networkPresent=True,
        )
        try:
            await self._stop_monitor()
            await self._run_adb_command(
                ["adb", "forward", "--remove", "tcp:9223"], timeout=1
            )
            # A partial attempt is removed before retry without stopping host services.
            for port in ports:
                await self._run_adb_command(
                    ["adb", "reverse", "--remove", f"tcp:{port}"], timeout=2
                )
            for port in ports[:3]:
                if not await asyncio.to_thread(self.host_listener_probe, port):
                    raise adb.OobAdbError(f"Host listener on tcp:{port} is unavailable")
            await self._ensure_coturn()
            for port in ports:
                proc = await self._run_adb_command(
                    ["adb", "reverse", f"tcp:{port}", f"tcp:{port}"], timeout=5
                )
                if proc.returncode:
                    raise adb.OobAdbError(
                        f"adb reverse tcp:{port}: {(proc.stderr or proc.stdout).strip()}"
                    )
            verification = await asyncio.to_thread(adb.probe_adb_reverse_rules, ports)
            if not verification.adb_available:
                raise adb.OobAdbError("ADB unavailable while verifying reverse rules")
            if verification.missing_ports:
                raise adb.OobAdbError(
                    f"USB reverse rules missing: {verification.missing_ports}"
                )
            self.last_rules_at = time.time()
        except (Exception, asyncio.CancelledError):
            await self._shielded_cleanup()
            raise
        await self._publish(
            "degraded",
            "AUTOMATING_BROWSER",
            "TURN and reverse rules verified",
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=True,
            turnPrerequisitesReady=True,
        )

    async def _prepare_device(self) -> None:
        """Wake the selected headset and clear stale browser state once."""
        await asyncio.to_thread(adb.assert_headset_awake, timeout=10.0)
        if not self._cache_cleared:
            try:
                await asyncio.to_thread(
                    adb.clear_headset_browser_cache, usb_local=self.usb_local
                )
            except Exception:
                log.warning(
                    "Browser cache cleanup unavailable; continuing", exc_info=True
                )
            self._cache_cleared = True

    async def _automate(self) -> None:
        await self._stop_monitor()
        self.generation += 1
        self.browser_ready = False
        self.browser_client = None
        self.connect_dispatched = False
        before = time.time()
        self.browser_probe_after = before
        await self._publish(
            "degraded",
            "AUTOMATING_BROWSER",
            "Opening browser and dispatching CONNECT",
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=self.usb_local,
            turnPrerequisitesReady=self.usb_local,
        )
        self.monitor = await adb.run_oob_connect(
            resolved_port=self.resolved_port,
            timeout=min(self.config.timeout_sec, 15.0),
            usb_local=self.usb_local,
            host_client=self.host_client,
        )
        self.connect_at = time.time()
        self.connect_dispatched = True
        await self._publish(
            "degraded",
            "VERIFYING_BROWSER",
            "Waiting for browser health report",
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=self.usb_local,
            turnPrerequisitesReady=self.usb_local,
        )
        await self._verify_browser()

    async def _verify_browser(self) -> None:
        """Wait for OOB proof after CONNECT without repeating the click on timeout."""
        if self.monitor is not None and self.monitor.done():
            raise adb.OobAdbError("CDP monitor ended")
        assert self.browser_probe_after is not None
        report = await self.hub.probe_browser(self.generation, self.browser_probe_after)
        if report is None:
            await self._publish(
                "degraded",
                "VERIFYING_BROWSER",
                "CONNECT dispatched; waiting for fresh browser health report",
                adbReady=True,
                networkPresent=True,
                reverseRulesVerified=self.usb_local,
                turnPrerequisitesReady=self.usb_local,
                connectDispatched=True,
            )
            return
        self.browser_client = report["clientId"]
        self.browser_ready = True
        self.last_browser_at = time.time()
        await self._publish(
            "browser_ready",
            "ACTIVE",
            "Browser ready; waiting for stream",
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=self.usb_local,
            turnPrerequisitesReady=self.usb_local,
            browserRegistered=True,
            healthProbeAcknowledged=True,
        )

    async def _observe_stream(self) -> None:
        state = await self.hub.get_snapshot()
        client = next(
            (h for h in state["headsets"] if h["clientId"] == self.browser_client), None
        )
        if client is None:
            self.browser_ready = False
            raise adb.OobAdbError("Browser OOB client disconnected")
        fresh = bool(
            client.get("lastMetricsAt")
            and (time.time() * 1000 - client["lastMetricsAt"])
            < self.metrics_stale_sec * 1000
        )
        after_connect = bool(
            client.get("lastMetricsAt")
            and self.connect_at
            and client["lastMetricsAt"] > self.connect_at * 1000
        )
        streaming = bool(client["streaming"])
        if streaming and not fresh:
            report = await self.hub.probe_browser(
                self.generation, 0, client_id=self.browser_client
            )
            if report is None:
                self.browser_ready = False
                raise adb.OobAdbError("Browser heartbeat lost while metrics were stale")
        health = "active" if streaming and fresh and after_connect else "browser_ready"
        if health == "active":
            self.last_stream_at = time.time()
        reason = (
            "Stream and fresh metrics confirmed"
            if health == "active"
            else "Browser ready; waiting for stream or fresh metrics"
        )
        await self._publish(
            health,
            "ACTIVE",
            reason,
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=self.usb_local,
            turnPrerequisitesReady=self.usb_local,
            browserRegistered=True,
            healthProbeAcknowledged=True,
            streaming=streaming,
            clientMetricsFresh=fresh,
            turnEndToEndHealthy=self.usb_local and health == "active",
        )

    async def _check_usb_prerequisites(self) -> None:
        """Rebuild on a real TURN or reverse-rule loss, even before OOB proof."""
        if not self.usb_local:
            return
        restarted = await self._ensure_coturn()
        verification = await asyncio.to_thread(
            adb.probe_adb_reverse_rules,
            [usb_ui_port(), self.resolved_port, usb_backend_port(), self.turn_port],
        )
        if not verification.adb_available:
            raise adb.OobAdbError("ADB unavailable while verifying reverse rules")
        if verification.missing_ports:
            raise adb.OobAdbError(
                f"USB reverse rules missing: {verification.missing_ports}"
            )
        if restarted:
            raise adb.OobAdbError("coturn restarted; renewing browser connection")

    async def run(self) -> None:
        token = adb.SELECTED_ADB_SERIAL.set(self.selected) if self.selected else None
        try:
            if not self.snapshot:
                await self._publish(
                    "starting", "WAITING_FOR_ADB", "Waiting for headset"
                )
            while True:
                devices = await asyncio.to_thread(adb.enumerate_adb_devices)
                ready = devices.ready
                observed_serials = {serial for serial, _ in devices.devices}
                if (
                    self._selected_once
                    and self.selected not in observed_serials
                    and ready
                ):
                    error = DeviceReplacedError(self.selected, ready[0])
                    await self._publish(
                        "fatal",
                        "FATAL",
                        "DEVICE_REPLACED",
                        expectedSerial=self.selected,
                        replacementSerial=ready[0],
                    )
                    if self.on_fatal:
                        self.on_fatal(error)
                    raise error
                if self.selected is None and len(ready) == 1:
                    self.selected = ready[0]
                    token = adb.SELECTED_ADB_SERIAL.set(self.selected)
                if self.selected in ready:
                    self._selected_once = True
                observation = (tuple(devices.devices), devices.diagnostic)
                if observation != self._last_observation:
                    self._restart_episode()
                    self._ready_count = 0
                    self._last_observation = observation
                    if self.selected in ready and len(ready) > 1:
                        log.warning(
                            "Extra ADB device present; continuing with pinned serial %s",
                            self.selected,
                        )
                selected_ready = bool(self.selected and self.selected in ready)
                if not selected_ready:
                    await self._stop_monitor()
                    self.browser_ready = False
                    self.connect_dispatched = False
                    self.browser_client = None
                    self._ready_count = 0
                    states = dict(devices.devices)
                    if len(ready) > 1 and self.selected is None:
                        reason = "Multiple ready devices; unplug extras or set ANDROID_SERIAL"
                    elif "unauthorized" in states.values():
                        reason = "Headset unauthorized; accept the USB debugging prompt"
                    elif "offline" in states.values():
                        reason = "Headset offline; reconnect the USB cable"
                    else:
                        reason = devices.diagnostic or "Waiting for selected headset"
                    await self._publish("degraded", "WAITING_FOR_ADB", reason)
                    await self.sleep(self.config.interval_sec)
                    continue
                self._ready_count += 1
                self.last_adb_at = time.time()
                if self._ready_count < 2:
                    await self.sleep(self.config.interval_sec)
                    continue
                network = await asyncio.to_thread(
                    adb.probe_headset_network, serial=self.selected
                )
                if network.state is not self.last_network_state:
                    self._restart_episode()
                    self.last_network_state = network.state
                if network.state is adb.HeadsetNetworkState.ADB_UNAVAILABLE:
                    await self._stop_monitor()
                    self.browser_ready = False
                    self.connect_dispatched = False
                    self.browser_client = None
                    await self._publish(
                        "degraded",
                        "WAITING_FOR_ADB",
                        "ADB unavailable: " + network.diagnostic,
                    )
                    await self.sleep(self.config.interval_sec)
                    continue
                if network.state is adb.HeadsetNetworkState.NETWORK_PRESENT:
                    self.last_network_at = time.time()
                if (
                    self.usb_local
                    and network.state is adb.HeadsetNetworkState.NO_NETWORK
                ):
                    await self._stop_monitor()
                    self.browser_ready = False
                    self.connect_dispatched = False
                    self.browser_client = None
                    await self._publish(
                        "degraded",
                        "PREPARING_DEVICE",
                        "Headset has no non-loopback network",
                        adbReady=True,
                    )
                    await self.sleep(self.config.interval_sec)
                    continue
                if not self.browser_ready:
                    state = await self.hub.get_snapshot()
                    clients = tuple(sorted(h["clientId"] for h in state["headsets"]))
                    rules = None
                    if self.usb_local:
                        rules = await asyncio.to_thread(
                            adb._run_adb,
                            "reverse observe",
                            ["adb", "reverse", "--list"],
                        )
                    signature = (
                        clients,
                        rules,
                        self.coturn.poll() if self.coturn else None,
                    )
                    if (
                        self._prerequisite_signature is not None
                        and signature != self._prerequisite_signature
                    ):
                        self._restart_episode()
                    self._prerequisite_signature = signature
                if (
                    self.clock() >= self.episode_start + self.config.timeout_sec
                    and not self.browser_ready
                    and not self.connect_dispatched
                ):
                    await self._publish(
                        "degraded",
                        "WAITING_FOR_ADB",
                        "Recovery episode expired; observing for a change",
                        adbReady=True,
                        networkPresent=True,
                    )
                    await self.sleep(self.config.interval_sec)
                    continue
                try:
                    if self.browser_ready or self.connect_dispatched:
                        await self._check_usb_prerequisites()
                    if self.browser_ready:
                        if self.monitor is not None and self.monitor.done():
                            raise adb.OobAdbError("CDP monitor ended")
                        await self._observe_stream()
                    elif self.connect_dispatched:
                        await self._verify_browser()
                    else:
                        self.attempts += 1
                        await self._publish(
                            "degraded",
                            "PREPARING_DEVICE",
                            "Preparing selected headset",
                            adbReady=True,
                            networkPresent=True,
                        )
                        remaining = (
                            self.episode_start + self.config.timeout_sec - self.clock()
                        )
                        async with asyncio.timeout(max(0.001, remaining)):
                            await self._prepare_device()
                            if self.usb_local:
                                await self._rebuild_usb()
                            await self._automate()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._stop_monitor()
                    self.browser_ready = False
                    self.connect_dispatched = False
                    self.browser_client = None
                    reason = (
                        "Recovery attempt timed out"
                        if isinstance(exc, TimeoutError)
                        else str(exc)
                    )
                    await self._publish(
                        "degraded",
                        "DEGRADED",
                        reason[:300],
                        adbReady=True,
                        networkPresent=True,
                    )
                await self.sleep(self.config.interval_sec)
        finally:
            await self._stop_monitor()
            await self._shielded_cleanup()
            if self.coturn is not None:
                await asyncio.to_thread(adb.stop_coturn, self.coturn)
            if token is not None:
                adb.SELECTED_ADB_SERIAL.reset(token)
