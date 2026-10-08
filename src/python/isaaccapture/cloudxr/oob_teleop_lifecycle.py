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
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from dataclasses import dataclass

from . import oob_teleop_adb as adb
from .oob_teleop_env import (
    USB_TURN_CREDENTIAL,
    USB_TURN_USER,
    redact_control_token,
    usb_backend_port,
)

log = logging.getLogger(__name__)


class _TransportDisrupted(adb.OobAdbError):
    """A USB-local substrate failed while browser survival remains possible."""

    def __init__(self, reason: str, *, adb_ready: bool, network_present: bool) -> None:
        super().__init__(reason)
        self.adb_ready = adb_ready
        self.network_present = network_present


@dataclass(frozen=True)
class RecoveryConfig:
    timeout_sec: float = 60.0
    interval_sec: float = 5.0
    client_reconnect_enabled: bool = True
    client_reconnect_max_attempts: int = 10
    client_reconnect_delay_ms: int = 3000

    @property
    def client_recovery_grace_sec(self) -> float:
        """Bound host takeover so the browser gets its configured retry budget first."""
        if not self.client_reconnect_enabled:
            return self.interval_sec
        return max(
            self.interval_sec,
            self.client_reconnect_max_attempts * self.client_reconnect_delay_ms / 1000.0
            + 10.0,
        )


def _display_serial(serial: str) -> str:
    """Bound untrusted ADB serials before putting them in status or logs."""
    return "".join(c if c.isprintable() and c not in "\r\n" else "?" for c in serial)[
        :80
    ]


def _display_serials(serials: tuple[str, ...]) -> list[str]:
    return [_display_serial(serial) for serial in serials[:8]]


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
        on_status: Callable[[dict], Awaitable[None] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = asyncio.sleep,
        metrics_stale_sec: float = 5.0,
        host_listener_probe: Callable[[int], bool] = _host_listener_ready,
    ) -> None:
        # Immutable topology and injected collaborators.
        self.hub = hub
        self.resolved_port = resolved_port
        self.usb_local = usb_local
        self.host_client = host_client
        self.config = config
        self.turn_port = turn_port
        self.on_status = on_status
        self.clock = clock
        self.sleep = sleep
        self.metrics_stale_sec = metrics_stale_sec
        self.host_listener_probe = host_listener_probe

        # Headset selection remains pinned after the first successful choice.
        self.selected = os.environ.get("ANDROID_SERIAL", "").strip() or None
        self.explicit_serial = self.selected is not None
        self.ignored_serials: tuple[str, ...] = ()
        self._cache_cleared = False

        # Browser, relay, and CONNECT state rebuilt after transport loss.
        self.generation = 0
        self.monitor: asyncio.Task | None = None
        self.coturn = None
        self.browser_client: str | None = None
        self.browser_ready = False
        self.client_loaded = False
        self.connect_dispatched = False
        self.last_metrics_at: float | None = None
        self.last_stream_at: float | None = None
        self.connect_at: float | None = None
        self.browser_probe_after: float | None = None
        self._transport_lost = False
        self._restore_existing_browser = False
        self._repair_started_at: float | None = None
        self._client_grace_deadline: float | None = None
        self._fresh_stream_without_cdp = False
        self._transport_disruption_signature: tuple | None = None
        self._handled_terminal_events: set[str] = set()
        self._soft_click_keys: set[str] = set()

        # Wall-clock status timestamps and latest prerequisite observations.
        self.last_adb_at: float | None = None
        self.last_network_at: float | None = None
        self.last_rules_at: float | None = None
        self.last_browser_at: float | None = None
        self.last_turn_at: float | None = None
        self._turn_listener_ready = False
        self.last_network_state: adb.HeadsetNetworkState | None = None

        # Monotonic time enforces the retry budget; wall time is for API consumers.
        self.episode_start = self.clock()
        self.episode_wall_start = time.time()
        self.attempts = 0
        self._ready_count = 0
        self._last_observation: tuple | None = None
        self._last_ignored_warning: tuple | None = None
        self._last_reason = ""
        self._last_health: str | None = None
        self._readiness_stage: str | None = None
        self._readiness_since: float | None = None
        self._prerequisite_signature: tuple | None = None
        self.snapshot: dict = {}

    def _restart_episode(self) -> None:
        self.episode_start = self.clock()
        self.episode_wall_start = time.time()
        self.attempts = 0

    def _bounded_client_grace_deadline(self) -> float:
        """Fit browser-owned retry grace inside the host recovery episode."""
        now = self.clock()
        desired = now + self.config.client_recovery_grace_sec
        # Reserve one observation cadence for the host fallback. Very short
        # episodes therefore take over immediately instead of making fallback
        # unreachable behind a longer browser retry policy.
        latest = self.episode_start + self.config.timeout_sec - self.config.interval_sec
        return min(desired, max(now, latest))

    async def _publish(self, health: str, state: str, reason: str, **flags) -> None:
        # Failure after a recovered browser/active session gets a fresh retry budget.
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
            "selectedSerial": _display_serial(self.selected) if self.selected else None,
            "explicitSerial": self.explicit_serial,
            "ignoredSerials": _display_serials(self.ignored_serials),
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
            "lastMetricsAt": self.last_metrics_at,
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
        transport_ready = bool(
            snapshot["adbReady"]
            and snapshot["networkPresent"]
            and (
                not self.usb_local
                or (
                    snapshot["reverseRulesVerified"]
                    and snapshot["turnPrerequisitesReady"]
                )
            )
        )
        client_loaded = transport_ready and (self.client_loaded or self.browser_ready)
        dispatched = client_loaded and snapshot["connectDispatched"]
        confirmed = bool(
            transport_ready
            and health == "active"
            and snapshot["streaming"]
            and snapshot["clientMetricsFresh"]
        )
        stage = (
            "streamConfirmed"
            if confirmed
            else "connectDispatched"
            if dispatched
            else "clientLoaded"
            if client_loaded
            else "transportReady"
            if transport_ready
            else "socketBound"
        )
        if stage != self._readiness_stage:
            self._readiness_stage = stage
            self._readiness_since = snapshot["updatedAt"]
        snapshot.update(
            socketBound=True,
            transportReady=transport_ready,
            clientLoaded=client_loaded,
            connectDispatched=dispatched,
            streamConfirmed=confirmed,
            readinessStage=stage,
            readinessSince=self._readiness_since,
        )
        self.snapshot = snapshot
        await self.hub.set_lifecycle_snapshot(snapshot)
        if self.on_status:
            publication = self.on_status(dict(snapshot))
            if isawaitable(publication):
                await publication
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
                # Shutdown must continue even if the monitor already failed.
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

    async def _remember_transport_loss(self) -> None:
        """Drop transport-owned state while preserving the prior browser intent."""
        if not self._transport_lost:
            self._restore_existing_browser = bool(
                self.browser_ready
                or self.connect_dispatched
                or self.last_stream_at is not None
            )
            self._transport_lost = self.usb_local and self._restore_existing_browser
            self._repair_started_at = None
            self._client_grace_deadline = None
            self._fresh_stream_without_cdp = False
        await self._stop_monitor()
        self.browser_ready = False
        self.browser_client = None
        self.client_loaded = False
        self.connect_dispatched = False

    def _invalidate_completed_repair(self, signature: tuple) -> bool:
        """Forget post-repair evidence once for each distinct hard disruption."""
        if signature == self._transport_disruption_signature:
            return False
        self._transport_disruption_signature = signature
        self._repair_started_at = None
        self._client_grace_deadline = None
        self._fresh_stream_without_cdp = False
        self._restart_episode()
        return True

    async def _enter_transport_recovery(
        self,
        reason: str,
        state: str,
        *,
        adb_ready: bool,
        network_present: bool,
    ) -> None:
        """Preserve browser intent and publish one USB-loss transition."""
        await self._remember_transport_loss()
        if self._transport_lost:
            self._invalidate_completed_repair(
                (state, reason, adb_ready, network_present)
            )
        await self._publish(
            "degraded",
            state,
            reason,
            adbReady=adb_ready,
            networkPresent=network_present,
        )

    def _transport_retry_interval(self) -> float:
        """Poll an interrupted active session promptly without busy looping."""
        if self._restore_existing_browser:
            return min(self.config.interval_sec, 1.0)
        return self.config.interval_sec

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
                # Preserve the original cancellation after the ADB task settles.
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
        """Ensure the TURN listener is healthy; return whether a process was launched."""
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
        ports = [self.resolved_port, usb_backend_port(), self.turn_port]
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
            for port in ports[:2]:
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
            (
                "VERIFYING_EXISTING_BROWSER"
                if self._restore_existing_browser
                else "AUTOMATING_BROWSER"
            ),
            "TURN and reverse rules verified",
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=True,
            turnPrerequisitesReady=True,
        )
        # The repaired substrate is the new baseline. A later identical cable
        # loss or topology change is therefore a new disruption, while
        # repeated observations before repair remain deduplicated.
        self._transport_disruption_signature = None
        self._prerequisite_signature = None

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
        self._transport_lost = False
        self._restore_existing_browser = False
        self._repair_started_at = None
        self._client_grace_deadline = None
        self._fresh_stream_without_cdp = False
        self._transport_disruption_signature = None
        self.generation += 1
        self.browser_ready = False
        self.browser_client = None
        self.client_loaded = False
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

        async def on_client_loaded() -> None:
            self.client_loaded = True
            await self._publish(
                "degraded",
                "CLIENT_LOADED",
                "Client loaded; waiting for CONNECT",
                adbReady=True,
                networkPresent=True,
                reverseRulesVerified=self.usb_local,
                turnPrerequisitesReady=self.usb_local,
                connectDispatched=False,
            )

        def on_dispatched() -> Awaitable[None]:
            self.connect_at = time.time()
            self.connect_dispatched = True
            return self._publish(
                "degraded",
                "CONNECT_DISPATCHED",
                "CONNECT dispatched; waiting for stream",
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
            on_dispatched=on_dispatched,
            on_client_loaded=on_client_loaded,
        )
        # A successful return still establishes dispatch if an adapter omits the callback.
        if not self.connect_dispatched:
            await on_dispatched()
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

    async def _attach_existing_monitor(self) -> bool:
        """Reattach passive CDP monitoring without treating failure as page loss."""
        if self.monitor is not None and not self.monitor.done():
            return True
        self.monitor = None
        try:
            self.monitor = await adb.attach_existing_oob_tab(click_connect=False)
        except adb.OobAdbError:
            log.info(
                "Existing browser is healthy through OOB but CDP is not attachable"
            )
            return False
        return True

    async def _same_tab_connect(self, key: str) -> bool:
        """Dispatch at most one trusted CONNECT click for a terminal/grace event."""
        if key in self._soft_click_keys:
            return False
        self._soft_click_keys.add(key)
        await self._stop_monitor()
        self.connect_dispatched = False

        async def on_client_loaded() -> None:
            self.client_loaded = True
            await self._publish(
                "degraded",
                "CLIENT_LOADED",
                "Existing browser CONNECT control is actionable",
                adbReady=True,
                networkPresent=True,
                reverseRulesVerified=self.usb_local,
                turnPrerequisitesReady=self.usb_local,
            )

        def on_dispatched() -> Awaitable[None]:
            self.connect_at = time.time()
            self.connect_dispatched = True

            async def publish() -> None:
                if not self.client_loaded:
                    await on_client_loaded()
                await self._publish(
                    "degraded",
                    "CONNECT_DISPATCHED",
                    "Existing browser CONNECT dispatched; waiting for fresh stream evidence",
                    adbReady=True,
                    networkPresent=True,
                    reverseRulesVerified=self.usb_local,
                    turnPrerequisitesReady=self.usb_local,
                    connectDispatched=True,
                )

            return publish()

        try:
            self.monitor = await adb.attach_existing_oob_tab(
                click_connect=True,
                on_client_loaded=on_client_loaded,
                on_dispatched=on_dispatched,
            )
        except adb.OobAdbError:
            self.monitor = None
            return False
        if not self.connect_dispatched:
            await on_dispatched()
        self._client_grace_deadline = self._bounded_client_grace_deadline()
        await self._publish(
            "degraded",
            "VERIFYING_EXISTING_BROWSER",
            "Existing browser CONNECT dispatched; waiting for fresh stream evidence",
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=self.usb_local,
            turnPrerequisitesReady=self.usb_local,
            connectDispatched=True,
        )
        return True

    async def _recover_existing_browser(self) -> None:
        """Restore a surviving page after USB repair before mutating browser state."""
        assert self._repair_started_at is not None
        assert self._client_grace_deadline is not None
        # Restore the host-owned CDP forward and passive monitor promptly. A
        # missing socket/tab is evidence for the bounded fallback below, not a
        # reason to mutate browser state yet.
        cdp_attached = await self._attach_existing_monitor()
        report = await self.hub.probe_browser(
            self.generation,
            0,
            timeout=min(self.config.interval_sec, 2.0),
        )
        if report is None:
            self._fresh_stream_without_cdp = False
            if self.clock() < self._client_grace_deadline:
                await self._publish(
                    "degraded",
                    "VERIFYING_EXISTING_BROWSER",
                    "USB repaired; waiting for the existing browser control channel",
                    adbReady=True,
                    networkPresent=True,
                    reverseRulesVerified=True,
                    turnPrerequisitesReady=True,
                )
                return
            if await self._same_tab_connect(f"grace:{self.generation}"):
                return
            # No usable tab survived the bounded grace. Only now perform the
            # destructive close/navigate/bootstrap path.
            await self._automate()
            return

        self.browser_client = report["clientId"]
        self.browser_ready = True
        self.client_loaded = True
        self.last_browser_at = time.time()
        phase = report.get("streamPhase") or (
            "streaming" if report.get("streaming") else "idle"
        )
        terminal_event = report.get("terminalEventId")
        metrics_at = report.get("lastMetricsAt")
        self.last_metrics_at = metrics_at
        fresh_post_repair = bool(
            metrics_at
            and metrics_at > self._repair_started_at * 1000
            and time.time() * 1000 - metrics_at < self.metrics_stale_sec * 1000
        )
        self._fresh_stream_without_cdp = bool(
            report.get("streaming") and fresh_post_repair and not cdp_attached
        )
        if report.get("streaming") and fresh_post_repair and cdp_attached:
            self.last_stream_at = time.time()
            self._transport_lost = False
            self._restore_existing_browser = False
            self._fresh_stream_without_cdp = False
            self._transport_disruption_signature = None
            self.connect_dispatched = True
            await self._publish(
                "active",
                "ACTIVE",
                "Existing browser resumed after USB repair",
                adbReady=True,
                networkPresent=True,
                reverseRulesVerified=True,
                turnPrerequisitesReady=True,
                browserRegistered=True,
                healthProbeAcknowledged=True,
                streaming=True,
                clientMetricsFresh=True,
                turnEndToEndHealthy=True,
                streamPhase=phase,
            )
            return
        if report.get("streaming") and fresh_post_repair:
            await self._publish(
                "degraded",
                "VERIFYING_EXISTING_BROWSER",
                "Stream resumed; retrying passive CDP monitoring",
                adbReady=True,
                networkPresent=True,
                reverseRulesVerified=True,
                turnPrerequisitesReady=True,
                browserRegistered=True,
                healthProbeAcknowledged=True,
                streaming=True,
                clientMetricsFresh=True,
                turnEndToEndHealthy=True,
                streamPhase=phase,
            )
            return
        if terminal_event and terminal_event not in self._handled_terminal_events:
            self._handled_terminal_events.add(terminal_event)
            if await self._same_tab_connect(f"terminal:{terminal_event}"):
                return
        if (
            phase == "terminal"
            and not self.connect_dispatched
            and self.clock() >= self._client_grace_deadline
        ):
            # The terminal event was handled once, but no usable CDP tab
            # accepted the same-tab click. Fall back only after the grace.
            await self._automate()
            return
        if (
            self.clock() >= self._client_grace_deadline
            and phase != "terminal"
            and not self.connect_dispatched
        ):
            if await self._same_tab_connect(f"grace:{self.generation}"):
                return
            # The control client survived, but without an attachable CDP tab
            # there is nowhere to dispatch the one trusted same-tab click.
            # Leave preservation mode through the normal full bootstrap so a
            # stable report cannot send us around this fallback again.
            await self._automate()
            return
        await self._attach_existing_monitor()
        await self._publish(
            "browser_ready",
            "VERIFYING_EXISTING_BROWSER",
            (
                "Existing browser is retrying the stream"
                if phase == "retrying"
                else "Existing browser is alive; waiting for fresh stream evidence"
            ),
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=True,
            turnPrerequisitesReady=True,
            browserRegistered=True,
            healthProbeAcknowledged=True,
            streaming=False,
            clientMetricsFresh=False,
            streamPhase=phase,
            terminalEventId=terminal_event,
        )

    async def _observe_passive_recovery(self) -> None:
        """Observe a repaired surviving page after the destructive budget expires."""
        assert self._repair_started_at is not None
        probe_kwargs = {}
        if self.browser_client is not None:
            probe_kwargs["client_id"] = self.browser_client
        report = await self.hub.probe_browser(
            self.generation,
            0,
            timeout=min(self.config.interval_sec, 2.0),
            **probe_kwargs,
        )
        if report is None:
            await self._publish(
                "degraded",
                "VERIFYING_EXISTING_BROWSER",
                "Recovery episode expired; observing the existing browser",
                adbReady=True,
                networkPresent=True,
                reverseRulesVerified=True,
                turnPrerequisitesReady=True,
            )
            return

        self.browser_client = report["clientId"]
        self.browser_ready = True
        self.client_loaded = True
        self.last_browser_at = time.time()
        phase = report.get("streamPhase") or (
            "streaming" if report.get("streaming") else "idle"
        )
        metrics_at = report.get("lastMetricsAt")
        self.last_metrics_at = metrics_at
        fresh_post_repair = bool(
            metrics_at
            and metrics_at > self._repair_started_at * 1000
            and time.time() * 1000 - metrics_at < self.metrics_stale_sec * 1000
        )
        if report.get("streaming") and fresh_post_repair:
            self.last_stream_at = time.time()
            self._transport_lost = False
            self._restore_existing_browser = False
            self._fresh_stream_without_cdp = False
            self._transport_disruption_signature = None
            await self._publish(
                "active",
                "ACTIVE",
                "Existing browser resumed after the recovery deadline",
                adbReady=True,
                networkPresent=True,
                reverseRulesVerified=True,
                turnPrerequisitesReady=True,
                browserRegistered=True,
                healthProbeAcknowledged=True,
                streaming=True,
                clientMetricsFresh=True,
                turnEndToEndHealthy=True,
                streamPhase=phase,
            )
            return
        await self._publish(
            "degraded",
            "VERIFYING_EXISTING_BROWSER",
            "Recovery episode expired; observing the existing browser",
            adbReady=True,
            networkPresent=True,
            reverseRulesVerified=True,
            turnPrerequisitesReady=True,
            browserRegistered=True,
            healthProbeAcknowledged=True,
            streaming=bool(report.get("streaming")),
            clientMetricsFresh=fresh_post_repair,
            streamPhase=phase,
            terminalEventId=report.get("terminalEventId"),
        )

    async def _verify_browser(self) -> None:
        """Wait for OOB proof after CONNECT without repeating the click on timeout."""
        if self.monitor is not None and self.monitor.done():
            if self.usb_local:
                raise _TransportDisrupted(
                    "CDP monitor ended while verifying the browser",
                    adb_ready=True,
                    network_present=True,
                )
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
        self.client_loaded = True
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
            if self.usb_local:
                raise _TransportDisrupted(
                    "Browser OOB control client disconnected",
                    adb_ready=True,
                    network_present=True,
                )
            self.browser_ready = False
            raise adb.OobAdbError("Browser OOB client disconnected")
        if self.monitor is not None and self.monitor.done():
            self.monitor = None
        if self.monitor is None:
            attached = await self._attach_existing_monitor()
            if self.usb_local and not attached:
                raise _TransportDisrupted(
                    "CDP monitor unavailable for the existing browser",
                    adb_ready=True,
                    network_present=True,
                )
        phase = client.get("streamPhase") or (
            "streaming" if client.get("streaming") else "idle"
        )
        if (
            self.usb_local
            and phase == "retrying"
            and not self._restore_existing_browser
        ):
            raise _TransportDisrupted(
                "Browser is retrying after a stream transport disruption",
                adb_ready=True,
                network_present=True,
            )
        terminal_event = client.get("terminalEventId")
        if terminal_event and terminal_event not in self._handled_terminal_events:
            if self.usb_local:
                # Repair host-owned substrate before trusting any browser
                # fallback. The terminal remains unseen so post-repair logic
                # can dispatch at most one same-tab CONNECT if still needed.
                raise _TransportDisrupted(
                    f"Browser stream reached terminal event {terminal_event}",
                    adb_ready=True,
                    network_present=True,
                )
            self._handled_terminal_events.add(terminal_event)
            if not await self._same_tab_connect(f"terminal:{terminal_event}"):
                # Continue the bounded existing-browser path without rebuilding
                # transport that is already healthy. At grace expiry this may
                # bootstrap only if no usable CDP tab can be recovered.
                self._transport_lost = True
                self._restore_existing_browser = True
                self._repair_started_at = time.time()
                self._client_grace_deadline = self._bounded_client_grace_deadline()
            return
        # Snapshot timestamps are epoch milliseconds; local time.time() values are seconds.
        self.last_metrics_at = client.get("lastMetricsAt")
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
                if self.usb_local:
                    raise _TransportDisrupted(
                        "Browser control heartbeat lost while metrics were stale",
                        adb_ready=True,
                        network_present=True,
                    )
                self.browser_ready = False
                raise adb.OobAdbError("Browser heartbeat lost while metrics were stale")
        health = "active" if streaming and fresh and after_connect else "browser_ready"
        if health == "active":
            self.last_stream_at = time.time()
        reason = (
            "Stream and fresh metrics confirmed"
            if health == "active"
            else (
                "Browser is retrying the stream"
                if phase == "retrying"
                else "Browser ready; waiting for stream or fresh metrics"
            )
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
            streamPhase=phase,
            terminalEventId=terminal_event,
        )

    async def _check_usb_prerequisites(self) -> None:
        """Rebuild on a real TURN or reverse-rule loss, even before OOB proof."""
        if not self.usb_local:
            return
        try:
            restarted = await self._ensure_coturn()
        except adb.OobAdbError as exc:
            raise _TransportDisrupted(
                f"TURN prerequisite unavailable: {exc}",
                adb_ready=True,
                network_present=True,
            ) from exc
        verification = await asyncio.to_thread(
            adb.probe_adb_reverse_rules,
            [self.resolved_port, usb_backend_port(), self.turn_port],
        )
        if not verification.adb_available:
            raise _TransportDisrupted(
                "ADB unavailable while verifying reverse rules",
                adb_ready=False,
                network_present=False,
            )
        if verification.missing_ports:
            raise _TransportDisrupted(
                f"USB reverse rules missing: {verification.missing_ports}",
                adb_ready=True,
                network_present=True,
            )
        if restarted:
            raise _TransportDisrupted(
                "coturn restarted; renewing browser connection",
                adb_ready=True,
                network_present=True,
            )

    async def run(self) -> None:
        """Keep one headset recovered through ADB, browser, and streaming states.

        The first ready headset stays pinned while unrelated devices are ignored. Recoverable
        failures run bounded retry episodes, then observe until a meaningful change starts a
        new episode. The loop runs until cancelled and always cleans up its owned resources.
        """
        token = adb.SELECTED_ADB_SERIAL.set(self.selected) if self.selected else None
        try:
            if not self.snapshot:
                await self._publish(
                    "starting", "WAITING_FOR_ADB", "Waiting for headset"
                )
            while True:
                # WAITING_FOR_ADB: select once, then wait only for that headset.
                devices = await asyncio.to_thread(adb.enumerate_adb_devices)
                ready = devices.ready
                if self.selected is None and len(ready) == 1:
                    self.selected = ready[0]
                    token = adb.SELECTED_ADB_SERIAL.set(self.selected)
                self.ignored_serials = (
                    tuple(
                        sorted(
                            serial
                            for serial, _ in devices.devices
                            if serial != self.selected
                        )
                    )
                    if self.selected
                    else ()
                )
                selected_state = dict(devices.devices).get(self.selected)
                observation = (tuple(sorted(devices.devices)), devices.diagnostic)
                if observation != self._last_observation:
                    self._restart_episode()
                    self._ready_count = 0
                    self._last_observation = observation
                warning_signature = (self.ignored_serials, selected_state)
                if warning_signature != self._last_ignored_warning:
                    self._last_ignored_warning = warning_signature
                    if self.ignored_serials:
                        ignored = ", ".join(_display_serials(self.ignored_serials))
                        if len(self.ignored_serials) > 8:
                            ignored += f", and {len(self.ignored_serials) - 8} more"
                        if selected_state == "device":
                            log.warning(
                                "Ignored ADB device(s) %s; continuing with selected headset %s.",
                                ignored,
                                _display_serial(self.selected),
                            )
                        else:
                            if selected_state in {"offline", "unauthorized"}:
                                log.warning(
                                    "Ignored ADB device(s) %s; selected headset %s is %s; waiting for it to reconnect.",
                                    ignored,
                                    _display_serial(self.selected),
                                    selected_state,
                                )
                            else:
                                log.warning(
                                    "Ignored ADB device(s) %s; waiting for selected headset %s to reconnect.",
                                    ignored,
                                    _display_serial(self.selected),
                                )
                selected_ready = bool(self.selected and self.selected in ready)
                if not selected_ready:
                    self._ready_count = 0
                    states = dict(devices.devices)
                    if len(ready) > 1 and self.selected is None:
                        reason = "Multiple ready devices; unplug extras or set ANDROID_SERIAL"
                    elif self.selected and states.get(self.selected) == "unauthorized":
                        reason = "Selected headset unauthorized; accept the USB debugging prompt"
                    elif self.selected and states.get(self.selected) == "offline":
                        reason = "Selected headset offline; reconnect the USB cable"
                    elif self.selected:
                        reason = "Waiting for selected headset to reconnect"
                    elif "unauthorized" in states.values():
                        reason = "Headset unauthorized; accept the USB debugging prompt"
                    elif "offline" in states.values():
                        reason = "Headset offline; reconnect the USB cable"
                    else:
                        reason = devices.diagnostic or "Waiting for selected headset"
                    await self._enter_transport_recovery(
                        reason,
                        "WAITING_FOR_ADB",
                        adb_ready=False,
                        network_present=False,
                    )
                    await self.sleep(self._transport_retry_interval())
                    continue
                self._ready_count += 1
                self.last_adb_at = time.time()
                if self._ready_count < 2 and not self._restore_existing_browser:
                    await self.sleep(self._transport_retry_interval())
                    continue
                network = await asyncio.to_thread(
                    adb.probe_headset_network, serial=self.selected
                )
                if network.state is not self.last_network_state:
                    self._restart_episode()
                    self.last_network_state = network.state
                if network.state is adb.HeadsetNetworkState.ADB_UNAVAILABLE:
                    await self._enter_transport_recovery(
                        "ADB unavailable: " + network.diagnostic,
                        "WAITING_FOR_ADB",
                        adb_ready=False,
                        network_present=False,
                    )
                    await self.sleep(self._transport_retry_interval())
                    continue
                if network.state is adb.HeadsetNetworkState.NETWORK_PRESENT:
                    self.last_network_at = time.time()
                if network.state is adb.HeadsetNetworkState.NO_NETWORK:
                    # An absent non-loopback interface cannot satisfy transport
                    # readiness or support browser automation in either mode.
                    await self._enter_transport_recovery(
                        "Headset has no non-loopback network",
                        "PREPARING_DEVICE",
                        adb_ready=True,
                        network_present=False,
                    )
                    await self.sleep(self._transport_retry_interval())
                    continue
                preserving_browser = (
                    self._transport_lost and self._restore_existing_browser
                )
                if not self.browser_ready or preserving_browser:
                    state = await self.hub.get_snapshot()
                    clients = tuple(
                        sorted(
                            str(headset["clientId"]) for headset in state["headsets"]
                        )
                    )
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
                    signature_changed = (
                        self._prerequisite_signature is not None
                        and signature != self._prerequisite_signature
                    )
                    topology_changed = bool(
                        signature_changed
                        and preserving_browser
                        and signature[1:] != self._prerequisite_signature[1:]
                    )
                    if topology_changed:
                        self._invalidate_completed_repair(
                            ("topology", rules, signature[2])
                        )
                    elif signature_changed:
                        self._restart_episode()
                    self._prerequisite_signature = signature
                if (
                    preserving_browser
                    and self.clock() >= self.episode_start + self.config.timeout_sec
                ):
                    if self._repair_started_at is not None:
                        await self._observe_passive_recovery()
                    else:
                        await self._publish(
                            "degraded",
                            "VERIFYING_EXISTING_BROWSER",
                            "Recovery episode expired; observing for a change",
                            adbReady=True,
                            networkPresent=True,
                        )
                    await self.sleep(self._transport_retry_interval())
                    continue
                if preserving_browser:
                    try:
                        if self._repair_started_at is None:
                            self.attempts += 1
                            await self._publish(
                                "degraded",
                                "REBUILDING_USB",
                                "Selected headset returned; repairing USB transport",
                                adbReady=True,
                                networkPresent=True,
                            )
                            await asyncio.to_thread(
                                adb.assert_headset_awake, timeout=10.0
                            )
                            await self._rebuild_usb()
                            self._repair_started_at = time.time()
                            self._client_grace_deadline = (
                                self._bounded_client_grace_deadline()
                            )
                        await self._recover_existing_browser()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self._repair_started_at = None
                        self._client_grace_deadline = None
                        await self._publish(
                            "degraded",
                            "REBUILDING_USB",
                            str(exc)[:300],
                            adbReady=True,
                            networkPresent=True,
                        )
                    await self.sleep(self._transport_retry_interval())
                    continue
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
                    # In USB-local mode, ACTIVE/VERIFYING_BROWSER revalidate USB first.
                    if self.browser_ready or self.connect_dispatched:
                        await self._check_usb_prerequisites()
                    if self.browser_ready:
                        await self._observe_stream()
                    elif self.connect_dispatched:
                        await self._verify_browser()
                    else:
                        # PREPARING_DEVICE wakes, rebuilds only in USB-local mode, then automates.
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
                except _TransportDisrupted as exc:
                    # Enumeration can miss a quick cable flap. Reverse/TURN
                    # evidence enters the same preservation path immediately,
                    # without the ordinary retry sleep or browser teardown.
                    await self._enter_transport_recovery(
                        str(exc),
                        "REBUILDING_USB",
                        adb_ready=exc.adb_ready,
                        network_present=exc.network_present,
                    )
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._stop_monitor()
                    self.browser_ready = False
                    self.browser_client = None
                    if self.connect_dispatched and isinstance(exc, TimeoutError):
                        await self._publish(
                            "degraded",
                            "VERIFYING_BROWSER",
                            "CONNECT dispatched; waiting for fresh browser health report",
                            adbReady=True,
                            networkPresent=True,
                        )
                        await self.sleep(self.config.interval_sec)
                        continue
                    self.connect_dispatched = False
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
