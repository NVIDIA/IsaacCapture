# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-band (OOB) teleop hub — WebSocket server for headset metrics and config.

Headsets register via WebSocket and report streaming metrics. Operators read
state and push config via the HTTP API on the same TLS port.

WebSocket: ``wss://<host>:<PORT>/oob/v1/ws``
HTTP API:  ``GET /api/oob/v1/state``, ``GET /api/oob/v1/config``
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger("isaaccapture.cloudxr.oob_teleop_hub")

OOB_WS_PATH = "/oob/v1/ws"


@dataclass
class _HeadsetState:
    client_id: str
    ws: Any
    registered_at: float
    device_label: str | None = None
    metrics_by_cadence: dict = field(default_factory=dict)
    # Updated by streamStatus messages; streaming_since is set on the rising
    # edge only (so a repeated True doesn't reset the timestamp).
    streaming: bool = False
    streaming_since: float | None = None
    last_seen_at: float | None = None
    last_metrics_at: float | None = None


class OOBControlHub:
    """Collects headset metrics and exposes state via HTTP.

    One instance per proxy process; WebSocket connections on ``OOB_WS_PATH``
    are dispatched via :meth:`handle_connection`.
    """

    def __init__(
        self,
        control_token: str | None = None,
        initial_config: dict | None = None,
    ) -> None:
        self._token = control_token
        self._headsets: dict[Any, _HeadsetState] = {}
        self._stream_config: dict = dict(initial_config or {})
        self._config_version: int = 0
        self._lock = asyncio.Lock()
        self._lifecycle: dict | None = None
        self._health_waiters: dict[tuple[str, int], tuple[Any, asyncio.Future]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def handle_connection(self, ws: Any) -> None:
        """Entry point for each new WebSocket client on ``OOB_WS_PATH``."""
        client_id = str(uuid.uuid4())
        registered = False

        try:
            async for raw in ws:
                if not isinstance(raw, str):
                    await self._send_error(ws, "BAD_REQUEST", "Expected text frame")
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(ws, "BAD_REQUEST", "Invalid JSON")
                    continue

                msg_type: str = msg.get("type", "")
                payload: dict = msg.get("payload") or {}

                if not registered:
                    if msg_type != "register":
                        await self._send_error(
                            ws, "BAD_REQUEST", "First message must be register"
                        )
                        return
                    ok = await self._handle_register(ws, client_id, payload)
                    if not ok:
                        return
                    registered = True
                    continue

                await self._dispatch_headset(ws, msg_type, payload)

        except Exception:
            log.debug("Teleop WS closed", exc_info=True)
        finally:
            async with self._lock:
                self._headsets.pop(ws, None)
                for key, (target_ws, waiter) in tuple(self._health_waiters.items()):
                    if target_ws is not ws:
                        continue
                    if not waiter.done():
                        waiter.set_result(None)
                    self._health_waiters.pop(key, None)
            log.info("Teleop client disconnected (clientId=%s)", client_id)

    async def get_snapshot(self) -> dict:
        """Build the JSON snapshot for ``GET /api/oob/v1/state``."""
        async with self._lock:
            headsets = [
                {
                    "clientId": s.client_id,
                    "connected": True,
                    "streaming": s.streaming,
                    "streamingSince": (
                        int(s.streaming_since * 1000)
                        if s.streaming_since is not None
                        else None
                    ),
                    "deviceLabel": s.device_label,
                    "registeredAt": int(s.registered_at * 1000),
                    "lastSeenAt": int(s.last_seen_at * 1000)
                    if s.last_seen_at
                    else None,
                    "lastMetricsAt": int(s.last_metrics_at * 1000)
                    if s.last_metrics_at
                    else None,
                    "metricsByCadence": s.metrics_by_cadence,
                }
                for s in self._headsets.values()
            ]
            return {
                "updatedAt": int(time.time() * 1000),
                "configVersion": self._config_version,
                "config": dict(self._stream_config),
                "headsets": headsets,
                "lifecycle": dict(self._lifecycle) if self._lifecycle else None,
            }

    async def set_lifecycle_snapshot(self, snapshot: dict) -> None:
        """Publish the lifecycle's already redacted status in the HTTP state."""
        async with self._lock:
            self._lifecycle = dict(snapshot)

    async def probe_browser(
        self,
        generation: int,
        after: float,
        timeout: float = 5.0,
        client_id: str | None = None,
    ) -> dict | None:
        """Require a reply from a client registered after browser navigation."""
        deadline = time.monotonic() + timeout
        target = None
        while target is None and time.monotonic() < deadline:
            async with self._lock:
                target = next(
                    (
                        s
                        for s in self._headsets.values()
                        if s.registered_at > after
                        and (client_id is None or s.client_id == client_id)
                    ),
                    None,
                )
            if target is None:
                await asyncio.sleep(0.1)
        if target is None:
            return None
        async with self._lock:
            probe_id = str(uuid.uuid4())
            key = (probe_id, generation)
            waiter = asyncio.get_running_loop().create_future()
            self._health_waiters[key] = (target.ws, waiter)
        await self._send(
            target.ws,
            "healthProbe",
            {"probeId": probe_id, "lifecycleGeneration": generation},
        )
        try:
            return await asyncio.wait_for(waiter, max(0.1, deadline - time.monotonic()))
        except TimeoutError:
            return None
        finally:
            async with self._lock:
                self._health_waiters.pop(key, None)

    async def wait_for_streaming(
        self, *, poll_seconds: float = 1.0
    ) -> tuple[str, float]:
        """Block until any headset reports ``streaming=True``; return ``(clientId, since)``."""
        while True:
            async with self._lock:
                for s in self._headsets.values():
                    if s.streaming and s.streaming_since is not None:
                        return s.client_id, s.streaming_since
            await asyncio.sleep(poll_seconds)

    def check_token(self, token: str | None) -> bool:
        """Return True if token satisfies the hub's auth requirement."""
        if not self._token:
            return True
        return token == self._token

    async def http_oob_set_config(self, payload: dict) -> tuple[int, dict]:
        """Merge stream config; for OOB HTTP ``GET /api/oob/v1/config``."""
        if not self.check_token(payload.get("token")):
            return 401, {"error": "Unauthorized"}

        new_config = payload.get("config")
        if not isinstance(new_config, dict):
            return 400, {"error": "config must be an object"}

        target_raw = payload.get("targetClientId")
        target_id: str | None = (
            None if target_raw is None or target_raw == "" else str(target_raw)
        )

        outcome = await self._merge_stream_config(new_config, target_id)
        if outcome[0] == "noop":
            return 200, {
                "ok": True,
                "changed": False,
                "configVersion": outcome[1],
            }
        if outcome[0] == "missing":
            return 404, {"error": f"Headset '{outcome[1]}' not connected"}

        _tag, version, config_snapshot, targets = outcome
        log.info(
            "OOB setConfig configVersion=%d → %d headset(s)", version, len(targets)
        )
        await self._push_config_to_headsets(version, config_snapshot, targets)
        return 200, {
            "ok": True,
            "changed": True,
            "configVersion": version,
            "targetCount": len(targets),
        }

    # ------------------------------------------------------------------
    # Private: registration
    # ------------------------------------------------------------------

    async def _handle_register(self, ws: Any, client_id: str, payload: dict) -> bool:
        """Validate and register a headset. Returns True on success."""
        if self._token and payload.get("token") != self._token:
            await self._send_error(ws, "UNAUTHORIZED", "Invalid or missing token")
            try:
                await ws.close(1008, "Unauthorized")
            except Exception:
                pass
            return False

        role = payload.get("role")
        if role != "headset":
            await self._send_error(ws, "BAD_REQUEST", "role must be 'headset'")
            return False

        async with self._lock:
            state = _HeadsetState(
                client_id=client_id,
                ws=ws,
                registered_at=time.time(),
                device_label=payload.get("deviceLabel"),
            )
            self._headsets[ws] = state
            log.info(
                "Headset registered: clientId=%s label=%s",
                client_id,
                state.device_label,
            )
            hello_payload = {
                "clientId": client_id,
                "configVersion": self._config_version,
                "config": dict(self._stream_config),
            }

        await self._send(ws, "hello", hello_payload)
        return True

    # ------------------------------------------------------------------
    # Private: message dispatch
    # ------------------------------------------------------------------

    async def _dispatch_headset(self, ws: Any, msg_type: str, payload: dict) -> None:
        if msg_type == "clientMetrics":
            await self._handle_client_metrics(ws, payload)
        elif msg_type == "streamStatus":
            await self._handle_stream_status(ws, payload)
        elif msg_type == "healthReport":
            await self._handle_health_report(ws, payload)
        else:
            await self._send_error(
                ws, "BAD_REQUEST", f"Unknown message type: {msg_type}"
            )

    # ------------------------------------------------------------------
    # Private: message handlers
    # ------------------------------------------------------------------

    async def _merge_stream_config(
        self, new_config: dict, target_id: str | None
    ) -> (
        tuple[Literal["noop"], int]
        | tuple[Literal["missing"], str]
        | tuple[Literal["push"], int, dict, list[_HeadsetState]]
    ):
        async with self._lock:
            merged = {**self._stream_config, **new_config}
            if merged == self._stream_config:
                return ("noop", self._config_version)

            if target_id is not None:
                targets = [
                    s for s in self._headsets.values() if s.client_id == target_id
                ]
                if not targets:
                    return ("missing", target_id)
                # Targeted push: send merged snapshot without mutating global state.
                return ("push", self._config_version, merged, targets)

            # Global push: update shared config and version.
            self._stream_config = merged
            self._config_version += 1
            return (
                "push",
                self._config_version,
                dict(self._stream_config),
                list(self._headsets.values()),
            )

    async def _push_config_to_headsets(
        self, version: int, config_snapshot: dict, targets: list[_HeadsetState]
    ) -> None:
        push_payload = {"configVersion": version, "config": config_snapshot}
        for headset in targets:
            await self._send(headset.ws, "config", push_payload)

    async def _handle_stream_status(self, ws: Any, payload: dict) -> None:
        """Wire format: ``{"streaming": true|false}``. Rising edge stamps ``streaming_since``."""
        streaming = bool(payload.get("streaming", False))
        async with self._lock:
            state = self._headsets.get(ws)
            if state is None:
                return
            state.last_seen_at = time.time()
            if streaming and not state.streaming:
                state.streaming_since = time.time()
            elif not streaming:
                state.streaming_since = None
            state.streaming = streaming
            log.info(
                "Headset %s streamStatus: streaming=%s", state.client_id, streaming
            )

    async def _handle_client_metrics(self, ws: Any, payload: dict) -> None:
        async with self._lock:
            state = self._headsets.get(ws)
            if state is None:
                return
            state.last_seen_at = time.time()
            state.last_metrics_at = state.last_seen_at
            cadence = str(payload.get("cadence", "unknown"))
            raw_metrics = payload.get("metrics", {})
            if not isinstance(raw_metrics, dict):
                raw_metrics = {}
            state.metrics_by_cadence[cadence] = {
                "at": int(payload.get("t", time.time() * 1000)),
                "metrics": {
                    str(k): float(v)
                    for k, v in raw_metrics.items()
                    if isinstance(v, (int, float))
                },
            }

    async def _handle_health_report(self, ws: Any, payload: dict) -> None:
        key = (str(payload.get("probeId", "")), payload.get("lifecycleGeneration"))
        async with self._lock:
            state = self._headsets.get(ws)
            pending = self._health_waiters.get(key)
            if (
                state is None
                or pending is None
                or pending[0] is not ws
                or pending[1].done()
            ):
                return
            state.last_seen_at = time.time()
            pending[1].set_result(
                {
                    "clientId": state.client_id,
                    "streaming": state.streaming,
                    "lastMetricsAt": state.last_metrics_at,
                    "receivedAt": state.last_seen_at,
                }
            )

    # ------------------------------------------------------------------
    # Private: send helpers
    # ------------------------------------------------------------------

    async def _send(self, ws: Any, msg_type: str, payload: dict) -> None:
        try:
            await ws.send(json.dumps({"type": msg_type, "payload": payload}))
        except Exception:
            log.debug("Failed to send '%s' message", msg_type, exc_info=True)

    async def _send_error(
        self,
        ws: Any,
        code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        p: dict = {"code": code, "message": message}
        if request_id is not None:
            p["requestId"] = request_id
        await self._send(ws, "error", p)
