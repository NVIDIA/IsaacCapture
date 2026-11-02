# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The production IsaacCapture acquisition adapter: owns SDK source-graph
configuration, the launcher/session context, the single rate-paced loop
that calls ``TeleopSession.step()``, and native-to-protocol mapping for
tracking samples and provider/device status.

Real ``step()`` is synchronous and unpaced (teleop_session_manager/
teleop_session.py: "session fields such as frame_count... are
intentionally owned by the application thread that calls step()") -- this
adapter is that application thread's loop: monotonic deadlines, skip
overdue ticks. This adapter is the sole acquisition-cadence owner, so no
second adapter ticker runs alongside it; ``_Session`` retains only
publication/sequence/counter responsibilities.

This file is an ordinary adapter, not a mock helper: everything mock-
specific (synthetic motion, native-style inventory, injected failures)
lives in :mod:`capture_server.mock.mock_isaac_capture`; everything here is
SDK-shape-agnostic composition/mapping that the real adapter sharing
this same class would also need.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

from capture_server import protocol as p

from ._thread_runner import ThreadRunner
from .clock import Clock, MonotonicClock
from .errors import SourceError, SourceExited
from .tensor_indices import ControllerInputIndex, HandInputIndex, HeadInputIndex

__all__ = ["MANUS_PLUGIN_NAME", "IsaacCaptureAdapter"]

#: The Manus glove plugin's manifest name. IsaacCapture's plugin manager registers every ``plugin.yaml``
#: found under the configured search paths by that file's ``name`` field and resolves a
#: ``PluginConfig.plugin_name`` by exact match (src/core/plugin_manager/cpp/plugin_manager.cpp,
#: teleop_session_manager/teleop_session.py ``_resolve_plugin``); the status monitor then names the
#: plugin's provider after that same field (``PluginProviderSpec.name = info.name``). The manifest in
#: src/plugins/manus/plugin.yaml says ``manus_hand_plugin``, so any other spelling is never discovered --
#: silently, when the plugin is optional. The mock mirrors this constant and a test keeps the two equal.
MANUS_PLUGIN_NAME = "manus_hand_plugin"


def _zero_controller_fields(prefix: str) -> dict[str, Any]:
    import numpy as np

    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return {
        f"{prefix}.present": np.uint8(0),
        f"{prefix}.grip_position": identity[:3],
        f"{prefix}.grip_orientation": identity[3:],
        f"{prefix}.grip_is_valid": np.uint8(0),
        f"{prefix}.aim_position": identity[:3],
        f"{prefix}.aim_orientation": identity[3:],
        f"{prefix}.aim_is_valid": np.uint8(0),
        f"{prefix}.primary_click": 0.0,
        f"{prefix}.secondary_click": 0.0,
        f"{prefix}.thumbstick_x": 0.0,
        f"{prefix}.thumbstick_y": 0.0,
        f"{prefix}.thumbstick_click": 0.0,
        f"{prefix}.menu_click": 0.0,
        f"{prefix}.squeeze_value": 0.0,
        f"{prefix}.trigger_value": 0.0,
    }


def _zero_hand_fields(prefix: str) -> dict[str, Any]:
    import numpy as np

    return {
        f"{prefix}.present": np.uint8(0),
        f"{prefix}.joint_positions": np.zeros((26, 3), dtype=np.float32),
        f"{prefix}.joint_orientations": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (26, 1)),
        f"{prefix}.joint_radii": np.zeros(26, dtype=np.float32),
        f"{prefix}.joint_valid": np.zeros(26, dtype=np.uint8),
    }


def _owned(value: Any) -> Any:
    """An owned copy of a value borrowed from a TensorGroup -- its
    underlying storage may be reused/overwritten by the SDK on the next
    ``step()`` call, so nothing borrowed may escape this adapter's worker
    thread uncopied."""
    import numpy as np

    return np.array(value, copy=True)


def _controller_fields(prefix: str, group: Any, index: type) -> dict[str, Any]:
    """*group* is a ``TensorGroup``-shaped object indexable by
    ``ControllerInputIndex`` -- see ``retargeting_engine/interface/
    tensor_group.py``'s ``__getitem__``/``is_none`` contract."""
    import numpy as np

    return {
        f"{prefix}.present": np.uint8(1),
        f"{prefix}.grip_position": _owned(group[index.GRIP_POSITION]),
        f"{prefix}.grip_orientation": _owned(group[index.GRIP_ORIENTATION]),
        f"{prefix}.grip_is_valid": np.uint8(1 if group[index.GRIP_IS_VALID] else 0),
        f"{prefix}.aim_position": _owned(group[index.AIM_POSITION]),
        f"{prefix}.aim_orientation": _owned(group[index.AIM_ORIENTATION]),
        f"{prefix}.aim_is_valid": np.uint8(1 if group[index.AIM_IS_VALID] else 0),
        f"{prefix}.primary_click": group[index.PRIMARY_CLICK],
        f"{prefix}.secondary_click": group[index.SECONDARY_CLICK],
        f"{prefix}.thumbstick_x": group[index.THUMBSTICK_X],
        f"{prefix}.thumbstick_y": group[index.THUMBSTICK_Y],
        f"{prefix}.thumbstick_click": group[index.THUMBSTICK_CLICK],
        f"{prefix}.menu_click": group[index.MENU_CLICK],
        f"{prefix}.squeeze_value": group[index.SQUEEZE_VALUE],
        f"{prefix}.trigger_value": group[index.TRIGGER_VALUE],
    }


def _hand_fields(prefix: str, group: Any, index: type) -> dict[str, Any]:
    """*group* is a ``TensorGroup``-shaped object indexable by
    ``HandInputIndex``."""
    import numpy as np

    return {
        f"{prefix}.present": np.uint8(1),
        f"{prefix}.joint_positions": _owned(group[index.JOINT_POSITIONS]),
        f"{prefix}.joint_orientations": _owned(group[index.JOINT_ORIENTATIONS]),
        f"{prefix}.joint_radii": _owned(group[index.JOINT_RADII]),
        f"{prefix}.joint_valid": _owned(group[index.JOINT_VALID]),
    }


@dataclass(frozen=True)
class TrackingSelection:
    """Which ic_tracking_v1 groups this session requested -- independent
    of whether the request is valid/tracked, which the native sample
    itself reports per group. ``manus_enabled`` is a separate, independent
    axis: a session can request the Manus plugin without selecting hand
    output, and selecting hand output never implies requesting Manus
    by itself."""

    head: bool
    controller_left: bool
    controller_right: bool
    hand_left: bool
    hand_right: bool
    manus_enabled: bool = False
    # Whether the client's own PluginRequest.required applies to the Manus
    # plugin specifically -- a required plugin that fails to load must
    # fail session establishment (IC_ERROR); an optional one may degrade
    # silently. Meaningless when manus_enabled is False (no plugin is
    # constructed at all in that case).
    manus_required: bool = False


class IsaacCaptureAdapter:
    """Wraps one ``TeleopSession``-shaped object for the lifetime of one
    accepted session. Identical code path for mock and real SDKs:
    ``sdk`` supplies only the real, SDK-shared source-selector/config
    construction surface (``HeadSource``/``ControllersSource``/
    ``HandsSource``/``TeleopSessionConfig`` -- the same public API a real
    adapter configures its graph with); ``session_factory`` is a
    ``Callable[[TeleopSessionConfig], TeleopSession]`` that, given that
    config, returns a ready-to-enter session. Launch composition supplies
    ``session_factory`` -- for mock mode a closure/partial over
    ``mock_isaac_capture.TeleopSession`` with scenario/seed/fault options
    already bound; for real mode, ``isaaccapture``'s ``TeleopSession``
    itself, unmodified. This class never sees or branches on
    scenario/seed/faults -- those are mock-only concepts that belong
    entirely inside ``mock_isaac_capture.py``."""

    def __init__(
        self,
        *,
        sdk: ModuleType,
        session_factory: Callable[[Any], Any],
        selection: TrackingSelection,
        rate_hz: float,
        clock: Clock | None = None,
    ) -> None:
        self._sdk = sdk
        self._session_factory = session_factory
        self._selection = selection
        self._interval = 1.0 / rate_hz
        self._clock = clock or MonotonicClock()
        self._runner = ThreadRunner()
        self._session: Any = None
        self._launcher: Any = None
        # Tracks whether each object's own __enter__() actually returned,
        # separately from whether the field itself is set -- close() must
        # never pair __exit__ with an __enter__ that failed or never ran,
        # the same contract a `with` block enforces.
        self._session_entered = False
        self._launcher_entered = False
        self._deadline: float | None = None

    def _build_config(self) -> Any:
        """Matches the real construction pattern -- examples/
        deviceio_live_view/.../deviceio_viser.py's
        ``build_all_human_pipeline``: construct named source nodes, wire
        each selected leaf through ``source.output(selector)`` into one
        ``OutputCombiner({leaf_key: ...})``, then hand that combiner to
        ``TeleopSessionConfig`` as its ``pipeline``."""
        outputs: dict[str, Any] = {}
        if self._selection.head:
            head = self._sdk.HeadSource(name="head")
            outputs["head"] = head.output("head")
        if self._selection.controller_left or self._selection.controller_right:
            controllers = self._sdk.ControllersSource(name="controllers")
            if self._selection.controller_left:
                outputs["controller_left"] = controllers.output(self._sdk.ControllersSource.LEFT)
            if self._selection.controller_right:
                outputs["controller_right"] = controllers.output(self._sdk.ControllersSource.RIGHT)
        if self._selection.hand_left or self._selection.hand_right:
            hands = self._sdk.HandsSource(name="hands")
            if self._selection.hand_left:
                outputs["hand_left"] = hands.output(self._sdk.HandsSource.LEFT)
            if self._selection.hand_right:
                outputs["hand_right"] = hands.output(self._sdk.HandsSource.RIGHT)
        pipeline = self._sdk.OutputCombiner(outputs)
        # An unrequested/disabled plugin is never constructed at all --
        # the wire protocol reports a missing plugin as having no
        # inventory entry at all, not a disabled-but-present one.
        plugins = (
            [
                self._sdk.PluginConfig(
                    plugin_name=MANUS_PLUGIN_NAME,
                    plugin_root_id="manus",
                    search_paths=[],
                    required=self._selection.manus_required,
                )
            ]
            if self._selection.manus_enabled
            else []
        )
        return self._sdk.TeleopSessionConfig(app_name="capture_server", pipeline=pipeline, plugins=plugins)

    def _enter_sync(self) -> Any:
        # CloudXRLauncher.launch_context is the shared launcher/context
        # boundary the approved API subset requires the adapter to
        # actually own/enter -- a mock CloudXRLauncher class that is
        # merely defined but never called would not exercise it.
        # The mock ignores its argument; the real SDK binding supplies attach-only CloudXR ownership
        # (see isaac_sdk.AttachOnlyCloudXR), so the namespace carries no launch options.
        # real launch options this call needs.
        launcher = self._sdk.CloudXRLauncher.launch_context(argparse.Namespace())
        # Record ownership BEFORE any further native work, on this same
        # worker thread -- if the asyncio await wrapping this call is
        # cancelled (or entry only partially succeeds) after this point,
        # close() must still be able to find and release whatever was
        # actually constructed, rather than seeing a still-None field set
        # only by an enter() continuation that may never run.
        self._launcher = launcher
        try:
            launcher.__enter__()
            self._launcher_entered = True
            session = self._session_factory(self._build_config())
            self._session = session
            session.__enter__()
            self._session_entered = True
        except Exception as exc:  # noqa: BLE001 -- native entry failure, mapped uniformly
            raise SourceError(str(exc) or type(exc).__name__) from exc
        return session

    async def enter(self) -> None:
        await self._runner.run(self._enter_sync)  # self._session is set inside _enter_sync itself
        self._deadline = self._clock.monotonic() + self._interval

    def _next_deadline(self, deadline: float) -> float:
        """Advance *deadline* to the next tick strictly after "now",
        skipping any ticks that are overdue rather than bursting to catch
        up."""
        now = self._clock.monotonic()
        while deadline <= now:
            deadline += self._interval
        return deadline

    def _step_and_map_sync(self) -> tuple[int, dict[str, Any]]:
        """Runs entirely on this adapter's dedicated worker thread: stamps
        the host-sampling timestamp immediately before calling ``step()``
        -- the wire protocol's ``sample_time_semantics`` is ``HOST_POLL``,
        so ``capture_time_ns`` must mark this host sampling step, not the
        time the server later finishes assembling the frame for transport --
        then immediately maps ``step()``'s result into an owned
        ic_tracking_v1 field dict, copying every array out of the
        TensorGroups ``step()`` returns before returning across the
        thread boundary. ``step()``'s returned TensorGroups borrow mutable
        storage the SDK may reuse/overwrite on the NEXT call -- mapping
        (and copying) must happen here, synchronously, on this same
        thread, not after the raw native result has already crossed back
        onto the asyncio loop."""
        sample_time_ns = int(self._clock.monotonic() * 1_000_000_000)
        try:
            native = self._session.step()
        except Exception as exc:  # noqa: BLE001 -- any step() failure is fatal, StopReason decided by the caller from real status evidence, never from this message's wording
            raise SourceError(str(exc) or type(exc).__name__) from exc
        return sample_time_ns, self._map_tracking(native)

    async def next_tracking_sample(self) -> tuple[int, dict[str, Any]]:
        """Owns the single acquisition cadence: sleeps until the next
        monotonic deadline (skipping overdue ticks), then calls
        ``step()`` on this adapter's dedicated thread and maps the result
        to one ic_tracking_v1 field dict, paired with the host-monotonic
        sampling timestamp taken at the actual ``step()`` call (not when
        this coroutine resumes or the caller later publishes the frame).
        ``_Session`` only loops calling this repeatedly and publishing
        what it returns; it never computes a deadline itself.

        A ``step()`` failure is always ``SourceError`` (IC_ERROR) unless
        this adapter can confirm, from real typed status evidence (a
        provider reporting ``PROCESS_EXITED``/``PROCESS_SIGNALED`` --
        teleop_session_manager/status.py's own ``StatusReason`` members,
        not this exception's message text), that the acquisition worker
        itself has exited -- only then is it reclassified as
        ``SourceExited`` (IC_EXITED). Message-substring matching is
        deliberately not used: it is fragile and can misclassify an
        ordinary operation error that merely happens to mention "exit\""""
        assert self._deadline is not None, "next_tracking_sample() called before enter()"
        import asyncio

        await asyncio.sleep(max(0.0, self._deadline - self._clock.monotonic()))
        self._deadline = self._next_deadline(self._deadline)
        try:
            return await self._runner.run(self._step_and_map_sync)
        except SourceError as exc:
            if await self._worker_exited():
                raise SourceExited(str(exc)) from exc
            raise

    async def _worker_exited(self) -> bool:
        """Real-signal check for whether the acquisition worker process
        itself has exited, via the SAME status surface a client observes
        -- never inferred from a step() exception's wording."""
        try:
            status = await self._runner.run(self._session.get_status)
        except Exception:  # noqa: BLE001 -- status itself failing is not exit evidence
            return False
        return any(provider.reason.name in ("PROCESS_EXITED", "PROCESS_SIGNALED") for provider in status.providers)

    def _map_tracking(self, native: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        fields: dict[str, Any] = {}
        head = native.get("head")
        if head is not None and not head.is_none:
            fields.update(
                {
                    "head.present": np.uint8(1),
                    "head.position": np.array(head[HeadInputIndex.POSITION], copy=True),
                    "head.orientation": np.array(head[HeadInputIndex.ORIENTATION], copy=True),
                    "head.is_valid": np.uint8(1 if head[HeadInputIndex.IS_VALID] else 0),
                    "head.is_tracked": np.uint8(1 if head[HeadInputIndex.IS_TRACKED] else 0),
                }
            )
        else:
            fields.update(
                {
                    "head.present": np.uint8(0),
                    "head.position": [0.0, 0.0, 0.0],
                    "head.orientation": [0.0, 0.0, 0.0, 1.0],
                    "head.is_valid": np.uint8(0),
                    "head.is_tracked": np.uint8(0),
                }
            )
        for prefix, key in (
            ("controller_left", "controller_left"),
            ("controller_right", "controller_right"),
        ):
            group = native.get(key)
            if group is not None and not group.is_none:
                fields.update(_controller_fields(prefix, group, ControllerInputIndex))
            else:
                fields.update(_zero_controller_fields(prefix))
        for prefix, key in (("hand_left", "hand_left"), ("hand_right", "hand_right")):
            group = native.get(key)
            if group is not None and not group.is_none:
                fields.update(_hand_fields(prefix, group, HandInputIndex))
            else:
                fields.update(_zero_hand_fields(prefix))
        return fields

    async def status(self, updated_at_ns: int) -> p.StatusSnapshot:
        native = await self._runner.run(self._session.get_status)
        return self._map_status(native, updated_at_ns)

    @staticmethod
    def _map_status(native: Any, updated_at_ns: int) -> p.StatusSnapshot:
        # TODO: getattr(..., native_enum_member.name) below assumes the
        # live SDK's own enum member names match this protocol's member
        # names exactly; the mock's names are defined to match by
        # construction, so this is unvalidated against a real SDK and
        # raises AttributeError on any live-only name it doesn't recognize.
        providers = tuple(
            p.ProviderStatus(
                id=provider.id,
                type=getattr(p.ProviderType, provider.type.name),
                # A plugin provider is named after its manifest, so the manifest name is the only key
                # that identifies the Manus plugin in live status.
                plugin=p.PluginType.MANUS if provider.name == MANUS_PLUGIN_NAME else None,
                name=provider.name,
                status=getattr(p.ProviderState, provider.status.name),
                reason=getattr(p.StatusReason, provider.reason.name),
                error=provider.error,
                pid=provider.pid,
                exit_code=provider.exit_code,
                term_signal=provider.term_signal,
            )
            for provider in native.providers
        )
        devices = tuple(
            p.DeviceStatus(
                id=device.id,
                path=device.path,
                type=device.type,
                description=device.description,
                provider_id=device.provider_id,
                status=getattr(p.DeviceState, device.status.name),
                reason=getattr(p.StatusReason, device.reason.name),
                error=device.error,
            )
            for device in native.devices
        )
        return p.StatusSnapshot(updated_at_ns=updated_at_ns, providers=providers, devices=devices)

    async def close(self) -> bool:
        def _close() -> None:
            # Read at EXECUTION time (on this adapter's single-worker
            # executor, strictly after any earlier-queued _enter_sync has
            # finished) -- never a value captured eagerly when close()
            # was first awaited, which could still be None if entry's own
            # job hadn't run yet. Session exits before the launcher it
            # runs inside of (reverse acquisition order) -- but a raising
            # session.__exit__ must never skip the launcher's own
            # __exit__, or the launcher's owned runtime leaks. Both are
            # always attempted; the first failure is what propagates.
            session = self._session
            launcher = self._launcher
            exc: Exception | None = None
            # Only an object whose own __enter__() actually returned gets
            # paired with __exit__() -- same contract a `with` block
            # enforces: a failed or never-attempted __enter__() must never
            # see a matching __exit__() call.
            if session is not None and self._session_entered:
                try:
                    session.__exit__(None, None, None)
                except Exception as session_exc:  # noqa: BLE001 -- recorded, the launcher close is still attempted
                    exc = session_exc
            if launcher is not None and self._launcher_entered:
                try:
                    launcher.__exit__(None, None, None)
                except Exception as launcher_exc:  # noqa: BLE001 -- only replaces exc if the session close already succeeded
                    if exc is None:
                        exc = launcher_exc
            if exc is not None:
                raise exc

        return await self._runner.close(_close)
