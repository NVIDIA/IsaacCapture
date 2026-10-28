# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public scheduling and lifetime facade for an application-owned OVXR scene."""

from __future__ import annotations
import logging
import math
import threading
from enum import Enum, auto
from collections.abc import Callable, Hashable, Iterable
from typing import Any
from .frame_pump import OvxrFramePump
from .runtime_owner import CloudXRRuntime, CloudXRRuntimeOwner
from .interfaces import (
    OvxrApplicationInterface as OvxrApplicationInterface,
    StageOrdinalLedger as StageOrdinalLedger,
    OvxrSessionBinding as OvxrSessionBinding,
    OvxrSessionConfig as OvxrSessionConfig,
    OvxrScenePublication as OvxrScenePublication,
    OvxrResourceOwner as OvxrResourceOwner,
)

_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_LOGGER = logging.getLogger(__name__)


class _Phase(Enum):
    NEW = auto()
    STARTING = auto()
    READY = auto()
    CLOSED = auto()


class OvxrSession:
    """Schedule one frame owner and close it after application borrowers stop."""

    def __init__(
        self,
        host: OvxrApplicationInterface,
        config: OvxrSessionConfig,
        *,
        required_openxr_extensions: Iterable[str] = (),
        max_blank_frames: int = 1,
        runtime_launcher_factory: Callable[..., CloudXRRuntime] | None = None,
    ) -> None:
        if max_blank_frames < 0:
            raise ValueError("max_blank_frames must be non-negative")
        self._host = host
        self._worker: threading.Thread | None = None
        self._worker_native_thread_id: int | None = None
        self._worker_condition = threading.Condition()
        self._worker_bootstrap_ready = False
        self._worker_error: BaseException | None = None
        self._shutdown_requested = threading.Event()
        self._borrowers_quiesced = threading.Event()
        self._retirement_complete = False
        self._worker_pending_ordinal: int | None = None
        self._worker_publication_requested = True
        self._worker_last_published_ordinal: int | None = None
        self._worker_handles = None
        self._phase = _Phase.NEW
        self._borrowers_exposed = False
        self._pump = OvxrFramePump(
            host,
            config,
            CloudXRRuntimeOwner(config, runtime_launcher_factory),
            shutdown_requested=lambda: self.shutdown_requested,
            request_shutdown=self.request_shutdown,
            required_openxr_extensions=required_openxr_extensions,
            max_blank_frames=max_blank_frames,
        )

    def start(self, timeout_s: float = 30.0) -> bool:
        """Transfer the borrowed pair and return whether XR startup completed.

        ``timeout_s`` must be finite, positive and within ``threading.TIMEOUT_MAX``.
        It bounds host attachment readiness, not the subsequent peer-gated XR wait.
        Expiry and interruption request cancellation; an active attachment must finish
        before the owner thread can retire its borrowed resources. Once start
        succeeds or input handles are borrowed, close() is required; use a context
        manager or try/finally. Failed startup without borrowers retires itself.
        """
        self._require_open()
        if self._worker is not None:
            raise RuntimeError("OVXR session is already started")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= threading.TIMEOUT_MAX:
            raise ValueError(
                "OVXR bootstrap timeout must be finite, positive and within threading.TIMEOUT_MAX"
            )
        self._pump.validate_provider()
        capability = self._host.offer_render_owner_transfer()

        self._phase = _Phase.STARTING

        def run() -> None:
            try:
                self._worker_native_thread_id = threading.get_native_id()
                self._pump.attach(capability)
                with self._worker_condition:
                    self._worker_bootstrap_ready = True
                    self._worker_condition.notify_all()
                if self._shutdown_requested.is_set():
                    return
                self._pump.create_context()
                with self._worker_condition:
                    if self._shutdown_requested.is_set():
                        return
                self._pump.bootstrap()
                with self._worker_condition:
                    if self._shutdown_requested.is_set():
                        return
                    self._worker_handles = self._pump.external_handles()
                    self._phase = _Phase.READY
                    self._worker_condition.notify_all()
                while True:
                    with self._worker_condition:
                        if self._shutdown_requested.is_set():
                            break
                        pending_ordinal = self._worker_pending_ordinal
                        self._worker_pending_ordinal = None
                        if pending_ordinal is not None:
                            self._worker_publication_requested = True
                            self._worker_condition.notify_all()
                    if pending_ordinal is not None:
                        frame_began = self._pump.render(pending_ordinal)
                    else:
                        frame_began = self._pump.render_reused()
                    # A peer can become ready after the bounded blank-frame
                    # bootstrap. Publish its input handles from the owning
                    # render thread once the session has actually appeared.
                    if self._worker_handles is None:
                        handles = self._pump.external_handles()
                        if handles is not None:
                            with self._worker_condition:
                                self._worker_handles = handles
                                self._worker_condition.notify_all()
                    # A begun frame is paced by OpenXR even when simulation is
                    # paused. Back off only without a scene or a ready session;
                    # sleeping after a reused frame misses runtime deadlines.
                    if not frame_began:
                        with self._worker_condition:
                            if (
                                not self._shutdown_requested.is_set()
                                and self._worker_pending_ordinal is None
                            ):
                                self._worker_condition.wait(0.01)
            except BaseException as exc:
                # Thread failures cannot reach the caller automatically. Preserve
                # interrupts too, for caller-side re-raising and cleanup.
                self._record_worker_failure(exc)
            finally:
                # Once handles are exposed, only the host can acknowledge that
                # its application step and DeviceIO borrowers have stopped.
                with self._worker_condition:
                    borrowers_exposed = self._borrowers_exposed
                if borrowers_exposed:
                    self._borrowers_quiesced.wait()
                try:
                    self._pump.close()
                    self._retirement_complete = True
                except BaseException as exc:
                    # Failed consumer retirement retains its borrowed owners.
                    # Report it to close() even when rendering itself succeeded.
                    self._record_worker_failure(exc)
                finally:
                    with self._worker_condition:
                        self._phase = _Phase.CLOSED
                        self._worker_condition.notify_all()

        try:
            self._worker = threading.Thread(
                target=run, name="isaacteleop-ovxr-render", daemon=False
            )
            self._worker.start()
        except Exception:
            # No owner thread accepted the borrowed pair; rollback stays on the application thread.
            self._worker = None
            self._phase = _Phase.CLOSED
            try:
                self._host.abort()
            except Exception:
                _LOGGER.exception(
                    "Cleanup after failed render-worker start also failed"
                )
            raise
        with self._worker_condition:
            try:
                bootstrap_ready = self._worker_condition.wait_for(
                    lambda: (
                        self._worker_bootstrap_ready
                        or self._worker_error is not None
                        or self._shutdown_requested.is_set()
                    ),
                    timeout_s,
                )
                if not bootstrap_ready:
                    raise TimeoutError(
                        "OVXR session worker bootstrap did not become ready"
                    )
                self._raise_worker_error()
                self._worker_condition.wait_for(
                    lambda: (
                        self._phase is _Phase.READY
                        or self._worker_error is not None
                        or self._shutdown_requested.is_set()
                    )
                )
                self._raise_worker_error()
                if self._phase is not _Phase.READY or self._shutdown_requested.is_set():
                    return False
                self._borrowers_exposed = True
                return True
            except BaseException:
                self._request_worker_shutdown()
                raise

    def request_render(
        self, publication_id: Hashable, *, force: bool = False
    ) -> int | None:
        """Publish the newest host generation when the render worker requests it.

        ``force`` preserves discontinuous reset/lifecycle generations even when
        an ordinary render demand is already satisfied.

        Return the published ordinal on admission, or ``None`` if demand is
        satisfied or shutdown has been requested. The application must retry
        its latest dirty generation while running, including while paused;
        the worker cannot read mutable host state. Worker failures still raise.
        """
        with self._worker_condition:
            self._raise_worker_error()
            self._require_open()
            # UI/signal shutdown can arrive during the current application step.
            # Decline its late publication, but never hide a latched worker error.
            if self._shutdown_requested.is_set():
                return None
            if self._worker is None:
                raise RuntimeError("OVXR session is not accepting requests")
            if (
                not force
                and self._worker_last_published_ordinal is not None
                and not self._worker_publication_requested
            ):
                return None
            self._worker_publication_requested = False

        try:
            with self._host.publication_guard():
                ordinal = self._host.publish_complete_scene(publication_id)
                self._worker_last_published_ordinal = ordinal
                with self._worker_condition:
                    if self._shutdown_requested.is_set():
                        self._raise_worker_error()
                        return None
                    self._worker_pending_ordinal = ordinal
                    self._worker_condition.notify_all()
                    self._raise_worker_error()
                return ordinal
        except BaseException:
            with self._worker_condition:
                if not self._shutdown_requested.is_set():
                    self._worker_publication_requested = True
                self._worker_condition.notify_all()
            raise

    def external_oxr_handles(self):
        """Return typed handles after peer-gated context startup, otherwise ``None``."""
        with self._worker_condition:
            self._raise_worker_error()
            self._require_open()
            if self._shutdown_requested.is_set():
                return None
            if self._worker_handles is not None:
                self._borrowers_exposed = True
            return self._worker_handles

    def get_target_frame_T_world(self, prim_path: str) -> Any:
        """Delegate a current target transform read to the simulation host."""
        self._require_open()
        return self._host.get_target_frame_T_world(prim_path)

    def close(self) -> None:
        """Acknowledge stopped app/DeviceIO borrowers, then drain the owner thread.

        The host must finish its current application step and stop every input
        borrower before calling this method, including after worker failure.
        A terminal worker is not proof of successful resource retirement; inspect
        ``retirement_complete`` and preserve any raised first error. After
        requesting exit, wait up to 30 seconds for the worker; a timeout leaves
        its borrowed resources and runtime alive for the worker to retire.
        """
        if self._phase is _Phase.CLOSED:
            # Service.stop supports retry; native destruction does not promise it.
            try:
                self._pump.retry_runtime_stop()
            except Exception as error:
                if self._worker_error is not None:
                    raise self._worker_error from error
                raise
            self._raise_worker_error()
            return
        if self._worker is None:
            self._host.abort()
            self._phase = _Phase.CLOSED
            self._retirement_complete = True
            return
        self._request_worker_shutdown()
        wake_error: BaseException | None = None
        try:
            self._pump.request_exit()
        except BaseException as exc:
            wake_error = exc
        self._borrowers_quiesced.set()
        # Slow retirement is not proof of a blocked OpenXR call. Keep the
        # runtime alive until its borrowers retire, even when this wait expires.
        self._worker.join(timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        if self._worker.is_alive():
            raise TimeoutError("OVXR session worker did not stop") from wake_error
        worker_error = self._worker_error
        if worker_error is not None:
            if wake_error is not None and worker_error is not wake_error:
                raise worker_error from wake_error
            raise worker_error
        if wake_error is not None:
            raise wake_error

    @property
    def owns_environment(self) -> bool:
        """Whether the worker accepted host-environment teardown ownership."""
        return self._worker is not None

    @property
    def shutdown_requested(self) -> bool:
        """Whether UI close or explicit teardown requested application shutdown."""
        return self._shutdown_requested.is_set()

    def request_shutdown(self) -> None:
        """Ask the embedding application to leave its active simulation loop."""
        self._request_worker_shutdown()

    def wait_for_shutdown(self, timeout_s: float) -> bool:
        """Wait interruptibly on the same event used by signals and mirror close."""
        return self._shutdown_requested.wait(timeout_s)

    @property
    def retirement_complete(self) -> bool:
        """Whether all owned resources retired successfully, not merely stopped."""
        with self._worker_condition:
            return self._retirement_complete

    @property
    def render_worker_bootstrap_ready(self) -> bool:
        """Whether the borrowed-pair owner transfer completed."""
        with self._worker_condition:
            return self._worker_bootstrap_ready

    @property
    def render_worker_native_thread_id(self) -> int | None:
        """Native ID of the sole OVXR owner thread, when started."""
        return self._worker_native_thread_id

    def _request_worker_shutdown(self) -> None:
        with self._worker_condition:
            if self._phase is _Phase.CLOSED or self._shutdown_requested.is_set():
                return
            self._shutdown_requested.set()
            self._worker_condition.notify_all()

    def _record_worker_failure(self, error: BaseException) -> None:
        self._shutdown_requested.set()
        with self._worker_condition:
            if self._worker_error is None:
                self._worker_error = error
            self._worker_condition.notify_all()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise self._worker_error

    def _require_open(self) -> None:
        if self._phase is _Phase.CLOSED:
            raise RuntimeError("OVXR session is closed")

    def __enter__(self) -> OvxrSession:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
