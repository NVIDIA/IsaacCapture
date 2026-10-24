# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-neutral ownership of an OVXR session.

The application keeps ownership of simulation publication and lends one exact
renderer/Stage/ordinal-ledger tuple through :class:`OvxrApplicationInterface`.  This
module owns the OVXR context, its sole render thread, CloudXR, frame
cancellation, recovery, and ordered teardown.  It intentionally has no
dependency on Isaac Lab or on any simulation type.
"""

from __future__ import annotations
import logging
import math
import threading
from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

_WORKER_SHUTDOWN_GRACE_SECONDS = 1.0
_LOGGER = logging.getLogger(__name__)


class OvxrApplicationInterface(Protocol):
    """Application operations required by the OVXR session."""

    @abstractmethod
    def offer_render_owner_transfer(self) -> object:
        raise NotImplementedError

    @abstractmethod
    def attach_on_render_thread(self, capability: object) -> OvxrSessionBinding:
        raise NotImplementedError

    @abstractmethod
    def owner_lane(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    @abstractmethod
    def publication_guard(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    @abstractmethod
    def publish_complete_scene(self, publication_id: Hashable) -> int:
        raise NotImplementedError

    @abstractmethod
    def validate_first_published_scene(self, ordinal: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_target_frame_T_world(self, prim_path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def close_publication(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_on_render_thread(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_environment_on_render_thread(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> None:
        raise NotImplementedError


class StageOrdinalLedger(Protocol):
    """Application-owned reservations for explicit OVXR Stage writes."""

    @abstractmethod
    def reserve(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def resolve_external(
        self, ordinal: int, *, expected_published: bool | None
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def abandon(self, ordinal: int) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class OvxrSessionBinding:
    """Exact application-owned objects borrowed by one OVXR context."""

    renderer: object
    stage: object
    ordinal_ledger: StageOrdinalLedger
    prepare_frame_scene: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if self.renderer is None or self.stage is None or self.ordinal_ledger is None:
            raise ValueError(
                "OVXR session requires renderer, Stage, and ordinal ledger"
            )


@dataclass
class OvxrSessionConfig:
    """Configure an application-owned renderer/Stage XR session."""

    application_name: str = "isaacteleop-ovxr-session"
    session_mode: Literal["ar", "vr"] = "vr"
    cloudxr_env_file: str | None = None
    auto_launch_cloudxr: bool = True
    accept_eula: bool = False
    render_products: tuple[str, ...] = ()
    camera_paths: tuple[str, ...] = ()
    create_render_products: bool = True
    meters_per_unit: float = 1.0
    depth_near_z: float = 0.0
    depth_far_z: float = 0.0
    tracking_anchor_path: str = ""
    floor_aligned_anchor: bool | None = None
    host_wait_render_output: bool = False
    extra_openxr_extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.session_mode not in ("ar", "vr"):
            raise ValueError("session_mode must be 'ar' or 'vr'")
        if not math.isfinite(self.meters_per_unit) or self.meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be finite and positive")
        if (self.depth_near_z, self.depth_far_z) != (0.0, 0.0) and (
            not math.isfinite(self.depth_near_z)
            or not math.isfinite(self.depth_far_z)
            or self.depth_near_z <= 0.0
            or self.depth_far_z <= self.depth_near_z
        ):
            raise ValueError(
                "depth_near_z and depth_far_z must both be zero or finite with 0 < near < far"
            )
        if len(self.camera_paths) != 2 or any(not path for path in self.camera_paths):
            raise ValueError("OVXR stereo requires exactly two non-empty camera paths")


def _import_ovxr():
    import ovxr

    return ovxr


def _create_cloudxr_launcher(env_config: str, *, accept_eula: bool = False):
    from isaaccapture.cloudxr import CloudXRLauncher

    return CloudXRLauncher(
        install_dir=str(Path.home() / ".cloudxr"),
        env_config=env_config,
        accept_eula=accept_eula,
        run_embedded=True,
    )


class OvxrSession:
    """Own one OVXR/CloudXR frame pump over an application-supplied pair."""

    def __init__(
        self,
        host: OvxrApplicationInterface,
        config: OvxrSessionConfig,
        *,
        required_openxr_extensions: Iterable[str] = (),
        max_blank_frames: int = 1,
    ) -> None:
        if max_blank_frames < 0:
            raise ValueError("max_blank_frames must be non-negative")
        self._host = host
        self._config = config
        self._required_openxr_extensions = tuple(required_openxr_extensions)
        self._max_blank_frames = max_blank_frames
        self._context = None
        self._pending_plan = None
        self._binding: OvxrSessionBinding | None = None
        self._cloudxr_launcher = None
        self._worker: threading.Thread | None = None
        self._worker_native_thread_id: int | None = None
        self._worker_condition = threading.Condition()
        self._context_lifecycle_lock = threading.Lock()
        self._teardown_state_lock = threading.Lock()
        self._worker_bootstrap_ready = False
        self._worker_ready = False
        self._worker_closing = False
        self._worker_error: BaseException | None = None
        self._shutdown_requested = threading.Event()
        self._borrowers_quiesced = threading.Event()
        self._retirement_complete = False
        self._ovxr_teardown_complete = threading.Event()
        self._cloudxr_stop_started = threading.Event()
        self._cloudxr_stop_complete = threading.Event()
        self._worker_pending_ordinal: int | None = None
        self._worker_publication_requested = True
        self._worker_last_ordinal: int | None = None
        self._worker_last_published_ordinal: int | None = None
        self._worker_first_deferred_ordinal_validated = False
        self._worker_handles = None
        self._closed = False

    def start(self, timeout_s: float = 30.0) -> bool:
        """Transfer the borrowed pair and return whether XR startup completed.

        ``timeout_s`` must be finite, positive and within ``threading.TIMEOUT_MAX``.
        It bounds host attachment readiness, not the subsequent peer-gated XR wait.
        Expiry and interruption request cancellation; an active attachment must finish
        before the owner thread can retire its borrowed resources.
        """
        self._require_open()
        if self._worker is not None:
            raise RuntimeError("OVXR session is already started")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= threading.TIMEOUT_MAX:
            raise ValueError(
                "OVXR bootstrap timeout must be finite, positive and within threading.TIMEOUT_MAX"
            )
        ovxr = _import_ovxr()
        required_phase_methods = (
            "begin_frame",
            "bind_frame",
            "submit_frame",
            "cancel_frame",
            "get_vulkan_handles",
            "set_frame_render_products",
        )
        missing = [
            method
            for method in required_phase_methods
            if not callable(getattr(ovxr.Context, method, None))
        ]
        if missing:
            raise RuntimeError(
                f"OVXR session is missing required Context methods: {', '.join(missing)}"
            )
        capability = self._host.offer_render_owner_transfer()

        def run() -> None:
            try:
                self._worker_native_thread_id = threading.get_native_id()
                self._binding = self._host.attach_on_render_thread(capability)
                with self._worker_condition:
                    self._worker_bootstrap_ready = True
                    self._worker_condition.notify_all()
                if self._shutdown_requested.is_set():
                    return
                self._create_context_on_render_thread()
                if self._shutdown_requested.is_set():
                    self._request_worker_shutdown()
                with self._worker_condition:
                    if self._worker_closing:
                        return
                self._bring_up_openxr()
                if self._shutdown_requested.is_set():
                    self._request_worker_shutdown()
                with self._worker_condition:
                    if self._worker_closing:
                        return
                    self._worker_handles = self._external_oxr_handles_on_render_thread()
                    self._worker_ready = True
                    self._worker_condition.notify_all()
                while True:
                    with self._worker_condition:
                        if self._worker_closing:
                            break
                        pending_ordinal = self._worker_pending_ordinal
                        self._worker_pending_ordinal = None
                        if pending_ordinal is not None:
                            self._worker_publication_requested = True
                            self._worker_condition.notify_all()
                    if pending_ordinal is not None:
                        frame_began = self._render_published_scene(pending_ordinal)
                    else:
                        frame_began = self._render_reused_scene()
                    if self._worker_handles is None:
                        handles = self._external_oxr_handles_on_render_thread()
                        if handles is not None:
                            with self._worker_condition:
                                self._worker_handles = handles
                                self._worker_condition.notify_all()
                    if not frame_began:
                        with self._worker_condition:
                            if (
                                not self._worker_closing
                                and self._worker_pending_ordinal is None
                            ):
                                self._worker_condition.wait(0.01)
            except BaseException as exc:
                # Thread failures cannot reach the caller automatically. Preserve
                # interrupts too, for caller-side re-raising and cleanup.
                self._record_worker_failure(exc)
            finally:
                if self._worker_ready:
                    self._borrowers_quiesced.wait()
                try:
                    self._teardown_on_render_thread()
                except BaseException as exc:
                    self._record_worker_failure(exc)
                finally:
                    with self._worker_condition:
                        self._closed = True
                        self._worker_condition.notify_all()

        try:
            self._worker = threading.Thread(
                target=run, name="isaacteleop-ovxr-render", daemon=False
            )
            self._worker.start()
        except Exception:
            self._worker = None
            self._closed = True
            try:
                self._host.abort()
            except BaseException:
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
                        self._worker_ready
                        or self._worker_closing
                        or self._worker_error is not None
                        or self._shutdown_requested.is_set()
                    )
                )
                self._raise_worker_error()
                if not self._worker_ready:
                    return False
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
            if self._shutdown_requested.is_set():
                return None
            if self._worker is None or self._worker_closing:
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
                    if self._worker_closing:
                        self._raise_worker_error()
                        if self._shutdown_requested.is_set():
                            return None
                        raise RuntimeError(
                            "OVXR session closed during scene publication"
                        )
                    self._worker_pending_ordinal = ordinal
                    self._worker_condition.notify_all()
                    self._raise_worker_error()
                return ordinal
        except BaseException:
            with self._worker_condition:
                if not self._worker_closing:
                    self._worker_publication_requested = True
                self._worker_condition.notify_all()
            raise

    def external_oxr_handles(self):
        """Return typed handles after peer-gated context startup, otherwise ``None``."""
        with self._worker_condition:
            self._raise_worker_error()
            self._require_open()
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
        ``retirement_complete`` and preserve any raised first error.
        """
        if self._closed:
            self._raise_worker_error()
            return
        if self._worker is None:
            self._host.abort()
            self._closed = True
            self._retirement_complete = True
            return
        self._request_worker_shutdown()
        wake_error: BaseException | None = None
        try:
            self._request_context_exit()
        except BaseException as exc:
            wake_error = exc
        self._borrowers_quiesced.set()
        self._ovxr_teardown_complete.wait(timeout=_WORKER_SHUTDOWN_GRACE_SECONDS)
        fallback_error: BaseException | None = None
        with self._teardown_state_lock:
            if not self._ovxr_teardown_complete.is_set() and self._worker.is_alive():
                _LOGGER.warning(
                    "OVXR teardown did not complete within %.3f seconds; "
                    "interrupting the owned OpenXR runtime",
                    _WORKER_SHUTDOWN_GRACE_SECONDS,
                )
                try:
                    self._interrupt_owned_openxr_runtime()
                except BaseException as exc:
                    fallback_error = exc
        self._worker.join(timeout=30.0)
        if self._worker.is_alive():
            if (
                self._cloudxr_stop_started.is_set()
                and not self._cloudxr_stop_complete.is_set()
            ):
                raise TimeoutError("CloudXR launcher did not stop") from wake_error
            raise TimeoutError("OVXR session worker did not stop") from wake_error
        worker_error: BaseException | None = None
        try:
            self._raise_worker_error()
        except BaseException as exc:
            worker_error = exc
        if worker_error is not None:
            if wake_error is not None and worker_error is not wake_error:
                if fallback_error is not None:
                    wake_error.__cause__ = fallback_error
                    wake_error.__suppress_context__ = True
                raise worker_error from wake_error
            if fallback_error is not None:
                raise worker_error from fallback_error
            raise worker_error
        if wake_error is not None:
            if fallback_error is not None:
                raise wake_error from fallback_error
            raise wake_error
        if fallback_error is not None:
            raise fallback_error

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

    def _create_context_on_render_thread(self) -> None:
        assert self._binding is not None
        if self._config.cloudxr_env_file and self._config.auto_launch_cloudxr:
            self._cloudxr_launcher = _create_cloudxr_launcher(
                self._config.cloudxr_env_file,
                accept_eula=self._config.accept_eula,
            )
            if self._shutdown_requested.is_set():
                return
        ovxr = _import_ovxr()
        extensions = list(
            dict.fromkeys(
                (
                    *self._config.extra_openxr_extensions,
                    *self._required_openxr_extensions,
                )
            )
        )
        with self._host.publication_guard():
            desc = ovxr.ContextDesc(
                ovrtx_renderer=self._binding.renderer,
                ovstage=self._binding.stage,
                application_name=self._config.application_name,
                display_mode=ovxr.DisplayMode.STEREO,
                session_mode={"ar": ovxr.SessionMode.AR, "vr": ovxr.SessionMode.VR}[
                    self._config.session_mode
                ],
                render_products=list(self._config.render_products),
                camera_paths=list(self._config.camera_paths),
                create_render_products=self._config.create_render_products,
                tracking_anchor_path=self._config.tracking_anchor_path,
                floor_aligned_anchor=self._config.floor_aligned_anchor,
                up_axis=ovxr.StageUpAxis.Z,
                meters_per_unit=self._config.meters_per_unit,
                depth_near_z=self._config.depth_near_z,
                depth_far_z=self._config.depth_far_z,
                scene_info_policy=ovxr.SceneInfoPolicy.AUTO,
                extra_openxr_extensions=extensions,
                host_wait_render_output=self._config.host_wait_render_output,
                ordinal_mode=ovxr.StageOrdinalMode.APPLICATION,
            )
            with self._host.owner_lane():
                self._context = ovxr.Context(desc)

    def _request_worker_shutdown(self) -> None:
        with self._worker_condition:
            if self._closed or self._worker_closing:
                return
            self._shutdown_requested.set()
            self._worker_closing = True
            self._worker_condition.notify_all()

    def _interrupt_owned_openxr_runtime(self) -> None:
        launcher = self._cloudxr_launcher
        if launcher is not None:
            launcher.interrupt_runtime()

    def _request_context_exit(self) -> None:
        """Wake an in-flight OpenXR frame wait without stopping CloudXR."""
        if not self._context_lifecycle_lock.acquire(blocking=False):
            return
        try:
            context = self._context
            if context is not None:
                context.request_exit()
        finally:
            self._context_lifecycle_lock.release()

    def _bring_up_openxr(self) -> None:
        ovxr = _import_ovxr()
        handles = self._external_oxr_handles_on_render_thread()
        for _ in range(self._max_blank_frames):
            if handles is not None:
                return
            result, _frame = self._context.submit_openxr_blank_frame()
            if result not in (
                ovxr.Result.SUCCESS,
                ovxr.Result.TIMEOUT,
                ovxr.Result.NOT_READY,
            ):
                raise RuntimeError(
                    f"OVXR blank-frame bootstrap failed: {ovxr.result_name(result)}"
                )
            handles = self._external_oxr_handles_on_render_thread()

    def _external_oxr_handles_on_render_thread(self):
        handles = self._context.get_openxr_handles()
        if handles is None:
            return None
        from isaaccapture.oxr import OpenXRSessionHandles

        return OpenXRSessionHandles(
            handles.xr_instance,
            handles.xr_session,
            handles.xr_space,
            handles.xr_get_instance_proc_addr,
        )

    def _render_published_scene(self, ordinal: int) -> bool:
        """Return whether OpenXR began a frame and supplied its pacing."""
        ovxr = _import_ovxr()
        result, plan = self._context.begin_frame()
        self._worker_last_ordinal = ordinal
        if result not in (
            ovxr.Result.SUCCESS,
            ovxr.Result.TIMEOUT,
            ovxr.Result.NOT_READY,
        ):
            raise RuntimeError(
                f"OVXR begin_frame failed: {ovxr.result_name(result)}: {self._context.last_error}"
            )
        if result != ovxr.Result.SUCCESS:
            return False
        active_plan = plan
        self._pending_plan = plan
        frame_error: BaseException | None = None
        try:
            with self._host.publication_guard():
                if not self._worker_first_deferred_ordinal_validated:
                    self._host.validate_first_published_scene(ordinal)
                    self._worker_first_deferred_ordinal_validated = True
                if self._binding.prepare_frame_scene is not None:
                    self._binding.prepare_frame_scene()
                frame_ordinal = self._binding.ordinal_ledger.reserve()
                try:
                    result, sealed = self._context.bind_frame(
                        active_plan, 0.0, stage_ordinal=frame_ordinal
                    )
                except BaseException as bind_error:
                    try:
                        self._binding.ordinal_ledger.resolve_external(
                            frame_ordinal, expected_published=None
                        )
                    except BaseException as reconciliation_error:
                        raise bind_error from reconciliation_error
                    raise
                self._binding.ordinal_ledger.resolve_external(
                    frame_ordinal,
                    expected_published=sealed,
                )
                if result not in (
                    ovxr.Result.SUCCESS,
                    ovxr.Result.TIMEOUT,
                    ovxr.Result.NOT_READY,
                ):
                    raise RuntimeError(
                        f"OVXR bind_frame failed: {ovxr.result_name(result)}: {self._context.last_error}"
                    )
            if result != ovxr.Result.SUCCESS:
                return True
            layers = []
            result, _frame = self._context.submit_frame(
                active_plan, additional_layers=layers
            )
            if result not in (
                ovxr.Result.SUCCESS,
                ovxr.Result.TIMEOUT,
                ovxr.Result.NOT_READY,
            ):
                raise RuntimeError(
                    f"OVXR submit_frame failed: {ovxr.result_name(result)}: {self._context.last_error}"
                )
            active_plan = None
            self._pending_plan = None
        except BaseException as error:
            frame_error = error
            raise
        finally:
            if active_plan is not None:
                try:
                    self._context.cancel_frame(active_plan)
                    self._pending_plan = None
                except BaseException as cleanup_error:
                    if frame_error is None:
                        raise
                    frame_error.add_note(f"Frame retirement failed: {cleanup_error}")
        return True

    def _render_reused_scene(self) -> bool:
        if self._worker_last_ordinal is None:
            return False
        return self._render_published_scene(self._worker_last_ordinal)

    def _teardown_on_render_thread(self) -> None:
        first_error: BaseException | None = None
        try:
            with self._host.publication_guard():
                self._host.close_publication()
        except BaseException as exc:
            first_error = exc
        try:
            if self._pending_plan is not None:
                self._context.cancel_frame(self._pending_plan)
                self._pending_plan = None
            for quad in reversed(()):
                quad.close()
        except BaseException as retirement_error:
            if first_error is not None:
                first_error.add_note(f"Consumer retirement failed: {retirement_error}")
                raise first_error from retirement_error
            raise
        context_retired = self._context is None
        if self._context is not None:
            try:
                assert self._binding is not None
                with self._context_lifecycle_lock:
                    try:
                        self._context.request_exit()
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                    try:
                        with self._host.publication_guard():
                            ordinal = self._binding.ordinal_ledger.reserve()
                            try:
                                with self._host.owner_lane():
                                    try:
                                        self._context.destroy(stage_ordinal=ordinal)
                                    except _import_ovxr().OvxrError:
                                        # OVXR raises its result error
                                        # only after native destroy consumed the context.
                                        # Pre-call/FFI errors still quarantine borrowed owners.
                                        context_retired = True
                                        raise
                                context_retired = True
                            except BaseException as destroy_error:
                                try:
                                    self._binding.ordinal_ledger.resolve_external(
                                        ordinal, expected_published=None
                                    )
                                except BaseException as reconciliation_error:
                                    raise destroy_error from reconciliation_error
                                raise
                            self._binding.ordinal_ledger.resolve_external(
                                ordinal, expected_published=None
                            )
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                        elif exc is not first_error:
                            first_error.add_note(f"OVXR retirement failed: {exc}")
            finally:
                if context_retired:
                    self._context = None
        if not context_retired:
            with self._teardown_state_lock:
                self._ovxr_teardown_complete.set()
            with self._worker_condition:
                if first_error is not None and self._worker_error is None:
                    self._worker_error = first_error
                self._closed = True
                self._worker_condition.notify_all()
            return
        with self._teardown_state_lock:
            self._ovxr_teardown_complete.set()
        if self._cloudxr_launcher is not None:
            self._cloudxr_stop_started.set()
            try:
                self._cloudxr_launcher.stop()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self._cloudxr_launcher = None
                self._cloudxr_stop_complete.set()
        else:
            self._cloudxr_stop_complete.set()
        try:
            self._host.close_on_render_thread()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        try:
            self._host.close_environment_on_render_thread()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        with self._worker_condition:
            if first_error is not None and self._worker_error is None:
                self._worker_error = first_error
            self._closed = True
            self._retirement_complete = first_error is None
            self._worker_condition.notify_all()

    def _record_worker_failure(self, error: BaseException) -> None:
        self._shutdown_requested.set()
        with self._worker_condition:
            if self._worker_error is None:
                self._worker_error = error
            self._worker_closing = True
            self._worker_condition.notify_all()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise self._worker_error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OVXR session is closed")

    def __enter__(self) -> OvxrSession:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
