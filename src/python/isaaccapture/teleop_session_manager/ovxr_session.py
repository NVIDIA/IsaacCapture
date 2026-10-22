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
from collections.abc import Callable, Hashable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from .camera_mailbox import CameraImage, CompletedCameraMailbox
from .openxr_quad import OpenXRCameraQuad
from .gpu_output import (
    completed_outputs,
    map_cuda_array_rgba,
    native_renderer_handle,
)


_WORKER_SHUTDOWN_GRACE_SECONDS = 1.0
_LOGGER = logging.getLogger(__name__)


class OvxrApplicationInterface(Protocol):
    """Application operations required by the OVXR session."""

    def offer_render_owner_transfer(self) -> object: ...

    def attach_on_render_thread(self, capability: object) -> OvxrSessionBinding: ...

    def owner_lane(self) -> AbstractContextManager[None]: ...

    def publication_guard(self) -> AbstractContextManager[None]: ...

    def publish_complete_scene(self, publication_id: Hashable) -> int: ...

    def validate_first_published_scene(self, ordinal: int) -> None: ...

    def get_target_frame_T_world(self, prim_path: str) -> Any: ...

    def close_publication(self) -> None: ...

    def close_on_render_thread(self) -> None: ...

    def close_environment_on_render_thread(self) -> None: ...

    def abort(self) -> None: ...


class StageOrdinalLedger(Protocol):
    """Application-owned reservations for explicit OVXR Stage writes."""

    def reserve(self) -> int: ...

    def resolve_external(
        self, ordinal: int, *, expected_published: bool | None
    ) -> bool: ...

    def abandon(self, ordinal: int) -> None: ...


@dataclass(frozen=True)
class OvxrSessionBinding:
    """Exact application-owned objects borrowed by one OVXR context."""

    renderer: object
    stage: object
    ordinal_ledger: StageOrdinalLedger
    auxiliary_render_product_path: str | None = None
    auxiliary_render_product_paths: tuple[str, ...] = ()
    prepare_frame_scene: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if self.renderer is None or self.stage is None or self.ordinal_ledger is None:
            raise ValueError(
                "OVXR session requires renderer, Stage, and ordinal ledger"
            )
        paths = self.selected_auxiliary_paths
        if (
            len(paths) > 3
            or len(paths) != len(set(paths))
            or any(not path.startswith("/") for path in paths)
        ):
            raise ValueError(
                "auxiliary RenderProduct paths must be absolute, unique, and at most three"
            )

    @property
    def selected_auxiliary_paths(self) -> tuple[str, ...]:
        if self.auxiliary_render_product_path is not None:
            if self.auxiliary_render_product_paths:
                raise ValueError("use singular or plural auxiliary paths, not both")
            return (self.auxiliary_render_product_path,)
        return self.auxiliary_render_product_paths


@dataclass(frozen=True)
class AuxiliaryRasterConfig:
    """Application-owned cadence and head-locked panel placement.

    The host applies ``max_fps`` when producing its RenderProduct. This owner
    copies completed outputs without imposing another cadence limit.
    """

    max_fps: float = 30.0
    max_image_bytes: int = 64 * 1024 * 1024
    panel_width_m: float = 0.48
    position: tuple[float, float, float] = (0.0, -0.15, -0.8)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    space: Literal["head_locked"] = "head_locked"
    layer_manifest: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_fps) or self.max_fps <= 0.0:
            raise ValueError("auxiliary raster max_fps must be finite and positive")
        if self.max_image_bytes <= 0:
            raise ValueError("auxiliary raster max_image_bytes must be positive")
        if not math.isfinite(self.panel_width_m) or self.panel_width_m <= 0.0:
            raise ValueError(
                "auxiliary raster panel_width_m must be finite and positive"
            )
        if len(self.position) != 3 or not all(
            math.isfinite(value) for value in self.position
        ):
            raise ValueError(
                "auxiliary raster position must contain three finite values"
            )
        if len(self.orientation_xyzw) != 4 or not all(
            math.isfinite(value) for value in self.orientation_xyzw
        ):
            raise ValueError(
                "auxiliary raster orientation must contain four finite values"
            )
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if norm <= 0.0:
            raise ValueError("auxiliary raster orientation must have non-zero norm")
        object.__setattr__(
            self,
            "orientation_xyzw",
            tuple(value / norm for value in self.orientation_xyzw),
        )
        if self.space != "head_locked":
            raise ValueError("auxiliary raster space must be 'head_locked'")


@dataclass
class OvxrSessionConfig:
    """Application-neutral OVXR and CloudXR configuration.

    The supported OVXR package supplies bound-frame outputs. Optional PiP and
    output consumers copy those outputs on the sole frame owner before retirement.
    """

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
    auxiliary_raster: AuxiliaryRasterConfig | None = None
    auxiliary_rasters: tuple[AuxiliaryRasterConfig, ...] = ()
    extra_openxr_extensions: tuple[str, ...] = ()

    @property
    def selected_auxiliary_rasters(self) -> tuple[AuxiliaryRasterConfig, ...]:
        if self.auxiliary_raster is not None:
            if self.auxiliary_rasters:
                raise ValueError(
                    "use singular or plural auxiliary raster policy, not both"
                )
            return (self.auxiliary_raster,)
        return self.auxiliary_rasters

    def __post_init__(self) -> None:
        if len(self.selected_auxiliary_rasters) > 3:
            raise ValueError("at most three auxiliary camera quads are supported")
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


class OvxrFrameConsumer(Protocol):
    """Frame-owner callbacks for an optional completed-output consumer.

    Initialize after context creation. ``finish_frame`` releases borrowed outputs
    before submit/cancel. ``close`` must tolerate incomplete startup and raise if
    retirement is unproved, so rendering providers stay alive.
    """

    def on_session_started(
        self,
        binding: OvxrSessionBinding,
        config: OvxrSessionConfig,
        request_shutdown: Callable[[], None],
    ) -> None: ...

    def on_outputs(
        self, renderer: int, outputs: Any, shared: Mapping[str, CameraImage | None]
    ) -> None: ...

    def on_frame_submitted(self) -> None: ...

    def finish_frame(self) -> None: ...

    def close(self) -> None: ...


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
        frame_consumers: Iterable[OvxrFrameConsumer] = (),
    ) -> None:
        if max_blank_frames < 0:
            raise ValueError("max_blank_frames must be non-negative")
        self._host = host
        self._config = config
        self._required_openxr_extensions = tuple(required_openxr_extensions)
        self._max_blank_frames = max_blank_frames
        self._frame_consumers = tuple(frame_consumers)
        self._context = None
        self._pending_plan = None
        self._binding: OvxrSessionBinding | None = None
        self._cloudxr_launcher = None
        self._camera_quad_attached = False
        self._camera_mailboxes: list[CompletedCameraMailbox] = []
        self._camera_quads: list[OpenXRCameraQuad] = []
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
        Expiry requests cancellation; an attachment already in progress must finish
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
                    # A peer can become ready after the bounded blank-frame
                    # bootstrap. Publish its input handles from the owning
                    # render thread once the session has actually appeared.
                    if self._worker_handles is None:
                        handles = self._external_oxr_handles_on_render_thread()
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
                                not self._worker_closing
                                and self._worker_pending_ordinal is None
                            ):
                                self._worker_condition.wait(0.01)
            except BaseException as exc:
                self._record_worker_failure(exc)
            finally:
                # Once handles are exposed, only the host can acknowledge that
                # its application step and DeviceIO borrowers have stopped.
                if self._worker_ready:
                    self._borrowers_quiesced.wait()
                try:
                    self._teardown_on_render_thread()
                except BaseException as exc:
                    # Failed consumer retirement retains its borrowed owners.
                    # Report it to close() even when rendering itself succeeded.
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
            # No owner thread accepted the borrowed pair; rollback stays on the application thread.
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
            bootstrap_ready = self._worker_condition.wait_for(
                lambda: (
                    self._worker_bootstrap_ready
                    or self._worker_error is not None
                    or self._shutdown_requested.is_set()
                ),
                timeout_s,
            )
            if not bootstrap_ready:
                # Expiry must set the same cancellation event checked after slow attach.
                self._request_worker_shutdown()
                raise TimeoutError("OVXR session worker bootstrap did not become ready")
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
        """Wait interruptibly for a shutdown request."""
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
        auxiliaries = self._config.selected_auxiliary_rasters
        auxiliary_paths = self._binding.selected_auxiliary_paths
        if len(auxiliaries) != len(auxiliary_paths):
            raise RuntimeError(
                "auxiliary raster policy and borrowed RenderProduct paths must be provided together"
            )
        if len({cfg.layer_manifest for cfg in auxiliaries}) > 1:
            raise ValueError("all camera panels must use the same native quad helper")
        for index, auxiliary in enumerate(auxiliaries):
            quad = OpenXRCameraQuad(auxiliary.layer_manifest, camera_index=index)
            self._camera_quads.append(quad)
            quad.configure(auxiliary)
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
                # The server mode must reach OVXR; the browser AR preference cannot
                # enable transparent rendering in an opaque server projection.
                session_mode={"ar": ovxr.SessionMode.AR, "vr": ovxr.SessionMode.VR}[
                    self._config.session_mode
                ],
                # Empty means OVXR creates its default eye products. Supplying
                # those names explicitly instead means existing products and
                # changes the production/authoring contract.
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
                self._camera_mailboxes = [
                    CompletedCameraMailbox(
                        path,
                        policy.max_image_bytes,
                        gpu_mapper=map_cuda_array_rgba if quad.supports_gpu else None,
                    )
                    for path, policy, quad in zip(
                        auxiliary_paths, auxiliaries, self._camera_quads, strict=True
                    )
                ]
                if auxiliaries or self._frame_consumers:
                    self._context.set_frame_render_products(list(auxiliary_paths))
        for consumer in self._frame_consumers:
            consumer.on_session_started(
                self._binding, self._config, self.request_shutdown
            )

    def _on_completed_outputs(self, user_data, renderer, outputs) -> None:
        """Frame-owner copies before result retirement; PiP owns its independent copy."""
        shared = {}
        for mailbox in self._camera_mailboxes:
            mailbox.on_outputs(user_data, renderer, outputs)
            if self._frame_consumers:
                image = mailbox.snapshot()
                if mailbox.error() is not None or (
                    image is not None and image.gpu is None
                ):
                    shared[mailbox.product_path] = image
        for consumer in self._frame_consumers:
            consumer.on_outputs(renderer, outputs, shared)

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
        # A startup failure may already be destroying under this lock. Never
        # wait here and prevent close() from reaching its runtime fallback.
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

        if self._camera_quads and not self._camera_quad_attached:
            vulkan = self._context.get_vulkan_handles()
            if vulkan is None:
                return None
            self._camera_quads[0].attach(handles, vulkan)
            self._camera_quad_attached = True
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
                    # Publication is independent of the eventual bind result:
                    # a later failure must not reuse an already sealed ordinal.
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
            if (
                self._camera_mailboxes or self._frame_consumers
            ) and active_plan.step_result:
                try:
                    renderer = native_renderer_handle(self._binding.renderer)
                    outputs = completed_outputs(renderer, active_plan.step_result)
                    self._on_completed_outputs(None, renderer, outputs)
                    for quad, mailbox in zip(
                        self._camera_quads, self._camera_mailboxes, strict=True
                    ):
                        try:
                            quad.publish(
                                mailbox.snapshot(), invalid=mailbox.error() is not None
                            )
                            if layer := quad.layer():
                                layers.append(layer)
                        except Exception:
                            _LOGGER.exception("OpenXR camera quad publication failed")
                            # Omit this layer even if optional native hiding fails.
                            try:
                                quad.hide()
                            except Exception:
                                _LOGGER.exception("OpenXR camera quad hiding failed")
                except Exception:
                    _LOGGER.exception("Optional bound-frame output observation failed")
                    layers.clear()
                    for quad in self._camera_quads:
                        try:
                            quad.hide()
                        except Exception:
                            _LOGGER.exception("OpenXR camera quad hiding failed")
                finally:
                    self._finish_frame_outputs()
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
            if (
                result == ovxr.Result.SUCCESS
                and _frame.submitted
                and self._frame_consumers
            ):
                # Submit consumed the plan; consumer callbacks must not
                # cancel it even if a callback fails.
                active_plan = None
                self._pending_plan = None
                for consumer in self._frame_consumers:
                    consumer.on_frame_submitted()
            active_plan = None
            self._pending_plan = None
        except BaseException as error:
            frame_error = error
            raise
        finally:
            if active_plan is not None:
                # An unproved copy/unmap keeps the plan and context alive. Never
                # consume borrowed output storage merely to recover a frame.
                try:
                    self._finish_frame_outputs()
                    self._context.cancel_frame(active_plan)
                    self._pending_plan = None
                except BaseException as cleanup_error:
                    if frame_error is None:
                        raise
                    frame_error.add_note(f"Frame retirement failed: {cleanup_error}")
        return True

    def _finish_frame_outputs(self) -> None:
        for mailbox in self._camera_mailboxes:
            mailbox.close()
        for consumer in self._frame_consumers:
            consumer.finish_frame()

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
            for consumer in reversed(self._frame_consumers):
                # A failed completion/unmap must keep the renderer alive: never
                # destroy storage that an optional consumer might still access.
                consumer.close()
            for mailbox in self._camera_mailboxes:
                mailbox.close()
            if self._pending_plan is not None:
                self._finish_frame_outputs()
                self._context.cancel_frame(self._pending_plan)
                self._pending_plan = None
            # The helper borrows the session/device, so its resources retire first.
            for quad in reversed(self._camera_quads):
                quad.close()
        except BaseException as retirement_error:
            if first_error is not None:
                first_error.add_note(f"Consumer retirement failed: {retirement_error}")
                # Keep the primary error and the failed retirement cause visible.
                raise first_error from retirement_error
            raise
        context_retired = self._context is None
        if self._context is not None:
            try:
                assert self._binding is not None
                with self._context_lifecycle_lock:
                    # The close caller normally requested this already to wake
                    # xrWaitFrame. Repeat idempotently on the owner thread so
                    # worker-failure teardown follows the same graceful path.
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
            # A returned worker does not prove a failed destroy retired its
            # borrowed Stage/renderer. Keep the context, binding, host and
            # runtime alive; the terminal owner must never retry or reuse them.
            with self._teardown_state_lock:
                self._ovxr_teardown_complete.set()
            with self._worker_condition:
                if first_error is not None and self._worker_error is None:
                    self._worker_error = first_error
                self._closed = True
                self._worker_condition.notify_all()
            return
        self._camera_quads.clear()
        self._camera_mailboxes.clear()
        # This is a terminal teardown milestone, including a failed destroy
        # call.  Once the owner thread has left OVXR teardown, interrupting the
        # runtime can only race the later, already-running CloudXR stop.
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
