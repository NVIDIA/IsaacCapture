# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner-thread OVXR frame processing and native resource retirement."""

from __future__ import annotations
import logging
import threading
from collections.abc import Callable, Iterable
from .interfaces import OvxrApplicationInterface, OvxrSessionBinding, OvxrSessionConfig
from .runtime_owner import CloudXRRuntimeOwner

_LOGGER = logging.getLogger(__name__)


def _import_ovxr():
    import ovxr

    return ovxr


class OvxrFramePump:
    """Own native state; the session supplies scheduling and shutdown signals."""

    def __init__(
        self,
        host: OvxrApplicationInterface,
        config: OvxrSessionConfig,
        runtime: CloudXRRuntimeOwner,
        *,
        shutdown_requested: Callable[[], bool],
        request_shutdown: Callable[[], None],
        required_openxr_extensions: Iterable[str],
        max_blank_frames: int,
    ) -> None:
        self._host = host
        self._config = config
        self._runtime = runtime
        self._shutdown_requested = shutdown_requested
        self.request_shutdown = request_shutdown
        self._required_openxr_extensions = tuple(required_openxr_extensions)
        self._max_blank_frames = max_blank_frames
        self._context = None
        self._binding: OvxrSessionBinding | None = None
        self._pending_plan = None
        self._context_lifecycle_lock = threading.Lock()
        self._last_ordinal: int | None = None
        self._first_ordinal_validated = False

    def attach(self, capability: object) -> None:
        self._binding = self._host.attach_on_render_thread(capability)

    @staticmethod
    def validate_provider() -> None:
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

    def create_context(self) -> None:
        assert self._binding is not None
        self._runtime.start()
        if self._shutdown_requested():
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
                up_axis=getattr(ovxr.StageUpAxis, self._config.up_axis),
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

    def request_exit(self) -> None:
        """Wake an in-flight OpenXR frame wait without stopping CloudXR."""
        # Startup failure may already be retiring the context on its owner.
        # Do not block the caller before its bounded worker join.
        if not self._context_lifecycle_lock.acquire(blocking=False):
            return
        try:
            context = self._context
            if context is not None:
                context.request_exit()
        finally:
            self._context_lifecycle_lock.release()

    def bootstrap(self) -> None:
        ovxr = _import_ovxr()
        handles = self.external_handles()
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
            handles = self.external_handles()

    def external_handles(self):
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

    def render(self, ordinal: int) -> bool:
        """Return whether OpenXR began a frame and supplied its pacing."""
        ovxr = _import_ovxr()
        result, plan = self._context.begin_frame()
        self._last_ordinal = ordinal
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
                if not self._first_ordinal_validated:
                    self._host.validate_first_published_scene(ordinal)
                    self._first_ordinal_validated = True
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
            result, _frame = self._context.submit_frame(active_plan)
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

    def render_reused(self) -> bool:
        if self._last_ordinal is None:
            return False
        return self.render(self._last_ordinal)

    def close(self) -> None:
        first_error: BaseException | None = None
        try:
            with self._host.publication_guard():
                self._host.close_publication()
        except BaseException as exc:
            first_error = exc
        if self._pending_plan is not None:
            try:
                self._context.cancel_frame(self._pending_plan)
            except BaseException as exc:
                # Native destroy remains responsible for the pending frame.
                if first_error is None:
                    first_error = exc
            else:
                self._pending_plan = None
        context_retired = self._context is None
        if self._context is not None:
            assert self._binding is not None
            with self._context_lifecycle_lock:
                try:
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
                                    else:
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
                        self._pending_plan = None
        if not context_retired:
            # A pre-call/FFI failure does not prove native consumption. Keep its
            # renderer, Stage and runtime alive; never retry an ambiguous destroy.
            assert first_error is not None
            raise first_error
        for close in (
            self._runtime.close,
            self._host.close_on_render_thread,
            self._host.close_environment_on_render_thread,
        ):
            try:
                close()
            except BaseException as exc:
                # Finish independent cleanup, then propagate the first failure.
                if first_error is None:
                    first_error = exc
                elif exc is not first_error:
                    first_error.add_note(f"Dependent cleanup failed: {exc}")
        if first_error is not None:
            raise first_error

    def retry_runtime_stop(self) -> None:
        """Retry service cleanup only after native borrowers have retired."""
        if self._context is None:
            self._runtime.close()
