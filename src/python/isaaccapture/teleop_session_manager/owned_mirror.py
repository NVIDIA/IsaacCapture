# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Desktop mirror over completed, application-owned eye and camera pixels."""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from .camera_mailbox import _copy_host_rgba, _latest_color_handle
from .gpu_output import MappedGpuImage, OutputMapCleanupError, OwnedGpuImage
from .desktop_mirror import (
    DesktopMirrorConfig,
    DesktopMirrorStatus,
    _CadenceSampler,
    _InputCopyCompletion,
)

_MAX_MIRROR_IMAGE_BYTES = 64 * 1024 * 1024


def _disable_glx_swap_interval() -> bool:
    """Keep the current GLX mirror paced by its FPS cap, not desktop refresh."""
    if sys.platform != "linux":
        return False
    try:
        glx = ctypes.CDLL("libGLX.so.0")
    except OSError:
        return False  # A non-GLX backend has no GLX swap interval.
    glx.glXGetCurrentDisplay.restype = ctypes.c_void_p
    glx.glXGetCurrentContext.restype = ctypes.c_void_p
    glx.glXGetCurrentDrawable.restype = ctypes.c_ulong
    display = glx.glXGetCurrentDisplay()
    context = glx.glXGetCurrentContext()
    drawable = glx.glXGetCurrentDrawable()
    if not (display and context and drawable):
        return False
    glx.glXQueryContext.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    screen = ctypes.c_int()
    # GLX_SCREEN identifies the extension set for this context's X screen.
    if glx.glXQueryContext(display, context, 0x800C, ctypes.byref(screen)) != 0:
        raise RuntimeError("cannot query mirror GLX context screen")
    glx.glXQueryExtensionsString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    glx.glXQueryExtensionsString.restype = ctypes.c_char_p
    extensions = glx.glXQueryExtensionsString(display, screen.value) or b""
    if b"GLX_EXT_swap_control" not in extensions.split():
        raise RuntimeError("GLX mirror requires GLX_EXT_swap_control")
    glx.glXGetProcAddressARB.argtypes = [ctypes.c_char_p]
    glx.glXGetProcAddressARB.restype = ctypes.c_void_p
    address = glx.glXGetProcAddressARB(b"glXSwapIntervalEXT")
    if not address:
        raise RuntimeError("mirror GLX swap control entry point unavailable")
    swap_interval = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int
    )(address)
    # OVUI enables vsync during init. Its native swap keeps the GIL, so a
    # sleeping display can stall headset submission even on another thread.
    # Set this window's interval after init; an environment override alone
    # did not prevent the stall. The existing max_fps cap still paces the UI.
    swap_interval(display, drawable, 0)
    glx.glXQueryDrawable.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint),
    ]
    glx.glXQueryDrawable.restype = None
    interval = ctypes.c_uint(0xFFFFFFFF)
    # GLX_SWAP_INTERVAL_EXT is queried on the same window we just configured.
    glx.glXQueryDrawable(display, drawable, 0x20F1, ctypes.byref(interval))
    if interval.value != 0:
        raise RuntimeError("mirror GLX swap interval did not become zero")
    return True


@dataclass(frozen=True)
class MirrorProduct:
    """An application-selected production identity, never a per-frame arrival."""

    path: str
    label: str
    generation: int = 0


@dataclass(frozen=True)
class OwnedMirrorFrame:
    sequence: int
    width: int
    height: int
    rgba: bytes | None
    predicted_display_time_ns: int
    camera_revision: int = 0
    product: MirrorProduct | None = None
    selection_revision: int = 0
    gpu: MappedGpuImage | OwnedGpuImage | None = None


class OwnedMirrorMailbox:
    """One selected completed output; copies and slow consumers never hold its lock."""

    def __init__(
        self, max_fps: float, *, copier=_copy_host_rgba, gpu_mapper=None
    ) -> None:
        self._period_ns = int(1_000_000_000 / max_fps)
        self._next_ns = 0
        self._lock = threading.Lock()
        self._products: tuple[MirrorProduct, ...] = ()
        self._selected: MirrorProduct | None = None
        self._revision = 0
        self._sequence = 0
        self._latest: OwnedMirrorFrame | None = None
        self._source_key = None
        self._copier = copier
        self._gpu_mapper = gpu_mapper
        self._mapped: list[MappedGpuImage | OwnedGpuImage] = []
        self._in_use: OwnedMirrorFrame | None = None
        self._gpu_error: str | None = None
        self._error: str | None = None
        self._copies = 0
        self._reuses = 0

    def _invalidate(self) -> None:
        self._revision += 1
        self._latest = None
        self._source_key = None
        self._next_ns = 0
        self._error = None

    def set_products(self, products: tuple[MirrorProduct, ...]) -> None:
        """Called only when application production selection/resources change."""
        products = tuple(products)
        if len({p.path for p in products}) != len(products) or any(
            not p.path.startswith("/") or not p.label for p in products
        ):
            raise ValueError("mirror products need unique absolute paths and labels")
        with self._lock:
            if products == self._products:
                return
            self._products = products
            if self._selected not in products:
                self._selected = next(
                    (
                        p
                        for p in products
                        if self._selected and p.path == self._selected.path
                    ),
                    products[0] if products else None,
                )
                self._invalidate()

    def select(self, product: MirrorProduct) -> None:
        with self._lock:
            if product not in self._products:
                return  # A dropdown event from an already retired production set.
            if product != self._selected:
                self._selected = product
                self._invalidate()

    def reset(self) -> None:
        """Invalidate in-flight copies when the application recreates resources."""
        with self._lock:
            self._invalidate()

    def view(self):
        with self._lock:
            return (
                self._products,
                self._selected,
                self._revision,
                self._latest,
                self._error,
            )

    def snapshot(self) -> OwnedMirrorFrame | None:
        with self._lock:
            return self._latest

    def acquire(self, after_sequence: int) -> OwnedMirrorFrame | None:
        """Pin only a new image; a GUI reference alone cannot retain a map."""
        with self._lock:
            image = self._latest
            if image is None or image.sequence <= after_sequence:
                return None
            if self._in_use is not None:
                raise RuntimeError("mirror already has an active consumer")
            self._in_use = image
            return image

    def release(self, image: OwnedMirrorFrame) -> None:
        # Called only after stock OVUI's final source read has completed.
        with self._lock:
            if self._in_use is not image:
                raise RuntimeError("mirror consumer identity changed")
            self._in_use = None

    def _drain_gpu(self, *, replace_latest: bool = False) -> None:
        # Only the renderer lane closes maps. No GPU call holds the UI lock.
        with self._lock:
            if replace_latest and self._latest is not self._in_use:
                self._latest = None
            keep = [
                f.gpu
                for f in (self._latest, self._in_use)
                if f is not None and f.gpu is not None
            ]
            retired = [m for m in self._mapped if not any(m is k for k in keep)]
        for mapped in retired:
            try:
                mapped.close()
            except Exception as error:
                self._gpu_error = str(error)
                raise
            self._mapped = [m for m in self._mapped if m is not mapped]

    def close(self) -> None:
        """Renderer owner, after joining the consumer and before context destruction."""
        with self._lock:
            if self._in_use is not None:
                raise RuntimeError("cannot retire an unfinished mirror GPU consumer")
            self._invalidate()
        self._drain_gpu()
        if hasattr(self._gpu_mapper, "close"):
            self._gpu_mapper.close()

    def finish_frame(self) -> None:
        """Prove the optional copy released all borrowed frame resources."""
        if hasattr(self._gpu_mapper, "finish_frame"):
            self._gpu_mapper.finish_frame()

    def report_error(self, error: Exception) -> None:
        """Expose an optional consumer failure without changing image ownership."""
        with self._lock:
            self._error = str(error)

    def configure_gpu_destination(self) -> None:
        """Called on the GUI lane while its newly created GL context is current."""
        if hasattr(self._gpu_mapper, "configure_opengl_destination"):
            self._gpu_mapper.configure_opengl_destination()

    def counters(self) -> dict[str, int]:
        with self._lock:
            return {"mirror_copies": self._copies, "shared_camera_reuses": self._reuses}

    def on_outputs(self, renderer: int, outputs_pointer: Any, shared=None) -> None:
        """Copy at most the selected output before the borrowed result expires.

        ``shared`` contains already-owned images independently required by PiP.
        A None entry denotes a produced-invalid camera, not a cadence omission.
        """
        now = time.perf_counter_ns()
        with self._lock:
            product, revision = self._selected, self._revision
            previous_key = self._source_key
        gpu = None
        try:
            if self._gpu_error is not None:
                raise RuntimeError(self._gpu_error)
            self._drain_gpu()
            if product is None or now < self._next_ns:
                return
            image = (shared or {}).get(product.path)
            if shared is not None and product.path in shared:
                if image is None:
                    raise RuntimeError("selected camera has no valid owned image")
                key = ("owned", image.revision, image.sensor_time)
                if key == previous_key:
                    return
                width, height, rgba = image.width, image.height, image.rgba
                sensor_time, camera_revision = image.sensor_time, image.revision
                reused = True
            else:
                result = _latest_color_handle(outputs_pointer.contents, product.path)
                if result is None:
                    return  # Cadence gap preserves both entry and last image.
                handle, sensor_time = result
                key = (handle, sensor_time)
                if key == previous_key:
                    return
                if self._gpu_mapper is not None:
                    self._drain_gpu(replace_latest=True)
                    if len(self._mapped) >= 2:
                        return  # Bounded latest/in-use storage; never wait for GUI.
                    gpu = self._gpu_mapper(renderer, handle, _MAX_MIRROR_IMAGE_BYTES)
                    if gpu is None:
                        return  # The GUI has not configured its copy device yet.
                    self._mapped.append(gpu)
                    width, height, rgba = gpu.width, gpu.height, None
                else:
                    width, height, rgba = self._copier(
                        renderer, handle, _MAX_MIRROR_IMAGE_BYTES
                    )
                camera_revision, reused = 0, False
            if (
                width <= 0
                or height <= 0
                or (
                    gpu is None
                    and (
                        len(rgba) != width * height * 4
                        or len(rgba) > _MAX_MIRROR_IMAGE_BYTES
                    )
                )
            ):
                raise ValueError("mirror output is not bounded contiguous RGBA8")
            rgba = bytes(rgba) if rgba is not None else None
            with self._lock:
                self._reuses += int(reused)
                self._copies += int(not reused)
                if (product, revision) != (self._selected, self._revision):
                    return  # Selection/reset changed while native copy ran.
                self._sequence += 1
                self._latest = OwnedMirrorFrame(
                    self._sequence,
                    width,
                    height,
                    rgba,
                    int(sensor_time * 1e9),
                    camera_revision,
                    product,
                    revision,
                    gpu,
                )
                self._source_key = key
                self._next_ns = now + self._period_ns
                self._error = None
        except Exception as error:
            if isinstance(error, OutputMapCleanupError):
                self._mapped.append(error.image)
                self._gpu_error = str(error)
            with self._lock:
                if (product, revision) == (self._selected, self._revision):
                    self._latest = None
                    self._source_key = None
                    self._error = str(error)


class OwnedDesktopMirrorWorker:
    """OVUI consumes only immutable snapshots supplied by the frame owner."""

    def __init__(
        self,
        acquire: Callable[[], OwnedMirrorFrame | None],
        config: DesktopMirrorConfig,
        on_user_close: Callable[[], None] | None = None,
        *,
        products: OwnedMirrorMailbox | None = None,
    ) -> None:
        self._products = products
        self._acquire = acquire
        self._config = config
        self._on_user_close = on_user_close
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status = DesktopMirrorStatus(status_text=config.initial_status_text)
        self._cadence = _CadenceSampler()

    def _update(self, **changes: object) -> None:
        with self._lock:
            self._status = replace(self._status, **changes)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("desktop mirror is already started")
        self._thread = threading.Thread(
            target=self._run, name="isaacteleop-ovui-owned-mirror"
        )
        self._thread.start()

    def _run(self) -> None:
        self._update(thread_id=threading.get_native_id())
        ui = None
        window = None
        failure = None
        completions = {}
        try:
            # CUDA destination discovery needs OpenGL. Select it only for this
            # optional worker, before OVUI initialization; preserve caller policy.
            os.environ.setdefault("OMNIUI_BACKEND", "opengl")
            import omni.ui as ui

            ui.init(
                self._config.title,
                width=self._config.window_width,
                height=self._config.window_width,
                max_fps=self._config.max_fps,
            )
            _disable_glx_swap_interval()
            if self._products is not None:
                self._products.configure_gpu_destination()
            provider = ui.ByteImageProvider()
            window = ui.Window(
                self._config.title,
                width=self._config.window_width,
                height=self._config.window_width,
                fill_app_window=True,
                flags=ui.WINDOW_FLAGS_NO_TITLE_BAR,
                raster_policy=ui.RasterPolicy.NEVER,
            )
            with window.frame:
                with ui.VStack():
                    selector_frame = ui.Frame(height=30) if self._products else None
                    status_label = ui.Label(
                        self.status.status_text,
                        height=48,
                        word_wrap=True,
                        style={"color": 0xFFFFFFFF, "font_size": 18},
                    )
                    with ui.ZStack():
                        ui.Rectangle(style={"background_color": 0xFF181818})
                        image_widget = ui.ImageWithProvider(
                            provider,
                            fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                        )
                        waiting_label = ui.Label("Waiting for a completed image")
            self._update(window_opened=True)

            def rebuild_selector(products, selected):
                def choose(model, _item):
                    index = model.get_item_value_model().get_value_as_int()
                    if 0 <= index < len(products):
                        previous_revision = self._products.view()[2]
                        self._products.select(products[index])
                        if self._products.view()[2] != previous_revision:
                            image_widget.visible = False
                            waiting_label.visible = True
                            waiting_label.text = f"Waiting for {products[index].label}"

                def build():
                    if products:
                        combo = ui.ComboBox(
                            products.index(selected), *(p.label for p in products)
                        )
                        combo.model.add_item_changed_fn(choose)
                    else:
                        ui.Label("No selected render products")

                selector_frame.set_build_fn(build)
                selector_frame.rebuild()

            async def drive() -> None:
                nonlocal failure
                last = 0
                next_status_ns = 0
                shown_products = None
                shown_selected = None
                shown_revision = None
                try:
                    while not self._stop.is_set():
                        # Status belongs to the application, not to a selected
                        # image. Update while waiting or paused, on the UI lane.
                        now_ns = time.perf_counter_ns()
                        if now_ns >= next_status_ns:
                            status_label.text = self.status.status_text
                            next_status_ns = now_ns + 250_000_000
                        if self._products is not None:
                            products, selected, revision, _latest, error = (
                                self._products.view()
                            )
                            if products != shown_products or selected != shown_selected:
                                rebuild_selector(products, selected)
                                shown_products = products
                                shown_selected = selected
                            # The provider owns its uploaded texture. A producer
                            # replacement gap must not hide it; identity or errors do.
                            if revision != shown_revision or error is not None:
                                image_widget.visible = False
                                waiting_label.visible = True
                                waiting_label.text = error or (
                                    f"Waiting for {selected.label}"
                                    if selected
                                    else "No selected render products"
                                )
                                shown_revision = revision
                            image = self._products.acquire(last)
                        else:
                            image = self._acquire()
                        if image is not None and image.sequence > last:
                            safe_to_release = image.gpu is None
                            try:
                                if image.gpu is not None:
                                    gpu = image.gpu
                                    if gpu.closed:
                                        raise RuntimeError(
                                            "mirror GPU map already retired"
                                        )
                                    if gpu.device not in completions:
                                        completions[gpu.device] = _InputCopyCompletion(
                                            gpu.device
                                        )
                                    completion = completions[gpu.device]
                                    completion._check(
                                        completion._cuda.cudaSetDevice(gpu.device)
                                    )
                                    try:
                                        provider.set_bytes_data_from_gpu(
                                            gpu.pointer,
                                            [image.width, image.height],
                                            ui.TextureFormat.RGBA8_UNORM,
                                            gpu.pitch,
                                        )
                                    finally:
                                        completion.wait()
                                        safe_to_release = True
                                else:
                                    if (
                                        image.width <= 0
                                        or image.height <= 0
                                        or len(image.rgba)
                                        != image.width * image.height * 4
                                    ):
                                        raise ValueError(
                                            "mirror snapshot is not tightly packed RGBA8"
                                        )
                                    pixels = np.frombuffer(image.rgba, dtype=np.uint8)
                                    provider.set_data_array(
                                        pixels, [image.width, image.height]
                                    )
                            finally:
                                if self._products is not None and safe_to_release:
                                    self._products.release(image)
                            last = image.sequence
                            current = self._products.view() if self._products else None
                            valid = current is None or (
                                image.product == current[1]
                                and image.selection_revision == current[2]
                                and current[4] is None
                            )
                            image_widget.visible = valid
                            waiting_label.visible = not valid
                            with self._lock:
                                status = self._status
                            if valid:
                                self._update(
                                    presented_frames=status.presented_frames + 1,
                                    last_sequence=last,
                                )
                        await asyncio.sleep(1.0 / self._config.max_fps)
                except GeneratorExit:
                    # Stock standalone can close its loop while drive is
                    # suspended. Coroutine finalization is not an upload error
                    # and must not overwrite the completed window-close status.
                    raise
                except BaseException as error:
                    # OVUI's standalone loop does not retrieve an async task's
                    # exception. Record it here so a failed copy is not
                    # mistaken for a human closing the mirror window.
                    failure = error
                    self._update(error=str(error))
                    self._stop.set()

            ui.run(drive())
        except BaseException as error:
            failure = error
            self._update(error=str(error))
        finally:

            def record_cleanup_error(error):
                nonlocal failure
                if failure is None:
                    failure = error
                    self._update(error=str(error))

            for device, completion in completions.items():
                try:
                    completion._check(completion._cuda.cudaSetDevice(device))
                    completion.close()
                except BaseException as error:
                    record_cleanup_error(error)
            if window is not None:
                try:
                    window.destroy()
                except BaseException as error:
                    record_cleanup_error(error)
            if ui is not None:
                try:
                    ui.shutdown()
                except BaseException as error:
                    record_cleanup_error(error)
            self._update(closed=True)
            if failure is None and not self._stop.is_set() and self._on_user_close:
                try:
                    self._on_user_close()
                except BaseException as error:
                    record_cleanup_error(error)

    def close(self, timeout_s: float = 10.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("owned desktop mirror worker did not stop")

    def record_host_frame(self) -> None:
        with self._lock:
            self._cadence.record_host_frame(time.perf_counter_ns())

    def record_render_frame(self) -> None:
        with self._lock:
            self._cadence.record_render_frame(time.perf_counter_ns())

    def set_status_text(self, text: str) -> None:
        self._update(status_text=str(text))

    @property
    def status(self) -> DesktopMirrorStatus:
        with self._lock:
            host, render = self._cadence.rates(time.perf_counter_ns())
            return replace(self._status, main_thread_fps=host, render_thread_fps=render)
