# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-driven run-loop for camera_viz.

VizRunner owns two threads:

  * **submit thread** — polls each source's ``latest()`` at ~1 kHz,
    calls ``layer.submit()`` on new frames, and notifies a
    condition variable.
  * **render thread** — waits on the condition. Wakes within ~µs of a
    new publish and calls ``session.render()``. A safety-net timeout
    re-runs render() periodically for window events / XR placement
    updates even without new frames.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Optional, Sequence

import isaaccapture.viz as viz

from ..dashboard import CameraRow, Dashboard, Snapshot

from .interface import FrameSource

logger = logging.getLogger("isaaccapture.camera_viz.pipeline.runner")


def _measure_ipd_mm(info) -> Optional[float]:
    """Headset IPD from the per-eye view poses, or None outside XR stereo.

    This is what the runtime believes the lens separation to be. It drives
    the stereo-distance maths, and a user whose headset IPD setting is wrong
    trims the difference out by hand on the stick.
    """
    views = getattr(info, "views", None)
    if views is None or len(views) < 2:
        return None
    left, right = views[0].pose.position, views[1].pose.position
    ipd = math.dist(tuple(left), tuple(right)) * 1000.0
    # A degenerate value means the runtime has not placed the eyes yet.
    return ipd if ipd > 1.0 else None


# Submit thread poll interval when no source has new data.
SUBMIT_POLL_S = 0.001

# Pipeline-stats print interval. One line per period on stderr showing
# per-source submit rate and the session's render rate — the first thing
# to look at when "the fps is low": a submit rate below the camera's
# capture rate points at the source/pairing/submit path, a render rate
# below the display rate points at the XR runtime pacing (missed
# deadlines / GPU time).
STATS_PERIOD_S = 5.0

# The panel is a snapshot, so it refreshes far more often than a log would.
LIVE_STATS_PERIOD_S = 0.5

# Window-mode stop-check granularity. stop() calls cond.notify_all()
# so this is normally a safety net, not a hot path — value isn't a
# render rate, it's "how long until Ctrl-C is honored if a notify
# is somehow lost." No XR equivalent: XR's render loop is paced by
# xrWaitFrame and iterates at display rate, which is itself a tight
# stop-check granularity (~one display period).
STOP_CHECK_INTERVAL_S = 0.5


class VizRunner:
    """Wires sources → layers and runs submit + render threads.

    Caller owns the ``VizSession`` and the layers. ``placement_strategies``
    is a parallel list; ``None`` entries are valid for layers whose
    placement is fixed at construction (window mode, or a kCustom XR
    placement set externally).
    """

    def __init__(
        self,
        session: viz.VizSession,
        sources: Sequence[FrameSource],
        layers: Sequence[viz.QuadLayer],
        placement_strategies: Optional[Sequence[Optional[object]]] = None,
        controls: Optional[object] = None,
        dashboard: Optional[Dashboard] = None,
        header: str = "",
        notes: Optional[Sequence[str]] = None,
    ) -> None:
        if len(sources) != len(layers):
            raise ValueError(
                f"sources / layers length mismatch: {len(sources)} vs {len(layers)}"
            )
        if placement_strategies is not None and len(placement_strategies) != len(
            layers
        ):
            raise ValueError(
                f"placement_strategies / layers length mismatch: "
                f"{len(placement_strategies)} vs {len(layers)}"
            )

        self._session = session
        self._sources = list(sources)
        self._layers = list(layers)
        self._strategies = (
            list(placement_strategies)
            if placement_strategies is not None
            else [None] * len(layers)
        )
        # Polled on the render thread once per XR frame. It owns the live
        # shape / lock-mode state, so the active layer and strategy are
        # resolved through it rather than read from the lists above.
        self._controls = controls
        self._dashboard = dashboard if dashboard is not None else Dashboard()
        self._header = header
        self._notes = list(notes or ())
        self._stop = threading.Event()
        self._submit_thread: Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None
        self._stats_thread: Optional[threading.Thread] = None
        # Submit thread bumps the version + notifies after each publish;
        # render thread compares versions under the lock, so wakeups
        # can't be lost.
        self._data_cond = threading.Condition()
        self._data_version = 0
        # First exception raised by either loop. ``wait()`` re-raises it
        # so the main thread sees a thread death instead of silently
        # falling through to ``return 0``.
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()
        # Per-source submit counters (submit thread writes, stats print
        # reads — plain ints under the GIL, approximate reads are fine).
        self._submit_counts = [0] * len(self._layers)

    def start(self) -> None:
        if self._submit_thread is not None or self._render_thread is not None:
            raise RuntimeError("VizRunner already started")
        self._stop.clear()
        # Roll back started sources on any failure so the runner doesn't
        # leak producer threads.
        started: list[FrameSource] = []
        try:
            for s in self._sources:
                s.start()
                started.append(s)
        except Exception:
            for s in reversed(started):
                try:
                    s.stop()
                except Exception:
                    pass
            raise
        self._submit_thread = threading.Thread(
            target=self._submit_loop, name="camera_viz_submit", daemon=False
        )
        self._submit_thread.start()
        self._render_thread = threading.Thread(
            target=self._render_loop, name="camera_viz_render", daemon=False
        )
        self._render_thread.start()
        # Daemon: it only draws, so it must never hold up an exit.
        self._stats_thread = threading.Thread(
            target=self._stats_loop, name="camera_viz_stats", daemon=True
        )
        self._stats_thread.start()

    def stop(self) -> bool:
        """Returns True iff both worker threads exited within the join budget.

        Callers MUST NOT destroy the VizSession on False — a thread is
        still inside session.render() / layer.submit() and tearing the
        session down under it is a use-after-free on Vulkan / CUDA
        handles. The non-daemon thread keeps the process alive until
        it exits; the OS reaps the session at process exit.
        """
        self._stop.set()
        # Wake the render thread's cond.wait.
        with self._data_cond:
            self._data_cond.notify_all()
        # Bounded joins so a wedged session.render() / source doesn't
        # block Ctrl-C. Sources always get stop()ped (camera / gst
        # handles) even if a thread is stuck.
        clean = True
        try:
            if self._render_thread is not None:
                self._render_thread.join(timeout=5.0)
                if self._render_thread.is_alive():
                    logger.warning("render thread did not exit within 5s")
                    clean = False
                else:
                    self._render_thread = None
            if self._submit_thread is not None:
                self._submit_thread.join(timeout=5.0)
                if self._submit_thread.is_alive():
                    logger.warning("submit thread did not exit within 5s")
                    clean = False
                else:
                    self._submit_thread = None
        finally:
            for s in self._sources:
                try:
                    s.stop()
                except Exception:
                    logger.exception("source.stop() raised")
        return clean

    def wait(self, health_check: Optional[Callable[[], None]] = None) -> None:
        """Block until the render thread exits, then re-raise any captured
        thread error. Polls so SIGINT is delivered and optional external
        dependencies can report failure on the main thread."""
        while self._render_thread is not None and self._render_thread.is_alive():
            self._render_thread.join(timeout=0.1)
            if health_check is not None:
                health_check()
        # The submit thread may still be running (it exits on _stop set
        # by render's exit / signal handler / record_error). Give it the
        # same poll-loop courtesy so its error has a chance to land
        # before we re-raise.
        while self._submit_thread is not None and self._submit_thread.is_alive():
            self._submit_thread.join(timeout=0.1)
        with self._error_lock:
            err = self._error
        if err is not None:
            raise err

    def __enter__(self) -> "VizRunner":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # Capture the first exception either loop raises, signal stop so the
    # peer thread exits cleanly, and let wait() re-raise to the main
    # thread. Without this, a dead thread silently leaves the main
    # process running.
    def _record_error(self, exc: BaseException, where: str) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc
        logger.error("VizRunner %s thread failed: %s", where, exc, exc_info=exc)
        self._stop.set()
        with self._data_cond:
            self._data_cond.notify_all()

    # ── Submit thread ──────────────────────────────────────────────────

    def _active_layer(self, index: int):
        """Layer to submit to / place this frame.

        Resolved through the controls, never from ``self._layers``: this
        class copies the lists it is handed, so a shape switch made by the
        controls would otherwise be invisible here and the runner would keep
        feeding the layer that just went hidden.
        """
        if self._controls is not None:
            return self._controls.active_layer(index)
        return self._layers[index]

    def _active_strategy(self, index: int):
        """Placement strategy for this frame; same reasoning as above (the
        A button swaps it)."""
        if self._controls is not None:
            return self._controls.strategy(index)
        return self._strategies[index]

    def _submit_loop(self) -> None:
        try:
            self._submit_loop_inner()
        except BaseException as e:  # noqa: BLE001 — propagate everything
            self._record_error(e, "submit")

    def _submit_loop_inner(self) -> None:
        # Pin to the source's GPU on the first frame.
        device_pinned = False
        # Which layer each source was last submitted to, and the frame it
        # sent, so a shape switch can re-send immediately (below).
        last_layers = [None] * len(self._layers)
        last_frames = [None] * len(self._layers)
        while not self._stop.is_set():
            published_any = False
            for i, source in enumerate(self._sources):
                # Re-read per pass: the controls swap this on a shape change.
                layer = self._active_layer(i)
                frame = source.latest()
                if frame is None:
                    # No new frame. If the shape just changed, re-send the
                    # last one so the newly visible layer isn't blank until
                    # the camera produces the next (up to 1/fps away). The
                    # source may have recycled the buffer, so the worst case
                    # is one torn frame on the switch, not a stall.
                    if layer is last_layers[i] or last_frames[i] is None:
                        continue
                    frame = last_frames[i]
                if not device_pinned:
                    self._pin_to_device(frame)
                    device_pinned = True
                if frame.image_right is not None:
                    # A stereo layer must always be fed two buffers, so the
                    # mono override ships the left frame to both eyes rather
                    # than dropping to the one-arg form (which would throw).
                    right = (
                        frame.image
                        if self._controls is not None and self._controls.force_mono(i)
                        else frame.image_right
                    )
                    layer.submit(frame.image, right, stream=frame.stream)
                else:
                    layer.submit(frame.image, stream=frame.stream)
                last_layers[i] = layer
                last_frames[i] = frame
                self._submit_counts[i] += 1
                published_any = True
            if published_any:
                with self._data_cond:
                    self._data_version += 1
                    self._data_cond.notify()
            else:
                self._stop.wait(timeout=SUBMIT_POLL_S)

    # ── Stats thread ───────────────────────────────────────────────────

    def _stats_loop(self) -> None:
        """The panel on its own thread, never the submit thread.

        Writing it from the submit thread put a blocking write() between the
        camera and the layer: a terminal that stops draining (a paused ssh
        session, tmux scrollback, ^S) fills the pty buffer in about 25 paints
        and then the feed stops until someone scrolls back down. A paint costs
        0.035 ms on a local pty -- it is the tail that matters, not the median.
        """
        period = LIVE_STATS_PERIOD_S if self._dashboard.live else STATS_PERIOD_S
        last = time.monotonic()
        while not self._stop.wait(timeout=period):
            now = time.monotonic()
            try:
                self._print_stats(now - last)
            except Exception:  # noqa: BLE001 — a broken pipe must not stop the feed
                logger.debug("status panel write failed", exc_info=True)
            last = now

    def _print_stats(self, elapsed: float) -> None:
        """Push a snapshot of everything to the status panel.

        The panel redraws in place on a terminal and falls back to one line
        per period when it isn't one, so this is the only stats path.
        """
        rows = []
        for i, source in enumerate(self._sources):
            rate = self._submit_counts[i] / elapsed if elapsed > 0 else 0.0
            self._submit_counts[i] = 0
            state = self._controls.status(i) if self._controls is not None else {}
            rows.append(
                CameraRow(
                    name=source.spec.name,
                    shape=state.get("shape", "quad"),
                    lock_mode=state.get("lock_mode", "-"),
                    stereo=bool(state.get("stereo", False)),
                    size_m=state.get("size_m"),
                    offset_y_m=state.get("offset_y_m"),
                    plane_distance_cm=state.get("plane_distance_cm"),
                    suggested_cm=state.get("suggested_cm"),
                    submit_fps=rate,
                )
            )
        self._dashboard.show(
            Snapshot(
                header=self._header,
                render=self._render_summary(),
                rows=rows,
                ipd_mm=self._controls.ipd_mm if self._controls is not None else None,
                notes=self._notes,
                last_event=self._controls.last_event
                if self._controls is not None
                else "",
            )
        )

    def _render_summary(self) -> str:
        try:
            t = self._session.get_frame_timing_stats()
        except Exception:
            return "render n/a"
        return (
            f"render {t.render_fps:.1f} fps"
            + (f" (target {t.target_fps:.0f})" if t.target_fps else "")
            + f"   missed {t.missed_frames}"
            + (f"   gpu {t.gpu_time_ms:.1f} ms" if t.gpu_time_ms else "")
            + (f"   stale {t.stale_layers}" if t.stale_layers else "")
        )

    def _pin_to_device(self, frame) -> None:
        try:
            import cupy as cp

            cp.cuda.runtime.setDevice(int(frame.image.device.id))
        except Exception:
            pass

    # ── Render thread ──────────────────────────────────────────────────

    def _render_loop(self) -> None:
        try:
            self._render_loop_inner()
        except BaseException as e:  # noqa: BLE001 — propagate everything
            self._record_error(e, "render")

    def _render_loop_inner(self) -> None:
        # Two distinct loop shapes, principled per mode:
        #   XR: tight loop, paced by xrWaitFrame inside session.render().
        #       The runtime requires xrEndFrame every display tick, so
        #       there's no "idle skip" option — we render even with
        #       stale data. Stop is checked once per iteration ≈ one
        #       display period.
        #   Window: pure event-driven. Render only on producer notify
        #       (cond.wait blocks indefinitely until notify or stop).
        #       The display already shows the last presented frame, so
        #       skipping idle renders is correct; window events go
        #       through pump_events on the main thread, not us.
        if self._session.is_xr_mode():
            self._render_loop_xr()
        else:
            self._render_loop_window()

    def _render_loop_xr(self) -> None:
        last = time.monotonic()
        ipd_mm = None
        while not self._stop.is_set():
            now = time.monotonic()
            dt, last = now - last, now
            if self._controls is not None:
                # Before placements: a lock-mode change this frame should
                # take effect on this frame's pose, not the next one. The IPD
                # is one frame stale, which is irrelevant -- it is fixed by
                # the headset's lens separation.
                self._controls.step(dt, ipd_mm)
            self._apply_xr_placements()
            info = self._session.render()
            ipd_mm = _measure_ipd_mm(info) or ipd_mm
            if self._session.should_close():
                self._stop.set()

    def _render_loop_window(self) -> None:
        last_seen_version = 0
        while not self._stop.is_set():
            with self._data_cond:
                if self._data_version == last_seen_version:
                    # Block until producer notifies OR stop() wakes us.
                    # The timeout is just a Ctrl-C safety net for the
                    # case where a notify is lost; not a render rate.
                    self._data_cond.wait(timeout=STOP_CHECK_INTERVAL_S)
                last_seen_version = self._data_version
            if self._stop.is_set():
                break
            self._session.render()
            if self._session.should_close():
                self._stop.set()

    def _apply_xr_placements(self) -> None:
        strategies = [self._active_strategy(i) for i in range(len(self._layers))]
        if not any(s is not None for s in strategies):
            return
        head = self._session.head_pose_now()
        if head is None:
            return
        for i, strategy in enumerate(strategies):
            if strategy is None:
                continue
            layer = self._active_layer(i)
            # An equirect sphere is centred on the operator: no pose to
            # re-lock, and its placement type isn't a QuadLayerPlacement.
            if isinstance(layer, viz.EquirectLayer):
                continue
            placement = strategy.update(head.position, head.orientation)
            if isinstance(layer, viz.CylinderLayer):
                # The cylinder's pose is the strategy's head anchor (its arc
                # bows out along the anchor's -z at radius); radius / angle /
                # aspect stay as configured.
                cyl = layer.placement()
                cyl.pose = viz.Pose3D(
                    placement.anchor_position, placement.anchor_orientation
                )
                layer.set_placement(cyl)
            else:
                layer.set_placement(
                    viz.QuadLayerPlacement(
                        viz.Pose3D(placement.position, placement.orientation),
                        placement.size_meters,
                    )
                )
