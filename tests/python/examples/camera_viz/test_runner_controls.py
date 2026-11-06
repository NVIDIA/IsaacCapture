# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VizRunner <-> controls integration.

These cover the seam the per-module tests structurally cannot: VizRunner
copies the lists it is handed, so a controls-side shape or lock-mode swap
only lands if the runner resolves it live. Without this, switching to
cylinder kept feeding the hidden quad and the feed went black.
"""

from __future__ import annotations

from isaaccapture_examples.camera_viz.pipeline.runner import VizRunner


class FakeLayer:
    def __init__(self, name) -> None:
        self.name = name
        self.submitted = 0

    def submit(self, *args, **kwargs) -> None:
        self.submitted += 1


class FakeSpec:
    name = "cam"


class FakeSource:
    spec = FakeSpec()

    def latest(self):
        return None

    def start(self) -> None: ...

    def stop(self) -> None: ...


class FakeSession:
    def is_xr_mode(self) -> bool:
        return True


class FakeControls:
    """Stands in for ControllerControls' resolver surface."""

    def __init__(self, layer, strategy=None) -> None:
        self.layer = layer
        self._strategy = strategy

    def active_layer(self, index):
        return self.layer

    def strategy(self, index):
        return self._strategy

    def force_mono(self, index) -> bool:
        return False


def _runner(controls=None, layers=None, strategies=None):
    layers = layers or [FakeLayer("quad")]
    return VizRunner(
        FakeSession(), [FakeSource()], layers, strategies, controls=controls
    )


def test_runner_follows_a_shape_switch():
    quad, cylinder = FakeLayer("quad"), FakeLayer("cylinder")
    controls = FakeControls(quad)
    runner = _runner(controls, [quad])

    assert runner._active_layer(0) is quad
    controls.layer = cylinder  # what pressing X does
    assert runner._active_layer(0) is cylinder


def test_runner_follows_a_lock_mode_swap():
    layer = FakeLayer("quad")
    first, second = object(), object()
    controls = FakeControls(layer, first)
    runner = _runner(controls, [layer], [first])

    assert runner._active_strategy(0) is first
    controls._strategy = second  # what pressing A does
    assert runner._active_strategy(0) is second


def test_runner_copies_stay_authoritative_without_controls():
    """No controls: the runner's own lists are the source of truth."""
    layer = FakeLayer("quad")
    strategy = object()
    runner = _runner(None, [layer], [strategy])
    assert runner._active_layer(0) is layer
    assert runner._active_strategy(0) is strategy


def test_mutating_the_caller_list_does_not_reach_the_runner():
    """Pins down why the resolver exists: VizRunner copies, so the old
    shared-list approach could not work."""
    layer, other = FakeLayer("quad"), FakeLayer("cylinder")
    caller_layers = [layer]
    runner = _runner(None, caller_layers)
    caller_layers[0] = other
    assert runner._active_layer(0) is layer


# ── The panel must not sit between the camera and the layer ───────────


def test_the_panel_is_drawn_off_the_submit_thread():
    """A terminal that stops draining (paused ssh, tmux scrollback, ^S) fills
    the pty buffer in ~25 paints and blocks the writer. On the submit thread
    that stops the video feed, so the panel gets its own thread."""
    import inspect

    from isaaccapture_examples.camera_viz.pipeline.runner import VizRunner

    submit_src = inspect.getsource(VizRunner._submit_loop_inner)
    assert "_print_stats" not in submit_src
    assert "_print_stats" in inspect.getsource(VizRunner._stats_loop)


def test_a_broken_panel_write_does_not_kill_the_stats_thread(monkeypatch):
    """stderr can go away mid-run (the terminal closes, the pipe breaks). The
    panel is decoration; losing it must not take the stats thread with it."""
    import threading

    from isaaccapture_examples.camera_viz.pipeline import runner as runner_mod

    monkeypatch.setattr(runner_mod, "LIVE_STATS_PERIOD_S", 0.005)

    calls = []
    barrier = threading.Event()

    class ExplodingDashboard:
        live = True

        def show(self, snapshot):
            calls.append(1)
            if len(calls) >= 3:
                barrier.set()
            raise OSError("terminal went away")

    runner = _runner()
    runner._dashboard = ExplodingDashboard()
    thread = threading.Thread(target=runner._stats_loop, daemon=True)
    thread.start()
    # It raised on every call and still came back for the next period.
    assert barrier.wait(timeout=3.0), f"stats thread stopped after {len(calls)} calls"
    runner._stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
