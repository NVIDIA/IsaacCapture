#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""camera_viz — camera-feed visualizer for Isaac Teleop.

Reads the unified pipeline YAML (cameras + streaming + display) and
runs the receiver side: either opens the configured cameras directly
(``source: local``) or listens for matching RTP H.264 streams
(``source: rtp``) from a ``camera_streamer.py`` instance on the robot.

The same YAML file drives ``camera_streamer.py``, so both ends of an
RTP-mode deployment share one config.

Usage:
    python camera_viz.py configs/v4l2.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import yaml

from isaacteleop.cloudxr import CloudXRLauncher

import cloudxr_env
import config
import display
from controls import (
    ControllerControls,
    ControlTarget,
    controls_config_from_yaml,
    make_hud,
)
from dashboard import Dashboard
from pipeline import VizRunner
from sources import resolve_video_paths, set_notify_sink, set_verbose


def _parse_args(argv: Optional[list[str]]):
    parser = argparse.ArgumentParser(description="Televiz camera_viz — display side")
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument(
        "--mode",
        choices=("window", "xr"),
        default=None,
        help="Override display.mode from the config "
        "(default: the config's value, or xr when the config omits it).",
    )
    parser.add_argument(
        "--xr-wait",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Override display.xr.system_wait_seconds: how long to wait for the "
        "headset to connect. Negative waits forever (the default); 0 fails fast. "
        "Ctrl-C is not delivered until the wait ends.",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    return parser.parse_args(argv)


def _build_display(session, entries, switch_shapes: bool):
    """Parallel arrays, one entry per camera stream.

    They stay parallel because the controls mutate them in place: a shape
    switch rewrites ``layers[i]`` and a lock-mode change ``strategies[i]``,
    and the runner reads the same lists.
    """
    sources, layers, strategies, shape_layers = [], [], [], []
    for entry in entries:
        per_shape = display.add_layers(session, entry, switch_shapes)
        sources.append(entry.source)
        shape_layers.append(per_shape)
        layers.append(per_shape[entry.shape])
        # Lock-mode strategies reposition quads AND cylinders (the runner
        # adapts the pose to the cylinder's head-anchored center). An
        # equirect sphere is centred on the operator with nothing to
        # re-snap, so the runner skips it by layer type -- the strategy is
        # still kept here, because switching away from equirect needs it.
        strategies.append(entry.placement)
    return sources, layers, strategies, shape_layers


def _build_controls(
    session, entries, layers, shape_layers, strategies, controls_cfg, dashboard
):
    """Wire the XR controller bindings to the layers they drive.

    ``strategies`` is handed over as-is rather than copied: the controls swap
    entries in place on a lock-mode change and the runner reads the same list.
    """
    targets = [
        ControlTarget(
            name=e.source.spec.name,
            layer=layer,
            shape=e.shape,
            stereo=e.stereo,
            plane_distance_cm=e.stereo_plane_distance_cm,
            lock_mode=e.lock_mode,
            placement_config=e.placement_config,
            shape_layers=per_shape,
            cylinder_radius_m=e.cylinder_radius_m,
            cylinder_angle_deg=e.cylinder_angle_deg,
            equirect_yaw_deg=e.equirect_yaw_deg,
        )
        for e, layer, per_shape in zip(entries, layers, shape_layers)
    ]
    # Added last so it composites over the feeds (insertion order is blend
    # order for runtime-composited layers).
    hud = make_hud(session, controls_cfg.hud)
    return ControllerControls(
        session,
        targets,
        strategies,
        controls_cfg,
        hud=hud,
        # The panel owns stderr while it is live; a second writer there lands
        # inside it and leaves every repaint misaligned.
        log_to_stderr=not dashboard.live,
    )


def _render_extent_note(session, entries) -> List[str]:
    """The per-eye extent the runtime asked us to render at.

    This is the number that says whether a foveation
    NV_CXR_RUNTIME_FOVEATION_UNWARPED_WIDTH override reached the app: the
    runtime hands it back through xrEnumerateViewConfigurationViews. Shown
    next to the widest source so the ratio is readable -- a feed wider than
    the extent is being minified before it is ever encoded, which is what
    aliases.
    """
    get = getattr(session, "get_recommended_resolution", None)
    if get is None:  # older isaacteleop wheel
        return []
    try:
        resolution = get()
        width, height = resolution.width, resolution.height
    except Exception as exc:  # noqa: BLE001 -- a note must not be fatal...
        # ...but it must not vanish either: swallowing this silently is why
        # the line went missing when get_recommended_resolution turned out to
        # return a Resolution rather than a tuple.
        print(f"camera_viz: render extent unavailable: {exc!r}", file=sys.stderr)
        return []
    note = f"render extent {width}x{height} per eye"
    widest = max((e.source.spec.width for e in entries), default=0)
    if widest:
        note += f"; widest source {widest} px ({width / widest:.2f}x)"
    # Printed here as well as returned: this is the one number that says
    # whether a foveation override reached the app, and the status panel
    # shows it dimmed on its last line, where it is easy to miss and gone
    # entirely if the run dies later in startup.
    print(f"camera_viz: {note}", file=sys.stderr, flush=True)
    return [note]


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"camera_viz: {args.config} must be a YAML mapping at the top level, "
            f"got {type(cfg).__name__}"
        )

    # Top-level ``verbose:`` enables per-source periodic breadcrumbs.
    set_verbose(bool(cfg.get("verbose", False)))
    resolve_video_paths(cfg, args.config.parent)

    source_mode = cfg.get("source", "local").lower()
    if source_mode not in ("local", "rtp"):
        raise ValueError(f"camera_viz: source must be local|rtp, got {source_mode!r}")

    effective_mode = (args.mode or cfg.get("display", {}).get("mode", "xr")).lower()
    config.check_shapes_are_displayable(cfg, effective_mode)

    # CloudXR runtime settings (display.cloudxr) go through a generated
    # --cloudxr-env-config file rather than os.environ: an env file is the one
    # tier that outranks a stale `source ~/.cloudxr/run/cloudxr.env` in the
    # shell, which otherwise beats both a setdefault here and the launcher's
    # own --cloudxr-device-profile. See cloudxr_env for the rest of the
    # reasoning. No effect under --no-launch-cloudxr-runtime: that runtime
    # already started with whatever environment it was given.
    cloudxr_settings = {}
    if effective_mode == "xr":
        cloudxr_settings = cloudxr_env.env_from_yaml(cfg.get("display", {}))

    # In XR mode, launch the in-process CloudXR runtime (+ WSS proxy for
    # headset clients) before creating the session — VizSession's OpenXR
    # instance needs XR_RUNTIME_JSON + a running service, both of which the
    # launcher provides. --no-launch-cloudxr-runtime skips this when a
    # runtime is already up (e.g. after sourcing ~/.cloudxr/run/cloudxr.env).
    # Window mode never launches a runtime.
    # Entered manually (not ``with``) so the unclean-stop path below can
    # SKIP the teardown: stopping the runtime while a worker thread is
    # still inside session.render() would rip the OpenXR service out from
    # under a live xrWaitFrame — the same hazard the skip-destroy
    # mitigation exists for. The launcher registers an atexit stop, which
    # fires once the stuck (non-daemon) thread finally exits.
    settings_stack = contextlib.ExitStack()
    if effective_mode == "xr" and not args.launch_cloudxr_runtime:
        # Nothing is launched, so nothing carries these to the runtime that is
        # already up -- it started with whatever environment it was given.
        extra = set(cloudxr_settings) - set(cloudxr_env.DEFAULT_ENV)
        if extra or cloudxr_settings != cloudxr_env.DEFAULT_ENV:
            print(
                "camera_viz: warning: display.cloudxr is ignored under "
                "--no-launch-cloudxr-runtime; set those variables in the "
                "environment of the runtime process instead",
                file=sys.stderr,
                flush=True,
            )
    if effective_mode == "xr" and args.launch_cloudxr_runtime:
        # An explicit --cloudxr-env-config is the operator overriding the
        # config file; do not overwrite it with the generated one.
        if getattr(args, "cloudxr_env_config", None) is None:
            work_dir = Path(settings_stack.enter_context(tempfile.TemporaryDirectory()))
            args.cloudxr_env_config = str(
                cloudxr_env.write_env_file(cloudxr_settings, work_dir / "cloudxr.env")
            )
    # A null context, not the launcher's NoopContext: that landed after the
    # isaacteleop this sample pins, and importing it would break the sample
    # against its own wheel. Swap it in (and drop the None guard below) once
    # the pin catches up.
    launch_ctx = (
        CloudXRLauncher.launch_context(args)
        if effective_mode == "xr"
        else contextlib.nullcontext(None)
    )
    launcher = launch_ctx.__enter__()
    stop_launcher = True
    try:
        controls_cfg = controls_config_from_yaml(cfg.get("display", {}))
        # Window mode has no controllers, so don't ask for their extensions.
        want_controls = controls_cfg.enabled and effective_mode == "xr"
        if effective_mode == "xr":
            # create() blocks here with the GIL held, so say so before it does:
            # an unexplained stall plus a dead Ctrl-C reads as a hang.
            xr_cfg = cfg.get("display", {}).get("xr", {})
            wait = (
                args.xr_wait
                if args.xr_wait is not None
                else int(xr_cfg.get("system_wait_seconds", display.WAIT_FOR_HEADSET))
            )
            if wait != 0:
                how = "indefinitely" if wait < 0 else f"up to {wait}s"
                print(
                    f"camera_viz: waiting {how} for the headset to connect "
                    "(Ctrl-C only lands once it does; kill the process to abort)",
                    flush=True,
                )
        session = display.make_session(
            cfg,
            mode_override=args.mode,
            required_extensions=(
                ControllerControls.required_extensions() if want_controls else None
            ),
            xr_wait_override=args.xr_wait,
        )
        is_xr = session.is_xr_mode()

        if source_mode == "local":
            entries = config.build_local_entries(cfg, is_xr)
        else:
            entries = config.build_rtp_entries(cfg, is_xr)

        # Shape switching needs every shape resident, and the shaped layers
        # are XR-only, so it is off outside XR regardless of the config.
        switch_shapes = want_controls and is_xr and controls_cfg.shape_switching

        sources, layers, strategies, shape_layers = _build_display(
            session, entries, switch_shapes
        )

        cameras = f"{len(sources)} camera" + ("s" if len(sources) != 1 else "")
        header = f"{effective_mode} · {source_mode} · {cameras}"
        notes = _render_extent_note(session, entries) if is_xr else []
        if switch_shapes:
            extra = sum(
                display.estimate_layer_bytes(e, shape)
                for e in entries
                for shape in config.VALID_SHAPES
                if shape != e.shape
            )
            notes.append(
                f"shape switching on — {len(config.VALID_SHAPES) - 1} extra layer(s) per "
                f"camera, about {extra / (1024 * 1024):.0f} MiB additional VRAM"
            )

        # Built before the controls and the sources' notifications are
        # rerouted: while it is live it owns stderr, and a second writer on
        # that stream lands inside the panel (see Dashboard.note).
        dashboard = Dashboard()
        if dashboard.live:
            set_notify_sink(dashboard.note)
        else:
            # Nothing is redrawing, so the header would never be seen.
            print(f"camera_viz: {header}", flush=True)
            for note in notes:
                print(f"camera_viz: {note}", file=sys.stderr, flush=True)

        controls = (
            _build_controls(
                session,
                entries,
                layers,
                shape_layers,
                strategies,
                controls_cfg,
                dashboard,
            )
            if want_controls and is_xr
            else None
        )

        runner = VizRunner(
            session,
            sources,
            layers,
            strategies,
            controls=controls,
            dashboard=dashboard,
            header=header,
            notes=notes,
        )

        def _on_signal(signum, frame):
            print(f"camera_viz: stopping (signal {signum})...", flush=True)
            runner.stop()

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        # Entered manually rather than with ``with``: teardown is ordered and
        # conditional. The controls' trackers borrow the XrInstance / XrSession
        # that VizSession owns, so they must detach AFTER the render thread
        # stops polling them but BEFORE destroy() frees those handles — and
        # neither may happen at all if a worker thread is still alive.
        if controls is not None:
            controls.__enter__()
        runner.start()
        try:
            runner.wait(
                health_check=launcher.health_check if launcher is not None else None
            )
        finally:
            # Skip destroy() when a worker thread is still alive — it may be
            # inside session.render() and destroying under it would UAF on the
            # Vulkan / CUDA handles. Non-daemon thread keeps the process alive;
            # OS reaps at exit. Leave the CloudXR runtime up too (see the
            # launch_ctx comment above).
            clean = runner.stop()
            # Unhook first: a source thread that outlives stop() would
            # otherwise notify into a panel that has closed.
            set_notify_sink(None)
            dashboard.close()
            if clean:
                if controls is not None:
                    controls.__exit__(None, None, None)
                session.destroy()
            else:
                stop_launcher = False
                print(
                    "camera_viz: worker thread did not exit; leaving VizSession, "
                    "the controller session and the CloudXR runtime alive to "
                    "avoid use-after-free. Process will keep running until the "
                    "stuck thread completes.",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if stop_launcher:
            launch_ctx.__exit__(None, None, None)
        # Held open until here: the launcher reads the generated env file at
        # launch, and keeping it on disk for the run makes the settings
        # actually in force inspectable while the runtime is up.
        settings_stack.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
