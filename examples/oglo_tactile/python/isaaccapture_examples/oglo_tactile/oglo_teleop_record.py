#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end OGLO + MetaQuest data-collection demo.

Runs on the Linux host. It:
  1. Creates a TeleViz XR session (CloudXR runtime) shared with a DeviceIOSession.
  2. Launches the two ``oglo_tactile`` BLE plugins (one per hand) which push
     tactile/IMU tensors into the shared runtime.
  3. Records Quest hand + head pose and both gloves to one time-synced MCAP.
  4. Draws a live tactile heatmap into the operator's Quest view via a QuadLayer.

Prereqs (see README): CloudXR running + MetaQuest connected; gloves powered and
advertising; the project built with -DBUILD_PLUGIN_OGLO=ON -DBUILD_VIZ=ON and
the Python wheel installed. ``cupy`` is required for the headset overlay
(recording still works without it).

    python oglo_teleop_record.py --plugin-bin <path-to>/oglo_tactile_plugin
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import FrameType

import numpy as np

import isaaccapture.deviceio as deviceio
import isaaccapture.viz as viz

from .oglo_heatmap import NUM_TAXELS, Normalizer, TactileHeatmapRenderer

OVERLAY_W, OVERLAY_H = 1024, 512
COLLECTION_PREFIX = "oglo"

_stop = False


def _on_signal(signum: int, frame: FrameType | None) -> None:
    global _stop
    _stop = True


def _default_plugin_bin() -> str:
    # Only meaningful from a source checkout: examples/oglo_tactile/python/
    # isaaccapture_examples/oglo_tactile/ is five levels below the repo root. An
    # installed copy lands in site-packages, where this finds nothing and the
    # PATH fallback below takes over.
    here = Path(__file__).resolve()
    root = here.parents[5]
    for cand in (
        root / "build/src/plugins/oglo_tactile/oglo_tactile_plugin",
        root / "install/plugins/oglo_tactile/oglo_tactile_plugin",
    ):
        if cand.exists():
            return str(cand)
    return "oglo_tactile_plugin"  # rely on PATH


def _popen_plugin(plugin_bin: str, side: str) -> subprocess.Popen:
    cmd = [plugin_bin, "--side", side, f"--collection-prefix={COLLECTION_PREFIX}"]
    print(f"launching: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def _qrot(q_wxyz: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Rotate vector v=(x,y,z) by quaternion q=(w,x,y,z)."""
    w, x, y, z = q_wxyz
    u = np.array([x, y, z], dtype=float)
    vv = np.array(v, dtype=float)
    t = 2.0 * np.cross(u, vv)
    return vv + w * t + np.cross(u, t)


def _head_locked_placement(
    head_tracker: "deviceio.HeadTracker",
    session: "deviceio.DeviceIOSession",
    args: argparse.Namespace,
) -> "viz.QuadLayerPlacement | None":
    """Place the panel as a first-person HUD: `panel_dist` ahead of the head,
    offset `panel_right` to the right and `panel_drop` down, facing the operator
    (bottom-right of view so it never blocks the center). None if no head pose."""
    h = head_tracker.get_head(session)
    if h is None or h.pose is None:
        return None
    p = h.pose.position
    o = h.pose.orientation
    q = (o.w, o.x, o.y, o.z)  # schema quaternion is (x,y,z,w); Pose3D wants (w,x,y,z)
    pos = np.array([p.x, p.y, p.z], dtype=float)
    fwd = _qrot(q, (0.0, 0.0, -1.0))
    up = _qrot(q, (0.0, 1.0, 0.0))
    right = _qrot(q, (1.0, 0.0, 0.0))
    center = (
        pos + fwd * args.panel_dist + right * args.panel_right - up * args.panel_drop
    )
    pose = viz.Pose3D(
        position=(float(center[0]), float(center[1]), float(center[2])), orientation=q
    )
    return viz.QuadLayerPlacement(
        pose=pose, size_meters=(args.panel_w, args.panel_w * OVERLAY_H / OVERLAY_W)
    )


def _make_xr_session(trackers, xr_wait_s: float) -> viz.VizSession:
    cfg = viz.VizSessionConfig()
    cfg.mode = viz.DisplayMode.kXr
    cfg.app_name = "OgloTeleopRecord"
    cfg.xr_near_z = 0.05
    cfg.xr_far_z = 100.0
    # Wait this long for the MetaQuest to connect via CloudXR before giving up
    # (otherwise xrGetSystem fails immediately with FORM_FACTOR_UNAVAILABLE).
    cfg.xr_system_wait_seconds = int(xr_wait_s)
    # The XrInstance must advertise the extensions the trackers need.
    cfg.required_extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)
    return viz.VizSession.create(cfg)


def _make_overlay_layer(
    session: viz.VizSession, args: argparse.Namespace
) -> viz.QuadLayer:
    layer_cfg = viz.QuadLayerConfig()
    layer_cfg.name = "oglo_tactile_overlay"
    layer_cfg.resolution = viz.Resolution(OVERLAY_W, OVERLAY_H)
    layer_cfg.format = viz.PixelFormat.kRGBA8
    if not args.overlay_fullscreen:
        # Fixed panel in stage space, in front of and slightly below eye line.
        # Tune at demo time with --panel-y / --panel-z if placement looks off;
        # --overlay-fullscreen is the guaranteed-visible fallback.
        pose = viz.Pose3D(
            position=(0.0, args.panel_y, args.panel_z), orientation=(1.0, 0.0, 0.0, 0.0)
        )
        layer_cfg.placement = viz.QuadLayerPlacement(
            pose=pose, size_meters=(args.panel_w, args.panel_w * OVERLAY_H / OVERLAY_W)
        )
    return session.add_quad_layer(layer_cfg)


def _taxels(tracked) -> np.ndarray | None:
    if not tracked:
        return None
    t = tracked.taxels
    if not t or len(t) < NUM_TAXELS:
        return None
    return np.asarray(t, dtype=np.float32)[:NUM_TAXELS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OGLO + Quest data-collection demo")
    parser.add_argument(
        "--plugin-bin",
        default=_default_plugin_bin(),
        help="path to oglo_tactile_plugin",
    )
    parser.add_argument(
        "--mcap", default=None, help="output MCAP path (default: timestamped)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to record (0 = until Ctrl+C)",
    )
    parser.add_argument(
        "--no-plugins",
        action="store_true",
        help="do not launch plugins (launch them yourself)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="record only; skip the headset heatmap",
    )
    parser.add_argument(
        "--raw", action="store_true", help="raw ADC heatmap (no baseline subtraction)"
    )
    parser.add_argument(
        "--overlay-fullscreen",
        action="store_true",
        help="fullscreen overlay (facing-safe fallback)",
    )
    parser.add_argument(
        "--no-headlock",
        action="store_true",
        help="fixed panel instead of head-locked overlay",
    )
    parser.add_argument(
        "--panel-dist",
        type=float,
        default=0.7,
        help="head-locked panel distance ahead (m)",
    )
    parser.add_argument(
        "--panel-right",
        type=float,
        default=0.28,
        help="head-locked panel rightward offset (m)",
    )
    parser.add_argument(
        "--panel-drop",
        type=float,
        default=0.24,
        help="head-locked panel drop below eye line (m)",
    )
    parser.add_argument(
        "--panel-y",
        type=float,
        default=0.0,
        help="fixed-panel height (m, --no-headlock)",
    )
    parser.add_argument(
        "--panel-z",
        type=float,
        default=-1.0,
        help="fixed-panel distance (m, -Z forward, --no-headlock)",
    )
    parser.add_argument(
        "--panel-w", type=float, default=0.42, help="overlay panel width (m)"
    )
    parser.add_argument(
        "--xr-wait",
        type=float,
        default=60.0,
        help="seconds to wait for the MetaQuest to connect",
    )
    parser.add_argument(
        "--plugin-stagger",
        type=float,
        default=8.0,
        help="seconds between launching the two gloves (avoids BLE scan contention)",
    )
    args = parser.parse_args(argv)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    mcap_path = (
        args.mcap or f"oglo_teleop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mcap"
    )

    # --- trackers: Quest hand/head + both gloves --------------------------------
    hand = deviceio.HandTracker()
    head = deviceio.HeadTracker()
    oglo_left = deviceio.OgloTactileTracker(f"{COLLECTION_PREFIX}/left")
    oglo_right = deviceio.OgloTactileTracker(f"{COLLECTION_PREFIX}/right")
    trackers = [hand, head, oglo_left, oglo_right]

    # --- shared XR session (CloudXR) + DeviceIOSession with recording -----------
    print(
        f"Creating XR session (waiting up to {args.xr_wait:.0f}s for MetaQuest via CloudXR)...",
        flush=True,
    )
    print("→ Now press CONNECT in the Quest web client if you haven't.", flush=True)
    viz_session = _make_xr_session(trackers, args.xr_wait)
    handles = deviceio.OpenXRSessionHandles(*viz_session.get_oxr_handles())

    # Launch the gloves staggered: the two plugins share one BLE adapter, and a
    # simultaneous BlueZ StartDiscovery from both makes one fail (DBus NoReply).
    # Start the right glove now; the left is launched after --plugin-stagger
    # seconds from inside the render loop (so the headset keeps rendering).
    plugins = []
    left_launched = args.no_plugins
    if not args.no_plugins:
        plugins.append(_popen_plugin(args.plugin_bin, "right"))

    # Overlay setup (optional / requires cupy for GPU upload).
    overlay = None
    cp = None
    if not args.no_overlay:
        try:
            import cupy as _cp

            _cp.asarray(
                np.zeros((1, 1, 4), dtype=np.uint8)
            )  # verify cupy actually works
            cp = _cp
            overlay = _make_overlay_layer(viz_session, args)
        except Exception as e:
            print(
                f"WARNING: headset overlay disabled — cupy unusable ({e}). Recording continues.",
                flush=True,
            )
            print(
                "  Fix: install the cupy build matching your CUDA driver "
                "(e.g. `pip install cupy-cuda12x`, or `cupy-cuda11x` for CUDA 11).",
                flush=True,
            )
            overlay = None
            cp = None

    renderer = TactileHeatmapRenderer(width=OVERLAY_W, height=OVERLAY_H)
    norm_l = Normalizer(raw=args.raw)
    norm_r = Normalizer(raw=args.raw)

    recording = deviceio.McapRecordingConfig(
        mcap_path,
        [
            (hand, "hands"),
            (head, "head"),
            (oglo_left, "oglo_left"),
            (oglo_right, "oglo_right"),
        ],
    )

    print(f"Recording → {mcap_path}", flush=True)
    print("Keep hands relaxed for ~1s so the tactile baseline can tare.", flush=True)

    try:
        with deviceio.DeviceIOSession.run(trackers, handles, recording) as session:
            start = time.time()
            left_due = start + args.plugin_stagger
            frames = 0
            while not _stop:
                # Staggered launch of the left glove (avoids BLE scan contention).
                if not left_launched and time.time() >= left_due:
                    plugins.append(_popen_plugin(args.plugin_bin, "left"))
                    left_launched = True

                session.update()

                if overlay is not None:
                    try:
                        if not args.no_headlock:
                            placement = _head_locked_placement(head, session, args)
                            if placement is not None:
                                overlay.set_placement(placement)
                        left = _taxels(oglo_left.get_glove_data(session))
                        right = _taxels(oglo_right.get_glove_data(session))
                        frame = renderer.render(
                            norm_l.normalize(left) if left is not None else None,
                            norm_r.normalize(right) if right is not None else None,
                        )
                        overlay.submit(
                            cp.asarray(frame)
                        )  # host->device for the GPU compositor
                    except Exception as e:
                        # Overlay is auxiliary — never let it kill the recording.
                        print(
                            f"WARNING: overlay error ({e}); disabling overlay, recording continues.",
                            flush=True,
                        )
                        overlay = None

                viz_session.render()

                frames += 1
                if frames % 90 == 0:
                    print(f"[{time.time() - start:5.1f}s] frames={frames}", flush=True)

                if viz_session.should_close():
                    break
                if args.duration and (time.time() - start) >= args.duration:
                    break
    finally:
        for p in plugins:
            p.terminate()
        for p in plugins:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        viz_session.destroy()

    print(f"Done. Saved {mcap_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
