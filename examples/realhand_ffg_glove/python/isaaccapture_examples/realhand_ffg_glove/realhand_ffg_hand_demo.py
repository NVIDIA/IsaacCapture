# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display standard FFG human skeletons with PC calibration."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

_EDGES = [(1, 0), (1, 2), (2, 3), (3, 4), (4, 5)]
for _start in (6, 11, 16, 21):
    _EDGES.append((1, _start))
    _EDGES.extend((index, index + 1) for index in range(_start, _start + 4))


def display_positions(points, side):
    points = np.asarray(points, dtype=float)
    if points.shape != (26, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Expected 26 finite OpenXR joint positions")
    # Rotate the display palm-up about the finger axis; never reflect or change calibration data.
    result = points[:, (0, 2, 1)] * (-1.0, -1.0, -1.0)
    result += (-0.15 if side == "left" else 0.15, -0.08, 0.20)
    return result


def bone_segments(points):
    points = np.asarray(points, dtype=float)
    starts = points[[first for first, _ in _EDGES]]
    ends = points[[second for _, second in _EDGES]]
    vectors = ends - starts
    return (starts + ends) * 0.5, vectors, np.linalg.norm(vectors, axis=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-python", type=Path)
    parser.add_argument("--capture-package", type=Path)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument(
        "--ports",
        help="Comma-separated verified glove ports; omit for identified-device discovery",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path.home() / ".config/realhand/ffg_human_demo.yml",
    )
    parser.add_argument(
        "--test-frames",
        type=Path,
        help="Replay C++ skeleton fixtures; no glove or runtime is used",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--no-launch-cloudxr-runtime", action="store_true")
    args = parser.parse_args(argv)
    if args.test_frames is None and any(
        value is None
        for value in (args.capture_python, args.capture_package, args.plugin_path)
    ):
        parser.error(
            "Live mode requires --capture-python, --capture-package, and --plugin-path"
        )
    if args.capture_python and not args.capture_python.is_file():
        parser.error(f"Capture Python interpreter not found: {args.capture_python}")
    return args


def worker_command(args, ipc_dir):
    command = [
        str(args.capture_python),
        "-m",
        "isaaccapture_examples.realhand_ffg_glove.realhand_ffg_hand_demo_bridge",
        "--ipc-dir",
        str(ipc_dir),
        "--plugin-path",
        str(args.plugin_path),
        "--calibration",
        str(args.calibration),
    ]
    # Omitted ports must reach native discovery, not silently select a single glove.
    if args.ports:
        command.extend(("--ports", args.ports))
    if args.no_launch_cloudxr_runtime:
        command.append("--no-launch-cloudxr-runtime")
    return command


def main():
    args = parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1440,
            "height": 900,
            "renderer": "RaytracedLighting",
            "fast_shutdown": True,
        }
    )
    process = None
    window = None
    menu_items = None
    exit_code = 0
    try:
        import omni.ui as ui
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.menu.utils import (
            MenuItemDescription,
            add_menu_items,
            remove_menu_items,
        )

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        dome = UsdLux.DomeLight.Define(stage, "/World/Light")
        dome.CreateIntensityAttr(450.0)
        key = UsdLux.DistantLight.Define(stage, "/World/Key")
        key.CreateIntensityAttr(900.0)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set((25, -30, -20))
        floor = UsdGeom.Cube.Define(stage, "/World/Workbench")
        floor.CreateSizeAttr(1.0)
        floor.CreateDisplayColorAttr([(0.14, 0.16, 0.18)])
        xf = UsdGeom.Xformable(floor)
        xf.AddTranslateOp().Set((0, 0, -0.02))
        xf.AddScaleOp().Set((0.78, 0.56, 0.03))
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(28.0)
        camera.CreateClippingRangeAttr((0.01, 100.0))
        UsdGeom.Xformable(camera).AddTransformOp().Set(
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(0, -0.65, 0.80), Gf.Vec3d(0, 0.01, 0.20), Gf.Vec3d(0, 0, 1)
            )
            .GetInverse()
        )
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(str(camera.GetPath()))
        skeletons = {}
        for side, color in (
            ("left", (0.14, 0.58, 0.85)),
            ("right", (0.90, 0.35, 0.27)),
        ):
            path = f"/World/{side.title()}Hand"
            root = UsdGeom.Xform.Define(stage, path)
            joints = []
            for index in range(26):
                joint = UsdGeom.Sphere.Define(stage, f"{path}/Joint_{index}")
                joint.CreateRadiusAttr(0.006 if index not in (0, 1) else 0.010)
                joint.CreateDisplayColorAttr([color])
                joints.append(UsdGeom.Xformable(joint).AddTranslateOp())
            # Solid bones must render in live RTX viewports, not only curve-enabled captures.
            bones = []
            for index in range(len(_EDGES)):
                bone = UsdGeom.Cylinder.Define(stage, f"{path}/Bone_{index}")
                bone.CreateAxisAttr(UsdGeom.Tokens.z)
                bone.CreateRadiusAttr(0.003)
                bone.CreateHeightAttr(1.0)
                bone.CreateDisplayColorAttr([(0.72, 0.77, 0.80)])
                transform = UsdGeom.Xformable(bone)
                bones.append(
                    (
                        bone,
                        transform.AddTranslateOp(),
                        transform.AddOrientOp(),
                        transform.AddScaleOp(),
                    )
                )
            root.MakeInvisible()
            skeletons[side] = (root, joints, bones)

        fixture = None
        if args.test_frames:
            import yaml

            fixture = yaml.safe_load(args.test_frames.read_text(encoding="utf-8"))
        window = ui.Window("RealHand FFG Calibration", width=390, height=580)
        with window.frame, ui.ScrollingFrame():
            with ui.VStack(spacing=10, margin=14):
                ui.Label("RealHand FFG Glove", height=26)
                connection = ui.Label("Starting DeviceIO...", height=46, word_wrap=True)
                ui.Separator(height=2)
                side_choice = ui.ComboBox(2, "Left", "Right", "Both", height=28)
                calibration_button = ui.Button(
                    "Start Calibration",
                    height=32,
                    clicked_fn=lambda: send(
                        {
                            "action": "begin",
                            "sides": (("left",), ("right",), ("left", "right"))[
                                side_choice.model.get_item_value_model().as_int
                            ],
                        }
                    ),
                )
                step_label = ui.Label("", height=24)
                instruction = ui.Label("", height=92, word_wrap=True)
                countdown = ui.Label("", height=28)
                capture_button = ui.Button(
                    "Capture Pose",
                    height=32,
                    clicked_fn=lambda: send({"action": "capture"}),
                )
                ui.Button(
                    "Cancel Capture",
                    height=28,
                    clicked_fn=lambda: send({"action": "cancel"}),
                )
                message = ui.Label("", height=110, word_wrap=True)
        window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

        def show_calibration():
            window.visible = True
            window.focus()

        menu_items = [
            MenuItemDescription(
                name="RealHand FFG Calibration", onclick_fn=show_calibration
            )
        ]
        add_menu_items(menu_items, "Window")

        rgb = None
        if args.screenshot:
            import omni.replicator.core as rep

            product = rep.create.render_product(str(camera.GetPath()), (1280, 800))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach(product)
        with tempfile.TemporaryDirectory(prefix="ffg-demo-") as temporary:
            ipc = Path(temporary)
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
                channel.bind(str(ipc / "frames.sock"))
                channel.setblocking(False)

                def send(command):
                    if fixture is not None:
                        message.text = "Fixture preview only. Launch live mode to calibrate a glove."
                        return
                    try:
                        channel.sendto(
                            json.dumps(command).encode(), str(ipc / "commands.sock")
                        )
                    except OSError:
                        message.text = (
                            "DeviceIO is not ready. Check the terminal output."
                        )

                if fixture is None:
                    environment = os.environ.copy()
                    environment["PYTHONPATH"] = os.pathsep.join(
                        (
                            str(args.capture_package),
                            str(Path(__file__).resolve().parents[2]),
                        )
                    )
                    # Isaac Sim's bundled shared libraries must not leak into the standalone OpenXR worker.
                    for key in (
                        "LD_LIBRARY_PATH",
                        "PYTHONHOME",
                        "CARB_APP_PATH",
                        "ISAAC_PATH",
                    ):
                        environment.pop(key, None)
                    process = subprocess.Popen(
                        worker_command(args, temporary), env=environment
                    )
                latest = None
                received = 0.0
                frame = 0
                started = time.monotonic()
                while app.is_running() and (not args.frames or frame < args.frames):
                    now = time.monotonic()
                    if fixture is not None:
                        pose_index = int((now - started) / 2.0) % 3
                        latest = {
                            "connected": {"left": True, "right": True},
                            "hands": {
                                side: [
                                    joint["position"]
                                    for joint in fixture[side][pose_index]["joints"]
                                ]
                                for side in ("left", "right")
                            },
                            "calibration": {
                                "state": "idle",
                                "message": "C++ skeleton fixture preview; no live glove data.",
                                "instruction": "",
                            },
                        }
                        received = now
                    else:
                        while True:
                            try:
                                latest = json.loads(channel.recv(65536))
                                received = now
                            except BlockingIOError:
                                break
                    fresh = latest is not None and now - received < 0.5
                    for side, (root, joints, bones) in skeletons.items():
                        points = latest["hands"].get(side) if fresh else None
                        if points is None:
                            root.MakeInvisible()
                            continue
                        points = display_positions(points, side)
                        root.MakeVisible()
                        for index, joint in enumerate(joints):
                            joint.Set(Gf.Vec3d(*points[index]))
                        centers, vectors, lengths = bone_segments(points)
                        for (
                            bone,
                            translate,
                            orient,
                            scale,
                        ), center, vector, length in zip(
                            bones, centers, vectors, lengths
                        ):
                            if length < 1e-6:
                                bone.MakeInvisible()
                                continue
                            bone.MakeVisible()
                            translate.Set(Gf.Vec3d(*center))
                            rotation = Gf.Rotation(
                                Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(vector / length))
                            )
                            orient.Set(Gf.Quatf(rotation.GetQuat()))
                            scale.Set(Gf.Vec3f(1, 1, length))
                    if fresh:
                        connection.text = " | ".join(
                            f"{side.title()}: {'connected' if latest['connected'][side] else 'not found'}"
                            for side in ("left", "right")
                        )
                        state = latest["calibration"]
                        step_label.text = (
                            f"Calibration: {state['step']} / {state['total']}"
                            if state["state"] in ("ready", "preparing", "capturing")
                            else f"Calibration: {state['state'].title()}"
                        )
                        instruction.text = state.get("instruction", "")
                        countdown.text = (
                            f"PREPARE: {state['remaining']:.1f} s"
                            if state["state"] == "preparing"
                            else f"HOLD: {state['remaining']:.1f} s"
                            if state["state"] == "capturing"
                            else ""
                        )
                        capture_button.enabled = state["state"] == "ready"
                        calibration_button.enabled = state["state"] not in (
                            "ready",
                            "preparing",
                            "capturing",
                        )
                        side_choice.enabled = state["state"] not in (
                            "ready",
                            "preparing",
                            "capturing",
                        )
                        errors = [
                            f"{side.title()}: {error}"
                            for side, error in latest.get("device_errors", {}).items()
                            if error
                        ]
                        message.text = "\n".join([state.get("message", ""), *errors])
                    elif process is not None and process.poll() is not None:
                        connection.text = (
                            f"DeviceIO exited (code {process.returncode})."
                        )
                        message.text = "Check the terminal for runtime, USB permission, or calibration errors."
                    elif now - started > 3.0:
                        connection.text = "Waiting for DeviceIO..."
                    app.update()
                    frame += 1
                    time.sleep(max(0.0, 1.0 / 60.0 - (time.monotonic() - now)))
                if rgb is not None:
                    from PIL import Image

                    rep.orchestrator.step(rt_subframes=2)
                    image = rgb.get_data()
                    if not isinstance(image, np.ndarray) or image.size == 0:
                        raise RuntimeError("Isaac Sim produced no screenshot pixels")
                    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(args.screenshot)
                    print(f"Screenshot saved: {args.screenshot}")
    except Exception:
        # Fast shutdown otherwise exits before Python can report a startup error.
        traceback.print_exc()
        exit_code = 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if window is not None:
            if menu_items is not None:
                remove_menu_items(menu_items, "Window")
            window.destroy()
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
