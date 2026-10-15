# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe native Franka recording without replacing its retargeting pipeline.

Robot actions come exclusively through the native WebXR/CloudXR device.
Only read-only feedback, visual presentation and orderly shutdown hooks are added.
"""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--lab-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--feedback-port", type=int, default=8765)
options, remaining = parser.parse_known_args()
ROOT = options.output.resolve()
REFERENCE = options.lab_root.resolve() / "scripts/tools/record_demos.py"
if not REFERENCE.is_file():
    raise FileNotFoundError(REFERENCE)
sys.argv = [str(REFERENCE), *remaining]
spec = importlib.util.spec_from_file_location("synthetic_xr_record_demos", REFERENCE)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)

state = {"phase": "initializing", "steps": 0, "success_steps": 0}
snapshot = json.dumps(state).encode()
last_publish = 0.0
environment = None
teleop = None
home = None
clutch_node = None
run_dir = ROOT
run_dir.mkdir(parents=True, exist_ok=False)
trace = (run_dir / "sim-feedback.jsonl").open("x", buffering=1)
state["run_dir"] = str(run_dir)


def values(data):
    data = data.torch if hasattr(data, "torch") else data
    return data.detach().cpu().tolist()


def publish(force=False):
    global snapshot, last_publish
    now = time.monotonic()
    if not force and now - last_publish < 0.1:
        return
    if environment is None:
        return
    last_publish = now
    env = environment
    robot = env.scene["robot"]
    ee = env.scene["ee_frame"].data
    state.update(
        {
            "timestamp": time.time(),
            "monotonic": now,
            "simulation_dt": float(env.step_dt),
            "joint_names": list(robot.joint_names),
            "joints": values(robot.data.joint_pos),
            "base_position_world": values(robot.data.root_pos_w),
            "base_quaternion_world": values(robot.data.root_quat_w),
            "ee_positions_base": values(ee.target_pos_source),
            "ee_quaternions_base": values(ee.target_quat_source),
            "ee_positions_world": values(ee.target_pos_w),
            "ee_quaternions_world": values(ee.target_quat_w),
            "body_names": list(robot.body_names),
            "body_positions_world": values(robot.data.body_pos_w),
            "body_quaternions_world": values(robot.data.body_quat_w),
            "cubes": {
                name: {
                    "position_world": values(env.scene[name].data.root_pos_w),
                    "quaternion_world": values(env.scene[name].data.root_quat_w),
                    "linear_velocity_world": values(
                        env.scene[name].data.root_lin_vel_w
                    ),
                }
                for name in ("cube_1", "cube_2", "cube_3")
            },
            "exported_successful_episodes": env.recorder_manager.exported_successful_episode_count,
            "configured_home_base_T_ee": home.tolist(),
            "gripper_joint_names": env.cfg.gripper_joint_names,
            "gripper_open_val": env.cfg.gripper_open_val,
        }
    )
    if teleop is not None:
        state["world_T_anchor"] = teleop._anchor_manager.get_world_matrix().tolist()
        target = (
            teleop._get_target_frame_T_world()
            if teleop._cfg.target_frame_prim_path is not None
            else None
        )
        state["base_T_world"] = None if target is None else np.asarray(target).tolist()
        ctrl = teleop.last_control_events
        state["control"] = {
            "is_active": ctrl.is_active,
            "should_reset": ctrl.should_reset,
        }
    snapshot = json.dumps(state, allow_nan=False).encode()
    trace.write(snapshot.decode() + "\n")
    temporary = ROOT / "sim-state.tmp"
    temporary.write_bytes(snapshot)
    temporary.replace(ROOT / "sim-state.json")


class Feedback(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/state":
            self.send_error(404)
            return
        body = snapshot
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


original_create = native.create_environment
original_config = native.create_environment_config


def create_environment_config(*args, **kwargs):
    cfg, success, use_capture = original_config(*args, **kwargs)
    if os.environ.get("SYNTHETIC_XR_VISIBLE") == "1":
        from isaaclab_visualizers.kit import KitVisualizerCfg

        cfg.sim.visualizer_cfgs = [
            KitVisualizerCfg(
                eye=(1.35, -1.35, 1.10),
                lookat=(0.43, 0.0, 0.20),
                focal_length=24.0,
                enable_live_plots=False,
                enable_markers=False,
            )
        ]
        cfg.scene.ee_frame.debug_vis = False
    return cfg, success, use_capture


def create_environment(cfg):
    global environment
    env = original_create(cfg)
    environment = env.unwrapped
    if os.environ.get("SYNTHETIC_XR_VISIBLE") == "1":
        import omni.ui

        # Keep the live viewport large enough to compare with the browser.
        for window in omni.ui.Workspace.get_windows():
            if window.title in {
                "Stage",
                "Property",
                "Content",
                "Content Browser",
                "Console",
                "IsaacLab",
                "Render Settings",
            }:
                window.visible = False

    def measure_home():
        global home
        ee = environment.scene["ee_frame"].data
        home = np.eye(4, dtype=np.float32)
        home[:3, 3] = values(ee.target_pos_source)[0][0]
        home[:3, :3] = Rotation.from_quat(
            values(ee.target_quat_source)[0][0]
        ).as_matrix()

    # The task randomizes joints at reset. Capture the actual reset pose each
    # time, before the native recorder forwards its RESET event to the device.
    original_reset = env.reset

    def reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        measure_home()
        return result

    env.reset = reset
    measure_home()
    state["phase"] = "scene_ready"
    state["adapter"] = "None: original Franka pipeline_builder retained"
    builder = cfg.isaac_teleop.pipeline_builder
    state["native_pipeline_builder"] = builder.__module__ + "." + builder.__qualname__
    state["native_pipeline_builder_sha256"] = hashlib.sha256(
        Path(inspect.getfile(builder)).read_bytes()
    ).hexdigest()
    state["native_record_script_sha256"] = hashlib.sha256(
        REFERENCE.read_bytes()
    ).hexdigest()
    original_step = env.step

    def step(actions):
        result = original_step(actions)
        state["phase"] = "running"
        state["steps"] += 1
        state["last_action"] = values(actions)
        if os.environ.get("SYNTHETIC_XR_VISIBLE") == "1" and state["steps"] == 1:
            from omni.kit.xr.core import XRSettings
            import omni.ui

            XRSettings.get_singleton().set_setting(
                "profile/persistent/mirror/displayMode", "viewport"
            )
            for window in omni.ui.Workspace.get_windows():
                if window.title in {
                    "Layer",
                    "Layers",
                    "XR",
                    "Stage",
                    "Property",
                    "Content",
                    "Content Browser",
                    "Console",
                    "IsaacLab",
                    "Render Settings",
                }:
                    window.visible = False
        publish(force=True)
        return result

    env.step = step
    original_render = env.sim.render

    def render(*args, **kwargs):
        result = original_render(*args, **kwargs)
        publish()
        return result

    env.sim.render = render
    publish(force=True)
    return env


original_setup = native.setup_teleop_device


def setup_device(*args, **kwargs):
    global teleop
    if os.environ.get("SYNTHETIC_XR_VISIBLE") == "1":
        import carb.settings

        # The GUI normally requires the operator to enable the XR profile.
        # Use the same native lifecycle auto-start path after bridge extensions load.
        carb.settings.get_settings().set("/isaaclab/xr/auto_start", True)
    teleop = original_setup(*args, **kwargs)
    original_advance = teleop.advance

    def advance(*args, **kwargs):
        if (run_dir / "stop").exists():
            raise KeyboardInterrupt("Synthetic XR example requested stop")
        return original_advance(*args, **kwargs)

    teleop.advance = advance
    state["phase"] = "waiting_for_xr"
    return teleop


original_success = native.process_success_condition


def process_success(env, term, count):
    result = original_success(env, term, count)
    state["success_steps"] = result[0]
    state["success_reset_needed"] = result[1]
    publish()
    return result


native.create_environment = create_environment
native.create_environment_config = create_environment_config
native.setup_teleop_device = setup_device
native.process_success_condition = process_success
server = ThreadingHTTPServer(("127.0.0.1", options.feedback_port), Feedback)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    native.main()
finally:
    server.shutdown()
    server.server_close()
    trace.close()
