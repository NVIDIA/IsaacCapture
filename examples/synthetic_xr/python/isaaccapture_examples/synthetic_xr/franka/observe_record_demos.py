# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe native Franka recording without replacing its retargeting pipeline.

Robot actions come exclusively through the native WebXR/CloudXR device.
Observation, optional task success and lifecycle hooks preserve native motion and export.
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
from types import SimpleNamespace

from .task_hooks import (
    check_driver_lease,
    evaluate_success,
    install_reset_observer,
    load_success_hook,
)
from .observer_process import supervise, write_snapshot

import numpy as np
from scipy.spatial.transform import Rotation


# Help must work before required paths are checked or any native worker starts.
parser = argparse.ArgumentParser(
    description="Observe Franka and supervise native episode recording.",
    epilog="Additional arguments are forwarded to Lab scripts/tools/record_demos.py; "
    "run that script with --help for native task options.",
)
parser.add_argument("--lab-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--feedback-port", type=int, default=8765)
parser.add_argument(
    "--objects",
    nargs="+",
    default=["cube_1", "cube_2", "cube_3"],
    help="Scene object names exposed in observations",
)
parser.add_argument(
    "--success",
    help="Trusted module:function accepting one observation and returning bool",
)
parser.add_argument(
    "--driver-timeout",
    type=float,
    default=30,
    help="Stop after this many seconds without an attached task driver heartbeat",
)
parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
parser.add_argument(
    "--session-timeout",
    type=float,
    default=900,
    help="Maximum total observer runtime in seconds",
)
options, remaining = parser.parse_known_args()
if not np.isfinite(options.session_timeout) or options.session_timeout <= 0:
    parser.error("Session timeout must be positive and finite")
if options.worker_run_id is None:
    raise SystemExit(
        supervise(options.output.resolve(), sys.argv[1:], options.session_timeout)
    )
if not np.isfinite(options.driver_timeout) or options.driver_timeout <= 0:
    parser.error("Driver timeout must be positive and finite")
if len(set(options.objects)) != len(options.objects):
    parser.error("Object names must be unique")
success_callback, success_identity = (
    (None, "native task")
    if options.success is None
    else load_success_hook(options.success)
)
ROOT = options.output.resolve()
REFERENCE = options.lab_root.resolve() / "scripts/tools/record_demos.py"
if not REFERENCE.is_file():
    raise FileNotFoundError(REFERENCE)
if (
    json.loads((ROOT / "supervisor.json").read_text())["run_id"]
    != options.worker_run_id
):
    raise RuntimeError("Worker does not belong to this supervisor")
sys.argv = [str(REFERENCE), *remaining]
spec = importlib.util.spec_from_file_location("synthetic_xr_record_demos", REFERENCE)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)

state = {
    "schema_version": 1,
    "run_id": options.worker_run_id,
    "coordinate_convention": "metres; world Z-up; quaternion xyzw",
    "success_hook": success_identity,
    "episode_id": 0,
    "phase": "initializing",
    "steps": 0,
    "success_steps": 0,
}
snapshot = json.dumps(state).encode()
last_publish = 0.0
environment = None
teleop = None
home = None
clutch_node = None
run_dir = ROOT
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
                for name in options.objects
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
    if not use_capture:
        raise RuntimeError(
            "Synthetic XR requires the native Isaac Teleop task pipeline"
        )
    if success_callback is not None:

        def custom_success(env):
            import torch

            # A reset increments episode_id before the throttled render publisher
            # refreshes scene data. Never initialize a task callback from that
            # mixed snapshot; read the native scene at this evaluation boundary.
            publish(force=True)
            return torch.tensor(
                [evaluate_success(success_callback, state)],
                dtype=torch.bool,
                device=env.device,
            )

        success = SimpleNamespace(func=custom_success, params={})
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

    def record_reset():
        state["episode_id"] += 1

    # Lab's step() resets terminated environments through _reset_idx, bypassing
    # reset(). Observe that shared boundary without replacing native reset behavior.
    install_reset_observer(environment, record_reset)
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
        episode_before = state["episode_id"]
        result = original_step(actions)
        if state["episode_id"] != episode_before:
            measure_home()
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
    original_close = env.close

    def close():
        result = original_close()
        state.update(
            native_environment_closed=True, phase="native_closed", timestamp=time.time()
        )
        write_snapshot(ROOT, state)
        return result

    env.close = close
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
        check_driver_lease(
            run_dir / "driver-heartbeat.json", state["run_id"], options.driver_timeout
        )
        if (run_dir / "stop").exists():
            raise KeyboardInterrupt("Synthetic XR example requested stop")
        return original_advance(*args, **kwargs)

    teleop.advance = advance
    state["phase"] = "waiting_for_xr"
    return teleop


original_success = native.process_success_condition


def process_success(env, term, count):
    before = state.get("exported_successful_episodes", 0)
    result = original_success(env, term, count)
    state["success_steps"] = result[0]
    state["success_reset_needed"] = result[1]
    publish(force=True)
    exported = state.get("exported_successful_episodes", 0)
    if exported > before:
        (ROOT / f"exported-episode-{exported}.json").write_bytes(snapshot)
    return result


native.create_environment = create_environment
native.create_environment_config = create_environment_config
native.setup_teleop_device = setup_device
native.process_success_condition = process_success
server = ThreadingHTTPServer(("127.0.0.1", options.feedback_port), Feedback)
threading.Thread(target=server.serve_forever, daemon=True).start()
completion_error = None
try:
    native.main()
except BaseException as error:
    completion_error = repr(error)
    raise
finally:
    # Preserve final counters after native cleanup without touching destroyed tensors.
    state.update(
        phase="worker_finished",
        timestamp=time.time(),
        completion_error=completion_error,
    )
    snapshot = json.dumps(state, allow_nan=False).encode()
    temporary = ROOT / "sim-state.tmp"
    temporary.write_bytes(snapshot)
    temporary.replace(ROOT / "sim-state.json")
    trace.write(snapshot.decode() + "\n")
    server.shutdown()
    server.server_close()
    trace.close()
