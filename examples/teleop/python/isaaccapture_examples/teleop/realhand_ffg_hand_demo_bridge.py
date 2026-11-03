# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the standard FFG DeviceIO stream separately from Isaac Sim's Python ABI."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import signal
import socket
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

_LOGGER = logging.getLogger("isaaccapture.examples.realhand_ffg_hand_demo_bridge")


def load_calibration_example(name="realhand_ffg_glove_calibration"):
    return importlib.import_module(f".{name}", __package__)


def validate_saved_calibration(path, example):
    # Previewing CTS calibration must never refit or rewrite the user's measurements.
    calibration = example._load_output(path)
    example._validate_calibration(calibration, tuple(calibration["hands"]))
    return calibration


class CalibrationCapture:
    def __init__(self, path, example):
        self.path = Path(path)
        self.example = example
        self.poses = example._CAPTURE_POSES
        self.prepare_seconds = example._DEFAULT_PREPARE_SECONDS
        self.hold_seconds = example._DEFAULT_HOLD_SECONDS
        self.sample_count = example._DEFAULT_SAMPLES
        self.state = "idle"
        self.sides = ()
        self.step = 0
        self.deadline = 0.0
        self.samples = {}
        self.captured = {}
        self.message = "Select the connected glove side, then start calibration."

    def command(self, command, connected, now):
        action = command.get("action")
        if action == "cancel":
            self.state = "idle"
            self.captured.clear()
            self.samples.clear()
            self.message = "Calibration cancelled. Existing calibration is unchanged."
        elif action == "begin":
            sides = command.get("sides")
            if (
                not isinstance(sides, list)
                or not sides
                or any(side not in ("left", "right") for side in sides)
            ):
                self.message = "Choose Left, Right, or Both."
                return
            if not all(connected.get(side, False) for side in sides):
                self.message = "Requested glove is not connected. Check its USB port and permissions."
                return
            self.sides = tuple(dict.fromkeys(sides))
            self.captured = {side: {} for side in self.sides}
            self.step = 0
            self.state = "ready"
            self.message = "Hold the requested pose, then click Capture Pose."
        elif action == "capture" and self.state == "ready":
            if not all(connected.get(side, False) for side in self.sides):
                self.message = (
                    "A requested glove is disconnected. Reconnect it before capturing."
                )
                return
            self.samples = {side: [] for side in self.sides}
            self.deadline = now + self.prepare_seconds
            self.state = "preparing"
            self.message = "Release the mouse and prepare both hands before HOLD."

    def update(self, raw, now):
        if self.state == "preparing":
            if now < self.deadline:
                return False
            self.deadline = now + self.hold_seconds
            self.state = "capturing"
            self.message = "Hold still until the countdown finishes."
        if self.state != "capturing":
            return False
        if any(raw.get(side) is None for side in self.sides):
            self.state = "error"
            self.message = "Glove data was lost during capture. Restart calibration."
            return False
        for side in self.sides:
            values = np.asarray(raw[side], dtype=float)
            if values.shape != (21,) or not np.all(np.isfinite(values)):
                self.state = "error"
                self.message = "Invalid glove data. Restart calibration."
                return False
            self.samples[side].append(values.copy())
        if now < self.deadline:
            return False
        if any(len(samples) < self.sample_count for samples in self.samples.values()):
            self.state = "error"
            self.message = "Too few valid samples. Restart calibration."
            return False
        key = self.poses[self.step][1]
        for side, samples in self.samples.items():
            self.example._store_capture(self.captured[side], key, samples)
        self.step += 1
        if self.step < len(self.poses):
            self.state = "ready"
            self.message = (
                "Pose captured. Prepare the next pose, then click Capture Pose."
            )
            return False
        try:
            self.example._fit_and_save(self.path, self.captured)
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.state = "error"
            self.message = str(error)
            return False
        self.state = "complete"
        self.message = "Calibration saved. Reloading the FFG hand plugin..."
        return True

    def status(self, now):
        step = min(self.step, len(self.poses) - 1)
        return {
            "state": self.state,
            "step": step + 1,
            "total": len(self.poses),
            "instruction": self.poses[step][0]
            if self.state in ("ready", "preparing", "capturing")
            else "",
            "remaining": max(0.0, self.deadline - now)
            if self.state in ("preparing", "capturing")
            else 0.0,
            "message": self.message,
            "file": str(self.path),
        }


def build_pipeline(hand_model=None, retargeters=None):
    from isaaccapture.retargeting_engine.deviceio_source_nodes import (
        HandsSource,
        JointStateSource,
    )
    from isaaccapture.retargeting_engine.interface import OutputCombiner

    hands = HandsSource(name="ffg_demo_hands")
    outputs = {
        side: hands.output(getattr(HandsSource, side.upper()))
        for side in ("left", "right")
    }
    if hand_model is not None:
        from isaaccapture.retargeters import DexHandRetargeter

        example = load_calibration_example("p7_realhand_bimanual_example")
        for side in ("left", "right"):
            # Scene diagnostics must use the same official solver as the combined example.
            retargeter = DexHandRetargeter(
                example.dex_hand_config(hand_model, side),
                name=f"demo_{hand_model}_{side}",
            )
            connected = retargeter.connect(
                {f"hand_{side}": hands.output(getattr(HandsSource, side.upper()))}
            )
            outputs[f"robot_{side}"] = connected.output("hand_joints")
            if retargeters is not None:
                retargeters[side] = retargeter
    for side in ("left", "right"):
        raw = JointStateSource(
            name=f"ffg_demo_raw_{side}",
            collection_id=f"realhand_ffg_glove_{side}",
            joint_names=[f"sensor_{sensor}" for sensor in range(21)],
        )
        outputs[f"raw_{side}"] = raw.output(JointStateSource.JOINTS)
    return OutputCombiner(outputs)


@lru_cache(maxsize=6)
def robot_link_names(urdf):
    # Pinocchio exposes JOINT and BODY frames; only URDF links are drawable bodies.
    return tuple(link.get("name") for link in ET.parse(urdf).getroot().findall("link"))


def robot_frame(retargeter, output):
    """Render FK from the official solver's named joint output and mimic adaptor."""
    optimizer = retargeter._dex_hand.optimizer
    robot = optimizer.robot
    names = retargeter._config.hand_joint_names
    values = (
        np.asarray([float(value) for value in output], dtype=float)
        if output is not None
        else np.zeros(len(names))
    )
    qpos = np.zeros(robot.dof)
    for name, value in zip(names, values):
        qpos[robot.get_joint_index(name)] = value
    if optimizer.adaptor is not None:
        qpos = optimizer.adaptor.forward_qpos(qpos)
    robot.compute_forward_kinematics(qpos)
    poses = {
        name: robot.get_link_pose(robot.get_link_index(name)).tolist()
        for name in robot_link_names(retargeter._config.hand_urdf)
    }
    return {"joint_names": names, "positions": values.tolist(), "link_poses": poses}


def replay_robot_frames(model, fixture):
    """Replay native skeleton fixtures through the official solver without OpenXR."""
    from isaaccapture.retargeters import DexHandRetargeter
    from isaaccapture.retargeting_engine.interface import (
        ComputeContext,
        ExecutionEvents,
        ExecutionState,
        OptionalTensorGroup,
        TensorGroup,
    )
    from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
        GraphTime,
    )
    from isaaccapture.retargeting_engine.tensor_types import HandInputIndex

    example = load_calibration_example("p7_realhand_bimanual_example")
    retargeters = {}
    ios = {}
    for side in ("left", "right"):
        retargeter = DexHandRetargeter(
            example.dex_hand_config(model, side), name=f"replay_{side}"
        )
        retargeters[side] = retargeter
        ios[side] = (
            {
                key: OptionalTensorGroup(spec)
                for key, spec in retargeter.input_spec().items()
            },
            {key: TensorGroup(spec) for key, spec in retargeter.output_spec().items()},
        )
    context = ComputeContext(
        graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
        execution_events=ExecutionEvents(
            reset=False, execution_state=ExecutionState.RUNNING
        ),
    )
    count = min(len(fixture[side]) for side in ("left", "right"))
    packets = []
    for start in range(count):
        for alpha in np.linspace(0.0, 1.0, 20, endpoint=False):
            packet = {
                "connected": {"left": True, "right": True},
                "hands": {},
                "robots": {},
                "calibration": {
                    "state": "idle",
                    "instruction": "",
                    "message": "Fixture replay; no live glove or calibration capture.",
                },
            }
            for side, retargeter in retargeters.items():
                first = fixture[side][start]["joints"]
                second = fixture[side][(start + 1) % count]["joints"]
                inputs, outputs = ios[side]
                group = inputs[f"hand_{side}"]
                positions = (1 - alpha) * np.asarray(
                    [j["position"] for j in first]
                ) + alpha * np.asarray([j["position"] for j in second])
                group[HandInputIndex.JOINT_POSITIONS] = positions.astype(np.float32)
                group[HandInputIndex.JOINT_ORIENTATIONS] = np.asarray(
                    [j["orientation"] for j in first], dtype=np.float32
                )
                group[HandInputIndex.JOINT_RADII] = np.asarray(
                    [j["radius"] for j in first], dtype=np.float32
                )
                group[HandInputIndex.JOINT_VALID] = np.ones(26, dtype=np.uint8)
                retargeter.compute(inputs, outputs, context)
                packet["hands"][side] = hand_positions(group)
                packet["robots"][side] = robot_frame(retargeter, outputs["hand_joints"])
            packets.append(packet)
    return packets


def hand_positions(group):
    from isaaccapture.retargeting_engine.tensor_types import HandInputIndex

    if group.is_none:
        return None
    valid = np.from_dlpack(group[HandInputIndex.JOINT_VALID])
    if not np.all(valid):
        return None
    positions = np.from_dlpack(group[HandInputIndex.JOINT_POSITIONS]).copy()
    orientations = np.from_dlpack(group[HandInputIndex.JOINT_ORIENTATIONS])
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(orientations)):
        return None
    q = orientations[1]
    length = np.linalg.norm(q)
    if length < 1e-6:
        return None
    # Fixed wrists in the viewer: remove the OpenXR wrist's global rigid transform.
    xyz, w = -q[:3] / length, q[3] / length
    local = positions - positions[1]
    cross = 2.0 * np.cross(xyz, local)
    local = local + w * cross + np.cross(xyz, cross)
    return local.tolist()


def sample_glove_inputs(session, result, example, status_tracker):
    report = status_tracker.get_device_status_snapshot(session.deviceio_session)
    devices = {}
    if (
        report is not None
        and report.schema_version == 1
        and 0 <= time.monotonic_ns() - report.report_time_ns < 2_000_000_000
    ):
        devices = {device.path: device for device in report.devices}
    raw = {}
    errors = {}
    for side in ("left", "right"):
        device = devices.get(f"/glove/realhand_ffg_glove_{side}")
        # JointStateSource may retain its last tensor after USB disconnection.
        raw[side] = (
            example._sample_group(result, f"raw_{side}")
            if device is not None and device.state.name == "CONNECTED"
            else None
        )
        errors[side] = device.error if device is not None else ""
        hand = devices.get(f"/hand/{side}")
        if (
            raw[side] is not None
            and hand is not None
            and hand.state.name != "CONNECTED"
        ):
            errors[side] = hand.error
    return raw, errors


def plugin_arguments(ports, calibration):
    # Fixed-wrist previews accept local, untracked skeletons; CTS still requires a tracked wrist.
    arguments = ["--wrist-source=auto"]
    if ports:
        arguments.append(f"--ports={ports}")
    if calibration.is_file():
        arguments.append(f"--calibration={calibration.resolve()}")
    return arguments


def main():
    from isaaccapture.cloudxr import CloudXRLauncher
    from isaaccapture.deviceio_trackers import PluginDeviceStatusTracker
    from isaaccapture.teleop_session_manager import (
        PluginConfig,
        TeleopSession,
        TeleopSessionConfig,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc-dir", type=Path)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--hand-model", choices=("l6", "o6", "l20"))
    parser.add_argument("--test-frames", type=Path)
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument(
        "--ports",
        help="Comma-separated verified glove ports; omit for identified-device discovery",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    CloudXRLauncher.add_launcher_arguments(parser)
    parser.set_defaults(
        cloudxr_device_profile="quest3",
        cloudxr_env_config=str(Path(__file__).with_name("realhand_ffg_hand_demo.env")),
    )
    args = parser.parse_args()
    if args.test_frames is not None:
        if args.hand_model is None or args.replay_output is None:
            parser.error(
                "Robot fixture replay requires --hand-model and --replay-output"
            )
        fixture = yaml.safe_load(args.test_frames.read_text(encoding="utf-8"))
        packets = replay_robot_frames(args.hand_model, fixture)
        args.replay_output.write_text(
            json.dumps(packets, allow_nan=False), encoding="utf-8"
        )
        return
    if any(
        value is None for value in (args.ipc_dir, args.plugin_path, args.calibration)
    ):
        parser.error("Live mode requires --ipc-dir, --plugin-path, and --calibration")
    example = load_calibration_example()
    capture = CalibrationCapture(args.calibration, example)
    try:
        validate_saved_calibration(args.calibration, example)
    except ValueError as error:
        capture.message = (
            f"Existing calibration invalid: {error}. Start calibration to replace it."
        )
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    frame_path = str(args.ipc_dir / "frames.sock")
    command_path = args.ipc_dir / "commands.sock"
    started = time.monotonic()
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
        channel.bind(str(command_path))
        channel.setblocking(False)
        try:
            with CloudXRLauncher.launch_context(args, host_client=False):
                while running:
                    if args.calibration.is_file():
                        example._load_output(args.calibration)
                    status_tracker = PluginDeviceStatusTracker(
                        "realhand_ffg_glove/device_status"
                    )
                    retargeters = {}
                    config = TeleopSessionConfig(
                        app_name="RealHandFFGValidation",
                        trackers=[status_tracker],
                        pipeline=build_pipeline(args.hand_model, retargeters),
                        plugins=[
                            PluginConfig(
                                plugin_name="realhand_ffg_glove_plugin",
                                plugin_root_id="realhand_ffg_glove",
                                search_paths=[args.plugin_path],
                                plugin_args=plugin_arguments(
                                    args.ports, args.calibration
                                ),
                                required=True,
                            )
                        ],
                    )
                    restart = False
                    with TeleopSession(config) as session:
                        if capture.state == "complete":
                            capture.message = "Calibration active. The standard 26-joint hand stream is now displayed."
                        while running and not restart:
                            now = time.monotonic()
                            if args.max_seconds and now - started > args.max_seconds:
                                running = False
                                break
                            result = session.step()
                            raw, errors = sample_glove_inputs(
                                session, result, example, status_tracker
                            )
                            connected = {
                                side: values is not None for side, values in raw.items()
                            }
                            hands = {
                                side: hand_positions(result[side])
                                if connected[side]
                                else None
                                for side in ("left", "right")
                            }
                            while True:
                                try:
                                    command = json.loads(channel.recv(4096))
                                except BlockingIOError:
                                    break
                                capture.command(command, connected, now)
                            restart = capture.update(raw, now)
                            packet = {
                                "time": now,
                                "connected": connected,
                                "device_errors": errors,
                                "hands": hands,
                                "calibration": capture.status(now),
                            }
                            if retargeters:
                                packet["robots"] = {
                                    side: robot_frame(
                                        retargeter,
                                        result[f"robot_{side}"]
                                        if connected[side]
                                        else None,
                                    )
                                    for side, retargeter in retargeters.items()
                                }
                            try:
                                channel.sendto(
                                    json.dumps(packet, allow_nan=False).encode(),
                                    frame_path,
                                )
                            except (BlockingIOError, FileNotFoundError):
                                pass
                            time.sleep(max(0.0, 1.0 / 90.0 - (time.monotonic() - now)))
        finally:
            command_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
