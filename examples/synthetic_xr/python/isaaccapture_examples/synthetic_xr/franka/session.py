# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-facing Franka tool commands over the native Synthetic XR path."""

import json
from pathlib import Path
import time
import math
import os
from urllib.error import URLError

import numpy as np
from scipy.spatial.transform import Rotation

from .browser_command import BrowserClient
from .xr_controller_bridge import XRControllerBridge, play, wait_steps

TOOL_OFFSET = np.array([0.0, 0.0, 0.1034])


def validate_observation(value):
    if value.get("schema_version") != 1:
        raise ValueError(
            "Expected Synthetic XR observation schema_version=1; start the matching observer"
        )
    if (
        not value.get("run_id")
        or value.get("coordinate_convention") != "metres; world Z-up; quaternion xyzw"
    ):
        raise ValueError(
            "Missing observation identity or incompatible coordinate convention"
        )
    if (
        not np.isfinite(value.get("simulation_dt", 0))
        or value.get("simulation_dt", 0) <= 0
    ):
        raise ValueError("Invalid simulation step duration")
    frame_count = None
    for name, width in (("ee_positions_world", 3), ("ee_quaternions_world", 4)):
        array = np.asarray(value[name])
        # Native ee_frame includes the tool followed by finger targets; index zero is the tool.
        if (
            array.ndim != 3
            or array.shape[0] != 1
            or array.shape[1] < 1
            or array.shape[2] != width
            or not np.isfinite(array).all()
        ):
            raise ValueError("Invalid observation field: " + name)
        if frame_count is not None and array.shape[1] != frame_count:
            raise ValueError("Position and orientation target counts differ")
        frame_count = array.shape[1]
    for obj in value["cubes"].values():
        for name, width in (
            ("position_world", 3),
            ("quaternion_world", 4),
            ("linear_velocity_world", 3),
        ):
            array = np.asarray(obj[name])
            if array.shape != (1, width) or not np.isfinite(array).all():
                raise ValueError("Invalid object observation: " + name)
    return value


class ObserverUnavailable(RuntimeError):
    """The supervisor has not yet confirmed worker completion."""


def check_timeout(timeout):
    if not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("Timeout must be in (0, 300] wall-clock seconds")


class FrankaTaskSession:
    """Attach to one local observer; release input and stop it on context exit."""

    def __init__(
        self,
        *,
        observation_dir,
        output,
        cdp_url="http://127.0.0.1:9222",
        feedback_url="http://127.0.0.1:8765/state",
        stop_observer=True,
    ):
        self.observation_dir = Path(observation_dir).resolve()
        self.output = Path(output).resolve()
        self.client = BrowserClient(cdp_url, feedback_url)
        self.controller = None
        self.run_id = None
        self.stop_observer = stop_observer
        self._trace = None
        self._entered = False

    def state(self, max_age=5):
        try:
            value = self.client.state(max_age)
        except (URLError, OSError):
            value = json.loads((self.observation_dir / "sim-state.json").read_text())
            # Only an orderly terminal snapshot can outlive its HTTP server.
            if value.get("phase") != "completed" or value.get("completion_error"):
                raise ObserverUnavailable(
                    "Observer disconnected before orderly completion"
                ) from None
        value = validate_observation(value)
        if value.get("phase") == "completed" and value.get("completion_error"):
            raise RuntimeError("Observer failed: " + value["completion_error"])
        if Path(value["run_dir"]).resolve() != self.observation_dir:
            raise RuntimeError("Feedback belongs to a different observation directory")
        if self.run_id is not None and value["run_id"] != self.run_id:
            raise RuntimeError("Observer restarted during the task")
        self._heartbeat()
        return value

    def _heartbeat(self):
        if self.run_id is not None:
            temporary = self.observation_dir / "driver-heartbeat.tmp"
            temporary.write_text(
                json.dumps({"run_id": self.run_id, "monotonic": time.monotonic()})
            )
            temporary.replace(self.observation_dir / "driver-heartbeat.json")

    def command(self, *args, **kwargs):
        self._heartbeat()
        return self.client.command(*args, **kwargs)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Task sessions cannot be entered twice")
        self._entered = True
        self.output.mkdir(parents=True, exist_ok=False)
        self.client.output = self.output / "commands"
        self.client.output.mkdir()
        self._trace = (self.output / "stages.jsonl").open("x", buffering=1)
        try:
            observation = self.state()
            if observation.get("phase") == "completed" or observation.get(
                "exported_successful_episodes", 0
            ):
                raise RuntimeError(
                    "Start a fresh observer before opening a task session"
                )
            # Reserve before sending input; another driver must not control this observer.
            with (self.observation_dir / "driver-owner.json").open("x") as owner:
                json.dump({"run_id": observation["run_id"], "pid": os.getpid()}, owner)
            self.run_id = observation["run_id"]
            self.command("inspect")
            self.command(
                "head", quaternion=Rotation.from_euler("x", -0.55).as_quat().tolist()
            )
            self.controller = XRControllerBridge(
                client=self, acknowledgement_path=self.output / "controller-acks.jsonl"
            )
            if not observation.get("control", {}).get("is_active"):
                play(client=self)
            self.controller.target_base(
                self.controller.target[:3, 3], self.controller.target[:3, :3]
            )
            self.mark("started")
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def observe(self):
        return self.state()

    def mark(self, name):
        observation = self.observe()
        self._trace.write(
            json.dumps({"stage": name, "state": observation}, allow_nan=False) + "\n"
        )
        return observation

    def wait(self, seconds, timeout=30):
        check_timeout(timeout)
        if not np.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError("Wait duration must be in (0, 120] simulation seconds")
        return wait_steps(
            max(1, int(np.ceil(seconds / self.state()["simulation_dt"]))),
            timeout,
            client=self,
        )

    def move_tool(self, position, quaternion, *, seconds=2):
        """Move the ee_frame in world metres, with a unit xyzw quaternion."""
        point, q = (
            np.asarray(position, dtype=float),
            np.asarray(quaternion, dtype=float),
        )
        if (
            point.shape != (3,)
            or not np.isfinite(point).all()
            or q.shape != (4,)
            or not np.isfinite(q).all()
            or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)
        ):
            raise ValueError(
                "Expected finite world position[3] and unit xyzw quaternion[4]"
            )
        rotation = Rotation.from_quat(q).as_matrix()
        self.controller.move_world(
            point - rotation @ TOOL_OFFSET, rotation, seconds=seconds
        )
        for _ in range(4):
            current = self.state()
            if current.get("exported_successful_episodes", 0):
                return current
            error = point - np.asarray(current["ee_positions_world"])[0, 0]
            if np.linalg.norm(error) < 0.002:
                return current
            target = (
                np.linalg.inv(self.controller.base_to_world) @ self.controller.target
            )
            self.controller.move_world(
                target[:3, 3] + np.clip(error, -0.01, 0.01), rotation, seconds=0.6
            )
        if (
            np.linalg.norm(point - np.asarray(self.state()["ee_positions_world"])[0, 0])
            > 0.01
        ):
            raise RuntimeError(
                "Tool did not converge within 0.01 m of its requested target"
            )
        return self.state()

    def set_gripper(self, *, closed):
        if type(closed) is not bool:
            raise TypeError("closed must be bool")
        if self.state().get("exported_successful_episodes", 0):
            raise RuntimeError("Episode already completed")
        self.controller.target_base(
            self.controller.target[:3, 3],
            self.controller.target[:3, :3],
            trigger=float(closed),
        )

    def wait_for_export(self, timeout=30):
        """Wait for native recorder confirmation; never force a success flag."""
        check_timeout(timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                observation = self.state()
            except ObserverUnavailable:
                time.sleep(0.05)
                continue
            if (
                observation.get("exported_successful_episodes", 0) > 0
                and observation.get("phase") == "completed"
            ):
                self.mark("native_export")
                (self.output / "result.json").write_text(
                    json.dumps(
                        {
                            "status": "PASS",
                            "run_id": self.run_id,
                            "native_exports": observation[
                                "exported_successful_episodes"
                            ],
                            "success_hook": observation.get(
                                "success_hook", "native task"
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                return observation
            if observation.get("phase") == "completed":
                raise RuntimeError(
                    "Observer completed without a native successful episode"
                )
            time.sleep(0.05)
        raise TimeoutError("Native success condition did not export an episode")

    def __exit__(self, error_type, error, traceback):
        cleanup_error = None
        try:
            if self.run_id is not None:
                self.command("stop")
        except Exception as failure:
            cleanup_error = failure
        finally:
            if self.controller is not None:
                self.controller.ack_trace.close()
            self.client.close()
            if self._trace is not None:
                self._trace.close()
            if self.run_id is not None and self.stop_observer:
                try:
                    current = json.loads(
                        (self.observation_dir / "sim-state.json").read_text()
                    )
                    if (
                        current.get("run_id") == self.run_id
                        and current.get("phase") != "completed"
                    ):
                        (self.observation_dir / "stop").touch()
                except Exception as failure:
                    cleanup_error = cleanup_error or failure
        if error is not None or cleanup_error is not None:
            (self.output / "result.json").write_text(
                json.dumps(
                    {
                        "status": "FAIL",
                        "error": repr(error),
                        "cleanup_error": repr(cleanup_error),
                    },
                    indent=2,
                )
                + "\n"
            )
            (self.output / "failure.json").write_text(
                json.dumps(
                    {"error": repr(error), "cleanup_error": repr(cleanup_error)},
                    indent=2,
                )
                + "\n"
            )
        if cleanup_error is not None and error is None:
            raise cleanup_error
        return False
