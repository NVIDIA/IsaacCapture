# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One task plan for native keyboard/gamepad and Synthetic XR backends.

No key names, axes, controller poses, SDK action writes or QA verdicts belong here.
World metres, tool-frame targets, xyzw unit quaternion; scene bounds are explicit.
"""

from dataclasses import dataclass
import math
from typing import Protocol


class TaskBackend(Protocol):
    """Tool targets are world metres/Z-up and unit xyzw; input stays in adapters."""

    name: str

    def observe(self) -> dict:
        """Return fresh tool/object poses, joints and object velocity."""
        ...

    def move_tool(self, position, quaternion): ...

    def set_gripper(self, *, closed: bool): ...

    def wait(self, seconds: float): ...

    def neutral(self):
        """Stop motion/release button edges without opening a held gripper."""
        ...


@dataclass(frozen=True)
class Scene:
    orientation: tuple | None = None
    clearance: float = 0.17
    contact_offset: float = 0.003
    retreat: float = 0.18
    minimum_lift: float = 0.10
    destination: tuple | None = None


def cube_top_down(quaternion):
    """Square cube symmetry: use the nearest equivalent top-down grasp yaw."""
    x, y, z, w = quaternion
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return (math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0)


def execute(backend: TaskBackend, phase, scene: Scene | None = None):
    # Construct per-call configuration rather than using a shared default object.
    scene = Scene() if scene is None else scene
    checkpoints = []
    failure = None

    def snapshot(label):
        s = backend.observe()
        checkpoints.append({"label": label, "state": s})
        return s

    def move(label, position):
        phase(label)
        backend.move_tool(position, orientation)
        return snapshot(label)

    def grip(label, closed):
        phase(label)
        backend.set_gripper(closed=closed)
        return snapshot(label)

    try:
        initial = snapshot("initial")
        orientation = scene.orientation or cube_top_down(
            initial["object_quaternion_xyzw"]
        )
        cube = list(initial["object"])
        destination = list(
            scene.destination or (cube[0], -0.15 if cube[1] > 0 else 0.15, cube[2])
        )
        if scene.destination is None and abs(destination[1] - cube[1]) < 0.15:
            destination[1] = cube[1] - 0.2
        phase("align gripper")
        backend.move_tool(initial["ee"], orientation)
        backend.set_gripper(closed=False)
        move("approach above cube", [cube[0], cube[1], cube[2] + scene.clearance])
        move("lower open gripper", [cube[0], cube[1], cube[2] + scene.contact_offset])
        grip("close gripper", True)
        lifted = move("lift cube", [cube[0], cube[1], cube[2] + scene.clearance])
        if lifted["object"][2] - cube[2] < scene.minimum_lift:
            raise RuntimeError("Task guard: cube did not lift")
        offset = [lifted["object"][i] - lifted["ee"][i] for i in range(3)]
        carry = move(
            "carry to destination",
            [
                destination[0] - offset[0],
                destination[1] - offset[1],
                cube[2] + scene.clearance,
            ],
        )
        if carry["object"][2] - cube[2] < scene.minimum_lift:
            raise RuntimeError("Task guard: cube dropped during carry")
        move(
            "lower cube onto table",
            [
                destination[i] - offset[i] + (scene.contact_offset if i == 2 else 0)
                for i in range(3)
            ],
        )
        grip("release cube", False)
        move(
            "retreat with open gripper",
            [
                destination[0] - offset[0],
                destination[1] - offset[1],
                cube[2] + scene.retreat,
            ],
        )
        phase("hands off: stable placement")
        backend.neutral()
        backend.wait(3)
        final = snapshot("stable placement")
        return {
            "task_completed": True,
            "initial_object": cube,
            "destination": destination,
            "placement_error_m": math.dist(final["object"][:2], destination[:2]),
            "checkpoints": checkpoints,
            "backend": backend.name,
        }
    except BaseException as error:
        failure = error
        raise
    finally:
        # Stop motion/release button edges. Backend retains explicit gripper state.
        try:
            backend.neutral()
        except BaseException as cleanup:
            if failure is None:
                raise
            failure.cleanup_error = repr(cleanup)
            if hasattr(failure, "add_note"):
                failure.add_note("Cleanup also failed: " + repr(cleanup))
