# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick-and-place destination and native success callback for the Franka scene."""

import numpy as np

_history = {}


# One application-defined destination keeps driver and callback decisions identical.
DESTINATION_XY = (0.45, -0.20)


class DestinationUnavailable(RuntimeError):
    """The configured placement region is unsuitable for this episode."""


def destination(observation):
    start = np.array(observation["cubes"]["cube_2"]["position_world"][0])
    goal = np.array([*DESTINATION_XY, start[2]])
    others = [
        np.array(value["position_world"][0])
        for name, value in observation["cubes"].items()
        if name != "cube_2"
    ]
    if np.linalg.norm((goal - start)[:2]) <= 0.18 or any(
        np.linalg.norm((goal - other)[:2]) <= 0.12 for other in others
    ):
        raise DestinationUnavailable(
            "Example destination is occupied or too close; choose a different task destination"
        )
    return goal


def is_success(observation):
    episode = observation["episode_id"]
    cube = np.array(observation["cubes"]["cube_2"]["position_world"][0])
    if _history.get("episode") != episode:
        try:
            goal = destination(observation)
        except DestinationUnavailable:
            goal = None
        _history.update(
            episode=episode,
            start=cube.copy(),
            goal=goal,
            lifted=False,
        )
    if _history["goal"] is None:
        return False
    _history["lifted"] |= bool(cube[2] - _history["start"][2] >= 0.08)
    goal = _history["goal"]
    ee = np.array(observation["ee_positions_world"][0][0])
    fingers = [
        value
        for name, value in zip(observation["joint_names"], observation["joints"][0])
        if name.startswith("panda_finger_")
    ]
    return bool(
        _history["lifted"]
        and np.linalg.norm((cube - goal)[:2]) < 0.015
        and abs(cube[2] - goal[2]) < 0.008
        and np.linalg.norm(observation["cubes"]["cube_2"]["linear_velocity_world"][0])
        < 0.02
        and ee[2] - goal[2] >= 0.16
        and len(fingers) == 2
        and all(abs(f - 0.04) < 0.005 for f in fingers)
    )
