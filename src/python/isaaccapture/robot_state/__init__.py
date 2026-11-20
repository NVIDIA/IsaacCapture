# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot state: what a robot is and its newest readings, read on the Teleop Session Server machine."""

from importlib import import_module

_EXPORTS = {
    "Fault": ".records",
    "GripperDescription": ".description",
    "GripperReading": ".reader",
    "JointDescription": ".description",
    "JointReading": ".reader",
    "JointsReading": ".reader",
    "Reading": ".reader",
    "RecordReading": ".reader",
    "RobotDescription": ".description",
    "RobotDescriptionError": ".description",
    "RobotRecord": "._robot_state",
    "RobotSeriesInfo": "._robot_state",
    "RobotState": ".reader",
    "RobotStateLocalService": "._local",
    "RobotStateStore": "._robot_state",
    "SeriesDescription": ".description",
    "StatusReading": ".reader",
    "ValueDescription": ".description",
    "default_socket_path": "._local",
    "load_robot_description": ".description",
    "parse_robot_description": ".description",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
