# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent URDF forward kinematics for the robot-hand viewers."""

from functools import lru_cache
import xml.etree.ElementTree as ET

from dex_retargeting.robot_wrapper import RobotWrapper
import numpy as np


class HandKinematics:
    def __init__(self, urdf, joint_names):
        self.robot = RobotWrapper(str(urdf))
        self.joint_names = tuple(joint_names)
        root = ET.parse(urdf).getroot()
        self.link_names = tuple(link.get("name") for link in root.findall("link"))
        self.mimics = {
            joint.get("name"): (
                mimic.get("joint"),
                float(mimic.get("multiplier", "1")),
                float(mimic.get("offset", "0")),
            )
            for joint in root.findall("joint")
            if (mimic := joint.find("mimic")) is not None
        }

    def full_qpos(self, values):
        values = dict(zip(self.joint_names, values, strict=True))
        pending = dict(self.mimics)
        while pending:
            resolved = [
                name for name, (source, _, _) in pending.items() if source in values
            ]
            if not resolved:
                raise ValueError("URDF mimic graph has an unresolved source or cycle")
            for name in resolved:
                source, multiplier, offset = pending.pop(name)
                values[name] = multiplier * values[source] + offset
        return np.asarray(
            [values.get(name, 0.0) for name in self.robot.dof_joint_names]
        )

    def positions(self, qpos, links):
        self.robot.compute_forward_kinematics(qpos)
        return np.asarray(
            [
                self.robot.get_link_pose(self.robot.get_link_index(name))[:3, 3]
                for name in links
            ]
        )

    def frame(self, values):
        self.robot.compute_forward_kinematics(self.full_qpos(values))
        return {
            "joint_names": list(self.joint_names),
            "positions": list(values),
            "link_poses": {
                name: self.robot.get_link_pose(self.robot.get_link_index(name)).tolist()
                for name in self.link_names
            },
        }


@lru_cache(maxsize=6)
def hand_kinematics(urdf, joint_names):
    return HandKinematics(urdf, joint_names)
