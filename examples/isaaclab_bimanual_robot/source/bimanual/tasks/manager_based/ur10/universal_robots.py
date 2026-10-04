# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_LEFT_CFG`: The UR10 arm without a gripper (left side).
* :obj:`UR10_RIGHT_CFG`: The UR10 arm without a gripper (right side).

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

UR10_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.28, 0.74, 0.827),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan_joint": 1.5707,
            "shoulder_lift_joint": -1.5707,
            "elbow_joint": 1.5707,
            "wrist_1_joint": -1.5707,
            "wrist_2_joint": -1.5707,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            effort_limit_sim=150.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


UR10_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.00, 0.74, 0.827),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan_joint": -1.5707,
            "shoulder_lift_joint": -1.5707,
            "elbow_joint": -1.5707,
            "wrist_1_joint": -1.5707,
            "wrist_2_joint": 1.5707,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            effort_limit_sim=150.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
