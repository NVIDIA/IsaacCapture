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

ASSET_URL = "https://github.com/flexivrobotics/isaac_sim_ws/raw/refs/heads/main/exts/isaacsim.robot.manipulators.examples/data/flexiv/Rizon4s_with_Grav.usd"
# ASSET_URL = "/home/linfan/Downloads/Rizon4s_with_Grav_without_gripper.usd"

FLEXIV_RIZON_4S_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ASSET_URL,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            effort_limit_sim=4000.0,
            stiffness=40000.0,
            damping=400.0,
            velocity_limit=2.0,
        ),
        "finger_joint": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "right_outer_knuckle_joint",
                "left_inner_knuckle_joint",
                "right_inner_knuckle_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                # "left_finguer_tip_joint",
                # "right_finguer_tip_joint",
            ],
            stiffness=0.0,
            damping=5.0,
            velocity_limit=0.5,  # rad/s — hard cap on closing speed
        ),
    },
)

