# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp


from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.sensors import CameraCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils

##
# Pre-defined configs
##

from bimanual.tasks.manager_based.common.base_env_cfg import BaseEnvCfg
from bimanual.tasks.manager_based.common.dummy_retargeter import DummyRetargeterCfg

from .universal_robots import UR10_LEFT_CFG, UR10_RIGHT_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class UR10BimanualEnvCfg(BaseEnvCfg):
    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.55, -0.3, -0.15),
        anchor_rot=(1, 0, 0, 0),
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.left_robot = UR10_LEFT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/left_robot",
        )
        self.scene.right_robot = UR10_RIGHT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/right_robot",
        )

        # Replace joint-position arm action with differential IK (relative pose)
        # NOTE: body_offset.pos is the TCP offset from ee_link along its z-axis (~130 mm for Robotiq 2F-85).
        # Verify against the actual USD if needed.
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="left_robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.13]),
        )

        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="right_robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.13]),
        )

        ## [Optional] Gripper control
        # self.actions.left_gripper_action = mdp.BinaryJointVelocityActionCfg(
        #     asset_name="left_robot",
        #     joint_names=["finger_joint"],
        #     open_command_expr={"finger_joint": -0.5},  # rad/s, opening direction
        #     close_command_expr={"finger_joint": 0.5},  # rad/s, closing direction — tune this value
        # )

        # self.actions.right_gripper_action = mdp.BinaryJointVelocityActionCfg(
        #     asset_name="right_robot",
        #     joint_names=["finger_joint"],
        #     open_command_expr={"finger_joint": -0.5},  # rad/s, opening direction
        #     close_command_expr={"finger_joint": 0.5},  # rad/s, closing direction — tune this value
        # )


        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=[
                        DummyRetargeterCfg(
                            enable_visualization=True,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.left_arm_action.joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
