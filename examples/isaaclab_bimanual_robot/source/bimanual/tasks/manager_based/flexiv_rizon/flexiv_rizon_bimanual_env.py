# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp


from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from .ik.task_space_actions import FlexivRizonDifferentialInverseKinematicsActionCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.sensors import CameraCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils

##
# Pre-defined configs
##
from .flexiv_rizon_robots import FLEXIV_RIZON_4S_CFG  # isort: skip
from bimanual.tasks.manager_based.common.base_env_cfg import BaseEnvCfg
from bimanual.tasks.manager_based.common.dummy_retargeter import DummyRetargeterCfg

from .ik.differential_ik_cfg import FlexivRizonDifferentialIKControllerCfg

##
# Environment configuration
##


@configclass
class FlexivRizonBimanualEnvCfg(BaseEnvCfg):
    def __init__(self):
        super().__init__()

        self.scene.table.init_state = AssetBaseCfg.InitialStateCfg(
            pos=[0.70, 0.5, 0.82],
            rot=[0.0, 0.0, 0.0, -1.0]
        )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.left_robot = FLEXIV_RIZON_4S_CFG.replace(
            prim_path="{ENV_REGEX_NS}/left_robot",
            init_state=FLEXIV_RIZON_4S_CFG.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                rot=(0.7071, 0, 0.7071, 0.0),
                joint_pos={
                    "joint1": 0.0,
                    "joint2": 1.5708,
                    "joint3": 0.0,
                    "joint4": 1.5708,
                    "joint5": 0.0,
                    "joint6": 0.0,
                    "joint7": 0.0,
                }
            ),
        )
        self.scene.right_robot = FLEXIV_RIZON_4S_CFG.replace(
            prim_path="{ENV_REGEX_NS}/right_robot",
            init_state=FLEXIV_RIZON_4S_CFG.InitialStateCfg(
                pos=(0.0, 0.8, 1.0),
                rot=(0.7071, 0, 0.7071, 0.0),
                joint_pos={
                    "joint1": 0.0,
                    "joint2": 1.5708,
                    "joint3": 0.0,
                    "joint4": 1.5708,
                    "joint5": 0.0,
                    "joint6": 0.0,
                    "joint7": 0.0,
                }
            ),
        )

        # Replace joint-position arm action with differential IK (relative pose)
        # NOTE: body_offset.pos is the TCP offset from ee_link along its z-axis (~130 mm for Robotiq 2F-85).
        # Verify against the actual USD if needed.
        self.actions.left_arm_action = FlexivRizonDifferentialInverseKinematicsActionCfg(
            asset_name="left_robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            body_name="gripper_base",
            controller=FlexivRizonDifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
                joint_weights=torch.tensor([10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            ),
            scale=1.0,
            body_offset=FlexivRizonDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        )

        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="right_robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            body_name="gripper_base",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.2]),
        )

        # [Optional] Gripper control
        self.actions.left_gripper_action = mdp.BinaryJointVelocityActionCfg(
            asset_name="left_robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.1},  # rad/s, opening direction
            close_command_expr={"finger_joint": -0.1},  # rad/s, closing direction — tune this value
        )

        self.actions.right_gripper_action = mdp.BinaryJointVelocityActionCfg(
            asset_name="right_robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.1},  # rad/s, opening direction
            close_command_expr={"finger_joint": -0.1},  # rad/s, closing direction — tune this value
        )


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
                    xr_cfg= XrCfg(
                        anchor_pos = (1.5, 0.4, -0.2),
                        anchor_rot = (0, 0.7071, 0, 0.7071)
                    ),
                ),
            }
        )
