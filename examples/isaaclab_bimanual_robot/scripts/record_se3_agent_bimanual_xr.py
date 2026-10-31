# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers)."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher
import time
import numpy as np
import os


# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--enable_gripper", action="store_true", default=False, help="Enable gripper control.")
parser.add_argument("--reverse_rotation_yz", action="store_true", default=False, help="Reverse rotation yz.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging

import gymnasium as gym
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import bimanual.tasks  # noqa: F401
from bimanual.tasks.manager_based.common.retargeter import BimanualOpenXRRetargeter

from isaaclab.devices.device_base import DeviceBase

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode


# import logger
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # /World/envs/env_.*/berkeley_table/BerkeleyRobotTable/overhead_camera.
    if args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # set up dataset recording
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    # no success conditions for now
    env_cfg.terminations.success = None

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure the teleop device in the environment config.")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    # render once
    env.sim.render()

    # init controller
    retargeter = BimanualOpenXRRetargeter(env, env.sim.device)

    print("Teleoperation started.")

    start_button_pressed = False
    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                raw_data = teleop_interface._get_raw_data()
                left_controller_data = np.array(raw_data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([])))
                right_controller_data = np.array(raw_data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([])))
                # robot root pose from observation/scene
                logger.debug("[time] %s", time.time())
                logger.debug("[left_controller_data]: %s", left_controller_data)
                logger.debug("[right_controller_data]: %s", right_controller_data)

                if left_controller_data.shape[0] == 0 or right_controller_data.shape[0] == 0:
                    env.sim.render()
                    continue

                logger.debug("[left_controller_data] buttons: %s", np.sum(left_controller_data[1]))
                logger.debug("[right_controller_data] buttons: %s", np.sum(right_controller_data[1]))

                # add an initialization condition; press left trigger to start
                if np.sum(left_controller_data[1]) > 0:
                    start_button_pressed = True

                if not start_button_pressed:
                    env.sim.render()
                    continue

                if not retargeter._initialized:
                    ctrl_left_pos_init = left_controller_data[0, 0:3]
                    ctrl_left_quat_init = left_controller_data[0, 3:7]
                    ctrl_right_pos_init = right_controller_data[0, 0:3]
                    ctrl_right_quat_init = right_controller_data[0, 3:7]
                    retargeter.reset(
                        ctrl_left_pos_init,
                        ctrl_left_quat_init,
                        ctrl_right_pos_init,
                        ctrl_right_quat_init,
                    )

                ctrl_left_pos_cur = torch.tensor(left_controller_data[0, 0:3], device=env.sim.device)
                ctrl_left_quat_cur = torch.tensor(left_controller_data[0, 3:7], device=env.sim.device)
                ctrl_right_pos_cur = torch.tensor(right_controller_data[0, 0:3], device=env.sim.device)
                ctrl_right_quat_cur = torch.tensor(right_controller_data[0, 3:7], device=env.sim.device)

                # obtain gripper commands
                if args_cli.enable_gripper:
                    gripper_left = 1.0 if np.sum(left_controller_data[1]) > 0 else -1.0
                    gripper_right = 1.0 if np.sum(right_controller_data[1]) > 0 else -1.0
                else:
                    gripper_left = None
                    gripper_right = None

                compute_action = retargeter.compute_action(
                    ctrl_left_pos_cur,
                    ctrl_left_quat_cur,
                    ctrl_right_pos_cur,
                    ctrl_right_quat_cur,
                    gripper_left=gripper_left,
                    gripper_right=gripper_right,
                    scale=args_cli.sensitivity,
                    reverse_yz=args_cli.reverse_rotation_yz,
                )

                obs, _, _, _, _ = env.step(compute_action)

                # press x or y to reset
                successful = False
                if np.sum(left_controller_data[1][4:6]) > 0:
                    should_reset_recording_instance = True

                # press a or b to reset env and save recording
                if np.sum(right_controller_data[1][4:6]) > 0:
                    should_reset_recording_instance = True
                    successful = True

                if should_reset_recording_instance:
                    if successful:
                        env.recorder_manager.export_episodes(env_ids=[0])  # export the current episode before resetting

                    env.sim.reset()
                    env.recorder_manager.reset()
                    env.reset()
                    teleop_interface.reset()

                    env.sim.render()

                    start_button_pressed = False
                    retargeter._initialized = False

                    should_reset_recording_instance = False
                    print("Environment reset complete")

        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
