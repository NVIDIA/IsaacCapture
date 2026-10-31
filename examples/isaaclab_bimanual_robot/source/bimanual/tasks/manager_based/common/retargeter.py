from isaaclab.utils.math import quat_mul, quat_inv, quat_apply
import torch

from dataclasses import dataclass

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class BimanualOpenXRRetargeter(RetargeterBase):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self._initialized = False

    def reset(
        self,
        ctrl_left_pos_init: list[float],
        ctrl_left_quat_init: list[float],
        ctrl_right_pos_init: list[float],
        ctrl_right_quat_init: list[float],
    ):
        """Call this once after env.reset(), before the control loop."""

        left_arm_action_term = self.env.action_manager.get_term("left_arm_action")
        ee_pos_curr, ee_quat_curr = left_arm_action_term._compute_frame_pose()

        right_arm_action_term = self.env.action_manager.get_term("right_arm_action")
        ee_pos_curr2, ee_quat_curr2 = right_arm_action_term._compute_frame_pose()

        # --- Controller initial poses (OpenXR world frame) ---
        self.ctrl_left_pos_init = torch.tensor(ctrl_left_pos_init, device=self.device)
        self.ctrl_right_pos_init = torch.tensor(ctrl_right_pos_init, device=self.device)
        self.ctrl_left_quat_init = torch.tensor(ctrl_left_quat_init, device=self.device)  # [w,x,y,z]
        self.ctrl_right_quat_init = torch.tensor(ctrl_right_quat_init, device=self.device)

        # --- Robot initial EE poses (in each robot's base frame) ---
        robot_left = self.env.scene["left_robot"]
        robot_right = self.env.scene["right_robot"]

        # EE frame is tracked by FrameTransformer
        # shape: (N, 3) and (N, 4) — take env 0
        self.ee_left_pos_init, self.ee_left_quat_init = ee_pos_curr[0], ee_quat_curr[0]
        self.ee_right_pos_init, self.ee_right_quat_init = ee_pos_curr2[0], ee_quat_curr2[0]

        # Cache robot base orientation (fixed-base robot, constant after reset)
        self.robot_left_quat_w = robot_left.data.root_quat_w[0]  # (4,) [w,x,y,z]
        self.robot_right_quat_w = robot_right.data.root_quat_w[0]

        self._initialized = True

    def compute_action(
        self,
        ctrl_left_pos_w: torch.Tensor,  # (3,) current left  controller pos in world
        ctrl_left_quat_w: torch.Tensor,  # (4,) [w,x,y,z]
        ctrl_right_pos_w: torch.Tensor,
        ctrl_right_quat_w: torch.Tensor,
        gripper_left: float = None,
        gripper_right: float = None,
        scale: float = 1.0,
        reverse_yz: bool = True,
    ) -> torch.Tensor:
        """Returns action tensor of shape (1, 14) = [left_7D, right_7D]."""

        has_gripper = gripper_left is not None and gripper_right is not None

        left = self._delta_action(
            ctrl_left_pos_w,
            ctrl_left_quat_w,
            self.ctrl_left_pos_init,
            self.ctrl_left_quat_init,
            self.ee_left_pos_init,
            self.ee_left_quat_init,
            self.robot_left_quat_w,
            scale,
            reverse_yz,
        )
        right = self._delta_action(
            ctrl_right_pos_w,
            ctrl_right_quat_w,
            self.ctrl_right_pos_init,
            self.ctrl_right_quat_init,
            self.ee_right_pos_init,
            self.ee_right_quat_init,
            self.robot_right_quat_w,
            scale,
            reverse_yz,
        )

        if has_gripper:
            return torch.cat(
                [
                    left,
                    torch.tensor([gripper_left], device=self.device),
                    right,
                    torch.tensor([gripper_right], device=self.device),
                ]
            ).unsqueeze(0)
        else:
            return torch.cat([left, right]).unsqueeze(0)

    def _delta_action(
        self,
        ctrl_pos_w: torch.Tensor,  # current controller pos in world
        ctrl_quat_w: torch.Tensor,  # current controller quat in world [w,x,y,z]
        ctrl_pos_init: torch.Tensor,
        ctrl_quat_init: torch.Tensor,
        ee_pos_init_b: torch.Tensor,  # robot EE initial pos in base frame
        ee_quat_init_b: torch.Tensor,  # robot EE initial quat in base frame
        robot_quat_w: torch.Tensor,  # robot base orientation in world [w,x,y,z]
        scale: float = 1.0, # scale factor for the delta position
        reverse_yz: bool = True, # swap y and z dimensions of delta_quat [w, x, y, z] -> swap indices 2 and 3, negate Z
    ) -> torch.Tensor:
        # 1. Position delta in world frame
        delta_pos_w = ctrl_pos_w - ctrl_pos_init  # (3,)

        # 2. Rotate delta into robot base frame
        #    (robot_quat_w rotates world→base via its inverse)
        delta_pos_b = quat_apply(quat_inv(robot_quat_w.unsqueeze(0)), delta_pos_w.unsqueeze(0)).squeeze(0)

        # 3. Desired EE position = initial + scaled delta
        desired_pos_b = ee_pos_init_b + delta_pos_b * scale

        # 4. Rotation delta: how much the controller rotated from its initial pose
        delta_quat = quat_mul(quat_inv(ctrl_quat_init.unsqueeze(0)), ctrl_quat_w.unsqueeze(0)).squeeze(0)

        # 5. Apply rotation delta to initial EE orientation
        # Interchange Y and Z dimensions of delta_quat [w, x, y, z] -> swap indices 2 and 3, negate Z
        if reverse_yz:
            delta_quat = delta_quat * torch.tensor([1.0, -1.0, 1.0, 1.0], device=delta_quat.device)
            delta_quat = delta_quat[[0, 1, 3, 2]]
        else: # reverse_xy
            delta_quat = delta_quat * torch.tensor([1.0, 1.0, -1.0, 1.0], device=delta_quat.device)
            delta_quat = delta_quat[[0, 2, 1, 3]]
        desired_quat_b = quat_mul(delta_quat.unsqueeze(0), ee_quat_init_b.unsqueeze(0)).squeeze(0)

        return torch.cat([desired_pos_b, desired_quat_b])  # (7,)

    def retarget(
        self,
        ctrl_left_pos_w,
        ctrl_left_quat_w,
        ctrl_right_pos_w,
        ctrl_right_quat_w,
        gripper_left,
        gripper_right,
        scale=1.0,
    ):
        return self.compute_action(
            ctrl_left_pos_w, ctrl_left_quat_w, ctrl_right_pos_w, ctrl_right_quat_w, gripper_left, gripper_right, scale
        )


@dataclass
class BimanualOpenXRRetargeterCfg(RetargeterCfg):
    """Configuration for the bimanual UR5E OpenXR retargeter."""

    enable_visualization: bool = False
    retargeter_type: type[RetargeterBase] = BimanualOpenXRRetargeter
