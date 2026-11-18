# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""SOMA body model wrapper for EgoHand.

Handles rotation matrix -> axis-angle conversion before calling SOMA FK.
"""

import torch
import torch.nn as nn

from egohand.utils.geometry import matrix_to_axis_angle


class NonPersistentModuleWrapper(nn.Module):
    """Wrap a module but drop all of its state_dict entries (SOMA weights excluded from checkpoints)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def load_state_dict(self, state_dict, strict=True):
        return torch.nn.modules.module._IncompatibleKeys([], [])


class SomaWrapper(nn.Module):
    """Wrapper around SOMA body model for EgoHand.

    Converts rotation matrices from the network to axis-angle for SOMA FK.
    SOMA weights are excluded from checkpoints via NonPersistentModuleWrapper.

    Supports two identity model types:
    - "mhr" (default): 45D identity_coeffs + 69D scale_params (global_scale + bone_scale)
    - "smpl": 10D identity_coeffs (SMPL betas), no scale_params
    """

    def __init__(
        self,
        data_root=None,
        low_lod=False,
        device="cuda",
        identity_model_type="mhr",
        mode="warp",
        identity_model_kwargs=None,
    ):
        super().__init__()
        from soma import SOMALayer as _SOMALayer
        from soma import Unit

        self.identity_model_type = identity_model_type

        # Resolve data_root from installed package if not explicitly provided
        if data_root is None or data_root == "inputs/soma_assets":
            import soma

            data_root = str(soma.get_assets_dir())

        soma_kwargs = dict(
            data_root=data_root,
            low_lod=low_lod,
            device=device,
            identity_model_type=identity_model_type,
            mode=mode,
            output_unit=Unit.METERS,
            enable_procedural_transforms=False,
        )
        if identity_model_kwargs:
            soma_kwargs["identity_model_kwargs"] = identity_model_kwargs

        soma = _SOMALayer(**soma_kwargs)
        self.soma = NonPersistentModuleWrapper(soma)
        # Freeze SOMA body model — these are fixed FK weights, not learned.
        # Critical for DDP: without this, DDP tries to allreduce gradients
        # for 268M SOMA params whose warp-kernel backward confuses
        # find_unused_parameters traversal, causing a multi-GPU deadlock.
        self.soma.requires_grad_(False)
        self.faces = soma.faces
        self.parents = soma.parents

    def forward(
        self,
        body_pose_rotmat,
        global_orient_rotmat,
        identity_coeffs,
        scale_params=None,
        transl=None,
    ):
        """Run SOMA forward kinematics from rotation matrices.

        Args:
            body_pose_rotmat: (B, 76, 3, 3) rotation matrices
            global_orient_rotmat: (B, 1, 3, 3) rotation matrices
            identity_coeffs: (B, 45) for MHR or (B, 10) for SMPL
            scale_params: (B, 69) for MHR, None for SMPL
            transl: (B, 3) optional translation
        Returns:
            dict with 'vertices' (B, V, 3) and 'joints' (B, 77, 3)
        """
        B = body_pose_rotmat.shape[0]

        # Convert rotation matrices to axis-angle for SOMA
        body_pose_aa = matrix_to_axis_angle(body_pose_rotmat)  # (B, 76, 3)
        global_orient_aa = matrix_to_axis_angle(
            global_orient_rotmat.squeeze(1)
        )  # (B, 3)

        # Concatenate into full pose (B, 77, 3)
        poses = torch.cat(
            [global_orient_aa[:, None], body_pose_aa], dim=1
        )  # (B, 77, 3)

        if transl is None:
            transl = torch.zeros(B, 3, device=poses.device, dtype=poses.dtype)

        return self.static_forward(poses, identity_coeffs, scale_params, transl)

    @torch.amp.autocast("cuda", enabled=False)
    def static_forward(
        self,
        poses,
        identity_coeffs,
        scale_params=None,
        transl=None,
        pose2rot=True,
        return_joints_only=False,
    ):
        """Single-frame SOMA forward kinematics.

        Args:
            poses: (B, 77, 3) axis-angle
            identity_coeffs: (B, 45) for MHR or (B, 10) for SMPL
            scale_params: (B, 69) for MHR, None for SMPL
            transl: (B, 3)
            pose2rot: whether to convert axis-angle to rotation matrices
        Returns:
            dict with 'vertices' (B, V, 3) and 'joints' (B, 77, 3)
        """
        # Force float32: SOMA sparse skinning ops don't support half precision
        poses = poses.float()
        identity_coeffs = identity_coeffs.float()
        if transl is None:
            transl = torch.zeros(
                poses.shape[0], 3, device=poses.device, dtype=poses.dtype
            )
        transl = transl.float()
        soma = self.soma.module

        if self.identity_model_type == "smpl":
            # SMPL identity: just betas. repose_to_bind_pose=True (default)
            # matches the official SOMA reference tools/smpl2soma_hard_samples.py
            # and the SOMA library docstring. Must match the value used during
            # conversion in scripts/preprocessing/convert_hmr2_to_soma.py.
            soma.prepare_identity(identity_coeffs)
        else:
            # MHR identity: decompose scale_params into global_scale + bone_scale.
            # repose_to_bind_pose=False is correct here — per the SOMA docstring,
            # use False when the target vertices are posed relative to the
            # identity model's native rest pose (MHR's case).
            scale_params = scale_params.float()
            global_scale = scale_params[:, :1]
            bone_scale = scale_params[:, 1:]
            soma.prepare_identity(
                identity_coeffs,
                bone_scale,
                repose_to_bind_pose=False,
                global_scale=global_scale,
            )

        out = soma.pose(poses, transl=transl, pose2rot=pose2rot)
        return {
            "vertices": out["vertices"] if not return_joints_only else None,
            "joints": out["joints"],
        }

    @torch.amp.autocast("cuda", enabled=False)
    def compute_world_rotations(self, poses, pose2rot=True):
        """Compute per-joint world rotations via rotation-only FK (no skinning).

        Uses the body model's joint hierarchy and joint orient to produce world
        rotation matrices.  Useful for extracting wrist/finger orientations
        when driving the hand model from body poses.

        Args:
            poses: (B, 77, 3) axis-angle or (B, 77, 3, 3) rotation matrices
            pose2rot: convert axis-angle to rotation matrices if True
        Returns:
            (B, 78, 3, 3) world rotation matrices for all 78 body joints
        """
        from soma.geometry.lbs import batch_rodrigues
        from soma.geometry.rig_utils import (
            apply_joint_orient_local,
            joint_local_to_world_levelorder,
        )

        poses = poses.float()
        B = poses.shape[0]
        soma = self.soma.module

        if pose2rot:
            poses_rot = batch_rodrigues(poses.reshape(-1, 3)).reshape(B, 77, 3, 3)
        else:
            poses_rot = poses.reshape(B, 77, 3, 3)

        # Pad with identity for virtual root (joint 0)
        poses_rot = soma._pad_poses(poses_rot)  # (B, 78, 3, 3)

        # Apply joint orient (same transform as batched_skinning.pose)
        bs = soma.batched_skinning
        if bs.joint_orient is not None:
            poses_rot = apply_joint_orient_local(
                poses_rot, bs.joint_orient, bs._orient_parent_T
            )

        # Rotation-only FK using precomputed level-order traversal
        return joint_local_to_world_levelorder(poses_rot, bs._levels)
