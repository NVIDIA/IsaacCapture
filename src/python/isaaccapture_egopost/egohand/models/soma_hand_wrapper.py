# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""SOMA hand model wrapper for EgoHand.

Thin wrapper around SOMAHandLayer (left + right) with frozen weights
excluded from checkpoints via NonPersistentModuleWrapper.
"""

import os

import torch
import torch.nn as nn

from egohand.models.soma_wrapper import NonPersistentModuleWrapper


def _resolve_hand_data_root():
    """Find assets directory that contains SOMAHand.npz.

    Checks third_party/soma-hand/assets/ first (local repo), then falls
    back to the HuggingFace-cached soma assets directory.
    """
    # Local third_party path (relative to repo root)
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "third_party",
        "soma-hand",
        "assets",
    )
    if os.path.isfile(os.path.join(local_path, "SOMAHand.npz")):
        return local_path

    import soma

    return str(soma.get_assets_dir())


class SomaHandWrapper(nn.Module):
    """Wrapper around SOMAHandLayer for left and right hands.

    SOMA hand weights are excluded from checkpoints via NonPersistentModuleWrapper.
    """

    def __init__(
        self, data_root=None, device="cuda", mode="warp", identity_model_type="soma"
    ):
        super().__init__()
        from soma.hand import SOMAHandLayer
        from soma.units import Unit

        if data_root is None:
            data_root = _resolve_hand_data_root()

        self.identity_model_type = identity_model_type
        self._hands = {}
        for hand_type in ("left", "right"):
            hand = SOMAHandLayer(
                data_root=data_root,
                hand_type=hand_type,
                device=device,
                identity_model_type=identity_model_type,
                mode=mode,
                output_unit=Unit.METERS,
                load_correctives_model=False,
            )
            wrapped = NonPersistentModuleWrapper(hand)
            wrapped.requires_grad_(False)
            setattr(self, f"hand_{hand_type}", wrapped)
            self._hands[hand_type] = hand

        self.left = self._hands["left"]
        self.right = self._hands["right"]

    @property
    def num_shape_components(self):
        return self.left.num_shape_components

    @property
    def num_scale_params(self):
        n = getattr(self.left.identity_model, "num_scale_params", None)
        return n if n is not None else 0

    def faces(self, hand_type):
        """Get face indices for the given hand type."""
        hand = self._hands[hand_type]
        return (
            hand.faces.cpu().numpy()
            if isinstance(hand.faces, torch.Tensor)
            else hand.faces
        )

    def wrist_global_id(self, hand_type):
        """Wrist joint index in 78-joint system."""
        return self._hands[hand_type].wrist_global_id

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        poses,
        identity_coeffs,
        hand_type,
        transl=None,
        pose2rot=True,
        global_scale=1.0,
        scale_params=None,
    ):
        """Run hand FK.

        Args:
            poses: (B, 25, 3) axis-angle or (B, 25, 3, 3) rot matrices.
                Joint 0 = wrist, joints 1-24 = fingers.
            identity_coeffs: (B, K) identity coefficients (K=20 SOMA, K=5 MHR)
            hand_type: 'left' or 'right'
            transl: optional (B, 3) wrist translation in meters
            pose2rot: convert axis-angle to rot matrices if True
            global_scale: uniform scale scalar or (B,) tensor. Default 1.0.
            scale_params: optional (B, S) per-identity scale params (MHR: S=26).
                None for SOMA.
        Returns:
            dict with 'vertices' (B, Vh, 3) and 'joints' (B, 25, 3) in meters
        """
        hand = self._hands[hand_type]
        return hand(
            poses.float(),
            identity_coeffs.float(),
            pose2rot=pose2rot,
            global_translation=transl,
            global_scale=global_scale,
            scale_params=scale_params,
        )

    @torch.no_grad()
    def body_to_hand_params(self, world_rotations, body_joints, hand_type):
        """Convert full-body FK outputs to hand-only parameters.

        Extracts wrist rotation/translation and finger local rotations from
        body world rotations, converting them to the hand model's input space.

        Args:
            world_rotations: (B, 78, 3, 3) per-joint world rotations from
                SomaWrapper.compute_world_rotations()
            body_joints: (B, 77, 3) joint positions from SOMALayer
                (virtual root excluded, so index = global_id - 1)
            hand_type: 'left' or 'right'
        Returns:
            poses: (B, 25, 3, 3) hand rotation matrices
                (joint 0 = wrist global rotation, joints 1-24 = finger local)
            transl: (B, 3) wrist translation in meters
        """
        hand = self._hands[hand_type]

        # Wrist world rotation from body FK
        wrist_rot = world_rotations[:, hand.wrist_global_id]  # (B, 3, 3)
        # Wrist translation from body joints (offset by 1: joints excludes virtual root)
        transl = body_joints[:, hand.wrist_global_id - 1]  # (B, 3)

        # Pre-conjugate wrist rotation to counteract apply_joint_orient_local.
        # The hand layer applies: R_out = orient[0].T @ R_in @ orient[0]
        # We want R_out = wrist_rot (the body's world rotation), so we set
        # R_in = orient[0] @ wrist_rot @ orient[0].T
        orient_wrist = hand.batched_skinning.joint_orient[0]  # (3, 3)
        wrist_rot = (
            orient_wrist[None] @ wrist_rot @ orient_wrist[None].transpose(-2, -1)
        )

        # Extract finger local rotations from body world rotations.
        # body_poses are deformation rotations relative to the BODY's joint_orient;
        # the hand model has its own joint_orient, so we cannot pass them directly.
        finger_gids = hand.hand_joint_ids_global[1:]  # (24,) body global IDs
        hand_parent_local = hand.joint_parent_ids[1:]  # (24,) hand-local parent indices
        parent_gids = [
            hand.hand_joint_ids_global[p] for p in hand_parent_local.tolist()
        ]

        R_world_finger = world_rotations[:, finger_gids]  # (B, 24, 3, 3)
        R_world_parent = world_rotations[:, parent_gids]  # (B, 24, 3, 3)
        R_local = R_world_parent.transpose(-2, -1) @ R_world_finger  # (B, 24, 3, 3)

        # Conjugate with hand T-pose so apply_joint_orient_local recovers R_local
        hand_t_rots = hand.t_pose_local[:, :3, :3]  # (25, 3, 3)
        T_parent = hand_t_rots[hand_parent_local]  # (24, 3, 3)
        T_self = hand_t_rots[1:]  # (24, 3, 3)
        finger_rot = T_parent[None] @ R_local @ T_self[None].transpose(-2, -1)

        # 25 poses: wrist world rotation + finger local rotations
        poses = torch.cat([wrist_rot.unsqueeze(1), finger_rot], dim=1)  # (B, 25, 3, 3)
        return poses, transl

    @torch.amp.autocast("cuda", enabled=False)
    def forward_from_body(
        self,
        world_rotations,
        body_joints,
        hand_type,
        identity_coeffs=None,
        global_scale=1.0,
    ):
        """Convenience: body_to_hand_params + forward in one call."""
        poses, transl = self.body_to_hand_params(
            world_rotations, body_joints, hand_type
        )
        B, device = poses.shape[0], poses.device
        if identity_coeffs is None:
            identity_coeffs = torch.zeros(B, self.num_shape_components, device=device)
        return self.forward(
            poses,
            identity_coeffs,
            hand_type,
            transl=transl,
            pose2rot=False,
            global_scale=global_scale,
        )
