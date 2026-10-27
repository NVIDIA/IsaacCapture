# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""EgoHand: GEP2-style training contract for SOMA hand regression.

This module keeps hand-specific data/model pieces (SOMA-hand FK, 25 joints,
right-hand-to-left canonicalization in the dataset) while matching the body
GEP2 training recipe where it matters:
- Hydra-instantiated backbone, including DINOv3 + sam-3d-body checkpoint.
- Full 256x256 DINOv3 crop via ``backbone_input_crop_w: null``.
- Weak-perspective camera path by default, with an opt-in calibrated
  ``sam3db_perspective`` path that projects through the dataset's full-image K.
- Sum-reduction HMR2-style keypoint / parameter / camera losses.
- Differential backbone/head learning rates.

The MHR hand shape vector follows the body convention: ``scale_params`` is a
single vector whose dim 0 is global scale and whose remaining dims are the
SOMA-hand MHR segment scales. It is split only at the FK API boundary.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from egohand.models.heads.rletr_occ_head import (
    RLEOccReadout,
    RLETransformerOccHead,
    RLEVisualMemory,
)
from egohand.models.heads.soma_hand_head_v2 import (
    SOMAHandTransformerDecoderHeadV2,
)
from egohand.models.rletr_occ_loss import RLEOccOutCoordLoss
from egohand.models.soma_hand_wrapper import SomaHandWrapper
from egohand.models.utils import zero_module
from egohand.utils.cam_utils import (
    canonical_camera_depth_mode,
    compute_transl_full_cam,
    depth_parameter_to_legacy_camera,
    effective_camera_depth_parameter,
)
from egohand.utils.cam_utils import (
    perspective_projection as cam_perspective_projection,
)
from egohand.utils.fixed_depth_translation import (
    camera_scale_to_translation_z,
    crop_keypoints_to_full_image,
    solve_lateral_translation_full_perspective,
)
from egohand.utils.geometry import perspective_projection
from egohand.utils.pylogger import Log
from egohand.utils.ray_pe import compute_crop_ray_angles, fourier_encode_angles

_MAX_TORCH_SEED = 2**63 - 1


def _seeded_kp2d_module(factory, *, base_seed: int, salt: int):
    """Build one branch module without advancing the caller's RNG state."""
    seed = (int(base_seed) + int(salt)) % _MAX_TORCH_SEED
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return factory()


def _build_kp2d_branch_modules(
    *,
    topology: str,
    in_channels: int,
    num_joints: int,
    d_model: int,
    depth: int,
    n_heads: int,
    feat_hw: tuple[int, int],
    hand_token_dim: int,
    negative_presence_weight: float | None,
    base_seed: int,
) -> dict[str, nn.Module | None]:
    """Build paired topology variants with identical shared initialization."""
    visual_memory = _seeded_kp2d_module(
        lambda: RLEVisualMemory(
            in_channels=in_channels,
            d_model=d_model,
            feat_hw=feat_hw,
        ),
        base_seed=base_seed,
        salt=101,
    )
    readout = _seeded_kp2d_module(
        lambda: RLEOccReadout(
            d_model=d_model,
            num_joints=num_joints,
        ),
        base_seed=base_seed,
        salt=102,
    )
    outcoord_loss = _seeded_kp2d_module(
        lambda: RLEOccOutCoordLoss(
            feat_hw=feat_hw,
            crop_bounds=(-0.5, 0.5),
            negative_presence_weight=negative_presence_weight,
        ),
        base_seed=base_seed,
        salt=103,
    )

    result = {
        "kp2d_head": None,
        "kp2d_visual_memory": None,
        "kp2d_token_adapter": None,
        "kp2d_readout": None,
        "kp2d_outcoord_loss": outcoord_loss,
    }
    if topology == "separate":
        result["kp2d_head"] = _seeded_kp2d_module(
            lambda: RLETransformerOccHead(
                in_channels=in_channels,
                num_joints=num_joints,
                d_model=d_model,
                depth=depth,
                n_heads=n_heads,
                feat_hw=feat_hw,
                visual_memory=visual_memory,
                readout=readout,
            ),
            base_seed=base_seed,
            salt=104,
        )
    elif topology == "merged":
        result["kp2d_visual_memory"] = visual_memory
        result["kp2d_token_adapter"] = _seeded_kp2d_module(
            lambda: nn.Linear(hand_token_dim, d_model),
            base_seed=base_seed,
            salt=104,
        )
        result["kp2d_readout"] = readout
    else:
        raise ValueError(f"unknown kp2d topology: {topology!r}")
    return result


class EgoHand(nn.Module):
    """GEP2-style SOMA hand parameter regression."""

    full_perspective_modes = ("perspective", "sam3db_perspective")

    def __init__(self, cfg: DictConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.freeze_backbone = bool(cfg.get("freeze_backbone", False))

        backbone_cfg = cfg.get("backbone", None)
        self.backbone = instantiate(backbone_cfg)

        backbone_weights = cfg.get("backbone_weights", None)
        if backbone_weights is not None:
            if hasattr(self.backbone, "load_pretrained"):
                self.backbone.load_pretrained(backbone_weights)
            else:
                Log.info(f"Loading backbone weights from {backbone_weights}")
                self.backbone.load_state_dict(
                    torch.load(backbone_weights, map_location="cpu")["state_dict"]
                )
        if self.freeze_backbone:
            self._freeze_backbone()

        crop_w = cfg.get("backbone_input_crop_w", [32, -32])
        if crop_w is None:
            self._backbone_crop_w = None
        else:
            self._backbone_crop_w = (int(crop_w[0]), int(crop_w[1]))

        self.focal_length = float(cfg.get("focal_length", 5000.0))
        self.image_size = int(cfg.get("image_size", 256))
        self.camera_mode = cfg.get("camera_mode", "weak_perspective")
        valid_camera_modes = ("weak_perspective", *self.full_perspective_modes)
        if self.camera_mode not in valid_camera_modes:
            raise ValueError(
                f"Unknown camera_mode={self.camera_mode!r}; "
                f"expected one of {valid_camera_modes}"
            )
        self.camera_depth_mode = canonical_camera_depth_mode(
            cfg.get("camera_depth_mode", "s")
        )
        if self.camera_depth_mode != "s" and self.camera_mode != "sam3db_perspective":
            raise ValueError(
                "log-depth camera modes currently require "
                "camera_mode='sam3db_perspective' so forward decoding, "
                "training targets, callbacks, and inference all use the "
                "same calibrated K_fullimg; got "
                f"camera_mode={self.camera_mode!r}"
            )
        camera_log_value_bounds = cfg.get(
            "camera_log_value_bounds",
            [-10.0, 10.0],
        )
        if len(camera_log_value_bounds) != 2:
            raise ValueError("camera_log_value_bounds must contain [minimum, maximum]")
        self.camera_log_value_bounds = (
            float(camera_log_value_bounds[0]),
            float(camera_log_value_bounds[1]),
        )
        if (
            not all(math.isfinite(value) for value in self.camera_log_value_bounds)
            or self.camera_log_value_bounds[0] >= self.camera_log_value_bounds[1]
        ):
            raise ValueError(
                "camera_log_value_bounds must contain two finite increasing values"
            )
        self.camera_depth_std = float(cfg.get("camera_depth_std", 1.0))
        if not math.isfinite(self.camera_depth_std) or self.camera_depth_std <= 0:
            raise ValueError("camera_depth_std must be finite and positive")
        self.camera_depth_loss_type = str(
            cfg.get("camera_depth_loss_type", "smooth_l1")
        )
        if self.camera_depth_loss_type not in {"smooth_l1", "mse"}:
            raise ValueError("camera_depth_loss_type must be 'smooth_l1' or 'mse'")
        # Camera parameterization and scalar optimization are deliberately
        # separate controls, each with its own default loss to keep prior
        # configurations bit-for-bit compatible while allowing a matched
        # s ablation.
        default_scalar_loss_type = (
            "legacy_mse"
            if self.camera_depth_mode == "s"
            else self.camera_depth_loss_type
        )
        self.camera_scalar_loss_type = str(
            cfg.get(
                "camera_scalar_loss_type",
                default_scalar_loss_type,
            )
        )
        if self.camera_scalar_loss_type not in {
            "legacy_mse",
            "smooth_l1",
            "mse",
        }:
            raise ValueError(
                "camera_scalar_loss_type must be 'legacy_mse', 'smooth_l1', or 'mse'"
            )
        self.camera_scalar_loss_std = float(
            cfg.get("camera_scalar_loss_std", self.camera_depth_std)
        )
        if (
            not math.isfinite(self.camera_scalar_loss_std)
            or self.camera_scalar_loss_std <= 0
        ):
            raise ValueError("camera_scalar_loss_std must be finite and positive")
        self.camera_scalar_delta_scaled = bool(
            cfg.get(
                "camera_scalar_delta_scaled",
                self.camera_depth_mode != "s",
            )
        )
        self.cliff_condition = bool(cfg.get("cliff_condition", False))
        self.ray_condition = bool(cfg.get("ray_condition", False))
        self.ray_inject = str(cfg.get("ray_inject", "head"))
        self.ray_pe_num_bands = int(cfg.get("ray_pe_num_bands", 8))
        if self.ray_condition:
            if self.ray_inject != "head":
                raise ValueError(
                    "The first EgoHand Ray PE variant supports "
                    f"ray_inject='head'; got {self.ray_inject!r}"
                )
            if self.ray_pe_num_bands <= 0:
                raise ValueError("ray_pe_num_bands must be positive")
            if self._backbone_crop_w is not None:
                raise ValueError(
                    "EgoHand Ray PE requires the backbone to consume the "
                    "full crop (backbone_input_crop_w=null)"
                )
        self.focal_length_mode = cfg.get("focal_length_mode", "fixed")
        self.root_joint_ids = tuple(cfg.get("root_joint_ids", [0]))

        # Optional direct-2D-to-translation path. It is intentionally limited
        # to calibrated-camera inference: s predicts tz, while direct
        # crop-space 2D joints and SOMA FK joints solve tx/ty using the
        # complete per-sample K_fullimg.
        self.kp2d_translation_cfg = dict(cfg.get("kp2d_translation", {}))
        self.use_kp2d_translation = bool(
            self.kp2d_translation_cfg.get("enabled", False)
        )
        self.kp2d_topology = self.kp2d_translation_cfg.get(
            "topology",
            "separate",
        )
        self.kp2d_num_joints = int(self.kp2d_translation_cfg.get("num_joints", 25))
        self.kp2d_model_dim = int(self.kp2d_translation_cfg.get("d_model", 256))
        self.kp2d_feat_hw = tuple(
            int(value)
            for value in self.kp2d_translation_cfg.get(
                "feat_hw",
                [16, 16],
            )
        )
        self.kp2d_presence_threshold = float(
            self.kp2d_translation_cfg.get(
                "presence_threshold",
                0.5,
            )
        )
        negative_presence_weight = self.kp2d_translation_cfg.get(
            "negative_presence_weight",
            1.0,
        )
        self.kp2d_negative_presence_weight = (
            None
            if negative_presence_weight is None
            else float(negative_presence_weight)
        )
        self.kp2d_min_points = int(self.kp2d_translation_cfg.get("min_points", 4))
        self.kp2d_min_depth = float(self.kp2d_translation_cfg.get("min_depth", 1e-4))
        self.kp2d_solver_damping = float(self.kp2d_translation_cfg.get("damping", 1e-6))
        self.kp2d_max_condition_number = float(
            self.kp2d_translation_cfg.get(
                "max_condition_number",
                1e6,
            )
        )
        # Emit a detailed source diagnostic once per process if a projected
        # GT joint is non-finite.  The loss still masks that individual joint,
        # but the warning identifies the first upstream tensor that went bad.
        self._warned_nonfinite_gt_projection = False
        if self.use_kp2d_translation:
            if self.camera_mode != "sam3db_perspective":
                raise ValueError(
                    "kp2d_translation requires "
                    "camera_mode='sam3db_perspective' with a real "
                    "K_fullimg"
                )
            if self.kp2d_topology not in {"separate", "merged"}:
                raise ValueError(
                    "kp2d_translation.topology must be 'separate' or "
                    f"'merged', got {self.kp2d_topology!r}"
                )
            if self.kp2d_num_joints != 25:
                raise ValueError(
                    "SOMA-hand translation solving requires exactly 25 "
                    "joint correspondences"
                )
            if len(self.kp2d_feat_hw) != 2 or any(
                value <= 0 for value in self.kp2d_feat_hw
            ):
                raise ValueError(
                    "kp2d_translation.feat_hw must contain two positive integers"
                )
            if not 0.0 <= self.kp2d_presence_threshold <= 1.0:
                raise ValueError(
                    "kp2d_translation.presence_threshold must be in [0, 1]"
                )
            if self.kp2d_negative_presence_weight is not None and (
                not math.isfinite(self.kp2d_negative_presence_weight)
                or self.kp2d_negative_presence_weight < 0
            ):
                raise ValueError(
                    "kp2d_translation.negative_presence_weight must be "
                    "finite and non-negative, or null"
                )
            if self.kp2d_min_points < 1:
                raise ValueError("kp2d_translation.min_points must be at least 1")
            if self.kp2d_min_depth < 0:
                raise ValueError("kp2d_translation.min_depth must be non-negative")
            if self.kp2d_solver_damping < 0:
                raise ValueError("kp2d_translation.damping must be non-negative")
            if self.kp2d_max_condition_number < 1:
                raise ValueError(
                    "kp2d_translation.max_condition_number must be at least 1"
                )

        head_cfg = dict(cfg.get("soma_hand_head", {}))
        if self.cliff_condition:
            head_cfg["cliff_cond_dim"] = 3
        if self.ray_condition:
            head_cfg["ray_cond_dim"] = 4 * self.ray_pe_num_bands
        configured_head_depth_mode = canonical_camera_depth_mode(
            head_cfg.get(
                "camera_depth_mode",
                self.camera_depth_mode,
            )
        )
        if configured_head_depth_mode != self.camera_depth_mode:
            raise ValueError(
                "model.camera_depth_mode and "
                "model.soma_hand_head.camera_depth_mode must match, got "
                f"{self.camera_depth_mode!r} and "
                f"{configured_head_depth_mode!r}"
            )
        head_cfg["camera_depth_mode"] = self.camera_depth_mode
        configured_delta_scaled = bool(
            head_cfg.get(
                "camera_scalar_delta_scaled",
                self.camera_scalar_delta_scaled,
            )
        )
        if configured_delta_scaled != self.camera_scalar_delta_scaled:
            raise ValueError(
                "model.camera_scalar_delta_scaled and "
                "model.soma_hand_head.camera_scalar_delta_scaled must "
                f"match, got {self.camera_scalar_delta_scaled} and "
                f"{configured_delta_scaled}"
            )
        head_cfg["camera_scalar_delta_scaled"] = configured_delta_scaled
        configured_camera_scalar_std = float(
            head_cfg.get(
                "camera_scalar_std",
                self.camera_scalar_loss_std,
            )
        )
        if self.camera_scalar_delta_scaled and not math.isclose(
            configured_camera_scalar_std,
            self.camera_scalar_loss_std,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "model.camera_scalar_loss_std and "
                "model.soma_hand_head.camera_scalar_std must match when "
                "camera scalar decoder deltas are scaled, got "
                f"{self.camera_scalar_loss_std} and "
                f"{configured_camera_scalar_std}"
            )
        head_cfg["camera_scalar_std"] = configured_camera_scalar_std
        configured_kp2d_queries = int(head_cfg.get("num_kp2d_queries", 0))
        expected_kp2d_queries = (
            self.kp2d_num_joints
            if self.use_kp2d_translation and self.kp2d_topology == "merged"
            else 0
        )
        if configured_kp2d_queries != expected_kp2d_queries:
            raise ValueError(
                "soma_hand_head.num_kp2d_queries must be "
                f"{expected_kp2d_queries} for kp2d topology "
                f"{self.kp2d_topology!r}, got "
                f"{configured_kp2d_queries}"
            )
        if "_target_" in head_cfg:
            self.hand_head = instantiate(head_cfg)
        else:
            self.hand_head = SOMAHandTransformerDecoderHeadV2(**head_cfg)

        self.kp2d_head = None
        self.kp2d_visual_memory = None
        self.kp2d_token_adapter = None
        self.kp2d_readout = None
        self.kp2d_outcoord_loss = None
        if self.use_kp2d_translation:
            backbone_feature_dim = int(cfg.get("backbone_feature_dim", 1280))
            kp2d_modules = _build_kp2d_branch_modules(
                topology=self.kp2d_topology,
                in_channels=backbone_feature_dim,
                num_joints=self.kp2d_num_joints,
                d_model=self.kp2d_model_dim,
                depth=int(
                    self.kp2d_translation_cfg.get(
                        "decoder_depth",
                        3,
                    )
                ),
                n_heads=int(self.kp2d_translation_cfg.get("num_heads", 8)),
                feat_hw=self.kp2d_feat_hw,
                hand_token_dim=int(head_cfg.get("transformer_dim", 1024)),
                negative_presence_weight=(self.kp2d_negative_presence_weight),
                base_seed=torch.initial_seed(),
            )
            for module_name, module in kp2d_modules.items():
                setattr(self, module_name, module)

        soma_hand_cfg = cfg.get("soma_hand", {})
        self.soma_hand_identity = soma_hand_cfg.get("identity_model_type", "mhr")
        self.soma_hand = SomaHandWrapper(
            data_root=soma_hand_cfg.get("data_root", None),
            device="cuda",
            identity_model_type=self.soma_hand_identity,
        )

        # Optional depth modality: per-modality encoder +
        # ``zero_module``-initialized fusion projection. When
        # ``cfg.depth_encoder`` is absent the model is unchanged from the
        # depth-free baseline; when present, a brand-new model loaded from a
        # depth-free checkpoint with ``strict=False`` produces identical
        # step-0 predictions because the fuser is zero.
        depth_encoder_cfg = cfg.get("depth_encoder", None)
        if depth_encoder_cfg is not None:
            self.depth_encoder = instantiate(depth_encoder_cfg)
            backbone_dim = int(cfg.get("backbone_feature_dim", 1280))
            depth_out_dim = int(depth_encoder_cfg.get("out_dim", backbone_dim))
            self.depth_fuser = zero_module(
                nn.Conv2d(depth_out_dim, backbone_dim, kernel_size=1)
            )
            self.use_depth = True
            # cfg-style drop: randomly zero the depth contribution during
            # training so the model is robust to missing depth at inference.
            # Set to 0.0 if depth is guaranteed to be available downstream.
            self.depth_drop_prob = float(cfg.get("depth_drop_prob", 0.0))
            Log.info(
                f"[EgoHand] depth modality enabled "
                f"(encoder out_dim={depth_out_dim}, fuser zero-init, "
                f"depth_drop_prob={self.depth_drop_prob})"
            )
        else:
            self.depth_encoder = None
            self.depth_fuser = None
            self.use_depth = False
            self.depth_drop_prob = 0.0

    def _freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        num_params = sum(p.numel() for p in self.backbone.parameters())
        Log.info(
            f"[EgoHand] Backbone permanently frozen/eval ({num_params:,} parameters)"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _get_bbox_components(self, batch: Dict, device, dtype):
        """Return the image/bbox components and a synthesized focal.

        Shared base for every camera mode: it resolves the bbox and image size,
        then derives a focal from ``focal_length_mode`` alone, ignoring any
        calibration the batch carries. Callers that need the real camera should
        go through :meth:`_get_camera_components`, which overrides the focal and
        principal point with ``K_fullimg`` in the calibrated modes.
        """
        if "img_size" in batch:
            img_wh = batch["img_size"].to(device=device, dtype=dtype)
        else:
            K = batch["K_fullimg"].to(device=device, dtype=dtype)
            img_wh = torch.stack([K[:, 0, 2] * 2, K[:, 1, 2] * 2], dim=-1)

        if "bbx_xys" in batch:
            bbx = batch["bbx_xys"].to(device=device, dtype=dtype)
            box_center = bbx[:, :2]
            box_size = bbx[:, 2]
        else:
            box_center = batch["box_center"].to(device=device, dtype=dtype)
            box_size = batch["box_size"].to(device=device, dtype=dtype)

        batch_size = img_wh.shape[0]
        if self.focal_length_mode == "diagonal":
            focal = torch.sqrt(img_wh[:, 0] ** 2 + img_wh[:, 1] ** 2)
        else:
            focal = self.focal_length * torch.ones(
                batch_size, device=device, dtype=dtype
            )
        return img_wh, box_center, box_size, focal

    def _get_camera_components(self, batch: Dict, device, dtype):
        """Return camera/bbox components for CLIFF and perspective projection.

        ``perspective`` preserves the historical synthetic camera: configured
        focal length and image-center principal point. ``sam3db_perspective``
        instead uses the dataset-provided ``K_fullimg`` unchanged, including its
        real focal and principal point.
        """
        img_wh, box_center, box_size, focal = self._get_bbox_components(
            batch,
            device,
            dtype,
        )
        principal = img_wh / 2
        K_fullimg = None
        if "K_fullimg" in batch:
            K_fullimg = batch["K_fullimg"].to(device=device, dtype=dtype)

        if self.camera_mode == "sam3db_perspective":
            if K_fullimg is None:
                raise KeyError(
                    "camera_mode='sam3db_perspective' requires batch['K_fullimg']"
                )
            if K_fullimg.dim() != 3 or K_fullimg.shape[-2:] != (3, 3):
                raise ValueError(
                    "batch['K_fullimg'] must have shape (B, 3, 3), got "
                    f"{tuple(K_fullimg.shape)}"
                )
            if not torch.isfinite(K_fullimg).all():
                raise ValueError("batch['K_fullimg'] contains non-finite values")
            if (K_fullimg[:, 0, 0] <= 0).any() or (K_fullimg[:, 1, 1] <= 0).any():
                raise ValueError(
                    "camera_mode='sam3db_perspective' requires positive fx/fy"
                )
            legacy_aug_rot = batch.get("aug_rot", None)
            if legacy_aug_rot is not None:
                legacy_aug_rot = torch.as_tensor(
                    legacy_aug_rot,
                    device=device,
                    dtype=dtype,
                )
                if (legacy_aug_rot.abs() > 1e-6).any():
                    raise ValueError(
                        "camera_mode='sam3db_perspective' received nonzero "
                        "batch['aug_rot']. Use dataset "
                        "rotation_mode='full_image_camera_roll', which applies "
                        "the roll to the full camera frame and emits aug_rot=0."
                    )
            focal = K_fullimg[:, 0, 0]
            principal = torch.stack(
                [K_fullimg[:, 0, 2], K_fullimg[:, 1, 2]],
                dim=-1,
            )
        return (
            img_wh,
            box_center,
            box_size,
            focal,
            principal,
            K_fullimg,
        )

    @staticmethod
    def _cliff_from_components(box_center, box_size, focal, principal):
        """Encode the crop relative to the active camera's optical center.

        The three-channel layout intentionally matches the established CLIFF
        convention: both offsets and bbox size are normalized by ``fx``. In
        calibrated mode the offsets use the *real* principal point from
        ``K_fullimg`` rather than assuming it is ``(W/2, H/2)``.

        Takes already-resolved components so ``forward`` can reuse the ones it
        needed anyway; :meth:`_get_cliff_conditioning` is the entry point for
        callers holding only a batch.
        """
        return torch.stack(
            [
                (box_center[:, 0] - principal[:, 0]) / focal,
                (box_center[:, 1] - principal[:, 1]) / focal,
                box_size / focal,
            ],
            dim=-1,
        )

    def _get_cliff_conditioning(self, batch: Dict, device, dtype):
        """Resolve the camera components for ``batch`` and build CLIFF from them."""
        (
            _img_wh,
            box_center,
            box_size,
            focal,
            principal,
            _K_fullimg,
        ) = self._get_camera_components(batch, device, dtype)
        return self._cliff_from_components(box_center, box_size, focal, principal)

    def _resolve_projection_K(
        self,
        batch_size: int,
        img_wh: torch.Tensor,
        focal: torch.Tensor,
        K_fullimg: torch.Tensor | None,
        device,
    ) -> torch.Tensor:
        """Pick the intrinsics the full-perspective path projects through.

        ``sam3db_perspective`` uses the dataset's ``K_fullimg`` verbatim;
        legacy ``perspective`` synthesizes K from the configured focal and an
        image-center principal point.
        """
        if self.camera_mode == "sam3db_perspective":
            if K_fullimg is None:
                raise KeyError(
                    "camera_mode='sam3db_perspective' requires batch['K_fullimg']"
                )
            return K_fullimg.float()
        K = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)
        K[:, 0, 0] = focal.float()
        K[:, 1, 1] = focal.float()
        K[:, 0, 2] = img_wh[:, 0].float() / 2
        K[:, 1, 2] = img_wh[:, 1].float() / 2
        K[:, 2, 2] = 1.0
        return K

    @staticmethod
    def _split_scale_params(scale_params: torch.Tensor):
        if scale_params.dim() != 2 or scale_params.shape[1] < 1:
            raise ValueError(
                f"scale_params must be (B, S>=1), got {tuple(scale_params.shape)}"
            )
        global_scale = scale_params[:, 0]
        hand_scale_params = scale_params[:, 1:]
        return global_scale, hand_scale_params

    @staticmethod
    def _project_full_perspective(
        joints_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        box_center: torch.Tensor,
        box_size: torch.Tensor,
        K: torch.Tensor,
    ):
        """Convert [s, tx, ty], pinhole-project with K, and crop-normalize."""
        with torch.autocast(
            device_type=joints_3d.device.type,
            enabled=False,
        ):
            bbx_xys = torch.cat(
                [box_center.float(), box_size.float().unsqueeze(-1)],
                dim=-1,
            )
            camera_translation = compute_transl_full_cam(
                pred_cam.float(),
                bbx_xys,
                K.float(),
            )
            pixels = cam_perspective_projection(
                joints_3d.float() + camera_translation.unsqueeze(1),
                K.float(),
            )
            crop_normalized = (
                pixels - box_center.float().unsqueeze(1)
            ) / box_size.float().view(-1, 1, 1)
            return camera_translation, crop_normalized

    @staticmethod
    def _project_camera_points_to_crop(
        camera_points: torch.Tensor,
        box_center: torch.Tensor,
        box_size: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """Project camera-frame points and return crop-centered coordinates."""
        with torch.autocast(
            device_type=camera_points.device.type,
            enabled=False,
        ):
            pixels = cam_perspective_projection(
                camera_points.float(),
                K.float(),
            )
            return (pixels - box_center.float().unsqueeze(1)) / box_size.float().view(
                -1, 1, 1
            )

    @staticmethod
    def _project_with_camera_translation_validated(
        joints_3d: torch.Tensor,
        camera_translation: torch.Tensor,
        box_center: torch.Tensor,
        box_size: torch.Tensor,
        K: torch.Tensor,
        *,
        min_depth: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Safely project only physically valid camera-frame joints.

        A pinhole projection is undefined for a point on or behind the camera
        plane.  Replace such points with a finite inert point before any
        projection arithmetic and return a detached validity mask for loss
        gating.  Other supervised non-finite projections are deliberately not
        sanitized, so they still reach the coordinated non-finite-loss check.
        """
        with torch.autocast(
            device_type=joints_3d.device.type,
            enabled=False,
        ):
            camera_points = joints_3d.float() + camera_translation.float().unsqueeze(1)
            projection_valid = torch.isfinite(camera_points).all(dim=-1) & (
                camera_points[..., 2] > float(min_depth)
            )
            inert_points = torch.zeros_like(camera_points)
            inert_points[..., 2] = 1.0
            safe_camera_points = torch.where(
                projection_valid.unsqueeze(-1),
                camera_points,
                inert_points,
            )
            projected = EgoHand._project_camera_points_to_crop(
                safe_camera_points,
                box_center,
                box_size,
                K,
            )
        return projected, projection_valid.detach(), camera_points

    @staticmethod
    def _inside_crop_half_open(
        keypoints_crop: torch.Tensor,
    ) -> torch.Tensor:
        """Return finite membership in the crop support ``[-0.5, 0.5)``."""
        return (
            torch.isfinite(keypoints_crop).all(dim=-1)
            & (keypoints_crop[..., 0] >= -0.5)
            & (keypoints_crop[..., 0] < 0.5)
            & (keypoints_crop[..., 1] >= -0.5)
            & (keypoints_crop[..., 1] < 0.5)
        )

    @staticmethod
    def _inside_full_image_half_open(
        keypoints_full: torch.Tensor,
        image_wh: torch.Tensor,
    ) -> torch.Tensor:
        """Return finite membership in ``[0, W) x [0, H)``."""
        if image_wh.ndim != 2 or image_wh.shape[-1] != 2:
            raise ValueError(
                f"image_wh must have shape (B, 2), got {tuple(image_wh.shape)}"
            )
        if image_wh.shape[0] != keypoints_full.shape[0]:
            raise ValueError("image_wh batch size must match keypoints_full")
        width = image_wh[:, 0:1]
        height = image_wh[:, 1:2]
        return (
            torch.isfinite(keypoints_full).all(dim=-1)
            & (keypoints_full[..., 0] >= 0)
            & (keypoints_full[..., 0] < width)
            & (keypoints_full[..., 1] >= 0)
            & (keypoints_full[..., 1] < height)
        )

    def _require_kp2d_image_size(
        self,
        batch: Dict,
        device,
        dtype,
    ) -> torch.Tensor:
        """Return real full-image ``[W,H]``; never infer it from principal point."""
        if "img_size" not in batch:
            raise KeyError(
                "kp2d_translation requires batch['img_size'] with the "
                "real full-image [W, H] for onscreen masking"
            )
        image_wh = batch["img_size"].to(device=device, dtype=dtype)
        if image_wh.ndim != 2 or image_wh.shape[-1] != 2:
            raise ValueError(
                f"batch['img_size'] must have shape (B, 2), got {tuple(image_wh.shape)}"
            )
        if not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
            raise ValueError("batch['img_size'] must contain finite positive [W, H]")
        return image_wh

    @staticmethod
    def _validate_kp2d_bbox(
        box_center: torch.Tensor,
        box_size: torch.Tensor,
    ) -> None:
        """Fail fast on a degenerate crop before projection or camera solving."""
        if (
            not torch.isfinite(box_center).all()
            or not torch.isfinite(box_size).all()
            or (box_size <= 0).any()
        ):
            raise ValueError(
                "kp2d_translation requires finite box centers and positive box sizes"
            )

    def _predict_direct_keypoints_2d(
        self,
        conditioning_feats: torch.Tensor,
        merged_joint_tokens: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        if not self.use_kp2d_translation:
            return None
        if self.kp2d_topology == "separate":
            return self.kp2d_head(conditioning_feats)
        if merged_joint_tokens is None:
            raise RuntimeError(
                "Merged kp2d topology did not receive hand-head joint tokens"
            )
        visual_memory = self.kp2d_visual_memory(conditioning_feats)
        joint_tokens = self.kp2d_token_adapter(merged_joint_tokens)
        return self.kp2d_readout(joint_tokens, visual_memory)

    def _build_kp2d_inference_point_mask(
        self,
        direct_keypoints_crop: torch.Tensor,
        crop_presence: torch.Tensor,
        bbx_xys: torch.Tensor,
        image_wh: torch.Tensor,
    ):
        """Gate translation correspondences by predicted visible crop support."""
        direct_keypoints_full = crop_keypoints_to_full_image(
            direct_keypoints_crop.float(),
            bbx_xys.float(),
        )
        point_mask = (
            (crop_presence.float() >= self.kp2d_presence_threshold)
            & self._inside_crop_half_open(direct_keypoints_crop)
            & self._inside_full_image_half_open(
                direct_keypoints_full,
                image_wh,
            )
        )
        return point_mask, direct_keypoints_full

    def _solve_translation_from_direct_keypoints(
        self,
        pred_joints: torch.Tensor,
        pred_cam: torch.Tensor,
        direct_keypoints_crop: torch.Tensor,
        point_mask: torch.Tensor,
        box_center: torch.Tensor,
        box_size: torch.Tensor,
        K: torch.Tensor,
    ):
        """Solve tx/ty at camera-s-predicted tz, with per-sample fallback."""
        bbx_xys = torch.cat(
            [box_center.float(), box_size.float().unsqueeze(-1)],
            dim=-1,
        )
        legacy_translation = compute_transl_full_cam(
            pred_cam.float(),
            bbx_xys,
            K.float(),
        )
        translation_z = camera_scale_to_translation_z(
            pred_cam[:, 0].float(),
            bbx_xys,
            K.float(),
        )
        direct_keypoints_full = crop_keypoints_to_full_image(
            direct_keypoints_crop.float(),
            bbx_xys,
        )
        solve = solve_lateral_translation_full_perspective(
            pred_joints.float(),
            direct_keypoints_full,
            translation_z,
            K.float(),
            point_mask=point_mask,
            min_points=self.kp2d_min_points,
            min_depth=self.kp2d_min_depth,
            damping=self.kp2d_solver_damping,
            max_condition_number=self.kp2d_max_condition_number,
        )
        solved_translation = torch.cat(
            [solve.translation_xy, translation_z.unsqueeze(-1)],
            dim=-1,
        )
        final_translation = torch.where(
            solve.valid.unsqueeze(-1),
            solved_translation,
            legacy_translation,
        )
        return (
            final_translation,
            legacy_translation,
            direct_keypoints_full,
            solve,
        )

    def _gt_scale_params_full(self, batch: Dict, batch_size: int, device, dtype):
        gt_global_scale = batch.get(
            "hand_global_scale",
            torch.ones(batch_size, device=device, dtype=dtype),
        )
        gt_global_scale = gt_global_scale.to(device=device, dtype=dtype)
        if gt_global_scale.dim() == 1:
            gt_global_scale = gt_global_scale.unsqueeze(-1)
        gt_scale_params = batch.get("hand_scale_params", None)
        if gt_scale_params is None:
            num_scale = int(
                self.cfg.get("soma_hand_head", {}).get("num_scale_params", 1)
            )
            gt_scale_params = torch.zeros(
                batch_size, max(num_scale - 1, 0), device=device, dtype=dtype
            )
        else:
            gt_scale_params = gt_scale_params.to(device=device, dtype=dtype)
        return torch.cat([gt_global_scale, gt_scale_params], dim=1)

    def process_conditioning_feats(
        self,
        conditioning_feats: torch.Tensor,
        batch: Dict,
        train: bool = False,
    ) -> torch.Tensor:
        """Hook for subclasses to modify backbone features before the hand head."""
        return conditioning_feats

    def process_pred_hand_params(
        self,
        pred_hand_params: Dict[str, torch.Tensor],
        pred_cam: torch.Tensor,
        batch: Dict,
        train: bool = False,
    ):
        """Hook for subclasses to refine predicted hand params without copying FK code."""
        return pred_hand_params, pred_cam

    def forward_step(
        self, batch: Dict, train: bool = False, un_rotate: bool = False
    ) -> Dict:
        x = batch["img"]
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        if self._backbone_crop_w is not None:
            backbone_in = x[
                :, :, :, self._backbone_crop_w[0] : self._backbone_crop_w[1]
            ]
        else:
            backbone_in = x
        conditioning_feats = self.backbone(backbone_in)

        # Optional depth fusion. Adds depth_fuser(depth_encoder(depth)) to the
        # backbone feature map, masked per-sample by ``has_depth``. The fuser
        # is zero-initialized, so at step 0 this is a no-op regardless of the
        # depth tensor's contents — a standard zero-module fusion pattern.
        if self.use_depth and "depth" in batch:
            depth = batch["depth"].to(device=device, dtype=dtype)
            depth_feat = self.depth_fuser(self.depth_encoder(depth))  # (B, C, h, w)
            # Spatial dims must match the backbone feature map — fix by adjusting
            # the DepthEncoder stride / backbone patch size, not by interpolating.
            assert depth_feat.shape[-2:] == conditioning_feats.shape[-2:], (
                f"depth feature spatial size {tuple(depth_feat.shape[-2:])} != "
                f"backbone feature spatial size {tuple(conditioning_feats.shape[-2:])}"
            )
            has_depth = batch.get("has_depth", None)
            if has_depth is None:
                gate = torch.ones(batch_size, device=device, dtype=dtype)
            else:
                gate = has_depth.to(device=device, dtype=dtype).reshape(batch_size)
            if train and self.depth_drop_prob > 0.0:
                drop = (torch.rand_like(gate) < self.depth_drop_prob).to(dtype)
                gate = gate * (1.0 - drop)
            conditioning_feats = (
                conditioning_feats + depth_feat * gate[:, None, None, None]
            )

        conditioning_feats = self.process_conditioning_feats(
            conditioning_feats,
            batch,
            train=train,
        )

        cliff_cond = None
        img_wh = box_center = box_size = focal = principal = K_fullimg = None
        is_full_perspective = self.camera_mode in self.full_perspective_modes
        if self.use_kp2d_translation:
            # Do this before _get_camera_components can exercise its legacy
            # principal-point-based image-size fallback.
            self._require_kp2d_image_size(batch, device, dtype)
        if is_full_perspective or self.cliff_condition or self.ray_condition:
            (
                img_wh,
                box_center,
                box_size,
                focal,
                principal,
                K_fullimg,
            ) = self._get_camera_components(batch, device, dtype)
        if self.use_kp2d_translation:
            self._validate_kp2d_bbox(box_center, box_size)
        if self.cliff_condition:
            cliff_cond = self._cliff_from_components(
                box_center,
                box_size,
                focal,
                principal,
            )

        ray_feats = None
        if self.ray_condition:
            if K_fullimg is None:
                raise KeyError("EgoHand ray_condition requires batch['K_fullimg']")
            legacy_aug_rot = batch.get("aug_rot", None)
            if legacy_aug_rot is not None:
                legacy_aug_rot_t = torch.as_tensor(
                    legacy_aug_rot,
                    device=device,
                    dtype=dtype,
                )
                if (legacy_aug_rot_t.abs() > 1e-6).any():
                    raise ValueError(
                        "EgoHand ray_condition received nonzero "
                        "batch['aug_rot']. Use dataset "
                        "rotation_mode='full_image_camera_roll', which "
                        "applies rotation augmentation to the full camera "
                        "frame (K_fullimg + bbx_xys) and emits aug_rot=0, "
                        "so per-token ray directions stay geometrically "
                        "correct."
                    )
            bbx_xys = torch.cat((box_center, box_size[:, None]), dim=-1)
            ray_angles = compute_crop_ray_angles(
                bbx_xys,
                K_fullimg,
                grid_hw=tuple(conditioning_feats.shape[-2:]),
                crop_hw=tuple(backbone_in.shape[-2:]),
            )
            ray_feats = fourier_encode_angles(
                ray_angles,
                num_bands=self.ray_pe_num_bands,
            )

        hand_head_output = self.hand_head(
            conditioning_feats,
            cliff_cond=cliff_cond,
            ray_feats=ray_feats,
        )
        if len(hand_head_output) == 3:
            (
                pred_hand_params,
                pred_cam,
                pred_hand_params_list,
            ) = hand_head_output
            merged_joint_tokens = None
        elif len(hand_head_output) == 4:
            (
                pred_hand_params,
                pred_cam,
                pred_hand_params_list,
                merged_joint_tokens,
            ) = hand_head_output
        else:
            raise RuntimeError(
                "hand_head must return 3 baseline values or 4 values with "
                "merged kp2d joint tokens"
            )
        direct_kp2d_output = self._predict_direct_keypoints_2d(
            conditioning_feats,
            merged_joint_tokens,
        )
        pred_hand_params_pre_refine = {
            key: value.clone() for key, value in pred_hand_params.items()
        }
        pred_cam_pre_refine = pred_cam.clone()
        pred_hand_params, pred_cam = self.process_pred_hand_params(
            pred_hand_params,
            pred_cam,
            batch,
            train=train,
        )

        output = {
            "_forward_train": bool(train),
        }
        output["pred_hand_params_pre_refine"] = pred_hand_params_pre_refine
        output["pred_cam_pre_refine"] = pred_cam_pre_refine
        output["pred_hand_params"] = {k: v.clone() for k, v in pred_hand_params.items()}
        if pred_hand_params_list is not None:
            output["pred_hand_params_list"] = pred_hand_params_list
        if direct_kp2d_output is not None:
            output["kp2d_output"] = direct_kp2d_output
            output["pred_keypoints_2d_direct"] = direct_kp2d_output["mu"]
            output["pred_kp2d_sigma"] = direct_kp2d_output["sigma"]
            output["pred_kp2d_crop_presence"] = direct_kp2d_output["presence"]
            output["pred_kp2d_presence_logit"] = direct_kp2d_output["pres_logit"]

        with torch.amp.autocast("cuda", enabled=False):
            poses_25 = torch.cat(
                [
                    pred_hand_params["wrist_orient"].float(),
                    pred_hand_params["hand_pose"].float(),
                ],
                dim=1,
            )
            pred_global_scale, pred_scale_params = self._split_scale_params(
                pred_hand_params["scale_params"].float()
            )
            hand_out = self.soma_hand(
                poses=poses_25,
                identity_coeffs=pred_hand_params["identity_coeffs"].float(),
                hand_type="left",
                pose2rot=False,
                global_scale=pred_global_scale,
                scale_params=pred_scale_params,
            )
            pred_joints = hand_out["joints"]
            pred_vertices = hand_out["vertices"]
            pred_camera_parameter_f = pred_cam.float()

            if un_rotate:
                aug_rot = batch.get("aug_rot", None)
                if aug_rot is not None:
                    aug_rot_f = torch.as_tensor(
                        aug_rot, device=device, dtype=torch.float32
                    )
                    if aug_rot_f.dim() == 0:
                        aug_rot_f = aug_rot_f.unsqueeze(0).expand(batch_size)
                    if (aug_rot_f != 0).any():
                        theta = torch.deg2rad(aug_rot_f)
                        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                        R_undo = torch.zeros(batch_size, 3, 3, device=device)
                        R_undo[:, 0, 0] = cos_t
                        R_undo[:, 0, 1] = -sin_t
                        R_undo[:, 1, 0] = sin_t
                        R_undo[:, 1, 1] = cos_t
                        R_undo[:, 2, 2] = 1.0
                        pred_joints = torch.einsum("bij,bnj->bni", R_undo, pred_joints)
                        pred_vertices = torch.einsum(
                            "bij,bnj->bni", R_undo, pred_vertices
                        )
                        camera_scalar = pred_camera_parameter_f[:, 0:1]
                        cam_tx = pred_camera_parameter_f[:, 1:2]
                        cam_ty = pred_camera_parameter_f[:, 2:3]
                        cam_tx_u = cos_t[:, None] * cam_tx - sin_t[:, None] * cam_ty
                        cam_ty_u = sin_t[:, None] * cam_tx + cos_t[:, None] * cam_ty
                        pred_camera_parameter_f = torch.cat(
                            [camera_scalar, cam_tx_u, cam_ty_u],
                            dim=-1,
                        )

            output["pred_keypoints_3d"] = pred_joints
            output["pred_vertices"] = pred_vertices
            pred_camera_parameter_raw = pred_camera_parameter_f
            pred_camera_parameter_effective = effective_camera_depth_parameter(
                pred_camera_parameter_raw,
                self.camera_depth_mode,
                log_value_bounds=self.camera_log_value_bounds,
            )
            # Three deliberately distinct camera contracts:
            #   raw:       unconstrained head value used by supervision;
            #   effective: exact mode-space value used by physical decoding;
            #   pred_cam:  canonical legacy [s, dx, dy] for old consumers.
            output["pred_camera_parameter_raw"] = pred_camera_parameter_raw
            output["pred_camera_parameter_valid"] = torch.isfinite(
                pred_camera_parameter_raw
            ).all(dim=-1)
            output["pred_camera_parameter"] = pred_camera_parameter_effective
            output["pred_native_camera_parameter"] = pred_camera_parameter_effective

            if is_full_perspective:
                K = self._resolve_projection_K(
                    batch_size,
                    img_wh,
                    focal,
                    K_fullimg,
                    device,
                )
                bbx_xys = torch.cat(
                    [
                        box_center.float(),
                        box_size.float().unsqueeze(-1),
                    ],
                    dim=-1,
                )
                pred_cam_f = depth_parameter_to_legacy_camera(
                    pred_camera_parameter_effective,
                    bbx_xys,
                    K,
                    self.camera_depth_mode,
                    log_value_bounds=self.camera_log_value_bounds,
                )
                # Keep the historical public contract. Consumers that need
                # the mode-space scalar use ``pred_camera_parameter``.
                output["pred_cam"] = pred_cam_f
                (
                    pred_cam_t_legacy,
                    pred_keypoints_2d_legacy,
                ) = self._project_full_perspective(
                    pred_joints,
                    pred_cam_f,
                    box_center,
                    box_size,
                    K,
                )
                pred_cam_t = pred_cam_t_legacy
                pred_keypoints_2d = pred_keypoints_2d_legacy
                output["pred_cam_t_legacy"] = pred_cam_t_legacy

                if self.use_kp2d_translation and not train:
                    direct_crop = direct_kp2d_output["mu"].float()
                    (
                        point_mask,
                        direct_full,
                    ) = self._build_kp2d_inference_point_mask(
                        direct_crop,
                        direct_kp2d_output["presence"],
                        bbx_xys,
                        img_wh.float(),
                    )
                    output["pred_keypoints_2d_direct_full"] = direct_full
                    (
                        pred_cam_t,
                        _legacy_translation,
                        _direct_full,
                        solve,
                    ) = self._solve_translation_from_direct_keypoints(
                        pred_joints,
                        pred_cam_f,
                        direct_crop,
                        point_mask,
                        box_center,
                        box_size,
                        K,
                    )
                    (
                        pred_keypoints_2d_safe,
                        pred_projection_valid,
                        _pred_camera_points,
                    ) = self._project_with_camera_translation_validated(
                        pred_joints,
                        pred_cam_t,
                        box_center,
                        box_size,
                        K,
                        min_depth=self.kp2d_min_depth,
                    )
                    # The inert projection returned above exists only to keep
                    # arithmetic finite.  It is not a real prediction: expose
                    # invalid inference joints as missing values so downstream
                    # consumers cannot mistake the camera principal point for
                    # a valid 2D joint.  Training recomputes a finite,
                    # validity-gated projection in ``compute_loss``.
                    pred_projection_valid = pred_projection_valid & torch.isfinite(
                        pred_keypoints_2d_safe
                    ).all(dim=-1)
                    pred_keypoints_2d = pred_keypoints_2d_safe.masked_fill(
                        ~pred_projection_valid.unsqueeze(-1),
                        float("nan"),
                    )
                    output["camera_solve_valid"] = solve.valid
                    output["camera_solve_num_points"] = solve.num_points
                    output["camera_solve_fallback"] = ~solve.valid
                    output["camera_solve_condition_number"] = solve.condition_number
                    output["camera_solve_point_mask"] = point_mask
                    output["pred_projection_valid"] = pred_projection_valid
                elif self.use_kp2d_translation:
                    output["camera_solve_valid"] = torch.zeros(
                        batch_size,
                        device=device,
                        dtype=torch.bool,
                    )
                    output["camera_solve_num_points"] = torch.zeros(
                        batch_size,
                        device=device,
                        dtype=torch.long,
                    )
                    output["camera_solve_fallback"] = torch.ones(
                        batch_size,
                        device=device,
                        dtype=torch.bool,
                    )
                output["pred_cam_t"] = pred_cam_t
                pred_depth = pred_cam_t[:, 2]
                pred_lambda = box_size.float() * pred_depth / K[:, 0, 0]
                output["pred_depth"] = pred_depth
                output["pred_log_depth"] = torch.log(pred_depth)
                output["pred_lambda"] = pred_lambda
                output["pred_log_lambda"] = torch.log(pred_lambda)
                output["focal_length"] = torch.stack(
                    [K[:, 0, 0], K[:, 1, 1]],
                    dim=-1,
                )
                output["K"] = K
            else:
                # Construction rejects log-depth with a weak-perspective
                # camera, so the raw parameter already has [s, dx, dy]
                # semantics here.
                pred_cam_f = pred_camera_parameter_effective
                output["pred_cam"] = pred_cam_f
                focal_length_t = self.focal_length * torch.ones(
                    batch_size, 2, device=device, dtype=torch.float32
                )
                pred_cam_t = torch.stack(
                    [
                        pred_cam_f[:, 1],
                        pred_cam_f[:, 2],
                        2
                        * focal_length_t[:, 0]
                        / (self.image_size * pred_cam_f[:, 0] + 1e-9),
                    ],
                    dim=-1,
                )
                output["pred_cam_t"] = pred_cam_t
                output["focal_length"] = focal_length_t
                pred_keypoints_2d = perspective_projection(
                    pred_joints,
                    translation=pred_cam_t,
                    focal_length=focal_length_t / self.image_size,
                )

            output["pred_keypoints_2d"] = pred_keypoints_2d.reshape(
                batch_size,
                -1,
                2,
            )

            # A non-finite raw camera value must never look like a plausible
            # bounded depth in standalone inference. During trainer-owned
            # train/validation passes, keep downstream arithmetic finite and
            # let the coordinated non-finite loss marker below stop every DDP
            # rank together.
            if not train and getattr(self, "_trainer", None) is None:
                invalid_camera = ~output["pred_camera_parameter_valid"]
                for key in (
                    "pred_cam",
                    "pred_cam_t",
                    "pred_depth",
                    "pred_log_depth",
                    "pred_lambda",
                    "pred_log_lambda",
                ):
                    value = output.get(key)
                    if value is None:
                        continue
                    mask = invalid_camera.reshape(
                        invalid_camera.shape + (1,) * (value.ndim - 1)
                    )
                    output[key] = value.masked_fill(mask, float("nan"))
                output["pred_keypoints_2d"] = output["pred_keypoints_2d"].masked_fill(
                    invalid_camera[:, None, None],
                    float("nan"),
                )
                if "pred_projection_valid" in output:
                    output["pred_projection_valid"] = (
                        output["pred_projection_valid"] & ~invalid_camera[:, None]
                    )

        return output

    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False, un_rotate=True)

    def validate_camera_depth_checkpoint(self, checkpoint) -> None:
        """Reject shape-compatible checkpoints with incompatible semantics."""

        # Lightweight infrastructure tests reuse the checkpoint hooks on a
        # shell model that intentionally does not construct the camera head.
        # Real EgoHand instances always define this attribute in __init__.
        if not hasattr(self, "camera_depth_mode"):
            return

        hyperparameters = checkpoint.get("hyper_parameters", {})
        checkpoint_cfg = (
            hyperparameters.get("cfg", None)
            if hasattr(hyperparameters, "get")
            else None
        )
        checkpoint_depth_mode = checkpoint.get(
            "egohand_camera_depth_mode",
            None,
        )
        if checkpoint_depth_mode is None:
            if hasattr(checkpoint_cfg, "get"):
                checkpoint_depth_mode = checkpoint_cfg.get(
                    "camera_depth_mode",
                    "s",
                )
            else:
                # Checkpoints without this key pre-date log-depth and
                # default to s.
                checkpoint_depth_mode = "s"
        checkpoint_depth_mode = canonical_camera_depth_mode(checkpoint_depth_mode)
        if checkpoint_depth_mode != self.camera_depth_mode:
            raise RuntimeError(
                "Refusing to reinterpret a camera head checkpoint across "
                "depth parameterizations: checkpoint uses "
                f"{checkpoint_depth_mode!r}, model uses "
                f"{self.camera_depth_mode!r}. Start a fresh head or use a "
                "mode-matched checkpoint."
            )
        checkpoint_scalar_loss_type = checkpoint.get(
            "egohand_camera_scalar_loss_type",
            None,
        )
        if checkpoint_scalar_loss_type is None:
            if hasattr(checkpoint_cfg, "get"):
                checkpoint_scalar_loss_type = checkpoint_cfg.get(
                    "camera_scalar_loss_type",
                    None,
                )
            if checkpoint_scalar_loss_type is None:
                checkpoint_scalar_loss_type = (
                    "legacy_mse"
                    if checkpoint_depth_mode == "s"
                    else (
                        checkpoint_cfg.get(
                            "camera_depth_loss_type",
                            "smooth_l1",
                        )
                        if hasattr(checkpoint_cfg, "get")
                        else "smooth_l1"
                    )
                )
        checkpoint_scalar_loss_type = str(checkpoint_scalar_loss_type)
        if checkpoint_scalar_loss_type != self.camera_scalar_loss_type:
            raise RuntimeError(
                "Refusing to load a camera checkpoint with a different "
                "scalar loss family: checkpoint uses "
                f"{checkpoint_scalar_loss_type!r}, model uses "
                f"{self.camera_scalar_loss_type!r}."
            )
        checkpoint_depth_std = checkpoint.get(
            "egohand_camera_depth_std",
            None,
        )
        if (
            checkpoint_depth_std is not None
            and checkpoint_depth_mode != "s"
            and not math.isclose(
                float(checkpoint_depth_std),
                self.camera_depth_std,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                "Refusing to resume a log-depth camera with a different "
                "normalization std: checkpoint uses "
                f"{float(checkpoint_depth_std):.9g}, model uses "
                f"{self.camera_depth_std:.9g}."
            )
        checkpoint_scalar_loss_std = checkpoint.get(
            "egohand_camera_scalar_loss_std",
            None,
        )
        if checkpoint_scalar_loss_std is None and hasattr(checkpoint_cfg, "get"):
            checkpoint_scalar_loss_std = checkpoint_cfg.get(
                "camera_scalar_loss_std",
                None,
            )
        if checkpoint_scalar_loss_std is None:
            checkpoint_scalar_loss_std = (
                checkpoint_depth_std
                if checkpoint_depth_std is not None
                else (
                    checkpoint_cfg.get("camera_depth_std", 1.0)
                    if hasattr(checkpoint_cfg, "get")
                    else 1.0
                )
            )
        if (
            self.camera_scalar_loss_type != "legacy_mse"
            or self.camera_scalar_delta_scaled
        ) and not math.isclose(
            float(checkpoint_scalar_loss_std),
            self.camera_scalar_loss_std,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Refusing to load a camera checkpoint with a different "
                "scalar-loss normalization std: checkpoint uses "
                f"{float(checkpoint_scalar_loss_std):.9g}, model uses "
                f"{self.camera_scalar_loss_std:.9g}."
            )
        checkpoint_delta_scaled = checkpoint.get(
            "egohand_camera_scalar_delta_scaled",
            None,
        )
        if checkpoint_delta_scaled is None:
            if hasattr(checkpoint_cfg, "get"):
                checkpoint_delta_scaled = checkpoint_cfg.get(
                    "camera_scalar_delta_scaled",
                    None,
                )
            if checkpoint_delta_scaled is None:
                checkpoint_delta_scaled = checkpoint_depth_mode != "s"
        checkpoint_delta_scaled = bool(checkpoint_delta_scaled)
        if checkpoint_delta_scaled != self.camera_scalar_delta_scaled:
            raise RuntimeError(
                "Refusing to load a camera checkpoint with a different "
                "decoder scalar normalization contract: checkpoint "
                f"scaled={checkpoint_delta_scaled}, model "
                f"scaled={self.camera_scalar_delta_scaled}."
            )
        checkpoint_log_bounds = checkpoint.get(
            "egohand_camera_log_value_bounds",
            None,
        )
        if checkpoint_log_bounds is None and hasattr(checkpoint_cfg, "get"):
            checkpoint_log_bounds = checkpoint_cfg.get(
                "camera_log_value_bounds",
                None,
            )
        if checkpoint_depth_mode != "s" and checkpoint_log_bounds is None:
            raise RuntimeError(
                "Refusing to load a log-depth camera checkpoint without "
                "camera_log_value_bounds metadata. The bounds are part of "
                "the physical decoder contract, so their semantics cannot "
                "be inferred safely."
            )
        if checkpoint_depth_mode != "s":
            if len(checkpoint_log_bounds) != 2:
                raise RuntimeError(
                    "Log-depth checkpoint camera_log_value_bounds must "
                    "contain exactly two values"
                )
            checkpoint_log_bounds = tuple(
                float(value) for value in checkpoint_log_bounds
            )
            if any(
                not math.isclose(
                    checkpoint_value,
                    model_value,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                for checkpoint_value, model_value in zip(
                    checkpoint_log_bounds,
                    self.camera_log_value_bounds,
                )
            ):
                raise RuntimeError(
                    "Refusing to load a log-depth camera checkpoint with "
                    "different decoder bounds: checkpoint uses "
                    f"{checkpoint_log_bounds}, model uses "
                    f"{self.camera_log_value_bounds}."
                )
