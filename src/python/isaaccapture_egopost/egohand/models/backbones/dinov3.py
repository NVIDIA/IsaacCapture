# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""DINOv3 ViT backbone wrapper.

Loads checkpoints with a `backbone.encoder.*` state_dict prefix into a
DINOv3 ViT and exposes the same `(B, C, H/16, W/16)` feature map contract
as egohand/models/backbones/vit.py so the rest of the model is unchanged.

DINOv3 is installed from the revision pinned in requirements-dinov3.txt.
The inference image includes the package at build time; model construction
uses the installed code and does not fetch source or pretrained weights.
"""

from typing import ClassVar

import torch
import torch.nn as nn

from egohand.utils.pylogger import Log


class DinoV3Backbone(nn.Module):
    """DINOv3 ViT-H/16+ backbone for body-feature extraction.

    Returns the last transformer block's spatial token map reshaped to
    `(B, embed_dim, H/p, W/p)`.
    CLS + storage tokens are dropped by `get_intermediate_layers(reshape=True)`.
    """

    # Per-architecture build kwargs. Mirrors the upstream dinov3 hub
    # factories (dinov3/hub/backbones.py in the pinned package) but lets
    # us override drop_path_rate (the factory hardcodes it to 0.0 and then
    # forwards **kwargs, so passing drop_path_rate=X to the factory crashes).
    _ARCH_KWARGS: ClassVar[dict] = {
        "dinov3_vith16plus": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1280,
            depth=32,
            num_heads=20,
            ffn_ratio=6.0,
            qkv_bias=True,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
    }

    def __init__(self, name: str = "dinov3_vith16plus", drop_path_rate: float = 0.1):
        super().__init__()
        if name not in self._ARCH_KWARGS:
            raise ValueError(
                f"Unsupported DINOv3 arch {name!r}. "
                f"Add it to DinoV3Backbone._ARCH_KWARGS."
            )
        from dinov3.models.vision_transformer import DinoVisionTransformer

        vit_kwargs = dict(self._ARCH_KWARGS[name])
        vit_kwargs["drop_path_rate"] = drop_path_rate
        self.encoder = DinoVisionTransformer(**vit_kwargs)
        self.encoder.init_weights()
        self.name = name
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim
        self._compiled_full_batch_size = None
        # Keep OptimizedModule wrappers in a plain list so the canonical
        # ModuleList, state_dict, and training paths remain unchanged. They are
        # created lazily by an eligible inference call.
        self._compiled_blocks = None

    def enable_full_batch_compile(self, full_batch_size: int) -> None:
        """Compile DINO blocks only for eval/no-grad batches of this exact size."""
        if isinstance(full_batch_size, bool) or not isinstance(full_batch_size, int):
            raise TypeError("full_batch_size must be an integer")
        if full_batch_size <= 0:
            raise ValueError("full_batch_size must be positive")
        if (
            self._compiled_blocks is not None
            and self._compiled_full_batch_size != full_batch_size
        ):
            raise RuntimeError(
                "cannot change full_batch_size after compiled wrappers are created"
            )
        self._compiled_full_batch_size = full_batch_size

    def _ensure_inference_compiled_blocks(self):
        if self._compiled_blocks is None:
            Log.info(
                "[DinoV3Backbone] Building eval-only static compiled blocks "
                f"for B={self._compiled_full_batch_size}"
            )
            self._compiled_blocks = [
                torch.compile(
                    block,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                    mode="default",
                )
                for block in self.encoder.blocks
            ]
        return self._compiled_blocks

    def _forward_compiled_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the exact get_intermediate_layers(n=1) contract via compiled blocks."""
        encoder = self.encoder
        tokens, (height, width) = encoder.prepare_tokens_with_masks(x)
        rope = (
            encoder.rope_embed(H=height, W=width)
            if encoder.rope_embed is not None
            else None
        )
        for block in self._ensure_inference_compiled_blocks():
            tokens = block(tokens, rope)

        special_count = encoder.n_storage_tokens + 1
        if encoder.untie_cls_and_patch_norms:
            spatial = encoder.norm(tokens[:, special_count:])
        else:
            spatial = encoder.norm(tokens)[:, special_count:]
        batch = x.shape[0]
        return (
            spatial.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, 3, H, W) — H, W must be multiples of patch_size (16).
        Returns:
            (B, embed_dim, H/16, W/16) feature map (CLS + storage tokens dropped).
        """
        use_compiled = (
            self._compiled_full_batch_size is not None
            and x.shape[0] == self._compiled_full_batch_size
            and not self.training
            and not torch.is_grad_enabled()
        )
        if use_compiled:
            return self._forward_compiled_spatial_features(x)
        # Training, grad-enabled validation, and every non-full/tail batch retain
        # the unmodified eager ModuleList path and therefore cannot recompile.
        return self.encoder.get_intermediate_layers(
            x,
            n=1,
            reshape=True,
            norm=True,
        )[-1]

    def load_pretrained(self, ckpt_path: str) -> None:
        """Load a checkpoint with a `backbone.encoder.*` state_dict prefix into self.encoder.

        The checkpoint's state_dict keys are namespaced as `backbone.encoder.*`
        (1127 keys total). We keep only those keys, strip the prefix, and
        `load_state_dict(strict=False)`. Expected unexpected: `mask_token`
        (DINOv3 SSL artifact, harmless at inference).
        """
        Log.info(f"[DinoV3Backbone] Loading pretrained weights from {ckpt_path}")
        # Checkpoints are data, not trusted code. Restricted loading rejects
        # pickle globals while preserving tensors and primitive metadata.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        prefix = "backbone.encoder."
        encoder_sd = {
            k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)
        }
        if not encoder_sd:
            raise RuntimeError(
                f"No keys with prefix {prefix!r} in {ckpt_path} — expected "
                f"a checkpoint with a backbone.encoder.* state_dict prefix"
            )

        missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=False)
        Log.info(
            f"[DinoV3Backbone] Loaded {len(encoder_sd)} backbone tensors. "
            f"Missing: {len(missing)}, unexpected: {len(unexpected)}"
        )
        if missing:
            Log.warning(f"[DinoV3Backbone] Missing keys: {missing}")
        if unexpected:
            Log.warning(f"[DinoV3Backbone] Unexpected keys: {unexpected}")
