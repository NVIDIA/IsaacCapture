# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Build an RT-DETRv2 hand bbox detector for ``detector_backend: rtdetr``.

Paired with ``rtdetr.yaml`` / ``rtdetr_docker.yaml`` in this directory.
"""

import dataclasses


def _rtdetr_config(data: dict | None):
    """Build an RTDetrConfig from the ``rtdetr_config:`` YAML mapping and apply
    the HF download fallback when weights are unset or empty. An explicit
    non-empty path is passed to the detector, which fails if it cannot load it.

    ``rtdetr_hf_weights`` is a pseudo-field, popped out before ``RTDetrConfig.from_dict``
    (which rejects unknown keys) rather than a real ``RTDetrConfig`` field -- it
    only controls what ``weights`` defaults to when unset, it isn't consulted
    once an explicit ``weights`` path is given.
    """
    from inference.hand_detector_2d.rtdetr import RTDetrConfig

    data = dict(data or {})
    rtdetr_hf_weights_path = data.pop("rtdetr_hf_weights", "hand_detector.onnx")
    config = RTDetrConfig.from_dict(data)
    if not config.weights:
        from inference.assets import RTDETR_HF_REPO, hf_download

        weights = hf_download(rtdetr_hf_weights_path, repo_id=RTDETR_HF_REPO)
        print(
            f"[rtdetr] weights not given, resolved via HF ({RTDETR_HF_REPO}/"
            f"{rtdetr_hf_weights_path}): {weights}"
        )
        config = dataclasses.replace(config, weights=weights)
    return config


def build(cfg: dict, device: str, batch_size: int, tracking: bool):
    return _rtdetr_config(cfg.get("rtdetr_config")).build(
        device=device, tracking=tracking
    )
