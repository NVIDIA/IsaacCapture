# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Hugging Face repo ids and download helper for default artifacts."""

EGOHAND_HF_REPO = "nvidia/EgoHand-v1"
RTDETR_HF_REPO = "nvidia/RT-DETR-Hand-Detector-v1"


def hf_download(filename: str, repo_id: str = EGOHAND_HF_REPO) -> str:
    """Download ``filename`` from an HF repo (default: EgoHand-v1), caching locally.

    Used to resolve default artifact paths (EgoHand checkpoint from EGOHAND_HF_REPO,
    RT-DETR weights from RTDETR_HF_REPO) when the user doesn't pass an explicit
    local path.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)
