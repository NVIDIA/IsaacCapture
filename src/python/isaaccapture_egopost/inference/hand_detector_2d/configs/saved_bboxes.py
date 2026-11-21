# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Build a SavedBBoxDetector for ``detector_backend: saved_bboxes``.

Paired with ``saved_bboxes.yaml`` in this directory. ``saved_bbox_path`` is one
root directory shared across every video in the run, so it's bound here at
construction; only the per-video subfolder is resolved later, in
``prepare_clip()``.
"""


def build(cfg: dict, device: str, batch_size: int, tracking: bool):
    from inference.hand_detector_2d.saved_bbox import SavedBBoxDetector

    return SavedBBoxDetector(bbox_dir=cfg.get("saved_bbox_path"))
