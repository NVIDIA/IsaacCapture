# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Post-processing applied to EgoHand predictions, either in-process or as a
separate post-hoc step against saved output.

``adaptive_smoothing`` operates on the per-frame result dicts
``EgoHandPredictor`` produces, so it changes the SOMA NPZs, the returned dict,
and every overlay rendered from it in one place rather than per consumer.
It filters SOMA rotations, translation and shape using frame timestamps, then
rebuilds geometry. The video CLI enables it by default; Python callers opt in
with ``smooth=True`` and provide frame timestamps in seconds.

``clip_quality_control`` evaluates the completed clip and reports quality
issues without modifying predictions. The video runner enables it with
``--postprocessing_qc``; it is off by default.

``world_coordinate_projection`` is a separate, standalone step: it fuses
already-written EgoHand hand-result NPZs with a camera-to-world pose track (e.g.
from ViPE v2) to produce stabilised world-frame hand keypoints. Unlike
``adaptive_smoothing`` it is not wired into ``run_egohand_inference_pipeline.py``
(it needs a pose track that script has no access to) and is run separately,
either as a library call or via its own CLI
(``python inference/postprocessing/world_coordinate_projection.py``).
"""

from inference.postprocessing.clip_quality_control import check_hand_results
from inference.postprocessing.adaptive_smoothing import (
    AdaptiveSmoothingConfig,
    smooth_adaptive_hand_parameters,
    smooth_adaptive_hand_results,
)
from inference.postprocessing.world_coordinate_projection import (
    PoseRefinementResult,
    load_hand_npz,
    load_pose_inds,
    load_pose_npz,
    refine_camera_poses,
    refine_camera_poses_from_npz,
)

__all__ = [
    "AdaptiveSmoothingConfig",
    "smooth_adaptive_hand_parameters",
    "smooth_adaptive_hand_results",
    "check_hand_results",
    "PoseRefinementResult",
    "refine_camera_poses",
    "refine_camera_poses_from_npz",
    "load_hand_npz",
    "load_pose_npz",
    "load_pose_inds",
]
