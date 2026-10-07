# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""RT-DETRv2 ONNX hand bbox detector, the bundled optional 2D detector.

``detector.RTDetrHandBBoxDetector`` satisfies the ``HandBBoxDetector`` Protocol
in ``hand_detector_2d.hand_bbox_detector``, so it drops into
``EgoHandInferencePipeline`` wherever any other conforming detector goes.

``onnx_model`` and ``tracking`` are vendored from the standalone
``hand_detector`` package in the ego-reconstruction repo; see their docstrings
for the deliberate deviations from upstream.
"""

from .detector import RTDetrConfig, RTDetrHandBBoxDetector

__all__ = ["RTDetrConfig", "RTDetrHandBBoxDetector"]
