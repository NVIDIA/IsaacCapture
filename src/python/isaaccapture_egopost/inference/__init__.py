# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""EgoHand hand inference: model wrapper, 2D hand bbox detection, and the pipeline that joins them.

``pipeline.pipeline.EgoHandInferencePipeline`` is the entrypoint. It pairs a
``pipeline.predictor.EgoHandPredictor`` with an optional
``hand_detector_2d.hand_bbox_detector.HandBBoxDetector`` and runs EgoHand/SOMA
inference over image batches. Input-frame and calibration code lives in
``camera/``; hand-crop preparation and right-hand mirroring live in
``preprocessing/``.

The ``hand_detector_2d.rtdetr`` package (RT-DETRv2 ONNX) implements that
Protocol, selected via ``--bbox_detector_config``.

The command-line entry point is ``scripts/run_egohand_inference_pipeline.py``.
"""
