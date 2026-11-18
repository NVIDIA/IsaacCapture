# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""One ``build(cfg, device, batch_size, tracking) -> HandBBoxDetector`` module per
detector_backend, dynamically imported by name from
``run_egohand_inference_pipeline.py::_build_detector_from_config``. That script never
imports a backend module or class directly -- it only knows the naming
convention ``inference.hand_detector_2d.configs.<detector_backend>``. Add a new
backend by adding ``<name>.py`` here with a ``build`` function; nothing in the
pipeline script needs to change.

Bbox detector YAML configs (``rtdetr.yaml``, ``saved_bboxes.yaml``, ...) live
alongside their matching builder module in this same directory.
"""
