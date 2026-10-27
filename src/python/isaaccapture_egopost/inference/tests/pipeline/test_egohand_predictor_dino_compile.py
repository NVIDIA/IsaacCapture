# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from types import SimpleNamespace

import pytest

from inference.pipeline.predictor import EgoHandPredictor


def _predictor(backbone, batch_size=640):
    predictor = object.__new__(EgoHandPredictor)
    predictor.model = SimpleNamespace(backbone=backbone)
    predictor.batch_size = batch_size
    return predictor


def test_predictor_configures_dino_compile_for_its_batch_size():
    configured = []
    backbone = SimpleNamespace(enable_full_batch_compile=configured.append)

    _predictor(backbone).enable_dino_compile()

    assert configured == [640]


def test_predictor_rejects_dino_compile_for_an_unsupported_backbone():
    with pytest.raises(RuntimeError, match="DINOv3 backbone"):
        _predictor(object()).enable_dino_compile()
