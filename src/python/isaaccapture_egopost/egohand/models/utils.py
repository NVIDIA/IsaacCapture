# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Small model-side helpers shared across hand/body networks."""

import torch.nn as nn


def zero_module(module: nn.Module) -> nn.Module:
    """Zero all parameters of ``module`` and return it.

    Used to make a new modality / fusion projection neutral at step 0 so that
    a model with the new modality enabled produces identical predictions to
    its modality-free baseline until the optimizer learns to use the new
    signal. A common zero-module fusion pattern
    (``Sequential(..., zero_module(Linear(...)))``).

    When a checkpoint is loaded after instantiation, the saved values
    overwrite these zeros — so this only affects the very first step of a
    fresh training run with the new modality.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
