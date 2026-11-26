# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Conditional MLP modules for TransformerDecoder."""

import copy
from typing import Optional

import torch


class AdaptiveLayerNorm1D(torch.nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm = torch.nn.LayerNorm(data_dim)
        self.linear = torch.nn.Linear(norm_cond_dim, 2 * data_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])
        return x * (1 + alpha) + beta


class SequentialCond(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(
                module, (AdaptiveLayerNorm1D, SequentialCond, ResidualMLPBlock)
            ):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1):
    if norm == "batch":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer":
        return torch.nn.LayerNorm(dim)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    elif norm is None:
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


class ResidualMLPBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_hidden_layers,
        output_dim,
        activation=None,
        bias=True,
        norm="layer",
        dropout=0.0,
        norm_cond_dim=-1,
    ):
        super().__init__()
        if activation is None:
            activation = torch.nn.ReLU()
        if not (input_dim == output_dim == hidden_dim):
            raise NotImplementedError(
                f"input_dim {input_dim} != output_dim {output_dim}"
            )
        layers = []
        prev_dim = input_dim
        for _i in range(num_hidden_layers):
            layers.append(
                _linear_norm_activ_dropout(
                    prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
                )
            )
            prev_dim = hidden_dim
        self.model = SequentialCond(*layers)
        self.skip = torch.nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.model(x, *args, **kwargs)


def _linear_norm_activ_dropout(
    input_dim, output_dim, activation, bias, norm, dropout, norm_cond_dim
):
    layers = []
    layers.append(torch.nn.Linear(input_dim, output_dim, bias=bias))
    if norm is not None:
        layers.append(normalization_layer(norm, output_dim, norm_cond_dim))
    layers.append(copy.deepcopy(activation))
    if dropout > 0.0:
        layers.append(torch.nn.Dropout(dropout))
    return SequentialCond(*layers)


class FrequencyEmbedder(torch.nn.Module):
    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        N = x.size(0)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_unsqueezed = x.unsqueeze(-1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(N, -1)
        return embedded
