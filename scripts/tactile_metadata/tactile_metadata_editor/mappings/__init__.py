# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry mappings have their own registry, independent of source semantics."""

from .lbs import LBSMapping

mapping_registry = {"LBS": LBSMapping()}


def mapping_for(device):
    name = device.mapping.get("type")
    if name not in mapping_registry:
        raise ValueError(f"Unsupported geometry mapping: {name}")
    return mapping_registry[name]
