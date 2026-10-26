# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source semantics are registered independently of geometry mappings."""

from .hand import HandProfile

profile_registry = {"HandSkeletonRef": HandProfile()}


def profile_for(device):
    name = device.profile.get("type")
    if name not in profile_registry:
        raise ValueError(f"Unsupported segment source: {name}")
    return profile_registry[name]
