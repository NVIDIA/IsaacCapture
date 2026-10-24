# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OVXR presentation over application-owned rendering resources."""

from .ovxr_session import (
    AuxiliaryRasterConfig,
    OvxrSessionBinding,
    OvxrSessionConfig,
    OvxrApplicationInterface,
    OvxrSession,
    StageOrdinalLedger,
)

__all__ = [
    "AuxiliaryRasterConfig",
    "OvxrSessionBinding",
    "OvxrSessionConfig",
    "OvxrApplicationInterface",
    "OvxrSession",
    "StageOrdinalLedger",
]
