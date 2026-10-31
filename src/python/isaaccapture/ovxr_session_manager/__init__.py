# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OVXR presentation over application-owned rendering resources."""

from .ovxr_session import (
    OvxrSessionBinding,
    OvxrScenePublication,
    OvxrResourceOwner,
    OvxrSessionConfig,
    OvxrApplicationInterface,
    OvxrFrameConsumer,
    OvxrSession,
    StageOrdinalLedger,
)
from .desktop_mirror import DesktopMirrorConfig, DesktopMirrorStatus

__all__ = [
    "OvxrSessionBinding",
    "OvxrSessionConfig",
    "OvxrApplicationInterface",
    "OvxrFrameConsumer",
    "OvxrSession",
    "StageOrdinalLedger",
    "DesktopMirrorConfig",
    "DesktopMirrorStatus",
]

from .camera_panels import CameraPanelConfig, CameraPanelConsumer
from .camera_mailbox import FrameOutputs

__all__ += ["CameraPanelConfig", "CameraPanelConsumer", "FrameOutputs"]

__all__ += ["OvxrScenePublication", "OvxrResourceOwner"]
