# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""See ``haptic_devices.glove`` -- same no-op-store rationale applies here."""

from typing import Any


class TactileVectorToFingerPower:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.last_tactile = None

    def set_tactile(self, values: Any) -> None:
        self.last_tactile = values
