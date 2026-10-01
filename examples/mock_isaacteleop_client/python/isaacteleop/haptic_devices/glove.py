# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Haptic glove stand-in. Per ``GumanIsaacTeleopUsage.md`` §8, this path is only exercised by
``exo`` stations and Guman never reads anything back -- a no-op store is sufficient. Exact
construction kwargs aren't pinned down by that spec (not exercised in the non-exo path it was
read from), so this accepts anything rather than guess wrong and crash.
"""

from typing import Any


class _HapticGloveDevice:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.last_tactile = None

    def set_tactile(self, values: Any) -> None:
        self.last_tactile = values


def haptic_glove_device(*args: Any, **kwargs: Any) -> _HapticGloveDevice:
    return _HapticGloveDevice(*args, **kwargs)
