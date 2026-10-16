# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure the session-wide ``isaaccapture`` logger tree.

The leader configures console/file handlers and, where supported, a Unix-socket
receiver for C++ and child-process records.
``isaaccapture.__init__`` installs them once; importing this package alone does
nothing. Raw fd 1/2 output is captured separately by ``_native_fd.py``.
"""

from ._console import (
    set_console_content_filter,
    set_console_format,
    set_console_level,
    set_console_logger_name_filter,
    set_logger_colors,
)

# Internal bootstrap; intentionally excluded from ``__all__``.
from ._setup import install as install
from ._setup import set_propagate_to_root

# Public configuration API.
__all__ = [
    "set_console_content_filter",
    "set_console_format",
    "set_console_level",
    "set_console_logger_name_filter",
    "set_logger_colors",
    "set_propagate_to_root",
]
