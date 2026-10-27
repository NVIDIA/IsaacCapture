# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep upstream extension headers separate from NVIDIA headers for license attribution.

add_library(teleop_openxr_extensions_upstream INTERFACE)
add_library(Teleop::openxr_extensions_upstream ALIAS teleop_openxr_extensions_upstream)

target_include_directories(teleop_openxr_extensions_upstream INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
