# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Carry the upstream headers through this target to preserve the public interface.
add_library(teleop_openxr_extensions INTERFACE)
add_library(Teleop::openxr_extensions ALIAS teleop_openxr_extensions)

target_include_directories(teleop_openxr_extensions INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(teleop_openxr_extensions INTERFACE
    Teleop::openxr_extensions_upstream
)
