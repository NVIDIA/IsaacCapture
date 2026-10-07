# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NVIDIA's own OpenXR extension headers. The ones written by Khronos and
# Collabora live in deps/third_party/openxr_extensions; this target carries them
# so that linking Teleop::openxr_extensions still reaches every extension header.
add_library(teleop_openxr_extensions INTERFACE)
add_library(Teleop::openxr_extensions ALIAS teleop_openxr_extensions)

target_include_directories(teleop_openxr_extensions INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(teleop_openxr_extensions INTERFACE
    Teleop::openxr_extensions_upstream
)
