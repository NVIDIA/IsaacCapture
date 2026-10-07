# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# OpenXR extension headers written by other people, vendored here verbatim:
#
#   openxr_extension_helpers.h   The Khronos Group Inc.   Apache-2.0
#   XR_MNDX_xdev_space.h         Collabora, Ltd.          BSL-1.0
#
# Kept apart from deps/cloudxr/openxr_extensions, which holds NVIDIA's own, so
# that what this project redistributes under someone else's terms is visible
# from the directory layout rather than only from the file headers.

add_library(teleop_openxr_extensions_upstream INTERFACE)
add_library(Teleop::openxr_extensions_upstream ALIAS teleop_openxr_extensions_upstream)

target_include_directories(teleop_openxr_extensions_upstream INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
