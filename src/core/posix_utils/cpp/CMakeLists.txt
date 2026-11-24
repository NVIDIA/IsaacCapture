# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(posix_utils INTERFACE)
target_include_directories(posix_utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)
add_library(posix_utils::posix_utils ALIAS posix_utils)
