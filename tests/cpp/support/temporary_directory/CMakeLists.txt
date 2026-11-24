# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(test_temporary_directory INTERFACE)
target_include_directories(test_temporary_directory INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)
add_library(test_support::temporary_directory ALIAS test_temporary_directory)
