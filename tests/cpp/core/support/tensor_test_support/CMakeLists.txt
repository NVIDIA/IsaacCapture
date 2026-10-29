# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Header-only INTERFACE library exposing <tensor_test_support/fake_tensor_runtime.hpp>: tensor collections
# served from memory, so a live schema tracker runs without an OpenXR runtime.
add_library(tensor_test_support INTERFACE)
target_include_directories(tensor_test_support INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)
target_link_libraries(tensor_test_support INTERFACE
    oxr::oxr_utils
    Teleop::openxr_extensions
)
add_library(tensor_test_support::tensor_test_support ALIAS tensor_test_support)
