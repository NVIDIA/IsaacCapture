# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(camera_ipc_tests
    test_host_pool.cpp
)

target_link_libraries(camera_ipc_tests PRIVATE
    camera::camera_ipc
    Catch2::Catch2WithMain
)

message(STATUS "camera_ipc_tests target enabled with Catch2")
catch_discover_tests(camera_ipc_tests ADD_TAGS_AS_LABELS)
