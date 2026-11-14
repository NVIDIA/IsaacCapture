# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(camera_endpoint_tests
    test_endpoint.cpp
)

# The tests play the reader's part with the socket and the codec themselves.
target_link_libraries(camera_endpoint_tests PRIVATE
    camera::camera_endpoint
    camera::camera_ipc
    camera::camera_wire
    Catch2::Catch2WithMain
    Threads::Threads
)

message(STATUS "camera_endpoint_tests target enabled with Catch2")
catch_discover_tests(camera_endpoint_tests ADD_TAGS_AS_LABELS)
