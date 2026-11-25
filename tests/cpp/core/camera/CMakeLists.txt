# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(camera_tests
    test_endpoint_path.cpp
    test_host_pool.cpp
    test_message_socket.cpp
    test_publisher.cpp
)

target_link_libraries(camera_tests PRIVATE
    camera::camera
    Catch2::Catch2WithMain
    Threads::Threads
)

message(STATUS "camera_tests target enabled with Catch2")
catch_discover_tests(camera_tests ADD_TAGS_AS_LABELS)
