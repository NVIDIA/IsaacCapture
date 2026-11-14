# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(camera_plugin_tests
    test_camera_host.cpp
)

target_link_libraries(camera_plugin_tests PRIVATE
    camera::plugin_core
    camera::camera_reader
    Catch2::Catch2WithMain
)

message(STATUS "camera_plugin_tests target enabled with Catch2")
catch_discover_tests(camera_plugin_tests ADD_TAGS_AS_LABELS)
