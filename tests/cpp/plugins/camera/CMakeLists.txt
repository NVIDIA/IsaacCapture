# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(camera_plugin_tests
    test_camera_bench.cpp
    test_camera_host.cpp
    test_camera_plugin.cpp
)

target_link_libraries(camera_plugin_tests PRIVATE
    camera::plugin_core
    camera::plugin_backends
    camera::backend_synthetic
    camera::camera_reader
    Catch2::Catch2WithMain
    Threads::Threads
)

# Some cases start the plugin or the bench as a process of its own.
target_compile_definitions(camera_plugin_tests PRIVATE
    CAMERA_PLUGIN_EXECUTABLE="$<TARGET_FILE:camera_plugin>"
    CAMERA_BENCH_EXECUTABLE="$<TARGET_FILE:camera_bench>"
)
add_dependencies(camera_plugin_tests camera_plugin camera_bench)

message(STATUS "camera_plugin_tests target enabled with Catch2")
catch_discover_tests(camera_plugin_tests ADD_TAGS_AS_LABELS)
