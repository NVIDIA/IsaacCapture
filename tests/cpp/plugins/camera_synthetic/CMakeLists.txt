# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(camera_synthetic_tests
    test_camera_bench.cpp
    test_camera_synthetic.cpp
)

target_link_libraries(camera_synthetic_tests PRIVATE
    camera_synthetic::camera_synthetic_core
    camera::camera
    teleop_plugin_manager
    Catch2::Catch2WithMain
    Threads::Threads
)

# Some cases start the plugin or the bench as a process of its own.
target_compile_definitions(camera_synthetic_tests PRIVATE
    CAMERA_SYNTHETIC_EXECUTABLE="$<TARGET_FILE:camera_synthetic_plugin>"
    CAMERA_SYNTHETIC_DIRECTORY="$<TARGET_FILE_DIR:camera_synthetic_plugin>"
    CAMERA_BENCH_EXECUTABLE="$<TARGET_FILE:camera_bench>"
)
add_dependencies(camera_synthetic_tests camera_synthetic_plugin camera_bench)
# Those paths are into the build tree, so this binary only runs where it was built: keep it
# off the venv PATH of a pip build.
set_target_properties(camera_synthetic_tests PROPERTIES ISAACTELEOP_NEEDS_SOURCE_TREE TRUE)

message(STATUS "camera_synthetic_tests target enabled with Catch2")
catch_discover_tests(camera_synthetic_tests ADD_TAGS_AS_LABELS)
