# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(camera_reader_tests
    test_camera_reader.cpp
    test_cross_process.cpp
)

# The reader is tested against a real endpoint, and against a stand-in plugin built from the
# socket and the codec for what a real one does not do on demand. The schema is linked for
# what the codec never writes: a group from a newer plugin.
target_link_libraries(camera_reader_tests PRIVATE
    camera::camera_reader
    camera::camera_endpoint
    camera::camera_ipc
    camera::camera_protocol
    isaacteleop_schema
    Catch2::Catch2WithMain
    Threads::Threads
)

# The integration cases run camera_test_peer as the other process.
target_compile_definitions(camera_reader_tests PRIVATE
    CAMERA_TEST_PEER_EXECUTABLE="$<TARGET_FILE:camera_test_peer>"
)
add_dependencies(camera_reader_tests camera_test_peer)

message(STATUS "camera_reader_tests target enabled with Catch2")
catch_discover_tests(camera_reader_tests ADD_TAGS_AS_LABELS)
