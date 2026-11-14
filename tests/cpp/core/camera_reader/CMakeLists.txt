# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(camera_reader_tests
    test_camera_reader.cpp
)

# The reader is tested against a real endpoint, and against a stand-in plugin built from the
# socket and the codec for what a real one does not do on demand.
target_link_libraries(camera_reader_tests PRIVATE
    camera::camera_reader
    camera::camera_endpoint
    camera::camera_ipc
    camera::camera_wire
    Catch2::Catch2WithMain
    Threads::Threads
)

message(STATUS "camera_reader_tests target enabled with Catch2")
catch_discover_tests(camera_reader_tests ADD_TAGS_AS_LABELS)
