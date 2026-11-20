# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(camera_tests
    test_camera_reader.cpp
    test_codec.cpp
    test_cross_process.cpp
    test_endpoint.cpp
    test_endpoint_path.cpp
    test_host_pool.cpp
    test_message_socket.cpp
    test_publisher.cpp
)

# The schema is linked to build what the codec never writes: messages from a newer peer and
# malformed ones.
target_link_libraries(camera_tests PRIVATE
    camera::camera
    isaacteleop_schema
    Catch2::Catch2WithMain
    Threads::Threads
)

# The integration cases run camera_test_peer as the other process.
add_subdirectory(test_peer)
target_compile_definitions(camera_tests PRIVATE
    CAMERA_TEST_PEER_EXECUTABLE="$<TARGET_FILE:camera_test_peer>"
)
add_dependencies(camera_tests camera_test_peer)
# That path is into the build tree, so this binary only runs where it was built: keep it off
# the venv PATH of a pip build.
set_target_properties(camera_tests PROPERTIES ISAACTELEOP_NEEDS_SOURCE_TREE TRUE)

message(STATUS "camera_tests target enabled with Catch2")
catch_discover_tests(camera_tests ADD_TAGS_AS_LABELS)
