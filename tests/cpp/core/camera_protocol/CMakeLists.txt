# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(camera_protocol_tests
    test_codec.cpp
)

# The schema is linked to build messages the encoder never would: ones from a newer peer and
# malformed ones.
target_link_libraries(camera_protocol_tests PRIVATE
    camera::camera_protocol
    isaacteleop_schema
    Catch2::Catch2WithMain
)

message(STATUS "camera_protocol_tests target enabled with Catch2")
catch_discover_tests(camera_protocol_tests ADD_TAGS_AS_LABELS)
