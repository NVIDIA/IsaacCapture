# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(pusherio_tests
    test_openxr_schema_push_channel.cpp
)

target_link_libraries(pusherio_tests PRIVATE
    pusherio::pusherio
    Catch2::Catch2WithMain
)

message(STATUS "pusherio_tests target enabled with Catch2")
catch_discover_tests(pusherio_tests ADD_TAGS_AS_LABELS)
