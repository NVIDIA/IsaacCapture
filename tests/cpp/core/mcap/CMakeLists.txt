# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Temp-file and reader scaffolding shared with other suites that write recordings.
add_library(mcap_test_support INTERFACE)
target_include_directories(mcap_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcap_test_support INTERFACE
    deviceio::deviceio_session
    mcap::mcap_core
    Catch2::Catch2
)

add_executable(mcap_tests
    test_mcap_tracker_channels.cpp
    test_schema_compat.cpp
)

target_link_libraries(mcap_tests PRIVATE
    mcap_test_support
    Catch2::Catch2WithMain
)

message(STATUS "mcap_tests target enabled with Catch2")
catch_discover_tests(mcap_tests)
