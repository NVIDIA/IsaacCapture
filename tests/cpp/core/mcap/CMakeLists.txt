# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(mcap_tests
    mcap_test_support.hpp
    test_guman_manus_glove_channels.cpp
    test_mcap_tracker_channels.cpp
    test_schema_compat.cpp
)

target_link_libraries(mcap_tests PRIVATE
    deviceio::deviceio_session
    mcap::mcap_core
    Catch2::Catch2WithMain
)

message(STATUS "mcap_tests target enabled with Catch2")
catch_discover_tests(mcap_tests)
