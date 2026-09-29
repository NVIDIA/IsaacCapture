# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(live_trackers_tests
    test_vendor_validation.cpp
    test_openxr_free_session.cpp
)

target_link_libraries(live_trackers_tests PRIVATE
    deviceio::live_trackers
    deviceio::deviceio_trackers
    # Provides the MCAP_IMPLEMENTATION symbols (mcap::McapWriter::*) that
    # live_trackers' McapTrackerChannels instantiations reference; MCAP is
    # compiled once inside deviceio_session (see src/core/mcap/cpp/CMakeLists.txt).
    deviceio::deviceio_session
    Catch2::Catch2WithMain
)

message(STATUS "live_trackers_tests target enabled with Catch2")
catch_discover_tests(live_trackers_tests)
