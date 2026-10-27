# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(remote_deviceio_session_tests
    test_remote_session.cpp
)

target_link_libraries(remote_deviceio_session_tests PRIVATE
    deviceio::deviceio_session
    deviceio::deviceio_trackers
    mcap::mcap_core
    Catch2::Catch2WithMain
)

message(STATUS "remote_deviceio_session_tests target enabled with Catch2")
catch_discover_tests(remote_deviceio_session_tests)
