# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(deviceio_trackers_tests
    test_keyboard_tracker.cpp
)

target_link_libraries(deviceio_trackers_tests PRIVATE
    deviceio::deviceio_trackers
    Catch2::Catch2WithMain
)

message(STATUS "deviceio_trackers_tests target enabled with Catch2")
catch_discover_tests(deviceio_trackers_tests)
