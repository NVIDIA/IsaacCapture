# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(replay_deviceio_session_tests
    test_replay_session.cpp
    test_xdev_hand_tracker_selection.cpp
    test_live_hand_fallback.cpp
)

target_include_directories(replay_deviceio_session_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/core/live_trackers/cpp
)

target_link_libraries(replay_deviceio_session_tests PRIVATE
    deviceio::deviceio_session
    deviceio::deviceio_trackers
    deviceio::live_trackers
    mcap::mcap_core
    Catch2::Catch2WithMain
)

message(STATUS "replay_deviceio_session_tests target enabled with Catch2")
catch_discover_tests(replay_deviceio_session_tests)
