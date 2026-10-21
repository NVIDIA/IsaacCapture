# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(pusherio_tests
    test_hand_tracking_pusher.cpp
    test_plugin_pull_channel.cpp
    test_schema_pusher_session.cpp
)

target_link_libraries(pusherio_tests PRIVATE
    deviceio::deviceio_trackers
    pusherio::pusherio
    Catch2::Catch2WithMain
)

catch_discover_tests(pusherio_tests ADD_TAGS_AS_LABELS)
