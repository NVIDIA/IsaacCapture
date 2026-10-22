# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(remote_plugin_session_tests
    test_websocket_transport.cpp
    test_wire_codec.cpp
)

target_link_libraries(remote_plugin_session_tests PRIVATE
    Teleop::remote_plugin_session
    pusherio::pusherio
    Catch2::Catch2WithMain
)

catch_discover_tests(remote_plugin_session_tests ADD_TAGS_AS_LABELS)
