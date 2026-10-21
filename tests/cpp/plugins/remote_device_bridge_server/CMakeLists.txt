# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(remote_device_bridge_server_tests
    test_bridge_server.cpp
)

target_link_libraries(remote_device_bridge_server_tests PRIVATE
    Teleop::remote_device_bridge_server
    Teleop::remote_plugin_session
    Catch2::Catch2WithMain
)

catch_discover_tests(remote_device_bridge_server_tests ADD_TAGS_AS_LABELS)
