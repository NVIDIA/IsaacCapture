# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

add_executable(remote_plugin_session_tests
    test_bridge_limits.cpp
    test_clock_offset_estimator.cpp
    test_latency_stats.cpp
    test_remote_address.cpp
    test_remote_plugin_session.cpp
    test_tcp_transport.cpp
    test_websocket_transport.cpp
    test_wire_codec.cpp
)

target_link_libraries(remote_plugin_session_tests PRIVATE
    Teleop::remote_plugin_session
    Threads::Threads
    Catch2::Catch2WithMain
)

catch_discover_tests(remote_plugin_session_tests ADD_TAGS_AS_LABELS)
