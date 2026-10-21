# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

# Transport-only library shared by the bridge, the synthetic sender and any
# plugin that offers --remote. It must never link the OpenXR loader or
# plugin_utils: plugin processes using it stay free of an OpenXR runtime.
add_library(remote_plugin_session STATIC
    inc/remote_plugin_session/bridge_limits.hpp
    clock_offset_estimator.cpp
    inc/remote_plugin_session/clock_offset_estimator.hpp
    latency_stats.cpp
    inc/remote_plugin_session/latency_stats.hpp
    remote_plugin_session.cpp
    inc/remote_plugin_session/remote_plugin_session.hpp
    tcp_socket.cpp
    inc/remote_plugin_session/tcp_socket.hpp
    inc/remote_plugin_session/transport.hpp
    websocket_transport.cpp
    inc/remote_plugin_session/websocket_transport.hpp
    wire_codec.cpp
    inc/remote_plugin_session/wire_codec.hpp
)

target_include_directories(remote_plugin_session
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(remote_plugin_session
    PUBLIC
        pusherio::pusherio
    PRIVATE
        log_bridge::log_bridge_core
        Threads::Threads
)

add_library(Teleop::remote_plugin_session ALIAS remote_plugin_session)
