# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

add_library(remote_device_bridge_server STATIC
    bridge_server.cpp
    inc/remote_device_bridge/bridge_server.hpp
    latency_stats.cpp
    inc/remote_device_bridge/latency_stats.hpp
)

target_include_directories(remote_device_bridge_server
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(remote_device_bridge_server
    PUBLIC
        Teleop::remote_plugin_session
    PRIVATE
        log_bridge::log_bridge_core
        Threads::Threads
)

add_library(Teleop::remote_device_bridge_server ALIAS remote_device_bridge_server)
