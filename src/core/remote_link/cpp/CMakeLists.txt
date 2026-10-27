# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

# Remote session link: link message header and types, TCP/WebSocket transports, Link threads, clock offset.
# MCAP-, OpenXR- and DeviceIO-free; socket code is POSIX-only (Windows bodies throw).
add_library(remote_link STATIC
    byte_order.hpp
    clock_offset.cpp
    link.cpp
    message_header.cpp
    messages.cpp
    reject.cpp
    tcp_transport.cpp
    websocket_transport.cpp
    ws_handshake.cpp
    ws_handshake.hpp
    inc/remote_link/clock_offset.hpp
    inc/remote_link/link.hpp
    inc/remote_link/message_header.hpp
    inc/remote_link/messages.hpp
    inc/remote_link/reject.hpp
    inc/remote_link/tcp_transport.hpp
    inc/remote_link/transport.hpp
    inc/remote_link/websocket_transport.hpp
)

target_include_directories(remote_link
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

# oxr_utils is used for os_time.hpp only (header-only, no OpenXR loader).
target_link_libraries(remote_link
    PUBLIC
        Threads::Threads
    PRIVATE
        oxr::oxr_utils
        log_bridge::log_bridge_core
)

add_library(remote_link::remote_link ALIAS remote_link)
