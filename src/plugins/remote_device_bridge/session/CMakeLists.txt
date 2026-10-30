# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Transport-only library shared by the bridge, the synthetic sender and any
# plugin that offers --remote. It must never link the OpenXR loader or
# plugin_utils: plugin processes using it stay free of an OpenXR runtime.
add_library(remote_plugin_session STATIC
    protocol_frame.cpp
    protocol_payloads.cpp
    protocol_io.hpp
    inc/remote_plugin_session/protocol_frame.hpp
    inc/remote_plugin_session/protocol_payloads.hpp
)

target_include_directories(remote_plugin_session
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(remote_plugin_session
    PUBLIC
        pusherio::pusherio
)

add_library(Teleop::remote_plugin_session ALIAS remote_plugin_session)
