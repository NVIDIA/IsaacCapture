# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# DeviceIOSession — wires trackers to the live factory and runs the update loop.
# ReplaySession — reads recorded tracker data from an MCAP file.
# RemoteSession — reads tracker data from a sample stream.
add_library(deviceio_session STATIC
    mcap_impl.cpp
    deviceio_session.cpp
    replay_session.cpp
    remote_session.cpp
    inc/deviceio_session/deviceio_session.hpp
    inc/deviceio_session/replay_session.hpp
    inc/deviceio_session/remote_session.hpp
)

target_include_directories(deviceio_session
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

# Public headers include oxr_funcs.hpp and oxr_session_handles.hpp. Propagate
# oxr::oxr_utils (OpenXR headers + oxr inc/) and Teleop::openxr_extensions (NVIDIA
# extension headers) so consumers (Python bindings, apps) compile without linking
# live_trackers.
target_link_libraries(deviceio_session
    PUBLIC
        deviceio::deviceio_base
        oxr::oxr_utils
        Teleop::openxr_extensions
        log_bridge::log_bridge_core
    PRIVATE
        deviceio::deviceio_trackers
        deviceio::live_trackers
        deviceio::replay_trackers
        mcap::mcap
        mcap::mcap_core
)

add_library(deviceio::deviceio_session ALIAS deviceio_session)
