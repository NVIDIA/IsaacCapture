# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

# The plugin's side of the conversation with readers: serves a Publisher's groups over a socket.
add_library(camera_endpoint STATIC
    endpoint.cpp
)

target_include_directories(camera_endpoint INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera_endpoint
    PUBLIC
        camera::camera_core
    PRIVATE
        camera::camera_ipc
        camera::camera_protocol
        log_bridge::log_bridge_core
        Threads::Threads
)

add_library(camera::camera_endpoint ALIAS camera_endpoint)
