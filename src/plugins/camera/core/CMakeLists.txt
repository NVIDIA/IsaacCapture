# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

# What a camera backend implements, and the host that gives each camera its pools and puts
# them all behind one endpoint.
add_library(camera_plugin_core STATIC
    camera_host.cpp
    config.cpp
    options.cpp
)

target_include_directories(camera_plugin_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera_plugin_core
    PUBLIC
        camera::camera_core
    PRIVATE
        camera::camera_endpoint
        camera::camera_ipc
        log_bridge::log_bridge_core
        yaml-cpp::yaml-cpp
        Threads::Threads
)

add_library(camera::plugin_core ALIAS camera_plugin_core)
