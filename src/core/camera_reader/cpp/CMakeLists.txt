# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# What an application links to read cameras from the camera plugin.
add_library(camera_reader STATIC
    camera_reader.cpp
    plugin_launcher.cpp
    plugin_launcher.hpp
)

target_include_directories(camera_reader INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera_reader
    PUBLIC
        camera::camera_core
    PRIVATE
        camera::camera_ipc
        camera::camera_wire
)

add_library(camera::camera_reader ALIAS camera_reader)
