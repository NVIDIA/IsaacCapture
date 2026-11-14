# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Turns the messages between the camera plugin and its readers into bytes and back. The
# FlatBuffers types stay inside this library; its users see camera_core types.
add_library(camera_wire STATIC
    codec.cpp
)

target_include_directories(camera_wire INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera_wire
    PUBLIC
        camera::camera_core
    PRIVATE
        camera::camera_wire_schema
)

add_library(camera::camera_wire ALIAS camera_wire)
