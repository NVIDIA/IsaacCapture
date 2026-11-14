# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Turns the messages between the camera plugin and its readers into bytes and back. The
# schema is schema/fbs/camera_protocol.fbs; its FlatBuffers types stay inside this library,
# and its users see camera_core types.
add_library(camera_protocol STATIC
    codec.cpp
)

target_include_directories(camera_protocol INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera_protocol
    PUBLIC
        camera::camera_core
    PRIVATE
        isaacteleop_schema
)

add_library(camera::camera_protocol ALIAS camera_protocol)
