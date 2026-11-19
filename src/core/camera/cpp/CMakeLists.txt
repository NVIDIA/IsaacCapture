# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# The camera library; this module's README.md says what is in it. Linux only.
add_library(camera STATIC
    description.cpp
    endpoint_path.cpp
    host_pool.cpp
    message_socket.cpp
    publisher.cpp
    inc/camera/endpoint_path.hpp
    inc/camera/host_pool.hpp
    inc/camera/message_socket.hpp
    inc/camera/publisher.hpp
    inc/camera/types.hpp
    inc/camera/unique_fd.hpp
)

target_include_directories(camera INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

add_library(camera::camera ALIAS camera)
