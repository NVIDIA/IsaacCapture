# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# How camera frames cross between processes on one machine: pools of shared host memory that
# readers can only map read-only. Linux only.
add_library(camera_ipc STATIC
    host_pool.cpp
)

target_include_directories(camera_ipc INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

add_library(camera::camera_ipc ALIAS camera_ipc)
