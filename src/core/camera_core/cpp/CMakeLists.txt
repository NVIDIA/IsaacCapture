# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Camera capture core: which pool slots hold which frame sets and who may still read them.
# Standard library only, so it builds and is tested on every platform.
add_library(camera_core STATIC
    description.cpp
    publisher.cpp
)

target_include_directories(camera_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

add_library(camera::camera_core ALIAS camera_core)
