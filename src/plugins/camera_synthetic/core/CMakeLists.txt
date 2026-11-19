# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

# The plugin's capture code: a camera that needs no hardware, frames with known content at a
# set rate. Kept apart from the executable so that the bench and the tests can check a frame
# against the same pattern.
add_library(camera_synthetic_core STATIC
    synthetic_backend.cpp
    inc/camera_synthetic/synthetic_backend.hpp
)

target_include_directories(camera_synthetic_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera_synthetic_core
    PUBLIC
        camera::camera
    PRIVATE
        Threads::Threads
)

add_library(camera_synthetic::camera_synthetic_core ALIAS camera_synthetic_core)
