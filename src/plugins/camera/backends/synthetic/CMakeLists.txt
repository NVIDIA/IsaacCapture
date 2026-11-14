# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

# A camera that needs no hardware: frames with known content at a set rate.
add_library(camera_backend_synthetic STATIC
    synthetic_backend.cpp
)

target_include_directories(camera_backend_synthetic INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera_backend_synthetic
    PUBLIC
        camera::plugin_core
    PRIVATE
        Threads::Threads
)

add_library(camera::backend_synthetic ALIAS camera_backend_synthetic)
