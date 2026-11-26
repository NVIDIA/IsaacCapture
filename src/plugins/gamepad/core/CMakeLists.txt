# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The SDL3 reader, kept apart from the OpenXR push loop so it can be unit-tested against SDL's
# virtual gamepads without a runtime or a device.
add_library(gamepad_core STATIC
    gamepad_connection.cpp
    inc/gamepad/gamepad_connection.hpp
)

target_include_directories(gamepad_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(gamepad_core
    PUBLIC
        log_bridge::log_bridge_core
        isaacteleop_schema
    PRIVATE
        SDL3::SDL3-static
)

add_library(gamepad::gamepad_core ALIAS gamepad_core)
