# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Robot state, server end: each data series' declaration and its newest record, with ages on this machine's clock.
# Reads the MCAP Schema, Channel and Message records of the robot-state link message payloads itself: MCAP-library-,
# schema-, OpenXR- and DeviceIO-free. Every time is an argument, so it reads no clock.
add_library(robot_state_store STATIC
    robot_state_store.cpp
    inc/robot_state_store/robot_state_store.hpp
)

target_include_directories(robot_state_store
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(robot_state_store
    PRIVATE
        log_bridge::log_bridge_core
)

add_library(robot_state::robot_state_store ALIAS robot_state_store)
