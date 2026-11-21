# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Robot state, robot end: declares data series and wraps record bytes in the MCAP Schema, Channel and Message
# records of the robot-state link message payloads. Writes those records itself: MCAP-library-, schema-, OpenXR-
# and DeviceIO-free.
add_library(robot_state_publisher STATIC
    robot_state_publisher.cpp
    inc/robot_state_publisher/robot_state_publisher.hpp
)

target_include_directories(robot_state_publisher
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(robot_state_publisher
    PRIVATE
        log_bridge::log_bridge_core
)

add_library(robot_state::robot_state_publisher ALIAS robot_state_publisher)
