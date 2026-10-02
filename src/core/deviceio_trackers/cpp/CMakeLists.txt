# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Public tracker classes (HeadTracker, HandTracker, etc.)
add_library(deviceio_trackers STATIC
    hand_tracker.cpp
    head_tracker.cpp
    controller_tracker.cpp
    message_channel_tracker.cpp
    tensor_push_tracker.cpp
    full_body_tracker.cpp
    haptic_command_reader_tracker.cpp
    keyboard_tracker.cpp
    ${GENERATED_TRACKER_FACADE_SOURCES}
    inc/deviceio_trackers/head_tracker.hpp
    inc/deviceio_trackers/hand_tracker.hpp
    inc/deviceio_trackers/controller_tracker.hpp
    inc/deviceio_trackers/message_channel_tracker.hpp
    inc/deviceio_trackers/full_body_tracker.hpp
    inc/deviceio_trackers/tensor_push_tracker.hpp
    inc/deviceio_trackers/haptic_command_reader_tracker.hpp
    inc/deviceio_trackers/keyboard_tracker.hpp
)

target_include_directories(deviceio_trackers
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<BUILD_INTERFACE:${GENERATED_TRACKER_DEVICEIO_TRACKERS_INC_DIR}>
        $<BUILD_INTERFACE:${GENERATED_TRACKER_DEVICEIO_BASE_INC_DIR}>
)

target_link_libraries(deviceio_trackers
    PUBLIC
        deviceio::deviceio_base
)

add_library(deviceio::deviceio_trackers ALIAS deviceio_trackers)
