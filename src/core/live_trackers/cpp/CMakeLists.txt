# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_library(live_trackers STATIC
    schema_tracker_base.cpp
    live_deviceio_factory.cpp
    live_head_tracker_impl.cpp
    live_hand_tracker_impl.cpp
    live_controller_tracker_impl.cpp
    live_message_channel_tracker_impl.cpp
    live_full_body_tracker_pico_impl.cpp
    live_full_body_tracker_noitom_impl.cpp
    live_full_body_tracker_xsens_impl.cpp
    live_tensor_push_tracker_impl.cpp
    live_haptic_command_reader_tracker_impl.cpp
    live_keyboard_tracker_impl.cpp
    live_gamepad_tracker_impl.cpp
    ${GENERATED_TRACKER_LIVE_SOURCES}
    inc/live_trackers/schema_tracker_base.hpp
    inc/live_trackers/schema_tracker.hpp
    inc/live_trackers/live_deviceio_factory.hpp
    live_head_tracker_impl.hpp
    live_hand_tracker_impl.hpp
    live_controller_tracker_impl.hpp
    live_message_channel_tracker_impl.hpp
    live_full_body_tracker_pico_impl.hpp
    live_full_body_tracker_noitom_impl.hpp
    live_full_body_tracker_xsens_impl.hpp
    live_tensor_push_tracker_impl.hpp
    live_haptic_command_reader_tracker_impl.hpp
    live_keyboard_tracker_impl.hpp
    live_gamepad_tracker_impl.hpp
)

target_include_directories(live_trackers
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<BUILD_INTERFACE:${GENERATED_TRACKER_INC_DIR}>
        $<BUILD_INTERFACE:${GENERATED_TRACKER_DEVICEIO_BASE_INC_DIR}>
        $<BUILD_INTERFACE:${GENERATED_TRACKER_LIVE_IMPL_INC_DIR}>
)

target_link_libraries(live_trackers
    PUBLIC
        deviceio::deviceio_base
        deviceio::deviceio_trackers
        mcap::mcap_core
        oxr::oxr_utils
        pusherio::pusherio
        Teleop::openxr_extensions
        log_bridge::log_bridge_core
)

add_library(deviceio::live_trackers ALIAS live_trackers)
