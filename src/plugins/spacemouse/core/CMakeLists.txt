# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# HID report parsing, device discovery and the reconnecting reader, kept apart from the OpenXR
# push loop so they can be unit-tested without a runtime or a device.
add_library(spacemouse_core STATIC
    device_discovery.cpp
    inc/spacemouse/device_discovery.hpp
    hid_report.cpp
    inc/spacemouse/hid_report.hpp
    spacemouse_connection.cpp
    inc/spacemouse/spacemouse_connection.hpp
)

target_include_directories(spacemouse_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(spacemouse_core
    PUBLIC
        log_bridge::log_bridge_core
)

add_library(spacemouse::spacemouse_core ALIAS spacemouse_core)
