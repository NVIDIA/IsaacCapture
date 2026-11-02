# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pose math, clocks, wrist validity, shape checks, serialization and the per-hand push pipeline,
# kept apart from OpenXR so they can be unit-tested without a runtime or a device.
add_library(revohuman_core STATIC
    device_clock.cpp
    inc/revohuman_plugin/device_clock.hpp
    device_status.cpp
    inc/revohuman_plugin/device_status.hpp
    event_log.cpp
    inc/revohuman_plugin/event_log.hpp
    pipeline.cpp
    inc/revohuman_plugin/pipeline.hpp
    pose_math.cpp
    inc/revohuman_plugin/pose_math.hpp
    serialize.cpp
    inc/revohuman_plugin/serialize.hpp
    inc/revohuman_plugin/source.hpp
    sha256.cpp
    inc/revohuman_plugin/sha256.hpp
    tactile.cpp
    inc/revohuman_plugin/tactile.hpp
    types.cpp
    inc/revohuman_plugin/types.hpp
    wrist_tracking.cpp
    inc/revohuman_plugin/wrist_tracking.hpp
)

target_include_directories(revohuman_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(revohuman_core
    PUBLIC
        isaacteleop_schema
        log_bridge::log_bridge_core
)

add_library(revohuman_plugin::revohuman_core ALIAS revohuman_core)
