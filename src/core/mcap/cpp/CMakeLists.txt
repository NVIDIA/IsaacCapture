# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# mcap_core provides McapTrackerChannels / McapTrackerViewers
# (mcap/tracker_channels.hpp), thin helpers that reduce boilerplate in the
# Live*TrackerImpl and Replay*TrackerImpl constructors and update() methods.
# The actual MCAP implementation (MCAP_IMPLEMENTATION) is compiled once inside
# deviceio_core (deviceio_session.cpp).
add_library(mcap_core STATIC
    recorded_schemas.cpp
    sample_stream.cpp
    inc/mcap/recorded_schemas.hpp
    inc/mcap/recording_traits.hpp
    inc/mcap/sample_stream.hpp
    inc/mcap/tracker_channels.hpp
)

target_include_directories(mcap_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<BUILD_INTERFACE:${GENERATED_TRACKER_INC_DIR}>
)

# Transitively expose mcap::mcap so consumers get <mcap/writer.hpp>
target_link_libraries(mcap_core
    PUBLIC
        mcap::mcap
        isaacteleop_schema
        schema::schema_compat
        log_bridge::log_bridge_core
)

add_library(mcap::mcap_core ALIAS mcap_core)
