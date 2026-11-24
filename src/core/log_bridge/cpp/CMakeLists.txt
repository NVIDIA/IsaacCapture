# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Core Logger library (static; consumed in-tree via log_bridge::log_bridge_core).
# It has no Python dependency; all C++ consumers use socket forwarding or
# process-local fallback sinks.
add_library(log_bridge_core STATIC
    logger.cpp
    sink_config.cpp
    sink_config.hpp
    socket_sink.cpp
    socket_sink.hpp
    inc/log_bridge/logger.hpp
)

target_include_directories(log_bridge_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(log_bridge_core
    PUBLIC
        spdlog::spdlog
    PRIVATE
        secure_directory::secure_directory
)

add_library(log_bridge::log_bridge_core ALIAS log_bridge_core)
