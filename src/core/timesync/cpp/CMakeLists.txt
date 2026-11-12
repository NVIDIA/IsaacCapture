# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_library(timesync STATIC
    chrony_tracking.cpp
    local_clock.cpp
    capture_qualifier.cpp
    command_runner.cpp
    chrony_monitor.cpp
)

target_include_directories(timesync
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(timesync
    PRIVATE
        Threads::Threads
)

target_compile_features(timesync PUBLIC cxx_std_17)
target_compile_options(timesync PRIVATE -Wall -Wextra)

add_library(timesync::timesync ALIAS timesync)
