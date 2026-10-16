# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Ensure OpenXR target is available (fetched via CMake FetchContent by parent)
if(NOT TARGET openxr_loader)
    message(FATAL_ERROR "openxr_loader target not found. Make sure OpenXR SDK is fetched via FetchContent first.")
endif()

# Create OpenXR::OpenXR alias for consistency
if(NOT TARGET OpenXR::OpenXR)
    add_library(OpenXR::OpenXR ALIAS openxr_loader)
endif()

# Core OXR session library (static; consumed in-tree via oxr::oxr_core)
add_library(oxr_core STATIC
    oxr_session.cpp
    inc/oxr/oxr_session.hpp
)

target_include_directories(oxr_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(oxr_core
    PUBLIC
        OpenXR::OpenXR
        oxr::oxr_utils
        log_bridge::log_bridge_core
    PRIVATE
        # oxr_time.hpp includes the NVIDIA action-context extension declarations.
        Teleop::openxr_extensions
)

# Create alias for consistent naming
add_library(oxr::oxr_core ALIAS oxr_core)
