# SPDX-FileCopyrightText: Copyright (c) 2026 HTC Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# VIVEHub VUT tracker SDK (static archive + headers), shipped with VIVEHub-Linux.
# Requires SDK >= 1.0.1 (Pose::serial on the wire). Point VUT_SDK_DIR at your
# VIVEHub-Linux/sdk directory (contains include/ and lib/libvut_sdk.a):
#   cmake ... -DVUT_SDK_DIR=/path/to/VIVEHub-Linux/sdk
# If it is not set / not found, the plugin is skipped (matches how the other
# vendor-SDK plugins behave), so a plain source or wheel build still configures.
set(VUT_SDK_DIR "" CACHE PATH "Path to the VIVEHub VUT SDK (contains include/ and lib/libvut_sdk.a)")
# Derive the lib path directly from VUT_SDK_DIR.
set(_vut_lib "${VUT_SDK_DIR}/lib/libvut_sdk.a")
if(NOT VUT_SDK_DIR OR NOT EXISTS "${_vut_lib}")
    message(STATUS "VIVEHub VUT SDK not found (set -DVUT_SDK_DIR=/path/to/VIVEHub-Linux/sdk) "
                   "— skipping vive_se3_tracker plugin build.")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(vive_se3_tracker_plugin
    main.cpp
    vive_se3_tracker_plugin.cpp
    vive_se3_tracker_plugin.hpp
)

target_include_directories(vive_se3_tracker_plugin PRIVATE "${VUT_SDK_DIR}/include")

target_link_libraries(vive_se3_tracker_plugin PRIVATE
    deviceio::deviceio_trackers
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
    "${_vut_lib}"
    Threads::Threads
    log_bridge::log_bridge_core
    secure_directory::secure_directory
)

install(TARGETS vive_se3_tracker_plugin RUNTIME DESTINATION plugins/vive_se3_tracker)
install(FILES plugin.yaml README.md DESTINATION plugins/vive_se3_tracker)
