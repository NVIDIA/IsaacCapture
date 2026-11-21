# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# Generated headers from FlatBuffer schemas with additional utilities.
add_subdirectory(schema)

# Grades one binary schema against another; used by the conform test and by replay.
add_subdirectory(schema_compat)

# isaaccapture::Logger. Added early -- before any module that might want to
# log during its own construction.
add_subdirectory(log_bridge)

# Build OXR Utils first - header-only utilities (no dependencies)
add_subdirectory(oxr_utils)

# Build Plugin Manager
add_subdirectory(plugin_manager)

# Build OXR library (depends on oxr_utils for types)
add_subdirectory(oxr)

# Build PusherIO library (depends on oxr for session management)
add_subdirectory(pusherio)

#   4. deviceio_session        — DeviceIOSession (wires trackers to live factory)
include(${CMAKE_SOURCE_DIR}/cmake/GenerateTrackers.cmake)
isaac_teleop_generate_trackers()

add_subdirectory(deviceio_base)
add_subdirectory(deviceio_trackers)
add_subdirectory(live_trackers)
add_subdirectory(replay_trackers)
add_subdirectory(deviceio_session)

# Build MCAP library (depends on deviceio for DeviceIOSession and tracker interfaces)
add_subdirectory(mcap)

if(BUILD_OVXR_CAMERA_QUADS)
    add_subdirectory(ovxr_pip_layer)
endif()

# CloudXR runtime helper (stages the vendored CloudXR runtime SDK bundle).
# Its pure-Python source lives in src/python/isaaccapture/cloudxr/, as does the
# retargeting engine's and the session manager's.
add_subdirectory(cloudxr)

# Python wheel packaging (combines both modules)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
