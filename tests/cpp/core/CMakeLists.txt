# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(oxr)
endif()

add_subdirectory(schema)
add_subdirectory(mcap)
add_subdirectory(deviceio_trackers)
add_subdirectory(replay_deviceio_session)
add_subdirectory(live_trackers)
add_subdirectory(tracker_metadata)
add_subdirectory(log_bridge)
if(NOT WIN32)
    add_subdirectory(timesync)
endif()
add_subdirectory(plugin_manager)

if(BUILD_OVXR_CAMERA_QUADS)
    add_subdirectory(ovxr_pip_layer)
endif()
