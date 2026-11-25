# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The Pose Provider runtime. It links no engine adapter; each executable links one. mcap_writer.cpp defines MCAP's
# writer for the run log; a program that also reads MCAP defines the reader and writer in one object of its own, as the
# tests' mcap_impl.cpp does, and the linker then leaves this copy out.
add_library(pose_provider_core STATIC
    mcap_writer.cpp
    pose_provider.cpp
    inc/pose_provider/pose_provider.hpp
)

target_include_directories(pose_provider_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

# Teleop::plugin_utils comes before oxr::oxr_core: the other way round, the static OpenXR loader takes a std::string
# symbol from deviceio_session and links DeviceIO and MCAP code that the plugin never calls.
target_link_libraries(pose_provider_core
    PUBLIC
        pose_provider::pose_camera_input
        pose_provider::pose_engine_api
        pusherio::pusherio
    PRIVATE
        Teleop::plugin_utils
        deviceio::deviceio_trackers
        isaacteleop_schema
        log_bridge::log_bridge_core
        mcap::mcap
        oxr::oxr_core
        oxr::oxr_utils
        pose_provider::pose_undistort
)

add_library(pose_provider::pose_provider_core ALIAS pose_provider_core)
