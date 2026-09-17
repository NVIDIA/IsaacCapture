# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(avatar_plugin_core STATIC
  avatar_glove_collection.hpp
  avatar_hand_tracking_plugin.cpp
  inc/avatar/avatar_hand_tracking_plugin.hpp
  inc/avatar/cli.hpp
)

target_include_directories(avatar_plugin_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_compile_definitions(avatar_plugin_core
  PRIVATE
    AVATAR_SDK_CONFIG_PATH="${_AVATAR_SDK_CONFIG}"
)

target_link_libraries(avatar_plugin_core
  PUBLIC
    AvatarSDK::AvatarSDK
    OpenXR::openxr_loader
  PRIVATE
    Teleop::openxr_extensions
    Teleop::plugin_utils
    deviceio::deviceio_session
    deviceio::deviceio_trackers
    isaacteleop_schema
    nlohmann_json::nlohmann_json
    oxr::oxr_core
    oxr::oxr_utils
    pusherio::pusherio
)

set_target_properties(avatar_plugin_core PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON
)
