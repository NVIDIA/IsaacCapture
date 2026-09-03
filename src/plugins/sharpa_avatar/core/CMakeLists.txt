# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(avatar_plugin_core SHARED
  avatar_hand_tracking_plugin.cpp
  inc/avatar/avatar_hand_tracking_plugin.hpp
)

target_include_directories(avatar_plugin_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(avatar_plugin_core
  PUBLIC
    AvatarSDK::AvatarSDK
    Teleop::plugin_utils
    deviceio::deviceio_session
    deviceio::deviceio_trackers
    OpenXR::openxr_loader
    oxr::oxr_core
    pusherio::pusherio
  PRIVATE
    oxr::oxr_utils
    Teleop::openxr_extensions
    isaacteleop_schema
)

set_target_properties(avatar_plugin_core PROPERTIES
  OUTPUT_NAME IsaacTeleopPluginsAvatar
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_WITH_INSTALL_RPATH NO
  SKIP_BUILD_RPATH NO
  BUILD_RPATH "${_ISAAC_AVATAR_BUILD_RPATH_STR}"
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS avatar_plugin_core
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT avatar
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT avatar
)
