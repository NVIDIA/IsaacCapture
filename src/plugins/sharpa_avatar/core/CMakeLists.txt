# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(avatar_plugin_core SHARED
  avatar_glove_collection.hpp
  avatar_hand_tracking_plugin.cpp
  avatar_hand_tracking_plugin_impl.hpp
  inc/avatar/avatar_hand_tracking_plugin.hpp
)

target_include_directories(avatar_plugin_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(avatar_plugin_core
  PRIVATE
    AvatarSDK::AvatarSDK
    OpenXR::openxr_loader
    Teleop::openxr_extensions
    Teleop::plugin_utils
    deviceio::deviceio_session
    deviceio::deviceio_trackers
    isaacteleop_schema
    oxr::oxr_core
    oxr::oxr_utils
    pusherio::pusherio
)

set_target_properties(avatar_plugin_core PROPERTIES
  OUTPUT_NAME IsaacTeleopPluginsAvatar
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "${_AVATAR_SDK_LIBRARY_DIR}"
  INSTALL_RPATH "${_AVATAR_SDK_LIBRARY_DIR};$ORIGIN"
)

add_library(isaac::avatar_plugin ALIAS avatar_plugin_core)

install(
  TARGETS avatar_plugin_core
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT avatar
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT avatar
)
