# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(manus_plugin_core SHARED
  manus_hand_tracking_plugin.cpp
  inc/manus/manus_hand_tracking_plugin.hpp
)

target_include_directories(manus_plugin_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(manus_plugin_core
  PUBLIC
    ManusSDK::ManusSDK
    deviceio::deviceio_trackers
    pusherio::pusherio
    log_bridge::log_bridge_core
  PRIVATE
    # Carries the static OpenXR loader; PUBLIC would link a second copy into every executable.
    Teleop::plugin_utils
    oxr::oxr_utils
    isaacteleop_schema
)

set_target_properties(manus_plugin_core PROPERTIES
  OUTPUT_NAME IsaacTeleopPluginsManus
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_WITH_INSTALL_RPATH NO
  SKIP_BUILD_RPATH NO
  BUILD_RPATH "${_ISAAC_MANUS_EFFECTIVE_BUILD_RPATH_STR}"
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS manus_plugin_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT manus
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT manus
)
