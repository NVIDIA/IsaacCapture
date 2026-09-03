# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(avatar_hand_plugin
  main.cpp
)

target_link_libraries(avatar_hand_plugin
  PRIVATE
    avatar_plugin_core
)

set_target_properties(avatar_hand_plugin PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_WITH_INSTALL_RPATH NO
  SKIP_BUILD_RPATH NO
  BUILD_RPATH "${_ISAAC_AVATAR_BUILD_RPATH_STR}"
  INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS avatar_hand_plugin
  RUNTIME DESTINATION plugins/sharpa_avatar
  COMPONENT avatar
)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../plugin.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../README.md"
  DESTINATION plugins/sharpa_avatar
  COMPONENT avatar
)
