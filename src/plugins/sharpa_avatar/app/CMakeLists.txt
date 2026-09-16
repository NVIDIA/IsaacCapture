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
  RUNTIME_OUTPUT_DIRECTORY "${_AVATAR_PLUGIN_BUILD_DIR}"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS avatar_hand_plugin
  RUNTIME DESTINATION plugins/sharpa_avatar
  COMPONENT avatar
)
