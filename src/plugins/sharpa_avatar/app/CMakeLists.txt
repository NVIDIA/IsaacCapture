# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(avatar_hand_plugin
  cli.hpp
  main.cpp
)

target_link_libraries(avatar_hand_plugin
  PRIVATE
    isaac::avatar_plugin
)

set_target_properties(avatar_hand_plugin PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${_AVATAR_PLUGIN_BUILD_DIR}"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "${_AVATAR_SDK_LIBRARY_DIR}"
  # All in-tree libraries are static; libavatar_sdk.so is the only shared dependency.
  INSTALL_RPATH "${_AVATAR_SDK_LIBRARY_DIR}"
)

install(TARGETS avatar_hand_plugin
  RUNTIME DESTINATION plugins/sharpa_avatar
  COMPONENT avatar
)
