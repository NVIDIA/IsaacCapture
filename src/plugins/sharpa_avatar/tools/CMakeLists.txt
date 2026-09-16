# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(avatar_hand_tracker_printer
  avatar_hand_tracker_printer.cpp
)

target_link_libraries(avatar_hand_tracker_printer
  PRIVATE
    avatar_plugin_core
)

set_target_properties(avatar_hand_tracker_printer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${_AVATAR_PLUGIN_BUILD_DIR}"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS avatar_hand_tracker_printer
  RUNTIME DESTINATION plugins/sharpa_avatar
  COMPONENT avatar
)

install(
  FILES sharpa_avatar_sample.py
  DESTINATION plugins/sharpa_avatar/tools
  COMPONENT avatar
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
