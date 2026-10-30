# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(manus_hand_plugin
  main.cpp
)

target_link_libraries(manus_hand_plugin
  PRIVATE
    manus_plugin_core
)

set_target_properties(manus_hand_plugin PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_WITH_INSTALL_RPATH NO
  SKIP_BUILD_RPATH NO
  BUILD_RPATH "${_ISAAC_MANUS_EFFECTIVE_BUILD_RPATH_STR}"
  INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS manus_hand_plugin
    RUNTIME DESTINATION plugins/manus
    COMPONENT manus
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../plugin.yaml"
    DESTINATION plugins/manus
    COMPONENT manus
)
