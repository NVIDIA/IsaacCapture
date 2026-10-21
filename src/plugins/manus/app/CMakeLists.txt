# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(manus_hand_plugin
  main.cpp
)

target_link_libraries(manus_hand_plugin
  PRIVATE
    manus_plugin_core
    Teleop::plugin_utils
)

# --remote=<host:port> needs the RemotePluginSession library (BUILD_PLUGIN_REMOTE_DEVICE_BRIDGE, non-Windows);
# without it the flag is rejected at startup.
if(TARGET Teleop::remote_plugin_session)
  target_link_libraries(manus_hand_plugin PRIVATE Teleop::remote_plugin_session)
  target_compile_definitions(manus_hand_plugin PRIVATE MANUS_HAVE_REMOTE_PLUGIN_SESSION)
endif()

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
