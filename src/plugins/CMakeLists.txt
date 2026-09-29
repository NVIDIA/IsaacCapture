# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Device plugins
# ==============================================================================
# Added only when BUILD_PLUGINS is ON (see src/CMakeLists.txt). Plugins that pull
# a third-party tree in via FetchContent must wrap it in
# isaac_teleop_third_party_scope_begin()/_end() -- this directory inherits the
# project warning set, and a fetched subdirectory would otherwise inherit it too.
# See cmake/CompilerWarnings.cmake.

# Shared plugin utilities, linked by the plugins below.
add_subdirectory(plugin_utils)

add_subdirectory(controller_se3_tracker)
add_subdirectory(vive_se3_tracker)
add_subdirectory(controller_synthetic_hands)
add_subdirectory(generic_3axis_pedal)
add_subdirectory(xsens_full_body)
add_subdirectory(so101_leader)
add_subdirectory(rebot_devarm_leader)
add_subdirectory(manus)
add_subdirectory(haptikos)

if(BUILD_PLUGIN_NOITOM_MOCAP)
    add_subdirectory(noitom_mocap)
endif()
if(BUILD_PLUGIN_OAK_CAMERA)
    add_subdirectory(oak)
endif()
if(BUILD_PLUGIN_OGLO)
    add_subdirectory(oglo_tactile)
endif()
# Gated: requires the wuji_sdk C library (set WUJI_SDK_INCLUDE_DIR + lib path).
if(BUILD_PLUGIN_WUJI_GLOVE)
    add_subdirectory(wuji_glove)
endif()
# Gated: Linux + Avatar SDK under /opt/avatar-sdk. Missing SDK skips the plugin.
if(BUILD_PLUGIN_SHARPA_AVATAR)
    add_subdirectory(sharpa_avatar)
endif()
