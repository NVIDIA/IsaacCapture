# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(pose_provider_plugin main.cpp)

target_link_libraries(pose_provider_plugin PRIVATE
    pose_provider::pose_provider_core
    ${POSE_ENGINE_TARGET}
    log_bridge::log_bridge_core
)

# Beside the copied plugin.yaml, so the build tree is a plugin directory.
set_target_properties(pose_provider_plugin PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${POSE_PROVIDER_PLUGIN_DIR}")
