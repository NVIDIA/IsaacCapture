# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(orbbec_ego_mcap_export_media export_embedded_media.cpp)
target_link_libraries(orbbec_ego_mcap_export_media PRIVATE ego::mcap_support log_bridge::log_bridge_core)
set_target_properties(orbbec_ego_mcap_export_media PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EGO_PLUGIN_OUTPUT_DIR}")
install(TARGETS orbbec_ego_mcap_export_media RUNTIME DESTINATION plugins/orbbec_ego COMPONENT orbbec_ego)
