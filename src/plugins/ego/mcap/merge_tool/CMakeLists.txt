# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ego_mcap_merge main.cpp)
target_link_libraries(ego_mcap_merge PRIVATE ego::mcap_merge log_bridge::log_bridge_core)
set_target_properties(ego_mcap_merge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EGO_PLUGIN_OUTPUT_DIR}")
# The recording helper resolves this tool beside the component-installed plugin.
install(TARGETS ego_mcap_merge RUNTIME DESTINATION plugins/ego COMPONENT ego)
