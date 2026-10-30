# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

add_executable(remote_hand_sender_synthetic
    remote_hand_sender_synthetic.cpp
)

target_link_libraries(remote_hand_sender_synthetic
    PRIVATE
        Teleop::remote_plugin_session
        log_bridge::log_bridge_core
)

set_target_properties(remote_hand_sender_synthetic PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

install(TARGETS remote_hand_sender_synthetic
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
