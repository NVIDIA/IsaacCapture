# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

add_executable(remote_device_bridge
    main.cpp
)

target_link_libraries(remote_device_bridge
    PRIVATE
        Teleop::plugin_utils
        Teleop::remote_device_bridge_server
        Teleop::remote_plugin_session
        log_bridge::log_bridge_core
        Threads::Threads
)

install(TARGETS remote_device_bridge
    RUNTIME DESTINATION plugins/remote_device_bridge
)
