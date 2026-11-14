# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

add_executable(camera_plugin
    main.cpp
)

target_link_libraries(camera_plugin PRIVATE
    camera::plugin_core
    camera::plugin_backends
    log_bridge::log_bridge_core
    Threads::Threads
)

install(TARGETS camera_plugin RUNTIME DESTINATION plugins/camera)
