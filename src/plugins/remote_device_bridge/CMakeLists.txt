# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The transports use POSIX sockets.
if(WIN32)
    return()
endif()

add_subdirectory(session)
add_subdirectory(server)
add_subdirectory(app)
add_subdirectory(tools)

install(FILES plugin.yaml
    DESTINATION plugins/remote_device_bridge
)
