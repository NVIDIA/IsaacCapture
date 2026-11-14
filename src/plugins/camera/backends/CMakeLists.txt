# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Which cameras this build of the plugin can open is decided here, when the build is configured.
# The synthetic backend needs no SDK and is always built. Every other backend sits behind its
# own option, off by default, and a backend that is off is not configured at all: its SDK is
# neither looked for nor built. See ../README.md, "Adding a backend".
add_subdirectory(synthetic)

add_library(camera_plugin_backends STATIC
    backends.cpp
)

target_include_directories(camera_plugin_backends INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera_plugin_backends
    PUBLIC
        camera::plugin_core
    PRIVATE
        camera::backend_synthetic
)

add_library(camera::plugin_backends ALIAS camera_plugin_backends)
