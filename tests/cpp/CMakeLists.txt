# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(support/temporary_directory)
add_subdirectory(core)

if(BUILD_VIZ)
    add_subdirectory(viz)
endif()

if(BUILD_PLUGINS)
    add_subdirectory(plugins)
endif()
