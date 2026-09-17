# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(plugin_utils)

if(BUILD_PLUGIN_OGLO)
    add_subdirectory(oglo_tactile)
endif()

if(BUILD_PLUGIN_REALHAND_FFG_GLOVE)
    add_subdirectory(realhand_ffg_glove)
endif()
