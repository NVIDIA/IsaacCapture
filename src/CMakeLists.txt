# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# First-party C++ tree
# ==============================================================================
# Everything under src/ is code we own, so the project warning set is enabled once
# here for the whole directory. Scoping it to this file is what lets the root
# CMakeLists.txt add deps/ and src/ in any order: third-party trees are simply
# never inside a directory that carries our flags.
isaac_teleop_enable_compiler_warnings()

# Core modules (OXR and DEVICEIO).
add_subdirectory(core)

# Stage the isaaccapture Python tree. Must follow core (defines BUILD_PYTHON_BINDINGS).
add_subdirectory(python)

# The isaacteleop compatibility shim, staged into the tree above. Must follow python.
add_subdirectory(compat)

# Televiz visualization module, a sibling of core.
# viz/CMakeLists.txt orchestrates its own sub-modules and tests.
if(BUILD_VIZ)
    add_subdirectory(viz)
endif()

if(BUILD_VIDEO_STAGE)
    add_subdirectory(video_stage)
endif()

if(BUILD_PLUGINS)
    add_subdirectory(plugins)
endif()
