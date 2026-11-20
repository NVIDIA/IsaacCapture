# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pybind11_add_module(robot_state_store_py
    robot_state_bindings.cpp
)

# oxr_utils is used for os_time.hpp only (header-only, no OpenXR loader).
target_link_libraries(robot_state_store_py
    PRIVATE
        robot_state::robot_state_store
        oxr::oxr_utils
)

set_target_properties(robot_state_store_py PROPERTIES
    OUTPUT_NAME "_robot_state"
    # Use genex in output directory - CMake won't add another config subdirectory
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/robot_state"
)
