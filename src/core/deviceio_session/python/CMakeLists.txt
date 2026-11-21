# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pybind11_add_module(deviceio_session_py
    session_bindings.cpp
)

target_link_libraries(deviceio_session_py
    PRIVATE
        deviceio::deviceio_session
        deviceio::deviceio_py_utils
)

# The module imports isaaccapture.robot_state._robot_state when it loads.
add_dependencies(deviceio_session_py robot_state_store_py)

set_target_properties(deviceio_session_py PROPERTIES
    OUTPUT_NAME "_deviceio_session"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/deviceio_session"
)
