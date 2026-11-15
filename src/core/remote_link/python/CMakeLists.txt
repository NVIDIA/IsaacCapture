# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pybind11_add_module(remote_link_py
    remote_link_bindings.cpp
    py_link.hpp
)

target_link_libraries(remote_link_py
    PRIVATE
        remote_link::remote_link
)

set_target_properties(remote_link_py PROPERTIES
    OUTPUT_NAME "_remote_link"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/remote_link"
)
