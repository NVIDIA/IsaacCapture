# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The camera reader for Python: isaaccapture.camera._camera.
pybind11_add_module(camera_reader_py
    camera_reader_bindings.cpp
)

target_link_libraries(camera_reader_py
    PRIVATE
        camera::camera_reader
)

set_target_properties(camera_reader_py PROPERTIES
    OUTPUT_NAME "_camera"
    # Use genex in output directory - CMake won't add another config subdirectory
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/camera"
)
