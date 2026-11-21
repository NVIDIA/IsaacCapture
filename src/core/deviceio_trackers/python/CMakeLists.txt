# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pybind11_add_module(deviceio_trackers_py
    tracker_bindings.cpp
)

target_link_libraries(deviceio_trackers_py
    PRIVATE
        deviceio::deviceio_trackers
)

target_include_directories(deviceio_trackers_py
    PRIVATE
        ${GENERATED_TRACKER_INC_DIR}
)

set_target_properties(deviceio_trackers_py PROPERTIES
    OUTPUT_NAME "_deviceio_trackers"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/deviceio_trackers"
)

# The module compiles in Chromium's BSD-3-Clause key table. CHROMIUM_LICENSE is what
# pyproject.toml names; the scikit-build install takes every non-.py file in the staged tree.
chromium_dom_codes_stage_license(deviceio_trackers_py)
