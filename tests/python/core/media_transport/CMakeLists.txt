# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
file(GLOB TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name "${test_file}" NAME_WE)
    add_test(NAME "media_transport_${test_name}"
        COMMAND uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} pytest -v --tb=short "${test_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
