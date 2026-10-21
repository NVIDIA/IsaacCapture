# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Registers the SBOM collector's tests with CTest; they need no built package.

file(GLOB TEST_FILES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py"
)

foreach(test_file ${TEST_FILES})
    get_filename_component(test_name "${test_file}" NAME_WE)

    add_test(
        NAME "sbom_${test_name}"
        COMMAND uv run --project "${CMAKE_CURRENT_SOURCE_DIR}" --python ${ISAAC_TELEOP_PYTHON_VERSION}
                --extra dev pytest -v --tb=short "${test_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # The SPDX license list deps/third_party fetched for this build, so the tests
    # read the same reference texts the collector will and never reach the
    # network; and this build tree, so the file API reader is exercised against
    # a codemodel CMake actually wrote.
    set_tests_properties("sbom_${test_name}" PROPERTIES
        ENVIRONMENT "ISAACCAPTURE_SBOM_LICENSE_DATA=${SPDX_LICENSE_DATA_JSON_DIR};ISAACCAPTURE_SBOM_BUILD_DIR=${CMAKE_BINARY_DIR}"
        LABELS "sbom"
    )
endforeach()
