# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Narrowed to the JSON the SBOM collector reads: upstream publishes every license
# in several formats, none of which anything here consumes.

foreach(_required GIT_EXECUTABLE SPDX_REPOSITORY SPDX_COMMIT SPDX_SOURCE_DIR)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

if(EXISTS "${SPDX_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${SPDX_SOURCE_DIR}"
        OUTPUT_VARIABLE _current_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rev_parse_result
    )
    if(_rev_parse_result EQUAL 0 AND _current_commit STREQUAL SPDX_COMMIT)
        return()
    endif()
endif()

file(REMOVE_RECURSE "${SPDX_SOURCE_DIR}")
execute_process(
    COMMAND "${GIT_EXECUTABLE}" clone --quiet --filter=tree:0 --no-checkout
        "${SPDX_REPOSITORY}" "${SPDX_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" sparse-checkout set "json/details" "json/licenses.json"
    WORKING_DIRECTORY "${SPDX_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" checkout --quiet --detach "${SPDX_COMMIT}"
    WORKING_DIRECTORY "${SPDX_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)
