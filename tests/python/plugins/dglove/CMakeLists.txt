# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(suite receiver source publisher)
    add_test(NAME dglove_${suite}
        COMMAND ${CMAKE_COMMAND} -E env "DGLOVE_PUBLISHER=$<TARGET_FILE:dglove_plugin>"
            uv run --python ${ISAAC_TELEOP_PYTHON_VERSION} pytest
                -c ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml -q test_${suite}.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
