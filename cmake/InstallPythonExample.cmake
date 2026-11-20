# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# InstallPythonExample.cmake
# ==============================================================================
# Macro to install a Python example directory with a generated pyproject.toml.
#
# Reads the source pyproject.toml and appends a [tool.uv] block only to the
# installed copy so that `uv run` works out of the box in the install tree. On the
# way it pins isaaccapture to the version this build produced and drops any
# [tool.uv.sources], so the installed example resolves the wheel next to it.
# A .python-version beside it pins `uv run` to the build's Python.
#
# Two source layouts are accepted while the examples are converted to the
# convention in examples/README.md (NVIDIA/IsaacTeleop#985), picked by where
# pyproject.toml sits. Converted: at the example root, sources under python/.
# Legacy: under python/. Drop the legacy branch once every example has moved.
#
# Usage:
#   install_python_example(DESTINATION examples/<name>)          # converted
#   install_python_example(DESTINATION examples/<name>/python)   # legacy
#   install_python_example(DESTINATION examples/teleop_ros2/python
#       EXTRA_UV_EXTRA_BUILD_DEPS "nlopt = [\"numpy\"]")
# ==============================================================================

macro(install_python_example)
    cmake_parse_arguments(_IPE "" "DESTINATION;EXTRA_UV_EXTRA_BUILD_DEPS" "" ${ARGN})
    if(NOT _IPE_DESTINATION)
        message(FATAL_ERROR "install_python_example: DESTINATION is required")
    endif()

    # Installed example is intended to run against a locally built wheel, so
    # required-environments must list only the current build arch.
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(_IPE_PLATFORM_MACHINE "aarch64")
    else()
        set(_IPE_PLATFORM_MACHINE "x86_64")
    endif()

    # Read the bare pyproject.toml and append uv configuration for the
    # installed environment. `find-links` is relative to the installed
    # pyproject, so its depth follows the layout: <prefix>/examples/<name>/ for
    # a converted example, one level deeper for a legacy one.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
        set(_IPE_PYPROJECT "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
        set(_IPE_FIND_LINKS "../../wheels")
    else()
        set(_IPE_PYPROJECT "${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml")
        set(_IPE_FIND_LINKS "../../../wheels")
    endif()
    file(READ "${_IPE_PYPROJECT}" _PYPROJECT_BASE)

    # A source-tree [tool.uv.sources] path is relative to the pyproject, so from
    # the installed copy it points at the install prefix, which is not a Python
    # project. Matched at line start, so prose naming the header is left alone.
    string(REGEX REPLACE "\n\\[tool\\.uv\\.sources\\][^[]*" "\n" _PYPROJECT_BASE "${_PYPROJECT_BASE}")

    # Pin to the wheel this build produced. Unversioned, uv would prefer the
    # newest release on an index whenever the local wheel is a pre-release --
    # which every CI build is (see IsaacTeleopVersion.cmake).
    # TODO(#880): no test covers the generated pyproject.
    string(REGEX REPLACE "\"(isaaccapture(\\[[^]]*\\])?)\""
        "\"\\1==${ISAAC_TELEOP_PYPROJECT_VERSION}\"" _PYPROJECT_BASE "${_PYPROJECT_BASE}")
    set(_TOOL_UV_BLOCK "[tool.uv]
find-links = [\"${_IPE_FIND_LINKS}\"]
python-preference = \"only-managed\"
environments = [\"python_version == '${ISAAC_TELEOP_PYTHON_VERSION}'\"]
required-environments = [\"sys_platform == 'linux' and platform_machine == '${_IPE_PLATFORM_MACHINE}'\"]
")
    if(_IPE_EXTRA_UV_EXTRA_BUILD_DEPS)
        string(APPEND _TOOL_UV_BLOCK "
[tool.uv.extra-build-dependencies]
${_IPE_EXTRA_UV_EXTRA_BUILD_DEPS}
")
    endif()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml"
        "${_PYPROJECT_BASE}\n${_TOOL_UV_BLOCK}")

    # Install generated pyproject.toml (with [tool.uv] appended)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml"
        DESTINATION ${_IPE_DESTINATION}
    )

    # requires-python is a range, so without this uv picks its newest managed
    # Python, which `environments` above then rejects.
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/.python-version" "${ISAAC_TELEOP_PYTHON_VERSION}\n")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/.python-version"
        DESTINATION ${_IPE_DESTINATION}
    )

    # Install Python example sources. A converted example keeps its python/
    # directory, so the namespace root in the install tree matches the source
    # tree; a legacy one flattens python/ into the destination.
    # `python` without a trailing slash installs the directory itself, so the
    # namespace root survives; `python/` installs its contents, flattening it.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
        set(_IPE_SOURCE_DIR "python")
    else()
        set(_IPE_SOURCE_DIR "python/")
    endif()
    install(DIRECTORY ${_IPE_SOURCE_DIR}
        DESTINATION ${_IPE_DESTINATION}
        FILES_MATCHING
            PATTERN "*.py"
            PATTERN "*.md"
        PATTERN ".venv" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
    )
endmacro()
