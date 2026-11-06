# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# isaaccapture pure-Python source
# ==============================================================================
# Single producer of the authored .py in python_package/<config>/isaaccapture/ --
# the canonical layout ctest reads via PYTHONPATH and both wheels build from.
# Build products share that tree but are NOT owned here: pybind11 extensions
# (each module's LIBRARY_OUTPUT_DIRECTORY), .pyi/py.typed (src/core/python), and
# cloudxr/native/ (src/core/cloudxr/python).

if(NOT BUILD_PYTHON_BINDINGS)
    return()
endif()

# CONFIGURE_DEPENDS: adding or removing a file re-runs configure. Globbing by
# extension rather than copying the directory keeps __pycache__ out by construction.
# The .xml are viz/robot/assets/'s MJCF wrappers, which assets.ensure_so101_scene()
# reads out of its own installed directory.
file(GLOB_RECURSE ISAAC_TELEOP_PYTHON_SOURCES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/isaaccapture/*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/isaaccapture/*.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/isaaccapture/viz/robot/*.png"
)

# cts/ is the CTS tooling, run in place from the source tree and never shipped.
list(FILTER ISAAC_TELEOP_PYTHON_SOURCES EXCLUDE REGEX "^isaaccapture/cts/")

# viz/__init__.py does a top-level `from ._viz import ...`, so without _viz it is
# unimportable -- and packages.find would enumerate it into a BUILD_VIZ=OFF wheel.
if(NOT BUILD_VIZ)
    list(FILTER ISAAC_TELEOP_PYTHON_SOURCES EXCLUDE REGEX "^isaaccapture/viz/")
endif()

set(ISAAC_TELEOP_PYTHON_STAGE_DIR "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>")

# Staging copies, it never removes: a .py deleted or renamed in the source tree
# survives in an incremental build dir until it is wiped. Wheels are built from a
# clean tree, so this only ever shows up locally -- rebuild from scratch if a
# module you deleted still imports.
set(ISAAC_TELEOP_PYTHON_STAGED_FILES "")
foreach(_rel_path IN LISTS ISAAC_TELEOP_PYTHON_SOURCES)
    set(_src_file "${CMAKE_CURRENT_SOURCE_DIR}/${_rel_path}")
    set(_dst_file "${ISAAC_TELEOP_PYTHON_STAGE_DIR}/${_rel_path}")
    get_filename_component(_dst_dir "${_dst_file}" DIRECTORY)

    add_custom_command(
        OUTPUT "${_dst_file}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_dst_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy "${_src_file}" "${_dst_file}"
        DEPENDS "${_src_file}"
        VERBATIM
    )

    list(APPEND ISAAC_TELEOP_PYTHON_STAGED_FILES "${_dst_file}")
endforeach()

add_custom_target(isaacteleop_python ALL
    DEPENDS ${ISAAC_TELEOP_PYTHON_STAGED_FILES}
    COMMENT "Staging isaaccapture pure-Python sources into the package structure"
)
