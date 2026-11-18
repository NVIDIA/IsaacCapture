# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only slot marker module shared by camera_receiver and the CUDA IPC consumers. The .so lands next to
# __init__.py so the package imports without an install step.

cmake_minimum_required(VERSION 3.20)
project(camera_ipc_protocol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# pybind11 is pip-installed into the active Python environment.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_rc
)
if(NOT _pybind11_rc EQUAL 0)
  message(FATAL_ERROR "pybind11 not found. `uv pip install pybind11` into the active venv.")
endif()
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(_slot_markers MODULE slot_markers_bindings.cpp)
target_include_directories(_slot_markers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(_slot_markers PRIVATE rt)
target_compile_options(_slot_markers PRIVATE -Wall -Wextra)

set_target_properties(_slot_markers PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
