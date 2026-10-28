# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(camera_viz_ego LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ORBBEC_SDK_ROOT)
  message(FATAL_ERROR "ORBBEC_SDK_ROOT is required")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_rc
)
if(NOT _pybind11_rc EQUAL 0)
  message(FATAL_ERROR "pybind11 not found in the active Python environment")
endif()
find_package(pybind11 REQUIRED CONFIG)

# Keep vendor SDK identifiers separate from EGO product target names.
find_library(ORBBEC_SDK_LIBRARY NAMES OrbbecSDK PATHS ${ORBBEC_SDK_ROOT}/lib NO_DEFAULT_PATH REQUIRED)

# This SDK bridge returns host payloads; GPU decoding stays in the shared codec.
pybind11_add_module(_ego_capture MODULE ego_capture.cpp)
target_include_directories(_ego_capture PRIVATE ${ORBBEC_SDK_ROOT}/include)
target_link_libraries(_ego_capture PRIVATE ${ORBBEC_SDK_LIBRARY})
# Place the extension beside the package entry point for in-tree viewer imports.
set_target_properties(_ego_capture PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  BUILD_RPATH ${ORBBEC_SDK_ROOT}/lib
  INSTALL_RPATH "$ORIGIN"
)
