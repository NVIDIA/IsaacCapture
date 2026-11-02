# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Native Argus camera source for camera_viz. The .so lands next to
# __init__.py so the package imports without an install step.

cmake_minimum_required(VERSION 3.20)
project(camera_viz_argus LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;87;89;90")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

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

list(APPEND CMAKE_MODULE_PATH "/usr/src/jetson_multimedia_api/argus/cmake")
find_package(Argus QUIET)
find_package(EGL QUIET)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_tegra_lib_dirs
    "/usr/lib/aarch64-linux-gnu/nvidia"
    "/usr/lib/aarch64-linux-gnu/tegra"
    "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia"
    "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra"
  )
else()
  set(_tegra_lib_dirs
    "/usr/lib/x86_64-linux-gnu/nvidia"
    "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia"
  )
endif()

if(NOT ARGUS_INCLUDE_DIRS)
  find_path(ARGUS_INCLUDE_DIRS Argus/Argus.h
    PATHS "/usr/src/jetson_multimedia_api/argus/include"
          "/usr/src/jetson_multimedia_api/include"
  )
endif()
if(NOT ARGUS_LIBRARIES)
  find_library(ARGUS_SOCKETCLIENT_LIBRARY
    NAMES nvargus_socketclient
    PATHS ${_tegra_lib_dirs}
  )
  find_library(ARGUS_LIBRARY
    NAMES nvargus
    PATHS ${_tegra_lib_dirs}
  )
  if(ARGUS_SOCKETCLIENT_LIBRARY)
    set(ARGUS_LIBRARIES ${ARGUS_SOCKETCLIENT_LIBRARY})
  elseif(ARGUS_LIBRARY)
    set(ARGUS_LIBRARIES ${ARGUS_LIBRARY})
  endif()
endif()
if(NOT ARGUS_INCLUDE_DIRS OR NOT ARGUS_LIBRARIES)
  message(FATAL_ERROR "Argus headers/libs not found. Install Jetson Multimedia API / libargus.")
endif()

if(NOT EGL_INCLUDE_DIRS)
  find_path(EGL_INCLUDE_DIRS EGL/egl.h)
endif()
if(NOT EGL_LIBRARIES)
  find_library(EGL_LIBRARIES NAMES EGL)
endif()
if(NOT EGL_INCLUDE_DIRS OR NOT EGL_LIBRARIES)
  message(FATAL_ERROR "EGL headers/libs not found.")
endif()

message(STATUS "camera_viz argus: ARGUS=${ARGUS_LIBRARIES}")
message(STATUS "camera_viz argus: EGL=${EGL_LIBRARIES}")

pybind11_add_module(_camera_viz_argus MODULE
  argus_bindings.cpp
  argus_camera.cpp
  yuv_to_rgba.cu
)

target_include_directories(_camera_viz_argus PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ARGUS_INCLUDE_DIRS}
  ${EGL_INCLUDE_DIRS}
)

target_compile_options(_camera_viz_argus PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
)

target_link_libraries(_camera_viz_argus PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
  ${ARGUS_LIBRARIES}
  ${EGL_LIBRARIES}
)

set_target_properties(_camera_viz_argus PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
