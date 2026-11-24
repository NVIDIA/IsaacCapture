# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Native H.264 NVENC/NVDEC pybind11 module for camera_viz. The .so
# lands next to __init__.py so the package imports without any
# install step.

cmake_minimum_required(VERSION 3.20)
# project() fills CMAKE_CUDA_ARCHITECTURES with nvcc's default, so tell a
# user-supplied value apart before it runs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(_codec_default_cuda_archs ON)
endif()
project(camera_viz_codec LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# nvcc's default (PTX JIT for older GPUs), Ampere / Ada / Hopper / Orin, plus
# RTX Blackwell sm_120 on nvcc 12.8+ (older nvcc rejects it). Cached so
# build.sh's reconfigure keeps it. Override with CMAKE_CUDA_ARCHITECTURES on
# the configure line for fast iteration.
if(_codec_default_cuda_archs)
  set(_codec_cuda_archs ${CMAKE_CUDA_ARCHITECTURES} 80 86 87 89 90)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND _codec_cuda_archs 120)
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${_codec_cuda_archs}" CACHE STRING "CUDA architectures" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# pybind11 is pip-installed into the venv by ``camera_viz.sh setup``.
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

# NVENC / NVDEC entry-point libraries (ship with the NVIDIA driver).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_lib_arch_dir "/usr/lib/aarch64-linux-gnu/nvidia")
else()
  set(_lib_arch_dir "/usr/lib/x86_64-linux-gnu")
endif()

find_library(NVENC_LIBRARY
  NAMES libnvidia-encode.so.1 libnvidia-encode.so nvidia-encode
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  PATHS "${_lib_arch_dir}"
)
if(NOT NVENC_LIBRARY)
  message(FATAL_ERROR "libnvidia-encode not found. Install the NVIDIA driver.")
endif()
message(STATUS "camera_viz codec: NVENC=${NVENC_LIBRARY}")

find_library(NVCUVID_LIBRARY
  NAMES libnvcuvid.so.1 libnvcuvid.so nvcuvid
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  PATHS "${_lib_arch_dir}"
)
if(NOT NVCUVID_LIBRARY)
  message(FATAL_ERROR "libnvcuvid not found. Install the NVIDIA driver.")
endif()
message(STATUS "camera_viz codec: NVCUVID=${NVCUVID_LIBRARY}")

# ---------------------------------------------------------------------------
# NVIDIA Video Codec SDK headers + reference encoder/decoder sources.
# ---------------------------------------------------------------------------
set(_nvc_sdk_url
  "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/holohub/operators/holohub_nvc_13.0.19.zip"
)
set(_nvc_sdk_zip "${CMAKE_BINARY_DIR}/holohub_nvc_13.0.19.zip")
set(_nvc_sdk_dir "${CMAKE_BINARY_DIR}/_deps/nvc_sdk")

if(NOT EXISTS ${_nvc_sdk_zip})
  message(STATUS "Downloading NVIDIA Video Codec SDK...")
  file(DOWNLOAD ${_nvc_sdk_url} ${_nvc_sdk_zip}
    SHOW_PROGRESS
    EXPECTED_MD5 "a08c33c282bc92469ab8cf863aca9087"
  )
endif()
if(NOT EXISTS ${_nvc_sdk_dir}/NvEncoder)
  message(STATUS "Extracting NVIDIA Video Codec SDK...")
  file(MAKE_DIRECTORY ${_nvc_sdk_dir})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${_nvc_sdk_zip}
    WORKING_DIRECTORY ${_nvc_sdk_dir}
  )
endif()

# ---------------------------------------------------------------------------
# pybind11 module
# ---------------------------------------------------------------------------
pybind11_add_module(_camera_viz_codec MODULE
  codec_bindings.cpp
  h264_encoder.cpp
  h264_decoder.cpp
  nv12_to_rgba.cu
  ${_nvc_sdk_dir}/NvEncoder/NvEncoder.cpp
  ${_nvc_sdk_dir}/NvEncoder/NvEncoderCuda.cpp
  ${_nvc_sdk_dir}/NvDecoder/NvDecoder.cpp
  ${_nvc_sdk_dir}/Logger.cpp
)

target_include_directories(_camera_viz_codec PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${_nvc_sdk_dir}
  ${_nvc_sdk_dir}/NvEncoder
  ${_nvc_sdk_dir}/NvDecoder
)

target_compile_definitions(_camera_viz_codec PRIVATE NV_WINDOWS=0 NV_LINUX=1)

# NVCODEC SDK reference sources carry legacy warnings.
target_compile_options(_camera_viz_codec PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder -Wno-uninitialized -Wno-unused-variable
                            -Wno-unused-parameter -Wno-missing-field-initializers -Wno-sign-compare>
)

target_link_libraries(_camera_viz_codec PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
  ${NVENC_LIBRARY}
  ${NVCUVID_LIBRARY}
)

# Drop the .so next to __init__.py so the package imports straight
# from the source tree.
set_target_properties(_camera_viz_codec PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
