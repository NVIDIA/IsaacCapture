# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Vulkan REQUIRED)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(CUDAToolkit REQUIRED)
endif()

add_library(ovxr_pip_layer SHARED pip_layer.cpp)
target_compile_features(ovxr_pip_layer PRIVATE cxx_std_20)
target_include_directories(ovxr_pip_layer PRIVATE
    $<TARGET_PROPERTY:openxr_loader,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(ovxr_pip_layer PRIVATE Vulkan::Vulkan log_bridge::log_bridge_core)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep static logger/fmt symbols private to the helper's existing C ABI.
    set(_pip_exports "${CMAKE_CURRENT_BINARY_DIR}/ovxr_pip_exports.map")
    file(GENERATE OUTPUT "${_pip_exports}"
        CONTENT "{ global: isaacteleop_pip_*; local: *; };\n")
    target_link_options(ovxr_pip_layer PRIVATE "-Wl,--version-script,${_pip_exports}")
    set_property(TARGET ovxr_pip_layer APPEND PROPERTY LINK_DEPENDS "${_pip_exports}")
endif()
if(CUDAToolkit_FOUND AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(ovxr_pip_layer PRIVATE ISAAC_TELEOP_PIP_CUDA)
    target_link_libraries(ovxr_pip_layer PRIVATE CUDA::cudart_static)
endif()
set_target_properties(ovxr_pip_layer PROPERTIES
    OUTPUT_NAME isaaccapture_camera_panels
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/ovxr_session_manager/native")

add_library(ovxr_pip_layer::ovxr_pip_layer ALIAS ovxr_pip_layer)
