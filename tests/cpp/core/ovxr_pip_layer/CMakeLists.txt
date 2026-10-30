# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep white-box lifecycle checks without exposing a plugin's private source
# tree as an interface include path. COPYONLY tracks both authoritative inputs.
set(_pip_whitebox_dir "${CMAKE_CURRENT_BINARY_DIR}/whitebox")
file(MAKE_DIRECTORY "${_pip_whitebox_dir}")
configure_file(
    "${PROJECT_SOURCE_DIR}/src/core/ovxr_pip_layer/cpp/pip_layer.cpp"
    "${_pip_whitebox_dir}/pip_layer.cpp" COPYONLY)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/pip_layer_contract_test.cpp"
    "${_pip_whitebox_dir}/pip_layer_contract_test.cpp" COPYONLY)
add_executable(ovxr_pip_layer_contract_test "${_pip_whitebox_dir}/pip_layer_contract_test.cpp")
target_compile_features(ovxr_pip_layer_contract_test PRIVATE cxx_std_20)
target_include_directories(ovxr_pip_layer_contract_test PRIVATE
    $<TARGET_PROPERTY:openxr_loader,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(ovxr_pip_layer_contract_test PRIVATE Vulkan::Vulkan log_bridge::log_bridge_core)
if(CUDAToolkit_FOUND AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(ovxr_pip_layer_contract_test PRIVATE ISAAC_TELEOP_PIP_CUDA)
    target_link_libraries(ovxr_pip_layer_contract_test PRIVATE CUDA::cudart)
endif()
add_test(NAME ovxr_pip_layer_contract_test COMMAND ovxr_pip_layer_contract_test)
