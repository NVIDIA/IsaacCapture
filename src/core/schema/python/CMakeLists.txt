# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pybind11_add_module(schema_py
    oak_bindings.h
    controller_bindings.h
    full_body_bindings.h
    hand_bindings.h
    haptic_command_bindings.h
    head_bindings.h
    joint_se3_pose_bindings.h
    joint_state_bindings.h
    keyboard_bindings.h
    message_channel_bindings.h
    oglo_tactile_bindings.h
    pedals_bindings.h
    plugin_device_status_bindings.h
    pose_bindings.h
    schema_array_views.h
    schema_serialized.h
    se3_tracker_bindings.h
    schema_module.cpp
)

target_link_libraries(schema_py
    PRIVATE
        isaacteleop_schema
)

# Ensure generated headers are available.
add_dependencies(schema_py schema_header_generation)

set_target_properties(schema_py PROPERTIES
    OUTPUT_NAME "_schema"
    # Use genex in output directory - CMake won't add another config subdirectory
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaaccapture/schema"
)
