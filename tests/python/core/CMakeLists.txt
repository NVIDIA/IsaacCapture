# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(schema)
add_subdirectory(retargeting_engine)
add_subdirectory(teleop_session_manager)

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(ego)
    add_subdirectory(cloudxr)
    # Needs `import isaaccapture`, and drives the C++ half through the emitter
    # built under tests/cpp/core/log_bridge.
    add_subdirectory(logging_config)
endif()
