# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(BUILD_VIZ)
    add_subdirectory(camera_viz)
    add_subdirectory(deviceio_live_view)
    add_subdirectory(soma_body_publisher)
    add_subdirectory(soma_hand_publisher)
endif()
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(realhand)
endif()
