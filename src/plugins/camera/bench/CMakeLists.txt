# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reads one group from a running camera plugin and prints where each frame set spent its time.
add_executable(camera_bench
    main.cpp
)

target_link_libraries(camera_bench PRIVATE
    camera::camera_reader
    camera::backend_synthetic
)

install(TARGETS camera_bench RUNTIME DESTINATION plugins/camera)
