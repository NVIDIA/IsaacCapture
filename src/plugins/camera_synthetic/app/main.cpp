// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera/camera_plugin.hpp>
#include <camera_synthetic/synthetic_backend.hpp>

// A camera plugin is its backend and nothing else: the arguments, the configuration, the
// endpoint and the serving are the camera library's.
int main(int argc, char** argv)
{
    return core::camera::run_camera_plugin(argc, argv, plugins::camera_synthetic::synthetic_backend());
}
