// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_plugin_backends/backends.hpp"

#include <camera_backend_synthetic/synthetic_backend.hpp>

namespace plugins::camera
{

std::vector<Backend> built_backends()
{
    std::vector<Backend> backends;
    backends.push_back(synthetic_backend());
    return backends;
}

} // namespace plugins::camera
