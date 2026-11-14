// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <camera_plugin/camera_source.hpp>

#include <vector>

namespace plugins::camera
{

// The backends this build of the plugin has. Which ones is decided when the build is configured.
std::vector<Backend> built_backends();

} // namespace plugins::camera
