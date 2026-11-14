// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_host.hpp"
#include "camera_source.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace plugins::camera
{

// The cameras a plugin is to serve and the size of their pools.
struct PluginConfig
{
    std::vector<CameraConfig> cameras;
    std::uint32_t slot_count = DEFAULT_SLOT_COUNT;
};

/**
 * @brief Read a configuration file. A camera takes three lines:
 *
 * @code{.yaml}
 * cameras:
 *   head:
 *     backend: synthetic
 *     width: 1920        # anything but `backend` is an option of that backend
 * slots: 8               # optional
 * @endcode
 *
 * @throws std::invalid_argument naming the file and the line that cannot be used.
 */
PluginConfig load_config_file(const std::string& path);

// Where a plugin looks for its configuration when it is not told:
// $XDG_CONFIG_HOME/isaaccapture/cameras.yaml, or ~/.config/isaaccapture/cameras.yaml.
std::string default_config_path();

// Every camera that the backends able to look for cameras find plugged in.
std::vector<CameraConfig> discover_cameras(std::span<const Backend> backends);

} // namespace plugins::camera
