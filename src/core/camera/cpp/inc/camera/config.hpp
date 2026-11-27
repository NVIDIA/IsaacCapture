// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_host.hpp"
#include "camera_source.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace core::camera
{

// The cameras a plugin is to serve and the size of their pools.
struct PluginConfig
{
    std::vector<CameraConfig> cameras;
    std::uint32_t slot_count = DEFAULT_SLOT_COUNT;
};

/**
 * @brief Read the cameras a configuration file names for one backend. The camera plugins of
 * a user share the file, and each camera in it names its backend. A camera takes two lines
 * and one per option:
 *
 * @code{.yaml}
 * cameras:
 *   head:
 *     backend: synthetic
 *     width: 1920        # anything but `backend` is an option of that backend
 * slots: 8               # optional
 * @endcode
 *
 * The cameras of other backends are left out: they belong to other plugins.
 *
 * @throws std::invalid_argument naming the file and the line that cannot be used; or, when
 * the file names cameras and none of them for `backend`, the file and its cameras with their
 * backends, which is how a misspelt backend shows.
 */
PluginConfig load_config_file(const std::string& path, std::string_view backend);

// Where a plugin looks for its configuration when it is not told:
// $XDG_CONFIG_HOME/isaaccapture/cameras.yaml, or ~/.config/isaaccapture/cameras.yaml.
std::string default_config_path();

} // namespace core::camera
