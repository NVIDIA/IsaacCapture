// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace core::camera
{

/**
 * @brief This user's endpoint directory: `$XDG_RUNTIME_DIR/isaaccapture`, or
 * `/tmp/isaaccapture-<uid>` when that variable is not set. The endpoint of every camera plugin
 * of the user is a socket file there.
 */
std::string default_endpoint_directory();

/**
 * @brief Where the camera plugin with this instance name listens when it is not given a path.
 *
 * `camera-<instance>.sock` in default_endpoint_directory(). An endpoint belongs to one camera
 * plugin, and a plugin's instance name is by default the name of its backend.
 *
 * @throws std::invalid_argument for an empty instance name or one with anything but letters,
 * digits, '-' and '_'.
 */
std::string default_endpoint_path(std::string_view instance);

/**
 * @brief The socket paths of the camera endpoints in this user's endpoint directory, sorted.
 *
 * These are the socket files with a name default_endpoint_path() gives; lock files and
 * anything else in the directory are left out. Nothing is connected to, so a socket file left
 * by a plugin that was killed is listed too, and the caller finds out when it connects. A
 * directory that does not exist gives an empty list and is not created.
 *
 * @throws std::system_error if the directory cannot be read.
 */
std::vector<std::string> default_endpoint_paths();

/**
 * @brief Make sure the directory of an endpoint path exists and belongs to this user alone.
 * Creates it with mode 0700 when it is missing.
 * @throws std::system_error if it cannot be created, or exists and is not a directory this
 * user owns with no access for anyone else.
 */
void prepare_endpoint_directory(const std::string& endpoint_path);

} // namespace core::camera
