// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

namespace core::camera
{

/**
 * @brief Where this user's camera plugin listens when nobody says otherwise.
 *
 * `$XDG_RUNTIME_DIR/isaaccapture/camera.sock`, or under `/tmp/isaaccapture-<uid>/` when that
 * variable is not set. A named instance, for a second plugin on the same machine, listens on
 * `camera-<instance>.sock` next to it.
 *
 * @throws std::invalid_argument for an instance name with anything but letters, digits, '-' and '_'.
 */
std::string default_endpoint_path(std::string_view instance = {});

/**
 * @brief Make sure the directory of an endpoint path exists and belongs to this user alone.
 * Creates it with mode 0700 when it is missing.
 * @throws std::system_error if it cannot be created, or exists and is not a directory this
 * user owns with no access for anyone else.
 */
void prepare_endpoint_directory(const std::string& endpoint_path);

} // namespace core::camera
