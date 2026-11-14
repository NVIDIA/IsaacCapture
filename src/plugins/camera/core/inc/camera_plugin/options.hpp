// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_source.hpp"

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <string_view>

namespace plugins::camera
{

/**
 * @brief Reads a camera's options for a backend and refuses what the backend does not know,
 * so a misspelt option is an error and not a silently ignored setting.
 */
class OptionReader
{
public:
    explicit OptionReader(const CameraConfig& config);

    /**
     * @brief The option's value, or `fallback` when it is not set.
     * @throws std::invalid_argument if the value is not a number between `minimum` and `maximum`.
     */
    std::uint32_t number(std::string_view key, std::uint32_t fallback, std::uint32_t minimum, std::uint32_t maximum);

    // The option's text, or `fallback` when it is not set.
    std::string text(std::string_view key, std::string_view fallback);

    /**
     * @brief Call once every option was read.
     * @throws std::invalid_argument naming an option that was set but never read.
     */
    void finish() const;

    // For a backend's own complaints: "camera 'head' (synthetic backend): <what>".
    [[noreturn]] void fail(const std::string& what) const;

private:
    const CameraConfig& m_config;
    std::set<std::string, std::less<>> m_read;
};

/**
 * @brief Parse one `--camera` argument: ID:BACKEND[:KEY=VALUE[,KEY=VALUE]...].
 * @throws std::invalid_argument if it does not have that form.
 */
CameraConfig parse_camera_argument(std::string_view argument);

} // namespace plugins::camera
